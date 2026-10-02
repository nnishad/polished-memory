"""Bounded loopback Qwen reranking service using the public TEI wire contract.

Optional torch/transformers dependencies belong in a separate model environment.
This does not register a route, grant inference, or enable Hindsight reranking.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = "Qwen/Qwen3-Reranker-0.6B"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
          "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
          "<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a query about a person's memories, retrieve relevant passages that answer the query"


class Unavailable(RuntimeError):
    pass


class ScoredRows(list):
    """Scores and per-job actual token counts; never shared mutable last-request state."""
    def __init__(self, rows, *, input_tokens, padded_tokens):
        super().__init__(rows)
        self.input_tokens = input_tokens
        self.padded_tokens = padded_tokens


def cohere_request(payload):
    """Local Cohere-compatible protocol, not a cloud provider or SDK dependency."""
    if not isinstance(payload, dict) or set(payload) - {"model", "query", "documents", "return_documents"}:
        raise ValueError("unsupported rerank request")
    if payload.get("model") != MODEL or payload.get("return_documents", False) is not False:
        raise ValueError("unknown model or document-return request")
    adapted = {"model": MODEL, "query": payload.get("query"), "texts": payload.get("documents")}
    validate(adapted)
    return adapted


def cohere_response(rows):
    return {"results": [{"index": row["index"], "relevance_score": row["score"]} for row in rows],
            "usage": {"input_tokens": rows.input_tokens, "total_tokens": rows.padded_tokens}}


def cohere_usage(body, count):
    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    if not isinstance(usage, dict):
        raise ValueError("invalid actual rerank token usage")
    actual, padded = usage.get("input_tokens"), usage.get("total_tokens")
    if type(actual) is not int or type(padded) is not int or not count <= actual <= padded <= count * 2048:
        raise ValueError("missing or invalid actual rerank token usage")
    return padded


def validate_cohere_response(body, count):
    """Refuse missing/duplicate candidates instead of native adapter's silent zero scores."""
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise ValueError("missing rerank results")
    rows = body["results"]
    if len(rows) != count:
        raise ValueError("incomplete rerank results")
    indices = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("index")) is not int:
            raise ValueError("invalid rerank index")
        score = row.get("relevance_score")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid rerank score")
        indices.append(row["index"])
    if sorted(indices) != list(range(count)):
        raise ValueError("invalid rerank indices")
    cohere_usage(body, count)


def validate(payload):
    if not isinstance(payload, dict) or set(payload) - {"query", "texts", "return_text", "top_n", "model"}:
        raise ValueError("unsupported rerank request")
    query, texts = payload.get("query"), payload.get("texts")
    if not isinstance(query, str) or not query.strip() or len(query) > 4096:
        raise ValueError("query must be nonempty and at most 4096 characters")
    if not isinstance(texts, list) or not 1 <= len(texts) <= 128:
        raise ValueError("requires 1..128 candidate texts")
    if any(not isinstance(t, str) or not t.strip() or len(t) > 65536 for t in texts):
        raise ValueError("candidate texts must be nonempty and at most 65536 characters")
    if type(payload.get("return_text", False)) is not bool:
        raise ValueError("return_text must be boolean")
    top = payload.get("top_n", len(texts))
    if type(top) is not int or not 1 <= top <= len(texts):
        raise ValueError("top_n is outside candidate bounds")
    if payload.get("model", MODEL) != MODEL:
        raise ValueError("unknown model")
    return query, texts, top, payload.get("return_text", False)


def batches(lengths, *, max_batch=64, token_budget=16384):
    """Group by length; charge padded batch size, not a sum of unpadded tokens."""
    if max_batch < 1 or token_budget < 1 or any(n < 1 or n > token_budget for n in lengths):
        raise ValueError("invalid batch limits or sequence length")
    ordered = sorted(range(len(lengths)), key=lambda i: lengths[i])
    result, current = [], []
    for index in ordered:
        if current and (len(current) >= max_batch or (len(current) + 1) * lengths[index] > token_budget):
            result.append(current)
            current = []
        current.append(index)
    if current:
        result.append(current)
    return result


class QwenReranker:
    def __init__(self, model_path, revision, *, max_batch=64, token_budget=16384,
                 allocation_mib=2400, reserve_mib=768, retain_workspaces=False):
        if model_path.name != revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("requires a pinned immutable model snapshot")
        if not 1 <= max_batch <= 128 or not 2048 <= token_budget <= 32768:
            raise ValueError("batch configuration outside benchmarked operating envelope")
        if not 1600 <= allocation_mib <= 2448 or reserve_mib < 768:
            raise ValueError("unsafe GPU memory allocation limits")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise Unavailable("CUDA unavailable")
        free, total = torch.cuda.mem_get_info()
        if free < (allocation_mib + reserve_mib) * 1024**2:
            raise Unavailable("insufficient GPU headroom for the selected envelope")
        torch.cuda.set_per_process_memory_fraction(allocation_mib * 1024**2 / total)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                                      padding_side="left", trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            trust_remote_code=False, dtype=torch.float16, attn_implementation="sdpa").eval().to("cuda")
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.yes, self.no = (self.tokenizer.convert_tokens_to_ids(s) for s in ("yes", "no"))
        self.max_batch, self.token_budget = max_batch, token_budget
        self.retain_workspaces = retain_workspaces
        self.info = {"model_id": MODEL, "revision": revision, "device": "cuda", "dtype": "float16",
            "max_input_tokens": 2048, "max_query_tokens": 512, "max_client_batch_size": 128,
            "max_batch_size": max_batch, "max_batch_tokens": token_budget,
            "allocation_mib": allocation_mib, "reserve_mib": reserve_mib,
            "truncation": "document tail only", "workers": 1, "retain_workspaces": retain_workspaces}

    def score(self, payload):
        query, texts, top, return_text = validate(payload)
        query_ids = self.tokenizer.encode(query, add_special_tokens=False)
        if len(query_ids) > 512:
            raise ValueError("query exceeds 512 tokens; query truncation is refused")
        head = self.tokenizer.encode(f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: ",
                                     add_special_tokens=False)
        available = 2048 - len(self.prefix) - len(head) - len(self.suffix)
        if available < 1:
            raise ValueError("query/template exceeds token limit")
        # Tokenize the complete body, matching Qwen's trained token boundaries.
        # Query/header is already bounded, so truncation can affect only the tail
        # of the document, never the query or required assistant suffix.
        bodies = [f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {text}" for text in texts]
        body_budget = 2048 - len(self.prefix) - len(self.suffix)
        docs = self.tokenizer(bodies, add_special_tokens=False, padding=False, truncation=True,
                              max_length=body_budget, return_attention_mask=False)["input_ids"]
        ids = [self.prefix + doc + self.suffix for doc in docs]
        scores = [None] * len(ids)
        chunks = batches([len(s) for s in ids], max_batch=self.max_batch, token_budget=self.token_budget)
        try:
            for indices in chunks:
                values = self._forward([ids[i] for i in indices])
                for index, value in zip(indices, values, strict=True):
                    if not math.isfinite(value):
                        raise Unavailable("nonfinite model score")
                    scores[index] = value
        except self.torch.cuda.OutOfMemoryError as error:
            error.__traceback__ = None
            gc.collect()
            self.torch.cuda.empty_cache()
            raise Unavailable("GPU allocation budget exceeded") from None
        finally:
            # Cached mode is explicit and still bounded by the allocator cap.
            if not self.retain_workspaces:
                self.torch.cuda.empty_cache()
        rows = [{"index": i, "score": s, **({"text": texts[i]} if return_text else {})}
                for i, s in enumerate(scores)]
        return ScoredRows(sorted(rows, key=lambda row: (-row["score"], row["index"]))[:top],
                          input_tokens=sum(map(len, ids)),
                          padded_tokens=sum(len(c) * max(len(ids[i]) for i in c) for c in chunks))

    def _forward(self, ids):
        inputs = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
        logits = None
        try:
            with self.torch.inference_mode():
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                return logits[:, [self.no, self.yes]].float().softmax(dim=-1)[:, 1].cpu().tolist()
        finally:
            del inputs, logits

    def memory_stats(self):
        return {"allocated_mib": self.torch.cuda.memory_allocated() / 1024**2,
                "reserved_mib": self.torch.cuda.memory_reserved() / 1024**2,
                "peak_allocated_mib": self.torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": self.torch.cuda.max_memory_reserved() / 1024**2}


class Admission:
    def __init__(self, model, slots=8):
        self.model = model
        self.slots = threading.BoundedSemaphore(slots)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker")

    def submit(self, payload):
        validate(payload)
        if not self.slots.acquire(blocking=False):
            raise Unavailable("request queue is full")
        def run():
            try:
                return self.model.score(payload)
            finally:
                self.slots.release()
        try:
            future = self.executor.submit(run)
        except Exception:
            self.slots.release()
            raise
        future.add_done_callback(lambda job: self.slots.release() if job.cancelled() else None)
        return future


def serve(model, port):
    admission = Admission(model)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(10)
            super().setup()

        def log_message(self, *_):
            pass  # Never log queries, passages or request bodies.

        def send(self, status, result):
            body = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/info":
                self.send(200, {**model.info, "gpu_memory": model.memory_stats()})
            elif self.path == "/health":
                self.send(200, {"status": "ok", "model_loaded": True})
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/rerank", "/v1/rerank"):
                self.send(404, {"error": "not found"})
                return
            future = None
            try:
                if self.headers.get("Origin") or self.headers.get("Transfer-Encoding"):
                    raise ValueError("browser-origin and chunked requests are refused")
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                    raise ValueError("application/json is required")
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 1024**2:
                    raise ValueError("request size must be 1..1048576 bytes")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete request body")
                payload = json.loads(raw)
                if self.path == "/v1/rerank":
                    payload = cohere_request(payload)
                future = admission.submit(payload)
                rows = future.result(timeout=30)
                self.send(200, cohere_response(rows) if self.path == "/v1/rerank" else rows)
            except (ValueError, UnicodeError):
                self.send(422, {"error": "invalid request or input limit exceeded"})
            except (Unavailable, concurrent.futures.TimeoutError):
                if future:
                    future.cancel()
                self.send(503, {"error": "model busy or GPU budget unavailable"})
            except (OSError, TimeoutError):
                if future:
                    future.cancel()
            except Exception:
                self.send(500, {"error": "model inference failed"})

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        handlers = threading.BoundedSemaphore(16)

        def process_request(self, request, client_address):
            if not self.handlers.acquire(blocking=False):
                request.close()
                return
            try:
                super().process_request(request, client_address)
            except Exception:
                self.handlers.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self.handlers.release()

    server = Server(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        admission.executor.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", type=int, default=8185)
    parser.add_argument("--allocation-mib", type=int, default=2400)
    parser.add_argument("--reserve-mib", type=int, default=768)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--retain-workspaces", action="store_true",
                        help="keep bounded allocator workspaces cached for repeated-request throughput")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port outside unprivileged TCP bounds")
    model = QwenReranker(args.model_path, args.revision, max_batch=args.max_batch,
        token_budget=args.token_budget, allocation_mib=args.allocation_mib, reserve_mib=args.reserve_mib,
        retain_workspaces=args.retain_workspaces)
    serve(model, args.port)


if __name__ == "__main__":
    main()
