"""Opt-in synthetic Qwen reranking benchmark; no bank, gateway or runtime changes.

Use the separate evals/.venv-reranker interpreter. Model scoring follows the
official Qwen model card, with the public logits_to_keep=1 optimization.
Receipts are synthetic-only and incremental so partial runs remain auditable.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL = "Qwen/Qwen3-Reranker-0.6B"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
          "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
          "<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a query about a person's memories, retrieve relevant passages that answer the query"
QUALITY = (
    ("english", "What drink does the owner prefer?", (
        "I do not want tea. I prefer coffee.", "The owner prefers tea, not coffee.", "My train leaves at six.")),
    ("hinglish", "Owner ko kaunsa drink pasand hai?", (
        "Mujhe chai nahi chahiye, coffee pasand hai.", "Kal train subah chhe baje hai.", "Laptop ki battery kharab hai.")),
    ("hindi", "मालिक को कौन सा पेय पसंद है?", (
        "मुझे चाय नहीं चाहिए, कॉफी पसंद है।", "मेरी ट्रेन सुबह छह बजे है।", "लैपटॉप की बैटरी खराब है।")),
)


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def gpu_snapshot():
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        values = [int(x.strip()) for x in result.stdout.strip().splitlines()[0].split(",")]
        return dict(zip(("total_mib", "used_mib", "free_mib", "utilization_percent"), values))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def emit(output, row):
    row = {"recorded_at": datetime.now(timezone.utc).isoformat(), **row}
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()
    print(encoded, flush=True)


def encode(tokenizer, query, document, max_length):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    body = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
    ids = tokenizer.encode(body, add_special_tokens=False)
    available = max_length - len(prefix) - len(suffix)
    if available < 1:
        raise ValueError("token limit does not leave space for query/document")
    return prefix + ids[:available] + suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--cpu-int8", action="store_true", help="also benchmark public PyTorch dynamic INT8 linear layers")
    parser.add_argument("--reserve-mib", type=int, default=640)
    parser.add_argument("--max-batch-seconds", type=float, default=15,
                        help="stop larger batches at a length when warmup exceeds this duration")
    parser.add_argument("--coactivity", action="store_true", help="synthetic concurrent vision/embedding check after timing")
    args = parser.parse_args()
    if not args.live or args.repeats < 3 or not 1 <= args.threads <= 12:
        parser.error("requires --live, at least three repeats, and 1..12 threads")
    if any(x < 96 or x > 2048 for x in args.lengths) or any(x < 1 or x > 128 for x in args.batches):
        parser.error("lengths must be 96..2048 and batches 1..128")
    if not 512 <= args.reserve_mib <= 2048:
        parser.error("safety reserve must be 512..2048 MiB")
    if not math.isfinite(args.max_batch_seconds) or not 0 < args.max_batch_seconds <= 60:
        parser.error("max-batch-seconds must be finite and in (0,60]")
    if args.cpu_int8 and args.device != "cpu":
        parser.error("dynamic INT8 option is CPU-only")
    if args.model_path.name != args.revision or len(args.revision) != 40:
        parser.error("model snapshot path must match the immutable revision")
    if args.output.exists():
        parser.error("refusing to overwrite an existing receipt")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    baseline = gpu_snapshot()
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    memory_limit = None
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; do not silently benchmark CPU instead")
        free, total = torch.cuda.mem_get_info()
        memory_limit = free - args.reserve_mib * 1024**2
        if memory_limit < 1400 * 1024**2:
            raise RuntimeError("insufficient CUDA allocation budget after safety reserve")
        torch.cuda.set_per_process_memory_fraction(memory_limit / total)

    emit(args.output, {"kind": "environment", "model": MODEL, "revision": args.revision,
        "contract": "qwen-memory-reranker-benchmark-v2",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": args.device, "dtype": str(dtype), "threads": args.threads,
        "quantization": "dynamic-int8-linear" if args.cpu_int8 else "none",
        "gpu_before": baseline, "allocator_limit_mib": memory_limit / 1024**2 if memory_limit else None,
        "reserve_mib": args.reserve_mib, "synthetic_only": True,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "psutil")},
        "corpus_sha256": hashlib.sha256(json.dumps(QUALITY, ensure_ascii=False).encode()).hexdigest(),
        "timing": "tokenization/padding/H2D/forward/scoring/D2H included; load and warmup excluded",
        "precision_comparison": "CPU FP32 versus GPU FP16, not identical precision",
        "p95_caveat": "few repetitions: empirical order statistic, not a production tail-latency estimate"})
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left", local_files_only=True,
                                             trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype,
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).eval().to(args.device)
    if args.cpu_int8:
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    yes, no = (tokenizer.convert_tokens_to_ids(x) for x in ("yes", "no"))
    process = psutil.Process()
    emit(args.output, {"kind": "loaded", "seconds": time.monotonic() - started,
        "gpu": gpu_snapshot(), "rss_mib": process.memory_info().rss / 1024**2})

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def score(pairs, length):
        # Cache immutable template tokens and use the public batched tokenizer.
        prefix = template_prefix
        suffix = template_suffix
        available = length - len(prefix) - len(suffix)
        if available < 1:
            raise ValueError("token limit does not leave space for query/document")
        bodies = [f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
                  for query, document in pairs]
        encoded = tokenizer(bodies, padding=False, truncation=True, max_length=available,
                            add_special_tokens=False, return_attention_mask=False)["input_ids"]
        ids = [prefix + body + suffix for body in encoded]
        inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(args.device)
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        scores = logits[:, [no, yes]].float().softmax(dim=-1)[:, 1].cpu().tolist()
        if not all(math.isfinite(x) for x in scores):
            raise ValueError("nonfinite reranking scores")
        return scores, int(inputs["input_ids"].numel())

    template_prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    template_suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    for language, query, documents in QUALITY:
        scores, _ = score([(query, doc) for doc in documents], 256)
        ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Both drink passages in the English case answer the topic; relevance
        # alone must not be mistaken for resolution of contradictory memories.
        expected = {0, 1} if language == "english" else {0}
        emit(args.output, {"kind": "quality", "language": language, "scores": scores,
            "ranking": ranking, "topic_top1_pass": ranking[0] in expected,
            "negation_faithfulness": "not_established_by_relevance_reranking"})

    for length in args.lengths:
        # Same non-private content and padded length for all CPU/GPU grid cells.
        document = "The owner prefers coffee, not tea. " + "A synthetic daily note records an ordinary task. " * length
        for batch in args.batches:
            pairs = [("What drink does the owner prefer?", document)] * batch
            try:
                warmup_started = time.monotonic()
                score(pairs, length)
                sync()
                warmup_seconds = time.monotonic() - warmup_started
                if warmup_seconds > args.max_batch_seconds:
                    emit(args.output, {"kind": "measurement", "state": "skipped_latency_budget",
                        "device": args.device, "sequence_tokens": length, "batch": batch,
                        "warmup_seconds": warmup_seconds, "budget_seconds": args.max_batch_seconds})
                    break
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(args.repeats):
                    sync()
                    start = time.perf_counter()
                    scores, tokens = score(pairs, length)
                    sync()
                    times.append(time.perf_counter() - start)
                emit(args.output, {"kind": "measurement", "state": "ok", "device": args.device,
                    "sequence_tokens": length, "batch": batch, "repeats": args.repeats,
                    "actual_padded_tokens_per_pair": tokens // batch,
                    "seconds": times, "batch_p50_s": statistics.median(times),
                    "batch_p95_s": quantile(times, .95),
                    "pairs_per_s": batch * len(times) / sum(times),
                    "tokens_per_s": tokens * len(times) / sum(times),
                    "rss_mib": process.memory_info().rss / 1024**2,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else None,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if args.device == "cuda" else None,
                    "gpu": gpu_snapshot(), "score": scores[0]})
            except torch.cuda.OutOfMemoryError:
                emit(args.output, {"kind": "measurement", "state": "allocator_oom", "device": args.device,
                    "sequence_tokens": length, "batch": batch, "gpu": gpu_snapshot()})
                gc.collect()
                torch.cuda.empty_cache()
                break  # Do not retry a larger batch at the same length.
            finally:
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    if args.coactivity:
        # A separately labelled bounded probe, not part of the isolated timings.
        import base64
        import concurrent.futures
        import struct
        import threading
        import urllib.request
        import zlib

        def post(url, payload):
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            started = time.monotonic()
            result = json.load(urllib.request.urlopen(req, timeout=60))
            return result, time.monotonic() - started

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 128, 128, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * 128) * 128)) + chunk(b"IEND", b""))
        barrier = threading.Barrier(3, timeout=10)

        def vision():
            barrier.wait()
            result, seconds = post("http://127.0.0.1:8080/v1/chat/completions", {
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "What is the dominant color? Reply with just the color name."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}],
                "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
            answer = result["choices"][0]["message"].get("content", "")
            return {"seconds": seconds, "answer": answer, "passed": "red" in answer.lower()}

        def embeddings():
            barrier.wait()
            result, seconds = post("http://127.0.0.1:11434/v1/embeddings", {
                "model": "qwen3-embedding:0.6b", "input": ["Mujhe coffee pasand hai, chai nahi."] * 8})
            vectors = [row["embedding"] for row in result["data"]]
            valid = len(vectors) == 8 and all(len(v) == 1024 and all(math.isfinite(x) for x in v) for v in vectors)
            return {"seconds": seconds, "vectors": len(vectors), "passed": valid}

        def rerank():
            barrier.wait()
            start = time.monotonic()
            scores, _ = score([("Which drink?", "Coffee is preferred, not tea. " * 120)] * 4, 512)
            sync()
            return {"seconds": time.monotonic() - start, "scores": scores, "passed": len(scores) == 4}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {name: pool.submit(fn) for name, fn in (("vision", vision), ("embeddings", embeddings), ("reranker", rerank))}
            observed = []
            while not all(f.done() for f in futures.values()):
                snapshot = gpu_snapshot()
                if snapshot:
                    observed.append(snapshot)
                time.sleep(.05)
            outcomes = {}
            for name, future in futures.items():
                try:
                    outcomes[name] = future.result()
                except Exception as error:
                    outcomes[name] = {"passed": False, "error_type": type(error).__name__}
        emit(args.output, {"kind": "coactivity", "outcomes": outcomes,
            "passed": all(row["passed"] for row in outcomes.values()),
            "sampled_min_free_mib": min((x["free_mib"] for x in observed), default=None),
            "scope": "one small synthetic image, eight short embeddings, four 512-token pairs; not a full-context peak-load test"})
    emit(args.output, {"kind": "finished", "gpu": gpu_snapshot()})


if __name__ == "__main__":
    main()
