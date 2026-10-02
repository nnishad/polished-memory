"""Synthetic resident-reranker checks with short and long multimodal inputs."""
import argparse
import base64
import concurrent.futures
import json
import math
import struct
import threading
import time
import urllib.request
import uuid
import zlib
from pathlib import Path

from benchmark_reranker import emit, gpu_snapshot


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.monotonic()
    result = json.load(urllib.request.urlopen(req, timeout=60))
    return result, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-prefix", action="store_true", help="vary the initial prompt to test uncached vision prefill")
    args = parser.parse_args()
    if not args.live or args.output.exists():
        parser.error("requires --live and a new output receipt")

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 128, 128, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * 128) * 128)) + chunk(b"IEND", b""))
    for long in (False, True):
        filler = "Synthetic note. " * (4000 if long else 0)
        if args.fresh_prefix:
            filler = "Synthetic validation run " + uuid.uuid4().hex + ".\n" + filler
        tokens, _ = post("http://127.0.0.1:8080/tokenize", {"content": filler})
        while len(tokens["tokens"]) > 14000:
            filler = filler[:int(len(filler) * .95)]
            tokens, _ = post("http://127.0.0.1:8080/tokenize", {"content": filler})
        barrier = threading.Barrier(3, timeout=10)

        def vision():
            barrier.wait()
            result, seconds = post("http://127.0.0.1:8080/v1/chat/completions", {
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": filler + "\nWhat is the dominant image color? Reply with just the color name."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}],
                "max_tokens": 128, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
            answer = result["choices"][0]["message"].get("content", "")
            return {"passed": "red" in answer.lower(), "answer": answer, "seconds": seconds, "usage": result.get("usage")}

        def embeddings():
            barrier.wait()
            text = "Coffee is preferred, not tea. " * (100 if long else 1)
            result, seconds = post("http://127.0.0.1:11434/v1/embeddings", {"model": "qwen3-embedding:0.6b", "input": [text] * 8})
            vectors = [row["embedding"] for row in result["data"]]
            return {"passed": len(vectors) == 8 and all(len(v) == 1024 and all(math.isfinite(x) for x in v) for v in vectors),
                    "vectors": len(vectors), "seconds": seconds}

        def rerank():
            barrier.wait()
            count = 16 if long else 4
            result, seconds = post("http://127.0.0.1:8185/rerank", {
                "query": "Which drink does the owner prefer?", "texts": ["Coffee, not tea. " * (600 if long else 20)] * count})
            return {"passed": len(result) == count and sorted(r["index"] for r in result) == list(range(count))
                    and all(math.isfinite(r["score"]) for r in result), "candidates": len(result), "seconds": seconds}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {name: pool.submit(fn) for name, fn in (("vision", vision), ("embeddings", embeddings), ("reranker", rerank))}
            snapshots = []
            while not all(f.done() for f in futures.values()):
                observed = gpu_snapshot()
                if observed:
                    snapshots.append(observed)
                time.sleep(.05)
            outcomes = {}
            for name, future in futures.items():
                try:
                    outcomes[name] = future.result()
                except Exception as error:
                    outcomes[name] = {"passed": False, "error_type": type(error).__name__}
        emit(args.output, {"kind": "coactivity", "long_prompt": long, "fresh_prefix": args.fresh_prefix,
            "filler_tokens": len(tokens["tokens"]),
            "outcomes": outcomes, "passed": all(r["passed"] for r in outcomes.values()),
            "sampled_min_free_mib": min((s["free_mib"] for s in snapshots), default=None), "synthetic_only": True})


if __name__ == "__main__":
    main()
