"""Bounded synthetic HTTP benchmark of the actual resident reranker service."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

from benchmark_reranker import QUALITY, emit, gpu_snapshot
from hermes_memory.models.reranker import INSTRUCTION, PREFIX, SUFFIX


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request("http://127.0.0.1:8185" + path, data=data,
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    result = json.load(urllib.request.urlopen(req, timeout=30))
    return result, time.monotonic() - started


def check(rows, count):
    if (not isinstance(rows, list) or len(rows) != count
            or sorted(r["index"] for r in rows) != list(range(count))
            or not all(math.isfinite(r["score"]) for r in rows)):
        raise ValueError("invalid reranker cardinality, index or score")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    if not args.live or args.output.exists():
        parser.error("requires --live and a new output path")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    info = None
    for _ in range(15):
        try:
            info, _ = request("/info")
            break
        except OSError:
            time.sleep(1)
    if info is None:
        raise RuntimeError("resident service did not become ready")
    emit(args.output, {"kind": "environment", "profile": args.profile, "info": info,
                      "gpu": gpu_snapshot(), "synthetic_only": True})
    for language, query, texts in QUALITY:
        rows, seconds = request("/rerank", {"query": query, "texts": list(texts), "return_text": False})
        check(rows, len(texts))
        expected = {0, 1} if language == "english" else {0}
        emit(args.output, {"kind": "quality", "language": language, "rows": rows,
                          "topic_top1_pass": rows[0]["index"] in expected, "seconds": seconds})

    query = "What drink does the owner prefer?"
    head = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: "
    query_header_tokens = len(tokenizer.encode(PREFIX, add_special_tokens=False))
    suffix_tokens = len(tokenizer.encode(SUFFIX, add_special_tokens=False))

    def document(target):
        original = tokenizer.encode("The owner prefers coffee, not tea. "
                                    + "A synthetic daily note records an ordinary task. " * target,
                                    add_special_tokens=False)
        low, high = 1, min(len(original), target)
        while low < high:
            middle = (low + high + 1) // 2
            doc = tokenizer.decode(original[:middle])
            size = query_header_tokens + len(tokenizer.encode(head + doc, add_special_tokens=False)) + suffix_tokens
            if size <= target:
                low = middle
            else:
                high = middle - 1
        doc = tokenizer.decode(original[:low])
        actual = query_header_tokens + len(tokenizer.encode(head + doc, add_special_tokens=False)) + suffix_tokens
        return doc, actual

    for length, candidates in ((128, 1), (128, 128), (512, 32), (512, 64), (2048, 8), (2048, 16)):
        doc, actual = document(length)
        payload = {"query": query, "texts": [doc] * candidates, "return_text": False}
        rows, _ = request("/rerank", payload)
        check(rows, candidates)
        times = []
        for _ in range(3):
            rows, seconds = request("/rerank", payload)
            check(rows, candidates)
            times.append(seconds)
        info, _ = request("/info")
        emit(args.output, {"kind": "measurement", "profile": args.profile,
            "tokens_per_pair": actual, "candidates": candidates, "seconds": times,
            "request_p50_s": statistics.median(times), "pairs_per_s": candidates * 3 / sum(times),
            "gpu_memory": info["gpu_memory"], "gpu": gpu_snapshot()})

    # Equal total work at different client concurrency. One GPU worker remains
    # intentional: more callers should queue, not multiply GPU memory peaks.
    doc, actual = document(512)
    payload = {"query": query, "texts": [doc] * 16, "return_text": False}
    for clients in (1, 2, 4):
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
            outcomes = list(pool.map(lambda _: request("/rerank", payload), range(4)))
        seconds = time.monotonic() - started
        for rows, _ in outcomes:
            check(rows, 16)
        emit(args.output, {"kind": "concurrency", "profile": args.profile, "clients": clients,
            "requests": 4, "tokens_per_pair": actual, "pairs_per_s": 64 / seconds,
            "elapsed_s": seconds, "request_seconds": [s for _, s in outcomes], "gpu": gpu_snapshot()})
    emit(args.output, {"kind": "finished", "profile": args.profile, "gpu": gpu_snapshot()})


if __name__ == "__main__":
    main()
