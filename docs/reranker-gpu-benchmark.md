# Resident GPU reranker: configuration and benchmark

Date: 2026-10-02. All inputs were synthetic. No private memory corpus was sampled, no Telegram message was sent, and the installed Hindsight recall configuration was not changed.

Later integration update: Hindsight now uses the local Qwen model through the authenticated gate. The reranker service is enabled at boot. The measurements below remain the original standalone benchmark; see [the integration/flow report](hindsight-qwen-reranker-integration.md) for subsequent activation, fixes and live end-to-end evidence.

## Result and active service

Qwen3-Reranker-0.6B is resident on the RTX 3060 and available on loopback at `http://127.0.0.1:8185`. Its core implementation is [models/reranker.py](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/models/reranker.py), installed normally as a wheel in the separate model environment. No upstream model/provider library was edited or monkeypatched.

The active unit is [hermes-memory-reranker.service](/home/jugaadu/.config/systemd/user/hermes-memory-reranker.service). It is started, but not automatically enabled at boot. The memory runtime/backend/gateway release was not switched for this model-service benchmark.

| Setting | Active value |
| --- | --- |
| Model revision | `e61197ed45024b0ed8a2d74b80b4d909f1255473` |
| Precision / attention | FP16 / PyTorch SDPA |
| GPU workers | 1 |
| Queued/running request slots | 8 total |
| Maximum candidates per request | 128 |
| Maximum individual input | 2,048 tokens, including template and query |
| Query bound | 512 tokens; overlong queries rejected, not silently truncated |
| Maximum padded batch tokens | 32,768 |
| Maximum pairs in an internal batch | 128 |
| GPU allocator ceiling | 2,400 MiB |
| Startup safety reserve | 768 MiB, after CUDA initialization |
| Allocator workspaces | Retained between requests, within the ceiling |
| HTTP body / handler bounds | 1 MiB / 16 simultaneous handlers |
| Request inference wait | 30 seconds; queued cancellations release their admission slot |

Length sorting minimizes padding, with scores restored to original candidate indices. The charge is **batch count × longest encoded sequence**, not the sum of unpadded lengths. Thus a full batch can hold up to 128 short 128-token inputs, 64 × 512-token inputs, or 16 × 2,048-token inputs. Larger requests are split into bounded internal batches. No result or conversation KV cache is used: only reusable GPU allocation workspaces remain resident.

GET `/info`, GET `/health` and POST `/rerank` implement the contract needed by Hindsight 0.10.1's remote TEI adapter. POST returns a sorted list of `{index, score}` objects. Invalid requests, unknown models, excess input and nonfinite scores are refused. The loopback listener rejects browser-Origin requests and does not log request bodies.

The startup reserve is not a guaranteed reservation against another process allocating memory later. The allocator cap is a fail-closed ceiling for this process, not a cluster-wide scheduler. Arbitrary larger images, simultaneous unrelated GPU workloads and a completely full 16K vision request remain outside the tested envelope.

## Does using more VRAM make it faster?

Three actual HTTP service profiles were measured with the same model, precision, synthetic inputs and one GPU worker. Each cell has a warmup and three timed repetitions, including HTTP handling, tokenization, inference and result validation. Load time is excluded.

| Workload | Smaller memory: 16K batch tokens | Larger memory: 32K batch tokens | Larger + cached workspaces | Cached vs smaller |
| --- | ---: | ---: | ---: | ---: |
| 128 candidates × 128 tokens | 122.43 scores/s | 122.74 scores/s | **124.44 scores/s** | +1.6% |
| 64 candidates × 512 tokens | 30.25 scores/s | 30.65 scores/s | **30.75 scores/s** | +1.6% |
| 16 candidates × 2,048 tokens | 6.81 scores/s | 6.89 scores/s | **6.90 scores/s** | +1.3% |

Median full-request latency in the cached profile was about 1.03 seconds, 2.08 seconds and 2.32 seconds respectively. A single 128-token pair took about 25 ms. A “score” is one query/passage pair, **not an entire memory query**.

The smaller profile used a 1,900 MiB ceiling, max batch 64 and released workspaces after requests. The larger profiles used a 2,400 MiB ceiling and max batch 128. Their observed peak reserved memory was 1,772 MiB, 2,380 MiB and 2,348 MiB respectively. In the cached profile, live tensor allocation afterward was about 1,144 MiB; reserved allocator memory was 2,348 MiB. Extra residency is working space, not another copy of the model.

Increasing callers from 1 to 2 to 4, for the same four 16-candidate/512-token requests, left aggregate throughput close to 30 scores/s. Request latency increased as callers queued. The GPU was near 100% compute utilization during larger cells. More inference workers would multiply memory pressure without a demonstrated throughput advantage.

**Conclusion:** using more VRAM for larger batches and cached workspaces gives a small observed throughput gain here, not a major speedup. Filling memory for its own sake does not add GPU compute capacity. These are short sequential trials on a shared desktop, not statistically established production tail-latency results or proof of global optimality. The high-memory cached profile is retained because the owner requested greater utilization and the bounded coactivity checks passed.

## CPU comparison and quality warning

The revised direct-scoring harness also measured CPU FP32 on six physical CPU threads. At 128 tokens, batch 4 achieved 3.31 scores/s versus GPU FP16's 94.93 scores/s at the same batch; batch 8 achieved 3.19 versus 104.32 scores/s. At 512 tokens and batch 1, CPU achieved 0.69 scores/s versus GPU's 21.28 scores/s. This is a CPU FP32/GPU FP16 operational comparison, not identical precision or an optimized ONNX/OpenVINO comparison. Larger CPU cells were bounded or stopped once they showed no throughput benefit.

CPU FP32 and GPU FP16 agreed on all three small English, Roman Hinglish and Hindi relevance rankings. Their largest quality-case score difference was about 0.000121. This is a smoke test, not a held-out multilingual quality benchmark.

The tested PyTorch **dynamic INT8 linear-layer** CPU configuration only modestly increased throughput and **failed the Hindi case**, ranking an unrelated passage above the relevant preference. Its Hindi score drift reached about 0.963 relative to FP32. Do not deploy that configuration as a quality-preserving fallback. This result does not establish that every INT8 method fails; quantization-aware or different quantization schemes require separate evaluation.

Both conflicting drink preferences in the English fixture are topically relevant. A reranker cannot decide which contradictory memory is true merely from relevance scores. Canonical evidence authority, time/revision, corrections and support validation remain the memory framework's responsibility.

## Resident coactivity and memory

After the high-memory cached service was warmed to its larger batches, two simultaneous-use rounds were run without unloading any model:

- Short round: vision identified a red image, embeddings returned eight valid 1,024-dimensional vectors, and reranking returned four valid scores. All passed; sampled free VRAM minimum 835 MiB.
- Longer round: vision processed **13,054 prompt tokens** including the image and returned `Red`; eight longer embeddings and 16 long reranker candidates also completed. Vision took 8.72 seconds, embeddings 0.60 seconds, reranking 4.78 seconds under contention. All passed; sampled free VRAM minimum **827 MiB**.
- Final observation: **11,136 MiB used**, 827 MiB reported free, out of 12,288 MiB. Reranker, vision, gateway, gate, Hindsight API and worker units all reported active.

The longer round is substantially more realistic than the original tiny image test, but it is not a full 16K context, maximum-resolution/multi-image stress test or a sustained reliability test. Sampled GPU memory can miss very short peaks.

## Reproduction and evidence

- [Direct scoring harness](/home/jugaadu/Projects/hermes-memory/evals/benchmark_reranker.py): batched tokenization, only last-position logits, no KV cache, explicit warmup and synchronized timings, actual padded input lengths, immutable revision and script hash.
- [Actual HTTP benchmark](/home/jugaadu/Projects/hermes-memory/evals/benchmark_reranker_service.py): profile comparison, cardinality/index/finite-score checks and equal-work caller-concurrency comparison.
- [Resident coactivity check](/home/jugaadu/Projects/hermes-memory/evals/check_reranker_coactivity.py): no second reranker model load, short/long synthetic multimodal tests.
- Raw receipts under `/home/jugaadu/data/model-benchmarks/`: `reranker-gpu-16k-vision.jsonl`, `reranker-cpu-v2.jsonl`, `reranker-http-low.jsonl`, `reranker-http-high.jsonl`, `reranker-http-high-cached.jsonl`, and `reranker-cached-coactivity.jsonl`.
- Historical pilot and interrupted receipts remain in the same directory. They are not substituted for completed current-profile measurements.
- Versions: torch 2.7.1+cu128, transformers 4.57.6, psutil 7.2.2. Models are cached under `/home/jugaadu/data/model-benchmarks/huggingface`; the separate environment is `evals/.venv-reranker`.
- Final source regression: **3,189 passed, 1 skipped**, 64.97 seconds. Tests include padded-cost batch limits and admission release on completion, cancellation and failure. The source compatibility artifact was regenerated; the existing memory/gateway release was not replaced.
- [Official Qwen model/scoring reference](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B). The model's template and yes/no logit probability are used; generation and embedding cosine scores are not relabeled as reranking.

## Remaining integration gap

At the end of the original benchmark, Hindsight still used RRF without the learned model. That historical integration gap is now closed through owned authenticated routing, shared physical-device admission, complete response validation, actual usage/time accounting and synthetic memory-flow checks. RRF remains the fusion stage before Qwen ranking. See [the integration report](hindsight-qwen-reranker-integration.md). Memory outbound delivery remains paused during validation.
