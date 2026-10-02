# Local reranker capacity assessment

Measured 2026-10-02 during memory validation. No reranker was installed or existing model unloaded for this assessment.

## Observed headroom

| Resource | Observation |
| --- | --- |
| System RAM | 31 GiB total, 15 GiB available; 5.3 GiB immediately free |
| Swap | 2.5 GiB used; sampled swap-in/out was zero during the two current vmstat intervals |
| CPU | Intel i5-10500, 6 cores / 12 threads, AVX2; sampled idle 94–96% |
| GPU | RTX 3060, 12,288 MiB total, 10,603 MiB used; 1,685 MiB remaining |
| Current GPU consumers | llama.cpp approximately 7,452 MiB; Ollama approximately 2,390 MiB; desktop applications use additional VRAM |
| Resident embedding model | qwen3-embedding:0.6b, Ollama reports 2.4 GB and 4,096-token context |
| Disk | 84 GiB available on the home filesystem |

These are point-in-time measurements, not peak-load capacity guarantees. Existing inference and desktop workloads can increase memory usage.

## Recommendation

There is reasonable **CPU/RAM capacity for a small reranker**, but GPU co-residency is not yet safe to promise. Start with a bounded CPU deployment, one request at a time, small batches, and short candidate passages. Measure peak RSS, CPU utilization, p50/p95 latency and memory-query latency before enabling it for normal recall. Do not evict the embedding or vision model merely to fit another service.

Qwen3-Reranker-0.6B is the recommended first multilingual candidate to benchmark. Its [official model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) documents 0.6B parameters, multilingual support, and the required query/document template with yes/no scoring. Multilingual support does not prove Roman Hinglish quality; explicitly test mixed scripts, transliteration and negation.

Approximate parameter storage alone is 1.2 GB at 16-bit or 2.4 GB at 32-bit precision, before runtime overhead, activations, temporary buffers and concurrency. With only about 1.65 GiB VRAM free, an unquantized GPU deployment has insufficient demonstrated safety margin. A quantized deployment may fit, but requires a supported backend, numerical-quality comparison and a measured peak-memory run; it is not an established result.

## Core integration prerequisites

- Use the correct Qwen reranker template and yes/no logit score, not generated prose or embedding similarity mislabeled as reranking.
- Preserve candidate indices and provenance. Validate finite scores, response cardinality, duplicate indices and out-of-range indices.
- Hindsight 0.10.1 supports a TEI-shaped remote provider using GET `/info` and POST `/rerank`. Check the selected serving backend against that exact contract.
- Extend the owned admission/resource accounting path for reranking. The current gate forwards only chat completions and embeddings; do not bypass it with an ungoverned GPU service.
- Bound candidates, input tokens, concurrency, deadlines and queue size. Clearly report degraded RRF-only recall if the learned reranker fails.
- Benchmark Hindi, Roman Hinglish and English against RRF-only recall, including unrelated distractors, corrections and negated preferences. Report relevance quality separately from service availability.

## Validation status

No learned reranker currently exists, so learned-reranking end-to-end validation remains pending. Hindsight's configured RRF is rank fusion, not a learned reranking model.

The reviewed schema-17 release was activated using the core installation workflow. Gate, Hindsight API, worker and Hermes gateway are active; gate and Hindsight health endpoints responded healthy. An isolated three-language retain/recall smoke run completed all three formation jobs and returned three results for each query. Recall latency was approximately 0.14–0.33 seconds and the run took 8.93 seconds. This small test does not establish result faithfulness, graph/consolidation quality, provider injection quality or learned-reranker performance.

Synthetic bank: `eval-2820194eec15410abe546844687c8421`. Durable test ledger: `/tmp/hermes-memory-synthetic-3q5co5qf/canonical.db`. All three test documents were verified absent after cleanup; the bank shell remains. Outbound memory delivery is paused for validation; no Telegram delivery test was performed.

## Vision context reduction, subsequently applied

The owner requested at least about 30K tokens for Hermes vision, with one slot acceptable. Changed the persistent [vision launcher](/home/jugaadu/services/deepseek-vision/run_service.sh) from `-c 60000` with automatic parallelism to `-c 32768 --parallel 1`, preserving the model, vision projector, port and other inference settings.

The previous running server was an orphan process outside its existing enabled systemd unit. Verified its exact command and idle slots, stopped that process gracefully, then started `deepseek-vision.service`. This unit was already enabled; no new boot-enable policy was added. No other model was unloaded.

- `/health` returned OK; `/props` confirmed one slot and 32,768-token context.
- Corrected an earlier conversational estimate: the old server used a unified context cache and each slot reported 60,160-token capacity. Dividing the old capacity by four was not valid for that build/configuration.
- Idle vision-process GPU allocation fell from **7,452 MiB to 6,052 MiB**, saving **1,400 MiB (about 1.37 GiB)**. Total reported GPU usage fell from 10,603 MiB to 9,203 MiB before the image probe.
- A synthetic 128×128 red PNG sent through the existing OpenAI-compatible multimodal endpoint returned `Red`, in 1.334 seconds, using 1,054 prompt tokens and two output tokens. This establishes a basic vision smoke test, not a full Hermes image-turn or 30K-input quality test.
- After the image probe, the driver reported 9,237 MiB used and **2,726 MiB free**. Use the driver's free figure for capacity decisions; reserved memory means total minus used need not equal reported free.
- Vision, Hermes gateway, memory gate, Hindsight API and worker services all reported active after the change.

The context limit includes image tokens, prompt/history and generated output; 32K does not provide 32K text tokens plus an unlimited image/output allowance. One slot serializes requests. Learned-reranker co-residency remains unverified even with the improved headroom. The existing vision server also logs an unauthenticated `0.0.0.0` listener with permissive CORS; that pre-existing network-exposure issue was not changed as part of the context adjustment.

## Current configuration: owner-requested 16K vision context

The owner subsequently requested a further reduction to make room for GPU reranking. The persistent vision launcher now uses **`-c 16384 --parallel 1`**. This supersedes the earlier 32K requirement and setting.

- Stopped the exact active reranker benchmark process before restarting vision; no benchmark process remained afterward. Partial benchmark receipts were retained, not discarded or reported complete.
- Restarted the same enabled `deepseek-vision.service`; health and `/props` confirmed one slot and 16,384-token context.
- A synthetic red-image check again passed, in 1.248 seconds.
- After that image probe, GPU usage was **8,637 MiB**, with **3,326 MiB (about 3.25 GiB) free**. Compared with the equivalent 32K post-image observation, this frees another **600 MiB (about 0.59 GiB)**. This measurement excludes a resident reranker, because the benchmark process was stopped.
- Vision, Hermes gateway, memory gate, Hindsight API and worker all remained active. Embeddings were not unloaded.

The earlier GPU pilot benchmark succeeded with the models resident at 32K vision context, but the repeat/coactivity run was interrupted for this requested change. No completed 16K coactivity or full-context peak-memory benchmark is claimed. Existing throughput observations remain observations of the earlier configuration, not measurements under the new 16K setting. CPU dynamic INT8 also failed the small Hindi relevance check; it is not approved as a quality-preserving fallback.

## Subsequent resident reranker validation

The preceding paragraph is historical. The reranker has since been loaded as a bounded local model service, benchmarked at the 16K vision setting with multiple memory/batch/cache profiles, and tested alongside a 13K-token synthetic vision request and longer embeddings. It remains resident in the selected high-memory cached profile. See [the current benchmark report](reranker-gpu-benchmark.md) for exact throughput, quality caveats, raw receipts and remaining Hindsight wiring work. About 827 MiB was reported free after this combined test; the earlier 3.25 GiB figure was the baseline without a resident reranker.
