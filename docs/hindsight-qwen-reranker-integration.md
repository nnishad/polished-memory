# Hindsight + local Qwen: integration and validation

Date: 2026-10-02. Owner requested core integration and complete synthetic path testing.

## Architecture

```text
Hermes / approved input
  → canonical evidence + current revision / visibility
  → approved formation → native Hindsight retain
  → text extraction through owned gate → Qwen embeddings through owned gate
  → Hindsight hybrid retrieval + RRF candidate fusion
  → native authenticated Cohere-compatible HTTP adapter
  → http://127.0.0.1:8123/v1/rerank (unique rerank credential)
  → shared local-gpu reservation
  → http://127.0.0.1:8185/v1/rerank
  → pinned Qwen/Qwen3-Reranker-0.6B, CUDA FP16
  → complete indexed scores + actual input/padded token usage
  → response validation + usage / elapsed-time accounting
  → Hindsight ranked facts → canonical support and caller checks
  → bounded broker packet → Hermes memory provider
```

RRF is still useful for candidate fusion; learned ranking now follows it. The native provider label is `cohere` because that supported adapter sends bearer credentials to an exact custom endpoint. No Cohere cloud model, SDK, cloud URL or cloud key is used. Native TEI did not offer the required credential header. We extended our core model service with this protocol, rather than modifying installed Hindsight.

## Core changes and operational configuration

- [Config](../src/hermes_memory/config.py) and [routes](../src/hermes_memory/processing/routes.py): optional owned reranker route, unique credential, `local-gpu`, interactive priority, no completion-token cap on scoring.
- [Gate](../src/hermes_memory/processing/gate_server.py): credentials are bound to operation/path; scoring requests cannot name upstreams. Require pinned model and bounded documents; reject missing, duplicate, invalid or nonfinite scores and invalid actual usage. Charge measured dispatch time and padded compute tokens. Malformed accounting is marked malformed; absurd unvalidated usage cannot poison the budget.
- [Model service](../src/hermes_memory/models/reranker.py): retains TEI `/rerank` and adds local Cohere-compatible `/v1/rerank`. Per-job counts stay with the job, not mutable last-request state. No additional Torch dependency in the base framework.
- [Owner configuration](../src/hermes_memory/install/reranking.py): reviewed paired private env files; owner-only, stopped services, separate route credential, private backups and atomic publication with rollback on write failure. Plans/receipts never print credentials.
- [Native client](../src/hermes_memory/backend/hindsight_client.py): optional `reflect(..., include_support=True)` reads exact cited memory identities in the same bank. Resolve observation source identities recursively; bound to 64 identities, depth 8 and 256 KiB. Refuse cycles, identity drift, invalidation and path injection. Hydration is not authority: SupportResolver and caller checks remain required. Ordinary foreground recall is unchanged and incurs no hydration requests.

Native candidate caps are low=16, mid=32, high=64, with a 3-second native rerank HTTP timeout and no configured provider fallback. Interactive memory requests do not queue behind an occupied GPU; they degrade explicitly. A caller timeout does not prove model execution stopped, so the gate retains the execution reservation until an actual response establishes completion.

The existing GPU profile remains FP16, 2,400 MiB allocator cap, 768 MiB startup reserve, maximum batch 128, padded batch budget 32,768 tokens, one inference worker, bounded queue and retained workspaces. The vision model remains at 16,384 context / one slot. The reranker unit is now enabled at boot. It only governs memory-originated requests through this gate; other direct vision/Ollama consumers are not magically serialized by it.

## Validation

- Full framework regression: **3,218 passed, 1 skipped**, 64.70 seconds. The skip is not a completed test. Focused bridge/model/gate/installer/summary tests also passed.
- [Synthetic flow harness](../evals/check_hindsight_reranked_flow.py): six canonical records with competing drink, travel and allergy facts; nine English, Roman Hinglish and Devanagari Hindi questions. Tests both native recall and the actual ContextBroker path, resolving expected source identities rather than checking only result counts.
- All nine queries ranked the expected canonical source first and delivered its supported semantic fact through the broker. English questions retrieved Hindi evidence; Hindi questions retrieved Hinglish/English evidence.
- Hiding evidence while its Hindsight projection still exists excludes it from delivered lexical and semantic context. Expected partial provenance after hiding is degradation reporting, not learned-reranker failure.
- Native reflection invoked learned recall, returned an answer mentioning the expected restriction, and its exact-ID hydrated citations resolved to canonical evidence.
- Explicit reranker outage: semantic channel unavailable, no semantic facts delivered, lexical memory still usable. Restart recovered learned ranking and the expected supported fact. Hidden evidence remained withheld.
- Every synthetic remote document is deleted and checked absent. Empty bank shells and durable local synthetic reconciliation ledgers are retained. No personal corpus or Telegram send was used.
- Installed Hermes gateway/provider probe passed profile A → B → A isolation with synthetic reply context and zero network calls. This is a separate host integration probe, not a live Telegram conversation.
- Final shared-GPU checks also passed after outage recovery. The first repeated vision prompt was heavily cached, so a second run used unique initial prefixes and verified **zero cached prompt tokens**. Its long round processed 13,091 vision prompt tokens plus an image, eight 1,024-dimensional embeddings, and 16 long reranker candidates concurrently. Vision=8.753 s, embeddings=.489 s, reranking=4.782 s; sampled free VRAM minimum=859 MiB. Receipt: `/home/jugaadu/data/model-benchmarks/hindsight-reranker-installed-fresh-coactivity.jsonl`. This is a coexistence stress check via direct model interfaces, not a foreground latency promise or a bypass used by Hindsight.

The first harness incorrectly looked for semantic facts in `Packet.items` instead of `Packet.facts`; it was corrected. Another recovery assertion incorrectly required `available` after deliberately hiding evidence; it now requires successful learned ranking and the expected supported fact while accepting truthful partial provenance. Early failed receipts remain available, rather than being presented as successful tests.

Pre-final-release complete live receipt: `/home/jugaadu/data/model-benchmarks/hindsight-reranked-flow-complete.json`: 22 successful gated rerank calls, 15,855 padded tokens, 44.087 seconds including formation, reflection, outage/recovery and cleanup.

Final activated-release receipt: `/home/jugaadu/data/model-benchmarks/hindsight-reranked-flow-installed.json`: **passed=true**, 9/9 query checks, 23 successful gated rerank calls, 19,056 actual padded tokens and 1.951 charged dispatch seconds. Each native-recall-plus-broker check took 0.3932–0.6602 seconds; total 54.4441 seconds includes formation, native reflection, outage/recovery and verified six-document cleanup. Recovered semantic channel is explicitly partial because hidden projection evidence was withheld, while Nisha's supported fact was delivered.

These are small-corpus timings, not production tail-latency guarantees. Final reflection also mentioned weight/diet records not present in the fixture; this reinforces the entailment/publication limitation below rather than turning its retrieval/support check into a faithfulness claim.

## Activated release identity

- `runtime/current` → `/home/jugaadu/data/hermes-memory/runtime/qwen-rerank-verified`.
- Immutable plugin registration pin: `b10b4944945690b2745e1fedcdf03288a36929b9`.
- Source input digest: `c667215f3f9f42946c734579719de197241eb535187315b4ca5a8b00cd86a03a`.
- Wheel digest: `5c098e582a205a5f6ab868f92f91cbda4d2916f455f631d298bc42f039592f03`.
- Runtime/worker/provider framework digest: `997b9e63e0f7751f74fa2cd5c6ee440c070caf7f6fcf099edabbf49d79ac12ca`; canonical schema 17, gate schema 3, native API 0.10.1.
- Core activation backup/receipt: `/home/jugaadu/data/hermes-memory/activation-backups/activation-ldejfsqa`.
- Paired private routing backup: `/home/jugaadu/data/hermes-memory/activation-backups/reranking-34umsu2m`.
- Post-activation doctor: `ok=true`, release/compatibility matching; historical queue-age and uncertain-delivery warnings remain. Installed host A → B → A probe passed again on this release.
- Final gate health `ok=true`; Hindsight health `healthy`; memory runtime, backend, worker, Qwen reranker and Hermes gateway all active. Reranker boot enablement verified.

## Limits and remaining acceptance work

Subsequent release: scoped faithfulness protection and the Hermes checking tool are now installed in `runtime/faithfulness-verified`; the activation identity above is historical. See [the implementation receipt](memory-faithfulness-implementation.md). Native free-form reflection remains untrusted; the limitation below records the original integration finding.

This validates the integration flow, not all P00–P14 intelligence requirements or general human-level memory. Six synthetic records do not establish large-corpus relevance, production p95 latency or fairness under sustained load. Existing GPU/CPU batching comparisons remain in [the benchmark report](reranker-gpu-benchmark.md).

The native reflection check establishes retrieval and source support, not sentence-level entailment. One synthetic response added generic allergy-severity/cross-contamination discussion absent from the stored statement, and `based_on` included retrieved facts beyond the directly relevant one. Do not turn the whole native answer into a canonical memory or claim every generated sentence is proven. The framework's scoped synthesis remains the publication path; native reflection is not an authority bypass. **Research-review clarification:** scoped synthesis currently validates claim structure and supplied citation identities, not whether evidence entails the claim text. Stronger semantic verification is therefore needed for scoped publication as well as native reflection. See [research and recommended design](memory-faithfulness-research.md).

Consolidation/association quality at scale, held-out multilingual tests, session-level production conversations and long-running mixed-device tail latency remain separate acceptance gates. Telegram delivery is still paused. The doctor's old waiting-job and uncertain-delivery warnings are historical state, not resolved by reranker activation.
