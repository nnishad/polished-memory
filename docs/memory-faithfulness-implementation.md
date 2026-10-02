# Memory claim faithfulness: implementation and validation

Date: 2026-10-02. Follow-up to [the research recommendations](memory-faithfulness-research.md). This implements the core synthesis/publication protection and a caller-scoped Hermes checking tool. It does not complete every recommendation or guarantee faithful prose from every model.

## Installed core changes

- `processing/faithfulness.py`: bounded canonical evidence/atomic claim contracts; exact source quotes with locally resolved offsets and revisions; strict complete verifier verdicts; source/claim/rendering digests checked again before publication. A model-generated verification attestation is not accepted as a claim field.
- `processing/synthesis.py`: evidence-first structured generation followed by a separate entailment call through the owned gate. Each claim is classified as supported, contradicted or insufficient evidence. Only supported claims are rendered. All-rejected results, malformed/truncated responses and unavailable verification withhold publication. There is no unverified fallback or automatic repair loop.
- Deterministic vetoes additionally reject new numeric details, unsupported severity and explicit strengthening of uncertainty. Their cross-language vocabulary is deliberately limited: these are conservative safety rules, not comprehensive semantic understanding. Semantic checking is still required.
- Generation offers finite exact canonical quote options in the output schema. Long sentences are offered as bounded source substrings. This prevents quote translation/drift when the configured server honors the grammar; local exact-membership validation still enforces the boundary if it does not.
- Structured model calls disable hidden thinking on the owned llama.cpp-compatible route, so bounded JSON output budgets are not exhausted before verdicts are emitted. No new resident model, GPU allocation or cloud endpoint was added.
- `processing/summarization.py`: a changed plan/fingerprint contract accounts conservatively for both model hops. Publication rejects absent receipts, changed evidence and prose added after checking. The existing watermark check still withholds output if authority changes during inference. Actual successful-hop token counts include generation and verification; the gate accounts reached failed hops separately.
- `processing/verification.py`: a read-only canonical-memory service for explicit draft claims. It resolves records itself, applies existing confirmed-identity scope/visibility rules and checks watermark changes during inference. It respects the operator inference pause and shared daily budget before dispatch. Caller-provided record text or authority is never trusted.
- Hermes provider: new `memory_verify` tool, manifest declaration, static guidance to check paraphrased personal-memory assertions and keep tentative inference/general advice separate. Rendered canonical evidence now includes its record ID so claims can identify their sources. Important unanswered questions use the existing clarification flow rather than becoming invented facts.

## Failures reproduced during development

1. Initial semantic-only live check incorrectly accepted “severe” from a plain peanut-allergy statement and Hinglish “pakka” from Hindi “शायद.” Both have regression coverage and deterministic vetoes. Fixing these examples is not evidence that every possible exaggeration is caught.
2. Hidden reasoning exhausted some bounded model outputs. Explicit non-thinking structured calls resolved the observed failure; truncated outputs remain fail-closed.
3. Hindi generation either omitted uncertain statements or changed an evidence quote. Prompt clarification preserves explicitly uncertain assertions, and finite canonical quote options prevent the observed quote drift. Invalid quotes are never silently translated or repaired.
4. Full-flow testing initially stopped because the shared daily budget was exhausted (205,384 used against 200,000). The owner explicitly approved increasing the configured budget to 300,000 for testing. The current setting remains 300,000; no ledger was reset or usage erased.

## Validation receipts

| Check | Result | Scope |
| --- | --- | --- |
| Full source regression | 3,263 passed, 1 skipped; 65.33 seconds | Offline contracts and isolated fixtures, not live language accuracy. |
| Expanded synthetic live verifier | 25/25 unsupported rejected; 16/16 supported retained | 41 human-authored English/Hindi/Hinglish examples, including additional names/reporting/uncertainty cases; not a calibrated representative corpus. |
| Installed verifier repeat | Same 41-case pass | Installed wheel, configured admission gate and existing remote text model. Three verifier batches took 1.7211, 1.6183 and 1.1718 seconds; these are not production p95 measurements. |
| Synthetic native memory flow | Passed; 9/9 recall queries, 22 learned rerank dispatches | Source coordinator against live Hindsight/Qwen: formation, canonical resolution, checking three candidate claims, verified summary publication and hidden-source exclusion. All six remote test documents verified absent afterward. |
| Installed Hermes profile/gateway probe | A → B → A passed; zero network calls | Installed provider and framework against real host imports, synthetic homes and simulated transport context; no Telegram delivery. |
| Post-activation services/doctor | All five services active; doctor `ok=true` | Memory runtime, Hindsight, worker, reranker and Hermes gateway. Historical queue-age and uncertain-delivery warnings remain. |

Raw receipts:

- `/home/jugaadu/data/model-benchmarks/memory-faithfulness-core.json`
- `/home/jugaadu/data/model-benchmarks/memory-faithfulness-installed.json`
- `/home/jugaadu/data/model-benchmarks/hindsight-faithfulness-flow-core.json`

The native reflection check is still only retrieval/provenance validation. The additional full-flow claim test checks deliberately supplied candidate claims using native-resolved canonical sources; it does **not** decompose and verify every sentence of the native reflection. Native free-form reflection remains untrusted derived prose. The live scoped summary passed the new publication contract, but model verdicts are not independent human annotations.

## Normal installation receipt

- Active managed release: `/home/jugaadu/data/hermes-memory/runtime/faithfulness-verified`.
- Matching runtime/worker/host framework digest: `9a651c38376ec5e13fb096c87f4e52163ac9c41cb757fc0f31e56caf509bfc86`.
- Reviewed frozen source digest: `9d37fa0002a7eac0c217a2fb2164690a06df1dcb94e758b151d512490c71b94f`.
- Wheel digest: `c7ae9f0103e5061b7f959148b3dc638cb12d869d3ab3892bd4abbb679fafcf1f`.
- Immutable plugin registration pin: `1788db91cccebe1f3cf5e9f02da4b02397b27224`.
- Core activation backup: `/home/jugaadu/data/hermes-memory/activation-backups/activation-32o7z02n`.
- Schema remains 17, gate schema 3, native API 0.10.1. No hand-edited installed packages or user-repository commit. Existing unrelated source changes were preserved in the reviewed working-tree snapshot.
- Telegram delivery pause was checked after activation and remains true. Synthetic remote documents were removed; their isolated local reconciliation ledgers remain for audit/recovery.

## Remaining work: do not call this the full faithfulness plan

1. **Mandatory final-answer boundary:** Hermes's current memory-provider contract has no mandatory pre-send validation hook. The new tool is agent-directed. A versioned host-core extension must cover streaming and non-streaming paths before claiming that every conversational answer is checked before delivery. Prompt guidance is not enforcement.
2. **Coverage of native derived prose:** free-form reflection, observations/consolidation, legacy summaries and agent-written `memory_remember` content are not made semantically safe by the new summary producer. Legacy summaries are not retroactively verified. Canonical typed promotions continue to require their existing owner decisions; do not infer new automatic authority from a verifier verdict.
3. **Independent multilingual evaluation/model choice:** the same text model generates and judges. Correlated errors remain. No MiniCheck deployment or independent Hindi/Hinglish calibration was performed. Add a representative held-out human-labeled corpus, assess false acceptance and supported-claim retention, then compare CPU verifier candidates.
4. **Performance work:** no persistent verdict cache or bounded repair was added. Omission/abstention is the current safety path. If introduced later, cache keys must include evidence revisions, scope, verifier and policy, with current visibility checks; repair must be reverified.
5. **Operational policy:** budget admission is a pre-dispatch estimate through the existing shared-ledger design, not a new atomic multi-hop quota reservation. Sustained concurrent load and final-response latency remain to be measured.

No claim of zero hallucinations, human-like memory completion or general medical correctness is made.
