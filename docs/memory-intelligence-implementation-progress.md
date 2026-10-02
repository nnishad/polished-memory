# Memory intelligence implementation progress

Date: 2026-10-02. Scope: ongoing core-source implementation from the [implementation plan](memory-intelligence-implementation-plan.md). Existing unrelated working-tree changes were preserved. Sections 1 and 3 retain the first-slice receipt; section 5 records the continuation.

**Historical exception:** an earlier isolation error caused unintended production writes; see the [incident and owner repair direction](memory-intelligence-live-test-incident.md). Subsequent core activation resolved the old installed-framework/schema mismatch. The current release and faithfulness testing are recorded in [the latest implementation receipt](memory-faithfulness-implementation.md). **The full P00–P14 plan and mandatory conversational-answer verification remain incomplete.**

## Latest: scoped faithfulness protection

Installed `runtime/faithfulness-verified` through the reviewed core workflow. Adds exact canonical quote contracts, separate per-claim semantic checking, deterministic severity/uncertainty/quantity vetoes, checked-rendering publication receipts and the caller-scoped Hermes `memory_verify` tool. Full regression: 3,263 passed, 1 skipped. Installed synthetic smoke: 25 unsupported rejected, 16 supported retained. Native Hindsight/Qwen flow passed, including guarded summary publication and hidden-source exclusion. Telegram delivery remains paused. This does not enforce every final conversational response or retroactively verify native/legacy derived prose; see [remaining boundaries](memory-faithfulness-implementation.md).

## 1. Implemented in this slice

### Versioned utterance context without changing legacy evidence

- New conversation-turn spool payloads declare source_context_version 2.
- The native turn-start hook records a host-receipt clock for the user's utterance. The later spool/completion clock is not substituted for it.
- Captures without a new turn hook have no utterance anchor. Initialization, session switches, and successful capture clear the prior clock. A replay uses the original persisted spool payload.
- The source SDK decodes versions explicitly, refuses unknown/malformed versions, and adds v2 metadata only to v2 events. Legacy v1 metadata/fingerprint semantics remain unchanged.
- The user anchor is not assigned to the assistant's answer. Missing, date-only, timezone-less, malformed, and oversized utterance timestamps stay unknown.
- Canonical occurred_at remains separate: a reliable utterance timestamp does not turn an undated preference or statement into a dated real-world event.

Source: [provider turn boundary](/home/jugaadu/Projects/hermes-memory/integrations/hermes-memory/provider.py:451), [capture](/home/jugaadu/Projects/hermes-memory/integrations/hermes-memory/provider.py:461), [SDK decoder/context](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/sources/sdk.py:152), [timestamp normalization](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/sources/base.py:146).

### Native single-record Hindsight extraction context

- Retain submissions now carry a bounded whitelist of role, origin, independence, session/event identity, revision/kind, and timestamp basis. Only author display name/ID are copied from the author object; arbitrary metadata and credentials are excluded.
- Native context distinguishes user statements from assistant/model output, tools, host instructions, and mirrored notes. First-person model text is not described as the owner's preference.
- With no event date, a validated v2 utterance timestamp can serve as Hindsight's extraction anchor, explicitly labeled as an utterance anchor rather than event validity. The speaker's local timezone is not inferred from the host's UTC clock.
- Unknown dates are labeled unknown. Actual language/date extraction remains to be verified with pinned prompt previews and the held-out corpus; a prompt instruction is not a correctness guarantee.
- The retain-context contract version is included in new formation processor fingerprints. This does **not** reselect old verified projections; generation-aware rebuilding remains P04 work.
- No multi-record conversational episode was introduced. Single-record provenance is preserved until full input manifests exist.

Source: [retain item construction](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/worker.py:397), [processor identity](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/formation.py:69).

### Caller-aware summary retrieval and dependency filtering

- Default SummaryStore provenance resolution now uses the canonical identity store.
- latest accepts the authenticated account_id and resolves each candidate with that caller, without falling back to an older unsupported revision.
- The broker passes its caller into summary lookup and checks every returned source dependency against the request's source/time filters before packing the summary.
- Unknown/incomplete source support is not packed merely because one lexical hit shares a project scope.
- This corrects the summary API/broker boundary. Hermes's existing automatic prefetch still does not establish an account_id from a reviewed host-principal binding; it must not guess one or silently widen access. That remaining end-to-end availability question is explicitly open below.

Source: [summary store](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/knowledge/summaries.py:80), [caller-aware latest](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/knowledge/summaries.py:234), [broker summary support checks](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/context/broker.py:417).

## 2. Finding/package status

| Finding / package | Source status | Remaining work | Installed status |
| --- | --- | --- | --- |
| P00 / M-034 | Initial regression baseline and new deterministic fixtures implemented | Language corpus, backend-quality runner, telemetry/latency/cost baseline | Not installed |
| P01 / M-004 | Role/session/author whitelist and native extraction context implemented | Full host context contract, quote/forward/bot treatment, pinned prompt-preview and quality validation; contextual episodes after P04 | Not installed |
| P01 / M-005 | New user-turn receipt anchor implemented separately from event occurrence | Historical enrichment only with evidence; speaker timezone and old transcript timing; relative-date quality evaluation | Not installed |
| P01 / M-024 | Summary store/broker caller propagation and all-dependency source/time checks implemented | Reviewed authenticated Hermes principal → source-account binding and end-to-end scoped availability | Not installed |
| P04 / M-006 | Versioned processor manifest and durable retain payload contract/digest implemented | Full generation registry, verified-record reselection, historical reconciliation, shadow-bank cutover | Not installed |
| P02 / M-001 | Explicit task policies and scoped consolidation client implemented | Separate stage grants, governed scheduling, real poller/cancellation integration proof | Not installed |
| P03 / M-002–M-003 | Bounded support resolution and explicit-input synthesis implemented | Faithfulness evaluations, persisted per-claim support and full end-to-end quality evidence | Not installed |
| P05 / Unicode query preprocessing | Combining marks preserved without rewriting evidence or indexes | Vector/template/instruction contract, transliteration and held-out language evaluation | Not installed |
| Other packages/findings | Not implemented in this slice | Follow the dependency/benchmark gates in the plan | Not installed |

## 3. Verification

New regressions cover legacy replay, v2 replay fingerprint stability, utterance/event separation, user/assistant anchor separation, missing/ambiguous/oversized clocks, unknown versions, session/capture clock clearing, metadata whitelisting, assistant attribution, processor-contract identity, caller-specific summary reads, and every-dependency source filtering.

- Pre-change targeted baseline: 249 passed.
- The scoped summary and v2/retention tests were first exercised against the old implementation and reproduced missing metadata/caller failures. Test-fixture mistakes were corrected before implementation results were accepted.
- An intermediate full suite passed with 3,062 passed and 1 skipped before the final one-shot capture-clock guard.
- Final-state full suite: **3,063 passed, 1 skipped**, in 54.78 seconds.
- Final targeted provider/SDK/formation/worker/broker/summary suite: **577 passed, 1 skipped**, in 5.01 seconds.
- The full suite needs temporary loopback sockets for isolated installer fixtures. A sandbox-denied socket attempt was retried with explicit escalation; this does not perform a production installation.
- git diff --check passed after source edits. Existing dirty changes are not being represented as new work from this slice.

No live model-quality, prompt-preview, production latency, gateway delivery, or installed-runtime test was performed. Offline green tests are not proof of Hinglish extraction accuracy or deployment health.

## 4. Next implementation batch and deployment blockers

1. Complete P01's authenticated caller binding and remaining source-context invariants using host-provided identity, not a model/tool account argument. Keep unknown scope fail-closed.
2. Complete P02's separately governed stage grants and actual poller integration checks. Do not enable consolidation merely because its task policy/client now exists.
3. Evaluate P03's explicit-input synthesis for claim faithfulness and multilingual quality; structural citation validation alone cannot prove entailment.
4. Implement P04's generation/manifest/reconciliation contracts before reprojecting existing evidence or changing vector indexes.

**Do not deploy this slice by itself.** In particular, changed retain payloads must be reconciled with durable pre-existing submissions before upgrade. Existing verified mappings are not refreshed by the new fingerprint alone. Prompt previews, full caller binding, generation migration, earlier source-fix deployment, and the plan's lifecycle/installer gates remain prerequisites for an approved installation.

Keep source-tested, release-tested, and installed-tested statuses separate in subsequent updates. No finding is fully closed merely because part of its implementation exists.

## 5. Continuation: core safety and intelligence contracts

### P02: pinned task policies, not aliases or new authority

- `TASK_POLICIES` distinguishes DB operation type from executor payload type. Native consolidation uses the consolidate route; mental-model refresh uses reflect. Mismatched/unknown kinds are refused, including malformed payload types.
- Native graph and vector-index maintenance receive a DB-only resource and a 300-second execution ceiling. They do not borrow the retain GPU route. The inspected pinned graph pass has a 240-second native time budget; vector-index maintenance reconciles PostgreSQL indexes without model inference.
- Added the pinned consolidate capability and strict nested `observation_scopes` request contract. Empty/unbounded-bank scopes are refused. Capability availability is not an allowance or scheduler permission.
- Tests cover native-shaped task kinds, payload mismatch, DB-only classification, and scoped consolidate request validation. Full cancellation/folded-task/poller integration and separate grants remain open.

References: [task policy](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/worker_launcher.py), [capabilities](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/capabilities.py), [client contract](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/hindsight_client.py), [worker contracts](/home/jugaadu/Projects/hermes-memory/tests/contracts/test_worker_launcher.py). Pinned evidence: `hindsight_api/worker/poller.py` operation-type injection and `engine/memory_engine.py` executor handlers, including `vector_index_maintenance`.

### P03: observations retain complete canonical support

- Recall preserves source-fact dictionary keys as IDs and rejects conflicting identities.
- A shared local resolver walks declared fact/document/record dependencies with depth, reference-count and manifest-byte bounds. Cycles, missing nodes, malformed references, cross-bank mappings, unverified projections, stale epochs/revisions and hidden evidence fail closed.
- The broker authorizes every resolved canonical dependency against caller/source/time constraints before packing a fact. A valid direct record cannot launder invalid additional support. Packed facts retain resolved record IDs.
- No support-resolution HTTP calls or private-text duplication were added. This is current-bank/current-epoch validation, not the still-pending generation registry.

References: [resolver](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/support.py), [broker](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/context/broker.py), [adversarial tests](/home/jugaadu/Projects/hermes-memory/tests/unit/test_backend_support.py).

### P03: canonical scoped synthesis and actual citations

- Summary generation no longer performs unrestricted Hindsight bank reflection. It supplies the reviewed canonical records (identity, revision, speaker role, text, source and occurrence time) to an explicit model through the owned loopback admission endpoint.
- Plans require explicit model/admission configuration and bound serialized input at 24,000 bytes. Token admission estimates reflect the selected evidence instead of being lowered to fit a budget. Canonical selection coverage does not falsely depend on projection coverage.
- Model output must contain bounded claims citing only supplied record IDs. Publication uses the actual union of cited records, and the report separately identifies planned-input count. Unsupported/malformed claims and malformed usage are controlled failures.
- Archive/authority watermark changes during synthesis prevent publication. Production hop accounting belongs to the admission gate; the coordinator does not charge it twice or reserve its device recursively.
- Summary processor identity includes the new strategy version, input ceiling and explicit model. A native reflection-only client cannot silently bypass the scoped contract.
- Structural support is not a semantic entailment proof. Per-claim durable manifests, claim-faithfulness scoring, full speaker/date/negation accuracy and real-model tests remain pending.

References: [synthesis adapter](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/synthesis.py), [summary producer](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/summarization.py), [adapter tests](/home/jugaadu/Projects/hermes-memory/tests/unit/test_scoped_synthesis.py), [publication tests](/home/jugaadu/Projects/hermes-memory/tests/integration/test_summarization.py).

### P04 prerequisites: reproducible retain bytes before rebuilding

- Source migration `0016_retain_payload_contract` adds a formatter-version marker and payload digest to backend mappings. It has only been exercised on disposable test databases; production has not been migrated.
- New mappings explicitly declare retain-context version 2. A known declared version 1 uses the frozen legacy formatter, preserving its original operation ID and bytes. Pre-migration mappings default to **unknown version 0**, not a guessed historical format; an unknown pending submission is withheld for reconciliation.
- The canonical revision must match the mapping. Before submission, a digest is pinned; reuse with different extraction bytes is refused. An identical retry does not bump the context revision or store another private-text copy.
- A versioned processor manifest now records extraction/context versions, bank, routes, explicit models, endpoints and output cap without credentials. Unknown bank prompt configuration and embedding dimensions remain explicitly unknown. Model/context changes alter identity; credential rotation does not.
- This does not implement generation-aware rebuilding, automatically reselect verified mappings, adopt unknown historical contracts or switch active banks. The existing verified-mapping shortcut remains a rebuild availability concern to address in P04.

References: [mapping contract](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/document_map.py), [migration](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/storage/migrations.py), [worker formatter](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/worker.py), [processor manifest](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/formation.py), [retry tests](/home/jugaadu/Projects/hermes-memory/tests/unit/test_worker.py).

### Unicode retrieval and warm-context invalidation

- Query token preprocessing now preserves Devanagari combining marks and in-token joiners. Tests cover intact Hindi words, exact lexical lookup and Hinglish negation/identifier tokens. Original evidence, IDs, SQLite tokenizer/indexes and the current AND matching policy are unchanged.
- Hermes's rendered warm packet now carries the canonical epoch/revision fence and is discarded if evidence changes before consumption. A regression hides a remembered record between warming and injection and verifies it is not returned.
- This is not proof of embedding-server instructions, transliteration recall or Hinglish model quality. Warm-packet time-based authority expiry and a reviewed host-principal/source-account binding are still separate work; the watermark fence must not be described as covering every expiry policy.

References: [Unicode terms](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/storage/evidence.py), [warm cache fence](/home/jugaadu/Projects/hermes-memory/integrations/hermes-memory/provider.py), [provider regressions](/home/jugaadu/Projects/hermes-memory/tests/hermes/test_hermes_plugin.py).

### Continuation verification and remaining roadmap

- Intermediate full suite: 3,096 passed, 1 skipped, 57.53 seconds.
- Later targeted safety/formation/summary families: 403 passed.
- Final synthesis/summary/provider targeted suite: 220 passed, 1 skipped, 2.16 seconds.
- Final full suite: **3,114 passed, 1 skipped**, 57.68 seconds. No live models were called.
- Regenerated the source compatibility artifact for schema 16 and current source identity. Post-regeneration installer/worker/client contracts: **569 passed**, 15.06 seconds. `git diff --check` passed.

Still pending: P00 corpus/telemetry and live measurements; P01 remaining source contexts and authenticated caller binding; P02 grants/scheduling; P03 semantic faithfulness and durable claim support; P04 generations/rebuild/cutover; P05 embedding contract and multilingual optimization; P06 automatic salience/assertions; P07 provider-priority/deadline work; P08 packing/modes/coverage; P09 evaluated consolidation/associations; P10 reranker benchmark; P11 typed goals/measurements/learning/informational clarification; P12 scale/fairness; P13 media; and P14 lifecycle/release activation. These are not marked complete by passing existing regression tests.

No production writes, model calls, service changes or installation were performed. Installation remains gated by the plan, not automatically authorized by this source implementation.

The preceding sentence is the historical section-5 receipt, not the section-6 live-test outcome.

## 6. Full-plan continuation and live-validation incident

Implemented further core-source work:

- Source migration `0017_projection_generations`: explicit legacy backfill, versioned generation manifests, shadow banks, active-binding uniqueness, per-document input manifests, and generation/target-bank fields on jobs.
- Owner-reviewed prepare/cutover ledger, coverage and lifecycle gates, distinct banks for changed processors, unchanged canonical revisions, retired-generation recall refusal, current binding resolution in Hermes, and generation-partitioned dispatch/reconciliation.
- `rebuild` planning/prepare/activate CLI and `form --generation`; planning opens read-only stores. Cutover remains blocked with unknown vector/prompt contracts or unfinished jobs/erasure duties.
- Stale verified mappings and acknowledgements cannot silently become current-epoch coverage. Hidden evidence does not create a new projection intent. Reset includes registry banks; recovery retains nonlegacy bank mappings instead of discarding them as unrelated banks.
- Authenticated private Telegram caller resolution uses existing `source_account / telegram:<user ID>` accounts only, without registration or inferred links. Bot/forward/group/internal/unattested/wrong-home/user turns remain unknown. Caller scope reaches tool recall and prefetch; warmed renders also compare current identity-group authority.
- Allowances distinguish formation/consolidation/synthesis/assertions/media stages. Cross-stage IDs cannot authorize, consume or revoke another stage. Wildcard grant conflict checks now also run inside the write lock. Atomic pass reservations and governed producers are still pending.
- Synthesis has an explicit schema-output contract with local citation/shape/size validation. The served model ignored plain JSON formatting; its grammar rejected richer bounds. A portable schema subset succeeded through owned admission. No fragment extraction, reasoning-output interpretation or automatic inference retry was introduced. The two supported dialects are explicit adapter options, not a server-detected permission fallback. [Primary structured-output reference](https://github.com/ggml-org/llama.cpp/blob/master/scripts/server-test-structured.py).
- Added an opt-in synthetic smoke runner and subsequent fail-closed path/bank isolation regressions after the incident.

Verification:

- Shadow-generation contracts: 8 passed.
- Worker/jobs/formation/provider targeted suite: 329 passed, 1 skipped before the later caller tests.
- Final isolation/generation/synthesis/provider targeted suite: 171 passed, 1 skipped.
- Final full suite after compatibility regeneration: **3,141 passed, 1 skipped**, 63.91 seconds. `git diff --check` passed.
- Three synthetic synthesis cases returned structurally valid, manually inspected negation-preserving claims in Roman Hinglish, Hindi and English, approximately 1.3–2.1 seconds per case. This is not a held-out extraction/recall quality score.
- The retain/recall run is **invalid as an isolated evaluation** and must not be reported as a successful benchmark. Its exact unintended writes, failed operation, containment, absent document, pending metadata and repair needs are recorded in the incident document.

P04 is still incomplete: contextual/multi-record manifests, durable generation-preserving restore/cutover proof across all crash boundaries, embedding/prompt validation receipts, live shadow backfill and supported production activation need further work. Earlier P00/P05 language-quality, P06/P09 governed producers, P07 lifetime/public-adapter spike, P08 packing, P10 optional reranker, P11 informational clarification/typed context, P12 scale, P13 media and P14 release gates remain open.

**Historical next action at the end of section 6:** owner repair direction was requested. The owner subsequently authorized continued repairs and confirmed all existing data is disposable. Section 7 supersedes the authorization blocker. Do not downgrade schema, edit migration history or blindly restore a snapshot.

## 7. Continued core repairs after disposable-data authorization

The [continuation repair register](memory-intelligence-repair-register.md) records I-001–I-009 with severity, consequences, core fix, regression names and staging evidence. Changes address deadline-aware polling, local claim detachment/cancellation, generation-preserving restore, stale synchronous reconciliation, reviewed frozen release inputs, explicit working-tree snapshot staging, source/target validation, service path boundaries, and mutable setuptools build metadata.

- Poll configuration is validated before queue or projection writes. Synthetic retain checks have a local maximum of 30 two-second polling intervals per job; this is not an end-to-end HTTP or native inference deadline guarantee.
- Restore retains empty and active post-snapshot generation identities, retires all generation histories when advancing the restore epoch, and therefore does not infer current coverage from an older bank.
- Source staging binds runtime/build/artifact bytes and permission modes into its reviewed digest. A separate build workspace prevents setuptools from rewriting the retained snapshot. No user repository commit or installed-package edit was performed.
- Real staging initially exposed a metadata-rewrite mismatch. The first candidate remains explicitly invalid and unactivated. The fixed workflow successfully staged `/home/jugaadu/data/hermes-memory/runtime/snapshot-820f2a988d824ac7`, with full core artifact verification passing.
- Independent staged interpreter checks verify matching runtime/backend framework digests, schema head 17 and Hindsight 0.10.1. A synthetic temporary database passed all migrations and SQLite integrity checking. This was not a migration or inference test against the live canonical corpus.
- Final full regression suite after source compatibility regeneration: **3,168 passed, 1 skipped**, 64.36 seconds. `git diff --check` passed.

No existing data was deleted. The release pointer, running services and registered Hermes provider remain unchanged, so the installed schema incompatibility is still pending. Next work is tested core activation/host pairing followed by isolated live validation, not manual pointer changes or installed-code patches. P00–P14 acceptance gates remain incomplete; neither disposable data nor passing regression tests establishes multilingual quality or production readiness.

## 8. Core activation and first isolated Hindsight round trip

This section supersedes section 7's unchanged-runtime status.

- Added a core `activate-release` workflow with owner/review gates, immutable snapshot registration pins, exclusive activation locking, owned-service quiescence, private backups, isolated migration rehearsal, official Hermes plugin installation, host/framework pairing checks, live forward migrations and atomic publication. Failure leaves services stopped for forward repair rather than restarting older code against a newer schema.
- Snapshot registration creates a deterministic Git pin only in the new release's retained source, not a commit in the user's dirty checkout.
- Focused activation/release tests: 46 passed. Final full regression: **3,172 passed, 1 skipped**, 64.56 seconds. `git diff --check` passed.
- Activated `/home/jugaadu/data/hermes-memory/runtime/snapshot-a0840de1caad49b5`; registration pin `83817e21bf43a78f188b08be72298ef0837af1fb`. Core receipt: `/home/jugaadu/data/hermes-memory/activation-backups/activation-8vyvw20b`. The activation receipt deliberately says `started-unverified`; subsequent checks, not that receipt alone, establish service availability.
- Gate, Hindsight API, worker and Hermes gateway are active. Gate `/health` and Hindsight `/health` returned healthy, with the latter reporting a connected database.
- The isolated Hindi/Roman-Hinglish/English round trip succeeded for all three formation jobs, with no unprojected records remaining. Recall returned three results per query, in 0.14–0.33 seconds; total run 8.93 seconds. All three remote documents were verified absent after cleanup. Bank shell and durable synthetic ledger remain for audit.
- The smoke receipt does not expose enough semantic detail to establish faithfulness; it is not a held-out language benchmark, graph/consolidation test, Hermes injection proof or learned-reranker test.
- Outbound memory delivery was explicitly paused before activation and remains paused. No real Telegram message was sent.
- The owner confirmed no reranker exists and requested a local headroom check. Measured capacity and the CPU-first recommendation are recorded in [local reranker capacity](local-reranker-capacity.md). Learned reranking remains pending; configured RRF must not be called learned reranking.

Full P00–P14 acceptance remains incomplete. Next validation must cover semantically asserted recall, actual provider context, governed consolidation/associations, and a measured learned-reranker deployment through owned resource admission.

## 9. Resident GPU reranker, capacity profiles and quality checks

- Added an optional core model-service module using pinned Qwen3-Reranker-0.6B weights, exact template/yes-no scoring, FP16 SDPA, last-position logits and no conversation KV cache. Torch/transformers are installed only in a separate model environment. No installed upstream package or Hermes provider was patched.
- Added length-aware padded-token batching, original-index restoration, finite score checks, loopback HTTP limits, one GPU executor, bounded admission, query-preserving document-tail truncation, and explicit capped allocator-workspace retention. Queue cancellation/completion/failure regression tests verify admission slots are released correctly.
- Loaded the service on loopback port 8185. The selected tested profile uses a 2,400 MiB allocation ceiling, 768 MiB startup reserve, maximum batch 128 and 32,768 padded batch tokens. It remains running; it is not boot-enabled yet.
- Compared actual HTTP low-memory, high-memory and high-memory-cached profiles. Larger memory/cached batches yielded about **1–2%** higher aggregate throughput, not a major speedup. Cached-profile throughput was 124.44, 30.75 and 6.90 query/passage scores per second at 128-, 512- and 2,048-token workloads respectively. These are not whole-query throughput figures.
- CPU FP32 was much slower; the tested CPU dynamic INT8 optimization failed the small Hindi relevance case and is not accepted for deployment. GPU FP16/CPU FP32 agreed on the three small language smoke rankings.
- Resident cached reranking passed simultaneous short and longer vision/embedding checks. The longer vision prompt used 13,054 tokens; sampled free VRAM minimum was 827 MiB. Maximum-resolution/multiple images, completely full-context vision, sustained load and held-out language quality remain unproved.
- Final full source regression: **3,189 passed, 1 skipped**, 64.97 seconds. Source compatibility artifact was regenerated after the new core module; the memory runtime/backend/gateway release itself was not changed. Installed model-service code is isolated from those canonical services.

Measurements, profile details, raw receipts, limitations and reproduction are in [the reranker benchmark report](reranker-gpu-benchmark.md). At the end of this historical benchmark slice, Hindsight still used RRF without the new learned model. The following integration slice supersedes that status. Outbound memory delivery remains paused.

## 10. Authenticated Hindsight/Qwen integration and flow validation

- Added the owned `rerank` configuration/credential route and local Cohere-compatible protocol. Hindsight's supported custom HTTP adapter reaches the gate with its unique route credential, then local CUDA Qwen. No cloud provider, unauthenticated admission bypass, direct installed-package patch or silent RRF provider fallback.
- Fixed operation/path credential binding, actual elapsed-time accounting and malformed rerank usage poisoning. Require complete finite indexed scores and bounded per-job actual token counts. Memory-originated GPU requests share physical admission with embeddings/vision routes; independent direct consumers remain outside that serialization.
- Added explicit exact-ID reflection support hydration because native ReflectFact omits source-document metadata. Recursive observation read-back is bounded and never substitutes for canonical/caller checks. Foreground recall does not acquire extra hydration requests.
- Full regression: **3,218 passed, 1 skipped**, 64.70 seconds. Matched runtime/worker/host wheel installed through core reviewed activation; Hermes plugin repinned through its official installer. Reranker unit now enabled at boot.
- Synthetic six-record competing corpus: nine English/Hindi/Hinglish questions all retrieved the intended canonical source first and delivered its supported semantic fact in ContextBroker. Hidden evidence was withheld; model outage gave explicit unavailable semantic retrieval with usable lexical memory; restart restored supported learned recall. Native reflection retrieval and exact-ID canonical support resolution also passed.
- Installed Hermes gateway/provider A → B → A profile isolation probe passed with zero network calls. This does not claim a real Telegram conversation was tested.
- Detailed configuration, receipts, code references, remaining entailment/scale acceptance work and activation identity are in [the integration report](hindsight-qwen-reranker-integration.md). Full P00–P14 acceptance remains incomplete. Native reflection may generate general discussion beyond recorded evidence and is not the framework's claim-validated publication path.
