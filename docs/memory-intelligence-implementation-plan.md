# Memory intelligence: detailed implementation plan

Date: 2026-10-02. Status: implementation started; full plan incomplete; not deployed.

Implementation has now started in core source; see the [implementation progress ledger](memory-intelligence-implementation-progress.md) for the exact first slice, verification, and remaining gaps. The full plan is not complete and installation remains deferred.

Latest validation exception: an isolated live-test runner incorrectly retained the production database path and applied migrations/injected test artifacts. Live validation is stopped; the owner has subsequently authorized continued repairs and confirmed all existing data is disposable. See the [incident record](memory-intelligence-live-test-incident.md) and [continuation repair register](memory-intelligence-repair-register.md). Historical source-only and schema-14 baseline statements below are not current production-state claims.

This plan implements the recommendations in the [memory intelligence and performance review](memory-intelligence-optimization-review.md), with traceability to all M-001–M-034 findings. It complements, rather than replaces, the [feature-flow review and earlier R-001–R-032 fixes](memory-framework-feature-review.md).

**This plan was originally produced as documentation only.** Subsequent source work is tracked separately in the progress ledger. Installation remains deferred. Future live inference, production migrations, service changes, real-data evaluation, and Telegram delivery require their own implementation/deployment authorization. Nothing below reports a completed optimization or measured performance gain.

## 1. Intended outcome and baseline

Hermes should receive a compact, relevant, correctly attributed memory packet on each appropriate turn. New conversations should become durable, connected memories under an explicit processing budget. Hinglish, Hindi, and English should work end to end, not just at the embedding stage. When memory needs clarification, Hermes should see the question, evidence, delivery state, and authenticated answer through its normal gateway context.

The design preserves separate layers for raw evidence, derived facts/observations, owner-approved assertions, goals, lessons, and forgetting. “Brain-like” means useful recall, cautious abstraction, temporal associations, verified learning, and controlled forgetting—not unlimited autonomous inference or authority.

The review observed the following baseline; these are historical observations from this review session, not continuously refreshed health claims:

- Source checkout HEAD: e26ae2309d73e969a7079372f529b0a4f1ca3148, with substantial uncommitted changes. The plan refers to inspected working-tree source.
- Installed release: 237133b492abed552e4567652b1f218f87bbf75b; Hindsight API slim 0.10.1. Installed source, not rolling documentation, determines supported interfaces.
- Live canonical schema: 0014_inquiry_delivery. Source already includes 0015_durable_fences and the earlier regression fixes, which remain undeployed in this analysis.
- Archive: 43 records, 40 visible; all 37 inspected Hermes records have no occurrence timestamp. This small archive cannot establish scaling behavior.
- Text upstream: http://192.168.68.69:8080/v1. Embeddings upstream: local Ollama, qwen3-embedding:0.6b, configured for 1,024 dimensions.
- Automatic consolidation is disabled; reranking is RRF; foreground memory deadline is six seconds. No live model-quality or contention benchmark was run for this plan.

Source-reference IDs such as C01 and H03 resolve to clickable files in section 12. External references were checked on 2026-10-02; rolling Hindsight docs may describe functionality beyond 0.10.1.

## 2. Non-negotiable implementation contracts

1. Canonical evidence stays immutable and authoritative. Re-extraction changes a projection generation, not a source revision. A model output is a candidate claim, not an owner-approved fact.
2. Scope comes from authenticated host/profile/account context. Model-supplied actor names, tags, URLs, chat IDs, or priority labels cannot grant authority.
3. Every admitted derived item resolves to a complete bounded support manifest of current, authorized source revisions in the current memory epoch. Missing support means withheld or explicitly partial, not trusted by default.
4. Forgetting and consent fences cover every generation, bank, cache, summary, candidate, graph/alias artifact, inquiry, and backup/restore path. Ranking decay is not deletion.
5. Preserve Hermes's stable system prompt and ordinary turn boundary. Use the existing memory provider/context mechanism; do not fabricate user turns, insert private mid-loop messages, or wake Hermes automatically by default. See C01 and A01.
6. All model work passes through the owned gate with a bounded deadline, resource, stage grant, token allowance, and operation identity. Do not hold an outer GPU claim while a native pipeline waits for the same nested gate.
7. Integrate through public Hindsight interfaces and versioned contracts. No site-packages edits, runtime monkeypatches, symlink workarounds, or bypass HTTP clients. If the public API cannot meet a requirement, ship a reviewed upstream/core change and pin the resulting version, or defer that feature.
8. Failure degrades to safe available evidence and explicit coverage diagnostics. “Available,” “authorized,” “complete,” and “answers the question” are different states.

These are release gates, not optional optimization preferences. C02–C07, C13, C19, H01–H04 underpin them.

## 3. Architecture decisions

### 3.1 Keep Hindsight as the derived-memory engine

Use its existing extraction, embeddings, facts, entities, temporal/semantic/causal links, recall, and observations. Keep canonical authorization and owner decisions in the custom framework. Measure native link quality before building a second graph; similarity edges must never become identity joins or proof of causation. H01, H03, H05; C04, C14.

### 3.2 Use native extraction context before modifying prompts

The installed extraction code renders both context and metadata into its prompt. Extend the current minimal payload with bounded, whitelisted speaker/source/time information and inspect prompt previews. Do not assume metadata is invisible, or replace Hindsight's prompt merely to add fields. The current API documentation also describes these fields as extraction inputs. [Hindsight retain documentation](https://hindsight.vectorize.io/developer/api/retain). H03; C03, C08.

### 3.3 Version projections explicitly

Introduce generational projection identity and an active-generation binding. Existing verified mappings short-circuit selection and submission; a processor fingerprint alone cannot refresh them. Changed vector dimensions or incompatible embedding contracts require a separate bank/index and controlled cutover. C05–C07.

### 3.4 Make strict summaries from bounded authorized input first

The first safe implementation synthesizes from a prepared canonical evidence packet, through the existing gate, with explicit output references. Native reflection can become an alternative only after pinned-version hard scope and complete support can be demonstrated. Prose instructions and temporal hints are not authorization filters. Current reflection documentation describes tag filtering, but that alone does not prove the inspected client uses it or the installed revision supports every documented option. [Hindsight reflect documentation](https://hindsight.vectorize.io/developer/api/reflect). C09; H01, H04.

### 3.5 Start with one embedding model and measured improvements

Retain Qwen 0.6B/1,024 dimensions as the baseline. Validate actual vector shape, normalization, truncation, and upstream templates. A/B-test a query-only instruction before changing models or document embeddings. Multilingual support is not a guarantee of Hinglish retrieval quality. Qwen recommends query instructions and supplies embedding-contract examples. [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). C06, C10; H02.

### 3.6 Clarification is a tracked memory workflow, not a second agent

Memory opens an inquiry with a dossier; delivery uses the existing gateway/outbox; Hermes receives that dossier at a normal authenticated turn. Informational answers and owner decisions have different schemas and privileges. Ordinary free text cannot silently authorize deletion, identity merging, budget changes, or goal activation. C01, C15–C17.

## 4. Execution order and complete finding traceability

Package IDs are proposed implementation work, not existing commands. Size denotes relative engineering scope: S = narrow existing boundary; M = several modules/contracts; L = schema or cross-service change. No calendar estimate is asserted before baseline/spike results.

| Package | Deliverable | Dependencies | Size / default |
| --- | --- | --- | --- |
| P00 | Reproducible baseline, test corpus, instrumentation contracts | None | M; required |
| P01 | Caller-aware capture and temporal/speaker context | P00 | M; required |
| P02 | Explicit worker task policies and pinned capabilities | P00 | M; required |
| P03 | Complete provenance resolution and scoped summaries | P01, P02 | L; required |
| P04 | Projection generations and per-item support manifests | P01, P03 | L; required |
| P05 | Hinglish extraction and multilingual query/lexical handling | P01, P03, P04 | L; required baseline improvements |
| P06 | Governed automatic formation and candidate assertions | P02, P04, P05 | L; required; inference initially off |
| P07 | Foreground purpose/deadline adapters and cancellation | P02, integration spike SP-1 | L; required; strategy gated by spike |
| P08 | Retrieval modes, packet packing, coverage, safe cache | P03, P05, P07 | L; required |
| P09 | Governed consolidation and native associations | P02, P03, P04, P06 | L; required validation; enable only after gates |
| P10 | Optional learned reranker | P00, P07, P08 | M; benchmark-gated |
| P11 | Goals, measurements, verified lessons, Hermes clarification | P01, P03, P08 | L; required integration; authority unchanged |
| P12 | Scaling, fairness, optional batching/concurrency | P04, P06, P07, P08 | L; measured bottlenecks only |
| P13 | Media pipeline validation and optional extraction | P03, P04, P06 | M/L; validation required, expansion conditional |
| P14 | Lifecycle proof, release packaging, rehearsal, activation | All selected packages | L; required before deployment |

P14's erasure/consent tests apply to every package from its first commit; they are not postponed until release. P00 instrumentation and quality evaluation remain active throughout. Sequence P01–P04 first: optimizing unsafe or incorrectly attributed output makes it harder to debug. P07 can be developed alongside P04–P06 once P02 is complete, without parallel production activation.

| Review findings | Package ownership | Required disposition |
| --- | --- | --- |
| M-001 | P02 | Real poller task names resolve to explicit policies |
| M-002, M-003 | P03 | Complete observation support; summaries have actual citations |
| M-004, M-005, M-024 | P01; P03 for summary authorization | Speaker/time preserved; scoped summaries usable only by authorized callers |
| M-006 | P04 | Generation change selects and rebuilds verified projections safely |
| M-007, M-014, M-015, M-016 | P05 | Language quality, vector contract, Unicode search, contextual query tests |
| M-008 | P06 | Supported candidate claims, not automatic truth promotion |
| M-009, M-010 | P06 | Bounded scheduled formation with selection fairness |
| M-011 | P12 | Per-item-safe batching benchmark or documented rejection |
| M-012, M-013 | P09 | Controlled observations and native link/association evaluation |
| M-017 | P10 | Reranker adopted only if quality/cost criteria pass |
| M-018, M-019, M-020, M-023, M-029 | P08; P05 for token/Unicode fixtures | Useful packed evidence, measured modes/cache, honest coverage |
| M-021, M-022 | P07 | Foreground embedding admission and end-to-end deadlines |
| M-025, M-026, M-027, M-028 | P11 | Personal-assistant context and authenticated clarification flow |
| M-030 | P14 and every package | No stale resurrection or cross-scope artifacts |
| M-031, M-032 | P12 | Resource and query-plan improvements supported by scale evidence |
| M-033 | P13 | End-to-end media evidence audit and bounded extension decision |
| M-034 | P00 and every package | Real quality/latency results distinct from synthetic correctness |

Not every opportunity requires new code: a benchmark-supported decision to retain RRF, single-item jobs, one worker, or native graph behavior is a valid disposition. Record the decision, measurements, and remaining limitations; do not label an untested idea fixed.

## 5. Shared data and API contracts to design before coding

### 5.1 SourceContext v2

Extend the capture/source envelope, not the canonical text itself. Proposed fields: envelope version; authenticated profile/source/account reference; speaker role and identity basis; conversation/thread/message identity; utterance time, timezone, precision and timestamp basis; event time if independently known; reply-to identity; quoted/forwarded/tool-origin flags; source-context schema version. Store only necessary fields and do not put credentials into model metadata.

Utterance time anchors “kal,” “next Friday,” and “pichhle hafte.” It is not the date a preference became valid. Capture ingest/spool time separately. Older records without a trustworthy utterance timestamp remain unknown; a timezone from the machine running a later replay is not evidence of the original speaker's timezone.

Version the decoder. Old v1 spool events must replay with their original normalization semantics and immutable identity. Do not reinterpret them as v2 and create conflicting content under the same record/revision. If historical context is enriched, store a derived overlay with evidence/basis; do not invent a canonical edit. C03; T02.

### 5.2 ProjectionGeneration and input manifest

Persist a versioned manifest containing backend/extractor versions, prompt/config digest, text model revision, chunk/context policy, embedding model revision/dimensions/instruction contract, bank binding, source-context version, and memory epoch. Separate extraction generation from query-only retrieval configuration so changing a query instruction does not automatically rebuild unchanged document vectors.

A proposed projection-registry extension keys entries by record revision, backend, bank, epoch, and generation. The existing uniqueness constraints exclude generation and need an additive migration or a new v2 table—not an in-place reset of verified rows. Track pending, submitted, verified, active/retired state and durable operation UUIDs. The active binding changes only after completeness, authorization, and deletion gates pass.

Every submission item has an input manifest: all contributing record IDs/revisions, selected spans, scope, epoch, generation, and document ID. Single-record jobs use one manifest entry. Contextual episodes and batches cannot share the existing single input_revision field as if all records had that revision. Preserve legacy job decoding and reconciliation. Document IDs must satisfy the pinned backend parser; use its tested opaque-ID convention rather than introducing unsupported delimiters. C05–C07.

### 5.3 RecallRequest and EvidenceCandidate

RecallRequest carries authenticated caller scope, bounded query text, language, explicit/derived time constraints with basis, permitted conversation/project referents, fast/balanced/deep mode, deadline, token ceiling, and source-context revision. Host-controlled fields cannot be overridden by the model. Resolved referents remain interpretations with confidence/basis, not identity joins.

EvidenceCandidate carries item kind, text/span, rank features, time bounds, complete support manifest, generation, epoch, authorization state, and coverage/withholding reason. Raw facts, observations, summaries, goals, and lessons stay distinguishable. Cache and packet code consume this representation instead of discarding support metadata between adapters. C02, C04, C10–C12.

### 5.4 InferenceRequest purpose and lifetime

The owned gate validates authenticated operation purpose, bank/profile, physical resource, stage allowance, input/output cap, priority, and expiry. Use a monotonic deadline within a process; cross-process propagation needs an absolute expiry plus bounded clock-skew handling. Queue admission and HTTP/model execution consume the same remaining budget.

“Query embedding” is not synonymous with “foreground”: reflection and maintenance also issue queries. An arbitrary request header is not trusted priority, and public Hindsight providers may not propagate it. SP-1 must establish how validated purpose reaches providers in API and worker paths. Missing purpose defaults to bounded background or refusal, never interactive privilege. C06, C10, C13; H01, H02, H04.

### 5.5 InquiryDossier and derived claim lifecycle

Dossiers carry inquiry ID/version, question type, alternatives or requested fact, relevant authorized citations, why clarification is needed, intended effect, expiry, correlation code, and delivery/answer/settlement status. Avoid exposing the entire archive when only one ambiguity is relevant.

Informational answers become attributed evidence and candidate claims. Owner decisions use the existing typed decision path and exact reviewed intent/digest. Candidate assertion status should distinguish proposed, supported, contradicted, rejected, and owner-activated where the existing ledger permits; extend it only where needed. A quoted assistant suggestion or Telegram bot message is not a user decision. C14–C17.

Migration numbers must be assigned at implementation time after checking the schema head. This plan does not reserve a guessed 0016/0017 or prescribe a second storage system. Extend existing ledgers where possible and document new indexes/tables only with their invariants and erasure obligations.

## 6. Detailed work packages

### P00 — Baseline, reproducibility, and measurement

Addresses M-034 and creates acceptance evidence for all later packages. References: C18, C20, T01, T09; review sections 2 and 7.

Implementation:

1. Record source diff digest, dependency lock digest, runtime/backend version, effective non-secret configuration, schema heads, corpus revision, hardware, and model revision. Distinguish source-tested from installed-tested results.
2. Freeze a language-balanced, manually labeled synthetic corpus; create independent train/tuning/held-out splits by scenario family, not by translated sentence. Include mixed scripts, spelling variation, pronouns, negation, dates, contradictions, assistant quotations, tools, forwarded messages, identity ambiguity, deletion, and unsupported queries.
3. Extend evaluation recording with per-stage timings: capture, queue, model, projection verification, lexical recall, embedding, backend recall, provenance, packing, and total provider latency. Record tokens and item counts; never log personal text or keys by default.
4. Preserve a pre-fix baseline showing reproducible defects. Then create a safety-repaired baseline from earlier source fixes in isolation. Compare optimizations against the latter so earlier bug fixes do not receive false performance credit.
5. Define opt-in backend-quality runs in an isolated profile/bank with synthetic input. A new live evaluation runner is a proposed deliverable, not a currently available command.

Tests/evidence: existing review, evaluation, and contract tests; reproducible JSON results and a short benchmark report in the project docs. Current synthetic evaluations must keep their “not measured against a backend/model” labeling until actual runs occur.

Exit: stable fixtures reproduce confirmed gaps; results identify code/model/config/corpus revisions and report each language separately. Rollback: disable telemetry/evaluation collection without changing source evidence or model behavior.

### P01 — Capture, attribution, time, and caller-aware summaries

Addresses M-004, M-005, M-024. References: C01, C03, C08–C09, H03; T02, T04.

Implementation:

1. Add SourceContext v2 at the Hermes capture boundary and source SDK. Preserve user/assistant/tool/quoted origins, session/message identity, and authenticated source-account bindings.
2. Derive utterance time only from an appropriate event timestamp. Keep occurred_at/event-validity separate. Preserve unknown historical timestamps rather than assigning the rebuild time.
3. Extend worker retain items with bounded context and whitelisted metadata. Start with single-turn formation; inspect pinned prompt previews to prove role/time/source reach extraction. Do not pass raw gateway objects or arbitrary untrusted metadata.
4. Pass authenticated caller identity and scope into summary lookup and the identity-aware ledger. Enforce the summary window and all dependency accounts, not merely the initiating account.
5. Add contextual episodes only after P04's complete input manifests exist. Reply chains must be bounded and every included source authorized; references outside scope remain unresolved.

Tests: v1 replay unchanged; v2 replay idempotent; missing timestamps stay unknown; midnight/timezone/DST cases; narrator/user/assistant distinction; quoted preferences not attributed to the owner; scoped summary returned to its caller and withheld across profiles/accounts. Use T02 and T04.

Exit: prompt previews show the intended fields; no source identity/content drift; account-aware summary reads pass authorization and availability tests. Rollback: disable new context projection while retaining canonical capture and compatible v2 decoding.

### P02 — Worker task policy and capability correctness

Addresses M-001; prerequisite for M-012. References: C06, C10, C13, H01, H04, H06; T03.

Implementation:

1. Introduce an explicit task policy registry separating database operation_type from executor payload type. Cover actual retained jobs and installed consolidation, refresh_mental_model, and graph_maintenance submission paths.
2. Map model-generating work to named gate operations and explicit stage grants. Give genuinely DB/CPU-only graph work its own bounded policy; never return a misleading “no resource” for unknown work.
3. Verify every supported kind through the real poller metadata injection path. A synthetic function-call test alone is insufficient because the database task kind can differ from the payload discriminator.
4. Extend the pinned capability table and compatibility artifact for the installed consolidation endpoint: POST /v1/default/banks/{bank_id}/consolidate, operation ID trigger_consolidation. Implement a named client method after request/response contract tests. Do not add unsupported APIs from rolling docs.
5. Preserve one admission per model hop. Prove no native operation acquires an outer claim and then waits for its own gate. Unknown kinds fail closed with an actionable status reason.

Tests: real-shaped poller task fixtures for every installed submission method; unknown task refused; granted stage admitted, expired/ungranted stage refused; DB-only work bounded; cancellation, retry, lost acknowledgement, and nested-gate deadlock regressions. T03 plus T05.

Exit: every supported task has an explicit policy and contract test; no implicit fallback or authority widening. Rollback: disable newly enabled stages, not task identity validation.

### P03 — Complete provenance and safe summary production

Addresses M-002, M-003. References: C02, C04, C09, H01; T04.

Implementation:

1. Carry RecallOutcome.source_facts alongside results into derivation. Resolve observation source_fact_ids through the supplied source-fact map to active document mappings and canonical support.
2. Share a bounded provenance resolver across facts, observations, summaries, associations, and later candidate assertions. Check bank/generation, current revision/epoch, all dependency scopes, and erasure fences. Bound depth, visited IDs, bytes, and support cardinality; reject cycles and report truncation reasons.
3. Do not perform unbounded extra backend fetches during foreground recall. If returned support is incomplete, withhold the item or use an explicitly allowed lower-level supported item, and report partial coverage.
4. Prepare a bounded authorized canonical packet for summary synthesis through the owned gate. Require structured output with claim-level input references. Verify that each reference belongs to actual supplied spans and that all cited dependencies remain live at publication.
5. Stop labeling all planned records as actual summary citations. Store selected coverage and actual support separately. ID validation does not establish semantic entailment; evaluate claim faithfulness and reject unsupported output.
6. Retain native reflect as an optional backend strategy only if pinned hard-scope and returned complete support tests pass. Temporal hints cannot substitute for canonical window enforcement. Hindsight describes temporal_window as a preference rather than a filter. [Hindsight recall documentation](https://hindsight.vectorize.io/developer/api/recall).

Tests: the valid observation/source-fact-map reproduction is admitted; absent, cyclic, cross-bank, cross-account, stale revision, or deleted support is withheld; mixed authorized/unauthorized support cannot be partially laundered into a trusted claim. Summary generation cannot cite records not provided or lose its scope after a concurrent revocation. T04 and T10.

Exit: complete-support observations work, and every published summary has an auditable actual support manifest. Rollback: turn off derived summaries/observations and keep authorized raw evidence; never disable verification.

### P04 — Generation-aware rebuilds and safe contextual manifests

Addresses M-006; enables episodes, reprocessing, and safe batching. References: C05–C07, C19; T05, T10.

Implementation:

1. Design and migrate the generational projection registry and active bank binding from section 5.2. Backfill existing mappings into an explicit legacy generation; unknown old processor inputs remain labeled unknown.
2. Update formation selection and DocumentMap.begin so a new generation schedules existing verified canonical records without mutating their revision or clearing the old verified state.
3. Persist a per-item revision/span manifest and operation identity before network submission. Recovery/reconciliation must resolve the submitted generation, not whichever generation is currently active.
4. Make incompatible embedding changes use a shadow bank/index. Test partial failure and deletion during backfill; verify live source coverage before atomically activating the binding.
5. Filter inactive generations during recall. For compatible same-bank rebuilds, prove candidate crowding is bounded and retire old documents after verification; otherwise use a shadow bank rather than silently accumulating duplicate vectors.
6. Keep all bank/generation erasure obligations until verified retirement/deletion. A forgotten record cannot be backfilled from an old job or restored manifest.

Tests: processor change actually reselects verified records; unchanged generation is idempotent; crash at each submission/cutover boundary; lost ack; heterogeneous revisions in a contextual item; concurrent edit/erasure; legacy jobs reconcile without guessed revisions. T05, T10.

Exit: deterministic rebuild and reversible active-binding cutover on a rehearsal store, with no duplicate canonical facts or stale resurrection. Rollback: switch binding only if the prior generation is still compatible, authorized, and complete; otherwise disable the affected derived path and forward-repair.

### P05 — Hinglish formation, query interpretation, and lexical matching

Addresses M-007, M-014, M-015, M-016; supplies token/span fixtures for M-020. References: C01–C03, C07–C08, C10–C12, H02–H03; T02, T04, T05.

Implementation:

1. Evaluate the existing extraction model before changing prompts: Romanized Hindi, Devanagari Hindi, English, and mixed-script Hinglish. Label speaker, modality, negation, entity, event/utterance time, and supporting span. Include “kal” as both tomorrow/yesterday depending on context and ambiguous cases requiring clarification.
2. Apply native context/metadata improvements first. Only introduce a versioned extraction-prompt/config change if residual failures demonstrate a need. Preserve raw language and named entities; an optional translation is an auxiliary retrieval aid with provenance, not a replacement for source text.
3. Inspect the embedding server/template and validate single/batched query/document output length, finite values, norm behavior, and truncation. Test a query-only instruction exactly once; prohibit double prefixes and document-query prefix contamination.
4. Replace the framework's combining-mark-splitting query preprocessing with a Unicode-aware strategy. Compare exact quoted identifiers, phrase matching, bounded OR/soft terms, and current AND behavior. Rebuild/version the auxiliary index only if tokenizer changes require it. SQLite documents configurable FTS5 tokenization; validate the actual tokenizer and emitted query, not just Python tokenization. [SQLite FTS5 documentation](https://www.sqlite.org/fts5.html).
5. Preserve raw spellings, identifiers, and negation. Optional spelling/transliteration variants go into auxiliary features, not canonical IDs or identity assertions. Evaluate variants such as “mujhe/muje,” mixed-script names, and punctuation without assuming every Devanagari query currently fails.
6. Build a bounded query descriptor from the authenticated current turn and relevant permitted recent context. Resolve “usko,” “wo project,” and relative dates only where evidence is adequate; ambiguity becomes an unresolved query/clarification, not an invented identity.

Tests: cross-language retrieval and extraction on held-out scenarios; negative preferences; assistant quotations; relative time with unknown timezone; current exact-ID search not regressed; combining marks retained; query prefix applied once; no raw-source/identity rewrite.

Exit: language-stratified gates in section 8 pass, with prompt/template/vector versions recorded. Rollback: retain raw evidence and baseline query mode; revert auxiliary features without undoing canonical capture. Document-vector changes use P04, never an in-place incompatible vector mix.

### P06 — Governed formation, salience, and candidate assertions

Addresses M-008, M-009, M-010. References: C07–C08, C13–C14, C20; T05, T06.

Implementation:

1. Add an owned scheduled formation entry point with a dry-run plan and explicit invocation policy. The existing three services do not prove archive formation is scheduled; inspect actual scheduling before adding duplicate producers.
2. Use explicit owner-granted stage allowances with expiry, record caps, token budgets, and global daily limits. Extend the current formation-only stage contract for new stages as needed. Agent invocations cannot mint or renew grants; no grant means capture-only behavior.
3. Select eligible records using bounded recency/salience/redundancy features and age-based fairness. Record selection reasons; prefer real user preferences, important commitments, changed facts, and unresolved dependencies without permanently starving ordinary old evidence.
4. Replace the uniform 2,000-token estimate with conservative measured per-type estimates, updated from actual usage. Keep pre-admission ceilings and reconcile usage without allowing overrun to silently widen budgets.
5. Reuse assertion storage for supported candidate claims where its lifecycle fits. Keep occurrence/validity intervals, source spans, confidence basis, contradictory alternatives, and status. Extracted assistant proposals remain assistant proposals.
6. Route uncertainty into P11 inquiries where useful. Owner activation remains a distinct host-authenticated operation; frequent repetition or a confident model does not make an assertion true.

Tests: no grant/expired grant/oversized selection refused; repeated scheduler ticks idempotent; outage catch-up bounded; budgets across profiles remain global where specified; stale leases/epoch invalidation; old-record fairness; conflicting candidates remain visible rather than overwritten. T05–T06.

Exit: fresh eligible evidence forms within the agreed scheduled budget, without unbounded model work or implicit owner decisions. Rollback: disable scheduler/new producers; capture and existing verified recall continue.

### P07 — Foreground admission, public adapters, and cancellation

Addresses M-021, M-022. References: C01, C06, C10, C13, H01–H02, H04; T03, T07.

SP-1 is a prerequisite integration spike:

1. Build an isolated composition proof using the public MemoryEngine constructor (embeddings/cross_encoder/query_analyzer hooks) and public create_app factory. Preserve API lifespan, authentication, tenant validation, DB initialization, and worker-specific migration behavior.
2. Demonstrate authenticated recall purpose and remaining deadline reaching an owned embeddings adapter in the API path, and maintenance purpose reaching it in the worker path. A request-local context must not leak across concurrent tasks. encode_query alone does not identify an interactive request.
3. Verify whether public extension/entry points suffice. If not, specify the smallest reviewed upstream/core extension and pin it, or retain bounded background admission until it exists. Do not instrument private methods or mutate installed objects to simulate support.

Implementation after SP-1:

4. Introduce explicit foreground-query and background-document/maintenance admission policies using verified purpose and bank ownership. New credentials/settings follow existing scoped secret handling; clients cannot self-declare an interactive priority.
5. Propagate one deadline across provider, broker, backend, gate queue, and HTTP execution. Leave time for packing and returning a safe packet; bounded queue wait must be shorter than remaining foreground lifetime, not the unrelated 120-second background default.
6. Cancel queued work before admission on expiry/disconnect. For running upstream inference, attempt supported cancellation but keep the physical resource accounted for until completion/cancellation is actually known. Cancelling a Python Future is not proof that HTTP/model execution ended.
7. Expose withheld/timeout/cancelled reasons and queue/model timings without leaking content. Keep non-preemptive resource semantics; priority does not kill already-started work.

Tests: concurrent foreground/background context separation; profile bank spoofing refused; provider timeout cannot launch a late queued embedding; running cancelled call does not falsely free GPU; resource reclaimed after verified completion; API/worker restart and lifespan correctness. T03, T07.

Exit: SP-1 proves a maintainable public integration, and end-to-end lifetime tests pass. Rollback: disable new priority policy/adapters through supported configuration; retain deadline bounds and safe fallback. Any changed backend pin must pass compatibility and P14 rehearsal.

### P08 — Useful retrieval packets and explicit coverage

Addresses M-018, M-019, M-020, M-023, M-029. References: C01–C02, C04, C09, C11–C12, H01; T04, T07.

Implementation:

1. Expose fast/balanced/deep policies in the broker with independent candidate, backend token, packet, and time budgets. Start by deriving actual limits from pinned source (currently 100/300/1,000 recall token limits); do not rely on stale “600” comments or assume a native budget is the final Hermes packet budget.
2. Fast: deterministic/raw and relevant already-verified items, no new reflection. Balanced: bounded native recall plus complete-support observations when available. Deep: explicit opt-in broader retrieval/associations; not the default for every short message.
3. Rank heterogeneous authorized candidates before packing. Reserve useful space for direct semantic matches rather than always filling the packet with raw lexical results and summaries first. Deduplicate repeated support while retaining attribution and contradictory alternatives.
4. Select question-relevant spans with a real host tokenizer if exposed through a stable interface, otherwise a calibrated conservative estimator. Count the exact rendered packet—including labels/citations—and preserve negation, qualifications, units, and time. Avoid blindly truncating every record at 600 characters.
5. Replace the current SUPPORTED heuristic with separate availability, authorized support, source completeness, and answerability states. Answerability remains unknown unless evaluated; backend record counts and nonempty results do not prove a question can be answered.
6. Cache only after measuring hit rate. Keys include normalized request, authenticated profile/account set, identity/consent/context revision, epoch, active generation, relevant configuration, and mode. Revalidate dependencies on read; erasure must invalidate even when TTL has not expired. Never share semantic cache entries across authority boundaries.

Tests: tight packet budget retains the best question-specific fact; tail negation preserved; exact rendered-token ceiling; conflicting facts labeled; partial backend support doesn't become complete; cache stale after identity/consent/epoch/generation change; empty and unsupported queries honestly reported.

Exit: useful packed recall and latency gates pass against the repaired baseline, without authorization regressions. Rollback: restore the baseline packing/mode strategy; cache can be disabled independently. Provenance verification remains mandatory.

### P09 — Controlled consolidation and associations

Addresses M-012, M-013. References: C04, C06–C08, C13–C14, H01, H04–H06; T03–T06.

Implementation:

1. Keep native automatic consolidation off initially. Schedule explicit incremental consolidation only for eligible active-generation scopes with a separately granted stage allowance and measured limits.
2. Prove pinned observation scope/tag behavior using contract fixtures and an isolated bank. If required hard scoping is not supported, do not publish cross-scope canonical abstractions; use a narrower bank/input strategy or defer.
3. Reconcile durable operations and support manifests. Source edits, consent changes, deletion, or new generations invalidate affected observations and dependent summaries; regeneration is budgeted work, not a hidden recall-side write.
4. Evaluate existing native entity/temporal/semantic/causal links on labeled scenarios: repeat preferences, project changes, deadlines, people with the same name, and corrected assumptions. Treat inferred relationships as hypotheses unless supported.
5. Expose a small number of relevant authorized associations to the broker where they improve recall. No second graph or broad alias expansion unless native evaluation demonstrates a specific gap and P14 lifecycle coverage is designed.

Tests: no grant means no consolidation; unknown task policy refused; duplicate consolidation reconciled; source withdrawal removes dependent observations; cross-account associations withheld; identical names do not merge owners; graph work respects DB/CPU bounds.

Exit: observations improve held-out recall without reduced claim precision, authority widening, or foreground SLO violation. Rollback: disable the consolidation scheduler/association exposure and continue factual recall; retain deletion/reconciliation obligations.

### P10 — Optional learned reranking

Addresses M-017. References: C02, C06, C10, H01–H02; T03, T07.

Implementation:

1. Establish RRF-only nDCG/recall and contention baseline first. Use a small bounded candidate set, initially at most 16, and a remaining-deadline check before reranking.
2. Implement a gated adapter only if a compatible scoring interface is proven. Qwen reranking relies on its relevance scoring protocol; a free-form chat “yes” is not that protocol, and its relevance score is not calibrated truth confidence. [Qwen3-Reranker-0.6B model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).
3. Evaluate on the local GPU while embeddings/other local work run. Do not introduce an always-loaded model that silently exceeds memory or blocks foreground embeddings.
4. Keep timeout/error fallback to RRF and record the fraction of requests actually reranked.

Tests: scoring contract; ordered candidate/result identity; gate and cancellation; deadline fallback; memory pressure; correct provenance unchanged by reranking. Exit: adopt only if section 8's optional-benefit gate passes. Otherwise record “retain RRF” with results. Rollback: disable the adapter; no reindex required for ranking-only changes.

### P11 — Personal-assistant memory and gateway clarifications

Addresses M-025, M-026, M-027, M-028. References: C01–C02, C14–C18; T02, T04, T08–T09.

Implementation:

1. Supply relevant open goals/commitments to ordinary context through the existing broker capability, with owner activation state, dates, and citations. Recall does not schedule/send reminders or create a new commitment.
2. Connect typed measurement queries where applicable. Calculate means/ranges/trends deterministically with units, time windows, and missing-data notes; an LLM cannot substitute invented arithmetic for the measurement store.
3. Surface only applicable verified lessons. Keep outcome evidence, evaluation provenance, approval state, scope, and expiry. Repeated use/popularity is not evidence of correctness.
4. Add typed informational inquiry schemas/producers for ambiguous speaker, referent, preference, date, or contradiction. Keep these separate from existing owner ACTS and destructive confirmation schemas.
5. Persist an InquiryDossier before delivery. Include it in Hermes's normal authenticated context whether pending, delivered, answered, or settled. No automatic Hermes wake by default; a future wake policy would be a separate feature/approval.
6. Route delivery through the existing outbox and gateway target. Record inquiry correlation, gateway message ID, attempt/receipt, profile/chat/thread, expiry, and content version. Preserve existing bot/forward/native-private-chat ownership checks at ingress.
7. At the normal user-turn boundary, settle a correlated informational answer once; persist an attributed evidence record and candidate interpretation. Re-read the settled dossier for Hermes. Model text, quoted code, forwarded replies, wrong account/thread, and expired inquiry cannot authorize owner acts.
8. For owner decisions, keep exact typed intent/review digest and idempotent decision receipts. If an answer changes the proposed action, create a newly reviewed decision rather than silently broadening the original approval.

Tests: synthetic gateway/native-turn fixtures first; correct reply connects to its dossier and resulting evidence; uncorrelated answer requests clarification; bot/forward/cross-profile/replayed answer refused; delivery retry does not duplicate settlement; forget request invalidates pending inquiry/support; ambiguous “haan” cannot confirm deletion or activate a goal. T08, T10.

Exit: Hermes can explain what memory asked, why, where it was delivered, what the owner answered, and what changed, all within scope. Real Telegram sends are a separately approved smoke test, not a unit-test side effect. Rollback: disable informational inquiry producers; preserve pending dossier/receipt visibility and existing owner safety.

### P12 — Scale, fairness, batching, and physical-resource utilization

Addresses M-011, M-031, M-032. References: C05–C08, C10, C13–C14; T05–T07.

Implementation:

1. Benchmark isolated synthetic archives at 1k, 10k, and 100k records with realistic scope/identity distributions. Capture query plans, DB timings, repeated provenance lookups, formation scans, packet size, and memory use.
2. Fix measured N+1 reads and scans with bounded batch queries/indexes. Preserve per-record authorization, revision checks, ordering, and count semantics; do not cache global authorization answers indefinitely.
3. Introduce microbatches only if per-item manifests/recovery from P04 are complete and job-overhead measurements justify them. Bound item count, text/tokens, account/scope, epoch, generation, and deadline. Maintain per-item success and operation reconciliation on partial failure.
4. Keep native provider batch processing disabled unless its every inference hop can be shown to pass through the gate. An upstream batch facility is not automatically compatible with this framework's admission model.
5. Measure remote text/local embeddings as separate physical resources. Improve fairness/worker scheduling only if a single task slot causes measurable head-of-line blocking. Model concurrency stays bounded by measured device capacity; increasing worker count alone is not a fix.
6. Measure actual GPU/RAM use and fragmentation before changing worker/concurrency settings. Add burst/background/foreground load tests and starvation bounds.

Tests: batch partial ack/retry/cancel/erasure; global allowances across workers; fairness under sustained background load; no duplicate submission; query-plan correctness; cross-profile scopes in batched provenance. Exit: adopt each change only with before/after scale and contention results. Rollback: single-item/single-worker defaults; additive indexes may remain if compatible and safe.

### P13 — Media evidence audit and optional extraction

Addresses M-033. References: C08, C10, C19–C20; T05, T10.

Implementation:

1. Trace actual blob capture → MIME/parser → transcript/OCR/vision → source spans → formation → recall → forgetting. Having blob storage and a vision route does not establish a complete production pipeline.
2. Define bounded text/transcript evidence with media time/page/region references and model/parser revision. Preserve the original only under the source's retention/consent policy.
3. Make OCR/transcription/vision an explicit stage grant, with type/size/output limits and no hidden backend file-upload bypass. Current OCR is disabled; enabling it is an opt-in configuration/deployment step after tests.
4. Evaluate Hinglish audio/code-switching separately if this extension is selected. Surface uncertainty; speech recognition output is not a verified user statement until attributed to the source.

Tests: malformed/unsupported media, oversized files, low-confidence transcript, parser crash/retry, scoped blob access, media deletion with derived artifact cleanup, restore fences. Exit: publish the pipeline gap/extension decision; deploy only selected verified media types. Rollback: disable media producers while preserving canonical metadata and lifecycle duties.

### P14 — Lifecycle proof and release/deployment readiness

Addresses M-030 and the undeployed earlier source fixes. References: C05, C13, C15–C20; T10–T11; [installation documentation](installation.md).

Implementation:

1. Add dependency/erasure handling for every new artifact before its producer is enabled. Cover pending jobs, all projection generations/banks, observation support, aliases/associations, packet/cache entries, goals/lessons, inquiries/receipts, media, and evaluation data.
2. Prove epoch/consent/revision fences at enqueue, submit, verify, publish, recall, delivery, answer settlement, and restore. A concurrent edit/forget during inference must prevent stale publication.
3. Build a release from reviewed source, lockfiles, tests, backend pin, compatibility artifact, and provider digest. Preserve unrelated user changes; do not stage a release by editing the currently installed runtime.
4. Rehearse the observed schema 0014 → existing source 0015 → selected new migrations on a disposable authorized copy or synthetic fixture. Validate legacy jobs, active generation bindings, inquiry receipts, and backup/recovery behavior.
5. Review the actual activation workflow. The current upgrade implementation produces a plan and performs no switch. If atomic owned activation/drain/health rollback is missing, implement and test it in installer core before deployment. Do not invent an upgrade --apply command or substitute manual symlink edits.

Exit: section 9's deployment gates pass, with a release manifest and an honest rollback plan. Installation remains a later, separately authorized action. Emergency mitigation favors disabling affected features and forward repair when an older binary cannot safely read the new schema.

## 7. End-to-end scenarios to prove

### Hinglish recollection and corrected date

1. At an authenticated turn the owner says: “Rohit ko kal budget bhejna tha, ab Friday ko bhejenge.” Capture speaker, utterance timestamp/timezone, conversation identity, and raw text. Do not assume which Rohit or what “kal” means without the available context.
2. Formation extracts the statement with its modality and changed plan; the support manifest includes every turn needed to resolve it. A commitment is only a candidate until the applicable owner activation path succeeds.
3. Later “usko kab bhejna hai?” creates a query descriptor from permitted recent context. If referent/date is unresolved, return honest partial context and create a typed informational inquiry rather than guessing.
4. Memory persists the dossier and sends the inquiry through the existing gateway, under approved delivery policy. Hermes sees the dossier on the user's next normal turn.
5. An authenticated correlated Telegram reply, “finance wala Rohit, is Friday 9 October,” is stored as attributed answer evidence. It resolves the candidate; a goal activation still requires the appropriate typed owner decision.
6. Later recall includes the corrected plan, not a superseded date as current truth, with citations and activation state. Forgetting the conversation prevents recall, pending inquiry settlement, old-generation backfill, and backup resurrection of the forgotten dependencies.

Acceptance fixtures must pin the scenario date/timezone; the example date is not a current reminder or an instruction to deliver anything.

### Other mandatory scenarios

- “Mujhe chai nahi chahiye” recalled from “tea preference?” retains negation and speaker; quoted assistant text does not create the owner's preference.
- A valid observation with source_fact_ids is accepted through complete support; one unauthorized/deleted support fact withholds the combined observation.
- Two accounts discuss different people named Rohit; associations do not merge identities or leak the other account's facts.
- Background formation/consolidation runs while a foreground query arrives; bounded admission meets the measured deadline or returns safe partial evidence, with no model job starting after the turn has expired.
- A changed embedding contract triggers shadow projection and verified cutover, not mixed-dimension vectors or silently skipped verified records.
- A pending clarification, a late model result, a cached packet, and a stale backup all encounter a concurrent forgetting fence and cannot restore visible content.

## 8. Evaluation matrix and proposed acceptance gates

These are initial acceptance proposals, not existing benchmark results or guarantees for unknown remote hardware. P00 must record feasibility and any negotiated change before implementation is called complete.

### Dataset and scoring

- Minimum initial language set: 60 independently labeled scenario families rendered in four variants—English, Romanized Hindi, Devanagari Hindi, mixed-script Hinglish—giving 240 cases. Split by family to avoid translation leakage. Expand rare time/negation/identity classes rather than relying on one aggregate score.
- Add separate adversarial/safety suites for profile/account isolation, prompt injection in stored text, bogus priority/owner identity, expiry, erasure races, schema upgrade, and gateway replay. Safety fixtures must not depend on a model's voluntary compliance.
- Measure atomic extraction precision/recall, speaker/time/negation accuracy, source-supported claim precision, recall@5/10, nDCG@10, useful packed recall, answerability labeling, contradiction handling, inquiry resolution accuracy, and actual token cost. Report denominators, per-language values, and confidence intervals where appropriate.
- Measure cold/warm p50/p95/p99 and failure/degraded rates under idle and contended load. Include gate queue wait, model time, support verification, and packing—not just backend HTTP time. State hardware, corpus scale, repetitions, and warm-up policy.

### Gates

| Gate | Initial target | Blocking rule |
| --- | --- | --- |
| Authority/lifecycle | Zero observed unauthorized admissions, false owner confirmations, stale publication/resurrection, or priority escalation in prescribed tests | Any violation blocks release regardless of recall score |
| Provenance | Every admitted derived item has complete live authorized support; no unsupported summary references | Missing support cannot be labeled complete |
| Extraction/support quality | At least 95% supported-claim precision in each language; no regression in critical speaker/negation/time fixtures | Aggregate English success cannot hide Hinglish failure |
| Retrieval | At least 90% recall@5 on labeled answerable cases per language, with no material repaired-baseline regression | Record actual denominators; tune only on tuning split |
| Packet utility | Useful packed recall improves or meets the agreed baseline while staying inside actual rendered-token ceiling | Backend-only gains do not count if Hermes never receives them |
| Foreground latency | Initial aspiration: fast p95 ≤1s, balanced p95 ≤2s on the agreed deployment/load; six-second current timeout remains an outer bound, not an SLO | If infeasible, document hardware/result and agree a revised gate before adoption |
| Formation | Within two scheduler intervals for eligible small inputs when grants/resources suffice; interval/backlog/cost selected at P06 | Budget exhaustion/outage must be explicit, not hidden completeness |
| Optional rerank/batch/concurrency | Adopt only for ≥5% relative nDCG@10 benefit or ≥20% relevant throughput/latency benefit, without precision/safety/SLO regression | Thresholds are proposals; no complexity without measured value |
| Clarification | Valid correlated native replies settle once and remain visible to Hermes; invalid fixtures never authorize action | Any ambiguous destructive confirmation blocks release |

Precision is claims supported divided by admitted extracted claims; recall uses labeled relevant items with a declared relevance unit. A refusal is not a false supported claim but must count against availability/recall where an authorized answer was available. Token/reranker confidence must not be reported as factual certainty.

Evaluate one change at a time: metadata/time → query instructions → lexical/query interpretation → packing → consolidation → optional reranker → batching/concurrency. Record model/config/vector generation at each step. Avoid crediting a model change for simultaneous corpus/prompt/index changes.

### Verification commands and boundaries

For future framework implementation, use the existing uv environment without implicitly updating its lockfile:

~~~sh
uv run --no-sync python -m pytest -q tests/review tests/contracts tests/hermes
uv run --no-sync python -m pytest -q tests/unit tests/integration tests/faults tests/installer
~~~

These are future verification instructions; they were not run for this documentation task. Existing synthetic/load eval entry points are C18; a backend/language-quality runner needs explicit implementation and isolation first. If Hermes host source changes become necessary, follow its repository instructions and scripts/run_tests.sh, not a guessed generic pytest invocation. Live model, real-data, service, and Telegram tests must be opt-in and separately scoped.

## 9. Release, deployment, and rollback gates

### Gate A — Source-ready

All required defect packages have passing regression tests; selected optimization packages have measured adoption/rejection decisions. Full M-register disposition and earlier R-register source status are recorded. Feature flags/defaults preserve capture-only and safe fallback. New owner-stage policies and migrations are documented.

### Gate B — Rehearsal-ready

Cut a reproducible release without including unrelated dirty changes. Verify source/plugin/backend/compatibility digests. Rehearse schema migration, interrupted migration recovery, legacy operation reconciliation, gen-1/gen-2 cutover, erasure, and restore on an isolated authorized store. Prove the installer activation path actually exists and is owned/atomic; current upgrade planning alone does not satisfy this gate. C19–C20, T10–T11.

### Gate C — Isolated backend-ready

Use synthetic data and an isolated profile/bank to test real 0.10.1 or newly pinned public contracts, vector shape, multilingual quality, task execution, cancellation, consolidation, and contention. No production recall/write or gateway delivery is implied. Include PG0/startup/auth checks when replacing native API composition.

### Gate D — Owner-approved production activation

Before activation, review exact release digest, changed services/settings, schema transitions, bank bindings, stage grants, backup/recovery obligations, and delivery policy. Drain/reconcile in-flight work through supported core operations, not blind termination or resetting leases. Apply the approved installer workflow and run read-only doctor/status and scoped smoke checks. Enable new producers in stages; keep consolidation/reranking/media off until their gates pass.

### Gate E — Post-activation observation

Watch stage error rates, formation lag, gate wait/model time, language quality samples under consent, provenance refusals, incomplete coverage, packet usefulness, inquiry correlation, and memory/GPU pressure. Agree an observation window and automatic stop thresholds before activation. A running service is not enough to declare the feature healthy.

### Rollback rules

- Disable the affected producer/ranking/adapter feature first where safe; canonical capture and authorized raw recall should remain available.
- Switch an active projection binding only to a still-authorized, compatible generation with all current erasure obligations satisfied. A previous bank is not safe merely because it existed before the release.
- Older binaries cannot run against a newer schema unless the compatibility contract explicitly permits it. Prefer forward repair when rollback compatibility is absent.
- Never blindly restore an old canonical/gate snapshot: that can erase newer forgetting, decision, inquiry, and delivery fences. Restoration must preserve/reconcile the authoritative durable fences and receipts under the tested recovery contract.
- No manual patch to installed packages, direct production SQL “fix,” or symlink switch substitutes for a verified installer/recovery operation.

Installation is still deferred. This plan is the checklist for a later implementation and separately approved deployment, not permission to activate it now.

## 10. Suggested commit boundaries and completion record

Each package should normally produce: a failing reproduction/contract fixture; the narrow core implementation and compatible migration if needed; tests and fault cases; documentation/configuration changes; and a benchmark/adoption decision where relevant. Commit tests with the implementation, not an unreviewable final mega-diff.

Recommended first implementation batch: P00 baseline → P01 context/caller correctness → P02 task policies → P03 support/summary correctness → P04 generations. This establishes the trustworthy foundation before automatic formation and retrieval optimization. Do not mix installation changes into these commits unless implementing an explicit P14 installer prerequisite.

Maintain a package/finding ledger with:

- Finding and package IDs; confirmed defect versus hypothesis/optimization.
- Reproduction fixture and source function; before/after code/model/config/corpus revision.
- Decision and rationale; implementation commit and migration; test/benchmark artifact.
- Source status, release status, and installed status separately.
- Acceptance gate passed/failed/not run; residual risks; safe fallback; rollback compatibility.

“Complete” requires evidence, not just a changed file. If a proposed optimization is rejected, its measured decision and limitation belong in the same ledger.

## 11. Decisions to finalize during implementation

No answer is needed to finish this documentation task. Recommended defaults for later work are:

1. Use synthetic evaluation only initially; personal-message sampling requires explicit consent and redaction/storage policy.
2. Retain current Qwen dimensions/model, RRF, and one worker until measurements justify changes.
3. Keep automatic consolidation off while explicit stage-granted consolidation is validated.
4. Use passive Hermes inquiry-dossier context; do not add automatic wakeups or broaden delivery destinations.
5. Use a shadow bank for incompatible vector contracts; retire old generations only after recall/deletion proof.
6. Let the owner choose processing grant duration, daily cost/record ceilings, quiet hours, and approved inquiry delivery target. The current stage-grant implementation imposes its own maximums; new stages must not evade them.
7. Agree hardware/load and realistic SLOs after SP-1/P00; reject optional features without meaningful benefits.
8. Keep media expansion and a duplicate custom graph out of the initial deployment unless the audit demonstrates a concrete required gap.

## 12. Reference catalog

Local references are the inspected working-tree files, or the immutable installed backend release below. Line numbers aid navigation and may move after implementation; function names specify the intended boundary. Tests listed here exist; proposed future helpers/CLI are described as proposed rather than linked as if present.

### Custom framework and Hermes integration

| ID | Evidence / implementation boundary |
| --- | --- |
| C01 | [Hermes provider: turn start](/home/jugaadu/Projects/hermes-memory/integrations/hermes-memory/provider.py:449), [broker construction/prefetch](/home/jugaadu/Projects/hermes-memory/integrations/hermes-memory/provider.py:819), [native turn settlement](/home/jugaadu/Projects/hermes-memory/integrations/hermes-memory/provider.py:927), [capture envelope](/home/jugaadu/Projects/hermes-memory/integrations/hermes-memory/capture.py) |
| C02 | [Broker assembly](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/context/broker.py:99), [derivation, coverage, support filtering](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/context/broker.py:254) |
| C03 | [Source SDK frame/stamp](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/sources/sdk.py:166), [canonical lexical preprocessing](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/storage/evidence.py:241), [search](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/storage/evidence.py:337) |
| C04 | [Recall client](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/hindsight_client.py:227), [RecallOutcome/source facts](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/hindsight_client.py:389), [provenance resolver](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/provenance.py:155) |
| C05 | [DocumentMap.begin](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/document_map.py:43), [jobs and submission fencing](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/jobs.py:91), [projection schema](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/storage/migrations.py:218) |
| C06 | [Pinned capability map](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/capabilities.py:19), [worker public constructor arguments](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/worker_launcher.py:497), [task resource selection](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/worker_launcher.py:793) |
| C07 | [Processor fingerprint/formation plan](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/formation.py:69), [formation apply](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/formation.py:227), [unprojected selection](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/formation.py:475) |
| C08 | [Projection worker](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/worker.py:113), [retain payload](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/worker.py:252) |
| C09 | [Summary production](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/summarization.py:264), [native reflection strategy](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/summarization.py:372), [summary read/latest](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/knowledge/summaries.py:202) |
| C10 | [Gate routes/priority/resources](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/routes.py:99), [queue wait](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/gate_server.py:89), [worker slot contract](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/backend/worker_launcher.py:236) |
| C11 | [Packet rendering](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/context/packet.py), [cache](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/context/cache.py) |
| C12 | [Identity/account evidence](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/storage/identity.py:582), [profile resolution](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/install/profiles.py:247) |
| C13 | [Owner stage allowances](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/allowance.py), [budget admission](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/processing/budgets.py:66) |
| C14 | [Assertion lookup/ledger](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/knowledge/assertions.py:295), [open goals](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/prospective/goals.py:293), [measurement series](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/storage/measurements.py:103) |
| C15 | [Inquiry native answer](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/proactive/inquiries.py:565), [dossier and answer settlement](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/proactive/inquiries.py:593) |
| C16 | [Gateway delivery](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/proactive/delivery.py:127), [durable outbox](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/proactive/outbox.py), [typed owner acts](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/operations/decisions.py) |
| C17 | [Applicable lessons](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/learning/lessons.py:232), [outcome recording](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/learning/outcomes.py:75), [evaluation ledger](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/learning/evaluation.py:155) |
| C18 | [Synthetic evaluation](/home/jugaadu/Projects/hermes-memory/evals/run_synthetic.py), [load evaluation](/home/jugaadu/Projects/hermes-memory/evals/run_load.py), [quality documentation](quality.md) |
| C19 | [Durable fence migration](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/storage/migrations.py:960), [erasure](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/lifecycle/erasure.py), [restore](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/lifecycle/recovery.py:186), [blob storage](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/storage/blobs.py) |
| C20 | [Upgrade planning—no activation](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/install/upgrade.py:44), [release plan/build/verification](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/install/release.py:90), [doctor](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/operations/doctor.py:84), [operations documentation](operations.md) |
| A01 | [Hermes architecture and prompt-cache invariants](/home/jugaadu/codes/hermes-agent/AGENTS.md:19) |

### Installed Hindsight 0.10.1 source

The immutable runtime target avoids treating a later runtime/current switch as the evidence inspected here. These are reference paths, not edit targets.

| ID | Pinned implementation evidence |
| --- | --- |
| H01 | [MemoryEngine public constructor](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/engine/memory_engine.py:2158), [recall](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/engine/memory_engine.py:7893), [recall result construction](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/engine/memory_engine.py:9770) |
| H02 | [Embeddings interface](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/engine/embeddings.py:109), [query/document methods](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/engine/embeddings.py:253), [OpenAI-compatible implementation](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/engine/embeddings.py:827) |
| H03 | [Extraction context/metadata prompt](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/engine/retain/fact_extraction.py:1810), [language instructions](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/engine/retain/fact_extraction.py:1044) |
| H04 | [Public API factory/lifespan](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/api/http.py:4798), [consolidation endpoint](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/api/http.py:9158) |
| H05 | [Native link creation](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/engine/retain/link_creation.py) |
| H06 | [Consolidation task submission](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/engine/memory_engine.py:22182), [graph task submission](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/engine/memory_engine.py:22243), [mental-model task submission](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/engine/memory_engine.py:22595), [poller metadata injection](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/worker/poller.py:711) |
| H07 | [Embedding prefix defaults](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/config.py:1215), [recall budget limits](/home/jugaadu/data/hermes-memory/runtime/237133b492abed552e4567652b1f218f87bbf75b/hindsight/lib/python3.12/site-packages/hindsight_api/config.py:1896) |

### Existing test boundaries

| ID | Tests to extend/use during implementation |
| --- | --- |
| T01 | [Feature review reproductions](/home/jugaadu/Projects/hermes-memory/tests/review/test_feature_review.py) |
| T02 | [Hermes provider](/home/jugaadu/Projects/hermes-memory/tests/hermes/test_hermes_plugin.py), [SDK](/home/jugaadu/Projects/hermes-memory/tests/hermes/test_sdk.py) |
| T03 | [Worker launcher contracts](/home/jugaadu/Projects/hermes-memory/tests/contracts/test_worker_launcher.py), [gate contracts](/home/jugaadu/Projects/hermes-memory/tests/contracts/test_gate_server.py) |
| T04 | [Broker](/home/jugaadu/Projects/hermes-memory/tests/unit/test_broker.py), [provenance](/home/jugaadu/Projects/hermes-memory/tests/unit/test_provenance.py), [summaries](/home/jugaadu/Projects/hermes-memory/tests/unit/test_summaries.py), [summary integration](/home/jugaadu/Projects/hermes-memory/tests/integration/test_summarization.py) |
| T05 | [Formation](/home/jugaadu/Projects/hermes-memory/tests/unit/test_formation.py), [worker](/home/jugaadu/Projects/hermes-memory/tests/unit/test_worker.py) |
| T06 | [Allowances](/home/jugaadu/Projects/hermes-memory/tests/unit/test_allowance.py) |
| T07 | [Resource gate](/home/jugaadu/Projects/hermes-memory/tests/unit/test_resource_gate.py), [identity](/home/jugaadu/Projects/hermes-memory/tests/unit/test_identity.py) |
| T08 | [Inquiry safety](/home/jugaadu/Projects/hermes-memory/tests/integration/test_inquiry_safety.py), [inquiries](/home/jugaadu/Projects/hermes-memory/tests/integration/test_inquiries.py), [inquiry reporting](/home/jugaadu/Projects/hermes-memory/tests/integration/test_inquiry_reporting.py), [delivery](/home/jugaadu/Projects/hermes-memory/tests/integration/test_delivery.py) |
| T09 | [Measurements](/home/jugaadu/Projects/hermes-memory/tests/unit/test_measurements.py), [evaluation door](/home/jugaadu/Projects/hermes-memory/tests/integration/test_evaluation_door.py), [evaluations](/home/jugaadu/Projects/hermes-memory/tests/integration/test_evals.py) |
| T10 | [Recovery faults](/home/jugaadu/Projects/hermes-memory/tests/faults/test_recovery.py), [review regressions](/home/jugaadu/Projects/hermes-memory/tests/review/test_feature_review.py) |
| T11 | [Install/upgrade](/home/jugaadu/Projects/hermes-memory/tests/installer/test_install_upgrade.py), [release](/home/jugaadu/Projects/hermes-memory/tests/installer/test_release.py) |

### Related analysis and external specifications

- [Memory intelligence review and M-001–M-034 register](memory-intelligence-optimization-review.md): primary finding evidence and limitations.
- [Feature-flow review and R-001–R-032 register](memory-framework-feature-review.md): earlier source fixes and deployment distinction.
- [Integration audit](hermes-memory-integration-audit.md), [gateway clarification review](hermes-gateway-clarification-review.md), [source inventory](hermes-memory-source-inventory.md): historical context; recheck current source before treating an old reported bug as still open.
- [Qwen embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [Qwen reranker model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B): model-specific input/scoring contracts; not evidence of this deployment's language scores.
- [Hindsight retain](https://hindsight.vectorize.io/developer/api/retain), [recall](https://hindsight.vectorize.io/developer/api/recall), [reflect](https://hindsight.vectorize.io/developer/api/reflect): current public semantics, with installed-version verification required.
- [SQLite FTS5](https://www.sqlite.org/fts5.html): tokenizer/query/index design reference. Exact application preprocessing behavior is established by C03 and regression fixtures.

The implementation should end with a smaller, trustworthy, measured memory system—not an accumulation of unbounded features. The foundation is correct context, complete support, controlled formation, relevant packets, and Hermes-visible clarification; optional models and extra concurrency come after that evidence.
