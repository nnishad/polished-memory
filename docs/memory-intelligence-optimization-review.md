# Memory intelligence and performance review

Date: 2026-10-02. Scope: the custom Hermes memory framework, its Hermes provider, the installed Hindsight integration, multilingual memory formation, retrieval, associations, and personal-assistant behavior.

Implementation follow-up: [detailed implementation plan](memory-intelligence-implementation-plan.md), including all M-001–M-034 findings, core contracts, dependencies, tests, benchmark gates, and deployment/rollback requirements. Subsequent core-source work and remaining gaps are tracked in the [implementation progress ledger](memory-intelligence-implementation-progress.md); installation remains deferred.

**Analysis only. No application code, configuration, installation, service state, or production memory was changed.** Synthetic checks used disposable/in-memory data; production inspection used read-only aggregate SQL. Installation remains deferred, as requested.

## 1. Main conclusion

The framework has a strong durability and consent foundation. It is not merely an embedding database: it already separates canonical evidence, derived facts, identity, assertions, summaries, goals, lessons, clarifications, delivery, and forgetting. Those distinctions should be preserved.

The largest improvement is not a bigger model or an unrestricted graph. It is making the existing layers work together: retain conversational meaning and attribution, reliably project new evidence, retrieve the right evidence with its provenance, and put a small useful packet into Hermes at the right moment.

The most important findings are:

1. Hindsight maintenance task names do not match the framework's route names. Enabling consolidation alone would not repair the worker path.
2. Observation recall loses the supporting `source_facts` before the broker verifies provenance. A valid observation can therefore be withheld even when its evidence was returned.
3. Summary production declares the planned local records as citations without checking that they are the records used by Hindsight's reflection.
4. Canonical Hermes events retain speaker metadata, but formation sends plain text and only source/record IDs. Extraction loses useful attribution and conversation context.
5. All 37 Hermes records in the inspected canonical archive lack an occurrence timestamp. Relative-time formation needs a separate, reliable utterance-time anchor.
6. Qwen's multilingual embeddings are a useful foundation, but this installation has no demonstrated Hinglish extraction-and-recall score. Query instructions, spelling variation, negation, reference resolution, and dates need their own validation.
7. Automatic consolidation is disabled, and recall uses RRF without a learned reranker. These are bounded-cost choices, not proof of a broken installation, but they limit the configured intelligence path.
8. The provider's packet is only 1,200 estimated tokens, with raw evidence packed before backend facts. Recall can spend time finding semantic matches that never reach Hermes.

“Human-brain-like” is a design analogy here: remember experiences, form cautious abstractions, connect related events, recall selectively, learn from verified outcomes, and forget reliably. It is not a claim of consciousness, perfect recall, psychological diagnosis, or autonomous authority over the owner.

## 2. Evidence, scope, and deployment boundaries

### What was inspected

- Framework source: `/home/jugaadu/Projects/hermes-memory`, including capture, source adapters, evidence/identity/lineage, formation/worker/gates, Hindsight client/provenance, broker/cache/packet, assertions/summaries, goals, proactivity, learning, measurements, lifecycle, and evaluation entry points.
- Hermes integration: `integrations/hermes-memory/provider.py`, `capture.py`, and the existing integration/clarification audit documents. The Hermes repository's architecture/cache invariants were also read.
- Installed backend source: `/home/jugaadu/data/hermes-memory/runtime/current/hindsight/lib/python3.12/site-packages/hindsight_api/`, particularly embeddings, retain extraction/chunking/link creation, recall, task submission, and worker polling.
- Whitelisted non-secret environment settings and the three memory service states.
- Canonical/gate database aggregate metadata only, with SQLite `mode=ro` and `query_only=ON`.
- Official Qwen model cards and Hindsight documentation. Installed source takes precedence where current documentation may describe a different revision.

This is a feature-family and important-path review, **not an exhaustive branch audit**. Existing feature-flow coverage and historical fixes are recorded in [the feature review](memory-framework-feature-review.md). Actual extraction quality, Telegram delivery, vector distributions, graph density, and model-contended latency were not benchmarked in this pass. No personal message bodies were sampled for language-quality evaluation.

### Snapshot observed during this review

| Item | Observation | Interpretation |
| --- | --- | --- |
| Framework checkout | HEAD `e26ae2309d73e969a7079372f529b0a4f1ca3148`, with substantial existing changes | Findings refer to working-tree source, not a clean HEAD |
| Current runtime target | `237133b492abed552e4567652b1f218f87bbf75b` | An immutable installed release; not equivalent to current working bytes |
| Backend distribution | `hindsight-api-slim`, version `0.10.1` | Verified from installed package metadata |
| Canonical schema | Latest migration `0014_inquiry_delivery` | Source now includes `0015_durable_fences`; the previous fixes have not been installed by this pass |
| Archive | 43 records, 40 visible | A small archive; not a basis for large-scale speed claims |
| Projection mappings | 40 `verified`, 1 `absent` | Mapping presence is not an extraction-quality score or a proof of exact per-record coverage |
| Hermes event timing | 37 Hermes records; all 37 have `occurred_at = NULL` | Confirmed timing coverage gap; source distinction discussed below |
| Backend operation ledger | 43 `retain` operations marked `finished` | Evidence of historical retention work, not of running consolidation/refresh |
| Services | Gate, Hindsight API, and Hindsight worker all active/running | Process health does not establish feature quality or continuous formation |

The earlier R-001–R-032 source fixes remain a separate register. Findings below are additional gaps/opportunities, not a claim that the earlier regression fixes were undone. Their deployment status must remain explicit.

### Configured inference path

| Function | Configured path / setting | Important implication |
| --- | --- | --- |
| Extraction and reflection | Hindsight → loopback gate `127.0.0.1:8123/v1` → `192.168.68.69:8080/v1` | The updated server address is configured; no live inference was sent to it |
| Text model | `Cobra91310/Ornith-1.5-9B-MTP-NVFP4-newhead:NVFP4` | Its Hinglish extraction accuracy is unmeasured here |
| Embeddings | Hindsight → loopback gate → `127.0.0.1:11434/v1` | The text model being remote does not make recall embeddings remote |
| Embedding model | `qwen3-embedding:0.6b`, 1,024 dimensions | Runtime vector length, norm, truncation, and server-side prompt behavior still need checks |
| Embedding prefixes | Not set in the inspected backend env; installed defaults are empty | Framework-side query instruction is absent from this configuration; an upstream template was not inspected |
| Retain chunk size | 1,500 **characters** | Not 1,500 model tokens; token cost varies by language |
| Retain completion cap | 2,048 tokens | A ceiling, not an extraction-quality guarantee |
| Extraction mode | No env override found; installed default `concise` | Per-bank overrides were not inspected |
| Automatic consolidation | `false` | No automatic observation-building should be assumed |
| Reranker | `rrf` | No learned cross-encoder reranking on this configured path |
| Worker/concurrency | One worker slot; inspected model concurrency limits are one | Safe resource bounds, with possible head-of-line blocking |
| Foreground memory deadline | 6 seconds | A maximum tolerated wait, not observed p95 latency |
| Embedding gate priority | `freshness`, not `interactive` | Foreground recall shares the background embedding admission policy |
| Background gate queue | No env override found; source default 120 seconds | Verify effective runtime settings before a performance change |
| Native provider batch / OCR | Disabled | Do not enable bypass paths merely to improve throughput |

Source anchors: `config.py`, `processing/routes.py:build_routes`, `processing/gate_server.py:wait_for`, `backend/worker_launcher.py:slot_contract`; installed `hindsight_api/config.py`, `engine/embeddings.py`; whitelisted env fields. Env/API credentials are intentionally omitted.

## 3. How the existing memory layers connect

```text
Hermes conversation / external sources
                │
       capture spool + bounded adapters
                │
      canonical evidence + identity + lineage
                │
        approved formation / durable jobs
                │
      Hindsight retain → facts / entities / links
                │                 │
                │       consolidation → observations
                │       [disabled; task route gap]
                │
query → local lexical/typed retrieval + Hindsight recall
                │
       provenance + access + freshness checks
                │
        bounded context packet → Hermes turn

canonical evidence → summaries / lessons / goals
                    → inquiry / attention / outbox
                    → Hermes gateway → owner reply
                    → authenticated decision + updated context

forget/correction/revocation → invalidate every dependent layer
```

The diagram shows architectural paths, not a claim that every producer is running automatically.

| Brain-like function | Existing substrate | What is missing or needs strengthening |
| --- | --- | --- |
| Episodic memory: what happened | Immutable canonical records, spool, timestamps, source provenance | Conversation episodes, utterance anchors, fast semantic access before slow formation |
| Semantic memory: what is known | Hindsight facts, canonical assertions, summaries | Reliable multilingual extraction, observation provenance, temporal supersession |
| Associative memory: what connects | Hindsight entities and semantic/temporal/causal link implementations | Safe alias handling, useful multi-hop recall, maintenance task attribution |
| Procedural memory: how to do things | Lessons, outcomes, evaluation ledger | Reliable capture of verified task outcomes and contextual lesson retrieval |
| Prospective memory: what must happen | Candidate/active goals, predicates, due events, attention policy | Relevant obligations in ordinary Hermes context, not just notification delivery |
| Working memory: what matters now | Session context and bounded broker packet | Context-aware query planning and better packet allocation |
| Uncertainty and correction | Contradictions, provenance verdicts, inquiries, owner decisions | Ambiguity detection with context-rich, gateway-routed clarification |
| Forgetting | Erasure obligations, visibility, reset/recovery, source fixes | Deployment of existing fixes and end-to-end graph/index/backup/cache verification |

Canonical lineage is a dependency/revision graph. It is **not** the same thing as a semantic association graph. Identity confirmation is an access-authority decision. It is **not** the same thing as discovering that two memories mention similar names.

## 4. Hinglish: what is supported, and what is not established

### Embeddings are only one stage

Qwen3-Embedding-0.6B is documented as multilingual and instruction-aware, with a maximum 1,024-dimensional representation. Its official usage distinguishes instructed queries from document text. This makes it a reasonable model to evaluate for cross-language personal recall; the published model card does not establish the accuracy of this installation on conversational Hinglish. [Official Qwen embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

The complete quality chain is:

```text
original message → contextual extraction → factual representation
                 → embedding/index → query interpretation/ranking
                 → authorized evidence selection → Hermes answer
```

An excellent embedding cannot recover a name, negation, speaker, date, or condition that extraction discarded. A correct fact does not help if provenance rejects it or packet packing omits it.

The installed Hindsight extractor already instructs preservation of the input language/script and includes reference-resolution and temporal guidance. Therefore, “Hindsight only supports English facts” is **not** a supported finding. The remaining question is whether the configured model follows that guidance on mixed-script, colloquial, ambiguous input. The relative-date fallback regex inspected in `_infer_temporal_date` is English-oriented; it does not by itself resolve `kal`, `parso`, or `agle hafte`.

### Recommended representation

Keep the original evidence immutable and recognizable. Add derived, versioned fields rather than replacing Hinglish with English:

- Original text, source, speaker, role, utterance timestamp, timezone, and exact supporting span.
- Minimal conversational context needed to resolve a reference; provenance for every contributing turn.
- Extracted claim: subject, relationship, value, negation/polarity, modality, conditions, and temporal validity.
- Original surface forms plus cautious aliases: `chai/chaai`, Hindi/Roman-script variants, declared nicknames, project names.
- Optional normalized retrieval text or translation, explicitly marked as derived. Never use it as a second independent witness.
- Uncertainty and unresolved reference/date fields. An ambiguous `kal` stays ambiguous until context or the owner resolves it.
- Processor identity covering extractor/prompt, model, embedding configuration, and normalization policy.

Do not globally rewrite names, remove negations as stopwords, infer identity from transliteration, or turn quoted/hypothetical preferences into confirmed preferences.

### Synthetic language cases for the later evaluation

These are invented examples, not excerpts from the owner's archive. Temporal interpretation assumes a known utterance date and owner timezone.

| Evidence / context | Recall question | Required behavior |
| --- | --- | --- |
| `mujhe chai pasand hai, coffee nahi` | “What drink do I prefer?” | Retrieve the preference and its coffee negation; do not rank a generic coffee mention as the answer |
| Same evidence | `मेरी पसंद की drink क्या है?` | Cross-script/cross-language retrieval with original evidence available |
| `chaai bina chini wali` | `chai kaise pasand hai?` | Preserve “without sugar”; spelling variation should not lose the match |
| “Rohit from finance called.” → `usko Friday ko reply karna` | “Who needs a reply?” | Resolve `usko` only from attributable conversation context; propose an obligation, not automatically authorize delivery |
| `kal Rohit se baat hui` | “When did we speak?” | Interpret past-tense `kal` relative to utterance time; retain uncertainty if the anchor is absent |
| `kal Rohit ko call karna` | “What is pending?” | Interpret future intent distinctly from a past event; confirm scheduling details when needed |
| `parso wala meeting cancel hai` | “Is the meeting still happening?” | Preserve cancellation and identify which meeting; ask if `parso` or the referent is ambiguous |
| `ab se coffee kam, chai zyada` | “What is my current preference?” | Recognize change without erasing historical evidence or promoting an uncertain value prematurely |
| `agar baarish hui toh cab lena` | “How should I travel?” | Conditional guidance, not an unconditional cab preference |
| `Priya ne bola 'mujhe coffee pasand hai'` | “What do I like?” | Attribute the quoted preference to Priya, not the owner |
| Assistant: “I booked it.”; no tool receipt | “Was it booked?” | Retrieve an assistant claim, not verified booking completion |
| Two different people named Rohit | `Rohit ka address?` | Keep people separate; clarify rather than transfer identity/access |
| `delete mat karna` | “Should this be deleted?” | Negation must survive; no destructive intent is inferred |
| `delete kar do` quoted inside a document | Any memory-management action | Treat retrieved text as evidence, not an executable owner command |

Use language combinations in both directions: Hinglish→English, English→Hinglish, Devanagari→Roman Hindi, and mixed-script→mixed-script. Include short replies, typos, sarcasm, uncertainty, corrections, and unknown-answer cases. Preserve a held-out set that was not used to tune prompts or aliases.

## 5. Findings and optimization register

Status vocabulary: **Verified** = source/config fact; **Probe** = reproduced with synthetic data; **Opportunity** = proposed design improvement; **Unmeasured** = requires real quality/latency evidence. Priority P0 means fix/validate the foundation before intelligence expansion, P1 means the next usefulness/performance work, and P2 means measurement-driven enhancement. It is not a CVSS rating.

All M-items remain **open for a later implementation/evaluation pass**. No fixes were made during this analysis.

### A. Correct memory formation and derived provenance

#### M-001 — Hindsight task-kind to route mapping is incomplete — P0, Probe

**Evidence:** `backend/worker_launcher.py:_resource_for` directly calls `routes.by_name(_kind(task))`. `processing/routes.py:build_routes` creates `consolidate`, while the installed engine submits `operation_type="consolidation"`. It also submits `refresh_mental_model` and `graph_maintenance`, neither of which has a corresponding route. The poller injects the DB operation type before executor invocation. A synthetic check returned a resource for `retain`, but refused all three maintenance kinds.

**Impact:** Those tasks cannot pass this attribution wrapper. Consolidation being disabled currently limits exposure; enabling it would reveal a separate worker integration blocker. DB-only graph/index upkeep must not be treated as an unknown model request.

**Recommendation:** Use an explicit, pinned task-kind policy mapping: model-backed tasks to admitted inference routes, DB-only maintenance to bounded DB/CPU accounting, unsupported tasks to clear refusal. Mental-model refresh also needs the correct configured model/cap/authorization semantics. Do not add aliases that conceal unsupported task classes.

**Acceptance:** Exercise actual pinned poller payloads for retention, consolidation, graph/index maintenance, refresh, cancellation, and folded tasks. Unknown kinds remain fail-closed. No extra owner authority or inference bypass is introduced.

#### M-002 — Valid observations lose their provenance at broker integration — P0, Probe

**Evidence:** `HindsightClient.recall` requests source facts, and `RecallOutcome` stores them. `context/broker.py:_derive` carries only `outcome.results` forward. `_fact_allowed` resolves canonical record/document IDs but does not follow `source_fact_ids`. Installed recall returns observation source IDs plus a separate source-fact map. Synthetic result: one observation + one mapped supporting source fact supplied; zero observation facts delivered; derived channel `partial`.

**Impact:** Consolidation could produce useful observations that the safe broker cannot use. Removing provenance checks would be the wrong fix.

**Recommendation:** Resolve the response's bounded provenance graph through the document map and canonical identity/visibility/revision state. Retain all dependencies; surface original evidence, not just synthesized text. Missing/truncated support stays partial or withheld.

**Acceptance:** Direct facts, multi-source observations, nested/missing references, budget truncation, forgotten support, and cross-account support are tested with pinned payload shapes. Authorized complete observations work; unsafe ones never render.

#### M-003 — Reflection citations are planned inputs, not verified model support — P0, Verified

**Evidence:** `processing/summarization.py:_reflect` calls `holder.reflect(_question(proposal), ...)` without a hard document/tag restriction. `_question` describes a scope/window in prose. `summarize_apply` publishes citations from `proposal["selected"]`; returned backend facts are counted but not reconciled with those citations.

**Impact:** The stored manifest can claim local evidence supports a reflection even if Hindsight used other facts or omitted planned records. This is a provenance-contract defect; an actual out-of-scope model answer was not generated in this pass.

**Recommendation:** Restrict retrieval using supported hard scopes where possible, map the returned support, validate it against the approved canonical set/window, and publish only genuinely supported content. If the backend cannot enforce the scope, synthesize from an explicitly bounded authorized evidence set instead. The returned support still needs checking.

**Acceptance:** An injected reflection citing an outside document or returning no support is refused/marked incomplete. Corrections and withdrawals invalidate the resulting summary. Planned coverage and actual cited coverage are reported separately.

An explicit Hindsight temporal window steers temporal ranking; it is not a hard period filter. Thus adding that field alone would not establish scoped summary provenance. [Official recall API semantics](https://hindsight.vectorize.io/developer/api/recall).

#### M-004 — Role and conversation context disappear on projection — P0, Verified

**Evidence:** `sources/sdk.py:_stamp` records `role`, `origin`, `independent`, author, and session metadata. `_turn` intentionally separates user and assistant records. `processing/worker.py:run_once` sends `evidence.text`, timestamp, and metadata containing only `source` and `record_id`.

**Impact:** A first-person assistant claim can resemble a user statement. A short utterance referring to a previous turn cannot be reliably interpreted in isolation. Repetition of assistant output must not become corroboration.

**Recommendation:** Project role/author/session/time/trust explicitly and provide a bounded context capsule or attributable conversation episode. Preserve separate canonical records. A `user` role is not by itself proof of the authenticated owner's identity, particularly in group conversations. Multi-turn extraction requires multi-record support, not a fake merged document with one misleading citation.

**Acceptance:** User preferences, quoted third-party statements, assistant guesses, delegation reports, and verified tool outcomes remain distinct. References resolve when supported and stay unresolved otherwise.

#### M-005 — Utterance time is missing from the formation contract — P0, Verified + aggregate evidence

**Evidence:** `provider.py:sync_turn` writes spool `created_at`, but not an explicit event timestamp in its payload. `sources/sdk.py:_frame` derives `occurred_at` from payload `occurred_at` or event `timestamp`; `_arrival_of` preserves `created_at` separately as `spooled_at`. Formation forwards only `occurred_at`. Read-only SQL found all 37 Hermes records with null occurrence time.

**Impact:** The extractor lacks a reliable anchor for relative statements. Missing occurrence time is not necessarily incorrect for an undated preference; losing the utterance time needed to interpret that preference/event is the gap.

**Recommendation:** Keep utterance time, ingestion time, event time, precision, timezone, and the basis of a derived date separate. Forward the original utterance anchor without pretending every claim happened then. Historical imports must not inherit today's clock.

**Acceptance:** Replay the same conversation later and obtain the same supported dates. `kal`/`parso`, timezone/DST boundaries, late imports, undated states, and missing anchors have explicit outcomes.

#### M-006 — Processor identity does not describe the actual intelligence pipeline — P1, Verified

**Evidence:** Formation and summary fingerprints include pinned backend version, bank, route/resource/upstream, and output cap. They do not explicitly include configured text-model identity, prompt/extraction version, embedding revision/dimensions/prefix, normalization, or chunk policy. `formation.py:_SELECT` considers a verified revision/epoch projected without a processor-generation condition; `DocumentMap.begin` short-circuits an already verified mapping.

**Impact:** Switching a model or extraction policy can leave old facts labeled covered, with no controlled refresh. A new job fingerprint alone does not guarantee re-extraction.

**Recommendation:** Version projection generations independently of canonical evidence revisions. Define compatible changes, re-extraction changes, and vector-space changes. For incompatible embeddings, use a shadow index/bank and approved cutover; retain erasure fences throughout.

**Acceptance:** A processor upgrade selects exactly affected projections, does not duplicate canonical evidence, can roll back safely, and cannot resurrect forgotten records.

#### M-007 — Hinglish extraction accuracy is not established by embedding capability — P1, Unmeasured

**Evidence:** Installed extraction has useful language/reference/date instructions; the configured 9B model performs extraction, not Qwen embedding. No live bilingual extraction benchmark was run. The inspected relative-date fallback is English-oriented.

**Impact:** Incorrect polarity, attribution, conditionality, names, or dates can become durable derived facts.

**Recommendation:** Evaluate the current extractor before changing models. Add focused examples/structured uncertainty through supported bank/extraction configuration if evaluation justifies it. Avoid unbounded prompts and do not auto-confirm difficult claims.

**Acceptance:** Measure per-field extraction precision/recall on held-out Hinglish and English cases, especially negation, speaker, reference, time, and current-vs-historical status.

#### M-008 — Semantic facts do not automatically become canonical assertions — P1, Opportunity

**Evidence:** `knowledge/assertions.py` provides candidates, confirmation, validity, contradiction, and supersession. Formation projects raw records to Hindsight; it is not a verified canonical assertion-extraction producer.

**Impact:** Hermes may repeatedly rediscover a preference without obtaining a structured, time-valid current answer or a explicit contradiction workflow.

**Recommendation:** Add a bounded candidate-assertion producer that cites exact spans, distinguishes observation from confirmation, and respects current supersession rules. Clear explicit owner statements and inferred patterns need different promotion policies.

**Acceptance:** Corrections supersede only on the appropriate authority; an inferred or tentative preference cannot silently replace a confirmed one. Historical questions retain historical answers.

### B. Continuous formation, consolidation, and associations

#### M-009 — Running workers do not prove continuous memory formation — P1, Verified path gap

**Evidence:** The inspected services run API/gate maintenance and Hindsight's existing task worker. `formation_apply` is reached by the operator CLI; no automatic caller was found in those service paths. Maintenance explicitly performs no inference. No separate memory formation service/timer file was found among the inspected user units.

**Impact:** New captured records can remain lexical-only until somebody invokes approved formation. An uninspected external scheduler could change this; none was established here.

**Recommendation:** Reuse the existing bounded owner allowance to authorize an incremental producer with backpressure, observable lag, pause handling, and per-profile fair scheduling. Do not reinterpret an active worker as standing permission to spend inference.

**Acceptance:** With an approved grant, new evidence forms within a measured freshness target; absent/expired grants cause an explicit backlog, not unapproved calls. Shutdown/restart preserves work and attribution.

#### M-010 — Formation treats all live record kinds similarly — P1, Verified + Opportunity

**Evidence:** `formation.py:_SELECT` selects live unprojected records in ingestion order, without a kind/salience-specific policy. Per-item token estimation is fixed rather than text-size/language dependent.

**Impact:** Greetings, repeated model prose, bulk telemetry, and valuable corrections compete for the same formation budget. Short records may be overestimated; long or token-dense records may be underestimated.

**Recommendation:** Keep approved raw capture separate from selective derived formation. Prioritize explicit remembers, corrections, commitments, project decisions, and independently supported outcomes; use fair aging so old records do not starve. Aggregate typed telemetry deterministically before deciding whether a narrative is useful.

**Acceptance:** Measure valuable-memory retention, formation lag by class, cost per useful fact, and starvation. Salience filtering must not erase canonical evidence or exclude all low-frequency personal details.

#### M-011 — One-record submission bounds provenance but has throughput overhead — P2, Opportunity

**Evidence:** Formation creates one job per record because jobs carry a single input revision; the worker submits individual records and polls operations. The Hindsight client supports bounded multi-item async requests, but the current job/mapping contract does not safely exploit arbitrary mixed revisions.

**Impact:** Large backlogs can spend disproportionate time on HTTP/polling/prompt overhead. This is not a measured bottleneck in the 40-visible-record archive.

**Recommendation:** First measure the components. If needed, implement safe micro-batches with per-item document/revision/submission accounting, exact recovery semantics, and bounded token-size buckets. Provider batch execution remains disabled unless it gains the same gate/budget guarantees.

**Acceptance:** Out-of-order acknowledgements, partial failure, cancellation, correction, and crash recovery do not lose or duplicate evidence. Compare records/sec and tokens/useful fact against individual submissions.

#### M-012 — Observation formation is deliberately disabled — P1, Verified

**Evidence:** Actual backend env sets `HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION=false`. Installed Hindsight has a consolidator and observation support.

**Impact:** The configured path does not automatically build recurring-pattern observations. This does not mean there are no facts or connections: retain already has entity and link machinery.

**Recommendation:** After M-001–M-003, introduce governed incremental consolidation of changed evidence under an explicit allowance. Prefer scoped batches, dirty-set tracking, and refresh thresholds over repeatedly reflecting over the entire bank. Keep observations tentative and attributable; avoid unsolicited personality judgments.

**Acceptance:** A recurring pattern produces supported, nonduplicated observations at a bounded measured cost; source correction/forgetting invalidates them. Ordinary recall does not trigger consolidation.

#### M-013 — Reuse native associations before building another graph — P2, Verified capability + Opportunity

**Evidence:** Installed retain code resolves entities and creates temporal, semantic, and causal link types; recall includes graph retrieval. `storage/lineage.py` tracks canonical dependencies, not semantic similarity. Actual link counts, causal-extraction settings, and graph quality were not measured.

**Impact:** A duplicate custom graph could create two inconsistent identity/erasure/index systems while solving a capability that largely exists upstream.

**Recommendation:** Measure native graph usefulness first. Use canonical typed relations only where needed for stable project/person/task semantics and authority. Aliases should preserve surface form, provenance, ambiguity, and scope; similarity is not confirmed identity. Bound multi-hop fanout and follow evidence-backed edges.

**Acceptance:** Questions requiring one/two hops improve recall over dense-only retrieval. Same-name people stay separate. Every returned connection has a source or is marked a hypothesis; deletion removes affected traversable support.

### C. Retrieval quality and foreground performance

#### M-014 — Qwen query instructions are not configured at this boundary — P1, Verified configuration; end-to-end behavior unmeasured

**Evidence:** The inspected env has no query/passage prefix. Installed `config.py` defaults both to empty; `Embeddings.encode_query` and `encode_documents` support separate prefixes. `OpenAIEmbeddings` otherwise forwards the input text. Server-side template behavior was not inspected.

**Recommendation:** Verify the actual Ollama model/template path, then A/B a task-specific query instruction exactly once. Do not apply the query instruction to stored document text or double-prefix it. Test vector ordering, length, finiteness, normalization expectations, and truncation as an end-to-end contract. Use existing supported configuration rather than patching installed packages.

**Acceptance:** Mixed-language held-out recall improves or remains acceptable with a measured cost. Prefix changes are versioned; incompatible document-side changes require a controlled reindex. Model-name similarity alone does not establish vector compatibility.

#### M-015 — Literal lexical fallback is brittle for natural language and script variation — P1, Probe + Opportunity

**Evidence:** `storage/evidence.py:fts_query` uses a conjunction of up to 24 quoted tokens. `fts_terms` uses the regex `[^\W]+`, which splits Devanagari words around combining marks. In an in-memory check, `मुझे चाय नहीं चाहिए` became `म, झ, च, य, नह, ह, ए`. The exact synthetic sentence still matched FTS; this is evidence of segmentation, **not** proof that all Hindi search fails.

**Impact:** Long conversational queries can require irrelevant words; Romanization variants and cross-script queries do not receive semantic support from this channel. Fragmentation can distort word-level matching/ranking. Dense recall should complement, not remove, the outage-safe exact channel.

**Recommendation:** Preserve an exact identifier/phrase route; add a bounded entity/key-term or weighted alternative route with Unicode-aware segmentation and tested aliases. Never strip meaningful negation. Fuse candidate ranks rather than pretending FTS and vector scores have the same scale.

**Acceptance:** Same-script literals, spelling variants, IDs with punctuation, irrelevant query words, and negated statements are evaluated separately. Adversarial FTS syntax remains safely quoted.

#### M-016 — Current-turn text is not a complete retrieval plan — P1, Verified + Opportunity

**Evidence:** `provider.py:prefetch` passes the current query string to the broker. The explicit recall tool can receive a task descriptor for lessons, but ordinary recall does not carry a full episode, resolved referents, project/person targets, or temporal intent.

**Impact:** `haan wahi`, `uska kya hua?`, and “continue the migration” may recall poorly despite relevant stored evidence.

**Recommendation:** Use a small structured retrieval descriptor from current Hermes state: topic/entities, task, session, utterance anchor, and supported references. Start with deterministic context carry-forward; use an optional bounded query reformulator only when necessary. Search expansions are hypotheses and never facts to retain.

**Acceptance:** Short follow-ups retrieve the correct task without cross-session/person leakage. Compare reformulation cost against quality benefit and retain the original query for attribution/debugging.

#### M-017 — RRF-only recall leaves learned relevance discrimination unused — P1, Verified configuration + Unmeasured benefit

**Evidence:** Actual reranker provider is `rrf`. Installed ranking supports this passthrough and a learned reranker path; the configured path is not a Qwen cross-encoder deployment.

**Recommendation:** A/B a small multilingual reranker over a bounded top candidate set, only for ambiguous/deep recall if worthwhile. Keep RRF as a fast fallback. Qwen3-Reranker-0.6B is a candidate, but its official scoring examples use a specialized query/document formulation and relevance token scoring; a generic chat response is not a compatible rerank API. [Official Qwen reranker model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).

**Acceptance:** Measure nDCG/MRR, wrong-person/negation errors, p95 latency, and GPU contention. An unavailable reranker degrades explicitly; it does not block all memory. Confirm the pinned Hindsight provider/API contract before wiring a model.

#### M-018 — Search depth is not adapted to query difficulty — P1, Verified + Opportunity

**Evidence:** The broker calls `client.recall` without a budget override, so the client defaults to `mid`, with `types=None`. Installed fixed budget defaults are 100/300/1,000 for low/mid/high; an installed method docstring still describes high as 600, so the actual config path—not that comment—is authoritative. Per-bank overrides were not inspected.

**Impact:** A tiny lookup and a multi-hop project question use the same client plan, despite different costs and evidence needs.

**Recommendation:** Support explicit fast/balanced/deep modes, with the foreground default selected by need. Restrict fact types when meaningful. Avoid foreground reflection. Search depth, candidate/reranker caps, and returned context size should be tuned independently. [Official retrieval tuning documentation](https://hindsight.vectorize.io/developer/retrieval).

**Acceptance:** Simple questions meet a measured fast budget; difficult questions improve with deeper search; automatic escalation is bounded and observable. Authorization is unchanged by any mode.

#### M-019 — Packet order can crowd out the semantic channel — P1, Verified

**Evidence:** `ContextBroker.assemble` reserves lessons/commitments/assertions, packs raw lexical evidence and summaries, then obtains/budgets backend facts. Hermes sets a 1,200-token packet ceiling and normally requests eight raw items.

**Impact:** A long lexical match can consume the room before a more useful semantic fact arrives. The backend work may therefore add latency but no delivered information.

**Recommendation:** Collect bounded candidates and then allocate across independent evidence, current assertions, obligations, and derived support. Deduplicate by canonical lineage. Keep a small semantic reserve or use query-dependent packing; fit useful spans rather than full documents. Do not duplicate a summary and all its sources as independent confirmation.

**Acceptance:** Measure recall before packing and after packing. Useful semantic support survives a long raw match, without losing cancellation/negation or exceeding the host's actual token budget.

#### M-020 — Character limits and token accounting do not match delivered utility — P1, Verified + Unmeasured severity

**Evidence:** Broker's default estimator is `len(text)//3`. It charges raw full-text candidates; rendered raw items use only the first 600 characters. Backend text is also prefix-clipped. Hindsight include sections have separate budgets; the broker requests entities but does not currently use them in the packet.

**Impact:** Prefixes can omit the answer or qualifying clause. Character-based estimates are not calibrated to the Hermes model or language, and full-text charges can waste scarce packet room. Unused entity payload can add overhead.

**Recommendation:** Use the actual host tokenizer when available, or calibrate a conservative bounded estimator per script. Select evidence-centered spans with small context and citation offsets. Budget metadata/labels/include sections too, and request only what is used. The existing injectable estimator is preferable to a new independent tokenizer subsystem.

**Acceptance:** Actual rendered token count respects the ceiling in English/Hinglish/Hindi; tail answers and negations remain intact. Record truncation and provenance truncation separately.

#### M-021 — Foreground embeddings have background gate semantics — P1, Verified configuration + Unmeasured contention

**Evidence:** `build_routes` gives the shared embeddings route `freshness` priority. `GateApplication.wait_for` treats only `interactive` routes as immediate foreground admission; other routes use the queue allowance. Foreground deadline is six seconds, while default background queue allowance is 120 seconds.

**Impact:** Recall can wait behind background local-GPU work under a policy designed for freshness processing. Queueing, not vector search, may dominate perceived latency. The user being served interactively is not conveyed by the current embedding credential.

**Recommendation:** Distinguish foreground query embeddings from background document embeddings with authenticated request purpose/deadline and correct bank attribution. Bound interactive queue wait, reserve small admission opportunities when possible, and degrade explicitly. Do not falsely label background work interactive or bypass the device gate.

**Acceptance:** Measure foreground recall while retain, vision, and local-model requests compete. Tail latency and background starvation both remain bounded.

#### M-022 — End-to-end cancellation/deadline behavior needs measurement — P1, Verified mechanics + Unmeasured effect

**Evidence:** Broker uses a two-worker executor and cancels a timed-out future; that cannot interrupt an already running synchronous HTTP request. Plugin client timeout is also bounded. Installed Hindsight checks disconnection before DB-heavy recall, but the whole embedding/gate/upstream cancellation chain was not exercised.

**Impact:** Hermes abandoning a request need not mean queued inference stopped. Repeated slow requests can consume capacity after they ceased being useful.

**Recommendation:** Propagate one end-to-end deadline, use bounded pending work/single-flight where safe, prevent stale queued admission, and preserve “outcome unknown” accounting for work already started. Optimize read cancellation separately from durable write reconciliation.

**Acceptance:** Repeated timeouts/disconnects do not create unbounded queues, block subsequent turns, or incorrectly release a device still working. Verify actual server behavior, not only futures in mocks.

#### M-023 — Safe cache policy may have low hit rates in active conversations — P2, Verified + Opportunity

**Evidence:** Packet cache uses normalized exact query, scope, epoch, and global context revision; defaults are 64 entries/20 seconds. Provider broker/cache ownership is per thread. Source fixes make authority invalidation stricter, correctly.

**Impact:** New canonical/derived writes can invalidate unrelated packets; changing thread or wording may miss reuse. This is a possible efficiency cost, not permission to relax forgetting or access checks.

**Recommendation:** Measure real hit/miss/invalidation reasons first. Consider dependency-scoped reuse, exact entity/goal subcaches, versioned query-embedding cache, and scoped single-flight. Always validate current authority and erased dependencies at handoff.

**Acceptance:** Warm latency improves under active writes; forget/revoke/withdraw/identity expiry immediately prevents stale disclosure. Similar wording never licenses reuse across callers or profiles.

### D. Personal-assistant completeness and trust

#### M-024 — Scoped summaries are not retrieved through the caller-aware read path — P1, Probe

**Evidence:** Broker `_summaries` calls `SummaryStore.latest(scope)` without caller/window/source arguments. `latest` resolves provenance without an account. Default `SummaryStore` also constructs a ledger without an identity resolver. Synthetic default-path read withheld an account-scoped summary; with an explicit identity-aware ledger, an account-aware read succeeded while `latest` still returned zero.

**Impact:** Correctly scoped summaries can disappear even for an authorized caller. Latest-scope selection also does not establish relevance to an explicit retrieval window. This is primarily a fail-closed availability/filtering gap, **not a demonstrated account leak**.

**Recommendation:** Carry caller identity and retrieval filters through summary selection and every cited dependency. Avoid broad source summaries that mix unrelated people/topics; use session/project/person scopes with explicit coverage.

**Acceptance:** Authorized summaries are available; unjoined callers get none. Time/source-filtered recall cannot quote a reading with unsupported outside-window content. Invalid newest summaries do not resurrect misleading older ones.

#### M-025 — Active obligations are not supplied by ordinary provider recall — P1, Verified

**Evidence:** Broker supports a commitments section, but provider `_recall`/`prefetch` calls supply lessons, not commitments. `GoalStore` is used for goal proposal, and the maintenance/delivery path handles reminders separately.

**Impact:** A question such as “what should I work on now?” need not include active commitments even though the prospective layer knows them. Notification delivery and contextual remembering are separate capabilities.

**Recommendation:** Retrieve a small authorized, time-valid set of relevant active goals/conditions into ordinary context. Include status, due precision, cancellation, and supporting evidence. Keep candidates separate from obligations.

**Acceptance:** Hermes can recall the owner's relevant active commitments, ignores cancelled/superseded ones, and does not turn a candidate or its own suggestion into an owner promise.

#### M-026 — Typed measurements deserve a specialized recall route — P2, Verified capability + Opportunity

**Evidence:** `storage/measurements.py` computes bounded series statistics, unit conflicts, intervals, and gaps deterministically. The ordinary provider recall path returns evidence/facts and does not invoke this specialized plane.

**Impact:** Numeric questions can reach prose recall instead of precise computations, or encourage expensive formation of each sample.

**Recommendation:** Route measurement questions to typed aggregates and evidence IDs; use the model to explain, not calculate unsupported values. Keep missing samples, mixed units, and unplaced time explicit. Do not infer health diagnoses from trends.

**Acceptance:** Counts/means/gaps are reproducible and scoped; wrong units and missing data are reported. Narrative summaries cite the actual selected series/window.

#### M-027 — Procedural learning must use verified outcomes, not memory popularity — P1, Opportunity

**Evidence:** `learning/lessons.py`, `outcomes.py`, and `evaluation.py` already distinguish candidate practice, prerequisites/exceptions, owner decisions, and versioned evaluation evidence.

**Recommendation:** Connect task execution receipts and explicit owner corrections to that existing ledger. Capture what was tried, under which conditions, what actually happened, and what changed. Successful retrieval/use can influence ranking, but must not strengthen a claim's truth merely because Hermes repeated it.

**Acceptance:** Unsupported assistant success claims cannot activate lessons. Lessons are retrieved only for matching conditions, and become stale when their model/tool/evaluation identity changes.

#### M-028 — Clarification should be a shared Hermes-visible episode — P1, Existing architecture + Opportunity

**Evidence:** Existing `proactive/inquiries.py`, provider turn-start settlement, delivery/outbox, and the previous [gateway clarification review](hermes-gateway-clarification-review.md) provide a durable owner-question path. This pass did not send a Telegram question or inspect a live reply.

**Recommendation:** Reuse that path for unresolved referents, dates, identity candidates, and conflicting claims. The dossier should include inquiry ID, authenticated owner/profile, question, candidate interpretations, source spans, pending decision, gateway destination/message correlation, expiry, and decision result. Hermes sees pending inquiry context before interpreting the user's reply and receives the outcome afterwards.

**Acceptance:** Native Telegram replies and explicit coded replies resolve the correct pending inquiry; “haan” is never applied to an unrelated question. A free-form ambiguous response remains a clarification, not destructive consent. Hermes can explain why it asked and what changed. Quoted source instructions, bots, and another profile cannot authorize settlement.

#### M-029 — Retrieval availability is not semantic answer support — P1, Verified heuristic + Opportunity

**Evidence:** `ContextBroker._coverage` primarily uses presence of raw items/facts, channel failures, conflicts, and truncation. It does not evaluate whether the evidence actually answers the question. Its `supported` label must not be interpreted as a truth or entailment guarantee.

**Impact:** A mention of the same subject can be delivered with a stronger apparent sufficiency label than its actual usefulness warrants. Conversely, typed-only answers need appropriate coverage reporting.

**Recommendation:** Separate channel availability, evidence completeness, provenance validity, and answerability. Use conservative labels and optionally a bounded support check for high-impact or conflicting questions. Do not fabricate calibrated confidence probabilities.

**Acceptance:** Matched-but-irrelevant text, missing qualifications, unknown answers, and contradictory evidence produce appropriate abstention/clarification rather than confident invention.

#### M-030 — More intelligence must not weaken forgetting and consent — P0, Existing fixes pending deployment + design constraint

**Evidence:** Source schema 15 and R-001–R-032 address erasure/recovery/cache/authority issues; inspected installed schema is still 14. Each new index, alias, episode, observation, graph edge, cache, and learned artifact creates another derivative that must obey the same lifecycle.

**Recommendation:** Use existing canonical IDs, dependency manifests, durable erasure fences, and owner decision infrastructure. Recency/utility decay may reduce ranking, not secretly delete owner-confirmed facts. Physical retention/backups need an explicit policy; a tombstone is not proof that every byte was destroyed.

**Acceptance:** Forget/revoke/correct during formation, consolidation, recall, cache reuse, and restore leaves no affected active derivative. Backup retention and remote deletion status are disclosed accurately. Profile isolation remains intact.

### E. Scale, multimodal coverage, and evidence-driven rollout

#### M-031 — Device serialization and task scheduling need separate policies — P2, Verified + Opportunity

**Evidence:** Routes distinguish remote text and local GPU resources, but worker composition enforces a single task slot. Gate priorities do not preempt running work. Actual per-bank queue fairness and combined memory/VRAM use were not measured.

**Impact:** A long task can occupy the sole worker while another task could use an independent resource. Conversely, casually raising worker slots could violate tested resource/budget contracts or increase RAM pressure.

**Recommendation:** Measure head-of-line wait, per-stage resource use, and resident memory. Consider bounded staged work that overlaps independent resources while retaining one admission slot per constrained device. Treat service memory limits as ceilings, not real consumption or guaranteed combined capacity.

**Acceptance:** Foreground performance and small-bank freshness remain acceptable under a bulk import; two models never exceed the same device's approved occupancy. Prove budget attribution and shutdown/cancellation under overlap before changing concurrency.

#### M-032 — Several local lookups need large-archive measurements — P2, Verified paths + Unmeasured bottleneck

**Evidence:** Assertions use bounded query terms but a substring `LIKE` scan over potential matches; provenance/authorization can perform per-reference lookups. Formation repeatedly selects/counts unprojected evidence. Current live archive is small.

**Recommendation:** Benchmark synthetic 1k/10k/100k archives with realistic metadata and graph fanout. Use query plans, batched authorization/provenance resolution, and targeted indexes only where needed. Cap processing work as well as response size. Preserve transactional authority checks and avoid broad full-bank scans on each turn.

**Acceptance:** p95/p99, examined rows, allocations, and query count remain bounded. Query-plan improvements do not change visibility, expiry, identity, or correction semantics.

#### M-033 — Attachment storage is not complete multimodal memory — P2, Capability gap requiring further tracing

**Evidence:** Blob storage, source/media ingestion branches, and a governed vision route exist; generic formation sends text. Native Hindsight OCR is disabled to prevent an ungated client path. End-to-end attachment interpretation was not exercised.

**Recommendation:** Trace which source attachments actually produce searchable canonical descriptions/transcripts. Add only necessary, approved extraction through the gate, with attachment hash/source IDs, language, timestamps, and uncertainty. A caption/transcript must be identifiable as derived, not the original media. Do not enable bypass OCR as a shortcut.

**Acceptance:** Image/audio/document evidence can be retrieved through a supported description or transcript; missing extraction is reported. Forgetting reaches bytes, descriptions, embeddings, and dependents; OCR/transcription errors are not promoted to facts without qualification.

#### M-034 — Existing green offline tests do not establish memory intelligence — P0, Verified measurement gap

**Evidence:** `docs/quality.md` and `evals/cases/checks.py` explicitly mark Hindsight/raw-fact/observation/model quality ablations as not measured without a backend. Local synthetic latency is mainly lexical/cache behavior. The load harness exercises synthetic durability/concurrency without model inference.

**Impact:** Passing core safety tests is necessary but not a Hinglish accuracy or contended recall-speed result. The previous 3,041-pass source test receipt remains useful; no full suite was rerun in this documentation-only pass.

**Recommendation:** Extend evaluation in a separate disposable profile/bank, not the owner's archive. Add bilingual, temporal, provenance, graph, task-context, and end-to-end packed-answer cases, plus model contention and cost measurements.

**Acceptance:** Publish denominators, language slices, failure examples, model/config fingerprints, cold/warm conditions, and unavailable metrics. No release gets a quality claim based on an unmeasured row.

## 6. Recommended target behavior: a useful personal assistant

### Formation: remember meaning without remembering every output as truth

1. Capture approved evidence durably and promptly, with role/trust/time/context preserved.
2. Make explicit remembers and current-turn evidence immediately accessible locally; show whether semantic formation is still pending.
3. Under a standing owner allowance, incrementally extract high-value claims/episodes, leaving difficult references uncertain.
4. Preserve per-record support even when processing a conversation window. Select useful evidence, not arbitrary character prefixes.
5. Create tentative typed assertions and links with exact support. Treat guesses, quotations, and model reports differently from direct statements and verified actions.
6. Consolidate changed evidence into cautious observations after the integration/provenance issues are solved. Refresh only affected abstractions.
7. Ask the owner through Hermes when resolving ambiguity materially changes a commitment, identity, preference, or destructive decision.

A fast episodic dense path before full extraction could improve cross-language recall of new evidence. This is a design option, not a requirement to add a second vector database. First check whether supported Hindsight chunk/verbatim modes can serve it without duplicating evidence or authority systems. Any alternative needs the same provenance, indexing version, forgetting, and scope contracts.

### Recall: three explicit service levels

These are proposed targets for later measurement, not current achieved performance.

| Mode | Intended behavior | Initial evaluation target |
| --- | --- | --- |
| Fast | Local scoped claims/goals/cache/exact evidence; low-depth semantic lookup if needed; no reflection | Added memory p95 ≤1 second when models are warm and the required resources are available; explicit fallback under contention |
| Balanced | Dense + keyword + useful native associations; bounded provenance; optional small rerank | Added memory p95 ≤2 seconds on the defined warm benchmark; compare useful evidence delivered to Hermes |
| Deep, explicit | Higher search depth, limited graph hops, expanded support and carefully budgeted synthesis | Separate user-visible deep request; measured ceiling and cancellable work, never hidden in every ordinary turn |

The current six-second foreground deadline should be a safety limit, not the default desired experience. Cold model loads, CPU-only fallback, unavailable inference, and saturation need separate distributions and honest user-facing outcomes.

The latency budget must include: capture freshness, gate wait, query embedding, DB/vector/keyword search, graph traversal, reranking, support fetch, local authorization, packet packing, and transport/serialization. Report per-request totals; do not sum unrelated p95 values as if they form a real p95.

### Connections: relevant, typed, and revisable

Useful personal links include person↔project, project↔decision, decision↔task, task↔outcome, event↔time/place, and preference↔conditions. Track validity and support. Distinguish “similar topic,” “same declared person,” “happened near the same time,” “contradicts,” “supersedes,” and “caused by”; these are not interchangeable edges.

A similarity edge should help find evidence, not let Hermes assert a relationship. High-degree generic nodes such as “user” or “work” need fanout caps/downweighting. Query-time graph expansion should be small and relevant, with access checks on every supporting branch. Existing Hindsight association machinery should be evaluated before creating a second graph.

### Working memory and gateway awareness

Hermes remains the conversational agent. The memory framework supplies bounded evidence and manages durable memory state; it should not become an independent assistant that talks to the owner without Hermes knowing.

For each inquiry, Hermes should receive the context dossier and the resulting decision event through its normal gateway/turn integration. The persistent system-prompt block remains static: the Hermes repository explicitly requires a byte-stable conversation prefix. Relevant memories and inquiry state belong in the existing per-turn context mechanism, not in a rebuilt system prompt or synthetic user message inserted mid-loop.

An owner's answer can update canonical evidence and approved decisions; it must not indiscriminately erase earlier evidence. Sensitive actions require authenticated, explicit consent to the exact pending decision. Non-sensitive clarifications can capture a new attributable statement and leave any consequent promotion to the appropriate policy.

### Forgetting and graceful uncertainty

Keep history useful without keeping stale claims authoritative. Stable, explicit preferences should not decay merely because they were not recently recalled. Recent temporary states should not override durable preferences forever. Use relevance/recency/utility for retrieval, separate from confidence, validity, confirmation, and retention.

When memory cannot answer, Hermes should know whether the reason is no evidence, unformed evidence, access withheld, ambiguity, contradictory accounts, incomplete provenance, timeout, or an unavailable stage. Those distinctions are more useful than an artificial high-confidence guess.

## 7. Evaluation plan before any optimization claim

No live execution of this plan was authorized/performed in this pass.

### Dataset and isolation

- Start with invented episodes spanning preferences, people/projects, task results, conditions, commitments, corrections, time ambiguity, quote attribution, and forgetting.
- Include at least 240 held-out recall questions across language/script combinations, with explicit source IDs and required spans. Expand if important categories have small denominators.
- If the owner later approves real samples, anonymize and label them privately. Do not log full conversations, credentials, prompts, or retrieved personal content in aggregate telemetry.
- Use a disposable profile and separate bank/index. Never contaminate the owner's memory with benchmark facts.
- Split by episode/person/template family, not merely paraphrase, so near-duplicates cannot leak into both tuning and evaluation.

### Measurements

| Layer | Metrics / checks |
| --- | --- |
| Capture | Durable capture success; duplicate canonical revision rate; p50/p95 capture→local visibility lag |
| Formation | Supported-claim precision/recall; speaker/negation/conditionality/date/entity accuracy; unresolved-reference rate; facts per useful record |
| Retrieval before packing | Recall@5/@10, MRR, nDCG; correct current/historical version; same-name-person errors; one/two-hop evidence recovery |
| Retrieval after packing | Required evidence actually delivered; qualifier/negation survival; canonical support completeness; duplication/diversity |
| Assistant answer | Evidence-supported answer rate; unknown-answer abstention; contradiction clarification; no fabricated action completion |
| Performance | p50/p95/p99 added latency by mode/language; cold/warm/resource-contended distributions; gate wait vs inference vs search |
| Throughput/cost | Useful records/sec; tokens per supported fact; consolidation cost per changed episode; background freshness by bank |
| Reliability | Queue bounds, timeout aftermath, retries/reconciliation, cancellation, model restart, unavailable rerank/backend, write/recall races |
| Safety | Zero unauthorized cross-account/profile disclosures in the test matrix; zero known erased support re-served; owner consent maintained |
| Proactivity | Precision, missed useful reminders, false interruptions, deduplication, quiet-hours/opt-out behavior, Hermes-visible inquiry context |

Use language-balanced reporting rather than one average hiding weak Hinglish performance. Date/reference/negation safety cases should be reviewed individually even if an aggregate score looks good. Zero failures in a finite safety suite is a release condition, not a proof of zero possible future failures.

Initial quality gates can start at required-evidence Recall@5 ≥90% and supported-claim precision ≥95% on the labeled corpus, with no known critical speaker/negation/identity/erasure failures. These are proposed project acceptance targets, not published model guarantees. Revise them using denominators, task risk, and baseline difficulty; do not relax authorization to obtain a higher recall score.

### Controlled ablations

Change one factor at a time:

1. Current baseline: literal lexical + existing Hindsight/RRF + current packet behavior.
2. Correct provenance/attribution/time contracts before comparing intelligence features.
3. Query instruction off/on, after verifying server templates.
4. Literal vs context-aware/alias-assisted query retrieval.
5. RRF alone vs bounded multilingual reranking.
6. Raw facts vs supported observations; graph on/off for multi-hop cases.
7. Current packet order vs lineage-deduplicated adaptive packing.
8. Current individual formation vs safe micro-batching.
9. Full-dimensional baseline vs supported reduced dimensions, only if storage/search cost warrants it and with separately indexed vectors.
10. Normal vs cold/saturated resources and foreground/background competition.

Keep embeddings/extractor/reranker model revision, quantization, tokenizer, endpoint, prefixes, dimensions, chunk policy, prompts, bank settings, and framework/backend digest in every receipt. A dimension reduction is an index-space migration, not a knob to turn on an existing populated index without revalidation.

## 8. Recommended implementation order for later

| Stage | Work | Exit condition |
| --- | --- | --- |
| 0. Establish safe baseline | Review/deploy the earlier source fixes only when the owner resumes installation; verify source/runtime/schema alignment in a rehearsal | Existing safety suite plus isolated migration/restore checks pass; production change separately approved |
| 1. Fix integration contracts | M-001–M-005 and M-024: task policies, observation support, scoped reflection, role/context/time, caller-aware summaries | Pinned real payloads work in an isolated integration test; unsafe support is withheld |
| 2. Establish multilingual quality | M-006–M-008, M-014–M-016, M-020, M-029, M-034 | Baseline Hinglish/English quality and vector-contract receipts exist; gaps are quantifiable |
| 3. Make learning continuous but governed | M-009–M-012: approved producer, selective formation, controlled consolidation, upgrade generations | Measured bounded lag/cost; no unauthorized inference; corrected/erased evidence never returns |
| 4. Improve recall speed and usefulness | M-017–M-023, then M-031/M-032 where measurements justify them | Better packed evidence and p95 under contention, without safety regressions |
| 5. Add personal-assistant depth | M-013, M-025–M-028, M-033; retain M-030 as an invariant | Hermes recalls relevant commitments, supported connections, verified practice, and inquiry context |

Do not enable consolidation, change vector dimensions, enlarge all budgets, add a bigger reranker, or raise worker concurrency together. Otherwise quality regressions and cost/latency changes become hard to attribute. A tested core contract change is preferable to editing installed package files, runtime monkeypatches, or duplicated fallback databases.

## 9. Evidence index and outstanding unknowns

### Stable source anchors

All framework paths below are relative to the project root; installed Hindsight paths are relative to the package directory given in section 2.

| Area | Anchors |
| --- | --- |
| Capture / event interpretation | `integrations/hermes-memory/provider.py:sync_turn`, `on_turn_start`, `prefetch`, `_broker`, `_derived_client`; `capture.py:transcript`; `sources/sdk.py:_frame`, `_stamp`, `_turn`, `_arrival_of` |
| Projection | `processing/formation.py:processor_fingerprint` (69), `_SELECT`, `_unprojected_rows`; `processing/worker.py:run_once` (retain payload near 252); `backend/document_map.py:begin` |
| Hindsight client / provenance | `backend/hindsight_client.py:recall`, `reflect`, `RecallOutcome.from_body`; `backend/provenance.py:resolve`; `backend/worker_launcher.py:attribute_tasks`, `_resource_for` (793) |
| Recall / packet | `context/broker.py:assemble`, `_derive` (254), `_recall_within_deadline`, `_fact_allowed` (384), `_summaries` (416), `_coverage` (332); `context/cache.py`; `context/packet.py:render` |
| Lexical / typed claims | `storage/evidence.py:fts_terms`, `fts_query`, `search`; `knowledge/assertions.py:matching` (295), confirmation/supersession; `storage/measurements.py` |
| Summaries | `processing/summarization.py:summarize_apply` (264), `_reflect`, `_question` (419); `knowledge/summaries.py:publish`, `read`, `latest` |
| Gates / services | `processing/routes.py:build_routes`; `gate_server.py:wait_for`; `processing/maintenance.py:pass_now`, `_summaries`; `service.py`; `config.py` |
| Assistant behavior / lifecycle | `prospective/goals.py`; `proactive/inquiries.py`, `outbox.py`, `delivery.py`; `learning/lessons.py`, `outcomes.py`, `evaluation.py`; `storage/lineage.py`; `lifecycle/`; migration `0015_durable_fences` |
| Installed extraction / links | `engine/retain/fact_extraction.py:_DEFAULT_LANGUAGE_RULE`, `_infer_temporal_date`, `_iter_conversation_chunks`; `orchestrator.py`, `entity_processing.py`, `link_creation.py` |
| Installed recall / scheduling | `engine/embeddings.py:Embeddings.encode_query`, `OpenAIEmbeddings`; `engine/memory_engine.py:recall_async`, recall result construction near 9770, `submit_async_consolidation`, `submit_async_graph_maintenance`, mental-model refresh submission; `worker/poller.py` payload injection near 711 |
| Evaluation | `docs/quality.md`; `evals/cases/checks.py`; `evals/run_load.py`; `tests/review/test_feature_review.py` |

### Checks actually performed in this pass

- Read-only systemd state check: three memory services active/running.
- Read-only aggregate SQLite checks: installed schema, record/visibility counts, occurrence-time coverage, projection states, and backend operation kind/state counts.
- Synthetic route-policy check: `retain` accepted; `consolidation`, `refresh_mental_model`, `graph_maintenance` refused by the current resource mapper.
- Synthetic observation provenance check: supported observation supplied but withheld by current broker integration.
- Synthetic scoped-summary checks: default ledger/read path withheld account-scoped content; explicit identity-aware account read succeeded, while caller-less `latest` still omitted it.
- In-memory Unicode/FTS check: verified Devanagari token fragmentation; exact synthetic sentence still matched.
- No model inference, production writes, Telegram sends, service changes, installation, or full regression suite execution.

### Unknowns that must not be presented as findings of measured failure

- Actual Hinglish extraction/retrieval/answer quality of the configured quantized 9B model and served Qwen model.
- Server-side Qwen prompt template, pooling/normalization, effective maximum input size, truncation, and vector quantization/revision.
- Effective per-bank Hindsight settings, graph density, association quality, and which optional link modes are enabled.
- Actual foreground p95/p99, cold starts, GPU/CPU memory pressure, and model contention.
- Whether an external, uninspected scheduler already invokes formation under a grant.
- Complete live Telegram native-reply/inquiry routing, multilingual free-form settlement, and cross-profile behavior.
- The provider's authenticated caller-account binding for within-bank authorization, particularly in group chats; profile isolation alone does not establish that contract.
- Full media/transcription/OCR recall lifecycle and exact backup/remote erasure completion.
- Whether a learned reranker, reduced dimensions, a new dense episodic stage, or additional concurrency improves this workload enough to justify cost/complexity.

The next useful result is a reproducible bilingual baseline with correct formation/provenance/time contracts—not a promise that a model upgrade alone will give Hermes perfect memory.
