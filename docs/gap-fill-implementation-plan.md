# Gap-fill implementation plan: completing the advertised integration

Date: 2026-10-02. Follows the capability audit of this tree (clone at
`~/scratch/hermes-memory-audit`). Every work item below names the exact files,
insertion points and existing conventions it builds on, so an implementer can
start without re-deriving the design. Line references are to this revision
(`ebdc972`); they will drift, the symbols will not.

**Scope honesty.** This plan closes the gaps the audit found between the
advertised layer table and the code: recall ranking/reranking, measurements
definitions/conversions/aggregates, an independent verification hop, chunked
consolidation, knowledge relationship/preferences/episodes, a real procedures
subsystem, learning replay/recovery, task dependencies, and encrypted backup.
It does not add HNSW locally (a pinned-backend concern — see 1.6), does not
loosen any fail-closed behavior, and does not change the claim: no zero
hallucination is promised anywhere.

## Governing conventions (inherited, not invented)

Every phase must obey the rules the codebase already enforces:

1. **Fail closed, no silent fallback.** Missing reranker, missing verifier
   route, missing conversion: refuse or degrade to an explicit
   `unavailable` channel, never to a guess.
2. **Plan → review digest → apply** for anything that mutates shared state
   (`formation.py:260,309-315` is the reference pattern).
3. **Owner-only mutations** through `owner_principal`
   (`config.py:226,399`), enforced in-store, never at the door.
4. **Immutable, revisioned, append-only rows.** Corrections are new
   revisions; the only mutation is a tombstone (`lifecycle/erasure.py:186`).
5. **Migrations are appended, never rewritten** (`MIGRATIONS`,
   `storage/migrations.py:930-1027`), and cache-invalidating tables are added
   to the `context_revision` trigger list (`migrations.py:989-995`).
6. **Bounded everything**: traversal caps (`CLOSURE_CAP=2000`), sample caps
   (`MAX_SAMPLES_PER_READ=20_000`), token caps, time caps. Every new loop
   states its cap and reports truncation.
7. **Deterministic checks beside model verdicts** (`faithfulness.py:38-64`).
8. **Tests inject transports**; the autouse `no_unstarted_socket` guard
   (`tests/conftest.py:30-67`) fails any real network attempt. Use the
   `store` fixture + `envelope()` factory (`conftest.py:70-89`) and fake
   clocks, per `jobs.py:86` / `due_events.py:91`.

---

## Phase 1 — Recall honesty (advertised but missing)

Goal: make the Recall row of the claim table true. The lexical channel is
FTS5 bm25 (`context/lexical.py:39-48` → `EvidenceStore.search`,
`storage/evidence.py:349-372`); the semantic channel is the Hindsight client;
`EvidenceItem.score` is hardcoded `1.0` (`context/broker.py:48-54`) and is not
even serialized (`context/packet.py:58-65`). Ranking today is FTS order.

### 1.1 Wire the cross-encoder reranker into `assemble` (highest value)

The service already exists and is accounted: `Qwen/Qwen3-Reranker-0.6B`,
score = softmax over last-token `[no, yes]` logits → P(yes)
(`models/reranker.py:197-204`), served at `127.0.0.1:8185` with a native
`/rerank` and Cohere-shaped `/v1/rerank` (`reranker.py:267-287`), caps: 128
texts/call, 2048 doc tokens (doc-tail truncation), 512 query tokens
(`reranker.py:155-171`). The gate already forwards `POST /v1/rerank` by
credential (`processing/gate_server.py:36,131-140,233-244`) and the route is
declared (`processing/routes.py:151-153`); **no framework code calls it today.**

Implementation:

- New module `src/hermes_memory/context/rerank.py`:
  - `RerankClient` with the two methods the codebase's clients always have:
    `health()` and `rerank(query, texts, top_n) -> list[Reranked(score)]`.
    It calls **through the owned gate** (`by_credential`), never the model
    port directly, mirroring how every other model call is routed.
  - Construction follows `formation.py:67`:
    `build_routes(settings, credentials=settings.route_credentials).by_name("rerank")`
    — absent route ⇒ client is `None` ⇒ channel reports `unavailable`, never
    an error.
- `ContextBroker.__init__` grows an optional `reranker=None` dependency
  (same injection style as `client`).
- Insertion point in `assemble`: between the authorization/filter producing
  `considered` (`broker.py:128-131`) and the budgeted selection loop
  (`broker.py:154-171`). Mechanics copied from the derived channel's
  deadline pattern (`_derive`/`_recall_within_deadline`,
  `broker.py:257-299`): bounded timeout (start at 3.0s, max 30s), thread
  pool reuse, and a channel status.
- Caps: rerank at most 64 candidates (matches the Hindsight tier wiring,
  `install/reranking.py:77-83`), only when `len(considered) > limit`, only
  when a route is configured and healthy.
- `Packet.channels` grows `rerank` in {"available","unavailable","skipped"}
  following `Channels.as_dict`'s existing discipline (`packet.py`).
- Budget: the call is admitted and charged as resource `local-gpu` through
  the instance gate — same shape as `verification.py:57`'s
  `Budgets(...).admit(...)` pre-dispatch estimate, and the gate's measured
  charge on release.
- Failure semantics: timeout/error/unhealthy ⇒ keep FTS order, set
  `rerank="unavailable"`, log to the gate ledger like any refused dispatch.
  This is degradation with a receipt, not a fallback.

Tests (`tests/unit/test_rerank.py`): stub transport asserting the Cohere
request shape; caps (64); timeout path; no-route path; selection order
follows returned scores; channel statuses; admission refusal surfaces as
`unavailable`.

### 1.2 Real scoring: rerank + recency + citation support

- Replace `score=1.0` in `_item()` (`broker.py:48-54`) with a deterministic
  blend, all components recorded on the item for `explain`:

  ```
  score = 0.7·P(yes)                    # from 1.1, else bm25-derived lexical rank score
        + 0.2·recency_decay(observed_at)  # e.g. exp(-age_days/180), computed at read
        + 0.1·min(support, 5)/5           # citation count via lineage.citations_of,
                                          # capped, batched into one bounded query
  ```

- No model in the scorer. When reranking is unavailable the lexical rank
  score substitutes for the P(yes) term and the packet says so.
- Emit `score` (and the three components) in `EvidenceItem.as_dict`
  (`packet.py:58-65`) and through `operations/explanations.py`, so "why this
  item ranked here" is answerable — the codebase's own `explain` standard.
- `citations_of` (`storage/lineage.py:164-185`) is per-artifact; add a
  batched `support_counts(ids, cap)` helper in `lineage.py` that stays
  inside one bounded `IN`-chunked query.

### 1.3 Surface the temporal window

`assemble(window=…)` and `_in_window` already exist (`broker.py`) but no
caller passes a window. Thread it:

- `integrations/hermes-memory/provider.py` recall path: accept an optional
  ISO interval from the tool schema, validate strictly (inverted ⇒ refuse).
- CLI `explain`/`status`: `--window since until`.
- The cache scope already keys on the window (`broker._scope`,
  `broker.py:522-539`) — no cache change needed.

### 1.4 Duplicate suppression inside a packet

After selection (`broker.py:154-171`): bounded pairwise shingle-Jaccard on
candidate texts — cap comparisons (≤200 pairs), drop the lower-scored item
when similarity > 0.9, record dropped ids in the packet (`deduped: [...]`)
so suppression is visible, not silent. Keep superseded-revision suppression
(where it already exists in the FTS layer) untouched.

### 1.5 Progressive rounds (bounded, explicit)

`assemble` gains exactly two rounds: round 2 fires only when round 1
under-fills (`len(items) < limit`) or every score is below threshold; it
widens the lexical limit (×2) and relaxes the window. Cache scope gains a
`rounds` component. Channel report includes `rounds: 2`. No unbounded
re-query loops — the codebase refuses open-ended work everywhere else and
this is no exception.

### 1.6 HNSW — decided: backend concern, not framework code

Vector indexing belongs to the pinned Hindsight backend (0.10.1); the gate
already proxies `/v1/embeddings` (`gate_server.py:36`). Action item: update
the claim table/docs to state this honestly, and track Hindsight's vector
index roadmap. Building a parallel local HNSW would duplicate state the
backend already owns and violate single-source-of-truth.

**Phase 1 acceptance:** full-flow eval passes with a live rerank step and a
new receipt showing `channels.rerank="available"` and non-1.0 scores;
withhold-order test proves FTS order survives reranker outage; `explain`
prints score components.

---

## Phase 2 — Measurements layer

Current state: samples are `records` rows of `kind='measurement'` with
`metadata` JSON `{measure, device, unit, value, quality}`
(`storage/measurements.py:27`); `available()` is a `GROUP BY` census
(`:78-101`); `series()` refuses unit mismatch (`:206-214`) and caps reads
(`MAX_SAMPLES_PER_READ=20_000`, `:200-205`); `summarise()` is Python
`statistics` (`:37-65`). No definitions, no conversions.

### 2.1 Migration `0018_measurements`

Append `Migration("0018_measurements")` (`migrations.py:930-1027`) with:

```sql
CREATE TABLE metric_definitions(
  id TEXT PRIMARY KEY,
  key TEXT NOT NULL UNIQUE,          -- canonical measure key, e.g. "weight"
  canonical_unit TEXT NOT NULL,      -- e.g. "kg"
  aliases_json TEXT NOT NULL,        -- immutable JSON array of alias units/keys
  description TEXT NOT NULL,
  created_by TEXT NOT NULL,          -- owner principal, enforced in-store
  created_at TEXT NOT NULL
);
CREATE TABLE unit_conversions(
  from_unit TEXT NOT NULL,
  to_unit TEXT NOT NULL,
  factor REAL NOT NULL,
  offset REAL NOT NULL DEFAULT 0.0,
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(from_unit, to_unit)
);
```

Both immutable (no UPDATE path at all, like `ingestion_receipts`). Both are
cache-invalidating: add to the `context_revision` trigger list
(`migrations.py:989-995`) because a new conversion changes what a packet may
contain.

### 2.2 Store API + read path

- `Measurements.define(...)` / `Measurements.conversion(...)` — owner-only
  writes (owner principal checked in-store, `allowance.py`-style), rejecting
  duplicate keys, non-positive factors, and self-conversions.
- `series()` (`measurements.py:103-227`): when a sample's unit differs from
  the requested unit, convert **only** when a registered chain exists
  (direct edge; no path search — fail closed), then proceed. Mismatched
  without a conversion keeps today's refusal, byte-for-byte.
- Conversion applied at read time only; stored rows are never rewritten
  (immutability rule).
- New `aggregate(measure, since, until, bucket)` (`"hour"|"day"|"week"|"month"`):
  SQL `count/min/max/avg/sum` with `GROUP BY strftime(bucket, occurred_at)`,
  result rows capped at 10_000 with a `truncated` flag (the census pattern,
  `available()`). Median/stdev remain Python `statistics` and are labeled
  `computed_at_read: true` in the report — never presented as SQL facts.

### 2.3 CLI

`hermes-memory measure define|conversion|aggregate` — define/conversion are
plan/review-digest/apply, owner-only (`formation.py:260,309-315` pattern);
aggregate is read-only. Validation error messages follow the house style:
say what was refused and why, and what to do next.

**Phase 2 acceptance:** tests for definition immutability, owner gating,
direct conversion success, refusal without conversion, no chained
conversions, aggregate caps and truncation flags; existing measurement tests
pass unchanged (backward compatibility of the refusal semantics).

---

## Phase 3 — Trust upgrades (Consolidation)

### 3.1 Independent verifier route (generate ≠ verify)

Today `ScopedSynthesizer` runs both hops — generation
(`memory_scoped_claims`) and entailment verification
(`memory_entailment_verdicts`) — on the same loopback admission URL, same
credential, same `self.model` (`processing/synthesis.py:68-125`; callers
differ only by route credential: `reflect` for summarization
`summarization.py:389-396` vs `foreground` for verification
`verification.py:47-66`). This is the correlated-error hole the
faithfulness doc admits.

Implementation:

- New env: `HERMES_MEMORY_VERIFIER_BASE_URL / _MODEL / _RESOURCE` parsed in
  `config.py` (`_route`, `config.py:466-477` — loopback/LAN allowlist rules
  apply automatically), credential
  `HERMES_MEMORY_ROUTE_CREDENTIAL_VERIFIER`.
- `build_routes` (`processing/routes.py:137-154`) grows a `verifier` route
  alongside the text routes (priority `interactive`).
- `ScopedSynthesizer(model=…, verifier=…)` — hop 2 uses the verifier route.
  **When a verifier route is configured, verification on the generation
  route is refused outright** (fail closed, no accidental correlation).
  When it is not configured, behavior is exactly today's, and reports keep
  saying `verifier: generation route` so nobody mistakes it for independent.
- `summary_fingerprint` (`summarization.py:64-69`) includes the verifier
  identity (model+route digest) so a summary's checking lineage is visible
  and re-verification with a different verifier is a new fact, not a silent
  reuse.
- Two-hop token accounting already covers both hops
  (`summarization.py` changed-plan/fingerprint contract) — verifier tokens
  land under the verifier resource in the gate ledger naturally.

Tests: same-model refusal when verifier configured; two-fake-transport
synthesizer asserting hop separation; fingerprint changes with verifier
identity; absent-route backward compatibility.

### 3.2 Whole-source chunk partitioning for summarization

`summarize_plan` currently selects `limit+1` to detect truncation and then
**blocks** with "raise --limit or narrow the scope, do not summarize a
prefix and call it the scope" (`summarization.py:203-225`), plus a byte
ceiling (`MAX_INPUT_BYTES`, `:212-213`). The honesty rule is right; the
blocking is the gap.

Implementation (plan-side only; formation stays one-job-per-record — the
`input_revision` invariant at `formation.py:154-158` is correct for
raw-facts and must not be chunked):

- `summarize_plan` proposes `N = ceil(scope / MAX_BATCH)` batches
  (`MAX_BATCH = MAX_CITATIONS_PER_SUMMARY`, `summarization.py:43`), each
  byte-bounded by `MAX_INPUT_BYTES`, each batch a separate job with stable
  key (`jobs.py:111-115` identity: kind, route, sorted inputs, revision,
  fingerprint, epoch) and `coverage="batch i of N"` recorded.
- A roll-up summary is produced **only when every batch completed**; its
  kind is `episode` (see 4.3) and its inputs are the batch summary ids.
  If any batch failed/withheld, the scope's coverage is `"partial"` and the
  report says which batches — never `full` from a prefix.
- Review digest covers the whole batch plan, so the owner approves one
  coherent partition, not batch-by-batch drift.

### 3.3 Local extractive adapter (explicit processor, not a fallback)

- A deterministic, model-free summarizer: sentence-rank over canonical
  quotes (frequency-based scoring over exact record text), output strictly
  verbatim quotes. Registered as `processor="extractive-local"` chosen at
  plan time.
- Because output is verbatim quote spans, publication validation is
  byte-exact membership only (`faithfulness.normalize_claims:96-129`) —
  entailment is not needed and is not run. No model ⇒ no gate charge, but
  the job still runs leased/quarantinable through the existing queue.
- Never selected implicitly: absent configuration keeps refusing scopes that
  need a model, exactly as today (`formation.py:140-142`).

**Phase 3 acceptance:** verifier-independence receipt in the live full-flow
run; chunked summarize of a >limit scope completes with all batches + roll-up
or reports `partial` naming failures; extractive path produces verbatim
quotes and passes byte-membership; no test regressions.

---

## Phase 4 — Knowledge completeness

### 4.1 Contextual preferences

- `kind='preference'` assertions gain an `applicability` JSON column
  (`0018` migration; nullable, default NULL = always applicable).
- Reuse the lessons rule engine wholesale: same operators
  `("eq","in","any","all")` and nesting (`lessons.py:52-53,413-433`,
  `_validate_rule :457-488`) — extract the validator into a shared
  `knowledge/rules.py` consumed by both, so preferences and lessons cannot
  drift apart.
- Broker read path: preferences outside applicability are filtered before
  packet assembly, and the packet records how many were filtered
  (visible, not silent).

### 4.2 Dated typed relationships

- Append-only `relationship_edges(id, record_id, related_id, kind,
  valid_from, valid_until, evidence_id, created_by, created_at)` in `0018`;
  `UNIQUE(record_id, related_id, kind, valid_from)`; evidence_id must
  resolve to a live record (FK-style check in-store).
- Exposed through `storage/lineage.py` next to `citations_of`, with the same
  bounded traversal (`_bounded`, caps 1..500); feeds the 1.2 support score
  (edges count as support, capped at 5 like citations).
- Broker may traverse exactly one hop at assembly time (bounded, cached in
  the packet's provenance section) — no deep graph walking in recall.

### 4.3 Episode summaries

- Add `"episode"` to `summaries.KINDS` (`knowledge/summaries.py:27`) with
  member record ids stored on the summary row (`0018` column, JSON, capped
  list).
- Produced by the 3.2 roll-up flow; same staleness/refresh ledger and
  provenance resolution as other kinds (`summaries.py:261-370`,
  `backend/provenance.py:155`). Owner approval requirements unchanged
  (`mental_model` keeps its owner gate, `summaries.py:113`).

---

## Phase 5 — Procedures subsystem (the genuine missing layer)

The audit's finding: "Procedures" is currently the inference gate + job
queue wearing a name. There is no binding of an evaluated lesson to an
installed capability, and no executor. This phase builds it for real.

New package `src/hermes_memory/procedures/`.

### 5.1 Binding (the administrator act)

- Migration (rides `0019_procedures`): `procedure_bindings(
  id TEXT PRIMARY KEY, lesson_id TEXT NOT NULL, lesson_version INT NOT NULL,
  capability TEXT NOT NULL, parameters_json TEXT NOT NULL,
  output_contract_json TEXT NOT NULL, bound_by TEXT NOT NULL, bound_at TEXT
  NOT NULL, state TEXT NOT NULL DEFAULT 'active')` — immutable while
  active; `state` moves active→retired only, owner-only, with reason.
- CLI `hermes-memory procedure bind --lesson <id>@<version> --capability
  <key> --parameters file.json --output-contract file.json` — plan prints
  the full binding, apply requires the printed review digest, actor must be
  the owner (`formation.py:260,309-315` pattern; owner enforcement
  in-store).
- Binding precondition (enforced, not advisory): the lesson@version must be
  **PASSED and current** — the same gate `activate_evaluation` uses
  (`lessons.py:182-209`, `evaluation.py:242-256`). Binding a non-evaluated
  lesson is refused.
- `capability` keys resolve against a new `capabilities()` registry
  (`procedures/registry.py`): installed executables declared with absolute
  path, allowed env names, and output schema. Declaring capabilities is an
  owner act too (plan/apply, like `install/reranking.py`'s reviewed env
  writes).

### 5.2 Execution (extending the proven runner)

`CommandRunner` (`learning/evaluator.py:47-98`) is the seam — it already
does absolute-path argv, no shell, scrubbed env (`ENV_ALLOWANCE` +
`HERMES_MEMORY_EVALUATOR_ENV`), timeout, bounded stdout, JSON outcome.
Extend rather than duplicate:

- `procedures/executor.py` runs a binding through CommandRunner with the
  binding's parameters, injecting credentials only via
  `scoped_secret` (`config.py:442-463`) — a binding names a credential
  scope; the executor resolves it at run time; the credential value never
  appears in job rows, logs, or outcomes.
- Jobs enqueue as `kind="procedure"` through the existing stable keys
  (`jobs.py:111-115`) — idempotent, leased (`claim`, `jobs.py:156`),
  renewable, quarantinable (`jobs.py:463`). Stable keys mean a retry of the
  same binding+params is the same job, not a duplicate execution.
- Stable job key must include the binding id and a parameters digest.

### 5.3 Output validation + learning loop

- The binding's `output_contract_json` is validated against the executor's
  parsed stdout (structure/types, bounded sizes). Invalid output ⇒ job
  `error` → quarantine path, never a partial success.
- Every run records an outcome via `outcomes.record`
  (`learning/outcomes.py:75`) with subject_kind `"procedure"`, so bound
  lessons accumulate support/against exactly like every other lesson
  source; retraction/promotion keep working unchanged
  (`lessons.py:165,182`).

### 5.4 Recall injection

Bound lessons already surface through `applicable()`
(`lessons.py:232-273` → `integrations/hermes-memory/provider.py:864-883`).
Bindings add the capability context to that injection (name + parameter
summary, never secrets), so the agent sees *how* to apply the lesson, not
only *that* it applies. The provider's task-vocabulary check
(`provider.py:1479-1499`) extends to procedure names.

**Phase 5 acceptance:** bind/promote gate tests; executor env-scrubbing test
(secret never in env dump or outcome); output-contract failure quarantines;
outcome tally drives lesson support; provider injection includes capability
context; an end-to-end fixture executes a trivial declared capability
(absolute-path script) start-to-outcome through the job queue.

---

## Phase 6 — Learning, Tasks, Ops gaps

### 6.1 Active-baseline replay

- `evaluation.py` already fixes baseline identity (`begin`, `:97`,
  fixture_digest; `evaluation.py:117` hashes the baseline lesson-set).
- Add `evaluation.replay(evaluation_id)` (read-only report): re-runs the
  recorded suite with the **stored baseline** lesson-set active instead of
  the current one, diffs per-case verdicts, emits
  `{case, was, now, delta}` with the divergence summary. Active lessons are
  never touched — replay is a measurement, not a mutation.

### 6.2 Lesson recovery

- Add `lessons`/`outcomes`/`lesson_versions` to `LEDGER_TABLES`
  (`lifecycle/recovery.py:36-37`) so restore carries them.
- Owner-only `hermes-memory lessons restore --lesson <id>` creates a **new
  version** whose status starts `candidate` and must pass a fresh
  evaluation before activation (`begin` already demands `candidate`,
  `evaluation.py:108`). Never resurrect in place — versioning rule.

### 6.3 Task DAG

- `0018` migration: `goal_dependencies(goal_id, depends_on_goal_id,
  created_at, PRIMARY KEY(goal_id, depends_on_goal_id))` (goals table has
  no dependency column today, `migrations.py:584-606`).
- Cycle safety: depth-capped DFS (cap 100) in `propose`/`revise`
  (`goals.py:94-152,211-261`) — refuse a cycle by naming the path found.
- `_waiting` (`predicates.py:186-198`) walks dependencies transitively with
  the same cap and reports `blocked_by` chains; `due_events._schedule`
  (`goals.py:358-371`) defers a dependent's events when a dependency
  settles or expires — deferral is a new due event, not a dropped one.
- One-level `waiting_for` remains valid and composes with `depends_on`.

### 6.4 Encrypted backup

Current: snapshot = one SQLite file via `db.backup`
(`lifecycle/snapshots.py:91`), sealed journal DELETE (`:278`), chmod 0600
(`:98`), sha256 manifest sidecar (`:103-110,298-300`); blobs live inside
the DB (`storage/blobs.py:1-8`) so the backup is db+manifest only. Restore
carries LEDGER_TABLES, guards, and re-buries tombstones
(`lifecycle/recovery.py:241-390`). `snapshots.py:96` states plainly:
plaintext.

Implementation, keeping the zero-dependency stance (no new Python deps):

- Encrypt via an external recipient-key tool (`age` preferred, `gpg`
  acceptable) invoked exactly like the delivery command is invoked today:
  absolute path, no shell, scrubbed environment (`config.py:259` documents
  the pattern; `evaluator.py:47-98` is the runner to copy).
- `snapshot create --encrypt <recipient>`: write the plaintext snapshot to
  a 0700 temp dir, stream through the encryptor to
  `canonical.db.age`, write the manifest with
  `encryption: {tool, recipient, plaintext_sha256}` (checksum still
  verifiable via decrypt), shred/remove the temp file, and only then
  publish the snapshot dir. Failure at any step leaves no snapshot rather
  than a half-encrypted one.
- `verify()`: with `age`/`gpg` available, decrypt to a temp copy and
  checksum; without it, report `encryption: unverifiable-here` (explicit,
  not an error masquerading as success).
- `restore` requires the recipient's key present at restore time — an
  owner decision made explicit in the CLI output.
- Default remains plaintext 0600 (behavior unchanged unless opted in), and
  the pre-restore guard dbs (`recovery.py:422-434`) follow the same
  opt-in.

### 6.5 Extension registry (minor)

- Replace the hardcoded `_export_readers` map (`cli.py:2367`) with a
  declarative table sourced from `SourceAdapter` declarations; validate
  source names against the registry at sync time (`sources/sync.py`
  `_check_name`) instead of accepting free text. Registry is code, not
  data — an extension is a module that declares itself.

---

## Sequencing and release discipline

Dependency graph:

```
Phase 1 (recall)  ──┐
Phase 2 (measurements) ──┴──> release A (migration 0018, one activation)
Phase 3 (trust)  ────────> release B (verifier route; re-run live full-flow + new receipts)
Phase 4 (knowledge)  ────> rides release B or C (0018 columns if A missed; else 0019)
Phase 5 (procedures) ────> release C (migration 0019, new package + CLI)
Phase 6 items       ────> independent; ship opportunistically with any release
```

Per release, the existing ritual is mandatory (it caught real failures
before):

1. `pytest` — full suite, zero regressions; every new module ships tests in
   the house style (`store` fixture, injected transports, fake clocks).
2. `hermes-memory compatibility --write` — regenerate
   `deployment/compatibility.json` for the new framework digest.
3. `hermes-memory release --source … --snapshot` (plan) → `--apply` with
   the printed review digest.
4. `hermes-memory activate-release …` (plan → `--apply --review <digest>`);
   activation backs up stores, rehearses migrations, flips the pointer,
   restarts services.
5. Re-run the live synthetic full-flow check
   (`evals/check_hindsight_reranked_flow.py --live --output …`) and
   `evals/check_memory_faithfulness.py --live`; keep receipts in
   `~/data/model-benchmarks/`.
6. Update the claim table and the docs' honest-scope notes so the
   advertisement and the code agree — the entire point of this plan.

## Risk register

| Risk | Mitigation |
| --- | --- |
| Rerank step adds latency to recall | 3s bounded timeout, only when `considered > limit`, channel degrades explicitly; measure in the live receipt |
| Conversion mistakes poison series | Conversions immutable + owner-only + read-time only; refusals without conversions unchanged |
| Verifier route misconfigured | Fail-closed refusal to verify on generation route; reports always name the verifier used |
| Chunked summaries claim false coverage | Roll-up only on all-batches-complete; `partial` coverage names failures |
| Procedures executes something unintended | Owner-gated binding + registry, absolute paths, scrubbed env, scoped secrets, output contracts, quarantine |
| DAG cycles or runaway traversal | Depth caps everywhere, cycles refused with the path named, deferrals are events not loops |
| Encrypted backup unreadable later | Recipient recorded in manifest; verify-with-decrypt; explicit `unverifiable-here` when tool absent |

## Effort estimate (order-of-magnitude)

| Phase | New/changed modules | New tests | Migrations |
| --- | --- | --- | --- |
| 1 | ~4 (rerank.py, broker, packet, provider) | ~6 files | — |
| 2 | ~2 + CLI | ~4 | 0018 |
| 3 | ~4 | ~5 | — |
| 4 | ~4 | ~4 | 0018 cols / 0019 |
| 5 | new package (4) + CLI | ~8 | 0019 |
| 6 | ~6 | ~6 | 0018/0019 |

Two migrations total (`0018_measurements_knowledge`,
`0019_procedures`), appended, never rewritten.
