# Tier 1 & Tier 2 implementation plan: agent-valuable memory improvements

Date: 2026-10-02. Companion to `gap-fill-implementation-plan.md` (the full
map) and the capability audit. This plan covers only the slice that
 measurably improves the Hermes agent's memory behavior, and deliberately
defers the rest.

Scope (7 features):

- **Tier 1** — F1 pre-send verification boundary; F2 verified agent-written
  memories; F3 independent verifier route; F4 temporal windows in recall;
  F5 in-packet duplicate suppression.
- **Tier 2** — F6 recall reranking with real scores; F7 chunked/episode
  summaries. Built only when corpus size demands them; specified now so
  the Tier 1 changes do not have to be redone.

All line references are to revision `1e74f84` in this tree.

## What exists today (the facts this builds on)

- Provider tools: nine, dispatched via `handle_tool_call` → `_dispatch`
  (`integrations/hermes-memory/provider.py:555,563`); `memory_remember`
  (:148) commits free prose **directly to canonical records**
  (`_remember` :935-959, `kind:"explicit_remember"`, no spool, no
  verification); `memory_verify` (:160) calls
  `verify_memory_claims(settings, store, claims, account_id=…)` in-process
  (:569-573).
- Verification service contract (`src/hermes_memory/processing/verification.py:16-75`):
  caller supplies 1-16 claims, each `{text, record_ids[1..32],
  evidence[{record_id, quote}]}`; the framework resolves records itself
  (live, visible, in the caller's identity scope), watermarks before/after,
  admits budget through the instance gate, then
  `ScopedSynthesizer.verify_claims` + `validate_publication`. Returns
  accepted text, per-claim labels, deterministic rejections, digests.
- Deterministic vetoes already exist: new numeric detail, unsupported
  severity, certainty strengthening (`faithfulness.py:38-64`); exact quote
  membership enforced in `normalize_claims` (:96-129).
- Recall: `ContextBroker.assemble(query, limit=8, include_derived, sources,
  window, account_id, commitments, lessons)` (`context/broker.py:100`) —
  `window` filters on `occurred_at` (inclusive; undated pass, :565-575) but
  **no caller passes it**; cache scope already keys on window
  (`context/cache.py:39`, `broker.py:522-539`). `EvidenceItem.score` is
  hardcoded 1.0 (`broker.py:48-54`) and not serialized
  (`context/packet.py:58-65`). Selection loop consumes FTS rank order
  (`broker.py:154-171`).
- Both generation and verification hops run on one model+credential inside
  `ScopedSynthesizer` (`processing/synthesis.py:68-125`).
- Summarization refuses oversized scopes rather than splitting
  (`processing/summarization.py:203-225`, `MAX_INPUT_BYTES` :212-213,
  `MAX_BATCH` :43).
- **Host contract limitation (honest constraint):** the plugin declares
  `provides_hooks: []` (`integrations/hermes-memory/plugin.yaml:42`) and the
  faithfulness doc's remaining-work #1 states the host has "no mandatory
  pre-send validation hook". Callbacks cover turn start, tool calls,
  completed-turn capture, compression, backup — nothing intercepts outgoing
  assistant text. F1 therefore has two halves: a framework-side boundary
  service (shippable and testable alone) and a small host-side hook
  (hermes-agent; versioned contract extension).
- Turn context the boundary can key on: the provider already tracks the
  last injected packet (`_last_injected`, set :998/:1006/:1009) with its
  record IDs, and stamps authority/generation on consumption
  (`_consume_queued` :1134-1149).

---

# Tier 1

## F1 — Pre-send verification boundary

**Goal.** No personal-memory assertion leaves the agent unchecked. Automatic
machine check; the owner sets strictness once and is never prompted per
message.

### F1.a Framework side: the boundary service

New module `src/hermes_memory/processing/boundary.py`:

```python
class SendBoundary:
    def __init__(self, *, synthesizer, store, settings, gate): ...

    def check(self, draft: str, *, packet: Packet | None,
              account_id: str | None, session_id: str | None) -> BoundaryVerdict
```

`BoundaryVerdict` (immutable dataclass, JSON-serializable):

```
disposition: "pass" | "revise" | "block"
claims: [{text, record_ids, label, source}]        # what was checked + verdicts
detector: {memory_shaped: bool, pattern_hits: [...], skipped_reason?: str}
action: {mode: "none" | "downgrade" | "strip" | "hold", notes: [...]}
receipt: {boundary_version, checked_at, packet_id, watermark, digests}
```

Pipeline, in order (cheap-first, the codebase's discipline):

1. **Detector** (deterministic, no model): memory-shaped-sentence scan of
   the draft using a bounded pattern lexicon — first/second-person
   assertions ("you are/have/had/will", "I remember that you…"), health,
   preference, schedule, relationship, numeric-about-person phrasing — plus
   any sentence whose entities overlap the injected packet's entities. No
   match ⇒ `pass` with `skipped_reason="no_memory_claims"`, zero model cost.
   The lexicon is data (a module constant), multilingual seeds included
   (EN/HI/Hinglish), conservative: over-triggering only costs a verifier
   call, under-triggering is the failure mode to avoid.
2. **Claim extraction + evidence binding:** each memory-shaped sentence
   becomes a candidate claim. Evidence comes from the **turn's injected
   packet** (`packet.items[].id` + their canonical text already resolved in
   the packet) — the agent is only permitted to assert what it was given.
   If the sentence references facts beyond the packet, its `record_ids` are
   empty and the claim proceeds as unverifiable.
3. **Verification:** claims run through `verify_memory_claims` semantics —
   reusing the existing service's resolution, watermark, gate admission,
   entailment and deterministic vetoes (extend `verification.py` with an
   internal entry point that takes a pre-resolved packet instead of
   caller-supplied evidence; the public contract stays unchanged).
4. **Disposition by policy:**

   | Outcome | Default action |
   | --- | --- |
   | all supported | `pass` |
   | insufficient evidence | `revise`: rewrite the sentence to tentative phrasing ("I recall … but can't confirm it from what's stored") or strip it, per policy |
   | contradicted / deterministic veto | `block`: hold the send, return the reason to the agent so it self-corrects |

5. **Receipts:** every check writes an append-only boundary receipt row
   (new table, migration `0020_boundary`: `boundary_receipts(id, session_id,
   packet_id, watermark, disposition, claims_json, digests, created_at)`) —
   auditable, cache-invalidation exempt (it does not change memory content,
   so no `context_revision` trigger; it is a log, not a fact store).

**Policy knob** (env, owner-set once): `HERMES_MEMORY_BOUNDARY_MODE` in
`off | warn | enforce` (default `warn` for the first release — the system
learns its false-positive rate from receipts before enforcement;
`enforce` = the table above; `warn` = disposition recorded, message still
sends with a marker). Fail-closed: boundary service unreachable in
`enforce` mode ⇒ send held with an explicit reason, never silently passed.

### F1.b Host side: the send hook

Versioned host-contract extension, one hook, minimal surface:

- `plugin.yaml` grows `provides_hooks: ["pre_send"]` and the host gains the
  corresponding call: `provider.pre_send(draft: str, turn_context) ->
  {disposition, revised_draft?, receipt}` invoked for both streaming
  (pre-first-chunk: check the composed draft before streaming begins) and
  non-streaming sends.
- Provider implements `pre_send` by delegating to the framework
  `SendBoundary` with the turn's `_last_injected` packet and
  `_caller_account_id`.
- Streaming note: full-draft checking means checking before the first token
  is streamed (draft is complete pre-stream in Hermes's send path); if the
  host ever streams token-by-token with no complete draft, the hook
  contract must say so and the boundary runs in `warn`-only there — stated
  in the contract, not silently degraded.
- Host change lands in a re-cloned `hermes-agent` with the saved integration
  patch applied (`~/uninstall-backups/hermes-agent-local-changes.patch`);
  the hook is additive and upstreamable.

**Tests.** Framework: detector recall/precision fixtures (EN/HI/Hinglish,
incl. the "severe"/"pakka"/"90 kg" regressions); packet-bound evidence
binding; each disposition; `warn` vs `enforce`; unreachable-verifier
fail-closed; receipt immutability. Host-half: a fake provider asserting the
hook shape and that `block` holds the send.

**Acceptance.** A live probe sends three drafts (faithful, exaggeration,
contradiction) through the boundary: pass / revise / block respectively,
with receipts; `warn` mode message rate measured over a session before
`enforce` is enabled.

## F2 — Verified agent-written memories

**Goal.** Agent-authored prose never enters the canonical store as
indistinguishable-from-evidence fact.

**Where.** `memory_remember` schema (`provider.py:148-158`) and `_remember`
(:935-959) — both small, the write path bypasses the spool today.

**Design.**

- Schema grows optional `evidence: [{record_id, quote}]` (same shape as
  `memory_verify`'s, `additionalProperties:false`, ≤16 entries).
- With evidence: `_remember` runs the deterministic checks + entailment
  (`verify_memory_claims` internals) **before** commit; only the supported
  subset is stored, with `metadata.verification` carrying the receipt
  (contract version, digests, labels). The tool's response reports accepted
  vs rejected parts.
- Without evidence (legitimate for operational notes — "user asked for a
  reminder", agent plans, session bookkeeping): still commits, but the
  record is marked `metadata.agent_authored: true` and rendered distinctly
  downstream — `packet.render()` labels such items
  "agent note (unverified)" and `Packet.coverage` accounts for them
  (they can never be cited as evidence for a boundary check in F1 —
  agent-authored records are excluded from evidence binding, closing the
  loop where the agent's own unverified prose launders into "verified"
  claims two turns later).
- Strict mode (env `HERMES_MEMORY_REMEMBER_STRICT=false|true`, default
  false): refuse unevidenced claims that the detector (F1's lexicon) marks
  personal-memory-shaped; operational kinds still pass.
- `memory_remember` guidance string updated to teach the evidence field;
  `_WRITING_TOOLS` gating unchanged.

**Tests.** Evidenced write passes and stores receipt; rejected part
excluded; unevidenced write flagged `agent_authored`; strict mode refuses a
personal claim without evidence and accepts an operational note;
agent-authored records excluded from F1 evidence binding and rendered
labeled.

**Acceptance.** Round-trip test: agent writes an evidenced memory →
recalled → asserted in a draft → boundary passes on it; the same sentence
from an unverified note never yields `pass`.

## F3 — Independent verifier route

**Goal.** Generation and verification run on different models, removing the
correlated-error hole the faithfulness doc admits.

**Design** (from the full plan's §3.1, restated for this slice):

- Env: `HERMES_MEMORY_VERIFIER_BASE_URL/_MODEL/_RESOURCE` +
  `HERMES_MEMORY_ROUTE_CREDENTIAL_VERIFIER`, parsed by the existing
  `_route` (`config.py:466-477` — loopback/LAN allowlist rules apply
  automatically).
- `build_routes` (`processing/routes.py:137-154`) grows a `verifier` route
  (`priority="interactive"`).
- `ScopedSynthesizer(model=…, verifier=…)` — hop 2 uses the verifier route.
  **Configured verifier ⇒ verifying on the generation route is refused
  outright** (fail-closed). Not configured ⇒ exactly today's behavior, with
  reports continuing to name the verifier used (they already do; now the
  name can differ).
- `summary_fingerprint` (`summarization.py:64-69`) includes the verifier
  identity so checking lineage is visible and a re-verification with a
  different verifier is a new fact.
- Serves F1 and F2 directly: boundary and write checks want an independent
  judge even more than summaries do.

**Tests.** Same-model refusal; two fake transports asserting hop
separation; fingerprint changes with verifier identity; absent-route
backward compatibility.

## F4 — Temporal windows in recall

**Where.** The machinery exists (`broker.assemble(window=…)`,
`_in_window`, cache scope includes window). The work is surfacing.

- `memory_recall` tool schema (`provider.py:123`) grows optional
  `since`/`until` (ISO-8601 date or datetime; strict validation, inverted ⇒
  refusal). `prefetch`/`_consume_queued` thread it through to `assemble`.
- Relative-time parsing is the agent's job (the model converts "last week"
  to dates; the tool accepts only absolute ISO) — no fuzzy date library, no
  new dependency.
- `render()` includes the applied window in the packet header so the agent
  sees the constraint it asked for.
- CLI `explain` gains `--window since until` for debugging.

**Tests.** Window threading end-to-end (store fixture with dated records);
inverted-window refusal; undated records still pass; cache separation by
window; packet header shows it.

## F5 — In-packet duplicate suppression

**Where.** After the selection loop (`broker.py:154-171`).

- Bounded pairwise shingle-Jaccard on candidate texts: ≤200 comparisons
  (8-gram shingles, hashed), drop the lower-priority item when similarity
  > 0.9; dropped IDs recorded on the packet (`deduped: [...]`) so
  suppression is visible and `explain` can name what was dropped and why.
- Superseded-revision handling in the FTS layer stays untouched.

**Tests.** Near-duplicates collapse; distinct items survive; cap respected;
`deduped` surfaced in `as_dict`/`explain`.

---

# Tier 2 (specified now, built on demand)

Trigger criteria stated up front so "on demand" is a policy, not a mood:
F6 when a live recall-quality eval shows the expected record outside the
top-3 in >10% of queries or the corpus passes ~5k live records; F7 when a
summarize scope exceeds the batch limit twice in a week.

## F6 — Recall reranking with real scores

- New `context/rerank.py`: `RerankClient.rerank(query, texts, top_n)` calling
  **through the gate** (route `by_name("rerank")`, built like
  `formation.py:67`) to the existing Qwen3-Reranker service
  (`127.0.0.1:8185`, Cohere `/v1/rerank`, P(yes) softmax —
  `models/reranker.py:197-204,267-287`). Optional constructor dependency on
  `ContextBroker` (same injection style as the Hindsight client); absent
  route ⇒ channel `unavailable`, FTS order kept, logged — degradation with
  a receipt, not a fallback.
- Insertion: between authorization (`broker.py:128-131`) and selection
  (`:154-171`), bounded 3s timeout copying the derived-channel deadline
  pattern (`:257-299`); ≤64 candidates, only when `len(considered) > limit`;
  admitted/charged as `local-gpu` through the gate.
- Scoring replaces `score=1.0` (`broker.py:48-54`):
  `0.7·P(yes) + 0.2·recency_decay + 0.1·min(support,5)/5`, components
  serialized on the item (`packet.py:58-65`) and shown by `explain`. When
  reranking is unavailable the lexical rank score substitutes and the
  packet says so. `lineage` gains a batched, capped `support_counts(ids)`.
- `Packet.channels` grows `rerank` in {available, unavailable, skipped}.

## F7 — Chunked/episode summaries

- `summarize_plan` proposes `N = ceil(scope / MAX_BATCH)` byte-bounded
  batches instead of blocking (`summarization.py:203-225`); each batch is a
  separately-keyed job (`jobs.py:111-115`) with `coverage="batch i of N"`;
  one review digest covers the whole partition.
- Roll-up summary of kind `episode` (add to `summaries.KINDS`,
  `knowledge/summaries.py:27`; member record IDs on the row) is produced
  only when **every** batch completed; otherwise coverage is `partial`
  naming the failed batches — never `full` from a prefix, preserving the
  house honesty rule.
- Formation stays one-job-per-record (the `input_revision` invariant is
  correct for raw facts).

---

# Sequencing, releases, ritual

```
Release A (Tier 1 core):   F3 verifier route → F2 write verification → F5 dedup → F4 windows
Release B (boundary):      F1.a framework service (warn mode) + receipts → live warn-rate measurement
Release C (enforcement):   F1.b host hook + HERMES_MEMORY_BOUNDARY_MODE=enforce
Tier 2 (on trigger):       F6, then F7
```

Rationale: F3 lands first because F1/F2 want an independent judge; F1 ships
in `warn` mode first so enforcement is switched on with measured
false-positive data, not hope.

Per release, the existing ritual unchanged: full `pytest` (house style —
store fixture, injected transports, fake clocks; `no_unstarted_socket`
guards every network claim); `compatibility --write`; `release --snapshot`
plan→apply with the printed digest; `activate-release` plan→apply; re-run
the live synthetic full-flow + faithfulness checks and keep receipts;
update the docs' honest-scope notes. Migrations: one new pair
(`0020_boundary`, plus `0021` for F7's episode column if Tier 2 lands) —
appended, never rewritten.

## Risk register

| Risk | Mitigation |
| --- | --- |
| Boundary false positives hold legitimate messages | `warn` mode first with measured rates; `revise` downgrades before `block`; receipts make every decision reviewable |
| Boundary latency on every send | Cheap deterministic detector first — non-memory messages skip the model entirely; model hop only for memory-shaped drafts, bounded tokens |
| Agent games the detector | Evidence binding is packet-bound (F1 can only verify against what was actually injected), and F2 excludes agent-authored records from evidence — the loops are closed structurally, not by prompt |
| Write verification blocks legitimate notes | Evidence optional; operational kinds unaffected; strict mode opt-in |
| Streaming path can't check a full draft | Contract states it; boundary runs warn-only there until the host exposes the composed draft |
| Rerank latency (Tier 2) | 3s bound, conditional invocation, explicit `unavailable` degradation |

## Effort

| Feature | Modules | New tests | Migration |
| --- | --- | --- | --- |
| F1 | boundary.py, verification.py internals, provider pre_send, plugin.yaml, host hook | ~7 files | 0020 |
| F2 | provider (_remember, schema), packet render | ~3 | — |
| F3 | config, routes, synthesis | ~3 | — |
| F4 | provider schema/prefetch, packet header, CLI explain | ~2 | — |
| F5 | broker, packet | ~2 | — |
| F6 | rerank.py, broker, packet, lineage helper | ~4 | — |
| F7 | summarization, summaries | ~3 | 0021 |

Tier 1 ≈ one focused week; each feature ships independently and none
regresses the others' tests.
