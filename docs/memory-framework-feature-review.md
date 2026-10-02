# Memory framework feature-flow review and bug register

Started 2026-10-02 as a review-only source/flow audit. The owner subsequently requested fixes for all confirmed issues in core logic. **R-001 through R-032 now have source fixes and expected-safe regression tests.** The implementation and verification receipt is below. No production archive mutations, sends, live inference calls, service changes or installs were performed in this remediation pass. Historical integration/clarification fixes are separate from this register.

Follow-up analysis, without code changes: [memory intelligence and performance review](memory-intelligence-optimization-review.md). Its **M-001–M-034 register remains open** and separates new integration gaps, reproduced availability/provenance issues, Hinglish quality unknowns, and optimization opportunities. It does not change the historical remediation status of the R-register. Installation remains deferred.

## Evidence standard

Each confirmed finding records an ID, severity, feature and flow, expected/actual behavior, source anchors, reproducible evidence, impact, proposed core fix and acceptance tests. Suspected risks and untested flows are separate. The original probes asserted defects; they have now been converted to expected-safe contracts and expanded with positive, replay and migration cases. Existing unrelated worktree changes are preserved.

This document is maintained incrementally. Coverage is a review ledger, not a claim of exhaustive branch or live end-to-end coverage. Companion reproductions: `tests/review/test_feature_review.py` (offline, synthetic only).

## Feature coverage ledger

| Feature family | Flow under review | Current coverage |
| --- | --- | --- |
| Evidence / attachments / lineage | Envelope validation → revision commit → visibility → journal / blob read | Selected source traced; immutable revision, erased replay and blob probes reproduced |
| Source connectors | File / structured / mail / WhatsApp / MCP / Hermes events → paging → lease → canonical commit → cursor / gaps | File, structured, WhatsApp, MCP and sync paths traced/probed; email redaction probed; Gmail, media, Hermes-event branches incomplete |
| Context retrieval | Query → lexical / assertions / summaries / backend facts → scope / budget → recheck → cache → Hermes turn | Broker/cache paths traced and authorization, freshness, invalidation and limit defects reproduced; live model quality untested |
| Identity | Account normalization → candidate → owner edge → group → access / revoke | Selected lifecycle/access code traced; cached revocation reproduced; expiry/race matrix pending |
| Assertions / contradictions | Quoted evidence → candidate / confirmation → supersession → retrieval / conflicts | Selected proposal/retrieval paths traced; scope, withdrawal cache and premature supersession reproduced; full contradiction matrix pending |
| Summaries / mental models | Scope selection → model reflection → provenance → publish / withdraw → retrieval / refresh | Store/planning paths traced; approval, citation identity and week interval reproduced; actual model provenance quality pending |
| Goals / prospective events | Proposal → activation → conditions / schedule → claim → durable handoff → revision / settle | Selected goal/event/predicate paths traced; audit and two predicate defects reproduced; complete scheduling/revision matrix pending |
| Proactivity / attention / delivery | Due event → eligibility → policy → decision → outbox → transport → receipt / recovery | Main outbox/transport paths traced; opt-out race, manifest loss, receipt and restore defects reproduced; actual transport untested |
| Clarifications | Subject fence → question → delivery attempt → native/coded reply → decision / dossier | Previous safety review retained; residual command receipt defect reproduced; live Telegram and cross-profile flow pending |
| Learning / outcomes / evaluation | Evidence → lesson proposal → evaluation / owner activation → applicability → retirement | Selected lesson/outcome/evaluation paths traced; applicability and empty receipt defects reproduced; full evaluation adversarial matrix pending |
| Processing / resources / budgets | Journal consumer → job → allowance / budget → device gate → submission → polling → projection | Main worker/job paths and selected budget/allowance/gate paths traced; lease/epoch and seconds defects reproduced; device concurrency pending |
| Forget / reset / restore | Preview → owner digest → tombstone → local bytes / derived obligations → snapshot overlay | Selected erasure/reset/snapshot/recovery paths traced; four byte/erasure and delivery rollback probes reproduced; remote deletion verification pending |
| Operations / installation | CLI / plugin entry → profile binding → status / doctor / explain → install / upgrade / remove | Selected entry/service/upgrade code inspected; previous installation audit and existing tests retained; full installer/doctor/security review not complete |

## Confirmed findings — historical evidence and remediation status

### Baseline and reproduction instructions

Review date: 2026-10-02. Framework source: `/home/jugaadu/Projects/hermes-memory`; Git HEAD `e26ae2309d73e969a7079372f529b0a4f1ca3148`, with pre-existing working-tree changes. Findings describe those working bytes, not a clean checkout of HEAD. Deployment manifest framework digest observed in the preceding integration work: `cbfd4914ac4b9135b4fecbcff20c9e036d0f274f1124c62cd825dce03d5118ab`; last installed runtime/plugin revision `237133b492abed552e4567652b1f218f87bbf75b`. This review does not establish that every finding was exercised against the installed runtime.

All 32 entries below were confirmed by synthetic offline reproduction. **Current status for every R-ID: fixed in source, regression verified; not deployed by this pass.** The original 39 cases now assert expected-safe behavior; additional cases bring this file to 65 cases. Fixtures use temporary stores and injected backends/transports; the suite's network guard prohibits external connections. Detailed finding entries retain the original evidence/source anchors for traceability; line numbers describe the reviewed pre-fix bytes and may have moved.

From the framework root:

```bash
.venv/bin/python -m pytest -q tests/review/test_feature_review.py
.venv/bin/python -m pytest -q tests/review/test_feature_review.py -k r001
```

Replace `r001` with the lowercase ID of any entry. Each entry names its exact probe below. Source anchors are relative to `src/hermes_memory/`; function names are more stable than line numbers in the dirty worktree. Priority: **High** = isolation, erasure, consent, durable state or data-loss risk; **Medium** = incorrect feature behavior or unreliable control; **Low** = bounded API-contract issue. These are local impact judgments, not CVSS scores.

### Backlog index

| ID | Priority | Defect |
| --- | --- | --- |
| R-001 | High | Cached authority and derived state survive revocation/withdrawal |
| R-002 | High | Assertions expose evidence withheld from the caller |
| R-003 | High | Backend-derived facts survive canonical hide during recall |
| R-004 | High | Mental models publish without a configured owner |
| R-005 | Medium | Immutable revision identity omits precision and parents |
| R-006 | High | Reset reports completion while retaining attachment bytes |
| R-007 | High | Replaying an erased revision recreates destroyed attachments |
| R-008 | Medium | Export pages exceed declared record caps |
| R-009 | Medium | Touching unchanged files causes immutable revision conflicts |
| R-010 | Medium | Edited structured exports reuse revision one |
| R-011 | High | Delivery proceeds after opt-out between lease and attempt |
| R-012 | Medium | Generator evidence disappears from outbox manifests |
| R-013 | Medium | Cancellation reasons can corrupt audit JSON |
| R-014 | Medium | Lesson prerequisites and exceptions are not enforced |
| R-015 | High | Empty host receipts count as verified successful outcomes |
| R-016 | Medium | Seconds budget does not block further dispatch |
| R-017 | High | Semantic facts bypass account, source and time filters |
| R-018 | Low | Broker ignores the requested raw item limit |
| R-019 | Medium | Summary identity omits its citation manifest |
| R-020 | High | Stale workers can mutate newly leased/old-epoch jobs |
| R-021 | Medium | Re-enqueued cancelled jobs retain obsolete epochs |
| R-022 | Medium | Weekly summary windows span thirteen days |
| R-023 | High | WhatsApp pagination silently loses a chat's remaining messages |
| R-024 | High | MCP pagination discards records before advancing remote cursor |
| R-025 | High | Import adapters bypass shared secret redaction |
| R-026 | High | Unapproved assertion candidates supersede confirmed claims |
| R-027 | Medium | Command transport invents positive receipts from invalid output |
| R-028 | High | Restore reintroduces erased attachment bytes |
| R-029 | High | Restore rewinds delivery history and permits duplicate sends |
| R-030 | High | Restore drops orphan tombstones, permitting forgotten re-import |
| R-031 | High | Reminder thresholds use forgotten, expired measurements |
| R-032 | Medium | Message-author predicates match arbitrary metadata substrings |

### R-001 — Cached authority/derived state survives revocation or withdrawal

- Flow/evidence: warm a packet with an owner-confirmed A↔B identity edge, revoke the edge, then repeat the query. Cached B evidence remains visible; manually invalidating the cache makes the fresh result withhold it. Separate probes retract an assertion and withdraw a summary; both remain in cached sections.
- Source: `storage/evidence.py:EvidenceStore.watermark` (313), `storage/identity.py:revoke` (251), `knowledge/assertions.py:retract`, `knowledge/summaries.py:withdraw` (160), `context/cache.py:PacketCache.get` (40). The watermark reflects epoch/journal changes, not all authority/materialized-state mutations.
- Probes: `test_r001_cached_identity_access_survives_edge_revocation`; `test_r001_cached_derived_section_survives_owner_withdrawal` (assertion/summary). Impact: revoked access or withdrawn guidance remains available until cache expiration.
- Core fix/acceptance: introduce a transactional authority/derived-state revision used in packet validity; recheck at handoff. Revocation, retraction and withdrawal must remove affected cached material immediately, without manual invalidation or waiting for TTL. Also verify time-based authority expiry separately.

### R-002 — Assertions bypass caller evidence authorization

- Flow/evidence: caller A cannot access account B's raw record: the broker returns no raw item and increments withheld count. An assertion citing the same record nevertheless exposes its value, quote and record ID in the packet/rendered context.
- Source: `context/broker.py:_assertions` (335), `knowledge/assertions.py:matching` (290). Retrieval does not carry the raw channel's per-caller evidence authorization into assertions.
- Probe: `test_r002_assertion_exposes_record_withheld_from_caller`. Impact: within-profile account isolation leaks through a derived channel. This is not evidence of a cross-profile bank leak; the actual plugin's caller binding requires a separate integration matrix.
- Core fix/acceptance: authorize every cited record before assertion/contradiction selection, budgeting and rendering. Test unjoined/joined accounts and anonymous callers, including contradictory values and quotations, with no unauthorized derivative disclosure.

### R-003 — Freshness recheck ignores backend facts

- Flow/evidence: an injected backend hides the supporting canonical record through a separate connection during recall and returns a fact derived from it. The broker's recheck drops raw evidence, but the fact still renders.
- Source: `context/broker.py:assemble` (99), `_derive` (236), `_recheck` (443); the latter only filters raw items.
- Probe: `test_r003_derived_fact_survives_canonical_hide_during_recall`. Impact: concurrent hide/forget can leave stale derived content in the delivered packet.
- Core fix/acceptance: resolve and revalidate dependency, visibility and authority for every section after external I/O. If freshness cannot be established, withhold the section; a caveat or refusing to cache is insufficient. Race hide/forget/revision against recall and verify no affected derivative survives.

### R-004 — Ownerless mental-model publication

- Flow/evidence: construct a summary store without an owner and publish a mental model without an approver. Both principal values are `None`, so the inequality-based approval check passes.
- Source: `knowledge/summaries.py:SummaryStore.publish` (90).
- Probe: `test_r004_mental_model_can_publish_without_configured_owner`. Impact: owner-only psychological interpretation can be published without any owner authorization.
- Core fix/acceptance: require a configured nonempty owner and an explicit host-authenticated owner decision. Missing owner, missing approval and non-owner approval must fail; a valid owner approval must succeed.

### R-005 — Incomplete immutable revision fingerprint

- Flow/evidence: replay a revision with changed occurrence precision and parent dependencies. It is accepted as a duplicate and the new lineage is lost because these fields do not participate in revision identity.
- Source: `storage/evidence.py:prepare_envelope` (101), fingerprint construction (~147), `write_prepared` (385).
- Probe: `test_r005_canonical_revision_ignores_precision_and_parent_dependencies`. Impact: revision semantics and dependency-driven invalidation can silently diverge from supplied evidence.
- Core fix/acceptance: fingerprint all canonical semantic fields, with a versioned compatibility/migration strategy. Same-revision changes to precision or parents must fail; a new revision must retain the changed lineage.

### R-006 — Reset leaves attachment payloads behind

- Flow/evidence: attach bytes, reset the store, and inspect canonical storage. Reset reports complete while attachment references and blob chunks remain. Records are hidden/deleted, so this does not demonstrate ordinary blob-read access.
- Source: `lifecycle/reset.py:ResetController` reset flow (78); missing `BlobStore.release` cleanup.
- Probe: `test_r006_reset_reports_complete_but_keeps_attachment_payloads`. Impact: forgotten attachment bytes remain in storage and subsequent snapshots despite the completed reset result.
- Core fix/acceptance: atomically release affected references and unreferenced chunks, or explicitly report/document retained bytes instead of claiming completion. Test all-reset, shared blobs, crashes and snapshot export after reset.

### R-007 — Erased revision replay recreates destroyed blobs

- Flow/evidence: normal erasure destroys attachment chunks. Replay the original envelope: duplicate handling attaches its payload again to the deleted record. The record remains hidden, but erased bytes have returned.
- Source: `storage/evidence.py:write_prepared` (385), attachment handling; `storage/blobs.py:attach_staged` (151).
- Probe: `test_r007_replay_of_erased_revision_recreates_destroyed_blob`. Impact: idempotent source replay reverses physical attachment cleanup.
- Core fix/acceptance: enforce erasure/deleted-state fences before attachment staging/commit and sensitive replay receipt persistence. Replaying erased SDK/import revisions must not recreate payloads or make records visible.

### R-008 — Export page caps are not enforced

- Flow/evidence: a single structured export containing twelve rows is read with a two-record page cap. File/structured adapters return all twelve with a terminal cursor. Their inner expansion is unbounded by the advertised page cap.
- Source: `sources/files.py:FileSource.read_page` (51), `sources/structured.py:StructuredSource.read_page` (69). Email has a similar source-inspected expansion pattern, not a separate reproduced case.
- Probe: `test_r008_single_export_overflows_declared_page_bound` (two adapters). Impact: oversized imports can violate downstream page ceilings and abort ingestion; aggregate byte bounds also require verification.
- Core fix/acceptance: persist intra-file offsets and enforce record/byte caps across expanded rows. A 1,001-row export under a two-record cap must ingest fully over bounded pages, without rejection or discarded tails.

### R-009 — Unchanged file touches conflict with immutable revisions

- Flow/evidence: ingest a file, change only its modification time, and ingest again. Content-derived revision stays the same while modification metadata changes the fingerprint, producing an immutable revision conflict.
- Source: `sources/files.py:_whole_file` (142), `_structured` (153).
- Probe: `test_r009_touch_unchanged_file_causes_immutable_revision_conflict`. Impact: ordinary filesystem operations interrupt incremental imports of unchanged content.
- Core fix/acceptance: separate observational metadata from semantic revision fields, or derive revisions consistently from the chosen semantic contract. Touching identical content must replay safely; real content edits must generate a distinct revision.

### R-010 — Structured edits always reuse revision one

- Flow/evidence: import a structured sample/series, change a value and import again. Stable source identity plus hardcoded revision `1` causes a revision conflict instead of recording the update.
- Source: `sources/structured.py:_sample`, `_series` (revision construction around 176/211).
- Probe: `test_r010_edited_structured_export_reuses_revision_one` (per-sample and series). Impact: supported incremental source edits cannot be ingested reliably.
- Core fix/acceptance: define stable logical identities and content/version-derived revisions. Unchanged reads must be idempotent; edited values, appended rows and reorderings must behave according to an explicit revision policy.

### R-011 — Opt-out after lease does not prevent delivery

- Flow/evidence: an injected lease wrapper opts the topic out after claiming an artifact. Revalidation would reject it, but `deliver_once` transitions to an attempt and calls the synthetic sink anyway.
- Source: `proactive/outbox.py:attempt` (284), `proactive/delivery.py:deliver_once` (129).
- Probe: `test_r011_artifact_can_be_attempted_after_topic_opt_out`. Impact: consent/eligibility changes before handoff do not reliably stop a send.
- Core fix/acceptance: make final eligibility, expiry, evidence and consent checks part of the atomic attempt transition. Define the irrevocable handoff boundary without holding a DB lock over network I/O. Opt-out/hide/pause/snooze before that boundary must produce zero transport calls.

### R-012 — Iterable evidence is consumed twice

- Flow/evidence: prepare an artifact with a generator of evidence. Digest generation consumes it; manifest serialization then stores an empty list, and digest revalidation fails.
- Source: `proactive/outbox.py:prepare` (139).
- Probe: `test_r012_generator_evidence_is_lost_by_outbox_prepare`. Impact: valid artifacts become undeliverable and lose their auditable dependency manifest.
- Core fix/acceptance: materialize and validate a bounded evidence tuple once, then reuse it for identity, digest and persistence. Lists, tuples and generators must produce equivalent manifests, with forgotten evidence still suppressing delivery.

### R-013 — Cancellation corrupts JSON audit metadata

- Flow/evidence: cancel a due event with a reason containing quotes. Interpolation produces invalid audit JSON even though the state transition completes.
- Source: `prospective/due_events.py:DueEventLog.cancel` (141).
- Probe: `test_r013_cancellation_reason_produces_invalid_json_audit`. Impact: explanation/audit consumers can fail or lose cancellation details.
- Core fix/acceptance: encode structured audit metadata through the JSON serializer in the same transaction. Quotes, backslashes, newlines and Unicode must round-trip through audit readers.

### R-014 — Lesson prerequisites/exceptions are annotations, not gates

- Flow/evidence: an active lesson requires owner approval and excludes production hosts. Applicability still returns it for production without that approval; only applicability/scope/support are checked.
- Source: `learning/lessons.py:propose` (111), `applicable` (232).
- Probe: `test_r014_lesson_prerequisites_and_exceptions_are_not_enforced`. Impact: retrieved guidance contradicts its declared conditions. The test does not execute the lesson's proposed action.
- Core fix/acceptance: define typed, machine-checkable conditions with fail-closed unknowns, or explicitly downgrade free-text fields to non-enforcing annotations. Missing prerequisites and matched exceptions must exclude actionable lessons; approved eligible contexts must include them.

### R-015 — Empty host receipt counts as checked success

- Flow/evidence: record a successful `host_receipt` outcome with no evidence spans using an outbox incapable of resolving a matching receipt. The verification loop executes zero checks and counts the result as support.
- Source: `learning/outcomes.py:OutcomeLog._check_authority` (140).
- Probe: `test_r015_empty_host_receipt_is_counted_as_checked_success`. Impact: unsupported success can influence learning/evaluation as if host-verified.
- Core fix/acceptance: require a nonempty, subject/event-bound verified receipt set from the authoritative host path. Empty, wrong-subject, unconfirmed and replayed receipts must not add verified support. Do not treat caller-supplied actor labels alone as authentication.

### R-016 — Seconds budget is only metered

- Flow/evidence: configure a one-second budget, charge ten seconds, then request another dispatch. Admission still allows it because it checks tokens/calls but not seconds.
- Source: `processing/budgets.py:BudgetLedger.admit` (66).
- Probe: `test_r016_seconds_budget_does_not_refuse_further_dispatch`. Impact: the exposed seconds budget does not bound dispatch.
- Core fix/acceptance: enforce remaining time using a documented admission estimate/reservation policy, or label the field metering-only. Exhaustion must refuse further work; failed/timed-out work must still be accounted for.

### R-017 — Semantic facts ignore query authorization/filter boundaries

- Flow/evidence: ask as A with a public-source filter and a future time window. Raw private B evidence is withheld; the injected semantic backend returns B's out-of-window fact, which enters context unchanged.
- Source: `context/broker.py:_derive` (236), `_recall_within_deadline` (~263). Neither remote arguments nor local fact validation establish equivalent account/source/window restrictions.
- Probe: `test_r017_backend_fact_bypasses_caller_source_window_and_account_scope`. Impact: filtered retrieval has a permissive semantic side channel. Unlike R-003, this is about scope/filtering rather than concurrent freshness.
- Core fix/acceptance: propagate supported remote filters and independently validate canonical provenance locally. Unresolved provenance must fail closed. All channels must honor account, source and temporal constraints.

### R-018 — Raw item limit is ignored

- Flow/evidence: request `limit=1` with five matching records. Candidate overfetch returns five, and packet assembly includes all five rather than capping raw results.
- Source: `context/broker.py:assemble` (99), raw inclusion loop (~158).
- Probe: `test_r018_broker_limit_one_returns_multiple_raw_items`. Impact: bounded callers receive more items than requested.
- Core fix/acceptance: separate candidate-pool size from result count, authorize before capping, and report truncation accurately. A one-item request must return at most one raw item.

### R-019 — Summary identity excludes citations

- Flow/evidence: publish the same title/body/kind/scope/processor/window with different citations. The second publication reuses the old ID/manifest instead of recording the new provenance.
- Source: `knowledge/summaries.py:publish` (90), summary ID construction (~126).
- Probe: `test_r019_summary_identity_omits_citation_manifest`. Impact: provenance changes are silently discarded and stale dependencies remain attached.
- Core fix/acceptance: include the effective citation manifest and relevant scope/coverage/epoch in immutable identity, or explicitly reject mismatched identity fields. Identical publication remains idempotent; different evidence must not silently inherit old provenance.

### R-020 — Job mutations lack original lease/epoch fencing

- Flow/evidence: worker A releases a claim; B claims it; A can still begin submission and alter B's row. Separately, epoch reconciliation quarantines an old job, but its old worker can mark it succeeded.
- Source: `processing/jobs.py:_transition` (469), worker mutation methods including `release` (181), `complete` (220). Job ID/state checks do not establish the original worker's lease and epoch.
- Probes: `test_r020_released_old_job_claim_can_mutate_new_holder_state`; `test_r020_pre_reset_worker_can_resurrect_epoch_quarantined_job`. Impact: stale workers corrupt current work or cross reset boundaries.
- Core fix/acceptance: CAS every worker mutation against original lease token, epoch and allowed state/generation; give owner reconciliation a distinct authority path. Stale A must never alter B's row or revive pre-reset work.

### R-021 — Re-enqueue leaves obsolete job generation fields

- Flow/evidence: cancel a job, bump epoch, enqueue the same logical job again. It reports a queued creation but retains the old epoch and cannot be claimed in the current epoch.
- Source: `processing/jobs.py:enqueue` (91), conflict-update branch.
- Probe: `test_r021_reenqueued_cancelled_job_keeps_obsolete_epoch`. Impact: retries/replanning silently produce stuck work.
- Core fix/acceptance: generate new epoch-bound work identity or fully and safely initialize every new-generation field. Define explicit cancelled-job re-enqueue policy. New jobs must be claimable; old submissions/receipts must not promote them.

### R-022 — Weekly summary interval spans thirteen days

- Flow/evidence: planning around September 25 includes September 19 and October 1, because both ends extend six days around the anchor.
- Source: `processing/summarization.py:_interval` (92).
- Probe: `test_r022_week_summary_window_covers_thirteen_days`. Impact: summaries include evidence outside an ordinary seven-day week and misstate their coverage.
- Core fix/acceptance: define calendar-week or rolling-seven-day semantics, then align planning, querying, citations and displayed dates. Exactly seven days must qualify; adjacent outside records must not. Verify timezone/end-boundary semantics separately.

### R-023 — WhatsApp page truncation silently loses messages

- Flow/evidence: import one five-message chat with a two-record page cap through adapter → connector runtime → canonical commit. Only two messages persist; the adapter advances to the end of the file, reports no gap, and coverage becomes current.
- Source: `sources/whatsapp_export.py:read_page` (88).
- Probe: `test_r023_whatsapp_page_drops_rest_of_single_chat_without_gap`. Impact: silent source data loss with falsely complete coverage.
- Core fix/acceptance: maintain an intra-chat message offset alongside file identity and byte bounds; terminal cursor only after consuming the file. Five messages must arrive across 2/2/1 pages, with idempotent restart and no silent loss.

### R-024 — MCP remote cursor skips unconsumed page tails

- Flow/evidence: an injected MCP tool returns three records and an after-three cursor despite a requested cap of two. Adapter keeps two and advances to that cursor; the next page is empty. Full runtime reports current coverage with only two records and no gap.
- Source: `sources/mcp.py:read_page` (175).
- Probe: `test_r024_mcp_page_limit_discards_tail_before_remote_cursor`. Impact: overfull responses lose records; local byte truncation needs the same safe cursor policy even for compliant record counts.
- Core fix/acceptance: retain resumable unconsumed tails, or reject/mark the page incomplete without committing the advanced cursor. Test remote overfill and byte caps: every record must survive or coverage must honestly show an unresolved gap.

### R-025 — File/mail/WhatsApp imports bypass shared secret redaction

- Flow/evidence: import a synthetic token recognized by the shared normalizer via file, email and WhatsApp adapters. All three persist the token verbatim, although the shared normalizer would redact it.
- Source: `sources/base.py:SourceAdapter.envelope` (111), `sources/files.py:_whole_file`, `sources/email.py:parse_message` (402), `sources/whatsapp_export.py:_record`.
- Probe: `test_r025_import_adapters_bypass_shared_secret_redaction` (three adapters). Impact: secrets can enter canonical storage and downstream context through uneven ingress policy. No real credential was used.
- Core fix/acceptance: enforce a shared, explicit normalization/redaction boundary before fingerprinting and sensitive persistence. Verify every adapter's record, receipt/spool and rendered output; separately define metadata/attachment policy without corrupting source identities.

### R-026 — Candidate proposal prematurely supersedes confirmed assertions

- Flow/evidence: propose an observed-pattern candidate that supersedes a confirmed assertion. The old assertion is immediately marked superseded while the new one remains unapproved; current assertions become empty before owner decision.
- Source: `knowledge/assertions.py:propose` (109), transactional supersession (~144).
- Probe: `test_r026_candidate_assertion_supersedes_confirmed_claim_before_approval`. Impact: speculative interpretation displaces authoritative memory without approval.
- Core fix/acceptance: record a pending replacement relation during proposal; retire old/promote new atomically only after an authorized qualifying decision. Pending or rejected candidates must leave the standing assertion unchanged.

### R-027 — Command transport manufactures an affirmative receipt

- Flow/evidence: command exits zero with non-JSON stdout. The sink returns default `sent: true` and `destination_unconfirmed: true`, although no explicit native acknowledgement was received.
- Source: `proactive/delivery.py:command_sink` (309), receipt default (~353).
- Probe: `test_r027_command_transport_manufactures_positive_receipt_from_invalid_stdout`. Impact: process success is mistaken for transport evidence. This is a residual transport-level issue, distinct from earlier clarification reply-fencing fixes.
- Core fix/acceptance: require explicit, validated transport receipts. Empty/malformed/partial output represents uncertainty, not invented success or definite failure; quarantine without blind resend. Validate destination/platform/native IDs where required.

### R-028 — Restore brings back destroyed attachment bytes

- Flow/evidence: snapshot an attachment, erase it normally (zero chunks), then restore the snapshot. Tombstone overlay keeps the record deleted, but restored attachment chunks remain in the database.
- Source: `lifecycle/recovery.py:_honour_tombstones` (326); applies visibility/FTS state without blob release.
- Probe: `test_r028_restore_reintroduces_destroyed_attachment_chunks`. Impact: restoring a backup reverses physical attachment erasure, even though ordinary reads remain blocked.
- Core fix/acceptance: replay physical attachment cleanup as part of erasure overlay before exposing restored state. Test shared references and re-export: erased chunks must not reappear in storage or new snapshots.

### R-029 — Restore rolls sent artifacts back into sendable state

- Flow/evidence: snapshot a prepared artifact, send it successfully into `accepted_unverified`, restore the pre-send snapshot, then deliver again. The same artifact/body reaches the synthetic sink twice.
- Source: `lifecycle/recovery.py:LEDGER_TABLES` (~39), ledger carry/apply (~239/252); delivery history is not preserved like erasure history.
- Probe: `test_r029_restore_rewinds_sent_outbox_to_sendable_state`. Impact: backup restore can duplicate notifications/side effects. Inquiry delivery tables need their own additional rollback matrix.
- Core fix/acceptance: preserve a non-rollbackable attempt/receipt ledger or carry/merge terminal and uncertain delivery facts by artifact generation. Accepted, confirmed and uncertain attempts must never become blind resend candidates after restore.

### R-030 — Orphan tombstone removal permits forgotten re-import

- Flow/evidence: snapshot before a record exists, add and erase the record, restore the older snapshot, then re-import the original record. Recovery drops the carried tombstone because its record is absent; retained erasure intent does not block ingestion. The forgotten text becomes normally searchable.
- Source: `lifecycle/recovery.py` ledger application (252), orphan filtering (~282); `storage/evidence.py:write_prepared` (385).
- Probe: `test_r030_orphan_tombstone_dropped_on_restore_allows_forgotten_record_reimport`. Impact: a real erasure fence is lost across restore, not merely hidden bytes retained.
- Core fix/acceptance: retain record-independent erasure identities/deny entries and consult them at every ingress path. Define how new revisions/source identities relate to erasure. Restoring an older database must preserve rejection of forgotten replay while keeping foreign keys valid.

### R-031 — Reminder thresholds use expired/forgotten assertions

- Flow/evidence: a confirmed measurement expired before evaluation and its supporting record is hidden. `AssertionStore.current(at=...)` returns nothing, but the threshold predicate still returns satisfied from its direct assertion query.
- Source: `prospective/predicates.py:_threshold` (~155).
- Probe: `test_r031_reminder_threshold_accepts_forgotten_expired_measurement`. Impact: a reminder can be triggered by evidence no longer valid or available.
- Core fix/acceptance: reuse authoritative assertion time/support/conflict checks and explicit unit/scope rules. Forgotten, hidden, expired and superseded measurements must not satisfy predicates; unresolved evidence should produce unknown rather than an invented decision.

### R-032 — Message-author predicate matches arbitrary metadata

- Flow/evidence: a record contains the queried account string only in an unrelated annotation. With no author binding, `new_message_from` still returns satisfied because it searches serialized metadata with `instr`.
- Source: `prospective/predicates.py:_new_message` (~117).
- Probe: `test_r032_new_message_from_accepts_account_string_in_arbitrary_metadata`. Impact: mentions, recipients or unrelated annotations can falsely satisfy author-based reminders.
- Core fix/acceptance: use typed sender/author fields with validated source/account identity mapping and relevant coverage. Annotation, recipient and mere-mention matches must not qualify; actual authors should qualify under documented identity rules.

## Core remediation — implemented in the requested order

Changes are grouped by shared invariant, rather than per-feature bypasses:

1. **Monotonic erasure and recovery:** R-030, R-028, R-007, R-006; preserve delivery history R-029. Introduce durable fences that survive snapshots/replays, with atomic physical cleanup.
2. **Uniform information/approval boundaries:** R-001, R-002, R-003, R-017, R-025, R-004, R-026. Centralize derivative authorization, freshness, ingress redaction and owner decisions.
3. **Durable worker ownership:** R-020, R-021. Establish lease/epoch CAS contracts before changing processing consumers.
4. **Consent, receipts and learning authority:** R-011, R-027, R-012, R-015, R-031, R-032. Tie final handoff and downstream learning to validated current evidence.
5. **Lossless connectors and revision semantics:** R-023, R-024, R-008, R-009, R-010, R-005. Define resumable bounded pages and complete canonical revision identity.
6. **Derived correctness and controls:** R-019, R-022, R-014, R-016, R-013, R-018.

### Implementation map

| R-IDs | Core change | Verification |
| --- | --- | --- |
| R-006, R-007, R-028, R-030 | Schema 15 introduces record-independent `erasure_fences`, including backfill from surviving confirmed intents when v14 already dropped orphan tombstones. Canonical ingress acknowledges fenced replay without storing bytes. Direct blob writes reject deleted targets. Reset and restore release attachment references/chunks; migration cleans attachment leftovers of deleted records. Restore honors independent fences even across multiple different snapshots. | Original four safe contracts; direct blob bypass; v14 orphan-intent migration; older→newer restore with forgotten attachment |
| R-029 | Durable `delivery_fences` survive rollback; restored sendable artifacts with prior handoffs become uncertain. Inquiry attempt fences are also carried and restored open inquiries are quarantined. Restore advances epoch and clears connector leases to fence in-flight writers. | Send→restore→deliver produces only one sink call; pre-restore job cannot report success |
| R-001 | Transactional `context_revision` triggers cover canonical visibility/dependencies, authority and materialized knowledge. Packet validity uses this revision. Identity group membership is part of scope validity, including natural edge expiry; authority changes during recall withhold old derivatives. | Revocation, assertion retraction, summary withdrawal and time-based edge expiry |
| R-002, R-003, R-017 | Assertions/contradictions and backend facts must resolve current authorized canonical support with source/window constraints. Missing or partly unresolved manifests fail closed. An archive/authority change during external recall clears stale derived sections and reauthorizes raw items. | Unauthorized assertion; concurrent hide; account/source/window bypass; authorized positive packet; mixed unresolved provenance |
| R-004, R-026 | Mental models require a configured owner and explicit owner approval. Candidate assertions retain only a pending replacement relationship; supersession occurs atomically with qualifying confirmation/publication. | Ownerless/non-owner refusal and owner success; candidate leaves standing claim intact until confirmation |
| R-020, R-021 | Worker mutation paths check original claim token/epoch within their write transaction. New job identity includes epoch; reopening cancelled/quarantined work resets generation fields. | Replaced claim rejected across submission, completion, release, retry, partial, uncertain and running; old-epoch work cannot revive; new-epoch re-enqueue is claimable |
| R-011, R-012, R-027 | Atomic handoff revalidates current consent/evidence/goal/expiry and lease before changing state. Evidence iterables are materialized once and bounded before persistence. Commands require an explicit positive receipt; invalid output raises uncertainty, never invented success. | Opt-out between lease/attempt causes zero sends; expired lease; generator manifest; oversized manifest refusal; invalid command output |
| R-015 | Host receipts must be nonempty, resolve sent artifacts and refer to that exact artifact's successful delivery. Delivery receipts cannot establish task/goal/lesson success. Repeated notes/actors about the same receipt do not multiply support. | Empty/unrelated receipt refusal; repeat receipt counted once; existing valid host receipt tests |
| R-008, R-023, R-024 | Shared local-export paging bounds envelope count/serialized bytes and retains content-bound per-file offsets (file, structured, WhatsApp, email). Changed files restart safely. MCP uses a content-bound local offset while replaying the unconsumed remote page; changed remote pages explicitly expire the cursor instead of silently losing the tail. | Twelve-row exports drain under two-record cap; five-message chat fully commits; MCP three-record response fully commits; changed local/remote pages; existing byte-bound suites |
| R-009, R-010, R-005 | File mtime is observational, not revision metadata; replay compatibility handles historical mtime/legacy fingerprints without rewriting evidence. Structured sample/series revisions derive from content. Fingerprints include occurrence precision and parent dependencies. | Touch replay; historical mtime; structured edits create new revisions; changed precision/parents conflict; compatible legacy replay |
| R-025 | Canonical preparation redacts recognized credential text and nested metadata before fingerprinting/persistence; receipts carry those sanitized values. Adapter envelope normalization also sanitizes text. | File/mail/WhatsApp imports; nested metadata and receipt checks; doctor still detects simulated legacy raw credentials |
| R-019, R-022 | Summary identity includes normalized citation manifest, account, coverage and epoch. Weekly scope is explicitly a seven-day interval starting on its named date. | Changed citations produce new identity; equivalent citation order is idempotent; week includes six-day endpoint, not preceding six days |
| R-014, R-016, R-013, R-018 | Lesson condition labels require explicit host task `conditions` booleans (required=true, exception=false; unknown withheld). Exhausted seconds budget blocks admission. Cancellation audits use JSON serialization. Raw packet item limits are enforced independently of candidate overfetch. | Missing conditions withheld/eligible positive case; exhausted seconds refuses; quoted reason round-trips; limit=1 returns one |

### Compatibility and operational boundaries

- New canonical schema: **15** (`0015_durable_fences`); `deployment/compatibility.json` agrees with the build. Opening an older supported database migrates transactionally. A source rollback after migration must not open schema 15 with an older binary; use a coherent backup and the supported restore/upgrade workflow.
- This pass changed framework core, migration/compatibility metadata, regression contracts and documentation. It did not edit Hermes host code, update an installed immutable release, restart services or touch production memory. The runtime revisions recorded above remain the last previously installed revisions, not this remediation.
- Facts lacking resolvable canonical provenance are now withheld, not shown as unsupported hints. This intentionally narrows permissive prior behavior; live backend payload/provenance and recall quality still need verification.
- Legacy lesson labels are not guessed from prose. A caller must supply explicit host-validated `task.conditions` values; conditional lessons with unknown labels fail closed. A successful send is not evidence that a lesson/task/goal worked.
- New ingress redaction follows the existing recognized-secret patterns. It is not a universal secret detector, retroactive text rewrite, credential rotation or binary attachment redaction. Existing historical plaintext should still be checked by doctor and handled through owner-approved erasure/rotation.
- Intra-page MCP replay requires a stable re-readable response. If the server consumes or changes it, the adapter reports cursor expiry and coverage cannot silently claim completeness. Local oversized individual records remain explicit gaps, not silently clipped evidence.
- Tests model concurrent state changes through separate store connections and injected hooks; a comprehensive multi-process crash/fault matrix and live gateway/model checks are still separate verification work.

## Suspected risks and verification gaps

- Live Telegram round trip, cross-bot/profile flows and general free-text clarification are not covered by offline inquiry tests.
- No live semantic relevance, model extraction quality or remote erasure proof is implied by this review.
- Full feature and branch coverage requires additional passes; the ledger will explicitly retain partial and unreviewed surfaces.
- Summary formation appears to rely on prompt-described windows/scope rather than proven backend evidence filtering. Verify citation/provenance laundering with injected out-of-scope backend results; this is not yet a confirmed additional bug.
- Budget/allowance admission and later charging need concurrent reservation tests; resource serialization alone may not establish an aggregate spending bound.
- Time-based identity expiry and pre-restore worker fencing now have regression proofs. Broader assertion retrieval validity, alias-confirmation races, restore quiescing across independent processes, source reconfiguration generations and all-channel packet size bounds remain additional audit surfaces.
- Media/Gmail/Hermes-event adapters, full contradiction/evaluation matrices, device routing/maintenance/resource failure paths, CLI/doctor/installer lifecycle and live gateway native/coded/free-text replies remain partial or unreviewed in this pass.
- Normal hidden/deleted evidence may intentionally retain historical audit text. Do not equate retention with ordinary retrieval access or claim universal secure deletion without checking the documented contract. Confirmed byte findings above target demonstrable attachment cleanup/reset/restore behavior.

## Review change log

- 2026-10-02: first broad source/flow pass; 32 confirmed open bug families, 39 offline reproduction cases. Added this register and test probes only; no production mutations, core fixes, installs or external sends. Dedicated reproduction run: **39 passed in 0.39s**. Complete framework suite with all probes: **3,015 passed, 1 skipped in 51.01s**. The green reproduction cases assert observed defects, not safe behavior. `git diff --check` passed.
- 2026-10-02: owner requested all confirmed core fixes. R-001–R-032 implemented, original defect assertions converted to safety contracts, and additional migration/positive/replay/expiry cases added. Final safety file: **65 passed in 0.80s**. Complete final framework suite: **3,041 passed, 1 skipped in 54.70s**. `git diff --check` passed. No deployment or production writes in this pass.
