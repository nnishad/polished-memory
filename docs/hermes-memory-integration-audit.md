# Hermes memory integration: understanding, requirements, and findings

Living investigation, started 2026-10-01. The findings below record the original installation; the implementation section tracks the source fixes and deployment authorized by the owner. No production data or credentials are included.

## Scope and evidence

- Hermes checkout: `/home/jugaadu/codes/hermes-agent`, revision `f97608f178d1ffeca59860195ab7da295f7c8e5f`.
- Framework checkout: `/home/jugaadu/Projects/hermes-memory`, HEAD `e26ae2309d73e969a7079372f529b0a4f1ca3148`. Existing uncommitted changes belong to the user and are preserved.
- Installed plugin: `/home/jugaadu/.hermes/plugins/hermes-memory`.
- Installed framework at investigation start: `/home/jugaadu/data/hermes-memory/runtime/current`, pointing to release `e26ae2309d73e969a7079372f529b0a4f1ca3148`. Current deployment is recorded below.
- Hermes profile: `/home/jugaadu/.hermes`; provider selection is `hermes-memory`, storage mode is `provider`, and `mcp_servers` is empty.
- Read-only service inspection found gateway, memory runtime, Hindsight API, and Hindsight worker active. Status reported 40 live, indexed records, 40 verified backend projections, and no pending capture events. This does not establish recall quality or current socket reachability.

Status vocabulary: **verified behavior** means a traced implementation; **confirmed defect** requires a concrete behavioral demonstration; **suspected defect** needs reproduction; **gap** means an uncovered requirement or capability; **design choice** is not automatically a bug.

## Installation and connection

Hermes's gateway runs the interpreter in `codes/hermes-agent/venv`. The plugin entry point registers a `HermesMemoryProvider` instance. Its `runtime.adopt()` resolves the framework package from the installed release before importing the provider implementation. Consequently, changing framework source does not change a running release or an already-imported process.

This is a native memory-provider integration, not an MCP integration. The plugin directly uses the release's framework library and profile-bound SQLite storage. Hindsight is a separate derived backend; the memory runtime exposes an admission/resource gate at loopback port 8123, while Hindsight serves on port 8888.

The primary archive is `data/hermes-memory/data/canonical.db`; turn capture first enters `data/hermes-memory/data/hermes-memory/capture-spool.db`. Background processing drains captured events into canonical evidence. Retrieval uses a context broker combining canonical lexical retrieval and derived channels. The installed profile bank is `hermes`.

## Investigation coverage (first pass completed)

| Area | Progress |
| --- | --- |
| Provider contract and manager | Read and mapped |
| Agent startup, turn capture, prefetch | Read and mapped |
| Native memory store and tool | Read core flags, tool dispatch, mutation and persistence boundaries |
| Compression, session switches, interruptions | Read checkpoint normalization, rotation calls, turn synchronization and rewind paths |
| Gateway, CLI, TUI/desktop, API teardown | Traced lifecycle entry points; API manager registry read; no live surface E2E |
| Session search and persistence | Read search contract and persistence metadata boundaries; no ranking benchmark |
| Skills, review forks, delegation, cron | Read native review fork, delegation notifications/blocked tools, cron initialization, procedural-memory boundary |
| Setup, migration, backup, dashboard | Read setup/clone/backup collectors and dashboard/native-edit entry points; no destructive operations run |
| Framework parity and defect reproduction | Offline installed-code probes reproduced failures below |

The companion [source inventory](hermes-memory-source-inventory.md) lists 124 matching production files. It is an entry-point discovery index, not evidence that every implementation branch was read. Large unrelated provider implementations and every desktop rendering component were not exhaustively reviewed.

## Initial requirements established by Hermes

The provider ABC is `agent/memory_provider.py`; orchestration is `agent/memory_manager.py`.

- Availability checks must be local and cheap, with a useful reason when unavailable.
- Initialization must bind the profile home, session, platform, execution context, and available identities.
- System prompt contributions must be static. Dynamic recalled evidence belongs in `prefetch()`.
- Prefetch must be bounded; the host timeout is eight seconds and the installed framework deadline is six seconds.
- Turn synchronization must preserve order, author attribution, and original conversation evidence.
- Session reset, resume, branch, compression rotation, and rewind require explicit handling.
- Pre-compression checkpoint API v2 supports durable capture and strict failure propagation before transcript loss.
- Native memory write mirroring must follow committed operations, including exact previous content for replacements and removals.
- Provider tool schemas must be valid, avoid reserved core names, and obey the session's memory toolset gate.
- Background tasks must inherit the caller's profile context; worker thread defaults are unsafe in a multiplexed process.
- Shutdown must flush and report abandoned work without hanging the host.
- Backup paths must resolve without initialization or network access.

## Findings register

The descriptions below are the **pre-fix baseline**, retained so each change has a
reproducible reason. They do not describe the corrected release's current behavior.

## Implementation and acceptance (2026-10-01)

Owner requested core fixes and reinstallation, without runtime patches. Changes are
in Hermes core and the framework source, not monkeypatches or edits to installed packages.

| Finding | Source resolution |
| --- | --- |
| F-001 | Cut an immutable release from a clean, locally committed build snapshot; reinstall the plugin at that same revision and restart live consumers. Preserve original source worktrees and pre-existing edits. Deployment receipt below records the outcome. |
| F-002 | Hermes recognizes `memory.store`: `both` (backward-compatible default), `builtin`, or `provider`. Provider-only requires a named external provider and disables both native snapshots and write surfaces. Builtin-only does not activate the configured external plugin. |
| F-003 | Fresh backup hook resolves the current enrolled profile read-only and returns canonical/spool directory plus enrollment, configuration, provider selection and gate state. No unregistered-profile fallback to another archive. Hindsight is a derived projection; framework full backup/rebuild rules remain separate from Hermes quick snapshots. |
| F-004 | Core allocates a capture ID before asynchronous dispatch and copies messages/author metadata. Provider uses explicit IDs for retry deduplication, plus generation/content identity for older hosts and unique IDs when no transcript is supplied. |
| F-005 | SDK ingests both delegation task and result with child linkage, marked nonindependent assistant evidence. |
| F-006 | Remove consumes exact `previous_content` (also accepts the older content-bearing shape). Replace/remove provenance is preserved through a whitelist that cannot forge trust metadata. Removal reports a note change; it is not evidence erasure. |
| F-007 | Native commits have transaction/operation identity rather than text-only identity. Explicit retry identities still deduplicate. |
| F-008 | Capture normalizes text parts in multimodal messages; it excludes synthetic compressed summaries and does not store image data as text. Required checkpoint failures remain fail-closed. |
| F-009 | Session end captures every eligible message at original position, with authors/timestamps and no 20-message/4,000-character cut. One durable event per message avoids oversized session batches. The canonical SDK's explicit per-record size limits still apply. |
| F-010 | Setup Hindsight default is 8888; binding rejects the model gate as the backend endpoint. |
| F-011 | Setup writes atomic profile `hermes-memory.json`, which binding reads. Legacy nonsecret profile `.env` settings are read for migration. Enrollment remains authoritative for storage/bank/scope; a setup `data_dir` is a proposal, not permission to relocate evidence. Profile routes cannot widen the instance's approved hosts. Host credentials use Hermes's context-local secret scope; setup names the configured key variable. |
| F-012 | Generic native-store post-commit observer covers normal tools, shared review stores, approvals and journey/UI edits using that store. Failed/staged/no-op writes emit no changes, and a mirror failure is reported without pretending the native commit failed. Manual file edits/imports outside the store API are not intercepted. Native reset remains native-only, not a custom archive erasure command. |
| F-013 | Intentional enrollment boundary retained. A cloned profile carries the JSON config, but must be explicitly enrolled before reading memory. It never silently inherits the source profile's archive. |
| F-014 | Failed binding raises; Hermes removes failed providers and their tool routes rather than claiming activation. Rebinding/shutdown closes the enrollment connection. |
| F-015 | Core wrapper describes recalled material as attributed evidence, not instructions or unquestionable facts. |
| F-016 | Turn-start clears prior reply state even when prefetch is skipped. Bot messages and mismatched authors cannot settle owner inquiries. Live messaging/inquiry E2E is still unverified. |
| F-017 | Fresh restore preserves snapshot erasure decisions, targets and tombstones, merged with newer live erasure history. The isolated rehearsal exposed this defect before production deployment. |
| F-018 | Generic Hermes plugin loader executes the entry point before eagerly loading remaining helpers. Release adoption now precedes framework imports; fresh installed-provider inspection confirms the selected release is actually loaded. |

Additional safety fix: native approval/journey stores disable writes when their
configuration cannot be read; they no longer fall back to enabled permissions.

Acceptance runs: final full framework suite **2,934 passed, 1 skipped**;
Hermes memory regression matrix **294 passed across 35 isolated files**. Subsequent
final tests and deployment verification are recorded in the deployment receipt.

Still not claimed: exhaustive Hermes audit, semantic recall benchmarks, real messaging
E2E, new unattended formation policy, or an automatic archive erasure operation for
native `/memory reset`. Raw disk edits are outside the native commit API; backups do
not imply restoration was tested unless the rehearsal result is recorded explicitly.

### Deployment receipt

- Framework deployment revision: `ec8977f442efbbb8209e8957cfa603bb25cb82b4`, built from the tested source in `/home/jugaadu/data/hermes-memory/build-source-20261001-OZEfsP`. Original source worktrees and pre-existing edits remain intact; no commits were made in either original repository.
- Official release staging built both runtime and Hindsight environments. `runtime/current` was switched atomically while services were stopped; older releases remain available.
- Plugin reinstalled through `hermes plugins install --ref ec8977f442efbbb8209e8957cfa603bb25cb82b4 --force --enable`. Framework wheel reinstalled into the Hermes virtualenv. Hermes core is an editable installation, so its source fixes are active after restart.
- All four services were restarted and subsequently confirmed active: gateway, memory runtime, Hindsight API and worker. Installed-provider inspection reports `from_release=true`, `matches_release=true`, and no release warning after the loader fix.
- Quiesced canonical snapshot `snap_85ac320cb897b4a5` is in `/home/jugaadu/data/hermes-memory/data/snapshots/20261001T181805.952414Z-snap_85ac320cb897b4a5`. Full instance and host-install archives are retained in owner-only `/home/jugaadu/data/hermes-memory-upgrade-backups/20261001-b0850c3`.
- Archive SHA-256: instance `462c73f093df63265cf9e7a3628ae7f8bb03bcd994cc54200d6cec3d0ad138c1`; host install `633c6972124cb186cf92fd91467c211af037ff35cc44272df40ecf833885f8b8`.
- Fresh isolated restore rehearsal passed with **40 live records, 3 tombstones and 13 migrations**. Production data was not restored or erased. The first failed rehearsal remains private alongside the corrected rehearsal for diagnosis.
- The installed synthetic probe passes: provider-only flags off, complete backup declarations, two same-length rewind captures, delegation evidence, removal/replacement provenance, repeated native adds, full 5,010-character session text with author, multimodal checkpoint and saved profile configuration propagation.
- Explicit compatibility test against the real Hermes provider ABC: **1 passed**. Final complete Hermes plugin suite: **1,817 passed, 5 skipped across 142 isolated files**. Final focused core regression suite: **68 passed across 7 files** (capture identity, native commits, provider-only permissions, startup, write bridge, approvals and disabled store). These overlap the earlier memory matrix and must not be added as independent totals.
- At the owner's request, both Hermes `model.base_url` (official configuration command) and the framework text route/approved inference host now use **`192.168.68.50:8080/v1`**. Local vision and embedding routes are unchanged.
- **External verification remains blocked:** two connection attempts timed out; routing selects `wlan0`, but neighbour resolution for `192.168.68.50` reports `FAILED`. No successful inference call or semantic retrieval benchmark is claimed.
- Doctor verified canonical schema, 40 indexed records/projections, 3 completed erasures and no capture gaps. It is not fully healthy: an existing uncertain remote execution reservation `wait_460672012efa43af991d65ac66df10dd` remains after `No route to host`; three delivered artifacts remain `accepted_unverified`. The gate was not bypassed and unknown execution was not declared finished without evidence. No unattended formation grant was added.

### Server address update (2026-10-01)

The owner subsequently supplied `192.168.68.69:8080` as the running server. Hermes's main model URL and the framework text route now both use `http://192.168.68.69:8080/v1`; the approved host list replaces `.50` with `.69`. Changes used the official Hermes configuration command and the instance's existing configuration file, followed by restart of all four services; all four report active.

Live checks succeeded: `/health` returned `{"status":"ok"}` and `/v1/models` advertised `Cobra91310/Ornith-1.5-9B-MTP-NVFP4-newhead:NVFP4`. This supersedes the earlier `.50` connectivity blocker, but is not an inference or recall-quality test. The existing uncertain remote execution reservation remains blocked; endpoint health alone does not establish that the earlier execution completed. No reservation was force-cleared.

### Live testing and fixes after server recovery (2026-10-01)

The owner authorized live tests and fixes with all models running. Results and source changes:

- The old reservation `wait_460672012efa43af991d65ac66df10dd` was settled **failed** through the official gate CLI, with an attributed reason. Its recorded urllib `errno 113` proves connection establishment failed; the new server's `/slots` also reported idle. The gate now reports no blocked or unresolved reservations.
- **F-019 — Failed connects stranded inference capacity.** `urllib_upstream` previously treated every transport failure as unknown execution. Core now marks only wrapped `ECONNREFUSED`, `EHOSTUNREACH` and `ENETUNREACH` as a proven failed connection and releases that reservation as failed. Timeouts, resets, broken pipes, generic errors and gate faults remain uncertain. Seven new regression cases plus a real refused local socket verified the distinction. No string-based automatic clearing of old reservations was added.
- **F-020 — Configured vision route lacked admission credentials.** The local multimodal server was running and advertised the configured model, but `vision` was withheld because its route credential was absent. A private, randomly generated credential was added to the owner-only instance configuration and the gate reloaded. Credentials are never printed by the probe or recorded here.
- Live authenticated gate checks pass: remote text replied `OK`, vision identified a synthetic red PNG as `red`, and embeddings returned a 1,024-dimensional vector. The repeatable [model probe](hermes_memory_model_probe.py) sends only synthetic input and no personal evidence.
- Official doctor synthetic test passed retain → derivation → recall in `hermes-doctor-probe`, with one derived unit and one recalled result. Its cleanup deletes the synthetic document; no personal canonical evidence is added by this test.
- **F-021 — Worker raced the API for pg0 ownership.** A cold restart reproduced API startup failure: `Instance already running`, while the worker had opened connections. API-alone startup succeeded. Despite comments promising one owner, the launcher constructed a default `MemoryEngine` with `database_url=pg0`, allowing the worker to start/stop the API's database. Core worker now waits for API health, reads the named pg0 instance's running DSN without starting it, and passes an explicit PostgreSQL URL with migrations disabled. Readiness has a bounded wait; failure refuses launch rather than taking ownership. Two regressions verify the wait and explicit DSN propagation.
- Transport fix was staged and installed as `c19ab0f809dba317cd216251b844fef810221618`. A new canonical backup `snap_dc7ff8099773956c` verified 40 records and 3 tombstones. The combined worker/transport release is `bb72c84a82bbcd6e80d268b6229bdc3420ec0c26`; final deployment and cold-start verification follow below.
- Full framework suite after transport fix: **2,941 passed, 1 skipped**; worker contracts after lifecycle fix: **57 passed**. These counts overlap, not independent totals. Original worktrees and user changes remain preserved.
- Remaining unrelated warning: three previously delivered artifacts are still `accepted_unverified`. Model connectivity does not prove delivery receipts; tests did not resend them or fabricate confirmation.

Final deployment receipt: the combined `bb72c84a82bbcd6e80d268b6229bdc3420ec0c26` release was installed through the release builder and pinned plugin installer; the pointer was switched with services stopped, and the framework wheel reinstalled in Hermes's virtualenv. Original and intermediate releases remain available. A **simultaneous cold start** of API and worker now succeeds, `/health` reports a connected database, and all four services are active. Fresh Hermes provider import confirms the exact release with `from_release=true`, `matches_release=true` and no warning. Text, actual image understanding and embeddings pass again after deployment. Final doctor reports all inference routes and gate healthy; synthetic retain/derive/recall passes with one unit and one result, cleaned up. Its only remaining warning is the three historical unverified deliveries. Final full framework suite: **2,943 passed, 1 skipped**.

Live worker verification also passed: durable synthetic submission `be9ff9d4-a3ce-4669-b42a-299b9523bc64` completed in isolated `hermes-worker-probe`, produced one unit, recalled one result, and its executable child was recorded `finished` by `hermes-memory-worker-default`. The [worker probe](hermes_memory_worker_probe.py) deletes its synthetic document after terminal completion. An earlier probe incorrectly looked for the batch parent's ID in the executable-worker ledger; inspection showed its child had completed and was correctly accounted. The probe was corrected to check child identities, not the framework. Both synthetic documents were cleaned up. Final gate reading: no held slots, blocked resources or unresolved reservations; all services remain active.

### Earlier restore and loader defects

**F-017 — Fresh restore dropped erasure history.** Recovery originally carried the destination's live erasure ledger but did not load the snapshot's history into a fresh destination. The backup rehearsal retained records but lost tombstones. Core recovery now unions snapshot and live history by primary key, with current live rows winning, and rebuilds dependent rows in foreign-key-safe order. A regression test and successful isolated restore verify the correction.

**F-018 — Plugin helper loading preceded release adoption.** The generic loader eagerly imported sibling files before executing `__init__.py`, letting a plugin client import the ambient framework package before `runtime.adopt()`. Core now executes the entry point first, preserving normal relative imports and subsequent helper exposure. A bootstrap-order regression and fresh installed-provider release identity check verify the correction.

## Original findings

### F-001 — Source checkout and active release are separate

**Verified deployment behavior.** The framework checkout contains changes not carried by the installed plugin/release. Treat source, installed plugin, release package, and live imported process as four distinct observations when diagnosing behavior. Evidence: installed `runtime.py`, `RELEASE.json`, plugin install metadata, and checkout status.

### F-002 — Native and external memory are different contracts

**Verified compatibility mismatch, not an established upstream bug.** Hermes's external provider is additive. `tools/memory_tool.py:get_builtin_memory_store_flags` only reads `memory_enabled` and `user_profile_enabled`. With the installed config's `store: provider` and both booleans true, the probe returned `(True, True)`. `agent/agent_init.py:_init_memory` therefore constructs the native `MemoryStore`; `agent/system_prompt.py:_memory_parts` renders both native snapshots and the provider's static block. The native `memory` tool stays available alongside `memory_remember`.

This corrects the earlier conversational claim that provider storage selection replaced native memory. In this checkout `store: provider` does not achieve that. The provider's own comments deliberately say native notes remain authoritative. A decision is needed about whether coexistence is intended, and which layer owns each fact. Do not disable native flags blindly: review forks, memory nudges and native UI features depend on them.

### F-003 — Fresh-instance backup hook omits the custom archive

**Confirmed defect; high priority.** `hermes_cli/backup.py:_collect_memory_provider_external_paths` loads a provider and directly calls `backup_paths()` without `is_available()` or `initialize()`. A fresh `HermesMemoryProvider` has `_settings`, `_activity`, and `_spool` unset; installed `provider.py:backup_paths` returns `[]`.

**Demonstration:** the installed-code probe returned `fresh_provider_backup_paths: []`. The contract explicitly requires this method to work without initialization/network. Canonical evidence outside `.hermes` therefore is not declared through this backup hook. No production backup archive was created or inspected; this proves the declaration omission, not the contents of existing user backups.

**Requirement:** resolve the owning profile read-only on the backup path. Back up the canonical archive, spool, necessary enrollment/configuration/erasure state, and either back up derived backend state or document rebuildability. Prove restore into an isolated installation. Quick snapshots use their own fixed file list and also must not be assumed to cover an external archive.

### F-004 — Capture IDs collide after rewind or missing transcript length

**Confirmed defect; high priority.** Installed `provider.py:sync_turn` uses `turn:<session-id>:<len(messages)>`. `spool.py:append` uses `INSERT OR IGNORE`. A new exchange at the same transcript length after `/undo` shares the old exchange's identity; `on_session_switch(rewound=True)` increments only the cache generation, not the capture identity.

**Demonstration:** two different synthetic exchanges, same session and length, with a rewind between them, stored one row containing the first exchange. Without `messages`, every turn in a session also uses length zero. Other paths, such as in-place compaction, need their own regression cases.

**Requirement:** stable replay identity must distinguish genuinely new exchanges and transcript generations while deduplicating retries. Returning successfully from capture must not conceal a conflicting payload.

### F-005 — Delegation capture has no consumer handler

**Confirmed defect; medium/high priority.** Hermes's `tools/delegate_tool_results.py:_notify_memory_manager` calls the parent's `on_delegation`. The custom provider appends `kind: delegation`. Framework `sources/sdk.py:_HANDLERS` handles only `conversation_turn`, `pre_compress`, `session_end`, and `native_memory_write`.

**Demonstration:** adapting the installed provider's delegation event returned zero envelopes with `unsupported event kind delegation`. The source SDK reports a skipped event/gap. The final parent response may independently contain the child's conclusions, but the dedicated task/result/child provenance event is not ingested.

**Requirement:** retain task, result, parent/child session linkage, origin and completion status. Treat a child's result as assistant-derived evidence, not an independent owner statement.

### F-006 — Native removal cannot be ingested; replacement provenance is lost

**Confirmed defects; high priority.** `MemoryManager.notify_memory_tool_write` sends an empty new `content` for `remove`, with the exact removed entry in `metadata.previous_content`. The provider spools that shape. `sources/sdk.py:_native_note` requires nonempty `payload.content` and never carries through `payload.metadata`.

**Demonstration:** removal produced zero envelopes with `the native note carried no text`. Replacement produced one envelope, but its previous-content provenance was absent. This contradicts the provider's comment that removal is filed as a report and that replacement records what it replaced.

**Requirement:** capture mutation events as mutation events, including previous entry, new entry, origin and identifiers. Supersede a curated assertion when appropriate. Removing a curated note and erasing all supporting evidence remain distinct operations.

### F-007 — Identical native actions across sessions are collapsed

**Confirmed defect; medium priority.** Installed `on_memory_write` identifies an event by target, action and a digest of content/previous content, excluding session, operation or occurrence. Add → remove → add of the same fact reuses the first add's key.

**Demonstration:** two identical adds with an intervening removal and different session metadata stored one add event. A factual deduplication policy is useful, but an operation audit must retain separate occurrences. A replay token should describe one operation, not every operation with matching text.

### F-008 — Strict multimodal checkpoint can succeed with no evidence captured

**Confirmed defect; high priority.** Hermes's `_direct_messages_for_pre_compress_memory` preserves a user message's list-valued content. Installed `on_pre_compress` skips any non-string body, including a list that contains text. `require_checkpoint=True` does not check whether eligible evidence was omitted.

**Demonstration:** a user message containing one text part returned successfully from a required checkpoint and added zero spool events. This is text loss in the checkpoint path, not a claim that images themselves must always be archived. Normal completed-turn capture flattens multimodal text, but checkpointing must also work for an uncompleted turn.

**Requirement:** normalize text parts, identify attachments separately, and either guarantee the promised evidence handoff or refuse strict checkpointing with a useful reason.

### F-009 — Session-end fallback silently truncates and loses authorship

**Confirmed information loss; medium priority.** Installed `on_session_end` stringifies content, takes the final 20 user/assistant messages, clips each to 4,000 characters, and drops original author metadata. The payload does not mark this clipping. SDK metadata then cannot reconstruct original positions, individual authors or lost text.

**Demonstration:** a synthetic 5,010-character message became 4,000 characters with no truncation marker and no per-message author. Per-turn capture can cover some omitted history, but a session-end fallback is not a full recovery guarantee—especially for interrupted turns that normal synchronization intentionally skips. Multimodal content is stringified rather than normalized.

**Requirement:** distinguish a bounded session summary from a complete transcript checkpoint, record explicit coverage/truncation, and preserve author/role/original-position metadata.

### F-010 — Setup schema points Hindsight at the resource gate

**Confirmed default mismatch; medium priority.** Installed `get_config_schema()` advertises `hindsight_url=http://127.0.0.1:8123`. The deployed backend and deployment templates use port 8888. `processing/gate_server.py:FORWARDED_PATHS` only forwards `/v1/chat/completions` and `/v1/embeddings`; it does not serve Hindsight bank/retain/recall APIs.

**Demonstration:** the probe reported the 8123 default. The current installation uses 8888, so it avoids this mismatch. Users accepting setup defaults can receive a valid-looking private endpoint that cannot answer Hindsight requests.

### F-011 — Provider setup saves a file the runtime does not read

**Confirmed configuration propagation defect; high priority for setup UX.** `provider.save_config` writes `<hermes-home>/hermes-memory.env`. Provider `is_available()` calls `load_settings()` with no file; that reads `<HERMES_MEMORY_HOME>/hermes-memory.env`. `client.bind` uses those instance settings and `Profile.scoped` overlays the enrollment ledger's paths/bank/credential scope, not the provider-saved file.

**Demonstration:** with an isolated instance home, `save_config` wrote a profile file containing a synthetic Hindsight URL and data directory; a subsequent `load_settings()` returned no URL and did not use that data directory. No production configuration was touched.

**Requirement:** decide which settings are instance-wide and which are profile-wide. The wizard/dashboard must write to the authority the runtime reads or state clearly that it produced only a proposal. An arbitrary profile data path must not bypass reviewed enrollment.

### F-012 — Native review, approval and UI edits bypass the provider mirror

**Verified integration gaps; not all are established defects.** Primary-agent native tool dispatch mirrors successful changes through `agent/inline_tool_executors.py:_memory`. Other writers do not use that path:

- `agent/background_review.py:build_cache_parity_fork` uses `skip_memory=True`, rebinds the parent's native store, and does not attach its memory manager. Reviews can update native notes without provider callbacks.
- `hermes_cli/write_approval_commands.py:_apply_one` applies staged native changes directly. The initial staged tool result is deliberately not mirrored, and the approval handler contains no provider notification.
- `agent/learning_mutations.py:_mutate_memory` directly mutates native files for journey edits/deletes.
- `hermes_cli/web_routers/ops.py:reset_memory` deletes native memory files; it does not clear canonical/Hindsight evidence.
- `hermes_cli/agent_import.py` imports/merges native files, independently of provider capture.

**Implication:** the custom archive cannot be treated as a complete mirror of native current state. UI reset is a native-file reset, not global forgetting. No UI deletion, review model call or approval mutation was run during this audit.

### F-013 — Cloned Hermes profiles need separate framework enrollment

**Verified lifecycle gap/design choice.** `hermes_cli/profile_memory_config.py:clone_memory_provider_config` copies `<provider>/` and `<provider>.json` conventions. The custom plugin writes a flat `hermes-memory.env`, uses an external installation ledger, and must be installed/discoverable in the new profile. Copying `memory.provider` does not enroll that home.

The custom framework deliberately refuses an unenrolled home. That isolation is correct; clone readiness must explain and surface it. Do not copy the existing archive/bank into a new profile implicitly. Full clone/import/delete profile E2E remains untested.

### F-014 — Provider availability and initialization success can disagree

**Verified reporting gap.** Custom `is_available()` checks instance configuration but intentionally defers enrollment to `initialize()`. An unenrolled `initialize()` stores a binding error and returns. Hermes's manager suppresses initialization errors and agent startup can log the provider as activated anyway. Tools remain registered even when no archive is bound. Actual reads return a binding error rather than another profile's data, which is correct, but activation and UI readiness do not express this state.

### F-015 — Recall wrapper elevates evidence to authoritative reference data

**Verified trust-policy mismatch; exploit not demonstrated.** `MemoryManager.build_memory_context_block` calls recalled content "authoritative reference data" that should inform all responses. The custom provider's static guidance says results are attributed evidence, not instructions. Native files undergo threat scanning; provider context fencing strips wrapper tags but is not equivalent to fact verification or content threat scanning.

**Requirement:** source quotations, model-derived claims, confirmed preferences and executable instructions need separate trust. Preserve provenance and uncertainty through the entire prompt, not just in the database. This audit did not perform a prompt-injection evaluation.

### F-016 — Owner reply handling needs per-turn identity/freshness verification

**Suspected defect; not reproduced against live inquiries.** Installed provider inherits the no-op `on_turn_start`, while `_owner_reply` is cleared only when `_settle_from_turn` runs through prefetch. Hermes skips prefetch on trivial/slash prompts. This creates a path where an earlier reply remains in provider state across a later turn. The reply handler compares against that state. The provider also checks destination chat but does not explicitly bind per-turn author identity to the decision path.

The configured private destination reduces exposure; it is not evidence of a group-chat vulnerability. Verify with synthetic inquiry state, quoted replies, declines, successive trivial turns, bot messages and session switches before calling this a confirmed defect. Existing uncommitted owner-reply/delivery changes must be reviewed before proposing fixes.

## What Hermes means by memory

Hermes does not have one storage abstraction covering all learning and recall. These layers coexist:

| Layer | What it retains | Hermes implementation | Custom provider's role |
| --- | --- | --- | --- |
| Curated personal notes | Environment facts, conventions and essentials | `tools/memory_tool_store.py`, `memories/MEMORY.md` | Mirrors some committed writes; does not replace the file store |
| Curated user profile | User preferences, persona, communication expectations | `memories/USER.md`, same native store | Query-time evidence is available, but no dedicated stable profile block is supplied |
| Episodic history | Messages, tool calls/results, session metadata and lineage | `hermes_state*.py`, `agent/session_persistence.py`, `tools/session_search_tool.py` | Captures a subset into evidence; not a replacement for the session database |
| Working context | Live transcript, API sidecars, compressed handoff and task state | `agent/turn_context.py`, `conversation_compression.py`, `context_engine.py`, `tools/todo_tool.py` | Recall enrichment and pre-compression user/assistant checkpointing |
| Procedural memory | Reusable instructions, workflows and skill refinements | `skills/`, `tools/skill_manager_tool.py`, curator and background review | Framework lessons are a separate mechanism; no automatic interchange with Hermes skills |
| Identity/instruction context | Agent personality and workspace rules | `SOUL.md`, `AGENTS.md`, `.hermes.md`, `.cursorrules`, prompt builder | Not provider facts; these retain their own loaders and trust rules |
| Prospective task state | Scheduled jobs, todo tasks and kanban state | `cron/`, `tools/todo_tool.py`, kanban providers | Framework goals/reminders use a distinct ledger and delivery drain |

A RAM-pressure monitor, token usage counter or HTTP credential store is not persistent personal memory even if its implementation contains the word "memory". Those are outside this audit.

## Hermes call paths and resulting requirements

| Trigger | Hermes path/symbol | What a compatible memory implementation must do |
| --- | --- | --- |
| Agent construction | `agent/agent_init.py:_init_memory`, `_memory_provider_init_kwargs` | Load static config; bind profile/session/platform/context; preserve startup performance; expose valid tools |
| Provider discovery | `plugins/memory/__init__.py:load_memory_provider` | Work as a user plugin; honor disabling; tolerate fresh instances for inspection; avoid dependency/service installs on import |
| System prompt build | `agent/system_prompt.py:_memory_parts` | Supply static instruction text; snapshot changes only at supported boundaries; avoid dynamic recall in the cached prefix |
| Turn start | `agent/turn_context.py:_memory_turn_start_and_prefetch` | Receive per-turn author/context; handle greeting/slash prefetch skips without stale state |
| Automatic recall | `MemoryManager.prefetch_all`, `_prefetch_provider` | Return relevant bounded attributed evidence; host waits at most eight seconds; preserve local fallback and indicate what was actually injected |
| Model tool call | `agent/inline_tool_executors.py`, `MemoryManager.handle_tool_call` | Return structured JSON, validate arguments, honor memory toolset and execution-context permissions |
| Native memory write | `_memory`, `notify_memory_tool_write` | Mirror only committed mutations; support atomic batches, full previous entries and provenance; ignore staged/failed writes |
| Completed turn | `run_agent.py:_sync_external_memory_for_turn` | Capture flattened user/assistant text and author; accept full messages as optional context; retain FIFO ordering |
| Interrupted turn | Same synchronization gate; surface teardown | Do not promote partial output as completed truth; provide a clearly marked recovery path if transcript evidence is retained |
| Background warm recall | `MemoryManager.queue_prefetch_all` | Cache by session/query/revision and invalidate on lifecycle changes; do not insert the previous question's answer into a different question |
| `/new` or reset | `hermes_cli/cli_session_mixin.py`, manager `commit_session_boundary_async` | Finish old capture then rebind atomically relative to queued writes; return control without backend wait |
| Resume/branch/undo | CLI session mixin, TUI prompt methods, manager `on_session_switch` | Respect parent/child lineage and rewinds; preserve historical evidence while distinguishing replacement turns |
| Compression | `conversation_compression.py:_pre_compress_memory_context` | API v2 receives direct user/assistant evidence; excludes summaries, system/tool rows; strict mode must fail if checkpoint cannot be guaranteed |
| Compression rotation | Compression `on_session_switch` calls | Rebind the active session correctly after rotation or in-place compaction; invalidate derived caches |
| Delegation finishes | `tools/delegate_tool_results.py:_notify_memory_manager` | Record the parent-side task/result with child provenance; children normally have no provider and cannot write native memory |
| Background review or `/refine` | `agent/background_review.py` | Respect detached review scope; understand native-memory/skill writes are separate from provider capture |
| Cron execution | `cron/scheduler.py` agent construction | Permit recall; context is `cron`, and the custom provider rejects automatic capture and writing tools; native store is still present |
| Gateway expiry/eviction/shutdown | `gateway/run_agent_cache.py`, `run_shutdown.py` | Execute flush/end/shutdown under the owning profile; be idempotent and bounded; handle cached agents and interruptions |
| CLI/TUI/desktop teardown | `hermes_cli/cli_shutdown.py`, `tui_gateway/session_lifecycle.py` | Commit correct transcript under correct profile even on force quit; tolerate repeated cleanup |
| API request continuity | `gateway/platforms/api_server_memory_sessions.py` | Keep provider manager state by `(profile home, session id)` across fresh request agents; exclusive checkout and bounded eviction |
| ACP use | `plugins/memory:import_memory_provider_module`, ACP agent lifecycle | Tolerate warm-up import on main thread and construction later; never infer profile/workspace from process cwd |
| Native approvals/journey/reset/import | Approval commands, learning mutations, dashboard ops, agent import | Surface limits of native-only changes; arrange explicit synchronization if full mirroring is promised |
| Backup/export/import | `hermes_cli/backup.py`, dump and profile utilities | Declare outside-home state without initialized provider; preserve WAL safely; distinguish canonical restore from rebuilding derived backend |
| Setup/config/profile clone | Memory setup, provider config dashboard, profile memory config | Persist settings to the actual authority; report enrollment readiness and separate profile-owned state from shared machine resources |

### Runtime limits and invariants

- The host allows at most one external memory provider. The manager supports a native provider concept, but this agent startup path keeps the native `MemoryStore` separately and registers only the selected external plugin.
- Native notes use character budgets, defaulting to 2,200 and 1,375 characters. Writes support add/replace/remove and atomic batches, with locking, drift detection and threat scans. Replacements replace the whole selected entry.
- The native system-prompt snapshot is frozen while disk/live tool state can change. Compression or an explicit supported refresh can reload volatile prompt state; ordinary recall must not rewrite prior prompt bytes.
- Automatic recall is appended to this turn's API-facing user content. The clean transcript and exact API replay bytes are separate (`content` versus `api_content` sidecar). New memory capture must not ingest recalled context as fresh owner evidence.
- The host's prefetch worker has an eight-second bound; a timed-out worker continues and subsequent recall skips that provider until it finishes. A timeout is not cancellation or proof the backend did no work.
- The host serializes queued writes and warm prefetch on one worker, with contextvars copied into background work. Shutdown gives the queue five seconds after stopping submissions; some surface callers attempt an additional ten-second flush first. Success shown to the user is not proof that asynchronous capture has committed.
- In checkpoint-required mode, an API v2 provider must complete a durable checkpoint or compression is refused. `compression.checkpoint_required` is opt-in; advertising v2 alone does not make every compression strict.
- Session search accesses actual session DB messages without a model call. It supports discovery, scrolling, reading and browsing, deduplicates compression lineage, demotes cron, and excludes subagent/tool/kanban sessions from normal discovery.
- Provider schemas and static provider guidance obey the memory toolset gate. Capture/provider initialization are not the same gate: disabling tools is not proven to disable all background recording or recall.

## Framework capability assessment

| Capability Hermes needs | Current custom integration | Remaining requirement/gap |
| --- | --- | --- |
| Persistent preference/environment facts | Canonical evidence, explicit remember, lexical and Hindsight recall | Decide native/provider ownership and supply predictable retrieval for preferences |
| Automatic conversational capture | Local durable spool and background consumption | Fix collision, multimodal, truncation and event-handler defects |
| Author/source/session provenance | Turn author, evidence IDs, profile/bank binding | Preserve provenance through every adapter; verify group/multi-author flows |
| User/assistant distinction | Turn adapter emits separate roles | Session-end and checkpoint paths need equal fidelity; model claims stay derived |
| Bounded contextual recall | Broker, 1,200-token target, local fallback, six-second derived timeout | Benchmark relevance, fresh corrections, pronouns/follow-ups and failure latency |
| Stable cached prompt | Static provider block and per-turn packets | Align host wrapper trust and native/provider guidance |
| Write/update/remove | Remember and native-event mirror | Correct mutation handling; authoritative current-state versus historical evidence |
| Compression resilience | v2 local checkpoint hook | Multimodal strict guarantee and coverage; do not promise tool-result capture this host omits |
| Lifecycle and profile isolation | Enrollment registry, scoped archive/bank, session generation | Clone/restore readiness; initialization reporting; full A→B→A E2E |
| Delegation/task outcomes | Parent-side spool callback; lessons/outcomes framework | Consumer missing; Hermes skills/review loop not connected to framework lessons |
| Backup/restore | Backup hook plus framework operation paths | Fresh-instance hook broken; isolated archive/ledger/backend restore proof missing |
| Operator visibility | Status tool, CLI status/doctor, projection/gate ledgers | Distinguish configured/available/bound/indexed/retrievable; add cross-layer host visibility |
| Forgetting and correction | Explicit request, owner decisions, erasure/tombstone obligations | Native/UI deletion is not erasure; stale assertions and cache retirement require end-to-end proof |
| Prospective memory/reminders | Goal candidates, due events, outbox, approved transport | Distinct from Hermes cron/todo/kanban; avoid duplicate delivery and incomplete proof |
| Owner decisions | Identity/assertion/lesson/forgetting/goal inquiries | Validate reply freshness and authentic per-turn source; avoid conflating attention with memory reads |

Contradiction handling, source connectors (files/email/WhatsApp/structured/MCP), identity resolution, summaries, erasure obligations and resource scheduling are framework capabilities beyond the minimum Hermes provider ABC. Hermes does not automatically use every capability merely because the framework implements it. Treat "library exists", "plugin exposes it", "host invokes it", "source event survives ingestion", and "model receives useful evidence" as separate acceptance criteria.

## Requirement acceptance checklist

These are requirements derived from actual host behavior; items are not claimed to pass unless measured.

1. Start a CLI/gateway/TUI/API session with correct profile, stable prompt and available tools. Unenrolled profiles clearly report unusable memory without leaking another archive.
2. Commit distinct completed turns, repeated same-text turns, no-message callers, rewinds, branches and in-place compaction without loss or false duplicate suppression.
3. Preserve role, author, time, session, parent lineage, original text and explicit truncation/provenance across every event producer/consumer pair.
4. Recall relevant preference/fact/history evidence inside the deadline under backend offline/busy/stale conditions; clearly distinguish no match from incomplete coverage.
5. Update/remove/batch curated facts without leaving the old version presented as current; preserve historical evidence and support separately confirmed erasure.
6. Checkpoint eligible direct evidence before compression; multimodal text and interrupted turns cannot yield false success. Tool-result archival is a separate contract from the host's v2 message filter.
7. Keep provider/tool visibility, native writes, review forks, cron and delegation permissions consistent. Verify memory toolset disabling separately from capture policy.
8. Carry source configuration through setup/restart/profile clone/import and report the exact loaded release. Never assume filesystem source edits updated a live process.
9. Backup and restore canonical evidence, spool, enrollment, tombstones/config and derived-backend recovery into an isolated home; verify fresh provider discovery.
10. Make operator status/doctor and user UI reflect the same archive. A reset/forget/delete operation must say which stores it affects and what remains retrievable.

## Original baseline validation and limitations (before fixes)

The supporting [offline probe](hermes_memory_audit_probe.py) imports the installed plugin and installed release, uses synthetic strings and a temporary SQLite spool, and prints metadata-only outcomes. It runs no backend/model/network calls and never initializes against the user's canonical archive. Its setup propagation probe writes only under its temporary directory. Run it with:

```sh
/home/jugaadu/codes/hermes-agent/venv/bin/python \
  /home/jugaadu/Projects/hermes-memory/docs/hermes_memory_audit_probe.py
```

Original measured outputs (the current installed probe now passes the corrected contracts):

| Probe | Result |
| --- | --- |
| Provider storage selection vs native flags | Both native flags true |
| Fresh provider backup declaration | Empty list |
| Distinct exchange after same-length rewind | Two attempted; one stored |
| Dedicated delegation event | Zero envelopes; unsupported event kind |
| Native remove event | Zero envelopes; missing-text refusal |
| Native replacement provenance | Previous content not preserved in envelope metadata |
| Add/remove/add repeated native text | One add occurrence stored |
| Session-end long text/author | 5,010 → 4,000 chars; clipping unmarked; author absent |
| Strict checkpoint with list-valued text | Successful return; zero events |
| Setup backend URL default | Port 8123 rather than backend 8888 |
| Saved profile settings read by instance loader | File written; URL/path not adopted |

The required Hermes runner was attempted for memory-write bridging, pre-compress handoff, disabled native memory, asynchronous sync, session switching and cron memory contracts. It stopped before collection because neither Hermes virtualenv had `pytest`. No dependencies were installed. Existing tests were inspected as evidence of intended contracts but are not reported as passing.

At this initial read-only stage, no production writes, erasures, profile mutations, model calls, backend probes, service restarts or fixes had been performed. Subsequent authorized implementation and deployment are recorded above. Exhaustive bug coverage, semantic recall quality and full multi-surface integration remain unclaimed.

## Gateway clarification review — 2026-10-01

The subsequent exploratory review is recorded in [hermes-gateway-clarification-review.md](hermes-gateway-clarification-review.md). It traces the existing partial text bridge and proposes a durable, profile-bound gateway conversation protocol. Eight additional defect classes (C-001–C-008) were reproduced offline, including negated approval, concurrent delivery leases, stale-state approval and missing command profile binding. At that review stage these findings were not fixed or deployed. Their subsequent authorized implementation and deployment are recorded below.

Targeted existing tests plus 12 synthetic reproduction cases: **205 passed, 1 skipped**. The reproduction tests intentionally demonstrate current defects; passing them is not a safety certification. No live Telegram messages, model calls, production archive changes or restarts were performed for this review. Recommended priority is deterministic authorization and atomic state fencing, followed by durable gateway correlation and contextual receipts.

## Clarification core fixes and deployment — 2026-10-02

Implemented in the recommended order: explicit authorization grammar and atomic subject/state fencing, then durable delivery tracking, exact native reply correlation, and deterministic context injection into the Hermes turn. The source implementation and regression coverage are detailed in the [clarification review](hermes-gateway-clarification-review.md).

- Negated or ambiguous prose cannot authorize a decision. Nested store writes use real SQLite savepoints, so subject mutation, inquiry settlement and audit receipt commit or roll back together.
- Delivery attempts are durable before transport I/O. Unknown outcomes are quarantined rather than automatically resent after a lease timeout; stale acknowledgements cannot overwrite a newer inquiry generation. Reopened generations clear obsolete delivery and answer fields.
- The command transport uses the enrolled Hermes profile home. Telegram native reply correlation requires an exact acknowledged platform/chat/message/thread/profile mapping, authenticated owner DM provenance and the current inquiry generation. Missing identity, forwarded/internal events, wrong profiles and uncorrelated short answers fail closed. Forgetting still requires the explicit confirmation code.
- Hermes passes a generic immutable inbound envelope through normal and queued turns. The plugin settles verified replies at turn start and supplies a bounded question/subject/outcome dossier to the current model-facing user turn, including when retrieval is skipped. Prefetch is read-only; the static prompt and old conversation prefix are not rewritten.

Deployment receipt:

| Item | Verified result |
| --- | --- |
| Runtime and plugin revision | `237133b492abed552e4567652b1f218f87bbf75b` |
| Active runtime | `/home/jugaadu/data/hermes-memory/runtime/current`, pointing to the revision above |
| Framework source digest | `cbfd4914ac4b9135b4fecbcff20c9e036d0f274f1124c62cd825dce03d5118ab` |
| Canonical schema | 14/14; SQLite integrity check passed |
| Preserved canonical state | 40 live records and 3 tombstones |
| Services | Gateway, memory API, worker and Hindsight all active after restart |
| Loaded provider | Installed plugin imports the active release; revision matches, no loader warning |
| Model checks | Foreground and vision probes passed at `192.168.68.69:8080`; embedding probe returned 1,024 dimensions |
| Doctor | `ok: true`; sole warning is 3 pre-existing historical `accepted_unverified` deliveries, retained unchanged |

Safety copies were taken before activation: official snapshot `snap_943e437a96896b33` (schema 13) and an owner-private coherent activation backup at `/home/jugaadu/data/hermes-memory-upgrade-backups/clarification-activation`. An isolated restore rehearsal verified migration and the same 40/3 counts. Previous runtime releases remain available. Downgrading after schema migration requires restoring the coherent backup, not pointing an old binary at the schema-14 store.

Validation:

- Complete framework suite: **2,976 passed, 1 skipped**.
- Broad Hermes gateway run: **8,758 passed, 2 failed, 39 skipped**. Both failures exposed missing-author test fixtures; the core helper was corrected to treat absent identity as unauthenticated. A focused 13-test rerun, including both failures and the new contracts, passed. The entire broad suite was not rerun after that correction.
- The real Hermes/plugin/temporary-canonical A→B→A profile-isolation probe passed against source and against the installed release. Its repeat A reply is idempotent. This is an offline native-envelope test, not proof of a live Telegram API round trip.
- Both source repositories pass `git diff --check`. Original dirty source worktrees were preserved; build-only commits supply the installable release.

Remaining boundaries: no live Telegram test messages were sent. Short native answers are supported only for exact owner-DM replies received through the owning profile; a different receiving bot/profile fails closed. This implements existing memory decision inquiries, not a general free-text clarification schema or an autonomous gateway wake-up service. Uncertain delivery needs operator reconciliation; withdrawal is not a guarantee that an unchanged terminal proposal can be reopened automatically. Semantic recall quality and broader lifecycle tests remain separate work.

## Original prioritization and remaining verification

Original priorities, now implemented where described above:

1. Preserve capture/restore guarantees: F-003, F-004, F-006 and F-008.
2. Repair configuration propagation and backend defaults: F-010 and F-011.
3. Complete consumer/provenance paths: F-005, F-007 and F-009.
4. Resolve intended native/provider coexistence and the review/approval/UI boundaries: F-002 and F-012.
5. Verify multi-profile clone/restore/init and reply-state behavior: F-013, F-014 and F-016.
6. Run host regressions once a test environment exists, then benchmark retrieval and test corrected facts/erasure through canonical, Hindsight, cached packets and actual prompt replay.

Pre-existing uncommitted framework changes were preserved and included in the tested deployment snapshot. Remaining work requires a reachable inference host, then semantic recall/correction/erasure benchmarks and live multi-surface lifecycle tests. Source fixes and isolated tests do not establish those results.

## Subsequent framework-wide review — 2026-10-02

The initial review-only [feature-flow bug register](memory-framework-feature-review.md) recorded 32 reproduced defect families (R-001 through R-032), separate from the historical F/C integration fixes above. At the owner's subsequent request, all 32 now have core source fixes and safe-contract regressions. The original 39 defect probes were converted and expanded to 65 passing safety cases; the complete final framework suite passed (3,041 passed, 1 skipped). Schema 15 supplies durable erasure/delivery fences and transactional context invalidation. The register maps every R-ID to implementation, evidence and remaining verification boundaries. These fixes have **not been installed into the running release by this pass**; no production mutations, external sends or Hermes host edits were made.
