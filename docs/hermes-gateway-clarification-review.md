# Memory clarification through the Hermes gateway

Review date: 2026-10-01. The original review below was read-only. The subsequently authorized implementation is recorded here; deployment status is tracked in the living installation audit.

## Authorized implementation

C-001–C-008 are fixed in source. Owner decision stores now use a composable write transaction; inquiry authentication, epoch/deadline/state fencing, mutation, answer receipt and audit commit together. A transport handoff has a durable attempt record, explicit acknowledgement semantics and a non-retryable unknown-send state. Schema migration 0014 preserves legacy sent rows and marks inherited live leases uncertain. Command delivery explicitly binds the enrolled Hermes home.

Hermes core now carries a frozen native inbound envelope from idle and queued gateway turns to provider `on_turn_start`. The optional returned context is attached to the current user turn even when retrieval is skipped. The memory provider processes replies only there; retrieval is read-only. Question dossiers and refused/successful outcomes are resolved directly from the durable inquiry ledger, without semantic recall or changes to the static prompt.

Native short replies are supported for authenticated Telegram owner DMs, matching the exact stored message ID, thread and owning/receiving home. Different bots/shared-bot satellite routes fail closed rather than guessing; those need a further receiving-transport handoff design. Forgetting still requires an explicit code. Plain `no` on a verified native reply withdraws only the question, not the underlying proposal. An uncorrelated `yes` does not choose the latest inquiry.

The existing CLI transport is retained; this implementation does **not** add a new live gateway IPC delivery service or wake the agent on background questions. The durable bridge is the framework delivery ledger plus host native ingress and turn-boundary context. Best-effort transcript mirroring remains separate from approval authority. Historical messages without new delivery mappings use the code fallback. General free-text informational clarification remains a distinct future schema, not something this approval protocol guesses at.

Validation: full framework suite **2,976 passed, 1 skipped**; actual Hermes/provider/temp-database integration passes profile A→B→A with no network. The broad Hermes gateway run passed 8,758 tests with two new-envelope compatibility failures subsequently corrected and verified by a 13-test rerun; one unrelated API drain timing file passed on retry. This is not a claim that the entire Hermes suite or a live Telegram round trip was tested.

The original 12 reproductions have been converted into desired-contract regressions in `tests/integration/test_inquiry_safety.py`, expanded with crash, unknown-send, native receipt and profile subprocess coverage. Additional provider and host tests cover unauthenticated/internal/forwarded messages, pending context and trivial-query skips.

## Recommendation

Yes: Hermes should own the user-facing clarification conversation, while the memory framework owns the inquiry record, evidence, expiry and authorized state transition. Hermes should receive a structured question dossier and its eventual outcome, not merely a notification saying something was approved. Do not make an LLM's interpretation of a reply the authority for a destructive or permission-granting action.

“Always aware” should mean the appropriate profile and conversation can recover relevant pending questions and outcomes across restarts. It should not mean broadcasting private memory into every Hermes session or invoking an agent/model for every background event.

## What is connected already

1. Framework maintenance discovers pending owner decisions and calls `InquiryStore.ask`. The framework stores a pinned subject digest, epoch, deadline and inquiry ID; asking itself does not send anything. See `src/hermes_memory/processing/maintenance.py:203` and `proactive/inquiries.py:276`.
2. The delivery drain sends a queued inquiry through the configured sink. The command transport is designed for `hermes send`, which uses Hermes's platform sending code. This standalone command need not run inside the live gateway process. See `proactive/delivery.py:309` and `hermes_cli/send_cmd.py` in the Hermes source.
3. Hermes's send tool attempts to mirror outbound text into an existing session. This means Hermes can already see some memory questions in its history. Mirroring is best-effort, can fail independently of transport success, and does not create a missing conversation. It stores ordinary text, not a typed inquiry-to-message association. See Hermes `tools/send_message_tool.py:419` and `gateway/mirror.py:36`.
4. Telegram builds a `MessageEvent` with reply-to message ID/text and sender/routing details. The inbound gateway renders the quote into text. The memory provider does not receive a durable inquiry correlation based on that native message ID. See Hermes `plugins/platforms/telegram/adapter.py:7014`, `:7041`, and `gateway/run_inbound.py:1581`.
5. The provider's `prefetch` calls `_settle_from_turn`, which can recognize an owner reply containing a code in the configured destination. A successful settlement contributes a short receipt to model context. An ordinary native Telegram reply saying only “yes” does not settle the inquiry. See `integrations/hermes-memory/provider.py:892` and `:927`.

Thus this is not a completely disconnected system, but the current bridge is text-based and best-effort. It cannot promise that the agent always knows which question a reply answers, why it was asked, or why an answer was refused.

## Confirmed defects

All original cases below used temporary SQLite databases and synthetic data or mocked subprocess execution. No Telegram messages or model requests were sent. The reproductions are now corrected regression contracts in [test_inquiry_safety.py](../tests/integration/test_inquiry_safety.py).

| ID | Finding and observed consequence | Core fix required |
| --- | --- | --- |
| C-001, critical approval safety | `_polarity` accepts positive words inside negation. `I am not sure CODE` actually activates a goal; `not confirmed`, `not correct` and `not approved` also parse as yes. | Use an explicit answer grammar for authorizations. Ambiguous/negated prose must not authorize. Keep free-text informational answers separate. |
| C-002, high delivery correctness | Two overlapping `send_next` calls claim the same open inquiry despite a live lease. Both send different codes; the first delivered code stops working. Completion reports success even after losing ownership. | Atomic compare-and-set lease acquisition, eligibility including lease expiry, and completion ownership/row-count checks. Preserve a stable request identity and reconcile uncertain sends rather than blindly reminting/retrying. |
| C-003, high consent integrity | The subject fence is checked before the mutation transaction. A second connection can revise a goal after the check; the answer then activates revision 2 that the question never represented. | Validate epoch and exact expected revision/digest inside the same transaction as the authorized mutation, inquiry transition and audit. |
| C-004, delivery correctness | A sink returning `{"sent": false}` is still recorded as sent. | Typed receipt validation: distinguish definite failure, acknowledged send, unknown send and confirmed conversation attachment. A normal function return is not a positive acknowledgement. |
| C-005, expiry correctness | An already-expired open question is sent before maintenance expires it. The freshly delivered code is unusable. | Enforce deadline at claim/send time as well as answer time; never send expired inquiries. |
| C-006, user-visible contract | The body says plain `no` stops asking, but even `answer(reply="no", inquiry_id=...)` refuses without a code and leaves the inquiry sent. | Align instructions with the authorized protocol. Explicitly distinguish withdrawing a question from rejecting/cancelling its underlying proposal. |
| C-007, lifecycle correctness | After an epoch bump voids a question, `ask` reopens the same deterministic row without updating its epoch. The reopened row retains the revoked epoch and cannot be answered. | Create a new epoch-bound question generation, or reinitialize all generation-specific fields atomically. Old delivered codes must remain invalid. |
| C-008, profile routing | `sink_for(settings, hermes_home)` ignores `hermes_home` for command delivery, and `command_sink` removes `HERMES_HOME` from its environment. The mocked child gets neither the requested profile home nor the caller's explicit binding. | Resolve and pass the enrolled profile's explicit, validated transport context. Do not inherit or guess a default profile. Custom commands that explicitly bind a profile may avoid this, but the factory currently provides no guarantee. |

Locations: `src/hermes_memory/proactive/inquiries.py:102` (parser), `:429` (send/lease/expiry/receipt), `:498` (answer/fence), `:276` (reopening); `src/hermes_memory/proactive/delivery.py:309` and `:375` (profile transport).

Validation command:

```sh
.venv/bin/python -m pytest -q \
  tests/integration/test_inquiry_safety.py \
  tests/integration/test_inquiries.py \
  tests/integration/test_inquiry_reporting.py \
  tests/integration/test_delivery.py \
  tests/hermes/test_hermes_plugin.py
```

Result: **205 passed, 1 skipped**, including **12 reproduction cases** covering the eight defect classes. Passing reproductions prove the defects still exist, not that they are fixed. Existing tests therefore do not cover these safety edges adequately. This is a targeted result, not a fresh full-suite or live-gateway certification.

## Further gaps and risks: inspected, not all experimentally reproduced

- Approval handling is coupled to retrieval. Hermes calls `on_turn_start` separately but invokes settlement inside provider `prefetch`. Trivial-input skips, a still-running prior prefetch thread, and timeouts can omit or delay this path. Reply authorization needs deterministic ingress processing with an immutable per-turn identity, independent of model invocation and retrieval. See Hermes `agent/turn_context.py:853`, `agent/memory_manager.py:459`, and provider `on_turn_start:449`.
- Native `reply_to_message_id`, profile/bot identity, thread and chat type need to survive into the bridge as structured metadata. Parsing rendered quoted text is not sufficient authentication or correlation. The provider's quote-stripping regex also does not cover every Hermes quote format.
- Pending inquiries are not an explicit ContextBroker source. Semantic recall should not be required to locate a known question. A dossier should include the full current proposal/preview, relevant supporting references, why clarification is needed, deadline and outcome/refusal reason. Inquiry-only changes need their own revision signal; canonical evidence watermarks alone may not invalidate this context.
- Current inquiries are constrained to recognized owner decision acts. They are not a general free-text ambiguity-resolution protocol. Stored/rendered choices are not a complete choice-answer implementation. Separate informational clarification from approval before extending the API.
- Settlement and marking the inquiry answered are separate commits. A crash between them could leave the subject changed but the inquiry unacknowledged. This is an inspected crash-window risk, not a fault-injection result in this review.
- Destination configuration alone is not proof of a private owner chat. Validate actual sender, chat type, receiving bot/profile and thread at the gateway. Never treat possession of quoted text, a guessed code, or an internal injected message as owner identity.
- A generic text mirror is not an authorization receipt. It can omit structured provenance and append an assistant-role message into history. Any new bridge must respect Hermes's turn-boundary ordering and prompt-cache constraints rather than rewriting old prompt prefixes.
- Failure/refusal context is currently thin: `_settle_from_turn` returns context for successful settlement but generally returns nothing when a code is stale, wrong or expired. Hermes should be able to explain those outcomes without inventing them.

## Proposed core integration

### 1. Durable, typed handoff

Keep framework `InquiryStore` as the authoritative state machine. Add a versioned request envelope containing event ID, inquiry ID/generation, subject revision/digest, epoch, owning profile, owner principal, approved transport/chat/thread, question, answer schema, supporting references, deadline and action risk.

Use a durable outbox with an authenticated local gateway consumer/bridge. A process-local plugin callback alone cannot connect the separate framework service to a live gateway. Avoid a memory-specific branch in generic Hermes core: expose a narrow typed external-event/receipt interface, with memory-specific semantics in the plugin/framework.

Record separate transport and context receipts: platform, receiving bot/profile, chat/thread, native message ID, session key/ID, event ID, body digest, send status and conversation-attachment status. Unknown delivery must remain unknown; a retry must not create conflicting live codes. Explicitly handle a missing or ambiguous conversation instead of silently choosing another profile or the newest chat.

### 2. Give Hermes context without making every event an agent turn

Persist a provenance-labelled external inquiry event in the correct conversation and expose its dossier at the next turn boundary. Pending questions and final outcomes must be recoverable after gateway restart or session rollover. Respect the host's cache-coherence and message-ordering rules; do not relabel a framework proposal as something the human said.

An optional agent-mediated follow-up may use Hermes's plugin injection capability only when enabled and appropriate. `PluginContext.inject_message` schedules an internal event in the live process; acceptance is not proof that a turn ran or a Telegram message arrived. It is not by itself a durable cross-process transport. Default to passive context plus deterministic delivery, with explicit budgets/permissions for autonomous follow-up.

### 3. Resolve replies before retrieval or model interpretation

At authenticated gateway ingress, capture immutable sender and canonical routing identity plus native reply-to ID and unquoted owner text. Resolve the reply against the stored delivery receipt in the same profile/bot/chat/thread. Obtain the original question dossier directly by ID, not by similarity search.

A plain “yes” can be supported only for a verified native reply to exactly one live, unchanged question under an explicitly enabled answer protocol. Uncorrelated answers still require the current code or explicit inquiry selector; never choose the latest pending question implicitly. Quoted/forwarded messages, bots, internal injection and replies from another account cannot confer approval. High-risk erasure may warrant a stronger explicit confirmation step.

Record informational free-text answers as attributed observations/candidates, not authorizations. For owner decisions, use deterministic parsing and one atomic expected-state transaction. Return a structured outcome/refusal event to Hermes so it can explain precisely what changed or why nothing changed.

### 4. Privacy and lifecycle

Apply existing owner reply grants, delivery holds, attention budgets and quiet hours. Expiry, reset, restore, revocation and erasure must invalidate old inquiry generations and receipts appropriately. Define retention/redaction of question previews and mirrored content; deleting canonical evidence does not automatically remove already-sent Telegram messages or chat-history copies.

## Implementation and acceptance sequence

1. Fix C-001/C-003 first, then leasing, receipts, expiry, epoch reopening and profile binding. Convert these reproduction probes into regression tests asserting refusal/no unintended action and correct delivery ownership.
2. Implement typed outbox/receipts and deterministic ingress correlation. Keep code replies compatible; add natural native replies only once provenance and routing checks pass.
3. Add pending inquiry/outcome dossiers to the appropriate turn context through the plugin and a narrow host interface where needed. Do not patch deployed package files or make the static prompt dynamic.
4. Verify two profiles and bots, shared chats/threads, profile A→B→A, absent sessions, busy/idle turns, restart, delayed/duplicate/out-of-order replies, multiple pending questions, partial quotes, forwards, spoofed/internal events, revoked grants, changed subjects, concurrent answers, expired requests and uncertain delivery. Include crash/recovery across the mutation/receipt boundaries.
5. Only then run an explicitly authorized end-to-end Telegram test using a synthetic proposal. Check actual owner reply, final canonical state and Hermes's ability to explain the original question and result. No such live test was performed in this review.

## Scope and outstanding decisions

This review followed inquiry production, state transitions, delivery, provider tools/prefetch, Telegram reply events, transcript mirroring and plugin injection. It is not an exhaustive audit of every framework subsystem.

Decisions still needed before the conversational extension: informational clarification versus approval schemas; whether “no” withdraws the question or rejects the proposal; whether new conversations may be created automatically; high-risk confirmation policy; and whether follow-up may wake the agent or should only be attached to its next user turn.
