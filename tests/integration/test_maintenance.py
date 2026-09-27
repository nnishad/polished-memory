"""The background pass: the one seam where the durable bookkeeping actually runs.

Every case here was written after the same discovery: a component existed, was correct on
its own terms, and had no caller. A due reminder was never swept, a stale summary was
reported but never queued for refresh, an identity candidate on forgotten evidence stayed
pending forever, and a backed-off job was visible while being unclaimable. The pass is what
closes those, and the rule it is held to is that it spends nothing: no model call, no device
slot, no attempt from a job's budget — because that is what makes it safe to run unattended.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
import time

import pytest

from conftest import envelope
from hermes_memory.backend.provenance import ProvenanceLedger
from hermes_memory.knowledge.summaries import SummaryStore
from hermes_memory.lifecycle.erasure import PENDING, ErasureManager
from hermes_memory.processing.jobs import QUARANTINED, RETRY_WAIT, JobQueue
from hermes_memory.backend.document_map import DocumentMap
from hermes_memory.processing.maintenance import (DEFAULT_LIMIT, MAX_LIMIT, SECTIONS,
                                                  Maintenance, Ticker, run)
from hermes_memory.processing.resource_gate import UNCERTAIN, ResourceGate
from hermes_memory.processing.routes import Route
from hermes_memory.proactive.engine import AWAITING
from hermes_memory.proactive.policy import POLICY_VERSION, AttentionPolicy
from hermes_memory.prospective.due_events import DueEventLog
from hermes_memory.prospective.goals import GoalStore
from hermes_memory.sources.sync import SyncController
from hermes_memory.storage.evidence import EvidenceError
from hermes_memory.storage.identity import STALE, IdentityStore

OWNER = "owner-principal"
AGENT = "agent:sess-1"
UTC = "UTC"
MORNING = "2026-09-15T09:00:00+00:00"
LATER = "2026-09-15T15:00:00+00:00"
AFTER_SNOOZE = "2026-09-16T09:00:00+00:00"
# The epoch form of MORNING: leases are counted in monotonic seconds, and a pass whose
# clock disagrees with the instant it was asked to speak for cannot say which one it means.
CLOCK = 1_789_462_800.0
ROUTE = Route("retain", "remote-9b", "chat", "http://127.0.0.1:8080/v1", "cred", "freshness",
              2048)
DAY = "day:2026-09-15"


class Settings:
    """The handful of attributes the door reads, and nothing else."""

    def __init__(self, root):
        self.db_path = Path(root) / "canonical.db"
        self.home = Path(root)
        self.owner_principal = OWNER
        self.profile = "test"


@pytest.fixture()
def sync(store):
    """A connector that has said it reached its tail, so coverage is a claim on file."""
    subject = SyncController(store)
    subject.register("gmail", policy_version="private-api")
    fence = subject.acquire("gmail", holder="test")
    subject.publish(fence, "page-1", [], next_cursor=None)
    subject.release(fence)
    return subject


@pytest.fixture()
def loop(store, sync):
    """The pass itself, over the same store a test seeds."""
    def build(**kwargs):
        return Maintenance(store, owner_principal=OWNER, sync=sync,
                           clock=lambda: CLOCK, **kwargs)
    return build


@pytest.fixture()
def live(store):
    """Delivery switched on. An installation starts in shadow mode, and this is the
    owner's act of turning it off — a pass may not assume it happened."""
    policy = AttentionPolicy(store, owner_principal=OWNER)
    policy.configure(actor=OWNER, timezone_name=UTC, max_immediate_per_day=10,
                     cooldown_minutes=0, shadow=False)
    return policy


def remind(store, *, title="Call the dentist", at=MORNING):
    return GoalStore(store, events=DueEventLog(store), owner_principal=OWNER).propose(
        title=title, statement="The appointment is booked.", timezone_name=UTC, due=at,
        proposed_by=OWNER, proposed_kind="owner")["id"]


def events(store):
    return DueEventLog(store)


def states(store, table="due_events", order="id"):
    return [row["state"] for row in store.db.execute(
        f"SELECT state FROM {table} ORDER BY {order}")]


def count(store, table, **where):
    clause = " WHERE " + " AND ".join(f"{key}=?" for key in where) if where else ""
    return int(store.db.execute(
        f"SELECT count(*) AS n FROM {table}{clause}", tuple(where.values())).fetchone()["n"])


def published(store, record, *, scope=DAY):
    summaries = SummaryStore(store, ledger=ProvenanceLedger(store), owner_principal=OWNER)
    summaries.publish(scope=scope, kind="day", title="The 04:12 incident",
                      body="It broke and recovered.", citations=[{"record_id": record}],
                      processor_fingerprint="summarizer-1.0")
    return summaries


def job(store, *, backoff: float = -5.0):
    """One job that has already failed once and is waiting out a backoff."""
    jobs = JobQueue(store, clock=lambda: CLOCK)
    made = jobs.enqueue(kind="retain", inputs=["rec_a"], input_revision="1", route=ROUTE,
                        processor_fingerprint="extractor-v3")
    jobs.retry(jobs.get(made["job_id"]), error="the backend hung", backoff=backoff)
    return made["job_id"]


def overdue_job(store):
    """One job whose deadline passed while nobody was looking at the queue."""
    return JobQueue(store, clock=lambda: CLOCK).enqueue(
        kind="retain", inputs=["rec_late"], input_revision="1", route=ROUTE,
        processor_fingerprint="extractor-v3", deadline=CLOCK - 60)["job_id"]


# -- the proactive section -----------------------------------------------------

def test_the_pass_is_the_thing_that_takes_a_due_reminder(store, loop, live):
    """The engine was correct on its own and unreachable; the pass is its caller."""
    remind(store)
    report = loop().pass_now(at=MORNING)
    assert report["proactive"]["decided"] == 1
    assert report["proactive"]["prepared"] == 1


def test_a_reminder_is_kept_in_the_owner_s_own_words_with_inference_off(store, loop, live):
    remind(store, title="Renew the passport before Friday")
    loop().pass_now(at=MORNING)
    artifact = store.db.execute("SELECT kind, payload, state FROM outbox").fetchone()
    assert artifact["payload"] == "Renew the passport before Friday"
    assert artifact["kind"] == "notify_owner"
    assert artifact["state"] == "prepared", "the framework prepares; Hermes delivers"


def test_a_shadow_installation_is_left_its_reminders(store, loop):
    """The destructive case.

    Shadow mode is the default state of a new installation, so a pass that read its own
    refusal as a decision would have consumed every reminder a person ever asked for
    before anyone had agreed to be interrupted at all.
    """
    remind(store)
    report = loop().pass_now(at=MORNING)
    assert report["proactive"]["deferred"] == 1
    assert report["proactive"]["suppressed"] == 0
    assert report["proactive"]["refused"] == 0, "a not-yet is not a no"
    assert states(store) == ["pending"], "the promise is still standing"
    assert count(store, "decision_intents") == 0, "no intention was invented in order to " \
                                                  "be refused"
    assert count(store, "outbox") == 0


def test_turning_delivery_on_is_enough_for_the_same_reminder_to_arrive(store, loop):
    remind(store)
    assert loop().pass_now(at=MORNING)["proactive"]["prepared"] == 0
    AttentionPolicy(store, owner_principal=OWNER).configure(
        actor=OWNER, timezone_name=UTC, max_immediate_per_day=10, cooldown_minutes=0,
        shadow=False)
    assert loop().pass_now(at=MORNING)["proactive"]["prepared"] == 1, \
        "the deferred reminder was still there to be delivered"


def test_an_operator_pause_defers_without_spending_the_event(store, loop, live):
    remind(store)
    store.set_control("topic:general", "attention", "paused", actor=OWNER,
                      reason="holding everything back", policy_version="test")
    report = loop().pass_now(at=MORNING)
    assert report["proactive"]["deferred"] == 1 and states(store) == ["pending"]
    store.set_control("topic:general", "attention", "active", actor=OWNER,
                      reason="resumed", policy_version="test")
    assert loop().pass_now(at=MORNING)["proactive"]["prepared"] == 1


def test_a_snoozed_reminder_is_not_due_and_survives_until_it_is(store, loop, live):
    """A snooze is the owner saying *later*; a pass may not read that as *never*.

    The clock filter is C8's own: while the snooze holds, the event is not due at all, so
    the pass neither takes it nor spends it. The interesting half is what happens when the
    snooze ends — the reminder has to still be there.
    """
    goal = remind(store)
    GoalStore(store, events=events(store), owner_principal=OWNER).snooze(
        goal_id=goal, actor=OWNER, until="2026-09-16T06:00:00+00:00", reason="busy today")
    assert loop().pass_now(at=MORNING)["proactive"]["deferred"] == 0, \
        "not due is not the same as deferred: nothing was even claimed"
    assert states(store) == ["pending"] and count(store, "outbox") == 0
    assert loop().pass_now(at=AFTER_SNOOZE)["proactive"]["prepared"] == 1


def test_a_snooze_after_the_handoff_waits_instead_of_being_refused(store, loop, live):
    """The recovery path: promised, then put off, then looked at again.

    An intention a crashed pass left standing goes through eligibility again, and by then
    the owner may have snoozed it. Suppressing it there would be the same destruction with
    a longer story: the reminder is owed after the snooze, not never.
    """
    goal = remind(store)
    event = events(store).due(MORNING)[0]
    claim = events(store).claim(event.id, holder="dead-worker", at=CLOCK)
    events(store).ack(event_id=event.id, token=claim.token, decision=AWAITING,
                      policy_version=POLICY_VERSION)
    GoalStore(store, events=events(store), owner_principal=OWNER).snooze(
        goal_id=goal, actor=OWNER, until="2026-09-16T06:00:00+00:00", reason="busy today")

    report = loop().pass_now(at=MORNING)
    assert report["proactive"]["deferred"] == 1
    assert report["proactive"]["suppressed"] == 0
    assert report["proactive"]["refused"] == 0, "waiting is not an answer about the change"
    assert states(store, "decision_intents") == [AWAITING], \
        "the promise stands, unfinished rather than answered"
    assert count(store, "outbox") == 0
    assert loop().pass_now(at=AFTER_SNOOZE)["proactive"]["prepared"] == 1


def test_a_topic_the_owner_opted_out_of_is_still_a_real_no(store, loop, live):
    """The durable refusals must keep being durable, or silence is only postponement."""
    remind(store)
    live.configure(actor=OWNER, opted_out=True)
    report = loop().pass_now(at=MORNING)
    assert report["proactive"]["suppressed"] == 1
    assert report["proactive"]["deferred"] == 0
    assert states(store) == ["suppressed"]


def test_the_second_pass_says_nothing_twice(store, loop, live):
    remind(store)
    assert loop().pass_now(at=MORNING)["proactive"]["decided"] == 1
    quiet = loop().pass_now(at=MORNING)["proactive"]
    assert quiet["decided"] == 0 and quiet["deferred"] == 0 and quiet["prepared"] == 0
    assert count(store, "outbox") == 1


# -- the summary section -------------------------------------------------------

def test_a_correction_becomes_a_refresh_promise_rather_than_a_complaint(store, loop):
    record = store.commit(envelope(source_id="msg-incident",
                                  text="The build broke at 04:12."))["id"]
    published(store, record)
    store.commit(envelope(source_id="msg-incident", revision="2",
                          text="Correction: it broke and stayed broken."))

    report = loop().pass_now(at=MORNING)
    assert report["summaries"]["stale"] == 1
    assert report["summaries"]["promised"] == 1
    assert report["summaries"]["pending"] == 1
    assert report["summaries"]["scopes"] == [DAY]


def test_four_corrections_are_one_refresh(store, loop):
    record = store.commit(envelope(source_id="msg-incident", text="Broke."))["id"]
    published(store, record)
    for revision in range(2, 6):
        store.commit(envelope(source_id="msg-incident", revision=str(revision),
                              text=f"Correction number {revision}."))
    first = loop().pass_now(at=MORNING)["summaries"]
    again = loop().pass_now(at=LATER)["summaries"]
    assert first["stale"] == 1 and first["promised"] == 1
    assert again["promised"] == 0 and again["already_promised"] == 1
    assert count(store, "summary_refreshes") == 1, \
        "the window that is already promised is not booked again by the next pass"


def test_a_promise_settled_without_the_work_is_booked_again(store, loop):
    """Settling is a bookkeeping act; staleness is a fact about the evidence.

    Marking a refresh done without rewriting the summary cannot make the summary stand, so
    the next pass asks again. The alternative — trusting the ledger entry over the evidence
    — is how a refresh that never happened comes to be reported as one that did.
    """
    record = store.commit(envelope(source_id="msg-incident", text="Broke."))["id"]
    summaries = published(store, record)
    store.commit(envelope(source_id="msg-incident", revision="2", text="Corrected."))
    loop().pass_now(at=MORNING)
    summaries.settle_refresh(scope=DAY, kind="day", through_at=MORNING, ok=True,
                             detail="marked done, nothing rewritten")
    assert loop().pass_now(at=LATER)["summaries"]["promised"] == 1
    assert count(store, "summary_refreshes") == 2, "one closed promise and one open"


def test_the_refresh_is_promised_and_not_performed(store, loop):
    """Performing one is a model call, and this pass makes none."""
    record = store.commit(envelope(source_id="msg-incident", text="Broke."))["id"]
    published(store, record)
    store.commit(envelope(source_id="msg-incident", revision="2", text="Corrected."))
    loop().pass_now(at=MORNING)
    assert states(store, "summary_refreshes", order="rowid") == ["pending"]
    assert count(store, "summaries") == 1, "no second revision was written"


# -- the identity section ------------------------------------------------------

def candidate(store, record, *, evidence=None):
    identity = IdentityStore(store, owner_principal=OWNER)
    made = identity.propose(
        account_a=identity.account("email", "a@example.com"),
        account_b=identity.account("email", "a@work.example"),
        rule="email-thread-participant", basis="one thread, two addresses",
        evidence=evidence or [record], proposed_by=AGENT, proposed_kind="agent")
    return identity, made["candidate_id"]


def state_of(store, identifier):
    return store.db.execute("SELECT state FROM identity_candidates WHERE id=?",
                            (identifier,)).fetchone()["state"]


def test_a_candidate_on_forgotten_evidence_expires(store, loop):
    record = store.commit(envelope(source_id="thread-1"))["id"]
    _, identifier = candidate(store, record)
    store.hide(record, reason="the owner forgot it", actor=OWNER)

    report = loop().pass_now(at=MORNING)
    assert report["identity"]["candidates_expired"] == 1
    assert state_of(store, identifier) == STALE
    assert loop().pass_now(at=LATER)["identity"]["candidates_expired"] == 0, \
        "an expired candidate is not counted again by the next pass"


def test_a_confirmed_edge_losing_its_evidence_goes_to_the_owner_not_the_bin(store, loop):
    record = store.commit(envelope(source_id="thread-1"))["id"]
    identity, identifier = candidate(store, record)
    identity.confirm(candidate_id=identifier, actor=OWNER, reason="yes, the same person")
    store.hide(record, reason="forgotten", actor=OWNER)

    report = loop().pass_now(at=MORNING)
    assert report["identity"]["candidates_expired"] == 0
    assert report["identity"]["confirmed_needing_review"] == 1
    assert state_of(store, identifier) != STALE, \
        "only the owner revokes what the owner confirmed"


def test_a_candidate_with_one_live_citation_survives(store, loop):
    kept = store.commit(envelope(source_id="thread-1"))["id"]
    gone = store.commit(envelope(source_id="thread-2", revision="1",
                                 text="A second message in the same thread."))["id"]
    _, identifier = candidate(store, kept, evidence=[kept, gone])
    store.hide(gone, reason="forgotten", actor=OWNER)
    assert loop().pass_now(at=MORNING)["identity"]["candidates_expired"] == 0
    assert state_of(store, identifier) != STALE, "one live citation is still a checkable claim"


# -- the queue section ---------------------------------------------------------

def test_a_backed_off_job_is_claimable_work_and_the_pass_says_so(store, loop):
    """The queue used to report a job as waiting while refusing to hand it out."""
    identifier = job(store)
    report = loop().pass_now(at=MORNING)
    assert report["queue"]["counts"] == {RETRY_WAIT: 1}
    assert report["queue"]["ready_for_retry"] == [identifier]


def test_a_job_still_in_backoff_is_not_offered(store, loop):
    identifier = job(store, backoff=600.0)
    report = loop().pass_now(at=MORNING)
    assert report["queue"]["ready_for_retry"] == []
    assert report["queue"]["counts"] == {RETRY_WAIT: 1}
    assert report["queue"]["quarantined"] == [] and identifier is not None


def test_the_pass_does_not_eat_the_attempt_budget(store, loop):
    """Reporting readiness must not spend an attempt: that is a worker's act."""
    identifier = job(store)
    loop().pass_now(at=MORNING)
    row = store.db.execute("SELECT state, attempts, lease FROM processing_jobs WHERE id=?",
                           (identifier,)).fetchone()
    assert (row["state"], row["attempts"], row["lease"]) == (RETRY_WAIT, 1, None)


def test_a_quarantined_job_is_named_rather_than_resurrected(store, loop):
    identifier = job(store, backoff=0.0)
    jobs = JobQueue(store, clock=lambda: CLOCK)
    for _ in range(3):
        jobs.retry(jobs.get(identifier), error="still failing", backoff=0.0)
    report = loop().pass_now(at=MORNING)
    assert report["queue"]["counts"] == {"quarantined": 1}
    assert report["queue"]["quarantined"] == [identifier]
    assert report["queue"]["ready_for_retry"] == [], \
        "work that ran out of tries is not put back in the queue by a pass"


def test_a_job_that_missed_its_deadline_is_given_its_ending(store, loop):
    """`claim()` was already refusing it; the pass is what says so where anyone can read it."""
    identifier = overdue_job(store)
    report = loop().pass_now(at=MORNING)
    assert report["queue"]["overdue_reaped"] == [identifier]
    assert report["queue"]["counts"] == {QUARANTINED: 1}
    again = loop().pass_now(at=LATER)
    assert again["queue"]["overdue_reaped"] == [], \
        "a closed row is not re-closed on every pass"


# -- the erasure section -------------------------------------------------------

def test_an_owed_backend_cleanup_stays_owed_and_says_so(store, loop):
    """Forgetting locally is done; the obligation to say so to the backend is not."""
    record = store.commit(envelope(source_id="msg-1", text="Mentioned in error."))["id"]
    DocumentMap(store).begin(record, "1")
    forget = ErasureManager(store, owner_principal=OWNER)
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    forget.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                   actor=OWNER)

    before = forget.pending()
    report = loop().pass_now(at=MORNING)
    assert len(before) >= 1
    assert report["erasure"]["backend_owed"] == len(before)
    assert report["erasure"]["oldest_owed"] == before[0]["requested_at"]
    after = forget.pending()
    assert [item["attempts"] for item in after] == [item["attempts"] for item in before], \
        "the pass cannot call a backend it is not authorized to reach"
    assert [item["state"] for item in after] == [PENDING] * len(after)


def test_an_unconfirmed_erasure_is_counted_without_being_quoted(store, loop):
    """A preview is private by design: sizes and digests, never record text."""
    record = store.commit(envelope(source_id="msg-1", text="A private matter."))["id"]
    ErasureManager(store, owner_principal=OWNER).preview(
        record_ids=[record], actor=OWNER, reason="withdrawn")
    report = loop().pass_now(at=MORNING)
    assert report["erasure"]["waiting_for_owner"] == 1
    assert "private matter" not in str(report)


# -- the pass as a whole -------------------------------------------------------

def test_every_section_runs_and_reports(store, loop):
    report = loop().pass_now(at=MORNING)
    assert list(report["sections"]) == list(SECTIONS)
    for name in SECTIONS:
        assert isinstance(report[name], dict), f"{name} reported nothing"


def test_one_instant_is_used_for_the_whole_pass(store, loop):
    assert loop().pass_now(at="2026-09-15T09:00:00+02:00")["at"] == "2026-09-15T07:00:00+00:00"
    assert loop().pass_now()["at"] == MORNING, \
        "with no instant the pass speaks for the clock it was given, which is this fixture's"


def test_an_instant_without_a_zone_is_refused(store, loop):
    with pytest.raises(ValueError, match="timezone"):
        loop().pass_now(at="2026-09-15T09:00:00")


def test_sections_can_be_taken_one_at_a_time(store, loop):
    report = loop().pass_now(at=MORNING, sections=("queue",))
    assert sorted(report) == ["at", "inference_paused", "queue", "sections"]
    assert report["sections"] == ["queue"]


def test_an_unknown_section_is_refused(store, loop):
    with pytest.raises(EvidenceError, match="unknown maintenance section"):
        loop().pass_now(sections=("everything",))


def test_the_pass_spends_nothing_it_was_not_given(store, loop, live):
    """The rule, checked where it would hurt: no claim, no client, no model.

    The engine is built with no analyst and no broker, so there is no model to reach even
    by mistake, and the job that inference would ride on is left exactly where it was.
    """
    identifier = job(store)
    remind(store)
    report = loop().pass_now(at=MORNING)
    assert report["proactive"]["analysed"] == 0
    assert loop().engine.analyst is None and loop().engine.broker is None
    row = store.db.execute("SELECT state, lease, attempts FROM processing_jobs WHERE id=?",
                           (identifier,)).fetchone()
    assert (row["state"], row["lease"], row["attempts"]) == (RETRY_WAIT, None, 1), \
        "the pass reported the queue rather than working it"
    assert count(store, "audit", action="projection_begin") == 0, \
        "nothing was projected, so nothing was retained"


def test_an_inference_hold_is_reported_rather_than_worked_around(store, sync, live):
    """The gate is a claim about inference. The pass says what it found, and does not
    depend on it, because it performs none."""
    gate = ResourceGate(store)
    gate.pause(actor=OWNER, reason="a hold for the test")
    subject = Maintenance(store, owner_principal=OWNER, sync=sync, gate=gate,
                          clock=lambda: CLOCK)
    remind(store)
    report = subject.pass_now(at=MORNING)
    assert report["inference_paused"] is True
    assert report["proactive"]["prepared"] == 1, \
        "a reminder needs no model, so a hold on inference must not swallow it"


def test_an_expired_device_lease_stops_occupying_its_resource_without_freeing_it(store):
    """A holder that died must stop blocking the device — and must not be called free.

    The request it was running may or may not have reached the model. Releasing the slot
    would pretend the answer is known; leaving it held would let one dead process block a
    physical resource until somebody else asked for it.
    """
    clock = {"now": 1000.0}
    gate = ResourceGate(store, clock=lambda: clock["now"], default_ttl=30.0)
    held = gate.try_acquire(route="retain", holder="worker-that-died", resource="local-gpu",
                            priority=2)
    assert held is not None
    clock["now"] = 2000.0

    subject = Maintenance(store, owner_principal=OWNER, sync=None, gate=gate,
                          clock=lambda: CLOCK)
    report = subject.pass_now(at=MORNING, sections=("queue",))
    assert report["queue"]["leases_uncertain"] == 1
    assert [row[0] for row in store.db.execute(
        "SELECT state FROM gate_reservations WHERE id=?", (held.id,))] == [UNCERTAIN]

    again = subject.pass_now(at=MORNING, sections=("queue",))
    assert again["queue"]["leases_uncertain"] == 0, \
        "one expired lease is converted once, not reported afresh every pass"


def test_the_bound_is_honoured_and_an_absurd_one_falls_back(store, loop):
    for index in range(4):
        remind(store, title=f"Reminder {index}")
    assert loop().pass_now(at=MORNING, limit=2)["proactive"]["deferred"] == 2, \
        "a pass takes what it was bounded to and leaves the rest standing"
    assert loop().pass_now(at=MORNING, limit=0)["proactive"]["deferred"] == 4, \
        f"a zero bound is not a bound (it falls back to {DEFAULT_LIMIT}), and a deferred " \
        "reminder is still due to be seen again"
    assert loop().pass_now(at=MORNING, limit=MAX_LIMIT + 1)["proactive"]["deferred"] == 4, \
        "an absurd bound falls back the same way rather than running unbounded"


def test_the_bound_is_a_bound_rather_than_a_suggestion(store):
    """An absurd bound falls back to the named default instead of running unbounded."""
    from hermes_memory.processing.maintenance import _bounded

    assert _bounded(7) == 7 and _bounded(1) == 1 and _bounded(MAX_LIMIT) == MAX_LIMIT
    for nonsense in (0, -1, MAX_LIMIT + 1, None, "25", 2.5):
        assert _bounded(nonsense) == DEFAULT_LIMIT, nonsense


# -- the door ------------------------------------------------------------------

def test_the_door_refuses_an_installation_with_no_store(tmp_path):
    report = run(Settings(tmp_path / "nowhere"), at=MORNING)
    assert report["ok"] is False and "no canonical store" in report["refused"]


def test_the_door_runs_the_pass_over_the_installation_it_names(tmp_path):
    """The door opens the store, hands it to the same pass the tests above drive."""
    from hermes_memory.storage.evidence import EvidenceStore

    settings = Settings(tmp_path)
    with EvidenceStore(settings.db_path) as store:
        remind(store)
    report = run(settings, at=MORNING)
    assert report["ok"] is True and report["profile"] == "test"
    assert report["sections"] == list(SECTIONS)
    assert report["proactive"]["deferred"] == 1, "shadow mode is still the default here"
    assert run(settings, at=MORNING, sections=("queue",))["sections"] == ["queue"]


# -- the ledger the pass writes to ----------------------------------------------

def an_expired_lease(settings, *, resource="local-gpu"):
    """One reservation in the instance admission ledger whose holder stopped answering."""
    from hermes_memory.processing.instance_gate import GateStore, gate_path

    clock = {"now": 1000.0}
    store = GateStore(gate_path(settings))
    gate = ResourceGate(store, clock=lambda: clock["now"], default_ttl=30.0)
    held = gate.try_acquire(route="retain", holder="worker-that-died", resource=resource,
                            priority=2)
    clock["now"] = 2000.0
    store.close()
    return held


def a_ledger_state(settings, reservation_id):
    from hermes_memory.processing.instance_gate import GateStore, gate_path

    store = GateStore(gate_path(settings), create=False)
    try:
        return [row[0] for row in store.db.execute(
            "SELECT state FROM gate_reservations WHERE id=?", (reservation_id,))]
    finally:
        store.close()


def test_the_door_reaps_against_a_ledger_it_is_allowed_to_write(tmp_path):
    """A reaping is a write, so the pass cannot ask the reading connection to perform it.

    The bug this pins: ``run()`` handed the pass ``status_gate`` — read-only, correctly,
    because a reading must not invent a ledger — and the queue section reaps expired
    leases. The raise landed after the identity section, so the scheduler kept ticking
    while the pass stopped recording that it ran, and an expired lease kept its device
    occupied for the rest of the installation's life.
    """
    from hermes_memory.storage.evidence import EvidenceStore

    settings = Settings(tmp_path)
    with EvidenceStore(settings.db_path):
        pass
    held = an_expired_lease(settings)

    report = run(settings, at=MORNING)
    assert report["ok"] is True and report["queue"]["leases_uncertain"] == 1
    assert a_ledger_state(settings, held.id) == [UNCERTAIN], \
        "the lease was reaped in the file, not only in the report"


def test_a_pass_invents_no_admission_ledger_to_reap_from(tmp_path):
    """Nothing has been queued from a fresh installation, and that is the answer.

    ``None`` from the ledger is not an error to work around: the pass runs, reports no
    uncertain leases, and leaves no file behind — the same rule ``status`` follows, which
    is why the writable form asks rather than creates.
    """
    from hermes_memory.processing.instance_gate import gate_path
    from hermes_memory.storage.evidence import EvidenceStore

    settings = Settings(tmp_path)
    with EvidenceStore(settings.db_path):
        pass
    report = run(settings, at=MORNING)
    assert report["ok"] is True and report["queue"]["leases_uncertain"] == 0
    assert not gate_path(settings).exists()


def test_the_two_ledger_forms_differ_exactly_in_whether_they_can_write(tmp_path):
    """``writable_gate`` and ``status_gate`` read one file; only one of them may change it.

    The distinction is the mode of the connection, so a test that reached the pass through
    the other form would not have caught what happened: the reading is correct for
    ``status``, and wrong for a reaping.
    """
    from hermes_memory.processing.instance_gate import (GateStore, gate_path, status_gate,
                                                        writable_gate)

    settings = Settings(tmp_path)
    with GateStore(gate_path(settings)):
        pass

    gate = writable_gate(settings)
    gate.pause(actor=OWNER, reason="the models are stopped for the night")
    reading = status_gate(settings)
    assert reading.paused is True and gate.hold()["actor"] == OWNER
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reading.pause(actor=OWNER, reason="a reading cannot hold the models")


# -- the heartbeat -------------------------------------------------------------

def test_a_pass_that_raised_records_where_it_broke_instead_of_recording_nothing(store, loop,
                                                                               monkeypatch):
    """A half-run pass and a skipped period have to be two different reports.

    The sections run in order, so an exception anywhere after the first loses the
    heartbeat the pass exists to write. Recording the failure is what turns "the scheduler
    is not running" — the wrong conclusion this machine actually produced — into the right
    one: it is running, and this section raises every period.
    """
    def broken(self, *, moment, limit):
        raise RuntimeError("attempt to write a readonly database")

    monkeypatch.setattr(Maintenance, "_queue", broken)
    with pytest.raises(RuntimeError, match="readonly"):
        loop().pass_now(at=MORNING)

    last = Maintenance.last_pass(store)
    assert last["failed_section"] == "queue"
    assert "readonly database" in last["error"]
    assert last["at"] == MORNING, "the sections that did run are still said to have run"

    monkeypatch.undo()
    loop().pass_now(at=LATER, sections=("queue",))
    assert "failed_section" not in Maintenance.last_pass(store), \
        "a healthy pass overwrites the failure rather than accumulating it"


def test_the_pass_writes_down_that_it_ran(store, loop):
    """Status and doctor run in other processes; this line is their only evidence."""
    assert Maintenance.last_pass(store) is None
    report = loop().pass_now(at=MORNING)
    last = Maintenance.last_pass(store)
    assert last["at"] == report["at"]
    assert last["sections"] == list(SECTIONS)
    assert last["created_at"], "the row is stamped when it was written, not when it is read"


def test_the_newest_heartbeat_is_the_one_reported(store, loop):
    loop().pass_now(at=MORNING, sections=("queue",))
    loop().pass_now(at=LATER, sections=("queue",))
    assert Maintenance.last_pass(store)["at"] == LATER


# -- the scheduler -------------------------------------------------------------

def counted():
    calls: list[int] = []
    return calls, lambda: (calls.append(len(calls)), {"ok": True, "at": MORNING})[1]


def test_the_period_is_validated_because_a_loop_with_no_period_is_a_busy_spin():
    with pytest.raises(EvidenceError, match="between 0 and 86400"):
        Ticker(runner=lambda: {}, interval_s=-1)
    with pytest.raises(EvidenceError, match="between 0 and 86400"):
        Ticker(runner=lambda: {}, interval_s="hourly")
    with pytest.raises(EvidenceError, match="between 0 and 86400"):
        Ticker(runner=lambda: {}, interval_s=99_999_999)


def test_a_zero_period_starts_no_thread_at_all():
    calls, runner = counted()
    ticker = Ticker(runner=runner, interval_s=0).start()
    assert ticker.state()["running"] is False and ticker.state()["never"] is True
    assert calls == []


def test_the_loop_runs_on_its_period_and_stops_when_asked():
    calls, runner = counted()
    ticker = Ticker(runner=runner, interval_s=0.02, first_delay_s=0.0).start()
    deadline = time.monotonic() + 5
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    state = ticker.stop(timeout=2)
    assert len(calls) >= 2, "the pass never ran on its own"
    assert state["running"] is False, "stop() left a thread behind"
    assert state["passes"] == len(calls)


def test_a_pass_that_raises_is_counted_rather_than_fatal():
    """A scheduler that dies quietly is worse than a reminder that arrives late."""
    def explodes():
        raise RuntimeError("the archive is locked")

    ticker = Ticker(runner=explodes, interval_s=60)
    assert ticker.tick()["ok"] is False
    assert ticker.state()["failures"] == 1
    assert "RuntimeError: the archive is locked" in ticker.last["error"]
    assert ticker.tick() and ticker.state()["passes"] == 0, "the loop is still alive"


def test_the_counts_that_matter_survive_into_the_state():
    ticker = Ticker(interval_s=60, runner=lambda: {
        "ok": True, "at": MORNING,
        "proactive": {"prepared": 1, "deferred": 2, "suppressed": 0}})
    assert ticker.tick() == {"ok": True, "at": MORNING, "prepared": 1, "deferred": 2,
                             "suppressed": 0}
    assert ticker.state()["passes"] == 1 and ticker.state()["failures"] == 0


def test_a_refused_pass_is_reported_as_refused_rather_than_succeeded():
    ticker = Ticker(interval_s=60, runner=lambda: {"ok": False, "refused": "no store"})
    assert ticker.tick() == {"ok": False, "at": None, "prepared": None, "deferred": None,
                             "suppressed": None}
