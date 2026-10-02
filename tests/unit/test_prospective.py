"""C8 prospective memory: versions, wall time, coverage honesty and the handoff."""
from __future__ import annotations

import pytest

from conftest import envelope
from hermes_memory.prospective.due_events import (CLAIMED, HANDED_OFF, PENDING, SUPPRESSED,
                                                  UNCERTAIN, DueEventLog)
from hermes_memory.prospective.goals import ACTIVE, CANDIDATE, CANCELLED, COMPLETED, GoalStore
from hermes_memory.prospective.predicates import (FAILED, SATISFIED, UNKNOWN,
                                                  resolve_wall_time, evaluate, validate)
from hermes_memory.storage.evidence import EvidenceError
from hermes_memory.storage.identity import IdentityStore
from hermes_memory.ids import now

OWNER = "owner-principal"
ZONE = "Europe/Amsterdam"
TUESDAY = "2026-09-29T09:00"


@pytest.fixture()
def events(store):
    return DueEventLog(store)


@pytest.fixture()
def goals(store, events):
    return GoalStore(store, events=events, owner_principal=OWNER)


@pytest.fixture()
def renewal(goals):
    return goals.propose(title="Renew the certificate",
                         statement="The wildcard certificate expires at the end of September.",
                         timezone_name=ZONE, due=TUESDAY, proposed_by=OWNER,
                         proposed_kind="owner")["id"]


def settle(goals, goal_id, action=COMPLETED, reason="done"):
    return goals.complete(goal_id=goal_id, actor=OWNER, reason=reason) \
        if action == COMPLETED else goals.cancel(goal_id=goal_id, actor=OWNER, reason=reason)


def claim(events, event_id, holder="worker-a", at=1000.0, lease=60.0):
    return events.claim(event_id, holder=holder, lease_s=lease, at=at)


# -- who may create an obligation ---------------------------------------------


def test_an_agents_proposal_schedules_nothing_until_the_owner_activates_it(goals, store):
    proposed = goals.propose(title="Follow up with the plumber",
                             statement="They never called back.",
                             timezone_name=ZONE, due=TUESDAY, proposed_by="agent-model")

    assert proposed["status"] == CANDIDATE
    assert store.db.execute("SELECT count(*) FROM due_events").fetchone()[0] == 0
    assert goals.open(now_iso="2026-10-01T00:00:00+00:00") == []

    with pytest.raises(EvidenceError, match="belongs to the owner"):
        goals.activate(goal_id=proposed["id"], actor="agent-model", reason="do it")

    goals.activate(goal_id=proposed["id"], actor=OWNER, reason="yes, remind me")
    assert store.db.execute("SELECT count(*) FROM due_events WHERE state='pending'"
                            ).fetchone()[0] == 1


def test_the_owner_may_open_a_goal_directly(goals, renewal):
    assert goals.get(renewal).status == ACTIVE
    assert [event["reason"] for event in queued(goals)] == ["due"]


def queued(goals):
    """Every due event on file, oldest first, as plain dictionaries."""
    rows = goals.db.execute("SELECT * FROM due_events ORDER BY revision, fire_at, id") \
        .fetchall()
    return [dict(row) for row in rows]


# -- wall time ------------------------------------------------------------------


def test_a_due_time_is_an_instant_understood_in_the_zone_it_was_said_in(goals, renewal):
    goal = goals.get(renewal)

    assert goal.due_at == "2026-09-29T07:00:00+00:00", "09:00 in Amsterdam"
    assert goal.timezone == ZONE and goal.due_precision == "minute"


def test_a_date_without_a_clock_time_becomes_a_day_not_a_midnight_guess(goals, store):
    out = goals.propose(title="Pay the invoice", statement="Due on the first.",
                        timezone_name=ZONE, due="2026-10-01", proposed_by=OWNER,
                        proposed_kind="owner")

    assert goals.get(out["id"]).due_precision == "day"
    assert goals.get(out["id"]).due_at == "2026-09-30T22:00:00+00:00"


def test_a_spring_forward_gap_is_refused_rather_than_silently_moved(goals):
    with pytest.raises(EvidenceError, match="does not exist"):
        goals.propose(title="Morning standup", statement="At 02:30, which nobody will see.",
                      timezone_name=ZONE, due="2026-03-29T02:30", proposed_by=OWNER,
                      proposed_kind="owner")


def test_a_repeated_autumn_hour_is_refused_because_only_one_is_meant(goals):
    with pytest.raises(EvidenceError, match="happens twice"):
        goals.propose(title="Late check-in", statement="02:00 happens twice that night.",
                      timezone_name=ZONE, due="2026-10-25T02:00", proposed_by=OWNER,
                      proposed_kind="owner")


@pytest.mark.parametrize("wall, zone, precision, expected", [
    ("2026-06-01T12:00", "UTC", "minute", "2026-06-01T12:00:00+00:00"),
    ("2026-06-01T12:00", "Asia/Tokyo", "minute", "2026-06-01T03:00:00+00:00"),
])
def test_the_same_wall_time_in_two_zones_is_two_instants(wall, zone, precision, expected):
    from datetime import timezone

    assert resolve_wall_time(wall, zone, precision).astimezone(timezone.utc).isoformat() \
        == expected


@pytest.mark.parametrize("precision, expected", [
    ("minute", "2026-09-29T13:30:00+00:00"),
    ("hour", "2026-09-29T13:00:00+00:00"),
    ("day", "2026-09-28T22:00:00+00:00"),
    ("week", "2026-09-27T22:00:00+00:00"),
    ("month", "2026-08-31T22:00:00+00:00"),
    ("year", "2025-12-31T23:00:00+00:00"),
])
def test_a_coarse_promise_is_anchored_at_the_start_of_its_period(precision, expected):
    """Every period begins at midnight local, so its instant is that zone's offset away."""
    from datetime import timezone

    moment = resolve_wall_time("2026-09-29T15:30", ZONE, precision)

    assert moment.astimezone(timezone.utc).isoformat() == expected


def test_an_unusable_zone_is_refused_at_the_door(goals):
    with pytest.raises(EvidenceError, match="not usable"):
        goals.propose(title="x", statement="y", timezone_name="Mars/Olympus_Mons",
                      due=TUESDAY, proposed_by=OWNER, proposed_kind="owner")


def test_a_goal_that_expires_before_it_is_due_is_incoherent(goals):
    with pytest.raises(EvidenceError, match="expires before"):
        goals.propose(title="Late notice", statement="Too late by then.",
                      timezone_name="UTC", due="2026-10-01T09:00", expires_at="2026-09-30T09:00",
                      proposed_by=OWNER, proposed_kind="owner")


# -- revisions ------------------------------------------------------------------


def test_revising_a_due_time_supersedes_the_queued_event_rather_than_moving_it(
        goals, events, renewal):
    revised = goals.revise(goal_id=renewal, actor=OWNER, reason="pulled forward",
                           due="2026-09-28T09:00")

    assert revised["revision"] == 2
    assert revised["superseded_events"] == 1
    assert [(item["revision"], item["state"]) for item in queued(goals)] == \
        [(1, "cancelled"), (2, "pending")]
    assert goals.get(renewal).due_at == "2026-09-28T07:00:00+00:00"


def test_history_names_every_decision_and_who_made_it(goals, renewal):
    goals.revise(goal_id=renewal, actor=OWNER, reason="moved a day earlier",
                 due="2026-09-28T09:00")
    settle(goals, renewal)

    log = goals.history(renewal)

    assert [item["revision"] for item in log] == [1, 2, 2]
    assert [item["status"] for item in log] == [ACTIVE, ACTIVE, COMPLETED]
    assert all(item["changed_by"] == OWNER for item in log)


def test_a_finished_goal_is_never_edited_into_an_active_one(goals, renewal):
    settle(goals, renewal)

    with pytest.raises(EvidenceError, match="already|reopen"):
        goals.revise(goal_id=renewal, actor=OWNER, reason="actually not done",
                     due="2026-10-05T09:00")
    assert goals.get(renewal).status == COMPLETED


def test_only_the_owner_may_settle_or_revise(goals, renewal):
    with pytest.raises(EvidenceError, match="belongs to the owner"):
        goals.complete(goal_id=renewal, actor="agent-model", reason="I finished it")
    with pytest.raises(EvidenceError, match="belongs to the owner"):
        goals.cancel(goal_id=renewal, actor="agent-model", reason="not needed")
    assert goals.get(renewal).status == ACTIVE


# -- conditions and coverage ----------------------------------------------------


def store_with_inbox(store, *, last_success="2026-09-25T11:00:00+00:00", coverage="complete"):
    store.db.execute(
        "INSERT INTO connectors(source, generation, policy_version, last_success_at, "
        "coverage_state, updated_at) VALUES('gmail', 1, 'p1', ?, ?, ?)",
        (last_success, coverage, now()))
    return store


def test_a_stale_inbox_cannot_answer_whether_anybody_replied(store):
    store_with_inbox(store, last_success="2026-09-20T09:00:00+00:00")
    verdict = evaluate(store.db, "new_message_from",
                       {"account_id": "acct_" + "0" * 32, "since": "2026-09-21T00:00:00+00:00"},
                       now_iso="2026-09-25T12:00:00+00:00")

    assert verdict.state == UNKNOWN
    assert "last read" in verdict.detail, "a gap is not an absence"


def test_a_source_that_never_reported_coverage_cannot_prove_silence(store):
    verdict = evaluate(store.db, "new_message_from",
                       {"account_id": "acct_" + "0" * 32, "since": "2026-09-21T00:00:00+00:00"},
                       now_iso="2026-09-25T12:00:00+00:00")

    assert verdict.state == UNKNOWN
    assert "ever reported coverage" in verdict.detail


def test_a_disconnected_source_reports_its_coverage_state_not_an_absence(store):
    store_with_inbox(store, coverage="partial")

    verdict = evaluate(store.db, "source_item_update",
                       {"source": "gmail", "source_id": "msg-1",
                        "since": "2026-09-21T00:00:00+00:00"},
                       now_iso="2026-09-25T12:00:00+00:00")

    assert verdict.state == UNKNOWN
    assert "partial" in verdict.detail


def test_a_message_from_a_confirmed_account_satisfies_the_condition(store):
    identity = IdentityStore(store, owner_principal=OWNER)
    account = identity.account("email", "plumber@example.com")
    store_with_inbox(store)
    arrived = store.commit(envelope(observed_at="2026-09-25T11:30:00+00:00",
                                    text="I can come Thursday.",
                                    metadata={"account_ids": [account], "author_account_id": account}))["id"]

    verdict = evaluate(store.db, "new_message_from",
                       {"account_id": account, "since": "2026-09-21T00:00:00+00:00"},
                       now_iso="2026-09-25T12:00:00+00:00")

    assert verdict.state == SATISFIED
    assert verdict.evidence_record_id == arrived


def test_a_fresh_covered_inbox_may_say_nobody_replied(store):
    identity = IdentityStore(store, owner_principal=OWNER)
    account = identity.account("email", "plumber@example.com")
    store.commit(envelope(source_id="other", text="Someone else wrote in."))
    store_with_inbox(store, last_success="2026-09-25T11:59:00+00:00")

    verdict = evaluate(store.db, "new_message_from",
                       {"account_id": account, "since": "2026-09-21T00:00:00+00:00"},
                       now_iso="2026-09-25T12:00:00+00:00")

    assert verdict.state == FAILED


def test_a_measurement_in_another_unit_is_not_an_answer(store):
    from hermes_memory.knowledge.assertions import AssertionStore

    record = store.commit(envelope(text="The humidity in the cellar was 95 percent."))["id"]
    AssertionStore(store, owner_principal=OWNER).propose(
        subject="cellar", predicate="humidity", value="95", kind="measurement", unit="%",
        evidence_kind="explicit_statement", record_id=record, quote="was 95 percent",
        proposed_by=OWNER)

    verdict = evaluate(store.db, "measured_threshold",
                       {"subject": "cellar", "predicate": "humidity", "operator": "<",
                        "value": 60, "unit": "%"}, now_iso="2026-09-25T12:00:00+00:00")
    other = evaluate(store.db, "measured_threshold",
                     {"subject": "cellar", "predicate": "humidity", "operator": "<",
                      "value": 60, "unit": "ratio"}, now_iso="2026-09-25T12:00:00+00:00")

    assert verdict.state == FAILED
    assert other.state == UNKNOWN and "not 'ratio'" in other.detail


@pytest.mark.parametrize("kind, params, message", [
    ("waiting_for", {"who": "the plumber"}, "missing params"),
    ("measured_threshold", {"subject": "s", "predicate": "p", "operator": "~",
                            "value": 1, "unit": "%"}, "operator must be one of"),
    ("due_at", {"at": "next tuesday"}, "not an ISO"),
    ("model_invented_this", {}, "unknown predicate"),
])
def test_a_condition_that_cannot_be_evaluated_never_gets_stored(kind, params, message):
    with pytest.raises(EvidenceError, match=message):
        validate(kind, params)


def test_a_goal_waiting_on_another_sees_its_real_state(goals, store):
    plumber = goals.propose(title="Plumber call-back", statement="Waiting on the quote.",
                            timezone_name=ZONE, proposed_by=OWNER, proposed_kind="owner")["id"]
    follow_up = goals.propose(title="Book the scaffolder",
                              statement="Only once the plumber has confirmed.",
                              timezone_name=ZONE, proposed_by=OWNER, proposed_kind="owner",
                              conditions=[{"kind": "waiting_for",
                                           "params": {"goal_id": plumber}}])["id"]

    assert goals.conditions(follow_up, now_iso="2026-09-26T00:00:00+00:00")[0]["state"] == \
        PENDING

    settle(goals, plumber)
    assert goals.conditions(follow_up, now_iso="2026-09-26T00:00:00+00:00")[0]["state"] == \
        SATISFIED


# -- the handoff ---------------------------------------------------------------


def test_a_claim_that_lapses_becomes_uncertain_rather_than_free(goals, events, renewal):
    event_id = queued(goals)[0]["id"]

    claim = events.claim(event_id, holder="worker-a", lease_s=30, at=1000.0)
    assert claim.event.state == CLAIMED
    expired = events.expire_claims(at=1040.0)

    assert expired == [event_id]
    assert events.get(event_id).state == UNCERTAIN
    with pytest.raises(EvidenceError, match="lease lapsed"):
        events.ack(event_id=event_id, token=claim.token, decision="notify_owner",
                   policy_version="policy-1")
    assert events.intents() == [], "no artifact was invented for a send nobody can account for"


def test_a_lapsed_event_is_not_offered_again_to_avoid_a_second_artifact(goals, events,
                                                                        renewal):
    event_id = queued(goals)[0]["id"]
    events.claim(event_id, holder="worker-a", lease_s=30, at=1000.0)
    events.expire_claims(at=1040.0)

    assert events.claim(event_id, holder="worker-b", at=1050.0) is None


def test_handoff_moves_responsibility_and_says_nothing_about_delivery(goals, events, renewal):
    event_id = queued(goals)[0]["id"]
    held = events.claim(event_id, holder="worker-a", at=1000.0)

    out = events.ack(event_id=event_id, token=held.token, decision="digest",
                     policy_version="policy-1", payload_digest="sha256:abc")

    assert events.get(event_id).state == HANDED_OFF
    intent = events.intents()[0]
    assert intent["kind"] == "digest" and intent["state"] == "prepared"
    assert intent["payload_digest"] == "sha256:abc"
    assert intent["revision"] == 1


def test_repeating_a_handoff_returns_the_intent_that_already_exists(goals, events, renewal):
    event_id = queued(goals)[0]["id"]
    held = events.claim(event_id, holder="worker-a", at=1000.0)
    first = events.ack(event_id=event_id, token=held.token, decision="draft",
                       policy_version="policy-1")

    again = events.ack(event_id=event_id, token=held.token, decision="draft",
                       policy_version="policy-1")

    assert again["replayed"] is True and again["intent"] == first["intent"]
    assert len(events.intents()) == 1


def test_a_suppressed_decision_is_still_a_durable_terminal_decision(goals, events, renewal):
    event_id = queued(goals)[0]["id"]
    held = events.claim(event_id, holder="worker-a", at=1000.0)

    out = events.ack(event_id=event_id, token=held.token, decision="silent",
                     policy_version="policy-1")

    assert events.get(event_id).state == HANDED_OFF
    assert [item["id"] for item in events.intents(state=SUPPRESSED)] == [out["intent"]]
    assert events.intents()[0]["state"] == SUPPRESSED


def test_analyis_required_transfers_responsibility_without_claiming_delivery(
        goals, events, renewal):
    event_id = queued(goals)[0]["id"]
    held = events.claim(event_id, holder="worker-a", at=1000.0)

    out = events.ack(event_id=event_id, token=held.token, decision="awaiting_analysis",
                     policy_version="policy-1")

    assert out["state"] == "awaiting_analysis"
    assert events.get(event_id).state == HANDED_OFF


def test_a_stale_claim_cannot_hand_off(goals, events, renewal):
    event_id = queued(goals)[0]["id"]
    events.claim(event_id, holder="worker-a", at=1000.0)

    with pytest.raises(EvidenceError, match="someone else"):
        events.ack(event_id=event_id, token="stolen", decision="notify_owner",
                   policy_version="policy-1")


def test_an_ack_of_a_superseded_revision_cannot_satisfy_the_current_one(goals, events,
                                                                        renewal):
    stale = queued(goals)[0]["id"]
    held = events.claim(stale, holder="worker-a", at=1000.0)
    goals.revise(goal_id=renewal, actor=OWNER, reason="pulled forward",
                 due="2026-09-28T09:00")

    with pytest.raises(EvidenceError, match="only a live claim"):
        events.ack(event_id=stale, token=held.token, decision="notify_owner",
                   policy_version="policy-1")
    assert events.intents() == []
    assert goals.get(renewal).revision == 2


def test_a_failed_handoff_leaves_the_claim_and_the_event_exactly_as_they_were(
        goals, events, store, renewal):
    """The two writes are one transaction: half a handoff is a lost reminder."""
    event_id = queued(goals)[0]["id"]
    held = events.claim(event_id, holder="worker-a", at=1000.0)

    class HalfWritten:
        """A connection that fails on the second statement of the handoff."""

        def __init__(self, inner):
            self.inner = inner

        @property
        def in_transaction(self):
            return self.inner.in_transaction

        def execute(self, statement, params=()):
            if "decision_intents" in statement:
                raise RuntimeError("the intent table is unwritable")
            return self.inner.execute(statement, params)

    store.db.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(RuntimeError, match="unwritable"):
            events.ack(event_id=event_id, token=held.token, decision="digest",
                       policy_version="policy-1", db=HalfWritten(store.db))
    finally:
        store.db.execute("ROLLBACK")

    assert events.get(event_id).state == CLAIMED
    assert events.intents() == []
    assert store.db.execute("SELECT count(*) FROM due_events WHERE state='handed_off'"
                            ).fetchone()[0] == 0


def test_cancelling_a_goal_while_a_reminder_is_queued_suppresses_it_in_one_transaction(
        store, goals, events, renewal):
    event_id = queued(goals)[0]["id"]

    out = settle(goals, renewal, reason="renewed it by hand")

    assert out["suppressed_events"] == 1
    assert events.get(event_id).state == "cancelled"
    assert events.due("2026-12-25T00:00:00+00:00") == []
    assert [row[0] for row in store.db.execute("SELECT status FROM goals")] == [COMPLETED]


def test_an_already_handed_off_event_survives_a_later_cancellation(goals, events, renewal):
    event_id = queued(goals)[0]["id"]
    held = events.claim(event_id, holder="worker-a", at=1000.0)
    events.ack(event_id=event_id, token=held.token, decision="notify_owner",
               policy_version="policy-1")

    assert settle(goals, renewal)["suppressed_events"] == 0
    assert events.get(event_id).state == HANDED_OFF
    assert len(events.intents()) == 1, "taking back a goal does not take back a send"


def test_a_snooze_holds_the_clock_without_rewriting_the_due_time(goals, renewal, store):
    goals.snooze(goal_id=renewal, actor=OWNER, until="2026-09-30T09:00",
                 reason="on holiday this week")

    assert goals.get(renewal).due_at == "2026-09-29T07:00:00+00:00"
    assert goals.due_now("2026-09-29T12:00:00+00:00") == []
    assert [item["id"] for item in goals.due_now("2026-10-01T12:00:00+00:00")] == \
        [queued(goals)[0]["id"]]


def test_a_snooze_that_suppresses_nothing_is_a_mistake_not_a_feature(goals, renewal):
    with pytest.raises(EvidenceError, match="would suppress nothing"):
        goals.snooze(goal_id=renewal, actor=OWNER, until="2026-09-29T06:00",
                     reason="five minutes")


def test_an_event_is_offered_only_at_its_instant_and_only_once_its_time_has_come(
        goals, events, renewal):
    """A worker ticking early, or a clock that stepped backwards, sees nothing."""
    event_id = queued(goals)[0]["id"]

    assert events.due("2026-09-29T06:59:59+00:00") == []
    assert [item.id for item in events.due("2026-09-29T07:00:00+00:00")] == [event_id]


def test_the_repeated_worker_tick_does_not_duplicate_a_due_event(goals, store, renewal):
    log = DueEventLog(store)
    goal = goals.get(renewal)

    first = log.schedule(goal_id=renewal, revision=goal.revision, fire_at=goal.due_at,
                         timezone=goal.timezone, precision=goal.due_precision, reason="due")
    second = log.schedule(goal_id=renewal, revision=goal.revision, fire_at=goal.due_at,
                          timezone=goal.timezone, precision=goal.due_precision,
                          reason="due")

    # propose() already queued this occurrence, so both calls must be no-ops.
    assert first["scheduled"] is False and second["scheduled"] is False
    assert store.db.execute("SELECT count(*) FROM due_events").fetchone()[0] == 1


def test_a_goal_past_its_expiry_leaves_the_open_list_rather_than_looming(goals, renewal):
    goals.propose(title="Book the referee", statement="For the Saturday match.",
                  timezone_name="UTC", due="2026-10-02T09:00", expires_at="2026-10-05T09:00",
                  proposed_by=OWNER, proposed_kind="owner")

    assert len(goals.open(now_iso="2026-10-01T00:00:00+00:00")) == 2
    assert [item.title for item in goals.open(now_iso="2026-10-06T00:00:00+00:00")] == \
        ["Renew the certificate"]
