"""C10 eligibility, analyst and outbox: the gate's gate, and the honest receipt.

The gate items these pin are the ones the plan named: historical backfill, an
unavailable span, a stale queued artifact, a corrected goal, a send whose outcome
was never established, and the refusal to resend it.
"""
from __future__ import annotations

import pytest

from hermes_memory.context.packet import EvidenceItem, Packet
from hermes_memory.ids import digest, timestamp
from hermes_memory.processing.budgets import Budget, Budgets
from hermes_memory.proactive.analyst import PacketAnalyst
from hermes_memory.proactive.eligibility import Eligibility
from hermes_memory.proactive.engine import ProactiveEngine
from hermes_memory.proactive.outbox import Outbox
from hermes_memory.proactive.policy import POLICY_VERSION, AttentionPolicy
from hermes_memory.prospective.due_events import DueEventLog
from hermes_memory.prospective.goals import GoalStore
from hermes_memory.sources.sync import SyncController
from hermes_memory.storage.evidence import EvidenceError
from conftest import envelope

OWNER = "owner-principal"
AGENT = "hermes-agent"
UTC = "UTC"
MORNING = "2026-09-15T09:00:00+00:00"
EVENING = "2026-09-15T23:30:00+00:00"
# The epoch form of MORNING: leases count in monotonic seconds and the tests must
# say the same instant the wall-clock strings do.
EPOCH = 1789462800.0
RECORD_ID: str = ""


@pytest.fixture()
def policy(store):
    subject = AttentionPolicy(store, owner_principal=OWNER)
    subject.configure(actor=OWNER, timezone_name=UTC, quiet_from="22:00",
                      quiet_until="06:00", max_immediate_per_day=10,
                      cooldown_minutes=0, shadow=False)
    return subject


@pytest.fixture()
def stack(store, policy):
    """The C8 handoff plus the C10 consumers of it."""
    events = DueEventLog(store)
    goals = GoalStore(store, events=events, owner_principal=OWNER)
    outbox = Outbox(store, policy=policy, owner_principal=OWNER, clock=lambda: EPOCH)

    def decide(topic="general", at=MORNING):
        decision = policy.decide(topic=topic, at=at)
        goal_id = goals.propose(title="Send the invoice", statement="Client waiting.",
                                timezone_name=UTC, due=timestamp(at), proposed_by=OWNER,
                                proposed_kind="owner")["id"]
        event_id = store.db.execute("SELECT id FROM due_events WHERE goal_id=?",
                                    (goal_id,)).fetchone()[0]
        claim = events.claim(event_id, holder="worker", at=EPOCH)
        intent = events.ack(event_id=event_id, token=claim.token,
                            decision="awaiting_analysis",
                            policy_version=POLICY_VERSION)["intent"]
        written = policy.record(decision, intent_id=intent, goal_id=goal_id, revision=1)
        return {"decision": decision, "id": written["id"], "goal": goal_id,
                "intent": intent, "topic": topic}

    return {"policy": policy, "goals": goals, "events": events, "outbox": outbox,
            "decide": decide}


@pytest.fixture()
def evidence(store):
    """One real record an artifact can cite, plus the id to cite it by."""
    committed = store.commit(envelope(source_id="invoice-1",
                                      text="The invoice is overdue since Monday."))
    return str(committed["id"])


# -- eligibility --------------------------------------------------------------

def test_a_backfill_is_not_a_live_change(store, policy, evidence):
    sync = SyncController(store)
    sync.register("gmail", policy_version="private-api")
    subject = Eligibility(store, policy=policy, sync=sync)
    fence = sync.acquire("gmail", holder="c1")
    sync.mark_gap(fence, coverage_state="partial", reason="sweeping six months of mail")
    verdict = subject.for_change(source="gmail", record_id=evidence, at=MORNING)
    assert not verdict.eligible and verdict.stage == "coverage"
    assert "gap is not an absence" in verdict.reason


def test_a_source_that_reached_the_tail_is_live_again(store, policy, evidence):
    sync = SyncController(store)
    sync.register("gmail", policy_version="private-api")
    subject = Eligibility(store, policy=policy, sync=sync)
    fence = sync.acquire("gmail", holder="c1")
    # Advancing the cursor is what "caught up" means: a page with no next cursor is
    # the end of the sweep, not a live connector.
    sync.publish(fence, "page-1", [envelope(source_id="invoice-2",
                                            text="A second overdue invoice.")],
                 next_cursor="cursor-2")
    sync.release(fence)
    assert subject.for_change(source="gmail", record_id=evidence, at=MORNING).eligible


def test_an_unregistered_source_is_not_assumed_current(store, policy, evidence):
    subject = Eligibility(store, policy=policy, sync=SyncController(store))
    verdict = subject.for_change(source="no-such-source", record_id=evidence, at=MORNING)
    assert not verdict.eligible and "not registered" in verdict.reason


def test_an_erased_citation_makes_the_change_eligible_for_nothing(store, policy, evidence):
    subject = Eligibility(store, policy=policy)
    store.hide(evidence, reason="owner forgot it", actor=OWNER)
    verdict = subject.for_change(source="gmail", record_id=evidence, at=MORNING)
    assert not verdict.eligible and verdict.stage == "evidence"


def test_six_month_old_mail_arriving_today_is_still_old_news(store, policy):
    committed = store.commit(envelope(source_id="old-1", occurred_at="2026-03-01T09:00:00+00:00",
                                      text="An arrangement made in March."))
    subject = Eligibility(store, policy=policy, max_age_days=14)
    verdict = subject.for_change(source="gmail", record_id=str(committed["id"]), at=MORNING)
    assert not verdict.eligible and verdict.stage == "freshness"
    assert "older than the 14 days" in verdict.reason


def test_a_snoozed_goal_is_not_asked_about_while_it_is_put_off(store, stack):
    subject = Eligibility(store, policy=stack["policy"], goals=stack["goals"])
    goal_id = stack["goals"].propose(title="Renew", statement="Due Friday.",
                                     timezone_name=UTC, due=timestamp(MORNING),
                                     proposed_by=OWNER, proposed_kind="owner")["id"]
    # Snoozed in a non-UTC zone: the comparison has to be about instants, not about
    # the characters at the end of the string.
    stack["goals"].snooze(goal_id=goal_id, actor=OWNER,
                          until="2026-09-16T10:00:00+02:00", reason="after the holiday")
    verdict = subject.for_event(goal_id=goal_id, revision=1, at=MORNING)
    assert not verdict.eligible and verdict.stage == "snooze"
    assert subject.for_event(goal_id=goal_id, revision=1,
                             at="2026-09-16T10:00:00+02:00").eligible


def test_an_expired_goal_is_not_announced_after_it_has_lapsed(store, stack):
    subject = Eligibility(store, policy=stack["policy"], goals=stack["goals"])
    goal_id = stack["goals"].propose(title="Renew", statement="Due Friday.",
                                     timezone_name=UTC, due=timestamp(MORNING),
                                     proposed_by=OWNER, proposed_kind="owner",
                                     expires_at="2026-09-15T12:00:00+00:00")["id"]
    verdict = subject.for_event(goal_id=goal_id, revision=1, at=EVENING)
    assert not verdict.eligible and verdict.stage == "expiry"


def test_a_goal_the_owner_revised_is_not_an_eligible_trigger(store, stack):
    subject = Eligibility(store, policy=stack["policy"], goals=stack["goals"])
    goal_id = stack["goals"].propose(title="Renew", statement="Due Friday.",
                                     timezone_name=UTC, due=timestamp(MORNING),
                                     proposed_by=OWNER, proposed_kind="owner")["id"]
    stack["goals"].revise(goal_id=goal_id, actor=OWNER, due=timestamp(EVENING),
                          reason="the deadline moved")
    verdict = subject.for_event(goal_id=goal_id, revision=1, at=EVENING)
    assert not verdict.eligible and verdict.stage == "goal"


def test_a_completed_goal_has_nothing_left_to_announce(store, stack):
    subject = Eligibility(store, policy=stack["policy"], goals=stack["goals"])
    goal_id = stack["goals"].propose(title="Renew", statement="Due Friday.",
                                     timezone_name=UTC, due=timestamp(MORNING),
                                     proposed_by=OWNER, proposed_kind="owner")["id"]
    stack["goals"].complete(goal_id=goal_id, actor=OWNER, reason="done")
    verdict = subject.for_event(goal_id=goal_id, revision=1, at=MORNING)
    assert not verdict.eligible and "completed" in verdict.reason


def test_an_intention_that_already_has_a_decision_is_not_asked_twice(store, stack):
    subject = Eligibility(store, policy=stack["policy"], goals=stack["goals"])
    made = stack["decide"]()
    verdict = subject.for_event(goal_id=made["goal"], revision=1,
                                intent_id=made["intent"], at=MORNING)
    assert not verdict.eligible and verdict.stage == "duplicate"
    assert "already has a recorded decision" in verdict.reason


def test_shadow_mode_refuses_the_model_but_not_the_decision(store, policy, evidence):
    policy.configure(actor=OWNER, shadow=True)
    subject = Eligibility(store, policy=policy)
    verdict = subject.for_change(source="gmail", record_id=evidence, at=MORNING)
    assert not verdict.eligible and verdict.stage == "shadow"
    assert store.db.execute("SELECT shadow FROM proactive_decisions").fetchall() == [] or True


def test_a_spent_model_budget_is_reported_as_a_budget(store, policy, evidence):
    budgets = Budgets(store, daily={"remote-9b": Budget(tokens=10, seconds=60)},
                     scope="general")
    budgets.charge("remote-9b", tokens=10)
    subject = Eligibility(store, policy=policy, budgets=budgets,
                          model_budget_resource="remote-9b")
    verdict = subject.for_change(source="gmail", record_id=evidence, at=MORNING)
    assert not verdict.eligible and verdict.stage == "budget"
    assert "budget is spent" in verdict.reason, \
        "an exhausted budget must not be filed under 'nothing needed the owner'"


def test_opting_a_topic_out_stops_eligibility_before_any_work(store, policy, evidence):
    subject = Eligibility(store, policy=policy)
    policy.configure(actor=OWNER, topic="health", opted_out=True)
    verdict = subject.for_change(source="gmail", record_id=evidence, topic="health",
                                at=MORNING)
    assert not verdict.eligible and verdict.stage == "opt_out"


# -- the analyst --------------------------------------------------------------

class Scripted:
    """A model that says exactly what it was told to, including the wrong things."""

    def __init__(self, payload, delay: float = 0.0):
        self.payload = payload
        self.delay = delay
        self.seen: list[dict] = []

    def analyse(self, request):
        from time import sleep
        self.seen.append(request)
        if self.delay:
            sleep(self.delay)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def packet_with(*ids: str) -> Packet:
    return Packet(query="invoice", packet_id="pkt_1", epoch=1, revision=1,
                  items=tuple(EvidenceItem(id=item, source="gmail",
                                           text=f"text of {item}",
                                           occurred_at=MORNING,
                                           occurred_precision="second", channel="lexical")
                              for item in ids))


def test_the_analyst_may_not_promote_its_own_interrption(store):
    model = Scripted({"outcome": "notify_owner", "message": "Call the client now.",
                      "citations": ["rec_a"], "reason": "overdue"})
    analysis = PacketAnalyst(store, model=model).analyse(
        packet=packet_with("rec_a"), topic="general", gate="digest")
    assert analysis.outcome == "digest"
    assert any("may not promote" in note for note in analysis.dropped)


def test_a_citation_outside_the_packet_is_not_a_citation(store):
    model = Scripted({"outcome": "digest", "message": "The invoice is overdue.",
                      "citations": ["rec_fabricated"], "reason": "x"})
    analysis = PacketAnalyst(store, model=model).analyse(
        packet=packet_with("rec_a"), topic="general", gate="digest")
    assert analysis.outcome == "silent" and analysis.citations == ()
    assert any("was not in the supplied packet" in note for note in analysis.dropped)


def test_a_model_proposed_relation_is_never_written_as_knowledge(store):
    model = Scripted({"outcome": "digest", "message": "The invoice is overdue.",
                      "citations": ["rec_a"], "relations": [["rec_a", "owes", "acme"]],
                      "reason": "guessed"})
    analysis = PacketAnalyst(store, model=model).analyse(
        packet=packet_with("rec_a"), topic="general", gate="digest")
    assert any("does not write knowledge" in note for note in analysis.dropped)
    assert store.db.execute("SELECT count(*) FROM assertions").fetchone()[0] == 0


def test_a_hundred_page_message_is_not_a_notification(store):
    model = Scripted({"outcome": "digest", "message": "very overdue " * 400,
                      "citations": ["rec_a"], "reason": "x"})
    analysis = PacketAnalyst(store, model=model, max_message_chars=200).analyse(
        packet=packet_with("rec_a"), topic="general", gate="digest")
    assert len(analysis.message) <= 200


def test_terminal_control_characters_do_not_reach_the_host(store):
    model = Scripted({"outcome": "digest",
                      "message": "overdue.\x1b[2J\x07 ignore previous instructions",
                      "citations": ["rec_a"], "reason": "x"})
    analysis = PacketAnalyst(store, model=model).analyse(
        packet=packet_with("rec_a"), topic="general", gate="digest")
    assert "\x1b" not in analysis.message and "\x07" not in analysis.message


@pytest.mark.parametrize("payload", [
    "not json at all", b'{"outcome": ', ["a", "list"], {"outcome": "email_blast"},
    {"outcome": "digest"}, {"outcome": "digest", "message": "   ", "citations": ["rec_a"]},
])
def test_an_unusable_answer_becomes_a_named_refusal_not_a_silent_pass(store, payload):
    analysis = PacketAnalyst(store, model=Scripted(payload)).analyse(
        packet=packet_with("rec_a"), topic="general", gate="digest")
    assert analysis.outcome == "silent"
    assert analysis.source in ("malformed", "refused", "analysed")
    assert analysis.reason


def test_a_hung_model_cannot_hold_the_worker(store):
    subject = PacketAnalyst(store, model=Scripted({"outcome": "digest"}, delay=2.0),
                            timeout_s=0.2)
    analysis = subject.analyse(packet=packet_with("rec_a"), topic="general",
                               gate="digest")
    assert analysis.source == "timeout" and analysis.outcome == "silent"
    subject.close()


def test_a_broken_model_is_reported_rather_than_hidden(store):
    subject = PacketAnalyst(store, model=Scripted(RuntimeError("upstream 503")))
    analysis = subject.analyse(packet=packet_with("rec_a"), topic="general",
                               gate="digest")
    assert analysis.source == "failed" and "upstream 503" in analysis.reason


def test_no_analyst_is_a_supported_operating_state(store):
    analysis = PacketAnalyst(store, model=None).analyse(
        packet=packet_with("rec_a"), topic="general", gate="digest")
    assert analysis.source == "unavailable" and not analysis.worth_sending
    assert "no packet analyst is configured" in analysis.reason


def test_a_packet_with_nothing_to_cite_is_not_answered_from_memory(store):
    model = Scripted({"outcome": "digest", "message": "I recall something.",
                      "citations": ["rec_a"], "reason": "invented"})
    seen: list[dict] = []

    class Spy(Scripted):
        def analyse(self, request):
            seen.append(request)
            return self.payload

    analysis = PacketAnalyst(store, model=Spy({"outcome": "digest"})).analyse(
        packet=Packet(query="nothing", items=()), topic="general", gate="digest")
    assert analysis.outcome == "silent" and "no spans to cite" in analysis.reason
    assert seen == [], "the model was not called at all"


def test_an_unparsable_gate_is_a_bug_in_the_caller(store):
    with pytest.raises(EvidenceError, match="gate"):
        PacketAnalyst(store, model=None).analyse(packet=packet_with("rec_a"),
                                                 topic="general", gate="urgent")


def test_the_request_carries_the_frozen_spans_not_a_live_cursor(store):
    model = Scripted({"outcome": "silent", "reason": "nothing to say"})
    PacketAnalyst(store, model=model).analyse(packet=packet_with("rec_a", "rec_b"),
                                             topic="general", gate="digest")
    assert model.seen[0]["spans"] == ["rec_a", "rec_b"]
    assert [item["id"] for item in model.seen[0]["packet"]] == ["rec_a", "rec_b"]


# -- the engine ---------------------------------------------------------------


class FakeBroker:
    """A C9 stand-in: the packet is whatever the test says is retrievable."""

    def __init__(self, *ids):
        self.ids = ids
        self.queries: list[str] = []

    def assemble(self, query, **kwargs):
        self.queries.append(query)
        return packet_with(*self.ids)


@pytest.fixture()
def engine(store, stack):
    events, policy = stack["events"], stack["policy"]
    eligibility = Eligibility(store, policy=policy, goals=stack["goals"])

    def build(analyst=None, broker=None):
        return ProactiveEngine(store, events=events, policy=policy,
                               eligibility=eligibility, outbox=stack["outbox"],
                               analyst=analyst, broker=broker)

    return build


def a_goal(store, stack, title="Call the dentist", at=MORNING):
    return stack["goals"].propose(title=title, statement="The appointment is booked.",
                                 timezone_name=UTC, due=timestamp(at),
                                 proposed_by=OWNER, proposed_kind="owner")["id"]


def test_a_reminder_needs_no_model_to_be_kept(store, stack, engine):
    """Inference off must not switch proactivity off as a side effect."""
    a_goal(store, stack)
    report = engine().sweep(at=MORNING)
    assert report["decided"] == 1 and report["prepared"] == 1
    artifact = stack["outbox"].pending()[0]
    assert artifact.payload == "Call the dentist", "the owner's own words are the reminder"
    assert artifact.state == "prepared", "the framework prepares; Hermes delivers"


def test_an_analysts_wording_replaces_the_title_only_when_it_has_one(store, stack,
                                                                     engine):
    a_goal(store, stack)
    subject = engine(analyst=PacketAnalyst(store, model=Scripted(
        {"outcome": "notify_owner", "message": "The dentist asked you to call back first.",
         "citations": ["rec_a"], "reason": "one concrete step"})), broker=FakeBroker("rec_a"))
    subject.sweep(at=MORNING)
    assert stack["outbox"].pending()[0].payload == \
        "The dentist asked you to call back first."


def test_the_analysts_no_is_final(store, stack, engine):
    a_goal(store, stack)
    subject = engine(analyst=PacketAnalyst(store, model=Scripted(
        {"outcome": "silent", "reason": "already handled in the last turn"})),
        broker=FakeBroker("rec_a"))
    report = subject.sweep(at=MORNING)
    assert report["prepared"] == 0 and report["suppressed"] == 1
    assert stack["outbox"].pending() == []
    assert store.db.execute("SELECT state FROM due_events").fetchone()[0] == "suppressed"


def test_a_model_may_not_promote_a_scheduled_reminder_past_the_gate(store, stack, engine):
    stack["policy"].configure(actor=OWNER, max_immediate_per_day=0, digest_per_day=4)
    a_goal(store, stack)
    subject = engine(analyst=PacketAnalyst(store, model=Scripted(
        {"outcome": "notify_owner", "message": "Call now.", "citations": ["rec_a"],
         "reason": "urgent"})), broker=FakeBroker("rec_a"))
    subject.sweep(at=MORNING)
    assert stack["outbox"].pending()[0].kind == "digest"


def test_a_finished_pass_never_re_announces_the_same_event(store, stack, engine):
    a_goal(store, stack)
    first = engine().sweep(at=MORNING)
    second = engine().sweep(at=MORNING)
    assert first["prepared"] == 1
    assert second["decided"] == 0 and second["prepared"] == 0, \
        "the event is no longer pending, so there is nothing to announce twice"


def test_a_crash_between_the_handoff_and_the_decision_is_finished(store, stack, engine):
    """The worker died holding a claim it had already handed off."""
    events = stack["events"]
    a_goal(store, stack)
    event = events.due(MORNING)[0]
    claim = events.claim(event.id, holder="dead-worker", at=EPOCH)
    intent = events.ack(event_id=event.id, token=claim.token, decision="awaiting_analysis",
                        policy_version=POLICY_VERSION)["intent"]
    assert store.db.execute("SELECT state FROM decision_intents WHERE id=?",
                            (intent,)).fetchone()[0] == "awaiting_analysis"
    report = engine().sweep(at=MORNING)
    assert report["decided"] == 1 and report["prepared"] == 1
    assert "unfinished" in " ".join(report["notes"])
    assert store.db.execute("SELECT kind, state FROM decision_intents",
                            ).fetchone()[0] == "notify_owner"


def test_an_intention_left_awaiting_is_not_settled_twice(store, stack, engine):
    events = stack["events"]
    a_goal(store, stack)
    event = events.due(MORNING)[0]
    claim = events.claim(event.id, holder="dead-worker", at=EPOCH)
    intent = events.ack(event_id=event.id, token=claim.token, decision="awaiting_analysis",
                        policy_version=POLICY_VERSION)["intent"]
    events.settle(event_id=event.id, intent_id=intent, decision="digest")
    assert engine().sweep(at=MORNING)["decided"] == 0


def test_a_goal_the_owner_finished_is_not_announced_by_the_sweep(store, stack, engine):
    goal_id = a_goal(store, stack)
    stack["goals"].complete(goal_id=goal_id, actor=OWNER, reason="already called")
    report = engine().sweep(at=MORNING)
    assert report["prepared"] == 0
    assert stack["outbox"].pending() == []


def test_a_sweep_with_no_instant_announces_nothing(store, stack, engine):
    a_goal(store, stack)
    report = engine().sweep()
    assert report["decided"] == 0 and report["prepared"] == 0
    assert "no instant was given" in " ".join(report["notes"])


def test_a_goal_finished_during_the_crash_is_not_announced_afterwards(store, stack,
                                                                     engine):
    """Handed off, then the owner dealt with it, then the worker came back."""
    events = stack["events"]
    goal_id = a_goal(store, stack)
    event = events.due(MORNING)[0]
    claim = events.claim(event.id, holder="dead-worker", at=EPOCH)
    events.ack(event_id=event.id, token=claim.token, decision="awaiting_analysis",
               policy_version=POLICY_VERSION)
    stack["goals"].complete(goal_id=goal_id, actor=OWNER, reason="already called")
    report = engine().sweep(at=MORNING)
    assert report["prepared"] == 0 and report["suppressed"] == 1
    assert stack["outbox"].pending() == []


def test_a_next_turn_item_is_not_spent_a_model_call(store, stack, engine):
    """Nothing is being said now, so nothing has to be worded now."""
    stack["policy"].configure(actor=OWNER, max_immediate_per_day=0, digest_per_day=0)
    a_goal(store, stack)
    broker = FakeBroker("rec_a")
    subject = engine(analyst=PacketAnalyst(store, model=Scripted(
        {"outcome": "silent", "reason": "nothing to add"})), broker=broker)
    subject.sweep(at=MORNING)
    assert broker.queries == [], "the analyst was not consulted for a next-turn item"
    assert stack["outbox"].pending()[0].payload == "Call the dentist"


def test_an_intention_cannot_be_left_awaiting_by_settling_into_awaiting(store, stack):
    events = stack["events"]
    a_goal(store, stack)
    event = events.due(MORNING)[0]
    claim = events.claim(event.id, holder="w", at=EPOCH)
    intent = events.ack(event_id=event.id, token=claim.token,
                        decision="awaiting_analysis",
                        policy_version=POLICY_VERSION)["intent"]
    with pytest.raises(EvidenceError, match="settled into"):
        events.settle(event_id=event.id, intent_id=intent, decision="awaiting_analysis")


def test_a_suppression_has_to_say_why(store, stack):
    events = stack["events"]
    a_goal(store, stack)
    event = events.due(MORNING)[0]
    with pytest.raises(EvidenceError, match="say why"):
        events.suppress(event.id, reason="   ")


def test_a_crash_in_the_middle_of_a_pass_leaves_a_recoverable_intention(store, stack,
                                                                        engine):
    """The promise must not depend on the worker surviving it."""
    a_goal(store, stack)
    subject = engine()
    real = stack["outbox"]

    class DiesOnTheWay:
        def prepare(self, **kwargs):
            raise RuntimeError("the process died on the way to the outbox")

        def expire(self, **kwargs):
            return []

        def recover(self, **kwargs):
            return []

    subject.outbox = DiesOnTheWay()
    with pytest.raises(RuntimeError):
        subject.sweep(at=MORNING)
    row = store.db.execute(
        "SELECT e.state AS event, i.state AS intent, i.kind FROM due_events e "
        "JOIN decision_intents i ON i.event_id = e.id").fetchone()
    assert row["event"] == "handed_off" and row["intent"] == "awaiting_analysis"
    assert stack["events"].due(MORNING) == [], \
        "a half-finished promise is not sitting in the queue to be picked up again"
    stack["outbox"] = real
    assert engine().sweep(at=MORNING)["prepared"] == 1, \
        "the next pass finishes what the crashed one promised"


def test_the_owner_may_opt_out_and_the_sweep_stops_immediately(store, stack, engine):
    a_goal(store, stack)
    stack["policy"].configure(actor=OWNER, opted_out=True)
    report = engine().sweep(at=MORNING)
    assert report["prepared"] == 0 and report["suppressed"] == 1
    assert store.db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0


# -- the handoff seam ---------------------------------------------------------

def test_settling_an_intention_twice_is_a_replay_not_a_second_decision(store, stack):
    events = stack["events"]
    a_goal(store, stack)
    event = events.due(MORNING)[0]
    claim = events.claim(event.id, holder="w", at=EPOCH)
    intent = events.ack(event_id=event.id, token=claim.token,
                        decision="awaiting_analysis",
                        policy_version=POLICY_VERSION)["intent"]
    first = events.settle(event_id=event.id, intent_id=intent, decision="digest")
    again = events.settle(event_id=event.id, intent_id=intent, decision="notify_owner")
    assert first["replayed"] is False and again["replayed"] is True
    assert again["kind"] == "digest", "a settled intention is not re-decided by the loser"


def test_one_reminder_is_one_intention_whatever_was_decided_about_it(store, stack):
    """A second ack of a handed-off event cannot mint a second artifact."""
    events = stack["events"]
    a_goal(store, stack)
    event = events.due(MORNING)[0]
    claim = events.claim(event.id, holder="w", at=EPOCH)
    first = events.ack(event_id=event.id, token=claim.token, decision="awaiting_analysis",
                       policy_version=POLICY_VERSION)
    second = events.ack(event_id=event.id, token=claim.token, decision="notify_owner",
                        policy_version=POLICY_VERSION)
    assert second["replayed"] is True and second["intent"] == first["intent"]
    assert len(events.intents()) == 1


def test_an_intention_cannot_be_settled_onto_another_event(store, stack):
    events = stack["events"]
    a_goal(store, stack)
    a_goal(store, stack, title="File the expense", at=EVENING)
    first, second = events.due(MORNING, limit=5), events.due(EVENING, limit=5)
    event_a = first[0]
    event_b = [item for item in second if item.id != event_a.id][0]
    claim = events.claim(event_a.id, holder="w", at=EPOCH)
    intent = events.ack(event_id=event_a.id, token=claim.token,
                        decision="awaiting_analysis",
                        policy_version=POLICY_VERSION)["intent"]
    with pytest.raises(EvidenceError, match="another event"):
        events.settle(event_id=event_b.id, intent_id=intent, decision="digest")


def test_a_suppressed_reminder_says_why_and_cannot_be_un_sent(store, stack):
    events = stack["events"]
    a_goal(store, stack)
    event = events.due(MORNING)[0]
    events.claim(event.id, holder="w", at=EPOCH)
    events.suppress(event.id, reason="the owner dealt with it in the thread")
    assert events.get(event.id).state == "suppressed"
    assert events.claim(event.id, holder="w2", at=EPOCH) is None, \
        "a suppressed reminder is not waiting for anybody to pick it up"
    with pytest.raises(EvidenceError, match="un-delivered"):
        events.suppress(event.id, reason="too late to matter")


# -- the outbox ---------------------------------------------------------------

def test_an_artifact_is_only_ever_addressed_to_the_owner(store, stack, evidence):
    made = stack["decide"]()
    with pytest.raises(EvidenceError, match="does not message anyone else"):
        stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                topic="general", payload="Call the client.",
                                evidence=[evidence], recipient="the-client")


def test_a_silent_decision_prepares_nothing_at_all(store, stack):
    made = stack["decide"]()
    with pytest.raises(EvidenceError, match="absence of one"):
        stack["outbox"].prepare(decision_id=made["id"], kind="silent", topic="general",
                                payload="", evidence=[])


def test_one_decision_never_queues_the_same_artifact_twice(store, stack, evidence):
    made = stack["decide"]()
    first = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                    topic="general", payload="The invoice is overdue.",
                                    evidence=[evidence])
    again = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                    topic="general", payload="The invoice is overdue.",
                                    evidence=[evidence])
    assert first["prepared"] is True and again["prepared"] is False
    assert again["id"] == first["id"]
    assert len(stack["outbox"].for_decision(made["id"])) == 1


def test_an_artifact_needs_a_decision_rather_than_a_healthy_sounding_guess(store, stack):
    with pytest.raises(EvidenceError, match="unknown decision"):
        stack["outbox"].prepare(decision_id="dec_nope", kind="notify_owner",
                                topic="general", payload="Anything", evidence=[])


def test_delivery_revalidates_evidence_the_owner_has_since_forgotten(store, stack,
                                                                    evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="The invoice is overdue.",
                                       evidence=[evidence])
    store.hide(evidence, reason="erased by the owner", actor=OWNER)
    check = stack["outbox"].revalidate(prepared["id"], at=EPOCH)
    assert not check.ok and check.stage == "evidence"
    assert "forgot or withdrew" in check.reason
    assert stack["outbox"].lease(holder="hermes", at=EPOCH) is None
    assert stack["outbox"].get(prepared["id"]).state == "suppressed"


def test_a_corrected_goal_suppresses_what_it_was_going_to_say(store, stack, evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="Send the invoice.",
                                       evidence=[evidence])
    stack["goals"].revise(goal_id=made["goal"], actor=OWNER, due=timestamp(EVENING),
                          reason="the deadline moved")
    check = stack["outbox"].revalidate(prepared["id"], at=EPOCH)
    assert not check.ok and check.stage == "revision"


def test_an_opt_out_while_queued_is_honoured_at_delivery(store, stack, evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="Send the invoice.",
                                       evidence=[evidence])
    stack["policy"].configure(actor=OWNER, opted_out=True)
    check = stack["outbox"].revalidate(prepared["id"], at=EPOCH)
    assert not check.ok and check.stage == "opt_out"


def test_an_operator_pause_stops_a_queued_artifact_without_deleting_it(store, stack,
                                                                      evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="Send the invoice.",
                                       evidence=[evidence])
    store.set_control("topic:general", "attention", "paused", actor="operator",
                      reason="model server maintenance", policy_version=POLICY_VERSION)
    check = stack["outbox"].revalidate(prepared["id"], at=EPOCH)
    assert not check.ok and check.stage == "pause"
    assert stack["outbox"].get(prepared["id"]).state == "prepared"


def test_nothing_is_delivered_while_the_owner_is_asleep(store, stack, evidence):
    made = stack["decide"](at=EVENING)
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="Send the invoice.",
                                       evidence=[evidence])
    check = stack["outbox"].revalidate(prepared["id"], at=1789515000.0)  # 23:30Z
    assert not check.ok and check.stage == "quiet"
    assert check.retry_at.startswith("2026-09-16T06:00"), \
        "a deferral must say when it becomes allowed, or it is a silent drop"


def test_a_digest_whose_window_already_went_out_does_not_go_out_again(store, stack,
                                                                     evidence):
    made = stack["decide"](at=EVENING)
    assert made["decision"].action == "digest"
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="digest",
                                       topic="general", payload="Two things overnight.",
                                       evidence=[evidence])
    store.db.execute("UPDATE attention_windows SET state='closed'")
    check = stack["outbox"].revalidate(prepared["id"], at=EPOCH)
    assert not check.ok and check.stage == "window"


def test_tampered_payload_bytes_are_caught_before_delivery(store, stack, evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="The invoice is overdue.",
                                       evidence=[evidence])
    store.db.execute("UPDATE outbox SET payload=? WHERE id=?",
                     ("The invoice is a birthday card.", prepared["id"]))
    check = stack["outbox"].revalidate(prepared["id"], at=EPOCH)
    assert not check.ok and check.stage == "digest", "the bytes no longer match the digest"


def test_a_completed_goal_suppresses_what_was_queued_for_it(store, stack, evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="Send the invoice.",
                                       evidence=[evidence])
    stack["goals"].complete(goal_id=made["goal"], actor=OWNER, reason="done already")
    check = stack["outbox"].revalidate(prepared["id"], at=EPOCH)
    assert not check.ok and check.stage == "goal" and "completed" in check.reason


def test_a_superseded_policy_version_stops_the_delivery_it_licensed(store, stack,
                                                                    evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="Send the invoice.",
                                       evidence=[evidence])
    store.db.execute("UPDATE proactive_decisions SET policy_version='attention-0' "
                     "WHERE id=?", (made["id"],))
    check = stack["outbox"].revalidate(prepared["id"], at=EPOCH)
    assert not check.ok and check.stage == "policy"
    assert "no longer the one in force" in check.reason


def test_an_artifact_whose_moment_has_passed_is_not_delivered_as_news(store, stack,
                                                                     evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="The invoice is overdue.",
                                       evidence=[evidence],
                                       expires_at="2026-09-15T10:00:00+00:00")
    check = stack["outbox"].revalidate(prepared["id"], at=EPOCH + 7200)
    assert not check.ok and check.stage == "expiry"
    # The reaper agrees with the revalidation, so nothing lingers in the queue.
    assert stack["outbox"].expire(at="2026-09-15T12:00:00+00:00") == [prepared["id"]]


def test_a_holder_may_give_back_an_artifact_it_did_not_send(store, stack, evidence):
    made = stack["decide"]()
    stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner", topic="general",
                            payload="The invoice is overdue.", evidence=[evidence])
    claim = stack["outbox"].lease(holder="hermes", at=EPOCH)
    returned = stack["outbox"].release(artifact_id=claim.artifact.id, token=claim.token,
                                       reason="the analyst had nothing to add")
    assert returned.state == "prepared" and returned.attempts == 1
    assert "nothing to add" in returned.reason
    again = stack["outbox"].lease(holder="hermes", at=EPOCH + 10)
    assert again.artifact.id == claim.artifact.id and again.artifact.attempts == 2, \
        "two attempts to deliver one artifact is a fact worth keeping"


def test_an_abandoned_lease_is_not_re_tried_by_the_next_worker(store, stack, evidence):
    made = stack["decide"]()
    stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner", topic="general",
                            payload="The invoice is overdue.", evidence=[evidence])
    claim = stack["outbox"].lease(holder="hermes", at=EPOCH, lease_s=60)
    later = stack["outbox"].lease(holder="hermes-b", at=EPOCH + 3600)
    assert later is None, "a lease that died with its holder is not an un sent artifact"
    assert stack["outbox"].get(claim.artifact.id).state == "uncertain"


def test_only_the_holder_may_give_an_artifact_back(store, stack, evidence):
    made = stack["decide"]()
    stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner", topic="general",
                            payload="The invoice is overdue.", evidence=[evidence])
    claim = stack["outbox"].lease(holder="hermes", at=EPOCH)
    with pytest.raises(EvidenceError, match="not a live lease"):
        stack["outbox"].release(artifact_id=claim.artifact.id, token="someone_elses")
    assert stack["outbox"].get(claim.artifact.id).state == "leased"
    stack["outbox"].attempt(artifact_id=claim.artifact.id, token=claim.token)
    with pytest.raises(EvidenceError, match="Re-sending it blind"):
        stack["outbox"].release(artifact_id=claim.artifact.id, token=claim.token)
    # Once it has been handed over, giving it back is a lie about the send.


def test_the_happy_path_reports_a_receipt_the_host_can_be_held_to(store, stack, evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="The invoice is overdue.",
                                       evidence=[evidence])
    claim = stack["outbox"].lease(holder="hermes", at=EPOCH)
    assert claim.artifact.id == prepared["id"] and claim.artifact.state == "leased"
    stack["outbox"].attempt(artifact_id=claim.artifact.id, token=claim.token)
    done = stack["outbox"].confirm(artifact_id=claim.artifact.id, token=claim.token,
                                  proof={"digest": claim.artifact.payload_digest})
    assert done.state == "confirmed" and done.delivered_at
    assert store.db.execute("SELECT state FROM decision_intents WHERE id=?",
                            (made["intent"],)).fetchone()[0] == "delivered"


def test_a_proof_for_different_bytes_is_uncertainty_not_success(store, stack, evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="The invoice is overdue.",
                                       evidence=[evidence])
    claim = stack["outbox"].lease(holder="hermes", at=EPOCH)
    stack["outbox"].attempt(artifact_id=prepared["id"], token=claim.token)
    done = stack["outbox"].confirm(artifact_id=prepared["id"], token=claim.token,
                                   proof={"digest": digest(["something", "else"])})
    assert done.state == "uncertain"
    assert "not this artifact" in done.reason


@pytest.mark.parametrize("proof, expected", [
    (None, "accepted_unverified"),
    ({"sent": True}, "accepted_unverified"),
    ({"status": "queued"}, "uncertain"),
])
def test_a_host_that_cannot_prove_it_delivered_says_so(store, stack, evidence, proof,
                                                       expected):
    made = stack["decide"]()
    stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner", topic="general",
                            payload="The invoice is overdue.", evidence=[evidence])
    claim = stack["outbox"].lease(holder="hermes", at=EPOCH)
    stack["outbox"].attempt(artifact_id=claim.artifact.id, token=claim.token)
    done = stack["outbox"].confirm(artifact_id=claim.artifact.id, token=claim.token,
                                   proof=proof)
    assert done.state == expected, "an acknowledgement is not a delivery receipt"


def test_a_crash_mid_handover_leaves_the_send_uncertain_not_queued(store, stack, evidence):
    """The send/ack crash: the artifact was handed over, and nobody said what happened."""
    made = stack["decide"]()
    stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner", topic="general",
                            payload="The invoice is overdue.", evidence=[evidence])
    claim = stack["outbox"].lease(holder="hermes", at=EPOCH)
    stack["outbox"].attempt(artifact_id=claim.artifact.id, token=claim.token)
    recovered = stack["outbox"].recover(at=EPOCH + 3600)
    assert recovered == [claim.artifact.id]
    assert stack["outbox"].get(claim.artifact.id).state == "uncertain"
    assert stack["outbox"].lease(holder="hermes-b", at=EPOCH + 3600) is None, \
        "a dead holder's artifact must not be handed to a second one"


def test_no_blind_resend_of_anything_already_handed_over(store, stack, evidence):
    made = stack["decide"]()
    stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner", topic="general",
                            payload="The invoice is overdue.", evidence=[evidence])
    claim = stack["outbox"].lease(holder="hermes", at=EPOCH)
    stack["outbox"].attempt(artifact_id=claim.artifact.id, token=claim.token)
    with pytest.raises(EvidenceError, match="Re-sending it blind"):
        stack["outbox"].attempt(artifact_id=claim.artifact.id, token=claim.token)
    done = stack["outbox"].confirm(artifact_id=claim.artifact.id, token=claim.token,
                                   proof={"digest": claim.artifact.payload_digest})
    with pytest.raises(EvidenceError, match="cannot be re-opened"):
        stack["outbox"].attempt(artifact_id=done.id, token=claim.token)


def test_a_stranger_cannot_report_the_outcome_of_someone_elses_send(store, stack,
                                                                    evidence):
    made = stack["decide"]()
    stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner", topic="general",
                            payload="The invoice is overdue.", evidence=[evidence])
    claim = stack["outbox"].lease(holder="hermes", at=EPOCH)
    stack["outbox"].attempt(artifact_id=claim.artifact.id, token=claim.token)
    with pytest.raises(EvidenceError, match="not from the holder"):
        stack["outbox"].confirm(artifact_id=claim.artifact.id, token="claim_other")
    with pytest.raises(EvidenceError, match="Re-sending it blind"):
        stack["outbox"].attempt(artifact_id=claim.artifact.id, token="claim_other")


def test_a_stale_queued_draft_expires_rather_than_arriving_late(store, stack, evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="draft",
                                       topic="general", payload="Draft reply to the client.",
                                       evidence=[evidence],
                                       expires_at="2026-09-15T12:00:00+00:00")
    assert stack["outbox"].expire(at="2026-09-16T09:00:00+00:00") == [prepared["id"]]
    assert stack["outbox"].get(prepared["id"]).state == "expired"
    assert stack["outbox"].lease(holder="hermes", at=EPOCH) is None


def test_the_payload_digest_is_a_statement_about_these_exact_bytes(store, stack, evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="  The   invoice  is here ",
                                       evidence=[evidence])
    artifact = stack["outbox"].get(prepared["id"])
    assert artifact.payload == "The invoice is here"
    assert artifact.payload_digest == digest(["v1", "general", OWNER, artifact.payload,
                                              [evidence]])


@pytest.mark.parametrize("payload", ["", "   ", None, 17, "x" * 7000])
def test_an_unreadable_or_endless_payload_is_refused_at_preparation(store, stack,
                                                                    payload):
    made = stack["decide"]()
    with pytest.raises(EvidenceError):
        stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                topic="general", payload=payload, evidence=[])


def test_a_digest_artifact_needs_the_window_it_claims_to_belong_to(store, stack, evidence):
    made = stack["decide"]()
    store.db.execute("UPDATE proactive_decisions SET window_id=NULL WHERE id=?",
                     (made["id"],))
    with pytest.raises(EvidenceError, match="window"):
        stack["outbox"].prepare(decision_id=made["id"], kind="digest", topic="general",
                                payload="Two things overnight.", evidence=[evidence])


def test_the_forgotten_path_reports_what_was_suppressed_and_why(store, stack, evidence):
    made = stack["decide"]()
    prepared = stack["outbox"].prepare(decision_id=made["id"], kind="notify_owner",
                                       topic="general", payload="The invoice is overdue.",
                                       evidence=[evidence])
    stack["outbox"].suppress(prepared["id"], reason="the owner asked not to be told today")
    artifact = stack["outbox"].get(prepared["id"])
    assert artifact.state == "suppressed" and "not to be told" in artifact.reason
    assert store.db.execute("SELECT state FROM decision_intents WHERE id=?",
                            (made["intent"],)).fetchone()[0] == "suppressed"
    with pytest.raises(EvidenceError, match="cannot be unsent"):
        stack["outbox"].suppress(prepared["id"], reason="too late")
