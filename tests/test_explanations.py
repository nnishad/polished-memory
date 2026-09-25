"""C14 explanations: answer "why" from the rows that were written when it was true.

The chain is built with the real C8/C10 components rather than by inserting rows,
because the point of these readings is that each rung recorded its own reason — an
explanation assembled from hand-placed rows would not prove that.
"""
from __future__ import annotations

import json

import pytest

from hermes_memory.ids import timestamp
from hermes_memory.proactive.outbox import Outbox
from hermes_memory.proactive.policy import POLICY_VERSION, AttentionPolicy
from hermes_memory.operations.explanations import Explanations
from hermes_memory.prospective.due_events import DueEventLog
from hermes_memory.prospective.goals import GoalStore
from hermes_memory.storage.evidence import EvidenceError

from conftest import envelope

OWNER = "owner-principal"
UTC = "UTC"
MORNING = "2026-09-15T09:00:00+00:00"
EVENING = "2026-09-15T23:30:00+00:00"
EPOCH = 1789462800.0
SECRET = "sk-supersecretvalue1"


@pytest.fixture()
def policy(store):
    subject = AttentionPolicy(store, owner_principal=OWNER)
    subject.configure(actor=OWNER, timezone_name=UTC, quiet_from="22:00",
                      quiet_until="06:00", max_immediate_per_day=5, cooldown_minutes=0,
                      shadow=False)
    return subject


@pytest.fixture()
def stack(store, policy):
    events = DueEventLog(store)
    goals = GoalStore(store, events=events, owner_principal=OWNER)
    outbox = Outbox(store, policy=policy, owner_principal=OWNER, clock=lambda: EPOCH)

    def decide(topic="general", at=MORNING, payload=None):
        decision = policy.decide(topic=topic, at=at)
        goal_id = goals.propose(title="Send the invoice", statement="Client waiting.",
                                timezone_name=UTC, due=timestamp(at),
                                proposed_by=OWNER, proposed_kind="owner")["id"]
        event_id = store.db.execute("SELECT id FROM due_events WHERE goal_id=?",
                                    (goal_id,)).fetchone()[0]
        claim = events.claim(event_id, holder="worker", at=EPOCH)
        intent = events.ack(event_id=event_id, token=claim.token,
                            decision="awaiting_analysis",
                            policy_version=POLICY_VERSION)["intent"]
        written = policy.record(decision, intent_id=intent, goal_id=goal_id, revision=1)
        artifact = None
        if decision.action != "silent":
            artifact = outbox.prepare(decision_id=written["id"], kind=decision.action,
                                      topic=topic,
                                      payload=payload or "Remember: send the invoice")
        return {"id": written["id"], "goal": goal_id, "intent": intent,
                "decision": decision, "artifact": artifact, "topic": topic}

    return {"policy": policy, "goals": goals, "events": events, "outbox": outbox,
            "decide": decide}


def as_owner():
    from types import SimpleNamespace

    return SimpleNamespace(owner_principal=OWNER)


@pytest.fixture()
def explained(store, stack):
    return Explanations(store, settings=as_owner(), policy=stack["policy"],
                        outbox=stack["outbox"])


def a_record(store, text="The kettle boiled.", **overrides):
    payload = envelope(text=text, **overrides)
    return store.commit(payload)["id"]


# -- why this came back ------------------------------------------------------

def test_a_live_indexed_record_is_retrievable_and_says_why(store, explained):
    record = a_record(store)
    report = explained.retrieval(record)
    assert report["retrievable"] is True
    assert all(item["met"] for item in report["gates"])
    assert report["coordinates"]["source"] == "gmail"


def test_withdrawn_evidence_names_the_gate_that_refused_it(store, explained):
    record = a_record(store)
    store.hide(record, reason="only mine to see", actor=OWNER)
    report = explained.retrieval(record)
    refused = {item["gate"]: item["evidence"] for item in report["gates"]
               if not item["met"]}
    assert refused["visible to retrieval"] == "only mine to see"
    # Withdrawing evidence takes it out of the index as well; both are reported so a
    # reader never has to guess which of the two the store meant.
    assert refused["in the search index"] == "no full-text entry exists for this record"


def test_forgotten_evidence_is_not_reported_as_merely_hidden(store, explained):
    record = a_record(store)
    store.db.execute("UPDATE records SET deleted=1 WHERE id=?", (record,))
    report = explained.retrieval(record)
    assert report["retrievable"] is False
    assert [item["gate"] for item in report["gates"] if not item["met"]] == ["not forgotten"]


def test_evidence_that_was_never_indexed_is_its_own_answer(store, explained):
    record = a_record(store)
    store.db.execute("DELETE FROM record_fts WHERE id=?", (record,))
    report = explained.retrieval(record)
    assert report["retrievable"] is False
    assert [item for item in report["gates"] if not item["met"]][0]["gate"] == \
        "in the search index"


def test_a_record_explains_only_what_it_supports(store, explained):
    record = a_record(store)
    other = a_record(store, source_id="msg-2", text="A different message entirely.")
    store.db.execute("INSERT INTO assertions(id, subject, predicate, value, category, "
                     "evidence_kind, record_id, quote_start, quote_end, quote, status, "
                     "created_by, created_at) VALUES('asr-other','owner','mood','calm',"
                     "'profile','explicit_statement',?,0,4,'A di','confirmed',?,?)",
                     (other, OWNER, MORNING))
    store.db.execute("INSERT INTO assertions(id, subject, predicate, value, category, "
                     "evidence_kind, record_id, quote_start, quote_end, quote, status, "
                     "created_by, created_at) VALUES('asr-1','owner','mood','calm',"
                     "'profile','explicit_statement',?,0,4,'The k','confirmed',?,?)",
                     (record, OWNER, MORNING))
    store.db.execute("INSERT INTO derived_citations(artifact_id, kind, coverage, record_id,"
                     " added_at) VALUES('sum-1','summary','full',?,?)", (record, MORNING))
    report = explained.retrieval(record)
    assert [item["id"] for item in report["supports"]["assertions"]] == ["asr-1"]
    assert report["supports"]["artifacts"][0]["artifact_id"] == "sum-1"
    cited = explained.retrieval(other)
    assert cited["supports"]["assertions"] == [{"id": "asr-other", "predicate": "mood",
                                                "status": "confirmed"}]


def test_an_unknown_record_is_answered_with_an_error_not_an_empty_shape(explained):
    with pytest.raises(EvidenceError, match="no record"):
        explained.retrieval("rec_does_not_exist")
    with pytest.raises(EvidenceError, match="nonempty text"):
        explained.retrieval("   ")


# -- why this was said, or not ----------------------------------------------

def test_a_prepared_artifact_says_who_still_owes_it(store, stack, explained):
    made = stack["decide"]()
    report = explained.notification(made["artifact"]["id"])
    assert report["answer"].startswith("prepared and waiting for the host transport")
    assert report["state"] == "prepared"
    assert report["still_sendable"]["ok"] is True


def test_a_delivered_artifact_is_answered_from_its_receipt(store, stack, explained):
    made = stack["decide"]()
    artifact_id = made["artifact"]["id"]
    claim = stack["outbox"].lease(holder="host", at=EPOCH)
    stack["outbox"].attempt(artifact_id=artifact_id, token=claim.token)
    stack["outbox"].confirm(artifact_id=artifact_id, token=claim.token,
                            proof={"digest": claim.artifact.payload_digest})
    report = explained.notification(artifact_id)
    assert report["answer"] == "delivered, and the host said so"
    assert report["state"] == "confirmed"


def test_the_whole_chain_is_reported_rung_by_rung(store, stack, explained):
    made = stack["decide"]()
    report = explained.notification(made["artifact"]["id"])
    assert report["decided"]["action"] == made["decision"].action
    assert report["decided"]["reason"], "a decision without a reason is not an explanation"
    assert report["handed_off"]["kind"] == "awaiting_analysis"
    assert report["trigger"]["fire_at"] == timestamp(MORNING)
    assert report["goal"]["status"] == "active"
    assert report["decided"]["model_used"] is False


def test_a_payload_never_travels_unless_it_is_asked_for(store, stack, explained):
    made = stack["decide"](payload=f"key is {SECRET}, please send")
    plain = json.dumps(explained.notification(made["artifact"]["id"]), default=str)
    assert "please send" not in plain
    assert SECRET not in plain
    assert explained.notification(made["artifact"]["id"])["payload_digest"]


def test_even_an_explicit_payload_is_redacted(store, stack, explained):
    made = stack["decide"](payload=f"key is {SECRET}, please send")
    report = explained.notification(made["artifact"]["id"], include_private=True)
    assert SECRET not in report["payload"]
    assert "[redacted]" in report["payload"]


def test_a_goal_statement_is_not_quoted_into_an_operators_log(store, stack, explained):
    made = stack["decide"]()
    report = explained.notification(made["artifact"]["id"])
    assert "title" not in json.dumps(report, default=str)
    assert "Client waiting." not in json.dumps(report, default=str)
    assert "Client waiting." not in json.dumps(
        explained.notification(made["artifact"]["id"], include_private=True),
        default=str)


def test_a_completed_goal_makes_its_queued_artifact_unsendable(store, stack, explained):
    made = stack["decide"]()
    stack["goals"].complete(goal_id=made["goal"], actor=OWNER, reason="done")
    report = explained.notification(made["artifact"]["id"])
    assert report["still_sendable"]["ok"] is False
    assert report["still_sendable"]["stage"] == "goal"
    assert "completed promise" in report["answer"]


def test_an_opted_out_topic_shows_up_in_the_explanation(store, stack, explained):
    made = stack["decide"](topic="health")
    stack["policy"].configure(actor=OWNER, topic="health", opted_out=True)
    report = explained.notification(made["artifact"]["id"])
    assert report["policy_now"]["state"] == "opted_out"
    assert report["still_sendable"]["stage"] == "opt_out"


def test_the_packet_is_named_but_never_quoted(store, stack, explained):
    made = stack["decide"]()
    store.db.execute("UPDATE proactive_decisions SET packet_id='pkt-1', citations=? "
                     "WHERE id=?", (json.dumps(["rec_a", "rec_b"]), made["id"]))
    report = explained.notification(made["artifact"]["id"])
    assert report["decided"]["packet_id"] == "pkt-1"
    assert report["decided"]["cited_records"] == ["rec_a", "rec_b"]


def test_a_withheld_artifact_says_why_it_was_withheld(store, stack, explained):
    made = stack["decide"]()
    artifact_id = made["artifact"]["id"]
    stack["outbox"].suppress(artifact_id, reason="the owner asked for a quiet week")
    report = explained.notification(artifact_id)
    assert report["state"] == "suppressed"
    assert report["answer"] == ("withheld before sending: the owner asked for a "
                                "quiet week")


def test_a_suppression_reason_is_not_echoed_verbatim(store, explained):
    store.db.execute("INSERT INTO goals(id, title, statement, status, timezone, "
                     "created_by, created_kind, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                     ("goal-1", "Call", "Call", "active", UTC, OWNER, "owner",
                      MORNING, MORNING))
    store.db.execute("INSERT INTO due_events(id, goal_id, revision, fire_at, reason, "
                     "timezone, precision, state, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                     ("due-1", "goal-1", 1, MORNING, "due", UTC, "minute", "pending",
                      MORNING))
    store.db.execute("INSERT INTO decision_intents(id, event_id, goal_id, revision, kind, "
                     "policy_version, state, created_at, updated_at) "
                     "VALUES(?,?,?,?,?,?,?,?,?)",
                     ("intent-1", "due-1", "goal-1", 1, "silent", POLICY_VERSION,
                      "prepared", MORNING, MORNING))
    store.db.execute("INSERT INTO proactive_decisions(id, intent_id, goal_id, revision, "
                     "topic, action, reason, policy_version, shadow, model_used, "
                     "citations, decided_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                     ("dec-1", "intent-1", "goal-1", 1, "general", "silent",
                      f"because {SECRET} leaked into the reason", POLICY_VERSION, 0, 0,
                      "[]", MORNING))
    dumped = json.dumps(explained.suppressed(topic="general"), default=str)
    assert SECRET not in dumped
    assert "[redacted]" in dumped


def test_an_uncertain_send_reports_who_lost_it(store, stack, explained):
    made = stack["decide"]()
    artifact_id = made["artifact"]["id"]
    claim = stack["outbox"].lease(holder="host", at=EPOCH)
    stack["outbox"].attempt(artifact_id=artifact_id, token=claim.token)
    stack["outbox"].uncertain(artifact_id=artifact_id, token=claim.token,
                              reason="the host stopped answering")
    report = explained.notification(artifact_id)
    assert report["answer"] == ("a send was attempted and its outcome is unknown: "
                                "the host stopped answering")


def test_an_unknown_artifact_is_an_error(store, explained):
    with pytest.raises(EvidenceError, match="no artifact"):
        explained.notification("out_nope")


# -- why the owner was not interrupted --------------------------------------

def test_the_suppressed_list_is_about_one_topic_and_the_newest_thing_first(store, stack,
                                                                          explained):
    first = stack["decide"](topic="general", at=MORNING)
    store.db.execute("UPDATE proactive_decisions SET action='silent', reason='not now' "
                     "WHERE id=?", (first["id"],))
    store.db.execute("DELETE FROM outbox WHERE decision_id=?", (first["id"],))
    second = stack["decide"](topic="health", at="2026-09-16T09:00:00+00:00")
    store.db.execute("UPDATE proactive_decisions SET action='silent', reason='health quiet'"
                     " WHERE id=?", (second["id"],))
    store.db.execute("DELETE FROM outbox WHERE decision_id=?", (second["id"],))
    general = explained.suppressed(topic="general")
    assert [item["reason"] for item in general["decided_against"]] == ["not now"]
    health = explained.suppressed(topic="health")
    assert [item["reason"] for item in health["decided_against"]] == ["health quiet"]


def test_the_suppressed_list_is_newest_first(store, stack, explained):
    for day, reason in (("15", "yesterday's reason"), ("16", "today's reason")):
        made = stack["decide"](at=f"2026-09-{day}T09:00:00+00:00")
        store.db.execute("UPDATE proactive_decisions SET action='silent', reason=? "
                         "WHERE id=?", (reason, made["id"]))
        store.db.execute("DELETE FROM outbox WHERE decision_id=?", (made["id"],))
    report = explained.suppressed(topic="general")
    assert [item["reason"] for item in report["decided_against"]] == ["today's reason",
                                                                     "yesterday's reason"]


def test_the_suppressed_list_answers_the_question_that_has_no_artifact(store, stack,
                                                                       explained):
    made = stack["decide"](topic="general")
    store.db.execute("UPDATE proactive_decisions SET action='silent', reason="
                     "'nothing in the change needed the owner' WHERE id=?", (made["id"],))
    store.db.execute("DELETE FROM outbox WHERE decision_id=?", (made["id"],))
    report = explained.suppressed(topic="general")
    assert report["decided_against"][0]["reason"] == "nothing in the change needed the owner"
    assert report["decided_against"][0]["artifact_prepared"] is False


def test_quiet_hours_are_answered_with_when_the_owner_would_be_woken(explained):
    """A notification at 23:30 has not informed anyone; it has woken them. The answer
    to "why was I not told" has to say when the telling was scheduled for."""
    report = explained.suppressed(topic="general", at=EVENING)
    assert report["quiet_until"] == "2026-09-16T06:00:00+00:00"
    assert report["policy"]["quiet_from"] == "22:00"
    assert explained.suppressed(topic="general", at=MORNING)["quiet_until"] is None


def test_a_silenced_topic_reports_its_owners_decision_as_the_reason(store, stack,
                                                                    explained):
    stack["policy"].configure(actor=OWNER, topic="general", opted_out=True)
    report = explained.suppressed(topic="general")
    assert report["policy"]["state"] == "opted_out"


def test_the_suppressed_list_has_a_bound(store, explained):
    with pytest.raises(EvidenceError, match="between 1 and 200"):
        explained.suppressed(topic="general", limit=0)
    with pytest.raises(EvidenceError, match="between 1 and 200"):
        explained.suppressed(topic="general", limit=5000)


# -- prospective memory ------------------------------------------------------

def test_an_active_goal_explains_what_is_still_outstanding(store, stack, explained):
    goal_id = stack["decide"]()["goal"]
    report = explained.goal(goal_id)
    assert report["status"] == "active"
    assert "due event" in report["why_now"] or "decided about" in report["why_now"]
    assert report["due_events"], "a goal with a clock must show it"


def test_a_settled_goal_says_so_rather_than_looking_overdue(store, stack, explained):
    goal_id = stack["decide"]()["goal"]
    stack["goals"].complete(goal_id=goal_id, actor=OWNER, reason="done")
    report = explained.goal(goal_id)
    assert report["why_now"] == "it is completed, so nothing about it is due"


def test_a_snoozed_goal_reports_who_put_it_off_and_until_when(store, stack, explained):
    goal_id = stack["decide"]()["goal"]
    stack["goals"].snooze(goal_id=goal_id, until="2026-09-20T09:00:00+00:00", actor=OWNER,
                          reason="later")
    report = explained.goal(goal_id)
    assert "put off until 2026-09-20T09:00:00+00:00" in report["why_now"]


def test_a_goal_statement_only_appears_when_it_is_asked_for(store, stack, explained):
    goal_id = stack["decide"]()["goal"]
    assert "statement" not in explained.goal(goal_id)
    assert "Client waiting." not in json.dumps(explained.goal(goal_id), default=str)
    assert explained.goal(goal_id, include_private=True)["statement"] == "Client waiting."


def test_an_unknown_goal_is_an_error(store, explained):
    with pytest.raises(EvidenceError, match="no goal"):
        explained.goal("goal_nope")


# -- the readings themselves -------------------------------------------------

def test_no_explanation_writes_anything(store, stack, explained):
    made = stack["decide"]()
    record = a_record(store)
    tables = [row[0] for row in store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]

    def footprint():
        return {table: store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in tables}

    before = footprint()
    explained.notification(made["artifact"]["id"], include_private=True)
    explained.retrieval(record)
    explained.suppressed(topic="general")
    explained.goal(made["goal"], include_private=True)
    assert footprint() == before
    assert store.db.in_transaction is False


def test_every_report_is_plain_data(store, stack, explained):
    made = stack["decide"]()
    json.dumps(explained.notification(made["artifact"]["id"]), default=str)
    json.dumps(explained.retrieval(a_record(store)), default=str)
    json.dumps(explained.suppressed(topic="general"), default=str)
    json.dumps(explained.goal(made["goal"]), default=str)


def test_the_audit_trail_is_reachable_from_explanation_without_being_rewritten(store,
                                                                              stack,
                                                                              explained):
    made = stack["decide"]()
    report = explained.notification(made["artifact"]["id"])
    assert [item["action"] for item in report["history"]] == ["outbox_prepare"]
    goal = explained.goal(made["goal"])
    assert "goal_propose" in [item["action"] for item in goal["history"]]
