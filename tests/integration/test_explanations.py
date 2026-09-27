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
    assert report["held"] == [], "a record with no attachment says so rather than omitting it"


def test_an_attached_file_is_named_without_its_bytes_reaching_the_reading(store, explained):
    """The owner asks what the store holds about a message; the answer is a name and a hash.

    Bytes are the thing a context assembly must never pull in on a listing's behalf, so the
    explanation reports the metadata the blob ledger recorded and leaves the content where
    it is.
    """
    from hermes_memory.storage.blobs import BlobStore

    record = a_record(store)
    blobs = BlobStore(store)
    payload = b"%PDF-1.4 the invoice the owner attached\n"
    store.db.execute("BEGIN IMMEDIATE")
    blobs.attach(record, [{"data": payload, "filename": "invoice.pdf",
                           "mime": "application/pdf"}], db=store.db)
    store.db.execute("COMMIT")
    report = explained.retrieval(record)
    assert [item["filename"] for item in report["held"]] == ["invoice.pdf"]
    assert report["held"][0]["size"] == len(payload)
    assert payload.decode() not in json.dumps(report, default=str), "no bytes in the reading"


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


# -- what the archive learned, and what it made of the evidence ---------------

def a_lesson(store, *, status="candidate"):
    """One habit, the record it came from, and the store that holds both."""
    from learning_cases import RULE
    from hermes_memory.learning.lessons import LessonStore
    from hermes_memory.learning.outcomes import OutcomeLog

    record = a_record(store, source_id="playbook-1",
                      text="Chase the invoice twice before phoning.")
    habits = LessonStore(store, outcomes=OutcomeLog(store, owner_principal=OWNER),
                         owner_principal=OWNER)
    made = habits.propose(lesson_id="chase-invoice",
                         text="Chase an overdue invoice twice by mail before phoning.",
                         applicability=RULE, evidence=[record], proposed_by="agent-x",
                         proposed_kind="agent")
    habits.record_confirmation(lesson_id="chase-invoice", version=made["version"],
                              note="it was right about the September one", actor=OWNER)
    if status == "active":
        habits.activate(lesson_id="chase-invoice", version=made["version"], actor=OWNER,
                       reason="the owner read it")
    return habits, made, record


def test_a_lesson_is_explained_by_its_versions_its_citations_and_its_tally(store, explained):
    _habits, made, record = a_lesson(store)
    report = explained.lesson("chase-invoice")
    assert report["lesson_id"] == "chase-invoice" and report["current"] == made["version"]
    only = report["versions"][0]
    assert only["status"] == "candidate" and only["proposed_kind"] == "agent"
    assert only["support"] == 1 and only["against"] == 0
    assert only["evidence"] == [{"citation": record, "record": record, "readable": True}]
    assert only["awaiting_owner_review"] is False and only["evaluation"] is None
    assert "text" not in only, "the wording of a rule is private until it is named"
    assert "teaches nothing" in report["note"]


def test_the_private_wording_is_still_redacted_when_it_is_named(store, explained):
    _habits, _made, _record = a_lesson(store)
    only = explained.lesson("chase-invoice", include_private=True)["versions"][0]
    assert "overdue invoice" in only["text"]


def test_a_lesson_the_archive_no_longer_stands_behind_says_so(store, explained):
    habits, made, record = a_lesson(store, status="active")
    store.hide(record, reason="a correction arrived", actor=OWNER)
    only = explained.lesson("chase-invoice")["versions"][0]
    assert only["evidence"][0]["readable"] is False
    assert only["awaiting_owner_review"] is True
    assert only["review_reason"] == "evidence gone"


def test_a_run_appears_against_the_lesson_it_licensed(store, explained):
    from hermes_memory.learning.evaluation import EvaluationLedger

    habits, made, _record = a_lesson(store)

    class Scored:
        name = "fixture-suite"

        def evaluate(self, *, lesson, case):
            return {"outcome": "pass", "detail": "both cases held"}

    ledger = EvaluationLedger(store, runner=Scored(), code_version="hm-test",
                             model_version="remote-9b", lessons=habits,
                             owner_principal=OWNER)
    habits.evaluations = ledger
    report = ledger.run(
        lesson_id="chase-invoice", version=made["version"],
        cases=[{"case_id": "overdue", "role": "targeted"},
               {"case_id": "meeting", "role": "regression"}])
    only = explained.lesson("chase-invoice")["versions"][0]
    assert only["evaluation"]["verdict"] == "passed"
    assert only["evaluation"]["runner"] == "fixture-suite"
    assert only["status"] == "active" and report["promoted"] is True


def test_an_unknown_lesson_is_named_as_absent(explained):
    with pytest.raises(EvidenceError, match="no lesson"):
        explained.lesson("never-proposed")


def a_reading(store, **kwargs):
    from hermes_memory.knowledge.summaries import SummaryStore

    record = a_record(store, source_id="survey-1",
                      text="Two notes about the survey were filed.")
    habits = SummaryStore(store, owner_principal=OWNER)
    made = habits.publish(scope="project:survey", kind="project", title="the survey",
                         body="Two notes were filed about the survey.",
                         citations=[{"record_id": record}],
                         processor_fingerprint="proc-1", **kwargs)
    return habits, made["id"], record


def test_a_reading_is_explained_by_the_records_it_cited(store, explained):
    _habits, summary, record = a_reading(store)
    report = explained.summary(summary)
    assert report["found"] is True and report["available"] is True
    assert report["cited"] == [{"record": record, "state": "visible", "coverage": "full"}]
    assert report["refresh_owed"] == []
    assert "body" not in report


def test_the_body_of_a_reading_travels_only_when_it_is_named(store, explained):
    _habits, summary, _record = a_reading(store)
    assert "Two notes were filed" in explained.summary(summary, include_private=True)["body"]


def test_a_reading_whose_evidence_went_explains_why_it_is_not_shown(store, explained):
    habits, summary, record = a_reading(store)
    from hermes_memory.lifecycle.erasure import ErasureManager

    manager = ErasureManager(store, owner_principal=OWNER)
    preview = manager.preview(record_ids=[record], actor=OWNER, reason="not ours")
    manager.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                    actor=OWNER)
    report = explained.summary(summary)
    assert report["available"] is False and report["reason"] == "withdrawn"
    assert [item["state"] for item in report["cited"]] == ["erased"]
    assert [item["scope"] for item in report["refresh_owed"]] == ["project:survey"]


def test_an_unknown_reading_is_named_as_absent(explained):
    with pytest.raises(EvidenceError, match="no summary"):
        explained.summary("sum_" + "0" * 32)


def test_a_topic_with_no_policy_names_the_ones_that_have_one(store, explained, stack):
    stack["decide"](topic="invoices")
    report = explained.suppressed(topic="no-such-topic")
    assert report["decided_against"] == []
    assert "general" in report["topics_under_policy"], (
    "an empty answer has to say whether the topic exists at all")


def test_two_artifacts_from_one_decision_are_named_together(store, explained, stack):
    made = stack["decide"](topic="invoices")
    first = made["artifact"]
    second = stack["outbox"].prepare(
        decision_id=made["id"], kind="draft", topic="invoices",
        payload="A second thing the same decision produced")
    report = explained.notification(first["id"])
    assert [item["id"] for item in report["from_the_same_decision"]] == [second["id"]]
    assert report["from_the_same_decision"][0]["kind"] == "draft"
    assert report["from_the_same_decision"][0]["state"] == "prepared"


def test_the_door_routes_a_lesson_and_a_reading_to_those_readings(tmp_path, monkeypatch):
    """The two newest explain targets have to be reachable, not merely implemented."""
    import contextlib
    import io
    import json

    from hermes_memory.cli import main
    from hermes_memory.storage.evidence import EvidenceStore

    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MEMORY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_MEMORY_OWNER_PRINCIPAL", OWNER)
    with EvidenceStore(tmp_path / "canonical.db") as store:
        _habits, made, _record = a_lesson(store)
        _summaries, summary, _other = a_reading(store)

    def read(*args):
        buffer = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(errors):
            code = main(list(args))
        text = buffer.getvalue()
        payload = json.loads(text) if text.strip().startswith("{") else {}
        return code, payload, errors.getvalue()

    code, report, _ = read("explain", "--lesson", "chase-invoice")
    assert code == 0 and report["versions"][0]["lesson"] == f"chase-invoice@{made['version']}"
    assert "text" not in report["versions"][0]

    code, report, _ = read("explain", "--summary", summary)
    assert code == 0 and report["id"] == summary and report["available"] is True
    assert "body" not in report

    code, report, refused = read("explain", "--summary", "sum_" + "0" * 32)
    assert code == 2 and report == {} and "no summary" in refused
