"""Clarification safety regressions; synthetic data and no external delivery."""
import re
from types import SimpleNamespace

import pytest

from hermes_memory.ids import timestamp
from hermes_memory.proactive.inquiries import InquiryStore, _polarity
from hermes_memory.proactive.delivery import sink_for
from hermes_memory.proactive.policy import AttentionPolicy
from hermes_memory.prospective.due_events import DueEventLog
from hermes_memory.prospective.goals import GoalStore
from hermes_memory.storage.evidence import EvidenceStore

OWNER = "synthetic-owner"
CHANNEL = "telegram:12345"


@pytest.fixture
def question(tmp_path):
    store = EvidenceStore(tmp_path / "review.db")
    AttentionPolicy(store, owner_principal=OWNER).configure(
        actor=OWNER, timezone_name="UTC", cooldown_minutes=0, max_immediate_per_day=10,
        quiet_from="00:00", quiet_until="00:01", shadow=False)
    goals = GoalStore(store, events=DueEventLog(store), owner_principal=OWNER)
    goal = goals.propose(title="Review synthetic ferns", statement="Synthetic initial plan",
                         timezone_name="UTC", due=timestamp("2027-01-01T09:00:00Z"),
                         proposed_by="agent:test", proposed_kind="agent")["id"]
    inquiries = InquiryStore(store, owner_principal=OWNER)
    inquiries.allow_replies(actor=OWNER, on=True, reason="Offline review fixture")
    opened = inquiries.ask(decision="goal-activation", subject_id=goal,
                           question="Should the synthetic fern reminder be activated?")
    yield SimpleNamespace(store=store, goals=goals, goal=goal, inquiries=inquiries,
                          inquiry=opened["id"])
    store.close()


def code(body):
    return re.search(r"`yes ([A-Z2-9]{6})`", body).group(1)


def send(q, sink):
    def acknowledged(body):
        result = sink(body)
        return {"sent": True} if result is None else result
    return q.inquiries.send_next(sink=acknowledged, destination=CHANNEL, holder="review", limit=1)


@pytest.mark.parametrize("words", ["not sure", "not confirmed", "not correct", "not approved"])
def test_negated_positive_words_are_not_authorizations(words):
    assert _polarity(words) is None


def test_negated_approval_does_not_activate_a_goal(question):
    q = question
    bodies = []
    send(q, bodies.append)
    result = q.inquiries.answer(reply="I am not sure " + code(bodies[0]), channel=CHANNEL)
    assert result["settled"] is False
    assert q.store.db.execute("SELECT status FROM goals WHERE id=?", (q.goal,)).fetchone()[0] == "candidate"


def test_a_second_sender_cannot_steal_an_existing_live_lease(question):
    q = question
    bodies = []
    inner = []

    def overlapping_sink(body):
        bodies.append(body)
        inner.append(send(q, bodies.append))
        return {"sent": True, "message_id": "outer"}

    outer = send(q, overlapping_sink)
    assert outer["sent"] == 1 and inner[0]["sent"] == 0
    assert len(bodies) == 1
    assert q.inquiries.answer(reply="yes " + code(bodies[0]), channel=CHANNEL)["settled"]


def test_explicit_negative_send_receipt_is_not_reported_as_sent(question):
    q = question
    result = send(q, lambda body: {"sent": False, "error": "synthetic refused delivery"})
    assert result["sent"] == 0
    assert q.inquiries.get(q.inquiry).state == "open"
    assert send(q, lambda body: {"sent": True})["sent"] == 1


def test_expired_question_is_not_sent_before_next_maintenance_pass(question):
    q = question
    q.store.db.execute("UPDATE inquiries SET expires_at=? WHERE id=?",
                       ("2000-01-01T00:00:00+00:00", q.inquiry))
    bodies = []
    assert send(q, bodies.append)["sent"] == 0
    assert not bodies and q.inquiries.get(q.inquiry).state == "expired"


def test_instructions_require_code_for_withdrawal(question):
    q = question
    bodies = []
    send(q, bodies.append)
    assert "`no " + code(bodies[0]) + "` stops the asking" in bodies[0]
    assert not q.inquiries.answer(reply="no", inquiry_id=q.inquiry, channel=CHANNEL)["settled"]
    assert q.inquiries.get(q.inquiry).state == "sent"


def test_concurrent_writer_cannot_change_subject_between_fence_and_approval(question, monkeypatch):
    import sqlite3
    q = question
    bodies = []
    send(q, bodies.append)
    check = q.inquiries._moved
    other = EvidenceStore(q.store.path)
    other.db.execute("PRAGMA busy_timeout=1")

    def check_then_concurrent_revision(inquiry):
        result = check(inquiry)
        # The same deterministic interleaving could be another process writing
        # between the separate fence read and the owner act's transaction.
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            GoalStore(other, events=DueEventLog(other), owner_principal=OWNER).revise(
                goal_id=q.goal, actor=OWNER, reason="Synthetic concurrent edit",
                statement="Different synthetic plan never shown in the question")
        return result

    monkeypatch.setattr(q.inquiries, "_moved", check_then_concurrent_revision)
    result = q.inquiries.answer(reply="yes " + code(bodies[0]), channel=CHANNEL)
    row = q.store.db.execute("SELECT status,revision FROM goals WHERE id=?", (q.goal,)).fetchone()
    assert result["settled"] and row["status"] == "active" and row["revision"] == 1
    other.close()


def test_reopened_question_has_current_epoch_and_invalidates_old_code(question):
    q = question
    bodies = []
    send(q, bodies.append)
    q.store.bump_epoch(reason="Synthetic trust revocation", actor=OWNER)
    assert not q.inquiries.answer(reply="yes " + code(bodies[0]), channel=CHANNEL)["settled"]
    reopened = q.inquiries.ask(decision="goal-activation", subject_id=q.goal,
                               question="Should the synthetic fern reminder be activated?")
    assert reopened["asked"] and q.inquiries.get(q.inquiry).epoch == q.store.epoch()
    assert send(q, bodies.append)["sent"] == 1 and len(bodies) == 2
    assert not q.inquiries.answer(reply="yes " + code(bodies[0]), channel=CHANNEL)["settled"]
    assert q.inquiries.answer(reply="yes " + code(bodies[-1]), channel=CHANNEL)["settled"]


def test_command_sink_uses_explicit_profile_binding(tmp_path, monkeypatch):
    import sys
    captured = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))

    def fake_run(*args, **kwargs):
        captured.append(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout=b'{"success":true,"chat_id":"12345"}',
                               stderr=b"")

    monkeypatch.setattr("hermes_memory.proactive.delivery.subprocess.run", fake_run)
    settings = SimpleNamespace(delivery_target=CHANNEL, delivery_command=(sys.executable,))
    sink, _ = sink_for(settings, tmp_path / "profile-b")
    sink("Synthetic question")
    assert captured[0]["HERMES_HOME"] == str(tmp_path / "profile-b")


def test_crash_after_owner_mutation_rolls_back_inquiry_and_subject(question, monkeypatch):
    from hermes_memory.operations import decisions
    q = question
    bodies = []
    send(q, bodies.append)
    original = decisions.settle

    def crash_after_change(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("Synthetic crash before receipt")

    monkeypatch.setattr(decisions, "settle", crash_after_change)
    with pytest.raises(RuntimeError, match="Synthetic crash"):
        q.inquiries.answer(reply="yes " + code(bodies[0]), channel=CHANNEL)
    assert q.inquiries.get(q.inquiry).state == "sent"
    assert q.goals.get(q.goal).status == "candidate"
    monkeypatch.setattr(decisions, "settle", original)
    assert q.inquiries.answer(reply="yes " + code(bodies[0]), channel=CHANNEL)["settled"]


def test_unknown_delivery_is_not_retried_after_lease_timeout(question):
    q = question
    bodies = []

    def lost_receipt(body):
        bodies.append(body)
        raise TimeoutError("Synthetic receipt loss after transport handoff")

    assert send(q, lost_receipt)["sent"] == 0
    q.store.db.execute("UPDATE inquiries SET lease_until=0")
    assert send(q, bodies.append)["sent"] == 0 and len(bodies) == 1
    assert q.store.db.execute("SELECT delivery_state FROM inquiries").fetchone()[0] == "uncertain"


def native_send(q, *, message="100", home="/synthetic/profile", thread=""):
    def transport(body):
        return {"sent": True, "platform": "telegram", "chat_id": "12345",
                "message_id": message, "thread_id": thread, "mirrored": False}
    transport.hermes_home = home
    return q.inquiries.send_next(sink=transport, destination=CHANNEL, holder="review", limit=1)


def native_answer(q, **overrides):
    fields = dict(reply="yes", platform="telegram", chat_id="12345", message_id="100",
                  thread_id="", profile_home="/synthetic/profile",
                  transport_home="/synthetic/profile", author_id="12345")
    fields.update(overrides)
    return q.inquiries.answer_native(**fields)


def test_native_reply_correlates_without_code_after_database_reopen(question):
    q = question
    assert native_send(q)["sent"] == 1
    with EvidenceStore(q.store.path) as restarted:
        q.inquiries = InquiryStore(restarted, owner_principal=OWNER)
        assert native_answer(q)["settled"]
        assert native_answer(q)["settled"] is False
        dossier = q.inquiries.dossier(q.inquiries.get(q.inquiry))
        assert dossier["question"] and dossier["outcome"]


@pytest.mark.parametrize("wrong", [dict(author_id="guest"), dict(chat_id="67890"),
    dict(profile_home="/synthetic/other"), dict(transport_home="/synthetic/other"),
    dict(message_id="999"), dict(thread_id="other"), dict(reply="I am not sure")])
def test_native_reply_refuses_wrong_provenance_and_ambiguous_words(question, wrong):
    assert native_send(question)["sent"] == 1
    assert not native_answer(question, **wrong)["settled"]
    assert question.goals.get(question.goal).status == "candidate"


def test_old_native_message_cannot_answer_reopened_question(question):
    q = question
    native_send(q)
    q.inquiries._void(q.inquiry, reason="Synthetic new question generation")
    q.inquiries.ask(decision="goal-activation", subject_id=q.goal,
                    question="Should this fresh synthetic question be activated?")
    assert native_send(q, message="101")["sent"] == 1
    assert not native_answer(q)["settled"]
    assert native_answer(q, message_id="101")["settled"]


def test_profile_command_binding_is_real_across_a_b_a(tmp_path):
    import sys
    for profile in ("a", "b", "a"):
        home = tmp_path / profile
        command = (sys.executable, "-c", "import os,json; print(json.dumps({'success':True, 'chat_id':os.environ['HERMES_HOME']}))")
        settings = SimpleNamespace(delivery_target="", delivery_command=command)
        sink, _ = sink_for(settings, home)
        assert sink("Synthetic question")["chat_id"] == str(home)
