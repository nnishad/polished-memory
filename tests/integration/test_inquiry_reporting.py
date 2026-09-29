"""What a report says about questions to the owner.

Three readings of one table, each with a different reader. `status` answers "is anything
being asked, and is anything stuck", `doctor` answers "should I do something tonight", and
`owner --list` answers "what is waiting on me, and have I already been asked about it". A
queue of decisions nobody was told about is the failure all three exist to prevent, because
the archive is otherwise silent in exactly the way a busy one is.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_memory.cli import ReadOnlyStore
from hermes_memory.operations.doctor import WARN, Doctor
from hermes_memory.operations.status import StatusReporter
from hermes_memory.proactive.inquiries import InquiryStore
from hermes_memory.storage.evidence import EvidenceStore

from conftest import envelope

STAMP = "2026-09-25T12:00:00+00:00"
FUTURE = "2099-01-01T00:00:00+00:00"
PAST = "2000-01-01T00:00:00+00:00"
OWNER = "owner"


def settings(**overrides):
    base = {"owner_principal": OWNER, "capture_only": False, "home": Path(tempfile.mkdtemp()),
            "delivery_enabled": False, "delivery_poll_s": 0,
            "maintenance_interval_s": 900}
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture()
def store(tmp_path):
    with EvidenceStore(tmp_path / "canonical.db") as opened:
        yield opened


def a_candidate(store, *, title="Water the ferns"):
    """A goal an agent proposed: a decision only the owner can make, with no clock."""
    from hermes_memory.prospective.due_events import DueEventLog
    from hermes_memory.prospective.goals import GoalStore

    return GoalStore(store, events=DueEventLog(store),
                     owner_principal=OWNER).propose(title=title, statement="It is dry.",
                                                    timezone_name="UTC",
                                                    proposed_by="agent:test",
                                                    proposed_kind="agent")["id"]


def asked(store, subject, *, question="Should I start reminding you about the ferns?",
          allow=True, state=None, expires=FUTURE, next_try=None):
    inquiries = InquiryStore(store, owner_principal=OWNER)
    if allow:
        inquiries.allow_replies(actor=OWNER, on=True, reason="ask me on my own channel")
    made = inquiries.ask(decision="goal-activation", subject_id=subject, question=question)
    if state is not None:
        store.db.execute("UPDATE inquiries SET state=?, expires_at=?, next_try_at=? "
                         "WHERE id=?", (state, expires, next_try, made["id"]))
    return made["id"]


# -- status --------------------------------------------------------------------

def test_an_installation_that_has_never_asked_says_so_and_who_said_so(store):
    report = StatusReporter(store, settings=settings()).questions()

    assert report["replies_enabled"] is False and report["waiting"] == 0
    assert "nobody has switched replies on" in report["why"]
    assert report["stage"] == "unset"


def test_a_question_the_owner_asked_for_is_reported_as_the_owners_decision(store):
    a_candidate(store)
    asked(store, a_candidate(store, title="Chase the invoice"))

    report = StatusReporter(store, settings=settings()).questions()

    assert report["replies_enabled"] is True and "owner" in report["why"]
    assert report["queued"] == 1 and report["awaiting_answer"] == 0
    assert report["live"][0]["decision"] == "goal-activation"
    # The sentence itself stays out of an operator's reading, exactly as a candidate goal's
    # title does: `status` says how many decisions wait, `owner --list` shows their words.
    assert "question" not in report["live"][0]


def test_a_question_held_by_quiet_hours_is_counted_as_held_not_stuck(store):
    identifier = a_candidate(store)
    asked(store, identifier, state="open", next_try=FUTURE)

    report = StatusReporter(store, settings=settings()).questions()

    assert report["held"] == 1 and report["queued"] == 1


def test_the_summary_says_when_everything_awaits_and_nothing_is_asked(store):
    """The commonest real state of a new installation, and the least findable."""
    a_candidate(store)

    notes = StatusReporter(store, settings=settings()).report()["notes"]

    assert [note for note in notes if note.startswith("questions:")] == [
        next(note for note in notes if note.startswith("questions:"))]
    said = next(note for note in notes if note.startswith("questions:"))
    assert "await the owner and nothing is asked" in said and "--switch-replies" in said


def test_a_quiet_archive_with_nothing_to_decide_makes_no_question_of_itself(store):
    notes = StatusReporter(store, settings=settings()).report()["notes"]

    assert [note for note in notes if note.startswith("questions:")] == []


def test_an_expired_question_is_still_reported_after_its_owner_stopped_being_asked(store):
    identifier = a_candidate(store)
    asked(store, identifier, state="expired", expires=PAST)

    report = StatusReporter(store, settings=settings()).questions()

    assert report["expired"] == 1 and report["waiting"] == 0
    assert "expired unanswered" in " ".join(
        StatusReporter(store, settings=settings()).report()["notes"])


def test_a_question_still_being_asked_about_a_moved_state_is_named(store):
    """The one disagreement this ledger is allowed to have, so it has to be able to say it.

    Restated as the real event: a goal adopted through a door that never looked at the asking,
    while the question about it sits `sent`.
    """
    from hermes_memory.prospective.due_events import DueEventLog
    from hermes_memory.prospective.goals import GoalStore

    identifier = a_candidate(store)
    question = asked(store, identifier, state="sent")
    GoalStore(store, events=DueEventLog(store),
              owner_principal=OWNER).activate(goal_id=identifier, actor=OWNER,
                                              reason="decided somewhere else")

    report = StatusReporter(store, settings=settings()).questions()

    assert [item["inquiry"] for item in report["decided_elsewhere"]] == [question]
    assert report["decided_elsewhere"][0]["decision"] == "goal-activation"
    assert "already moved" in " ".join(StatusReporter(store, settings=settings()).report()["notes"])


def test_a_question_the_report_leaves_alone_is_one_still_asked_about_the_state_it_showed(store):
    identifier = a_candidate(store)
    asked(store, identifier, state="sent")

    assert StatusReporter(store, settings=settings()).questions()["decided_elsewhere"] == []


def a_question_finding(store):
    """The questions check on its own: the rest of the report belongs to another file."""
    return Doctor(store, settings=settings()).questions().as_dict()


# -- doctor --------------------------------------------------------------------

def test_the_doctor_is_quiet_about_questions_when_there_are_none(store):
    finding = a_question_finding(store)

    assert finding["severity"] == "ok" and finding["remedy"] is None


def test_the_doctor_warns_when_a_question_outranked_its_own_window(store):
    """`expire_due` is the pass's job; a row past its window means nobody ran it."""
    identifier = a_candidate(store)
    asked(store, identifier, state="sent", expires=PAST)

    finding = a_question_finding(store)

    assert finding["severity"] == WARN
    assert "past the window" in finding["detail"] and "questions" in finding["remedy"]


def test_the_doctor_distinguishes_a_queued_question_from_one_being_held(store):
    """Held is a policy working: the owner is asleep. Queued with no sender is a fault."""
    identifier = a_candidate(store)
    asked(store, identifier, state="open", next_try=FUTURE)

    held = a_question_finding(store)

    assert held["severity"] == "ok", "quiet hours are the owner's own instruction"

    store.db.execute("UPDATE inquiries SET next_try_at=NULL")
    stuck = a_question_finding(store)

    assert stuck["severity"] == WARN and "nothing is scheduled to ask them" in stuck["detail"]
    assert "hermes-memory deliver" in stuck["remedy"]


def test_the_doctor_says_out_loud_when_an_answer_was_not_what_decided_it(store):
    """A `sent` question over a decided subject is evidence, not litter: it says who moved."""
    from hermes_memory.prospective.due_events import DueEventLog
    from hermes_memory.prospective.goals import GoalStore

    identifier = a_candidate(store)
    question = asked(store, identifier, state="sent")
    GoalStore(store, events=DueEventLog(store),
              owner_principal=OWNER).activate(goal_id=identifier, actor=OWNER,
                                              reason="an agent with a terminal decided it")

    finding = a_question_finding(store)

    assert finding["severity"] == WARN and "already moved" in finding["detail"]
    assert question in finding["remedy"] and "the audit names who" in finding["remedy"]
    assert a_question_finding(store) == finding, "a reading that reports this erases it"


# -- owner --list --------------------------------------------------------------

def test_the_owners_listing_shows_the_question_and_the_switch_beside_the_decision(store,
                                                                                  tmp_path):
    """One screen: what awaits, whether it was asked, and what the answer would decide.

    The code is in none of it. A listing the agent can read must not carry the thing that
    proves a reply came from a human, which is why this shows the state of the switch rather
    than anything that could be replayed.
    """
    from hermes_memory.cli import _awaiting

    identifier = a_candidate(store)
    asked(store, identifier)

    entry = _awaiting([settings(profile="default", db_path=tmp_path / "canonical.db")])[0]

    assert entry["replies_enabled"] is True
    assert [item["decision"] for item in entry["questions"]] == ["goal-activation"]
    assert entry["questions"][0]["state"] == "open"
    assert "code" not in str(entry["questions"]).lower()
    assert entry["goal_candidates"][0]["goal"] == identifier
