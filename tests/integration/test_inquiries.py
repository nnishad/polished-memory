"""Questions to the owner, and what an answer is allowed to decide.

The framework has always been able to *wait* for a human: an erasure sits in
`awaiting_confirmation`, an identity candidate sits `pending`, a goal an agent proposed is a
candidate. What it could not do was ask. This covers the part that turns waiting into
asking — and mostly covers the refusals, because a reply that arrives over a transport is
text somebody typed, and the value of the code it carries is that it is the only thing
between "the owner answered" and "somebody answered".
"""
from __future__ import annotations

import json
import re

import pytest

from conftest import envelope
from hermes_memory.ids import timestamp
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.proactive.delivery import DeliveryPolicy, deliver_ready, local_sink
from hermes_memory.proactive.inquiries import FENCES, InquiryStore, code_in, fence
from hermes_memory.proactive.policy import AttentionPolicy
from hermes_memory.processing.maintenance import Maintenance
from hermes_memory.prospective.due_events import DueEventLog
from hermes_memory.prospective.goals import GoalStore
from hermes_memory.storage.evidence import EvidenceError
from hermes_memory.storage.identity import IdentityStore

OWNER = "jugaadu"
UTC = "UTC"
FERNS = "Should I start reminding you about the ferns?"
CHANNEL = "telegram:787655730"


@pytest.fixture()
def outdir(tmp_path):
    """Where a local transport puts what it was asked to deliver: the owner's side of the wire."""
    return tmp_path / "home"


@pytest.fixture()
def waiting(store):
    """A goal an agent proposed, which is a candidate until the owner says otherwise."""
    def make(title="Water the ferns"):
        return GoalStore(store, events=DueEventLog(store),
                         owner_principal=OWNER).propose(
            title=title, statement="It is dry.", timezone_name=UTC,
            due=timestamp("2026-09-20T09:00:00+00:00"), proposed_by="agent:test",
            proposed_kind="agent")["id"]
    return make


@pytest.fixture()
def asking(store, outdir):
    """The question store, an owner who has agreed to be asked, and a working transport."""
    AttentionPolicy(store, owner_principal=OWNER).configure(
        actor=OWNER, timezone_name=UTC, cooldown_minutes=0, max_immediate_per_day=5,
        quiet_from="00:00", quiet_until="00:01", shadow=False)

    def make(*, allow=True):
        inquiries = InquiryStore(store, owner_principal=OWNER)
        if allow:
            inquiries.allow_replies(actor=OWNER, on=True,
                                    reason="a reply on my own channel, carrying a code")
        return inquiries, local_sink(outdir)
    return make


def asked(store, waiting, asking, outdir, *, question=FERNS, decision="goal-activation"):
    """One live question, already sent: the store, its id, and the subject it asks about."""
    identifier = waiting()
    inquiries, sink = asking()
    opened = inquiries.ask(decision=decision, subject_id=identifier, question=question)
    assert opened["asked"] is True, opened
    assert inquiries.send_next(sink=sink, destination=CHANNEL, holder="test")["sent"] == 1
    return inquiries, opened["id"], identifier


# -- asking -------------------------------------------------------------------

def test_a_question_about_a_row_that_awaits_nobody_is_not_asked(store, waiting, asking):
    """Adopt the goal first and there is nothing left to ask: the fence says so."""
    identifier = waiting()
    inquiries, _sink = asking()
    GoalStore(store, events=DueEventLog(store),
              owner_principal=OWNER).activate(goal_id=identifier, actor=OWNER,
                                              reason="it is mine now")

    opened = inquiries.ask(decision="goal-activation", subject_id=identifier,
                           question=FERNS)

    assert opened["asked"] is False and "already active" in opened["reason"]
    assert inquiries.list() == []


def test_a_question_about_a_row_that_does_not_exist_is_refused(store, asking):
    inquiries, _sink = asking()
    with pytest.raises(EvidenceError, match="nothing to ask about"):
        inquiries.ask(decision="goal-activation", subject_id="gol_nonesuch",
                      question=FERNS)


def test_the_fence_is_computed_here_rather_than_handed_in(store, waiting, asking):
    """The caller cannot supply the digest, so it cannot ask about a state that never was."""
    identifier = waiting()
    inquiries, _sink = asking()
    opened = inquiries.ask(decision="goal-activation", subject_id=identifier,
                           question=FERNS)

    assert opened["subject_digest"] == fence(
        store, decision="goal-activation", subject_id=identifier)["digest"]


def test_nothing_the_store_holds_leaks_the_code_that_answers(store, waiting, asking, outdir):
    """The code is in the message and nowhere readable, including not in this table.

    The agent may read the archive; that is what makes it useful. So the ability to answer a
    question must be recoverable from nothing the archive holds — which is why only the
    digest of a random code is kept, and why the digest cannot be walked backwards.
    """
    inquiries, question, _identifier = asked(store, waiting, asking, outdir)
    code = code_of(outdir, question)

    dumped = json.dumps([item.as_dict() for item in inquiries.list()])
    columns = str(store.db.execute("SELECT * FROM inquiries").fetchone())
    receipts = str(store.db.execute("SELECT COALESCE(group_concat(proof, ''), '') || "
                                    "COALESCE(group_concat(settled, ''), '') FROM "
                                    "inquiries").fetchone()[0])

    assert code not in dumped and code not in columns and code not in receipts
    assert store.db.execute("SELECT code_digest FROM inquiries").fetchone()[0] != code


def test_a_question_asks_what_it_will_do_before_it_asks(store, waiting, asking, outdir):
    body = message_holding(outdir, asked(store, waiting, asking, outdir)[1])

    assert "It would settle: goal-activation" in body
    assert "answer with the code" not in body, "a bare code is not an instruction"
    assert "`yes " in body and "`no`" in body and "Until" in body


# -- answering ----------------------------------------------------------------

def test_a_reply_carrying_the_code_adopts_the_thing_it_asks_about(store, waiting, asking,
                                                                  outdir):
    inquiries, question, identifier = asked(store, waiting, asking, outdir)
    code = code_of(outdir, question)

    answered = inquiries.answer(reply=f"yes {code}", channel=CHANNEL)

    assert answered["settled"] is True, answered
    assert store.db.execute("SELECT status FROM goals WHERE id=?",
                            (identifier,)).fetchone()[0] == "active"
    assert inquiries.get(question).state == "answered"


def test_a_reply_without_a_code_is_somebodys_words_and_decides_nothing(store, waiting,
                                                                       asking, outdir):
    """The relay can put any text in this field, so agreement has to be specific.

    "yes" is the shape of sentence an agent could produce on its own; the code is what says a
    human read the message that carried it.
    """
    inquiries, _question, identifier = asked(store, waiting, asking, outdir)

    answered = inquiries.answer(reply="yes, obviously, do it")

    assert answered["settled"] is False and "no code" in answered["reason"]
    assert store.db.execute("SELECT status FROM goals WHERE id=?",
                            (identifier,)).fetchone()[0] == "candidate"


def test_naming_the_question_without_the_code_still_decides_nothing(store, waiting, asking,
                                                                    outdir):
    inquiries, question, identifier = asked(store, waiting, asking, outdir)

    answered = inquiries.answer(reply="adopt it", inquiry_id=question)

    assert answered["settled"] is False and "carries no code" in answered["reason"]
    assert inquiries.get(question).state == "sent"


def test_a_guessed_code_is_refused_and_written_down(store, waiting, asking, outdir):
    """Two wrong codes, two refusals, both recorded.

    A code that belongs to no question is a guess; a code that belongs to another question is
    either a copy-paste slip or somebody working through a list. Both say the same thing to
    the owner and both leave a line in the audit, because a refusal nobody can see is a
    refusal that can be tried again for free.
    """
    inquiries, question, _identifier = asked(store, waiting, asking, outdir)

    unknown = inquiries.answer(reply="yes QQQQQQ", channel=CHANNEL)
    other = inquiries.answer(reply="yes XXXXXX", inquiry_id=question, channel=CHANNEL)

    assert unknown["settled"] is False and "no question is open under" in unknown["reason"]
    assert other["settled"] is False and "does not belong" in other["reason"]
    assert store.db.execute("SELECT count(*) FROM audit WHERE action='inquiry_refused'"
                            ).fetchone()[0] == 2
    assert inquiries.get(question).state == "sent"          # still live, still askable


def test_the_same_answer_cannot_be_used_twice(store, waiting, asking, outdir):
    inquiries, question, _identifier = asked(store, waiting, asking, outdir)
    code = code_of(outdir, question)

    first = inquiries.answer(reply=f"yes {code}")
    second = inquiries.answer(reply=f"yes {code}")

    assert first["settled"] is True
    assert second["settled"] is False and "already answered" in second["reason"]


def test_a_bare_code_proves_the_owner_read_the_message_and_not_that_they_agreed(
        store, waiting, asking, outdir):
    """The code says who sent it. Only a word says what to do, and this one is missing."""
    inquiries, question, identifier = asked(store, waiting, asking, outdir)

    answered = inquiries.answer(reply=code_of(outdir, question), channel=CHANNEL)

    assert answered["settled"] is False and "neither a yes nor a no" in answered["reason"]
    assert store.db.execute("SELECT status FROM goals WHERE id=?",
                            (identifier,)).fetchone()[0] == "candidate"
    assert inquiries.get(question).state == "sent"


def test_a_no_stops_the_asking_without_stopping_the_decision(store, waiting, asking, outdir):
    """"No" is not a rejection of the goal, because rejecting it is a different act.

    It withdraws the question — the owner answered, and asking again would be arguing — and
    leaves the candidate exactly where it was, which is where `owner --list` can still show
    it. Reading a refusal as a decision is how a personal system ends up deleting the thing
    its owner only paused.
    """
    inquiries, question, identifier = asked(store, waiting, asking, outdir)

    declined = inquiries.answer(reply=f"no {code_of(outdir, question)}", channel=CHANNEL)

    assert declined["settled"] is False and declined["declined"] is True
    assert store.db.execute("SELECT status FROM goals WHERE id=?",
                            (identifier,)).fetchone()[0] == "candidate"
    assert inquiries.get(question).state == "withdrawn"
    assert store.db.execute("SELECT count(*) FROM audit WHERE action='inquiry_declined'"
                            ).fetchone()[0] == 1
    again = inquiries.ask(decision="goal-activation", subject_id=identifier, question=FERNS)
    assert again["asked"] is False and "ended as withdrawn" in again["note"]


def test_an_answer_about_a_moved_preview_is_refused(store, waiting, asking, outdir):
    """The owner was shown one state and the archive is another: the answer is stale.

    Restated as a goal revision, which is the ordinary thing that happens between a question
    being asked and an owner getting round to it.
    """
    inquiries, question, identifier = asked(store, waiting, asking, outdir)
    code = code_of(outdir, question)
    GoalStore(store, events=DueEventLog(store),
              owner_principal=OWNER).revise(goal_id=identifier, actor=OWNER,
                                            reason="it is a different plant",
                                            statement="It is very dry.")

    refused = inquiries.answer(reply=f"yes {code}")

    assert refused["settled"] is False and "changed since" in refused["reason"]
    assert inquiries.get(question).state == "void"


def test_a_reset_memory_voids_the_question_and_the_answer_with_it(store, waiting, asking,
                                                                  outdir):
    inquiries, question, _identifier = asked(store, waiting, asking, outdir)
    code = code_of(outdir, question)

    store.bump_epoch(reason="rebuilding from a corrected source", actor=OWNER)
    refused = inquiries.answer(reply=f"yes {code}")

    assert refused["settled"] is False and "reset" in refused["reason"]
    assert inquiries.get(question).state == "void"


def test_an_expired_question_cannot_be_answered_by_its_own_code(store, waiting, asking,
                                                                outdir):
    inquiries, question, _identifier = asked(store, waiting, asking, outdir)
    code = code_of(outdir, question)
    store.db.execute("UPDATE inquiries SET expires_at=? WHERE id=?",
                     ("2020-01-01T00:00:00+00:00", question))

    refused = inquiries.answer(reply=f"yes {code}")

    assert refused["settled"] is False and "expired" in refused["reason"]


# -- what a live question closes ----------------------------------------------

def test_a_second_door_will_not_decide_what_the_owner_is_being_asked_about(store, waiting,
                                                                          asking, outdir):
    """The failure this closes is a real one: an agent with a terminal read the question,
    answered itself, and left the owner holding a message that was still asking."""
    from hermes_memory.operations.decisions import settle

    inquiries, question, identifier = asked(store, waiting, asking, outdir)

    with pytest.raises(EvidenceError, match=f"{question} is being asked of the owner"):
        settle(store, owner_principal=OWNER, name="goal-activation", subject_id=identifier,
               actor=OWNER, reason="the owner presumably wants it")

    assert store.db.execute("SELECT status FROM goals WHERE id=?",
                            (identifier,)).fetchone()[0] == "candidate"
    assert inquiries.get(question).state == "sent", "a refusal does not end the asking"


def test_the_exemption_belongs_to_the_question_rather_than_to_whoever_asks(
        store, waiting, asking, outdir):
    """`via_inquiry` names the reply that is answering; naming it is not the same as being it."""
    from hermes_memory.operations.decisions import settle

    inquiries, _question, identifier = asked(store, waiting, asking, outdir)

    with pytest.raises(EvidenceError, match="being asked of the owner"):
        settle(store, owner_principal=OWNER, name="goal-activation", subject_id=identifier,
               actor=OWNER, reason="answered, more or less", via_inquiry="inq_other")


def test_declining_a_question_gives_the_other_door_back(store, waiting, asking, outdir):
    """A question the owner refused is not a fence around the decision forever after."""
    from hermes_memory.operations.decisions import settle

    inquiries, question, identifier = asked(store, waiting, asking, outdir)
    code = code_of(outdir, question)
    assert inquiries.answer(reply=f"no {code}")["declined"] is True

    answer = settle(store, owner_principal=OWNER, name="goal-activation",
                    subject_id=identifier, actor=OWNER, reason="decided it at the door instead")

    assert answer["status"] == "active"


# -- what the switch and the ladder decide ------------------------------------

def test_a_question_is_not_asked_at_all_where_replies_are_not_allowed(store, waiting, asking):
    """Asking would be a promise nobody can collect on, so it is not made."""
    identifier = waiting()
    inquiries, sink = asking(allow=False)
    opened = inquiries.ask(decision="goal-activation", subject_id=identifier,
                           question=FERNS)

    sent = inquiries.send_next(sink=sink, destination=CHANNEL, holder="test")

    assert sent["sent"] == 0
    assert inquiries.get(opened["id"]).state == "void"


def test_a_switch_flipped_off_after_the_ask_refuses_the_answer(store, waiting, asking,
                                                               outdir):
    """Turning the capability off has to take effect on the way in, not only on the way out."""
    inquiries, question, _identifier = asked(store, waiting, asking, outdir)
    code = code_of(outdir, question)
    inquiries.allow_replies(actor=OWNER, on=False, reason="changed my mind about this")

    refused = inquiries.answer(reply=f"yes {code}")

    assert refused["settled"] is False and "switched off" in refused["reason"]


def test_only_the_owner_principal_may_allow_replies(store, asking):
    inquiries, _sink = asking(allow=False)

    with pytest.raises(EvidenceError, match="owner principal"):
        inquiries.allow_replies(actor="agent:helper", on=True, reason="seemed useful")


def test_a_question_spends_the_owners_one_interruption_for_the_day(store, waiting, asking,
                                                                   outdir):
    """Two questions under a one-a-day ceiling: one goes, and the other says why it did not."""
    first, second = waiting("Water the ferns"), waiting("Send the invoice")
    inquiries, sink = asking()
    for identifier, title in ((first, "ferns"), (second, "invoice")):
        inquiries.ask(decision="goal-activation", subject_id=identifier,
                      question=f"Should I start reminding you about the {title}?")
    store.db.execute("UPDATE topic_policies SET max_immediate=1 WHERE topic='general'")

    sent = inquiries.send_next(sink=sink, destination=CHANNEL, holder="test", limit=5)

    assert sent["sent"] == 1, sent
    assert "budget" in sent["reason"]
    held = inquiries.list(states=("open",))
    assert len(held) == 1 and held[0].state == "open"
    assert "budget" in (held[0].reason or "")


def test_the_outbox_sees_what_the_question_queue_spent(store, waiting, asking, outdir):
    """The cap is on being interrupted, so both halves have to read one counter.

    Nothing in the outbox was told about inquiries; it asks the same policy the reminders
    always asked, and the answer now already carries what the queue spent. Two components each
    allowed five a day is ten interruptions the owner never agreed to.
    """
    asked(store, waiting, asking, outdir)
    store.db.execute("UPDATE topic_policies SET max_immediate=1 WHERE topic='general'")
    policy = AttentionPolicy(store, owner_principal=OWNER)

    gate = policy.decide(topic="general", urgency="proactive")

    assert gate.action != "notify_owner", gate.reason
    assert any("budget" in item for item in gate.downgrades), gate.downgrades
    counts = policy.counts("general")
    assert counts["asked_today"] == 1 and counts["interrupted_today"] == 1


def test_quiet_hours_hold_a_question_and_say_when_to_ask_it(store, waiting, asking):
    identifier = waiting()
    inquiries, sink = asking()
    inquiries.ask(decision="goal-activation", subject_id=identifier, question=FERNS)
    store.db.execute("UPDATE topic_policies SET quiet_from='00:00', quiet_until='23:59' "
                     "WHERE topic='general'")

    sent = inquiries.send_next(sink=sink, destination=CHANNEL, holder="test")

    assert sent["sent"] == 0
    report = sent["reports"][0]
    assert report["ok"] is True and "held, not refused" in report["reason"]
    assert report.get("retry_at"), "a held question has to say when it is worth asking again"
    assert inquiries.for_subject("goal-activation", identifier).state == "open"


def test_a_transport_that_failed_lets_the_question_be_asked_again(store, waiting, asking,
                                                                  outdir):
    """A question that never arrived is not a question the owner declined."""
    identifier = waiting()
    inquiries, _sink = asking()
    inquiries.ask(decision="goal-activation", subject_id=identifier, question=FERNS)

    def broken(body):
        raise RuntimeError("the gateway is not running")

    failed = inquiries.send_next(sink=broken, destination=CHANNEL, holder="test")
    assert failed["sent"] == 0 and failed["ok"] is False
    assert "gateway is not running" in inquiries.for_subject(
        "goal-activation", identifier).reason

    again = inquiries.send_next(sink=local_sink(outdir), destination=CHANNEL, holder="test")
    assert again["sent"] == 1, again


# -- the subjects a question can have -----------------------------------------

def test_a_forgetting_question_carries_the_digest_the_door_demands(store, asking, outdir):
    """The erasure fence is the real one: the same 64 characters `confirm` insists on.

    So a reply confirms exactly the blast radius that was previewed, and nothing about the
    act differs from the same confirmation typed at a keyboard — which is the whole reason
    both handles call one function.
    """
    record = store.commit(envelope(source="gmail", source_id="fern-1",
                                   text="The fern report is in the drawer."))["id"]
    preview = ErasureManager(store, owner_principal=OWNER).preview(
        record_ids=[record], actor="agent:helper", reason="it was a mistake to keep")
    inquiries, sink = asking()

    opened = inquiries.ask(decision="forgetting", subject_id=preview["intent_id"],
                           question="May I forget the fern report and everything derived "
                                    "from it?")
    inquiries.send_next(sink=sink, destination=CHANNEL, holder="test")

    assert opened["subject_digest"] == preview["preview_digest"]
    settled = inquiries.answer(
        reply=f"yes {code_of(outdir, opened['id'])}", channel=CHANNEL)
    assert settled["settled"] is True, settled
    assert store.db.execute("SELECT deleted FROM records WHERE id=?",
                            (record,)).fetchone()[0] == 1


def test_an_identity_question_is_voided_when_the_candidate_is_decided_elsewhere(store, asking,
                                                                                outdir):
    """Two handles on one decision: whoever settles it first, the question stops asking."""
    identities = IdentityStore(store, owner_principal=OWNER)
    one = identities.account("email", "a@example.test")
    two = identities.account("email", "b@example.test")
    made = identities.propose(
        account_a=one, account_b=two, rule="email-normalized-equal",
        basis="one mailbox, two spellings",
        evidence=[store.commit(envelope(source="gmail", source_id="id-1",
                                        text="same signature block"))["id"]],
        proposed_by="agent:helper")["candidate_id"]
    inquiries, sink = asking()
    inquiries.ask(decision="identity", subject_id=made,
                  question="Are these two addresses the same person?")
    inquiries.send_next(sink=sink, destination=CHANNEL, holder="test")
    question = inquiries.for_subject("identity", made).id
    code = code_of(outdir, question)

    identities.confirm(candidate_id=made, actor=OWNER, reason="I checked with them")
    refused = inquiries.answer(reply=f"yes {code}")

    assert refused["settled"] is False
    assert inquiries.get(question).state == "void"


def test_the_answer_is_attributed_to_the_owner_and_says_which_channel_relayed_it(
        store, waiting, asking, outdir):
    inquiries, question, identifier = asked(store, waiting, asking, outdir)

    inquiries.answer(reply=f"yes {code_of(outdir, question)}", channel=CHANNEL)

    history = store.db.execute("SELECT changed_by FROM goal_history WHERE goal_id=? ORDER "
                               "BY rowid DESC LIMIT 1", (identifier,)).fetchone()
    assert history["changed_by"] == OWNER
    row = inquiries.get(question)
    assert row.answer_channel == CHANNEL
    audit = store.db.execute("SELECT metadata FROM audit WHERE action='inquiry_answered' "
                             "AND object_id=?", (question,)).fetchone()
    assert json.loads(audit["metadata"])["channel"] == CHANNEL


def test_every_answerable_subject_is_listed_with_where_its_row_lives():
    """A subject with no fence entry cannot be asked about, so the table is the surface.

    A reply settling something absent from here would have no way to know what it confirmed,
    which is the failure mode the fence exists to close.
    """
    from hermes_memory.operations.decisions import ACTS

    covered = set(FENCES) | {"forgetting"}
    assert covered == set(ACTS), sorted(covered ^ set(ACTS))
    for decision, (table, keys, state_column, awaiting) in FENCES.items():
        assert table and keys and state_column and awaiting, decision


# -- who opens a question, and who sends one -----------------------------------

def a_pass(store):
    return Maintenance(store, owner_principal=OWNER, sync=None, gate=None)


def test_the_pass_opens_a_question_and_opens_no_channel(store, waiting, asking, outdir):
    """Asking is a write to a queue. The transport belongs to the drain, not to the pass.

    The background pass has one rule that outlives every feature added to it: no model call,
    no channel. A question that reached the owner from inside it would break that rule
    quietly, so the pass that finds the backlog is not the thing that rings the bell.
    """
    identifier = waiting()
    inquiries, _sink = asking()

    report = a_pass(store).pass_now(sections=("questions",))["questions"]

    assert report["asked"] == 1 and report["subjects"] == [f"goal-activation:{identifier}"]
    assert inquiries.for_subject("goal-activation", identifier).state == "open"
    assert not list((outdir / "memory" / "delivered").glob("*.md"))


def test_a_second_pass_asks_the_same_thing_once(store, waiting, asking):
    """The id comes from the state, so a pass on a timer cannot grow a question per period."""
    identifier = waiting()
    inquiries, _sink = asking()

    for _ in range(3):
        report = a_pass(store).pass_now(sections=("questions",))["questions"]

    assert report["asked"] == 0
    assert len(inquiries.list()) == 1


def test_the_pass_does_not_ask_about_what_the_owner_proposed(store, asking):
    """An owner who typed a candidate at the door does not need to be asked if they meant it."""
    identifier = GoalStore(store, events=DueEventLog(store),
                           owner_principal=OWNER).propose(
        title="Water the ferns", statement="It is dry.", timezone_name="UTC",
        proposed_by=OWNER, proposed_kind="owner")["id"]
    inquiries, _sink = asking()

    report = a_pass(store).pass_now(sections=("questions",))["questions"]

    assert report["asked"] == 0
    assert inquiries.for_subject("goal-activation", identifier) is None


def test_the_pass_asks_nothing_while_replies_are_switched_off(store, waiting, asking):
    """Switched off means the question could be asked and could not be answered. Writing it
    down anyway would be a queue of things nobody can drain."""
    waiting()
    inquiries, _sink = asking(allow=False)

    report = a_pass(store).pass_now(sections=("questions",))["questions"]

    assert report["asked"] == 0 and "switched off" in report["note"]
    assert inquiries.list() == []


def test_the_drain_asks_the_question_beside_the_reminders(store, waiting, asking, outdir):
    """One drain, one transport, both kinds of message — and the report says how many of each."""
    identifier = waiting()
    inquiries, sink = asking()
    opened = inquiries.ask(decision="goal-activation", subject_id=identifier, question=FERNS)
    policy = DeliveryPolicy(enabled=True, destination=CHANNEL, owner_principal=OWNER)

    report = deliver_ready(store, policy=policy, sink=sink, limit=5, holder="test")

    assert report["delivered"] == 0 and report["asked"] == 1, report
    assert "1 question(s) asked" in report["reason"]
    asked_row = inquiries.get(opened["id"])
    assert asked_row.state == "sent"
    assert code_of(outdir, asked_row.id)


# -- helpers ------------------------------------------------------------------

def message_holding(outdir, question_id) -> str:
    """The message a local transport delivered for this question."""
    for path in sorted((outdir / "memory" / "delivered").glob("*.md")):
        body = path.read_text(encoding="utf-8")
        if question_id in body:
            return body
    raise AssertionError(f"nothing was delivered that names {question_id}")


def code_of(outdir, question_id) -> str:
    """The code that answers a question, read from the owner's side of the wire.

    Deliberately read out of the delivered message rather than out of the store: the store
    cannot answer this question, which is the property half of this file keeps checking.
    """
    found = re.search(r"`yes ([A-Z2-9]{6})`", message_holding(outdir, question_id))
    assert found, "the message that asks must carry the code that answers"
    assert found.group(1) == code_in(found.group(1))
    return found.group(1)
