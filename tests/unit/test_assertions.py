"""C6 assertions: typed claims, their intervals, and the quote that holds them up."""
from __future__ import annotations

import pytest

from conftest import envelope
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.knowledge.assertions import (CANDIDATE, CONFIRMED, RETRACTED, SUPERSEDED,
                                                AssertionStore)
from hermes_memory.storage.evidence import EvidenceError

OWNER = "owner-principal"
TEXT = "I work from home on Tuesdays and I take the coffee without sugar."


@pytest.fixture()
def claims(store):
    return AssertionStore(store, owner_principal=OWNER)


@pytest.fixture()
def note(store):
    return store.commit(envelope(text=TEXT))["id"]


def propose(claims, record, **overrides):
    args = {"subject": "julia", "predicate": "works_from_home_on", "value": "Tuesday",
            "kind": "preference", "evidence_kind": "owner_declared", "record_id": record,
            "quote": "I work from home on Tuesdays", "proposed_by": OWNER}
    args.update(overrides)
    return claims.propose(**args)


# -- quoting ------------------------------------------------------------------


def test_a_claim_enters_confirmed_only_when_a_human_said_the_quoted_thing(
        store, claims, note):
    direct = propose(claims, note)

    assert direct["status"] == CONFIRMED
    assert direct["quote_start"] == TEXT.index("I work from home")
    assert claims.current()[0].confirmed_by == OWNER


@pytest.mark.parametrize("evidence_kind", ["observed_pattern", "derived"])
def test_what_a_model_noticed_starts_as_a_hypothesis(store, claims, note, evidence_kind):
    created = propose(claims, note, evidence_kind=evidence_kind,
                      proposed_by="agent-model")

    assert created["status"] == CANDIDATE
    assert created["confirmed_by"] is None
    assert claims.current() == []
    assert [item.id for item in claims.current(include_candidates=True)] == [created["id"]]


def test_a_claim_cannot_borrow_a_citation_from_a_sentence_it_does_not_match(
        store, claims, note):
    with pytest.raises(EvidenceError, match="does not appear in the cited evidence"):
        propose(claims, note, quote="I work from home on Mondays")
    assert store.db.execute("SELECT count(*) FROM assertions").fetchone()[0] == 0


def test_an_ambiguous_quote_is_refused_unless_the_span_is_named(store, claims):
    record = store.commit(envelope(source_id="msg-repeat",
                                   text="Thursday works. Thursday works better."))["id"]

    with pytest.raises(EvidenceError, match="appears 2 times"):
        propose(claims, record, quote="Thursday works")
    sentence = "Thursday works. Thursday works better."
    second = sentence.index("Thursday works", 1)
    named = propose(claims, record, quote="Thursday works",
                    span=(second, second + len("Thursday works")),
                    value="the later mention")
    assert named["quote_start"] == second


def test_a_span_that_does_not_hold_what_it_claims_is_refused(store, claims, note):
    with pytest.raises(EvidenceError, match="not taken on trust"):
        propose(claims, note, quote="without sugar", span=(0, 13))


def test_evidence_that_cannot_be_read_cannot_support_anything(store, claims, note):
    store.hide(note, reason="retracted by the author", actor=OWNER)

    with pytest.raises(EvidenceError, match="not retrievable"):
        propose(claims, note)
    assert store.db.execute("SELECT count(*) FROM assertions").fetchone()[0] == 0


def test_a_backend_document_id_is_not_evidence(store, claims):
    with pytest.raises(EvidenceError, match="canonical record id"):
        propose(claims, "hdoc0123456789abcdef0123456789abcdef")


def test_replaying_the_same_claim_does_not_fork_a_second_row(store, claims, note):
    first = propose(claims, note)
    second = propose(claims, note, quote="I work from home on Tuesdays")

    assert second["created"] is False and second["id"] == first["id"]
    assert [row[0] for row in store.db.execute("SELECT status FROM assertions")] == [CONFIRMED]


# -- shape of the vocabulary ---------------------------------------------------


@pytest.mark.parametrize("kind, value, unit, message", [
    ("measurement", "95", None, "needs a unit"),
    ("measurement", "high", "%", "not a number"),
    ("preference", "Tuesday", "day", "only a measurement carries a unit"),
])
def test_a_measurement_has_to_be_comparable_and_nothing_else_carries_a_unit(
        store, claims, note, kind, value, unit, message):
    with pytest.raises(EvidenceError, match=message):
        propose(claims, note, kind=kind, value=value, unit=unit)


def test_the_same_claim_written_two_ways_is_one_claim(store, claims, note):
    one = propose(claims, note, value="Tuesday")
    other = propose(claims, note, value="  Tuesday  ")

    assert one["id"] == other["id"]
    assert len(claims.current()) == 1


def test_an_unparseable_time_window_is_refused(store, claims, note):
    with pytest.raises(EvidenceError, match="valid_to precedes valid_from"):
        propose(claims, note, valid_from="2026-05-01T00:00:00+00:00",
                valid_to="2026-04-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="timezone"):
        propose(claims, note, valid_from="2026-05-01T00:00:00")


# -- decisions -----------------------------------------------------------------


def test_confirmation_and_retraction_belong_to_the_owner(store, claims, note):
    candidate = propose(claims, note, evidence_kind="observed_pattern",
                        proposed_by="agent-model")["id"]

    with pytest.raises(EvidenceError, match="belongs to the owner"):
        claims.confirm(assertion_id=candidate, actor="agent-model", reason="looks right")
    with pytest.raises(EvidenceError, match="belongs to the owner"):
        claims.retract(assertion_id=candidate, actor="agent-model", reason="undo")

    assert claims.confirm(assertion_id=candidate, actor=OWNER,
                          reason="yes, that is me")["status"] == CONFIRMED


def test_without_an_owner_named_nothing_can_be_confirmed(store, note):
    anonymous = AssertionStore(store)
    candidate = anonymous.propose(subject="julia", predicate="prefers", value="quiet",
                                 kind="preference", evidence_kind="observed_pattern",
                                 record_id=note, quote="without sugar",
                                 proposed_by="agent-model")["id"]

    with pytest.raises(EvidenceError, match="OWNER_PRINCIPAL"):
        anonymous.confirm(assertion_id=candidate, actor="anyone", reason="trust me")


def test_a_retracted_claim_stays_retracted(store, claims, note):
    assertion = propose(claims, note)["id"]
    claims.retract(assertion_id=assertion, actor=OWNER, reason="no longer true")

    with pytest.raises(EvidenceError, match="un-deleting"):
        claims.confirm(assertion_id=assertion, actor=OWNER, reason="actually keep it")
    assert claims.current() == []


def test_supersession_moves_the_claim_forward_without_deleting_the_belief(store, claims, note):
    old = propose(claims, note, value="Tuesday")["id"]
    new = propose(claims, note, value="Wednesday", supersedes=old,
                  quote="I take the coffee without sugar",
                  valid_from="2026-09-01T00:00:00+00:00")

    assert new["status"] == CONFIRMED
    assert claims.get(old).status == SUPERSEDED
    assert [item.value for item in claims.current()] == ["Wednesday"]
    assert claims.current(at="2020-01-01T00:00:00+00:00") == []


def test_supersession_cannot_cross_the_thing_being_talked_about(store, claims, note):
    other = propose(claims, note, subject="someone-else", predicate="prefers",
                    value="tea")["id"]

    with pytest.raises(EvidenceError, match="cannot cross"):
        propose(claims, note, value="coffee", supersedes=other)


def test_a_claim_that_lost_its_evidence_stops_being_returned(store, claims, note):
    assertion = propose(claims, note)["id"]
    store.hide(note, reason="the author withdrew it", actor=OWNER)

    assert claims.current() == []
    review = claims.needs_review()
    assert [item["assertion"]["id"] for item in review] == [assertion]
    assert review[0]["reason"] == "the cited evidence is hidden or forgotten"


def test_a_correction_sends_the_claim_that_quoted_the_old_wording_to_review(
        store, claims, note):
    assertion = propose(claims, note)["id"]
    store.commit(envelope(revision="2", text="Correction: I work from home on Wednesdays."))

    # Ingress supersedes the revision it replaces, so a claim resting on the withdrawn
    # wording leaves `current()` and becomes reviewable. This is the same door an erasure
    # reaches, arrived at by a source that simply changed its mind: the correction does not
    # delete the claim, it stops pretending the evidence is still there.
    assert claims.current() == []
    review = claims.needs_review()
    assert [item["assertion"]["id"] for item in review] == [assertion]
    assert review[0]["reason"] == "the cited evidence is hidden or forgotten"


def test_forgetting_the_evidence_invalidates_the_claim_through_the_real_path(
        store, claims, note):
    claim = propose(claims, note)
    manager = ErasureManager(store, owner_principal=OWNER)
    preview = manager.preview(record_ids=[note], actor=OWNER, reason="withdrawn")
    manager.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                    actor=OWNER)

    assert claims.current() == []
    assert claims.get(claim["id"]) is not None, "the claim is kept, and reported as reviewable"
    assert claims.needs_review()[0]["reason"] == "the cited evidence is hidden or forgotten"


def test_a_quoted_span_that_no_longer_matches_is_not_support(store, claims, note):
    """A store restored from an older snapshot still looks live.

    The row is rewritten directly because nothing in the runtime may do it: the
    point is that support is re-read against the evidence, so even a database
    that changed underneath cannot leave a quotation standing on other text.
    """
    created = propose(claims, note)
    store.db.execute("BEGIN IMMEDIATE")
    store.db.execute("UPDATE records SET text=? WHERE id=?",
                     ("A different sentence that was never said.", note))
    store.db.execute("COMMIT")

    assertion = claims.get(created["id"])
    assert claims.support(assertion) == (False, "the quoted span no longer matches the "
                                                "cited evidence")
    assert claims.current() == []
    assert claims.needs_review()[0]["reason"].startswith("the quoted span")


# -- retrieval ------------------------------------------------------------------


def test_a_question_finds_claims_through_any_of_their_parts(store, claims, note):
    propose(claims, note, value="Tuesday")
    propose(claims, note, subject="coffee-order", predicate="takes", value="black",
            quote="I take the coffee")

    by_subject = [item.subject for item in claims.matching("what about julia")]
    by_value = [item.value for item in claims.matching("tuesday")]
    ranked = claims.matching("julia works_from_home_on Tuesday")

    assert by_subject == ["julia"]
    assert by_value == ["Tuesday"]
    assert ranked[0].value == "Tuesday", "the claim the question named most goes first"


def test_single_letters_are_not_treated_as_a_question(store, claims, note):
    created = propose(claims, note)

    assert claims.matching("a i o") == [], "one letter matches everything and means nothing"
    assert claims.matching("") == []
    assert [item.id for item in claims.matching("work home Tuesdays")] == [created["id"]]


def test_a_query_in_any_script_finds_its_claim(store, claims, note):
    record = store.commit(envelope(source_id="msg-jp", text="会議は木曜日に変更しました"))["id"]
    created = propose(claims, record, subject="会議", predicate="日", value="木曜日",
                      kind="fact", evidence_kind="explicit_statement",
                      quote="会議は木曜日に", proposed_by=OWNER)

    assert [item.id for item in claims.matching("会議")] == [created["id"]]


def test_retrieval_skips_unconfirmed_and_unsupported_claims(store, claims, note):
    kept = propose(claims, note)["id"]
    hypothesis = propose(claims, note, value="Friday", subject="someone-else",
                         evidence_kind="derived", proposed_by="agent-model")["id"]
    assert kept != hypothesis
    store.hide(note, reason="withdrawn", actor=OWNER)

    assert claims.matching("julia someone-else") == []


# -- contradictions ------------------------------------------------------------


def test_two_supported_claims_that_cannot_both_be_true_are_reported(store, claims, note):
    propose(claims, note, value="Tuesday")
    propose(claims, note, value="Friday", quote="I take the coffee without sugar")

    found = claims.contradictions()

    assert [(item.subject, item.predicate) for item in found] == [("julia", "works_from_home_on")]
    assert found[0].as_dict()["values"] == ["Friday", "Tuesday"]
    assert len(found[0].assertions) == 2, "both stay on file; nothing picks a winner"


def test_beliefs_and_hypotheses_are_not_contradictions(store, claims, note):
    propose(claims, note, kind="belief", value="Tuesday is luckier")
    propose(claims, note, kind="belief", value="Friday is luckier",
            quote="I take the coffee without sugar")

    assert claims.contradictions() == []
    unproven = propose(claims, note, evidence_kind="derived", value="Monday",
                       proposed_by="agent-model")
    assert claims.contradictions() == [], f"{unproven['status']} cannot contradict anything"


def test_claims_that_do_not_overlap_in_time_are_not_a_contradiction(store, claims, note):
    propose(claims, note, value="Tuesday", valid_to="2026-01-01T00:00:00+00:00")
    propose(claims, note, value="Wednesday", quote="I take the coffee without sugar",
            valid_from="2026-01-02T00:00:00+00:00")

    assert claims.contradictions() == []
    assert [item.value for item in claims.current(at="2026-06-01T00:00:00+00:00")] == \
        ["Wednesday"]
    assert [item.value for item in claims.current(at="2025-06-01T00:00:00+00:00")] == \
        ["Tuesday"]


def test_every_claim_carries_the_span_a_caller_can_quote(store, claims, note):
    propose(claims, note)

    assertion = claims.current()[0]
    payload = assertion.as_dict()

    assert set(payload) >= {"id", "subject", "predicate", "value", "kind", "unit",
                            "evidence_kind", "record_id", "quote", "quote_start",
                            "quote_end", "valid_from", "valid_to", "status",
                            "confirmed_by", "supersedes", "revision"}
    assert store.get(note).text[payload["quote_start"]:payload["quote_end"]] == \
        payload["quote"]
