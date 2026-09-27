"""C9 context broker: per-channel honesty, bounded output, authority by identity.

The recurring shape in these tests is that two very different situations produce
one packet, and the packet has to say which one it is: found nothing versus
could not look, slow versus broken, cached-versus-current, one person's evidence
versus another's.
"""
from __future__ import annotations

import time

import pytest

from conftest import envelope
from hermes_memory.backend.hindsight_client import RecallOutcome
from hermes_memory.context import ContextBroker, PacketCache, fts_expression
from hermes_memory.storage.evidence import EvidenceStore
from hermes_memory.storage.identity import IdentityStore

ME = "me@example.com"
STRANGER = "stranger@example.com"


class FakeBackend:
    """Stands in for Hindsight, using the real outcome shape."""

    def __init__(self, *, results=(), truncated=(), error=None, sleep_s=0.0, during=None):
        self.results = tuple(results)
        self.truncated = tuple(truncated)
        self.error = error
        self.sleep_s = sleep_s
        self.during = during
        self.calls: list[tuple[str, dict]] = []

    def recall(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.sleep_s:
            time.sleep(self.sleep_s)
        if self.during is not None:
            self.during()
        if self.error is not None:
            raise self.error
        return RecallOutcome(results=self.results, truncated=self.truncated)


@pytest.fixture()
def broker(store):
    built = ContextBroker(store, cache=None)
    yield built
    built.close()


# -- the two channels report separately --------------------------------------


def test_a_stopped_backend_still_yields_a_useful_packet(store, broker):
    store.commit(envelope(source_id="msg-a", text="The kickoff meeting is Monday at 10."))
    store.commit(envelope(source_id="msg-b", text="The budget meeting moved to Friday."))

    packet = broker.assemble("meeting")

    assert [item.source for item in packet.items] == ["gmail", "gmail"]
    assert packet.channels.lexical == "available"
    assert packet.channels.derived == "not_configured"
    assert "these results are lexical only" in packet.render()
    assert packet.coverage == "supported"


def test_derived_text_without_local_evidence_is_never_claimed_as_supported(store, broker):
    backend = FakeBackend(results=[{"text": "The user prefers terse answers."}])
    broker.client = backend

    packet = broker.assemble("preferences")

    assert packet.facts and not packet.items
    assert packet.coverage == "partial"
    assert "derived fact" in packet.render()


def test_a_down_backend_is_not_reported_as_an_empty_archive(store, broker):
    broker.client = FakeBackend(error=ConnectionError("connection refused"))

    packet = broker.assemble("anything at all")

    assert packet.channels.derived == "unavailable"
    assert "connection refused" in packet.channels.detail
    # Nothing was found, but nothing was ruled out either.
    assert packet.coverage == "partial"
    assert "derived retrieval is unavailable" in packet.render()


def test_nothing_found_on_both_answered_channels_is_unknown(store, broker):
    store.commit(envelope(text="Unrelated note about the garden."))
    broker.client = FakeBackend()

    packet = broker.assemble("quantum tunnelling")

    assert packet.coverage == "unknown"
    assert packet.render() == ""
    assert packet.channels.derived == "available"


def test_a_slow_backend_costs_the_caller_the_deadline_and_not_the_request(store):
    broker = ContextBroker(store, cache=None, derived_timeout_s=0.05)
    broker.client = FakeBackend(sleep_s=0.5, results=[{"text": "late"}])
    store.commit(envelope(text="The kickoff meeting is Monday."))

    started = time.monotonic()
    packet = broker.assemble("kickoff meeting")
    waited = time.monotonic() - started

    assert waited < 0.4, "the derived deadline bounded the report, not the wait"
    assert packet.channels.derived == "timeout"
    assert packet.items, "the lexical half is still a useful answer"
    broker.close()


def test_truncated_provenance_is_reported_instead_of_assumed(store, broker):
    broker.client = FakeBackend(results=[{"text": "derived"}], truncated=("source_facts",))

    packet = broker.assemble("anything")

    assert packet.channels.derived == "partial"
    assert "source_facts" in packet.truncated
    assert "provenance" in packet.render()


# -- bounded output -----------------------------------------------------------


def test_the_token_ceiling_holds_and_says_so(store, broker):
    for index in range(12):
        store.commit(envelope(source_id=f"msg-{index}", text="meeting notes " * 60))
    tight = ContextBroker(store, cache=None, budget_tokens=120)

    packet = tight.assemble("meeting", limit=20)

    assert packet.tokens_used <= 120
    assert 0 < len(packet.items) < 12
    assert packet.items[-1].span_truncated is True, "a prefix beats amnesia"
    assert "packet" in packet.truncated
    assert packet.coverage == "partial"
    assert "not the whole answer" in packet.render()

    # A sliver of room cannot hold a sentence, so nothing is claimed at all.
    starving = ContextBroker(store, cache=None, budget_tokens=40)
    assert starving.assemble("meeting", limit=20).items == ()
    tight.close()
    starving.close()


def test_derived_text_is_charged_against_the_same_ceiling(store):
    tight = ContextBroker(store, cache=None, budget_tokens=60)
    tight.client = FakeBackend(results=[{"text": "word " * 4000}])

    packet = tight.assemble("anything")
    tight.close()

    assert packet.facts == ()
    assert packet.tokens_used <= 60
    assert "packet" in packet.truncated


def test_an_open_commitment_is_not_crowded_out_by_evidence(store):
    for index in range(6):
        store.commit(envelope(source_id=f"msg-{index}", text="meeting " * 200))
    broker = ContextBroker(store, cache=None, budget_tokens=90)

    packet = broker.assemble("meeting", commitments=[{"title": "Renew the TLS certificate",
                                                      "due_at": "2026-10-01"}])
    assert packet.commitments and "Renew the TLS certificate" in packet.render()
    broker.close()


def test_an_over_long_query_reports_the_terms_it_dropped(store, broker):
    store.commit(envelope(text="alpha"))
    query = " ".join(f"term{i}" for i in range(40))

    packet = broker.assemble(query)

    assert "terms" in packet.truncated
    assert packet.channels.lexical == "available"
    assert fts_expression(query).count('"') == 48


# -- authority ----------------------------------------------------------------


def test_punctuation_in_a_question_is_not_mistaken_for_an_outage(store, broker):
    store.commit(envelope(text="The meeting moved to Thursday."))

    packet = broker.assemble('the meeting "quoted" (parens) AND NOT OR * ^')

    assert packet.channels.lexical == "available"
    assert packet.items == ()


def test_a_broken_index_is_an_outage_and_not_zero_hits(store, broker, monkeypatch):
    store.commit(envelope(text="The meeting moved to Thursday."))

    def broken(_match, **_kwargs):
        raise RuntimeError("no such module: fts5")

    monkeypatch.setattr(store, "search", broken)
    packet = broker.assemble("meeting")

    assert packet.channels.lexical == "unavailable"
    assert "fts5" in packet.channels.detail
    assert packet.coverage == "partial"
    assert "lexical retrieval unavailable" in packet.render()


def test_account_scoped_evidence_never_reaches_another_account(store, broker):
    identity = IdentityStore(store, owner_principal="owner")
    mine, theirs = identity.account("email", ME), identity.account("email", STRANGER)
    broker.identity = identity
    store.commit(envelope(source_id="mine", text="My dentist appointment.",
                          metadata={"account_ids": [mine]}))
    store.commit(envelope(source_id="theirs", text="Their dentist appointment.",
                          metadata={"account_ids": [theirs]}))
    store.commit(envelope(source_id="shared", text="An appointment reminder note."))

    packet = broker.assemble("appointment", account_id=mine)

    assert [item.source_id for item in packet.items] == ["mine", "shared"]
    assert packet.withheld == 1
    assert "withheld as outside the caller's scope" in packet.render()


def test_an_unconfirmed_candidate_does_not_widen_access(store, broker):
    identity = IdentityStore(store, owner_principal="owner")
    mine, theirs = identity.account("email", ME), identity.account("email", STRANGER)
    broker.identity = identity
    support = store.commit(envelope(text="A thread naming both addresses."))
    store.commit(envelope(source_id="theirs", text="Their private prescription.",
                          metadata={"account_ids": [theirs]}))
    candidate = identity.propose(account_a=mine, account_b=theirs,
                                 rule="email-thread-participant", basis="same thread",
                                 evidence=[support["id"]], proposed_by="agent")

    before = broker.assemble("prescription", account_id=mine)
    assert before.items == () and before.withheld == 1

    identity.confirm(candidate_id=candidate["candidate_id"], actor="owner",
                     reason="owner says they are the same person")
    after = broker.assemble("prescription", account_id=mine)
    assert len(after.items) == 1, "confirmed identity expansion is what C9 gathers"


def test_an_expired_join_stops_widening_access(store, broker):
    """The packet asks who this caller is *now*.

    A confirmed edge whose validity interval has closed is a true statement about a
    period that ended; keeping it open here would widen one caller's view into another
    person's evidence on a decision the owner had already bounded.
    """
    identity = IdentityStore(store, owner_principal="owner")
    mine, theirs = identity.account("email", ME), identity.account("email", STRANGER)
    broker.identity = identity
    support = store.commit(envelope(text="A thread naming both addresses."))
    store.commit(envelope(source_id="theirs", text="Their private prescription.",
                          metadata={"account_ids": [theirs]}))
    candidate = identity.propose(account_a=mine, account_b=theirs,
                                 rule="email-thread-participant", basis="same thread",
                                 evidence=[support["id"]], proposed_by="agent")
    identity.confirm(candidate_id=candidate["candidate_id"], actor="owner",
                     reason="the same person, until they were not",
                     valid_from="2019-01-01T00:00:00+00:00",
                     valid_until="2020-01-01T00:00:00+00:00")

    packet = broker.assemble("prescription", account_id=mine)

    assert packet.items == () and packet.withheld == 1


def test_scoping_does_not_depend_on_someone_having_injected_identity(store, broker):
    """An identity store needs the same database the packet reads, so there is no
    configuration in which the caller's account is honoured and one in which it is
    quietly ignored."""
    identity = IdentityStore(store, owner_principal="owner")
    mine = identity.account("email", ME)
    stranger = identity.account("email", STRANGER)
    store.commit(envelope(source_id="private", text="My private appointment.",
                          metadata={"account_ids": [mine]}))

    packet = broker.assemble("appointment", account_id=stranger)

    assert packet.items == () and packet.withheld == 1


def test_a_lesson_learned_from_another_person_is_not_taught_to_the_caller(store, broker):
    """A lesson is a conclusion about someone's records; repeating it is repeating
    that evidence. The rule is the evidence rule, one step back.

    The last line is the one that matters for the tool surface: a caller that hands
    the broker a lesson it has no business having cannot make the packet show it.
    """
    identity = IdentityStore(store, owner_principal="owner")
    mine, theirs = identity.account("email", ME), identity.account("email", STRANGER)
    broker.identity = identity
    store.commit(envelope(source_id="mine", text="My invoice is due.",
                          metadata={"account_ids": [mine]}))
    cited = store.commit(envelope(source_id="theirs", text="Their invoice is due.",
                                  metadata={"account_ids": [theirs]}))["id"]
    private = {"id": "chase-theirs", "version": 1, "text": "Chase their invoice by phone.",
               "evidence": [cited]}
    shared = {"id": "file-mine", "version": 1, "text": "File my receipt.",
              "evidence": [str(store.db.execute(
                  "SELECT id FROM records WHERE source_id='mine'").fetchone()["id"])]}

    packet = broker.assemble("invoice", account_id=mine, lessons=[private, shared])

    assert [item["id"] for item in packet.lessons] == ["file-mine"]
    assert "lessons" in packet.truncated
    assert "1 lesson(s) belong to evidence this caller is not joined to" in \
        packet.channels.detail
    assert packet.coverage == "partial", "the packet is missing practice it could not show"

    candidate = identity.propose(account_a=mine, account_b=theirs,
                                 rule="email-thread-participant", basis="same thread",
                                 evidence=[cited], proposed_by="agent")
    identity.confirm(candidate_id=candidate["candidate_id"], actor="owner",
                     reason="one person")
    joined = broker.assemble("invoice", account_id=mine, lessons=[private, shared])
    assert [item["id"] for item in joined.lessons] == ["chase-theirs", "file-mine"]


def test_a_caller_the_installation_cannot_identify_gets_unscoped_practice_only(store, broker):
    """The lesson rule is the evidence rule, so it fails the same way: an unknown
    caller sees the practice derived from records that name nobody, and nothing else."""
    identity = IdentityStore(store, owner_principal="owner")
    theirs = identity.account("email", STRANGER)
    broker.identity = identity
    cited = store.commit(envelope(source_id="theirs", text="Their invoice.",
                                  metadata={"account_ids": [theirs]}))["id"]

    packet = broker.assemble("invoice", account_id=None,
                             lessons=[{"id": "chase", "text": "Chase it.",
                                       "evidence": [cited]}])

    assert [item["id"] for item in packet.lessons] == [], "unknown caller, so not this lesson"
    shared = broker.assemble("invoice", account_id=None,
                             lessons=[{"id": "file", "text": "File the receipt.",
                                       "evidence": [store.commit(
                                           envelope(source_id="public",
                                                    text="A receipt worth filing."))["id"]]}])
    assert [item["id"] for item in shared.lessons] == ["file"], \
        "practice from evidence that names nobody is nobody's private business"


def test_a_record_naming_a_stranger_is_scoped_by_confirmed_identity_alone(store, broker):
    identity = IdentityStore(store, owner_principal="owner")
    mine = identity.account("email", ME)
    broker.identity = identity
    store.commit(envelope(text="My note.", metadata={
        "participants": [{"namespace": "email", "address": STRANGER}]}))

    packet = broker.assemble("note", account_id=mine)

    # The address is not a registered account, so it scopes nothing: the record
    # is unscoped, and unscoped evidence is shared rather than quarantined.
    assert len(packet.items) == 1


def test_an_unknown_caller_sees_unscoped_evidence_only(store, broker):
    identity = IdentityStore(store, owner_principal="owner")
    mine = identity.account("email", ME)
    broker.identity = identity
    store.commit(envelope(source_id="private", text="My private appointment.",
                          metadata={"account_ids": [mine]}))
    store.commit(envelope(source_id="public", text="An appointment note."))

    packet = broker.assemble("appointment")

    assert [item.source_id for item in packet.items] == ["public"]
    assert packet.withheld == 1


# -- typed assertions ----------------------------------------------------------


def assertions(store):
    from hermes_memory.knowledge.assertions import AssertionStore

    return AssertionStore(store, owner_principal="owner")


def test_a_supported_assertion_goes_in_ahead_of_the_raw_spans(store, broker):
    record = store.commit(envelope(text="I take the coffee without sugar."))["id"]
    claims = assertions(store)
    claims.propose(subject="jugaadu", predicate="sweetens", value="nothing",
                   kind="preference", evidence_kind="owner_declared", record_id=record,
                   quote="without sugar", proposed_by="owner")
    broker.assertions = claims

    packet = broker.assemble("coffee sugar")

    assert [item["predicate"] for item in packet.assertions] == ["sweetens"]
    rendered = packet.render()
    assert "asserted preference: jugaadu sweetens = nothing" in rendered
    assert record in rendered, "the claim names the evidence it came from"
    assert rendered.index("asserted preference") < rendered.index("- [gmail @")


def test_an_unconfirmed_assertion_is_not_presented_as_knowledge(store, broker):
    record = store.commit(envelope(text="I take the coffee without sugar."))["id"]
    claims = assertions(store)
    claims.propose(subject="jugaadu", predicate="sweetens", value="nothing",
                   kind="preference", evidence_kind="observed_pattern", record_id=record,
                   quote="without sugar", proposed_by="agent-model")
    broker.assertions = claims

    assert broker.assemble("sweetens coffee").assertions == ()


def test_forgetting_the_evidence_under_a_claim_removes_it_from_the_packet(store, broker):
    record = store.commit(envelope(text="I take the coffee without sugar."))["id"]
    claims = assertions(store)
    claims.propose(subject="jugaadu", predicate="sweetens", value="nothing",
                   kind="preference", evidence_kind="owner_declared", record_id=record,
                   quote="without sugar", proposed_by="owner")
    broker.assertions = claims
    store.hide(record, reason="withdrawn", actor="owner")

    packet = broker.assemble("sweetens coffee")

    assert packet.assertions == ()
    assert packet.items == ()


def test_two_supported_claims_that_disagree_make_the_packet_conflicting(store, broker):
    record = store.commit(envelope(text="The standup is at 09:00 on Tuesday."))["id"]
    claims = assertions(store)
    for value, quote in (("Tuesday", "on Tuesday"), ("Wednesday", "at 09:00")):
        claims.propose(subject="standup", predicate="day", value=value, kind="fact",
                       evidence_kind="explicit_statement", record_id=record, quote=quote,
                       proposed_by="owner")
    broker.assertions = claims

    packet = broker.assemble("standup day")

    assert packet.coverage == "conflicting"
    assert packet.conflicts == ("standup day is disputed: Tuesday vs Wednesday",)
    assert "conflicting accounts" in packet.render()


def test_a_broken_knowledge_layer_costs_a_section_and_not_the_answer(store, broker,
                                                                      monkeypatch):
    from hermes_memory.knowledge.assertions import AssertionStore

    record = store.commit(envelope(text="I take the coffee without sugar."))["id"]
    claims = AssertionStore(store)
    claims.propose(subject="jugaadu", predicate="sweetens", value="nothing",
                   kind="preference", evidence_kind="owner_declared", record_id=record,
                   quote="without sugar", proposed_by="owner")
    broker.assertions = claims

    def explode(*_args, **_kwargs):
        raise RuntimeError("assertion table is unreadable")

    monkeypatch.setattr(claims, "matching", explode)
    packet = broker.assemble("coffee sugar")

    assert "assertions could not be read" in packet.channels.detail
    assert "assertions" in packet.truncated
    assert packet.conflicts == (), "a missing section is not a disagreement"
    assert packet.items, "the lexical half still answers"
    assert packet.coverage == "partial"


def test_assertions_are_charged_against_the_same_ceiling(store):
    from hermes_memory.context import ContextBroker

    record = store.commit(envelope(text="I take the coffee without sugar."))["id"]
    claims = assertions(store)
    for index in range(6):
        claims.propose(subject=f"person-{index}", predicate="sweetens", value="nothing",
                       kind="preference", evidence_kind="owner_declared", record_id=record,
                       quote="without sugar", proposed_by="owner")
    tight = ContextBroker(store, assertions=claims, cache=None, budget_tokens=40)

    packet = tight.assemble("sweetens")

    assert packet.tokens_used <= 40
    assert len(packet.assertions) < 6
    tight.close()


# -- correctness under change -------------------------------------------------


def test_evidence_revoked_during_recall_is_dropped_from_the_packet(store, tmp_path):
    committed = store.commit(envelope(text="The kickoff meeting is Monday."))
    path = tmp_path / "canonical.db"

    def revoke():
        with EvidenceStore(path) as other:
            other.hide(committed["id"], reason="owner revoked it", actor="owner")

    broker = ContextBroker(store, cache=None, client=FakeBackend(during=revoke))
    packet = broker.assemble("kickoff meeting")

    assert packet.items == ()
    assert "revoked_during_recall" in packet.truncated
    assert "forgotten or hidden during retrieval" in packet.channels.detail
    broker.close()


def test_a_stale_preview_of_a_moved_archive_is_never_cached(store, tmp_path):
    store.commit(envelope(text="The kickoff meeting is Monday."))
    path = tmp_path / "canonical.db"

    def change():
        with EvidenceStore(path) as other:
            other.commit(envelope(source_id="msg-late", text="A late correction."))

    broker = ContextBroker(store, client=FakeBackend(during=change))
    first = broker.assemble("kickoff meeting")
    assert "store_moved" in first.truncated
    assert len(broker.cache) == 0, "an answer built across a change cannot be reused"
    assert "not cached" in first.channels.detail

    second = broker.assemble("kickoff meeting")
    assert "store_moved" not in second.truncated
    assert len(broker.cache) == 1
    broker.close()


class FakeClock:
    """A clock the test advances, so nothing here waits on wall time."""

    def __init__(self, start=1000.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now


def test_a_down_backend_is_not_hammered_but_stays_reported_as_down(store):
    broker = ContextBroker(store, client=FakeBackend(error=ConnectionError("refused")))

    first, second = broker.assemble("anything"), broker.assemble("anything")

    assert broker.cache.as_dict()["hits"] == 1
    assert second.channels.derived == "unavailable"
    assert first.coverage == second.coverage == "partial"
    broker.close()


def test_an_expired_entry_asks_the_backend_again(store):
    clock = FakeClock()
    backend = FakeBackend(results=[{"text": "a derived statement"}])
    broker = ContextBroker(store, client=backend,
                           cache=PacketCache(store, clock=clock, ttl_s=5), clock=clock)

    broker.assemble("anything")
    broker.assemble("anything")
    clock.now += 6
    broker.assemble("anything")

    assert len(backend.calls) == 2, "a short ttl is what makes recovery observable"
    assert broker.cache.as_dict() == {"entries": 1, "max_entries": 64, "ttl_s": 5.0,
                                      "hits": 1, "misses": 2, "stale": 1}
    broker.close()


def test_a_repeat_question_is_served_from_cache_until_something_changes(store):
    store.commit(envelope(text="The kickoff meeting is Monday."))
    broker = ContextBroker(store)

    first, second = broker.assemble("kickoff"), broker.assemble("kickoff")

    assert first.packet_id == second.packet_id
    assert broker.cache.as_dict()["hits"] == 1

    hidden = store.search("kickoff")[0].id
    store.hide(hidden, reason="correction", actor="owner")
    third = broker.assemble("kickoff")

    assert third.items == ()
    assert broker.cache.as_dict()["stale"] == 1, "a hide invalidates without being told to"
    broker.close()


def test_an_epoch_bump_invalidates_every_cached_packet(store):
    store.commit(envelope(text="The kickoff meeting is Monday."))
    broker = ContextBroker(store)
    broker.assemble("kickoff")

    store.bump_epoch(reason="operator distrusts the archive", actor="owner")
    packet = broker.assemble("kickoff")

    assert broker.cache.as_dict()["hits"] == 0
    assert packet.epoch == 2
    broker.close()


def test_the_cache_never_answers_a_different_question_or_caller(store):
    identity = IdentityStore(store, owner_principal="owner")
    mine, theirs = identity.account("email", ME), identity.account("email", STRANGER)
    store.commit(envelope(source_id="mine", text="kickoff meeting",
                          metadata={"account_ids": [mine]}))
    broker = ContextBroker(store, identity=identity)

    mine_packet = broker.assemble("kickoff", account_id=mine)
    theirs_packet = broker.assemble("kickoff", account_id=theirs)

    assert mine_packet.items and theirs_packet.items == ()
    assert broker.cache.as_dict()["misses"] == 2
    broker.close()


def test_a_cache_rejects_an_impossible_shape():
    with pytest.raises(ValueError, match="max_entries"):
        PacketCache(None, max_entries=0)
    with pytest.raises(ValueError, match="ttl_s"):
        PacketCache(None, ttl_s=0)


# -- conflict and coverage ----------------------------------------------------


def test_contradictory_dates_are_surfaced_and_nobody_picks_a_winner(store, broker):
    store.commit(envelope(source_id="a", text="Kickoff happened Monday.",
                          occurred_at="2026-03-02T09:00:00+00:00",
                          metadata={"subject": "kickoff-2026"}))
    store.commit(envelope(source_id="b", text="Kickoff happened Wednesday.",
                          occurred_at="2026-03-04T09:00:00+00:00",
                          metadata={"subject": "kickoff-2026"}))

    packet = broker.assemble("kickoff")

    assert packet.coverage == "conflicting"
    assert len(packet.items) == 2, "both accounts stay visible; we do not silently choose"
    assert any("occurred_at is disputed" in text for text in packet.conflicts)
    assert "conflicting accounts" in packet.render()


def test_declared_claims_that_disagree_are_conflicts(store, broker):
    for index, venue in enumerate(("Room A", "Room B")):
        store.commit(envelope(source_id=f"claim-{index}",
                              text=f"The venue is {venue}.",
                              metadata={"subject": "kickoff-2026",
                                        "claims": {"venue": venue}}))

    packet = broker.assemble("venue")

    assert packet.conflicts == ("kickoff-2026 venue is disputed: Room A vs Room B",)


def test_an_evolving_file_is_not_a_contradiction(store, broker):
    for index in range(3):
        store.commit(envelope(source="filesystem", source_id="notes.md",
                              revision=str(index + 1), kind="file",
                              occurred_at=f"2026-09-2{index}T09:00:00+00:00",
                              text=f"Meeting notes version {index}."))

    packet = broker.assemble("meeting notes")

    # Same source_id, three different timestamps and no declared subject: that is
    # a document changing over time, and calling it a contradiction would bury
    # every real one under a wall of noise.
    assert packet.conflicts == ()
    assert packet.coverage == "supported"


def test_a_window_filter_is_not_reported_as_degradation(store, broker):
    store.commit(envelope(source_id="old", text="Old kickoff.",
                          occurred_at="2020-01-01T09:00:00+00:00"))
    store.commit(envelope(source_id="new", text="New kickoff.",
                          occurred_at="2026-09-01T09:00:00+00:00"))
    store.commit(envelope(source_id="undated", text="Undated kickoff.",
                          occurred_at=None, occurred_precision="unknown"))

    packet = broker.assemble("kickoff",
                             window=("2026-01-01T00:00:00+00:00", "2027-01-01T00:00:00+00:00"))

    assert [item.source_id for item in packet.items] == ["new", "undated"]
    assert packet.coverage == "supported"


def test_a_cross_source_question_answers_from_the_sources_it_has(store, broker):
    store.commit(envelope(source="gmail", source_id="g1", text="Kickoff on Monday."))
    store.commit(envelope(source="whatsapp", source_id="w1", text="Kickoff on Tuesday."))

    both = broker.assemble("kickoff")
    chat = broker.assemble("kickoff", sources=["whatsapp"])

    assert {item.source for item in both.items} == {"gmail", "whatsapp"}
    assert [item.source for item in chat.items] == ["whatsapp"]
    assert chat.coverage == "supported" and chat.conflicts == ()


def test_an_empty_query_is_a_caller_error_not_an_empty_packet(broker):
    with pytest.raises(ValueError, match="must not be empty"):
        broker.assemble("   ")


def test_impossible_budgets_are_refused_at_construction(store):
    with pytest.raises(ValueError, match="budget_tokens"):
        ContextBroker(store, budget_tokens=0)
    with pytest.raises(ValueError, match="derived_timeout_s"):
        ContextBroker(store, derived_timeout_s=0)


def test_the_packet_carries_a_revision_and_a_content_scoped_id(store, broker):
    committed = store.commit(envelope(text="The kickoff meeting is Monday."))

    packet = broker.assemble("kickoff")
    same = broker.assemble("kickoff", include_derived=False)

    assert packet.epoch == 1 and packet.revision == 1
    assert packet.packet_id.startswith("ctx_") and packet.packet_id == same.packet_id

    store.hide(committed["id"], reason="correction", actor="owner")
    after = broker.assemble("kickoff", include_derived=False)
    assert after.packet_id != packet.packet_id


def test_the_packet_asks_the_backend_for_no_more_than_it_can_send(store, broker):
    backend = FakeBackend()
    broker.client = backend
    broker.budget_tokens = 500

    broker.assemble("anything")

    query, kwargs = backend.calls[0]
    assert query == "anything"
    assert kwargs["max_tokens"] == 500
    assert kwargs["types"] is None, "a type filter here would hide whole classes of memory"


# -- the cache key covers what the caller brought ------------------------------

def test_a_packet_warmed_for_one_task_is_not_sold_to_another(store):
    """The caller's own material changes the answer, so it belongs in the key.

    Evidence, filters and the account were all in the scope; the practice and the
    obligations handed in by the caller were not, which let one task's lesson be
    rendered into another task's turn — a wrong answer with a cache hit behind it.
    """
    subject = ContextBroker(store)
    first = subject.assemble("kickoff", lessons=[{"id": "greet", "version": 1,
                                                 "text": "Greet by name."}])
    second = subject.assemble("kickoff", lessons=[{"id": "greet", "version": 2,
                                                  "text": "Greet by name and company."}])
    assert [item["text"] for item in first.lessons] == ["Greet by name."]
    assert [item["text"] for item in second.lessons] == ["Greet by name and company."]
    again = subject.assemble("kickoff", lessons=[{"id": "greet", "version": 1,
                                                "text": "Greet by name."}])
    assert [item["text"] for item in again.lessons] == ["Greet by name."], \
        "the same inputs still earn the cache hit"


def test_a_different_obligation_list_is_a_different_packet_too(store):
    subject = ContextBroker(store)
    one = subject.assemble("kickoff", commitments=[{"title": "Renew the TLS certificate",
                                                   "due_at": "2026-09-25T09:00:00+00:00"}])
    two = subject.assemble("kickoff", commitments=[{"title": "File the expense report",
                                                   "due_at": "2026-09-26T09:00:00+00:00"}])
    assert [item["title"] for item in one.commitments] == ["Renew the TLS certificate"]
    assert [item["title"] for item in two.commitments] == ["File the expense report"]
    assert len(subject.cache) == 2, "two different asks are two entries, not one collision"
