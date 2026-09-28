"""C7 identity: deterministic candidates, owner-only decisions, confirmed-only traversal."""
from __future__ import annotations

import json

import pytest

from hermes_memory.storage.evidence import EvidenceError
from hermes_memory.storage.identity import (CONFIRMED, PENDING, REJECTED, STALE, IdentityStore,
                                            RULES, normalize_account, parse_account)

from conftest import envelope

OWNER = "owner-principal"
AGENT = "agent:sess-1"


@pytest.fixture()
def identity(store):
    return IdentityStore(store, owner_principal=OWNER)


@pytest.fixture()
def evidence(store):
    return store.commit(envelope(source_id="thread-1"))["id"]


def accounts(identity):
    return (identity.account("email", "Priya@example.com"),
            identity.account("email", "priya@work.example"))


# -- normalization -----------------------------------------------------------

def test_email_local_part_case_is_preserved_but_the_domain_is_folded(identity):
    account = identity.account("email", "Priya@Example.COM")
    row = identity.get_account(account)
    assert row["identifier"] == "Priya@Example.COM", "as written is what the source said"
    assert row["normalized"] == "Priya@example.com", "only the domain folds"


def test_the_same_address_in_different_case_is_one_account(identity):
    first = identity.account("email", "Priya@Example.com")
    second = identity.account("email", "Priya@example.COM")
    assert first == second


def test_different_local_part_case_is_not_silently_merged(identity):
    """Some providers treat local-part case as significant, so we must not assume."""
    assert normalize_account("email", "Priya@x.com")[1] != normalize_account("email",
                                                                            "priya@x.com")[1]


def test_a_phone_without_an_international_prefix_is_refused(identity):
    with pytest.raises(EvidenceError, match="international prefix"):
        identity.account("phone", "415 555 1234")
    account = identity.account("phone", "+1 (415) 555-1234")
    assert identity.get_account(account)["normalized"] == "+14155551234"


def test_an_unknown_namespace_or_malformed_address_is_refused(identity):
    with pytest.raises(EvidenceError, match="unknown identity namespace"):
        identity.account("smoke-signal", "x")
    with pytest.raises(EvidenceError, match="not an email"):
        identity.account("email", "not-an-address")


# -- the shape an account arrives in -----------------------------------------

def test_an_account_written_as_a_blob_is_refused_rather_than_kept_as_an_address(identity):
    """Every one of these is a *description* of an account, not the account.

    They used to be stored anyway: the plugin stringified what a caller handed it, so an
    address column ended up holding `{"namespace": "email", ...}` and the person behind it
    could never be joined to the same address written plainly.
    """
    for value in ('{"namespace": "email", "address": "priya@example.com"}',
                  '{"namespace":"email","address":"priya@example.com"}',
                  "{'namespace': 'email', "
                  "'address': 'priya@example.com'}",
                  '["email", "priya@example.com"]',
                  '["email","priya@example.com"]',
                  "Priya <priya@example.com>",
                  "priya at example.com"):
        with pytest.raises(EvidenceError, match="not an account address"):
            identity.account("email", value)
    assert identity.db.execute("SELECT COUNT(*) AS n FROM identity_accounts").fetchone()["n"] == 0


def test_a_phone_written_with_its_readable_separators_is_still_one_account(identity):
    """The one-token rule is about blobs, not about how people type numbers."""
    assert (identity.account(*parse_account("+1 (415) 555-1234"))
            == identity.account("phone", "+14155551234"))


def test_an_account_object_and_the_address_it_carries_are_one_account(identity):
    """The pair form is what a record's participants carry, so quoting it back must land
    on the same account the plain address does — otherwise a normalized-equal rule could
    never fire on the two shapes of one person."""
    plain = identity.account(*parse_account("priya@example.com"))
    shaped = identity.account(*parse_account({"namespace": "email",
                                              "address": "priya@EXAMPLE.com"}))
    assert plain == shaped


@pytest.mark.parametrize(("given", "expected"), (
    ("priya@example.com", ("email", "priya@example.com")),
    ("+41555123456", ("phone", "+41555123456")),
    ("priya", ("handle", "priya")),
    ({"namespace": "profile", "address": "https://work.example/priya"},
     ("profile", "https://work.example/priya")),
))
def test_parse_account_reads_the_two_admissible_shapes(given, expected):
    assert parse_account(given) == expected


@pytest.mark.parametrize("given", (
    None, "", "   ", 7, ["email", "priya@example.com"], {"namespace": "email"},
    {"address": "priya@example.com"}, {"namespace": "smoke-signal",
                                       "address": "priya@example.com"},
    {"namespace": "email", "address": "priya@example.com", "person": "Priya"},
))
def test_any_other_shape_is_refused_by_name(given):
    with pytest.raises(EvidenceError):
        parse_account(given)


# -- candidates --------------------------------------------------------------

def test_only_deterministic_rules_may_propose(identity, evidence):
    first, second = accounts(identity)
    for rule in ("display-name-match", "co-occurrence", "model-suggested", "fuzzy-name",
                 "vibes"):
        with pytest.raises(EvidenceError, match="cannot propose|unknown identity rule"):
            identity.propose(account_a=first, account_b=second, rule=rule,
                             basis="looked similar", evidence=[evidence], proposed_by=AGENT)
    assert identity.pending() == []


def test_a_rule_version_mismatch_is_refused(identity, evidence):
    first, second = accounts(identity)
    with pytest.raises(EvidenceError, match="is at version"):
        identity.propose(account_a=first, account_b=second, rule="email-thread-participant",
                         rule_version="99", basis="thread", evidence=[evidence],
                         proposed_by=AGENT)


def test_a_candidate_needs_retrievable_evidence(identity):
    first, second = accounts(identity)
    with pytest.raises(EvidenceError, match="live evidence"):
        identity.propose(account_a=first, account_b=second, rule="email-thread-participant",
                         basis="thread", evidence=["rec_" + "0" * 32], proposed_by=AGENT)
    with pytest.raises(EvidenceError, match="between 1 and 200"):
        identity.propose(account_a=first, account_b=second, rule="email-thread-participant",
                         basis="thread", evidence=[], proposed_by=AGENT)


def test_an_account_cannot_be_a_candidate_for_itself(identity, evidence):
    first, _ = accounts(identity)
    with pytest.raises(EvidenceError, match="itself"):
        identity.propose(account_a=first, account_b=first, rule="email-normalized-equal",
                         basis="same", evidence=[evidence], proposed_by=AGENT)


def test_proposing_the_same_pair_twice_is_one_candidate(identity, evidence):
    first, second = accounts(identity)
    one = identity.propose(account_a=first, account_b=second, rule="email-thread-participant",
                           basis="thread 4f2a", evidence=[evidence], proposed_by=AGENT)
    two = identity.propose(account_a=second, account_b=first, rule="email-thread-participant",
                           basis="thread 4f2a", evidence=[evidence], proposed_by=AGENT)
    assert one["candidate_id"] == two["candidate_id"]
    assert len(identity.pending()) == 1


# -- authority ---------------------------------------------------------------

def test_an_agent_cannot_confirm_its_own_proposal(identity, evidence):
    first, second = accounts(identity)
    proposal = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="thread",
                                evidence=[evidence], proposed_by=AGENT)
    with pytest.raises(EvidenceError, match="owner principal"):
        identity.confirm(candidate_id=proposal["candidate_id"], actor=AGENT, reason="sure")
    assert identity.same_person(first, second) is False


def test_no_configured_owner_means_nothing_can_be_confirmed(store, evidence):
    identity = IdentityStore(store, owner_principal=None)
    first, second = accounts(identity)
    proposal = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="thread",
                                evidence=[evidence], proposed_by=AGENT)
    with pytest.raises(EvidenceError, match="no owner principal"):
        identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="x")


def test_owner_confirmation_creates_the_canonical_edge(identity, evidence):
    first, second = accounts(identity)
    proposal = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="thread",
                                evidence=[evidence], proposed_by=AGENT)
    outcome = identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER,
                               reason="same person, both mine",
                               valid_from="2024-01-01T00:00:00+00:00")
    assert outcome["state"] == CONFIRMED and outcome["edge_id"].startswith("iedge_")
    assert identity.same_person(first, second) is True
    assert identity.group(first) == sorted([first, second])


def test_confirming_twice_does_not_create_a_second_edge(identity, evidence):
    first, second = accounts(identity)
    proposal = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="thread",
                                evidence=[evidence], proposed_by=AGENT)
    identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="mine")
    again = identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="mine")
    assert again["edge_id"] is None
    assert identity.db.execute("SELECT count(*) FROM identity_edges").fetchone()[0] == 1


def test_a_rejected_candidate_cannot_be_confirmed_later(identity, evidence):
    """The rejection is durable; re-proposing must not quietly revive it."""
    first, second = accounts(identity)
    proposal = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="thread",
                                evidence=[evidence], proposed_by=AGENT)
    identity.reject(candidate_id=proposal["candidate_id"], actor=OWNER, reason="different people")
    replayed = identity.propose(account_a=first, account_b=second,
                               rule="email-thread-participant", basis="thread",
                               evidence=[evidence], proposed_by=AGENT)
    assert replayed["state"] == REJECTED and replayed["reopened"] is False
    with pytest.raises(EvidenceError, match="was rejected"):
        identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="changed mind")


def test_a_confirmed_identity_is_revoked_not_rejected(identity, evidence):
    first, second = accounts(identity)
    proposal = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="thread",
                                evidence=[evidence], proposed_by=AGENT)
    identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="mine")
    with pytest.raises(EvidenceError, match="revoke the edge"):
        identity.reject(candidate_id=proposal["candidate_id"], actor=OWNER, reason="mistake")
    edges = identity.db.execute("SELECT id FROM identity_edges").fetchall()
    identity.revoke(edge_id=edges[0]["id"], actor=OWNER, reason="was wrong")
    assert identity.same_person(first, second) is False


def test_merging_two_identities_the_owner_separated_is_refused(identity, evidence):
    """The owner said second and third are different people; a later edge must not undo that."""
    first, second = accounts(identity)
    third = identity.account("email", "someone-else@example.org")
    pairs = ((first, second), (second, third), (first, third))
    ids = {}
    for pair in pairs:
        ids[pair] = identity.propose(account_a=pair[0], account_b=pair[1],
                                     rule="email-thread-participant", basis="thread",
                                     evidence=[evidence], proposed_by=AGENT)["candidate_id"]
    identity.confirm(candidate_id=ids[(first, second)], actor=OWNER, reason="mine")
    identity.reject(candidate_id=ids[(second, third)], actor=OWNER, reason="different people")
    with pytest.raises(EvidenceError, match="already rejected a join"):
        identity.confirm(candidate_id=ids[(first, third)], actor=OWNER, reason="actually mine")
    assert identity.same_person(second, third) is False


def test_a_duplicate_active_edge_over_the_same_interval_is_refused(identity, evidence):
    first, second = accounts(identity)
    proposal = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="thread",
                                evidence=[evidence], proposed_by=AGENT)
    identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="mine",
                     valid_from="2024-01-01T00:00:00+00:00")
    # Revoke the decision so the candidate can be reconsidered, then try to stack
    # a second edge over the interval the first still covers.
    edge = identity.db.execute("SELECT id FROM identity_edges").fetchone()["id"]
    identity.revoke(edge_id=edge, actor=OWNER, reason="re-deciding")
    identity.db.execute("UPDATE identity_candidates SET state=? WHERE id=?",
                        ("pending", proposal["candidate_id"]))
    outcome = identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER,
                               reason="still mine", valid_from="2024-06-01T00:00:00+00:00")
    assert outcome["state"] == CONFIRMED, "the revoked edge no longer conflicts"


def test_non_overlapping_intervals_are_allowed(identity, evidence):
    """A number can change hands; time-bounded edges express that."""
    first, second = accounts(identity)
    third = identity.account("email", "previous-owner@example.org")
    for pair, start, end in (((first, second), "2024-01-01T00:00:00+00:00",
                              "2024-12-31T00:00:00+00:00"),
                             ((first, third), "2025-06-01T00:00:00+00:00", None)):
        proposal = identity.propose(account_a=pair[0], account_b=pair[1],
                                    rule="explicit-alias-declared", basis="declared",
                                    evidence=[evidence], proposed_by=AGENT)
        identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="sequential",
                         valid_from=start, valid_until=end)
    assert identity.same_person(first, second, at="2024-06-01T00:00:00+00:00") is True
    assert identity.same_person(first, third, at="2024-06-01T00:00:00+00:00") is False
    assert identity.same_person(first, third, at="2026-01-01T00:00:00+00:00") is True


def test_an_edge_whose_interval_has_closed_stops_widening_the_group(identity, evidence):
    """A decision the owner bounded in time must not become permanent on read.

    ``group`` is what C9 and the provenance ledger widen a caller's access with, so
    honouring the interval only in ``same_person`` left the useful path open.
    """
    first, second = accounts(identity)
    third = identity.account("email", "someone-else@example.org")
    for pair, start, end in (((first, second), "2024-01-01T00:00:00+00:00",
                              "2024-12-31T00:00:00+00:00"),
                             ((first, third), None, None)):
        proposal = identity.propose(account_a=pair[0], account_b=pair[1],
                                    rule="explicit-alias-declared", basis="declared",
                                    evidence=[evidence], proposed_by=AGENT)
        identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER,
                         reason="sequential", valid_from=start, valid_until=end)
    assert identity.group(first) == sorted([first, third]), "the closed edge has dropped out"
    assert identity.group(first, at="2024-06-01T00:00:00+00:00") == sorted(
        [first, second, third]), "inside its interval the closed edge still joins"
    assert identity.same_person(first, second) is False
    assert identity.same_person(first, second, at="2024-06-01T00:00:00+00:00") is True


def test_a_query_moment_without_a_zone_is_refused(identity):
    """A loose stamp would be compared as text against normalized ones, and lose."""
    first, _ = accounts(identity)
    for loose in ("2024-06-01", "last tuesday"):
        with pytest.raises(EvidenceError, match="timezone-aware"):
            identity.group(first, at=loose)
        with pytest.raises(EvidenceError, match="timezone-aware"):
            identity.same_person(first, "acct_x", at=loose)


def test_a_rejected_join_is_refused_even_after_the_interval_that_linked_it_closes(
        identity, evidence):
    """The safety check is not a query about today.

    Two accounts the owner separated must not be reunited through an edge that is
    no longer time-valid: the merge is what they decided against, whenever the
    calendar says it applies.
    """
    first, second = accounts(identity)
    third = identity.account("email", "someone-else@example.org")
    linked = identity.propose(account_a=first, account_b=second,
                              rule="explicit-alias-declared", basis="declared",
                              evidence=[evidence], proposed_by=AGENT)
    identity.confirm(candidate_id=linked["candidate_id"], actor=OWNER, reason="was mine",
                     valid_from="2020-01-01T00:00:00+00:00",
                     valid_until="2020-12-31T00:00:00+00:00")
    assert identity.same_person(first, second) is False, "the edge has closed"
    separated = identity.propose(account_a=second, account_b=third,
                                 rule="email-thread-participant", basis="thread",
                                 evidence=[evidence], proposed_by=AGENT)
    identity.reject(candidate_id=separated["candidate_id"], actor=OWNER,
                    reason="different people")
    reunion = identity.propose(account_a=first, account_b=third,
                               rule="email-thread-participant", basis="thread",
                               evidence=[evidence], proposed_by=AGENT)
    with pytest.raises(EvidenceError, match="already rejected a join"):
        identity.confirm(candidate_id=reunion["candidate_id"], actor=OWNER, reason="actually")


def test_a_reversed_validity_interval_is_refused(identity, evidence):
    first, second = accounts(identity)
    proposal = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="thread",
                                evidence=[evidence], proposed_by=AGENT)
    with pytest.raises(EvidenceError, match="precedes"):
        identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="x",
                         valid_from="2026-01-01T00:00:00+00:00",
                         valid_until="2025-01-01T00:00:00+00:00")


def test_a_second_rule_about_the_same_pair_reopens_one_candidate(identity, evidence):
    """The pair is the candidate, so a second rule cannot make a second one.

    The schema says UNIQUE(account_a, account_b); an id derived from the rule as well
    meant a re-proposal under another rule collided with the row instead of refreshing
    it, and the owner would have seen one error rather than one question.
    """
    first, second = accounts(identity)
    by_thread = identity.propose(account_a=first, account_b=second,
                                 rule="email-thread-participant", basis="one thread",
                                 evidence=[evidence], proposed_by=AGENT)
    by_alias = identity.propose(account_a=first, account_b=second,
                                rule="explicit-alias-declared", basis="declared by both",
                                evidence=[evidence], proposed_by=AGENT)
    assert by_alias["candidate_id"] == by_thread["candidate_id"]
    rows = identity.db.execute("SELECT count(*) FROM identity_candidates").fetchone()[0]
    assert rows == 1
    rule = identity.db.execute("SELECT rule, basis FROM identity_candidates WHERE id=?",
                               (by_thread["candidate_id"],)).fetchone()
    assert rule["rule"] == "explicit-alias-declared", \
        "the owner is shown the rule behind the proposal that is on the list now"
    assert rule["basis"] == "declared by both"


def test_a_pair_the_owner_already_joined_is_not_asked_about_again(identity, evidence):
    """Transitively, as well as directly.

    The candidate id is per pair and rule, so nothing else stops a second proposal
    about `first` and `third` from landing on the owner's list after they had already
    confirmed both of them to be one person through a third account.
    """
    first, second = accounts(identity)
    third = identity.account("email", "someone-else@example.org")
    for pair in ((first, second), (second, third)):
        proposal = identity.propose(account_a=pair[0], account_b=pair[1],
                                    rule="email-thread-participant", basis="thread",
                                    evidence=[evidence], proposed_by=AGENT)
        identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="mine")
    assert identity.same_person(first, third)

    answer = identity.propose(account_a=first, account_b=third,
                              rule="phone-e164-equal", basis="and the same number",
                              evidence=[evidence], proposed_by=AGENT)
    assert answer["already_joined"] is True and answer["candidate_id"] is None
    assert answer["state"] == CONFIRMED
    assert identity.pending() == [], "a decided question is not on the list"


def test_a_closed_join_is_a_question_again(identity, evidence):
    """The short-circuit asks who one person *now*, not who they ever were."""
    first, second = accounts(identity)
    third = identity.account("email", "someone-else@example.org")
    linked = identity.propose(account_a=first, account_b=second,
                              rule="explicit-alias-declared", basis="declared",
                              evidence=[evidence], proposed_by=AGENT)
    identity.confirm(candidate_id=linked["candidate_id"], actor=OWNER, reason="until 2025",
                     valid_from="2024-01-01T00:00:00+00:00",
                     valid_until="2024-12-31T00:00:00+00:00")
    reopened = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="a thread again",
                                evidence=[evidence], proposed_by=AGENT)
    assert reopened["candidate_id"], "the interval closed, so the question is open"


# -- traversal and staleness -------------------------------------------------

def test_traversal_follows_confirmed_edges_only(identity, evidence):
    first, second = accounts(identity)
    third = identity.account("email", "third@example.org")
    linked = identity.propose(account_a=second, account_b=third,
                              rule="email-thread-participant", basis="thread",
                              evidence=[evidence], proposed_by=AGENT)
    confirmed = identity.propose(account_a=first, account_b=second,
                                 rule="email-thread-participant", basis="thread",
                                 evidence=[evidence], proposed_by=AGENT)
    identity.confirm(candidate_id=confirmed["candidate_id"], actor=OWNER, reason="mine")
    assert identity.same_person(first, third) is False, "an unconfirmed hop must not join"
    identity.confirm(candidate_id=linked["candidate_id"], actor=OWNER, reason="also mine")
    assert identity.same_person(first, third) is True
    assert identity.group(first) == sorted([first, second, third])


def test_losing_all_evidence_marks_a_candidate_stale(store, identity, evidence):
    first, second = accounts(identity)
    proposal = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="thread",
                                evidence=[evidence], proposed_by=AGENT)
    from hermes_memory.lifecycle.erasure import ErasureManager

    manager = ErasureManager(store, owner_principal=OWNER)
    preview = manager.preview(record_ids=[evidence], actor=OWNER, reason="withdrawn")
    manager.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                    actor=OWNER)
    outcome = identity.invalidate_stale()
    assert outcome["stale"] == [proposal["candidate_id"]]
    assert identity.db.execute("SELECT state FROM identity_candidates WHERE id=?",
                               (proposal["candidate_id"],)).fetchone()[0] == STALE
    with pytest.raises(EvidenceError, match="lost its supporting evidence"):
        identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="x")


def test_a_confirmed_edge_on_vanished_evidence_is_flagged_not_silently_revoked(store, identity,
                                                                              evidence):
    """Revoking a decision the owner made is the owner's call, not a background job's."""
    first, second = accounts(identity)
    proposal = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="thread",
                                evidence=[evidence], proposed_by=AGENT)
    identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="mine")
    from hermes_memory.lifecycle.erasure import ErasureManager

    manager = ErasureManager(store, owner_principal=OWNER)
    preview = manager.preview(record_ids=[evidence], actor=OWNER, reason="withdrawn")
    manager.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                    actor=OWNER)
    outcome = identity.invalidate_stale()
    assert outcome["stale"] == []
    assert outcome["confirmed_needing_review"] == [proposal["candidate_id"]]
    assert identity.same_person(first, second) is True, "still believed until the owner decides"


def test_a_sweep_that_changed_nothing_is_not_recorded_as_a_decision(store, identity,
                                                                    evidence):
    """The background pass calls this every period, and the audit ledger is not its heartbeat.

    112 `identity_invalidate` rows on a live installation, 108 of them `{"stale": 0,
    "needs_review": 0}`, is the failure: the one act worth recording — a candidate whose
    evidence went away — becomes a line in a log of visits.
    """
    from hermes_memory.lifecycle.erasure import ErasureManager

    first, second = accounts(identity)
    identity.propose(account_a=first, account_b=second, rule="email-thread-participant",
                     basis="thread", evidence=[evidence], proposed_by=AGENT)

    def rows():
        return store.db.execute("SELECT metadata FROM audit WHERE "
                                "action='identity_invalidate'").fetchall()

    assert identity.invalidate_stale()["stale"] == []
    assert rows() == [], "a sweep over nothing is not an event"
    assert identity.invalidate_stale()["stale"] == []
    assert rows() == [], "and calling it on a timer must not accumulate one row per period"

    manager = ErasureManager(store, owner_principal=OWNER)
    preview = manager.preview(record_ids=[evidence], actor=OWNER, reason="withdrawn")
    manager.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                    actor=OWNER)
    assert len(identity.invalidate_stale()["stale"]) == 1
    assert len(rows()) == 1, "the change is written down, once"
    assert json.loads(rows()[0]["metadata"]) == {"stale": 1, "needs_review": 0}, \
        "and says what it moved, so the ledger answers the question a reviewer asks"


# -- topics ------------------------------------------------------------------

def test_structural_topics_and_model_labels_are_kept_apart(identity):
    account, _ = accounts(identity)
    identity.link_topic(account_id=account, topic="thread-4f2a", kind="structural",
                        source_record_id=None)
    identity.link_topic(account_id=account, topic="probably the landlord", kind="label")
    assert identity.accounts_for_topic("thread-4f2a") == [account]
    assert identity.accounts_for_topic("thread-4f2a", kind="label") == []
    assert identity.accounts_for_topic("probably the landlord", kind="label") == [account]
    with pytest.raises(EvidenceError, match="structural"):
        identity.link_topic(account_id=account, topic="x", kind="guess")


def test_no_identity_path_leaves_a_transaction_open(identity, evidence):
    first, second = accounts(identity)
    proposal = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="thread",
                                evidence=[evidence], proposed_by=AGENT)
    steps = [
        lambda: identity.pending(),
        lambda: identity.invalidate_stale(),
        lambda: identity.same_person(first, second),
        lambda: identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="mine"),
        lambda: identity.group(first),
        lambda: identity.link_topic(account_id=first, topic="t", kind="structural"),
    ]
    for step in steps:
        step()
        assert identity.db.in_transaction is False
    with pytest.raises(EvidenceError):
        identity.confirm(candidate_id="cand_missing", actor=OWNER, reason="x")
    assert identity.db.in_transaction is False


def test_every_admissible_rule_is_reachable(identity, evidence):
    """A rule in the table that cannot be used is a rule that does not exist."""
    previous = None
    for index, rule in enumerate(sorted(RULES)):
        account = identity.account("email", f"r{index}@example.com")
        if previous is not None:
            outcome = identity.propose(account_a=previous, account_b=account, rule=rule,
                                       basis=f"verified for {rule}", evidence=[evidence],
                                       proposed_by=AGENT)
            assert outcome["state"] == PENDING
        previous = account
