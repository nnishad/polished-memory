"""C7 identity: deterministic candidates, owner-only decisions, confirmed-only traversal."""
from __future__ import annotations

import pytest

from hermes_memory.storage.evidence import EvidenceError
from hermes_memory.storage.identity import (CONFIRMED, PENDING, REJECTED, STALE, IdentityStore,
                                            RULES, normalize_account)

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


def test_a_reversed_validity_interval_is_refused(identity, evidence):
    first, second = accounts(identity)
    proposal = identity.propose(account_a=first, account_b=second,
                                rule="email-thread-participant", basis="thread",
                                evidence=[evidence], proposed_by=AGENT)
    with pytest.raises(EvidenceError, match="precedes"):
        identity.confirm(candidate_id=proposal["candidate_id"], actor=OWNER, reason="x",
                         valid_from="2026-01-01T00:00:00+00:00",
                         valid_until="2025-01-01T00:00:00+00:00")


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
