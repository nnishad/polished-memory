"""C6 provenance: derived artifacts are only as good as the evidence behind them."""
from __future__ import annotations

import pytest

from conftest import envelope
from hermes_memory.backend.provenance import (COMPLETE, INVALID, PARTIAL, UNRESOLVED,
                                             ProvenanceLedger)
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.storage.evidence import EvidenceError
from hermes_memory.storage.identity import IdentityStore
from hermes_memory.storage.lineage import Lineage

OWNER = "owner-principal"
SUMMARY = "A weekly wrap of the incident channel."


@pytest.fixture()
def ledger(store):
    return ProvenanceLedger(store)


@pytest.fixture()
def two(store):
    first = store.commit(envelope(source_id="msg-1", text="The cluster failed at 04:12."))["id"]
    second = store.commit(envelope(source_id="msg-2", kind="message", revision="1",
                                   text="The recovery completed at 05:00."))["id"]
    return [first, second]


def declare(ledger, citations, **kwargs):
    return ledger.declare("sum_incident_week", kind="summary", citations=citations, **kwargs)


# -- declaration ---------------------------------------------------------------


def test_a_manifest_names_the_records_and_the_spans(store, ledger, two):
    report = declare(ledger, [{"record_id": two[0], "quote": "failed at 04:12"},
                              {"record_id": two[1]}])

    assert report == {"artifact": "sum_incident_week", "citations": 2, "coverage": "full"}
    assert ledger.resolve("sum_incident_week").verdict == COMPLETE


def test_declaring_again_replaces_the_manifest_instead_of_growing_it(store, ledger, two):
    declare(ledger, [{"record_id": one} for one in two])
    second = declare(ledger, [{"record_id": two[0]}])

    assert second["citations"] == 1
    assert [item["citations"] for item in ledger.artifacts()] == [1]
    assert Lineage(store).artifacts([two[1]]) == [], \
        "the old dependency is no longer claimed"


def test_a_quote_that_could_belong_to_two_places_cannot_be_cited_vaguely(store, ledger):
    record = store.commit(envelope(source_id="msg-repeat",
                                   text="Retry succeeded. Retry succeeded again."))["id"]

    with pytest.raises(EvidenceError, match="appears 2 times"):
        declare(ledger, [{"record_id": record, "quote": "Retry succeeded"}])


def test_a_backend_document_id_is_not_a_citation(store, ledger, two):
    with pytest.raises(EvidenceError, match="canonical record id"):
        declare(ledger, [{"record_id": "hdoc0123456789abcdef0123456789abcdef"}])


def test_an_artifact_cannot_over_claim_its_support(store, ledger):
    from hermes_memory.backend.provenance import MAX_CITATIONS

    citations = [{"record_id": store.commit(envelope(
        source_id=f"bulk-{index}", text=f"Note number {index}."))["id"]}
        for index in range(MAX_CITATIONS + 1)]

    with pytest.raises(EvidenceError, match="ceiling"):
        declare(ledger, citations)


def test_a_manifest_cannot_be_written_outside_a_transaction_it_was_lent(store, ledger, two):
    # Passing a connection is a promise to share its transaction: writing on a
    # bare connection would commit the citations before the artifact that owns
    # them exists, which is exactly the orphan this arrangement avoids.
    with pytest.raises(EvidenceError, match="ambient transaction"):
        ledger.declare("sum_orphan", kind="summary",
                       citations=[{"record_id": two[0]}], db=store.db)

    assert ledger.artifacts() == []


# -- verdicts ------------------------------------------------------------------


def test_an_undeclared_artifact_is_unresolved_rather_than_supported(ledger):
    resolved = ledger.resolve("sum_never_declared")

    assert resolved.verdict == UNRESOLVED
    assert resolved.usable_for_decision is False
    assert resolved.problems == ("no one has declared what this was built from",)


def test_forgetting_one_dependency_invalidates_the_whole_synthesis(store, ledger, two):
    declare(ledger, [{"record_id": one} for one in two])
    manager = ErasureManager(store, owner_principal=OWNER)
    preview = manager.preview(record_ids=[two[1]], actor=OWNER, reason="withdrawn")
    manager.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                    actor=OWNER)

    resolved = ledger.resolve("sum_incident_week")

    assert resolved.verdict == INVALID
    assert resolved.withheld is True
    assert resolved.verified == 1
    assert "forgotten or hidden" in resolved.problems[0]


def test_a_withheld_synthesis_is_replaced_by_the_evidence_that_still_stands(
        store, ledger, two):
    declare(ledger, [{"record_id": one} for one in two])
    store.hide(two[1], reason="not mine to share", actor=OWNER)

    resolved = ledger.resolve("sum_incident_week")
    safe = ledger.safe_evidence("sum_incident_week")

    assert resolved.verdict == INVALID
    assert [item.id for item in safe] == [two[0]], "the survivor is still worth handing over"


def test_citations_outside_the_callers_scope_withhold_the_artifact(store, ledger, two):
    identity = IdentityStore(store, owner_principal=OWNER)
    mine, theirs = identity.account("email", "me@example.com"), \
        identity.account("email", "partner@example.com")
    scoped = store.commit(envelope(source_id="msg-3", text="Their private salary detail.",
                                   metadata={"account_ids": [theirs]}))["id"]
    shared = ProvenanceLedger(store, identity=identity)
    shared.declare("sum_salaries", kind="summary",
                   citations=[{"record_id": mine and two[0]}, {"record_id": scoped}])

    for_caller = shared.resolve("sum_salaries", account_id=mine)
    for_owner = shared.resolve("sum_salaries", account_id=theirs)

    assert for_caller.verdict == INVALID
    assert "outside this scope" in for_caller.problems[0]
    assert [item.id for item in for_caller.evidence] == [two[0]]
    # Their own view is complete: one record is theirs and the other claims no
    # scope at all. Withholding is per caller, not a property of the artifact.
    assert for_owner.verdict == COMPLETE


def test_a_truncated_manifest_is_admitted_as_partial_and_is_not_a_decision_basis(
        store, ledger, two):
    declare(ledger, [{"record_id": one} for one in two], full_coverage=False)

    resolved = ledger.resolve("sum_incident_week")

    assert resolved.verdict == PARTIAL
    assert resolved.usable_for_decision is False
    assert resolved.problems == ()


def test_a_span_that_no_longer_lands_on_its_quote_is_invalid_not_merely_stale(
        store, ledger, two):
    declare(ledger, [{"record_id": two[0], "quote": "failed at 04:12"}])
    store.db.execute("BEGIN IMMEDIATE")
    store.db.execute("UPDATE records SET text=? WHERE id=?",
                     ("The cluster recovered quietly.", two[0]))
    store.db.execute("COMMIT")

    resolved = ledger.resolve("sum_incident_week")

    assert resolved.verdict == INVALID
    assert resolved.verified == 0
    assert "no longer matches" in resolved.problems[0]


def test_a_citation_to_a_record_that_never_existed_is_unresolved(store, ledger, two):
    declare(ledger, [{"record_id": two[0]}, {"record_id": two[1]}])
    # FK enforcement makes a dangling citation impossible from inside the
    # application, so this simulates the one thing that can do it: a database
    # file restored by hand from an older snapshot. The pragma is set outside
    # the transaction, because SQLite ignores it inside one.
    store.db.execute("PRAGMA foreign_keys=OFF")
    store.db.execute("BEGIN IMMEDIATE")
    store.db.execute("DELETE FROM record_fts WHERE id=?", (two[1],))
    store.db.execute("DELETE FROM records WHERE id=?", (two[1],))
    store.db.execute(
        "UPDATE derived_citations SET record_id=? WHERE artifact_id=? AND record_id=?",
        ("rec_" + "0" * 32, "sum_incident_week", two[0]))
    store.db.execute("COMMIT")
    store.db.execute("PRAGMA foreign_keys=ON")

    resolved = ledger.resolve("sum_incident_week")

    assert resolved.verdict == UNRESOLVED
    assert resolved.verified == 0
    assert set(resolved.problems) == {
        f"cited evidence {two[1]} does not exist",
        "cited evidence rec_" + "0" * 32 + " does not exist",
    }, "a vanished row says nothing was found, not that something was forgotten"


def test_a_record_can_be_shown_everything_that_leans_on_it(store, ledger, two):
    """One walker of "what quotes this", and it says what kind of thing each one is."""
    declare(ledger, [{"record_id": one} for one in two])
    ledger.declare("lesson_retry", kind="lesson", citations=[{"record_id": two[1]}])

    leaning = Lineage(store).artifacts([two[1]])
    assert leaning == [{"kind": "lesson", "id": "lesson_retry"},
                       {"kind": "summary", "id": "sum_incident_week"}], \
        "a lesson called a summary is a wrong answer in the one preview that is checked"
    assert Lineage(store).artifacts([]) == []


def test_no_lifecycle_event_drops_a_manifest_and_leaves_the_artifact_behind(
        store, ledger, two):
    declare(ledger, [{"record_id": one} for one in two])
    with pytest.raises(AttributeError, match="forget"):
        ledger.forget


def test_the_verdict_is_recomputed_rather_than_recalled(store, ledger, two):
    declare(ledger, [{"record_id": one} for one in two])
    before = ledger.resolve("sum_incident_week")
    store.hide(two[0], reason="retracted", actor=OWNER)
    after = ledger.resolve("sum_incident_week")

    assert before.verdict == COMPLETE and after.verdict == INVALID
    assert ledger.artifacts()[0]["declared_at"] == ledger.artifacts()[0]["declared_at"]
