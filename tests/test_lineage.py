"""C1 lineage: one traversal for revisions, consequence and derived products.

These tests read the same edges the erasure preview and the doctor report on, so a
drift here is a drift in what an owner is told a deletion will cost.
"""
from __future__ import annotations

import pytest

from conftest import envelope
from hermes_memory.backend.provenance import ProvenanceLedger
from hermes_memory.knowledge.assertions import AssertionStore
from hermes_memory.knowledge.summaries import SummaryStore
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.storage.evidence import EvidenceError
from hermes_memory.storage.identity import IdentityStore
from hermes_memory.storage.lineage import Lineage

OWNER = "owner-principal"


@pytest.fixture()
def lineage(store):
    return Lineage(store)


def commit(store, source_id, text, **overrides):
    return str(store.commit(envelope(source_id=source_id, text=text, **overrides))["id"])


def correct(store, source_id, revision, text):
    """A correction is a new revision of the same source item."""
    return commit(store, source_id, text, revision=revision)


# -- revisions and chains ----------------------------------------------------

def test_corrections_stay_answerable_as_history(store, lineage):
    first = commit(store, "msg-1", "The meeting moved to Thursday.")
    second = correct(store, "msg-1", "2", "The meeting moved to Friday.")
    store.supersede(first, second, reason="corrected by the sender", actor=OWNER)
    revisions = lineage.revisions("gmail", "msg-1")
    assert [item["revision"] for item in revisions] == ["1", "2"]
    assert revisions[0]["hidden"] == 1 and revisions[0]["replacement_id"] == second
    assert lineage.current("gmail", "msg-1") == second


def test_a_hidden_record_is_current_by_nobody(store, lineage):
    first = commit(store, "msg-1", "Private note.")
    assert lineage.current("gmail", "msg-1") == first
    store.hide(first, reason="the owner withdrew it", actor=OWNER)
    assert lineage.current("gmail", "msg-1") is None
    assert lineage.revisions("gmail", "msg-1", include_hidden=False) == []


def test_a_supersession_chain_is_walked_to_its_head(store, lineage):
    ids = [commit(store, "msg-1", f"Version {n}.", revision=str(n)) for n in (1, 2, 3)]
    store.supersede(ids[0], ids[1], reason="corrected", actor=OWNER)
    store.supersede(ids[1], ids[2], reason="corrected again", actor=OWNER)
    chain = lineage.chain(ids[0])
    assert chain["head"] == ids[2]
    assert chain["ancestors"] == [ids[1], ids[2]]
    assert chain["replaced_by"] == ids[1]
    assert lineage.chain(ids[2])["ancestors"] == []


def test_a_chain_of_corrections_builds_itself(store, lineage):
    ids = [commit(store, "msg-1", f"Version {n}.", revision=str(n)) for n in (1, 2, 3)]

    # Nobody called supersede. Each commit retired the revision it replaced, and the
    # pointers still run one step at a time rather than sending the first straight to the
    # last, so "what did we believe then" has an answer for every date rather than only
    # for the newest one.
    assert lineage.current("gmail", "msg-1") == ids[2]
    assert lineage.chain(ids[0])["ancestors"] == [ids[1], ids[2]]
    assert lineage.chain(ids[0])["head"] == ids[2]
    assert [item["revision"] for item in lineage.revisions("gmail", "msg-1",
                                                           include_hidden=False)] == ["3"]


def test_a_supersession_cycle_is_reported_rather_than_walked_forever(store, lineage):
    first = commit(store, "msg-1", "One.")
    second = commit(store, "msg-1", "Two.", revision="2")
    store.supersede(first, second, reason="corrected", actor=OWNER)
    # Corrupt the other way round deliberately: the traversal must not hang on it.
    store.db.execute("INSERT INTO record_visibility(record_id, hidden, replacement_id, "
                     "reason, changed_at) VALUES(?,1,?,?,?) ON CONFLICT(record_id) DO "
                     "UPDATE SET hidden=1, replacement_id=excluded.replacement_id",
                     (second, first, "corrupt", "2026-09-15T09:00:00+00:00"))
    chain = lineage.chain(first)
    assert chain["truncated"] is True and chain["cycle_at"] == first


def test_a_hide_is_not_reported_as_a_correction(store, lineage):
    record = commit(store, "msg-1", "Withdrawn.")
    store.hide(record, reason="the owner withdrew it", actor=OWNER)
    chain = lineage.chain(record)
    assert chain["replaced_by"] is None and chain["head"] == record


# -- consequence -------------------------------------------------------------

def test_dependents_are_followed_transitively(store, lineage):
    a = commit(store, "src-a", "One.")
    b = commit(store, "src-b", "Two.")
    c = commit(store, "src-c", "Three.")
    store.db.execute("BEGIN IMMEDIATE")
    store.db.execute("INSERT INTO record_dependencies(child_id, parent_id) VALUES(?,?)",
                     (b, a))
    store.db.execute("INSERT INTO record_dependencies(child_id, parent_id) VALUES(?,?)",
                     (c, b))
    store.db.execute("COMMIT")
    closure, truncated = lineage.closure([a])
    assert closure == sorted([a, b, c]) and truncated is False
    assert lineage.parents(c) == [b]


def test_a_dependency_cycle_terminates(store, lineage):
    a = commit(store, "src-a", "One.")
    b = commit(store, "src-b", "Two.")
    store.db.execute("BEGIN IMMEDIATE")
    store.db.execute("INSERT INTO record_dependencies(child_id, parent_id) VALUES(?,?)",
                     (b, a))
    store.db.execute("INSERT INTO record_dependencies(child_id, parent_id) VALUES(?,?)",
                     (a, b))
    store.db.execute("COMMIT")
    closure, _ = lineage.closure([a])
    assert closure == sorted([a, b])


def test_a_hub_is_reported_as_truncated_rather_than_endless(store, lineage):
    seed = commit(store, "src-a", "The hub.")
    for index in range(12):
        child = commit(store, f"msg-{index}", f"Leaf {index}.")
        store.db.execute("BEGIN IMMEDIATE")
        store.db.execute("INSERT INTO record_dependencies(child_id, parent_id) VALUES(?,?)",
                         (child, seed))
        store.db.execute("COMMIT")
    closure, truncated = lineage.closure([seed], cap=5)
    assert len(closure) <= 6 and truncated is True, \
        "an impact list that says 'and everything' is not something an owner can agree to"


def test_nothing_reached_from_nothing_is_an_error(store, lineage):
    with pytest.raises(EvidenceError, match="at least one record"):
        lineage.impact([])


def summary_for(store, record, quote="overdue since Monday", scope="project-x"):
    ledger = ProvenanceLedger(store)
    return SummaryStore(store, ledger=ledger).publish(
        scope=scope, kind="project", title="Invoices", body="The invoice is overdue.",
        citations=[{"record_id": record, "quote": quote}],
        processor_fingerprint="test-processor-1")


# -- derived products --------------------------------------------------------

def test_a_summary_quoting_evidence_appears_in_the_impact_radius(store, lineage):
    record = commit(store, "msg-1", "The invoice is overdue since Monday.")
    published = summary_for(store, record)
    impact = lineage.impact([record])
    assert {"kind": "summary", "id": published["id"]} in impact.artifacts
    assert impact.digest() == lineage.impact([record]).digest()


def test_an_assertion_quoting_evidence_appears_too(store, lineage):
    record = commit(store, "msg-1", "I prefer the Thursday slot.")
    assertions = AssertionStore(store)
    made = assertions.propose(subject="owner", predicate="prefers", value="Thursday",
                             kind="preference", evidence_kind="explicit_statement",
                             record_id=record, quote="the Thursday slot",
                             proposed_by=OWNER)
    artifacts = lineage.artifacts([record])
    assert {"kind": "assertion", "id": made["id"]} in artifacts


def test_the_radius_digest_moves_when_a_product_appears(store, lineage):
    record = commit(store, "msg-1", "The invoice is overdue since Monday.")
    before = lineage.impact([record]).digest()
    summary_for(store, record)
    assert lineage.impact([record]).digest() != before, \
        "a confirmation digest that ignores new dependents would let one slip past"


def test_an_erasure_preview_shows_the_products_it_will_orphan(store, lineage):
    record = commit(store, "msg-1", "The invoice is overdue since Monday.")
    summary = summary_for(store, record)
    erasure = ErasureManager(store, owner_principal=OWNER)
    preview = erasure.preview(record_ids=[record], actor=OWNER, reason="forget it")
    assert {"kind": "summary", "id": summary["id"]} in preview["derived_products"]


def test_citations_report_the_live_state_of_what_they_quote(store, lineage):
    record = commit(store, "msg-1", "The invoice is overdue since Monday.")
    summary = summary_for(store, record)
    assert all(item["live"] for item in lineage.citations_of(summary["id"]))
    store.hide(record, reason="withdrawn", actor=OWNER)
    assert all(not item["live"] for item in lineage.citations_of(summary["id"]))


def test_an_unknown_artifact_cites_nothing(store, lineage):
    assert lineage.citations_of("sum_nothing") == []


# -- explanation -------------------------------------------------------------

def test_explain_says_why_a_record_is_not_retrievable(store, lineage):
    record = commit(store, "msg-1", "Meeting notes.")
    assert lineage.explain(record)["retrievable"] is True
    store.hide(record, reason="the owner withdrew it", actor=OWNER)
    account = lineage.explain(record)
    assert account["state"] == "hidden" and account["retrievable"] is False
    assert "withdrew" in account["reason"]


def test_explain_tells_an_erased_record_apart_from_a_typo(store, lineage):
    missing = lineage.explain("rec_does_not_exist")
    assert missing["present"] is False and missing["state"] == "unknown"
    assert "never held" in missing["reason"]
    record = commit(store, "msg-1", "Something to forget.")
    erasure = ErasureManager(store, owner_principal=OWNER)
    preview = erasure.preview(record_ids=[record], actor=OWNER, reason="finished with it")
    erasure.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                    actor=OWNER)
    erased = lineage.explain(record)
    assert erased["state"] == "erased" and preview["intent_id"] in erased["reason"]


def test_explain_names_the_accounts_without_granting_them(store, lineage):
    identity = IdentityStore(store, owner_principal=OWNER)
    account = identity.account("email", "dana@example.com", label="Dana")
    record = commit(store, "msg-1", "Dana's invoice.",
                   metadata={"participants": [{"namespace": "email",
                                               "address": "dana@example.com"}]})
    assert lineage.explain(record)["accounts"] == [account]


# -- integrity ---------------------------------------------------------------

def test_an_edge_to_a_record_that_never_existed_is_surfaced(store, lineage):
    record = commit(store, "msg-1", "One.")
    store.db.execute("PRAGMA foreign_keys=OFF")
    store.db.execute("INSERT INTO record_dependencies(child_id, parent_id) "
                     "VALUES(?, 'rec_never_existed')", (record,))
    store.db.execute("PRAGMA foreign_keys=ON")
    found = lineage.dangling()
    assert any(item["table_name"] == "record_dependencies"
               and item["to_id"] == "rec_never_existed" for item in found)
    assert store.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_a_clean_store_reports_nothing_dangling(store, lineage):
    commit(store, "msg-1", "One.")
    assert lineage.dangling() == []


def test_a_leaf_record_is_normal_and_an_unlinked_capture_is_visible(store, lineage):
    lonely = commit(store, "msg-1", "A plain note with no links.")
    assert lonely in lineage.orphans()
    parent = commit(store, "msg-2", "A note something was built from.")
    child = commit(store, "msg-3", "Built from another.")
    store.db.execute("BEGIN IMMEDIATE")
    store.db.execute("INSERT INTO record_dependencies(child_id, parent_id) VALUES(?,?)",
                     (child, parent))
    store.db.execute("COMMIT")
    assert child not in lineage.orphans()
