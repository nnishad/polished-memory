"""C6 summaries: hierarchy, coverage, and what happens when the evidence moves."""
from __future__ import annotations

import time

import pytest

from conftest import envelope
from hermes_memory.backend.provenance import COMPLETE, INVALID, PARTIAL, ProvenanceLedger
from hermes_memory.knowledge.summaries import MAX_CITATIONS_PER_SUMMARY, SummaryStore
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.storage.evidence import EvidenceError
from hermes_memory.context import ContextBroker

OWNER = "owner-principal"
BROKE = "The build broke at 04:12 and recovered at 05:00."


@pytest.fixture()
def ledger(store):
    return ProvenanceLedger(store)


@pytest.fixture()
def summaries(store, ledger):
    return SummaryStore(store, ledger=ledger, owner_principal=OWNER)


@pytest.fixture()
def incident(store):
    return store.commit(envelope(source_id="msg-incident", text=BROKE))["id"]


def publish(summaries, citations, **overrides):
    args = {"scope": "day:2026-09-25", "kind": "day", "title": "The 04:12 incident",
            "body": "The build broke and recovered within the hour.",
            "citations": citations, "processor_fingerprint": "summarizer-1.0"}
    args.update(overrides)
    return summaries.publish(**args)


# -- publication ---------------------------------------------------------------


def test_a_published_summary_carries_its_verdict_and_its_window(store, summaries, incident):
    out = publish(summaries, [{"record_id": incident, "quote": "recovered at 05:00"}],
                  window=("2026-09-25T00:00:00+00:00", "2026-09-25T23:59:59+00:00"))

    assert out["published"] is True and out["verdict"] == COMPLETE
    read = summaries.read(out["id"])
    assert read["available"] is True and read["usable_for_decision"] is True
    assert read["window_to"].startswith("2026-09-25T23:59:59")
    assert summaries.verdict(out["id"]) == COMPLETE


def test_republishing_the_same_words_does_not_add_a_revision(store, summaries, incident):
    first = publish(summaries, [{"record_id": incident}])
    again = publish(summaries, [{"record_id": incident}])

    assert again == {**first, "published": False}
    assert summaries.summarize() == {"day.published": 1}


def test_a_new_revision_withdraws_the_one_it_replaces(store, summaries, incident):
    old = publish(summaries, [{"record_id": incident}])
    new = publish(summaries, [{"record_id": incident}], body="Corrected: two hours, not one.",
                  supersedes=old["id"])

    assert new["revision"] == 2
    assert summaries.get(old["id"]).status == "withdrawn"
    assert [item.revision for item in summaries.latest("day:2026-09-25")] == [2]


def test_supersession_cannot_move_a_summary_into_another_scope(store, summaries, incident):
    old = publish(summaries, [{"record_id": incident}])

    with pytest.raises(EvidenceError, match="cannot move between scopes"):
        publish(summaries, [{"record_id": incident}], scope="project:build",
                kind="project", supersedes=old["id"])


@pytest.mark.parametrize("citations, message", [
    ([], "cites between 1 and"),
    ([{"record_id": f"rec_{index:032x}"} for index in range(MAX_CITATIONS_PER_SUMMARY + 1)],
     "cites between 1 and"),
])
def test_a_summary_without_a_bounded_manifest_is_refused(store, summaries, citations, message):
    with pytest.raises(EvidenceError, match=message):
        publish(summaries, citations)


def test_an_unattributed_processor_is_refused(store, summaries, incident):
    with pytest.raises(EvidenceError, match="processor version"):
        publish(summaries, [{"record_id": incident}], processor_fingerprint="  ")


def test_only_the_owner_may_publish_a_mental_model(store, summaries, incident):
    citations = [{"record_id": incident, "quote": "broke at 04:12"}]

    with pytest.raises(EvidenceError, match="needs the owner's approval"):
        publish(summaries, citations, kind="mental_model", scope="model:reliability",
                title="We ship on a fragile build", approved_by="agent-model")

    approved = publish(summaries, citations, kind="mental_model", scope="model:reliability",
                       title="We ship on a fragile build", approved_by=OWNER)
    assert summaries.read(approved["id"])["available"] is True


def test_rolling_back_a_publish_leaves_no_orphan_manifest(store, summaries, incident):
    store.db.execute("BEGIN IMMEDIATE")
    summary_id = "sum_" + "a" * 32
    store.db.execute(
        "INSERT INTO summaries(id, scope, kind, title, body, revision, status, "
        "processor_fingerprint, epoch, budget_tokens, created_at, published_at) "
        "VALUES(?,?,?,?,?,?, 'published', ?,?,?,?,?)",
        (summary_id, "day:x", "day", "t", "b", 1, "p-1", store.epoch(), 100,
         "2026-09-25T00:00:00+00:00", "2026-09-25T00:00:00+00:00"))
    summaries.ledger.declare(summary_id, kind="summary:day",
                             citations=[{"record_id": incident}], db=store.db)
    store.db.execute("ROLLBACK")

    assert summaries.get(summary_id) is None
    assert summaries.ledger.artifacts() == []
    assert store.db.execute("SELECT count(*) FROM derived_citations").fetchone()[0] == 0


# -- the summary is not evidence ------------------------------------------------


def test_a_summary_never_becomes_a_source_for_the_next_one(store, summaries, incident):
    out = publish(summaries, [{"record_id": incident, "quote": "broke at 04:12"}])
    broker = ContextBroker(store, assertions=None, cache=None)

    packet = broker.assemble("build broke recovered")

    assert [item.id for item in packet.items] == [incident]
    assert out["id"] not in {item.id for item in packet.items}
    assert len(store.search("fragile")) == 0
    assert summaries.get(out["id"]).body not in [item.text for item in store.search("broke")]
    broker.close()


# -- what the evidence does afterwards ----------------------------------------


def test_forgetting_a_cited_record_withholds_the_summary(store, summaries, incident, ledger):
    out = publish(summaries, [{"record_id": incident, "quote": "broke at 04:12"}])
    manager = ErasureManager(store, owner_principal=OWNER)
    preview = manager.preview(record_ids=[incident], actor=OWNER, reason="not ours")
    manager.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                    actor=OWNER)

    read = summaries.read(out["id"])

    assert read["available"] is False
    assert read["body"] is None, "a caveat on an unsupported claim is still the claim"
    assert read["verdict"] == INVALID
    assert summaries.latest("day:2026-09-25") == []


def test_a_summary_that_lost_its_evidence_is_absent_rather_than_substituted(
        store, summaries, incident):
    first = publish(summaries, [{"record_id": incident}], body="First reading.")
    other = store.commit(envelope(source_id="msg-2",
                                  text="A second note about the build."))["id"]
    publish(summaries, [{"record_id": other}], body="Second reading.",
            supersedes=first["id"])
    store.hide(other, reason="retracted", actor=OWNER)

    assert summaries.latest("day:2026-09-25") == [], (
        "the first reading is superseded, and a superseded summary is not a current one")
    assert store.live_and_visible(incident) is True, "the evidence is still there to read"


def test_a_truncated_manifest_is_readable_but_is_not_a_decision_basis(
        store, summaries, incident):
    out = publish(summaries, [{"record_id": incident}], full_coverage=False)

    read = summaries.read(out["id"])

    assert read["available"] is True
    assert read["verdict"] == PARTIAL
    assert read["usable_for_decision"] is False


def test_a_correction_after_publication_makes_it_stale(store, summaries, incident):
    out = publish(summaries, [{"record_id": incident}])
    store.commit(envelope(source_id="msg-incident", revision="2",
                          text="Correction: the build broke at 04:12 and stayed broken."))

    assert summaries.is_stale(summaries.get(out["id"])) is True
    # Two reasons, because a correction says two things at once: the window has new
    # material (so the summary is out of date) and the sentence the summary quoted is no
    # longer live evidence (so the summary is unsupported). Reporting only the first
    # would let a refresh be scheduled while the old body kept being served as if it
    # still stood.
    assert summaries.needs_refresh() == [
        {"scope": "day:2026-09-25", "kind": "day", "summary": out["id"],
         "reasons": ["unsupported", "changed since publication"], "revisions": 1}]
    assert summaries.read(out["id"])["available"] is False


def test_a_refresh_promise_is_measured_against_the_clock_it_was_made_with(store):
    record = store.commit(envelope(text=BROKE))["id"]
    past = SummaryStore(store, owner_principal=OWNER,
                        clock=lambda: time.mktime(time.strptime("2026-09-25T12:00:00",
                                                                "%Y-%m-%dT%H:%M:%S")))
    soon = publish(past, [{"record_id": record}], refresh_after="2026-09-25T06:00:00+00:00")
    later = publish(past, [{"record_id": record}], body="A second take on the same day.",
                    title="Second", refresh_after="2026-09-26T06:00:00+00:00")

    assert past.is_stale(past.get(soon["id"])) is True
    assert past.is_stale(past.get(later["id"])) is False


def test_many_changes_ask_for_one_refresh_per_scope(store, summaries, incident):
    publish(summaries, [{"record_id": incident}])
    publish(summaries, [{"record_id": incident}], body="Another take on the same day.",
            title="Second summary of the day")
    store.commit(envelope(source_id="msg-incident", revision="2",
                          text="Correction: the build broke at 04:12 and stayed broken."))

    needed = summaries.needs_refresh()

    assert len(needed) == 1, "two revisions of one afternoon are one refresh, not two"
    assert needed[0]["revisions"] == 2


def test_refresh_requests_coalesce_and_settle(store, summaries):
    summaries.request_refresh("day:2026-09-25", kind="day",
                              through_at="2026-09-25T12:00:00+00:00")
    summaries.request_refresh("day:2026-09-25", kind="day",
                              through_at="2026-09-25T12:00:00+00:00")
    pending = summaries.pending_refreshes()

    assert len(pending) == 1 and pending[0]["state"] == "pending"
    assert summaries.settle_refresh(scope="day:2026-09-25", kind="day",
                                    through_at="2026-09-25T12:00:00+00:00",
                                    ok=True)["settled"] == 1
    assert summaries.pending_refreshes() == []
    assert summaries.settle_refresh(scope="day:2026-09-25", kind="day",
                                    through_at="2026-09-25T12:00:00+00:00",
                                    ok=False)["settled"] == 0, "already settled"


def test_only_the_owner_can_withdraw(store, summaries, incident):
    out = publish(summaries, [{"record_id": incident}])

    with pytest.raises(EvidenceError, match="only the owner may withdraw"):
        summaries.withdraw(out["id"], actor="agent-model", reason="not useful")

    assert summaries.withdraw(out["id"], actor=OWNER, reason="superseded by hand")["withdrawn"]
    assert summaries.latest("day:2026-09-25") == []
    assert summaries.read(out["id"])["reason"] == "withdrawn"


def test_an_unsupported_window_is_refused_before_anything_is_written(store, summaries,
                                                                     incident):
    with pytest.raises(EvidenceError, match="ends before it begins"):
        publish(summaries, [{"record_id": incident}],
                window=("2026-09-25T12:00:00+00:00", "2026-09-25T00:00:00+00:00"))
    with pytest.raises(ValueError, match="timezone"):
        publish(summaries, [{"record_id": incident}],
                window=("2026-09-25T12:00:00", "2026-09-26T00:00:00"))
    assert summaries.summarize() == {}
