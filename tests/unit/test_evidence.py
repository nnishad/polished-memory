"""C1 canonical evidence store invariants."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_memory.ids import backend_document_id, record_id
from hermes_memory.backend.document_map import DocumentMap
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.storage.lineage import Lineage
from hermes_memory.storage.evidence import EvidenceError, EvidenceStore
from hermes_memory.storage.migrations import (
    MIGRATIONS,
    apply_migrations,
    connect,
    current_version,
)

from conftest import OBSERVED, SOURCE, envelope


def test_replaying_the_same_revision_is_idempotent(store):
    first = store.commit(envelope())
    second = store.commit(envelope())
    assert first["duplicate"] is False
    assert second == {"id": first["id"], "duplicate": True}
    assert store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 1


def test_same_revision_with_different_bytes_is_refused_not_forked(store):
    store.commit(envelope())
    with pytest.raises(EvidenceError, match="content conflict"):
        store.commit(envelope(text="silently edited after the fact"))


def test_record_id_is_reproducible_from_source_coordinates(store):
    committed = store.commit(envelope())
    assert committed["id"] == record_id(SOURCE, "msg-1", "1")


def test_unknown_event_time_stays_unknown_and_is_not_ingestion_time(store):
    committed = store.commit(
        envelope(occurred_at=None, occurred_precision="unknown", observed_at=OBSERVED)
    )
    stored = store.get(committed["id"])
    assert stored.occurred_at is None
    assert stored.occurred_precision == "unknown"
    assert stored.ingested_at != OBSERVED  # ingestion time must not masquerade as event time


def test_precision_without_a_time_is_rejected(store):
    with pytest.raises(EvidenceError, match="requires an occurred_at"):
        store.commit(envelope(occurred_at=None, occurred_precision="day"))


def test_naive_timestamp_is_refused(store):
    with pytest.raises(ValueError, match="timezone"):
        store.commit(envelope(occurred_at="2026-09-24T09:15:00"))


def test_a_later_revision_supersedes_the_earlier_one_without_anyone_asking(store):
    old = store.commit(envelope())
    new = store.commit(envelope(revision="2", text="Correction: the meeting is Friday."))

    # The corrected revision is what the source says *now*, so it is the only thing
    # retrieval may offer. Leaving revision 1 in the live set would answer "when is the
    # meeting" with both dates, and would leave every claim that quoted the old sentence
    # reporting itself supported.
    assert [hit.id for hit in store.search("meeting")] == [new["id"]]
    assert store.get(old["id"]) is None                             # not visible
    assert store.get(old["id"], include_hidden=True) is not None     # still stored
    row = store.db.execute(
        "SELECT replacement_id, reason FROM record_visibility WHERE record_id=?",
        (old["id"],)).fetchone()
    assert row["replacement_id"] == new["id"]
    assert row["reason"] == "revised to 2"
    # The index is not the source of truth, but a withdrawn sentence must not sit in it:
    # every read that goes through the index would otherwise find a row the store has
    # already said is not current.
    assert store.db.execute("SELECT count(*) FROM record_fts WHERE id=?",
                            (old["id"],)).fetchone()[0] == 0
    journaled = store.db.execute(
        "SELECT change FROM change_journal WHERE record_id=? ORDER BY seq", (old["id"],),
    ).fetchall()
    assert [row["change"] for row in journaled] == ["add", "supersede"]
    audited = store.db.execute(
        "SELECT object_id, metadata FROM audit WHERE action='evidence_supersede'").fetchall()
    assert [row["object_id"] for row in audited] == [old["id"]]
    assert json.loads(audited[0]["metadata"]) == {"actor": "gmail", "replacement": new["id"]}


def test_a_correction_does_not_disturb_forgotten_evidence(store):
    first = store.commit(envelope())
    manager = ErasureManager(store, owner_principal="owner-principal")
    preview = manager.preview(record_ids=[first["id"]], actor="owner-principal",
                              reason="withdrawn")
    manager.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                    actor="owner-principal")

    second = store.commit(envelope(revision="2", text="Correction: the meeting is Friday."))

    # The tombstone is not a revision to supersede. Pointing a forgotten record at a live
    # one would write it back into a chain, and a chain is a thing the store reads out loud.
    assert [hit.id for hit in store.search("meeting")] == [second["id"]]
    assert store.db.execute("SELECT count(*) FROM record_visibility WHERE record_id=?",
                            (first["id"],)).fetchone()[0] == 0


def test_replaying_a_superseded_revision_does_not_un_correct_it(store):
    old = store.commit(envelope())
    store.commit(envelope(revision="2", text="Correction: the meeting is Friday."))

    replay = store.commit(envelope())

    # The replay is acknowledged, not rewritten — and the acknowledgement must not put
    # the withdrawn sentence back in the index, or a source that re-serves a stale page
    # would silently undo a correction.
    assert replay == {"id": old["id"], "duplicate": True}
    assert store.get(old["id"]) is None


def test_supersede_hides_from_search_but_preserves_history(store):
    old = store.commit(envelope(source_id="msg-1"))
    new = store.commit(envelope(source_id="msg-2", text="Correction: the meeting is Friday."))

    assert {hit.id for hit in store.search("meeting")} == {old["id"], new["id"]}

    store.supersede(old["id"], new["id"], reason="user correction", actor="owner")

    assert [hit.id for hit in store.search("meeting")] == [new["id"]]
    assert store.get(old["id"]) is None                             # not visible
    assert store.get(old["id"], include_hidden=True) is not None     # still stored
    row = store.db.execute(
        "SELECT replacement_id FROM record_visibility WHERE record_id=?", (old["id"],)
    ).fetchone()
    assert row["replacement_id"] == new["id"]
    # Journaled like any other change: the packet cache and every derived index decide what
    # to drop by the journal sequence, so an unreported supersession keeps serving a
    # reading the store has already withdrawn.
    assert store.db.execute(
        "SELECT count(*) FROM change_journal WHERE record_id=? AND change='supersede'",
        (old["id"],)).fetchone()[0] == 1


def test_the_latest_report_wins_even_when_its_revision_token_is_older(store):
    late = store.commit(envelope(revision="2", text="Correction: the meeting is Friday."))
    earlier = store.commit(envelope(text="The meeting moved to Thursday at 3pm."))

    # Revision tokens are not comparable here — a file adapter reports a content digest,
    # an IMAP revision a mod-sequence — so the source's latest report is the only ordering
    # available. What must never happen is the store retiring the row it just wrote and
    # leaving the item unfindable.
    assert [hit.id for hit in store.search("meeting")] == [earlier["id"]]
    assert store.live_and_visible(late["id"]) is False


def test_supersede_refuses_an_edge_it_cannot_name(store):
    kept = store.commit(envelope())

    with pytest.raises(EvidenceError, match="cannot supersede itself"):
        store.supersede(kept["id"], kept["id"], reason="typo", actor="owner")
    with pytest.raises(EvidenceError, match="unknown record"):
        store.supersede(kept["id"], "rec_" + "0" * 32, reason="typo", actor="owner")

    # A refused supersession leaves no trace to read back: neither the visibility row nor
    # the withdrawn revision, because both were written in the transaction that rolled back.
    assert store.db.execute("SELECT count(*) FROM record_visibility").fetchone()[0] == 0
    assert [hit.id for hit in store.search("meeting")] == [kept["id"]]


def test_a_hidden_record_leaves_retrieval_but_stays_answerable_as_history(store):
    """Hiding is a visibility decision; only forgetting destroys the revision.

    Nothing in this build un-hides a record. The reading that replaced it is a new row,
    so the hidden one has to stay quotable — "what did we believe then" is exactly the
    question asked after a correction.
    """
    committed = store.commit(envelope())
    assert len(store.search("meeting")) == 1
    store.hide(committed["id"], reason="scope revocation", actor="owner")
    assert store.search("meeting") == []
    assert store.get(committed["id"]) is None
    assert store.live_and_visible(committed["id"]) is False
    kept = store.get(committed["id"], include_hidden=True)
    assert kept is not None and "meeting" in kept.text


def test_parent_must_resolve_to_live_evidence(store):
    with pytest.raises(EvidenceError, match="does not resolve to live evidence"):
        store.commit(envelope(parent_record_ids=["rec_" + "0" * 32]))
    parent = store.commit(envelope(source_id="msg-0"))
    child = store.commit(envelope(source_id="msg-1", parent_record_ids=[parent["id"]]))
    lineage = Lineage(store)
    assert lineage.parents(child["id"]) == [parent["id"]]
    reached, truncated = lineage.closure([parent["id"]])
    assert truncated is False and child["id"] in reached


def test_audit_never_contains_evidence_bodies(store):
    store.commit(envelope(text="passport number 42 secret"))
    rows = store.db.execute("SELECT metadata FROM audit").fetchall()
    assert rows
    for row in rows:
        assert "passport" not in row[0]
        assert "42 secret" not in row[0]


def test_a_projection_is_recorded_under_an_unambiguous_document_id(store):
    """The id a chunk is derived from has to survive Hindsight's escaping rules.

    The store no longer accepts a hand-written mapping: the only way a
    ``backend_documents`` row appears is through the same call that submits it, which is
    what makes the underscore-free guarantee structural rather than a validation of
    somebody's input.
    """
    committed = store.commit(envelope())
    revision = store.get(committed["id"]).revision
    first = DocumentMap(store).begin(committed["id"], revision)
    assert "_" not in first["document_id"] and "~" not in first["document_id"]
    assert first["document_id"] == backend_document_id(committed["id"], revision)
    again = DocumentMap(store).begin(committed["id"], revision)
    rows = store.db.execute("SELECT state, count(*) AS n FROM backend_documents "
                            "GROUP BY state").fetchall()
    assert again["state"] == first["state"] == "queued"
    assert [(row["state"], row["n"]) for row in rows] == [("queued", 1)], \
        "re-recording one revision is an idempotent intent, not a second submission"
    with pytest.raises(EvidenceError, match="cannot project unknown record"):
        DocumentMap(store).begin("rec_" + "0" * 32, "1")


def test_operator_pause_is_persisted_and_authoritative(store):
    assert store.stage_is_paused(SOURCE, "formation") is False
    store.set_control(SOURCE, "formation", "paused", actor="owner", reason="budget",
                      policy_version="v1")
    assert store.stage_is_paused(SOURCE, "formation") is True
    # A stage pause is per-stage: capture continues.
    assert store.stage_is_paused(SOURCE, "capture") is False
    store.commit(envelope())  # capture still works while formation is paused


def test_control_rejects_an_unauthorised_state(store):
    with pytest.raises(EvidenceError, match="paused"):
        store.set_control(SOURCE, "formation", "approved-by-model", actor="agent",
                          reason="x", policy_version="v1")


def test_reopening_an_existing_database_does_not_replay_migrations(store):
    store.commit(envelope())
    path = store.path
    store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    store.close()
    reopened = EvidenceStore(path)
    try:
        assert reopened.db.execute("SELECT count(*) FROM records").fetchone()[0] == 1
    finally:
        reopened.close()


def test_migrations_refuse_a_schema_from_a_newer_build(tmp_path: Path):
    path = tmp_path / "future.db"
    db = connect(path)
    apply_migrations(db, at=lambda: "2026-09-25T00:00:00+00:00")
    db.execute("INSERT INTO schema_migrations(name, applied_at) VALUES('9999_from_the_future', ?)",
               ("2026-09-25T00:00:00+00:00",))
    assert current_version(db) == len(MIGRATIONS) + 1  # real migrations plus the planted future
    db.close()

    fresh = connect(path)
    try:
        with pytest.raises(RuntimeError, match="newer schema"):
            apply_migrations(fresh, at=lambda: "2026-09-25T00:00:00+00:00")
    finally:
        fresh.close()


def test_a_failed_commit_leaves_no_partial_state(tmp_path: Path):
    with EvidenceStore(tmp_path / "canonical.db") as store:
        parent = store.commit(envelope(source_id="msg-0"))
        before = store.db.execute("SELECT count(*) FROM records").fetchone()[0]
        with pytest.raises(EvidenceError):
            store.commit(envelope(source_id="msg-9", parent_record_ids=["rec_" + "f" * 32]))
        assert store.db.execute("SELECT count(*) FROM records").fetchone()[0] == before
        assert store.get(parent["id"]) is not None


def test_store_opens_and_writes_with_no_backend_or_network(tmp_path: Path):
    """The durability core must not import or require Hindsight."""
    import subprocess
    import sys

    script = (
        "import sys;"
        "assert 'hindsight_client' not in sys.modules;"
        "sys.path.insert(0, {});"
        "from hermes_memory.storage.evidence import EvidenceStore;"
        "s = EvidenceStore({});"
        "print(s.commit({{'source':'x','source_id':'1','revision':'1','kind':'message',"
        "'text':'offline','observed_at':'2026-09-25T00:00:00+00:00'}})['id'][:4])"
    ).format(repr(str(Path(__file__).resolve().parents[2] / "src")),
             repr(str(tmp_path / "x.db")))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "rec_"


def test_an_unknown_id_is_not_visible_rather_than_an_error(store):
    """Callers legitimately hold ids an erasure or a stale cursor invalidated."""
    assert store.live_and_visible("rec_" + "0" * 32) is False
    assert store.get("rec_" + "0" * 32) is None


# -- the read-only view ------------------------------------------------------

def test_a_read_only_store_reads_what_the_writable_one_wrote(store, tmp_path):
    from hermes_memory.storage.evidence import ReadOnlyStore

    record = store.commit(envelope())["id"]
    with ReadOnlyStore(tmp_path / "canonical.db") as reading:
        assert reading.get(record).text == store.get(record).text
        assert reading.epoch() == store.epoch()
        assert reading.db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert reading.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_a_read_only_store_refuses_every_write(store, tmp_path):
    from hermes_memory.storage.evidence import ReadOnlyStore

    record = store.commit(envelope())["id"]
    with ReadOnlyStore(tmp_path / "canonical.db") as reading:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reading.db.execute("UPDATE records SET text='edited' WHERE id=?", (record,))
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reading.db.execute("INSERT INTO audit(action, object_id, created_at, metadata) "
                               "VALUES('x','y','z','{}')")
    assert store.get(record).text == envelope()["text"]


def test_opening_a_missing_store_read_only_does_not_create_one(tmp_path):
    from hermes_memory.storage.evidence import ReadOnlyStore

    path = tmp_path / "never-existed.db"
    with pytest.raises(EvidenceError, match="nothing to read"):
        ReadOnlyStore(path)
    assert not path.exists()


def test_a_read_only_store_still_reports_the_controls_it_cannot_change(store, tmp_path):
    from hermes_memory.storage.evidence import ReadOnlyStore

    store.set_control("gmail", "capture", "paused", actor="owner", reason="review",
                      policy_version="v1")
    with ReadOnlyStore(tmp_path / "canonical.db") as reading:
        assert reading.stage_is_paused("gmail", "capture") is True
