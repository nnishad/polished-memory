"""C1 canonical evidence store invariants."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_memory.ids import backend_document_id, record_id
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


def test_supersede_hides_from_search_but_preserves_history(store):
    old = store.commit(envelope())
    new = store.commit(envelope(revision="2", text="Correction: the meeting is Friday."))
    assert {hit.id for hit in store.search("meeting")} == {old["id"], new["id"]}

    store.supersede(old["id"], new["id"], reason="user correction", actor="owner")

    assert [hit.id for hit in store.search("meeting")] == [new["id"]]
    assert store.get(old["id"]) is None                             # not visible
    assert store.get(old["id"], include_hidden=True) is not None     # still stored
    row = store.db.execute(
        "SELECT replacement_id FROM record_visibility WHERE record_id=?", (old["id"],)
    ).fetchone()
    assert row["replacement_id"] == new["id"]


def test_hidden_records_leave_the_lexical_index(store):
    committed = store.commit(envelope())
    assert len(store.search("meeting")) == 1
    store.hide(committed["id"], reason="scope revocation", actor="owner")
    assert store.search("meeting") == []
    store.show(committed["id"], reason="revocation reversed", actor="owner")
    assert [hit.id for hit in store.search("meeting")] == [committed["id"]]


def test_parent_must_resolve_to_live_evidence(store):
    with pytest.raises(EvidenceError, match="does not resolve to live evidence"):
        store.commit(envelope(parent_record_ids=["rec_" + "0" * 32]))
    parent = store.commit(envelope(source_id="msg-0"))
    child = store.commit(envelope(source_id="msg-1", parent_record_ids=[parent["id"]]))
    assert store.parents(child["id"]) == [parent["id"]]
    assert store.dependents(parent["id"]) == [child["id"]]


def test_audit_never_contains_evidence_bodies(store):
    store.commit(envelope(text="passport number 42 secret"))
    rows = store.db.execute("SELECT metadata FROM audit").fetchall()
    assert rows
    for row in rows:
        assert "passport" not in row[0]
        assert "42 secret" not in row[0]


def test_backend_projection_requires_underscore_free_document_id(store):
    committed = store.commit(envelope())
    revision = store.get(committed["id"]).revision
    with pytest.raises(EvidenceError, match="must not contain"):
        store.project(committed["id"], revision, backend="hindsight", bank_id="hermes",
                      document_id="rec_ambiguous_id")
    document = backend_document_id(committed["id"], revision)
    assert "_" not in document and "~" not in document
    store.project(committed["id"], revision, backend="hindsight", bank_id="hermes",
                  document_id=document)
    pending = store.pending_projections(backend="hindsight", bank_id="hermes")
    assert [row["document_id"] for row in pending] == [document]
    # Re-projecting the same revision is an idempotent upsert, not a second row.
    store.project(committed["id"], revision, backend="hindsight", bank_id="hermes",
                  document_id=document)
    assert len(store.pending_projections(backend="hindsight", bank_id="hermes")) == 1


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
    ).format(repr(str(Path(__file__).resolve().parents[1] / "src")), repr(str(tmp_path / "x.db")))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "rec_"


def test_an_unknown_id_is_not_visible_rather_than_an_error(store):
    """Callers legitimately hold ids an erasure or a stale cursor invalidated."""
    assert store.live_and_visible("rec_" + "0" * 32) is False
    assert store.get("rec_" + "0" * 32) is None
