"""C1 attachments: content-addressed bytes, reference counts and the forget path.

The plan keeps exactly one blob authority on the initial port, and it is the
database. That choice pays for itself in the gate cases: a crash between writing
a blob and committing its record cannot happen when both are the same
transaction, and forgetting is not complete while the bytes survive underneath
it.
"""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from conftest import envelope
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.storage.blobs import (MAX_ATTACHMENTS_PER_RECORD, Attachment, BlobError,
                                         BlobStore, normalize_filename, stage_bytes)
from hermes_memory.storage.evidence import EvidenceError, EvidenceStore

PDF = b"%PDF-1.4 " + bytes(range(256)) * 40
OWNER = "owner-principal"


@pytest.fixture()
def blobs(store):
    return BlobStore(store)


@pytest.fixture()
def record(store):
    return store.commit(envelope(source_id="msg-1"))["id"]


def attach(store, blobs, record_pk, payloads):
    """One attachment write inside its own transaction, as a source commit would."""
    store.db.execute("BEGIN IMMEDIATE")
    try:
        written = blobs.attach(record_pk, payloads, db=store.db)
        store.db.execute("COMMIT")
    except BaseException:
        store.db.execute("ROLLBACK")
        raise
    return written


def counts(store):
    return {table: store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("blob_contents", "blob_chunks", "attachments")}


# -- write and read -----------------------------------------------------------


def test_an_attachment_round_trips_and_is_attributed_to_its_record(store, blobs, record):
    written = attach(store, blobs, record, [{"data": PDF, "filename": "invoice.pdf",
                                             "mime": "application/pdf"}])

    assert [item.mime for item in written] == ["application/pdf"]
    assert written[0].size == len(PDF)
    assert written[0].sha256 == hashlib.sha256(PDF).hexdigest()
    assert blobs.read(written[0].id) == PDF
    assert [item.position for item in blobs.list_for(record)] == [0]
    ordered = attach(store, blobs, record, [{"data": b"cover", "filename": "b.pdf",
                                             "mime": "application/pdf", "position": 1}])
    assert [item.position for item in ordered] == [1]
    assert [item.position for item in blobs.list_for(record)] == [0, 1]


def test_a_large_attachment_is_chunked_and_reassembled(store, blobs, record):
    data = bytes(range(256)) * 8000  # ~2MB, several chunks

    written = attach(store, blobs, record, [{"data": data, "filename": "big.bin",
                                             "mime": "application/octet-stream"}])

    chunk_count = store.db.execute(
        "SELECT count(*) FROM blob_chunks WHERE sha256=?",
        (written[0].sha256,)).fetchone()[0]
    assert chunk_count > 1
    assert blobs.read(written[0].id) == data


def test_the_same_bytes_on_two_records_are_stored_once_and_referenced_twice(
        store, blobs, record):
    other = store.commit(envelope(source_id="msg-2", text="Fwd: the same invoice"))["id"]
    first = attach(store, blobs, record, [{"data": PDF, "filename": "invoice.pdf",
                                           "mime": "application/pdf"}])[0]
    second = attach(store, blobs, other, [{"data": PDF, "filename": "fwd-invoice.pdf",
                                           "mime": "application/PDF"}])[0]

    assert first.sha256 == second.sha256
    assert counts(store)["blob_contents"] == 1
    assert store.db.execute("SELECT refs FROM blob_contents").fetchone()[0] == 2


def test_replaying_the_same_page_does_not_double_count_or_fork_a_row(store, blobs, record):
    payload = [{"data": PDF, "filename": "invoice.pdf", "mime": "application/pdf"}]

    again = attach(store, blobs, record, payload)

    assert counts(store) == {"blob_contents": 1, "blob_chunks": 1, "attachments": 1}
    assert [item.id for item in again] == [item.id for item in blobs.list_for(record)]
    assert store.db.execute("SELECT refs FROM blob_contents").fetchone()[0] == 1


def test_a_cited_position_never_changes_its_mind(store, blobs, record):
    attach(store, blobs, record, [{"data": PDF, "filename": "invoice.pdf",
                                    "mime": "application/pdf"}])

    with pytest.raises(BlobError, match="already holds different bytes"):
        attach(store, blobs, record, [{"data": b"different", "filename": "invoice.pdf",
                                        "mime": "application/pdf"}])
    assert blobs.read(blobs.list_for(record)[0].id) == PDF


def test_a_write_without_a_transaction_is_refused(store, blobs, record):
    # The reference count and the row must rise and fall together; an ambient
    # transaction is the only thing that makes that true.
    with pytest.raises(BlobError, match="ambient transaction"):
        blobs.attach(record, [{"data": PDF, "filename": "a.pdf", "mime": "application/pdf"}])

    assert counts(store) == {"blob_contents": 0, "blob_chunks": 0, "attachments": 0}


# -- ingress limits ------------------------------------------------------------


def test_an_unbounded_or_malformed_payload_never_reaches_the_database(store, blobs, record):
    bad = [
        ({"data": b"", "filename": "empty.pdf", "mime": "application/pdf"},
         "empty attachment"),
        ({"data": "text, not bytes", "filename": "note.txt", "mime": "text/plain"},
         "must be bytes"),
        ({"data": PDF, "filename": "invoice.pdf", "mime": "not-a-type"},
         "type/subtype"),
        ({"data": PDF, "filename": "invoice.pdf", "mime": ""}, "type/subtype"),
    ]
    for payload, message in bad:
        with pytest.raises(BlobError, match=message):
            attach(store, blobs, record, [payload])
        assert counts(store) == {"blob_contents": 0, "blob_chunks": 0, "attachments": 0}


def test_an_oversized_attachment_is_refused_at_ingress(blobs, record):
    from hermes_memory.storage import blobs as module

    huge = b"x" * (module.MAX_ATTACHMENT_BYTES + 1)
    with pytest.raises(BlobError, match="ceiling"):
        stage_bytes(huge, filename="huge.bin", mime="application/octet-stream")


def test_too_many_attachments_on_one_record_is_refused(store, blobs, record):
    payloads = [{"data": bytes([index]), "filename": f"p{index}.bin",
                 "mime": "application/octet-stream"}
                for index in range(MAX_ATTACHMENTS_PER_RECORD + 1)]

    with pytest.raises(BlobError, match="per"):
        attach(store, blobs, record, payloads)
    assert counts(store)["attachments"] == 0


def test_a_source_supplied_name_is_made_inert():
    assert normalize_filename("..\\..\\windows\\system32\\config") == "config"
    assert normalize_filename("inv\x00oice\tpdf") == "invoicepdf"
    assert normalize_filename("   ") == "unnamed"
    assert normalize_filename(None) == "unnamed"
    assert len(normalize_filename("x" * 5000)) == 200


def test_one_write_cannot_claim_a_position_twice(store, blobs, record):
    payloads = [{"data": bytes([index]), "filename": f"p{index}.bin",
                 "mime": "application/octet-stream", "position": 0} for index in (1, 2)]

    with pytest.raises(BlobError, match="claimed twice"):
        attach(store, blobs, record, payloads)
    assert counts(store)["attachments"] == 0


@pytest.mark.parametrize("position", [-1, MAX_ATTACHMENTS_PER_RECORD, "0", True, 1.5])
def test_a_position_outside_the_ordering_is_refused(blobs, record, position):
    with pytest.raises(BlobError, match="position"):
        stage_bytes(b"x", filename="a.bin", mime="application/octet-stream",
                    position=position)


def test_attaching_to_an_unknown_record_is_refused(store, blobs):
    with pytest.raises(BlobError, match="unknown record"):
        attach(store, blobs, "rec_" + "0" * 32,
               [{"data": PDF, "filename": "a.pdf", "mime": "application/pdf"}])


# -- visibility and destruction ------------------------------------------------


def test_a_hidden_records_bytes_are_not_retrievable(store, blobs, record):
    written = attach(store, blobs, record, [{"data": PDF, "filename": "invoice.pdf",
                                             "mime": "application/pdf"}])[0]
    store.hide(record, reason="retracted by the author", actor="owner")

    assert blobs.list_for(record) == []
    assert blobs.summarize(record)["bytes"] == len(PDF), "the bytes are still there"
    with pytest.raises(BlobError, match="not retrievable"):
        blobs.read(written.id)


def test_an_unknown_attachment_id_is_not_the_same_answer(store, blobs):
    with pytest.raises(BlobError, match="unknown attachment"):
        blobs.read("blb_" + "0" * 32)


def test_forgetting_destroys_the_bytes_and_not_the_copy_still_cited(store, blobs, record):
    other = store.commit(envelope(source_id="msg-2", text="Fwd: the same invoice"))["id"]
    shared = attach(store, blobs, record,
                    [{"data": PDF, "filename": "invoice.pdf", "mime": "application/pdf"}])[0]
    kept = attach(store, blobs, other,
                  [{"data": PDF, "filename": "fwd.pdf", "mime": "application/pdf"}])[0]
    only_here = attach(store, blobs, record,
                       [{"data": b"personal note", "filename": "note.txt",
                         "mime": "text/plain", "position": 1}])[0]

    manager = ErasureManager(store, owner_principal=OWNER)
    preview = manager.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    result = manager.confirm(intent_id=preview["intent_id"],
                             preview_digest=preview["preview_digest"], actor=OWNER)

    assert result["attachments"] == {"attachments": 2, "contents": 1, "bytes": len(PDF) + 13}
    with pytest.raises(BlobError, match="unknown attachment"):
        blobs.read(only_here.id)
    assert blobs.read(kept.id) == PDF, "a shared copy survives its other carrier"
    assert store.db.execute("SELECT refs FROM blob_contents WHERE sha256=?",
                            (shared.sha256,)).fetchone()[0] == 1
    assert counts(store)["attachments"] == 1


def test_the_preview_shows_the_bytes_that_will_go_and_a_new_one_invalidates_it(
        store, blobs, record):
    manager = ErasureManager(store, owner_principal=OWNER)
    first = manager.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    assert first["attachments"] == {"files": 0, "bytes": 0}

    attach(store, blobs, record, [{"data": PDF, "filename": "invoice.pdf",
                                   "mime": "application/pdf"}])

    with pytest.raises(EvidenceError, match="attachments changed"):
        manager.confirm(intent_id=first["intent_id"], preview_digest=first["preview_digest"],
                        actor=OWNER)
    assert blobs.list_for(record), "the refused confirm destroyed nothing"


def test_a_rollback_after_the_blob_write_leaves_no_orphan_bytes(store, blobs, record):
    """The gate case, answered by construction rather than by a sweeper."""
    store.db.execute("BEGIN IMMEDIATE")
    blobs.attach(record, [{"data": PDF, "filename": "invoice.pdf",
                           "mime": "application/pdf"}], db=store.db)
    assert store.db.execute("SELECT count(*) FROM blob_chunks").fetchone()[0] == 1
    store.db.execute("ROLLBACK")

    assert counts(store) == {"blob_contents": 0, "blob_chunks": 0, "attachments": 0}


def test_a_restored_database_can_be_recounted_and_swept(store, blobs, record):
    written = attach(store, blobs, record, [{"data": PDF, "filename": "invoice.pdf",
                                             "mime": "application/pdf"}])[0]
    # Simulate a restore that carried the content forward but lost the citation.
    store.db.execute("BEGIN IMMEDIATE")
    store.db.execute("DELETE FROM attachments WHERE id=?", (written.id,))
    store.db.execute("COMMIT")

    report = blobs.reconcile()

    assert report["contents_collected"] == 1
    assert counts(store) == {"blob_contents": 0, "blob_chunks": 0, "attachments": 0}
    assert blobs.get(written.id) is None


def test_reconcile_never_deletes_bytes_still_under_a_tombstone(store, blobs, record):
    written = attach(store, blobs, record, [{"data": PDF, "filename": "invoice.pdf",
                                             "mime": "application/pdf"}])[0]
    store.hide(record, reason="retracted", actor="owner")

    report = blobs.reconcile()

    assert report["contents_collected"] == 0
    assert store.db.execute("SELECT refs FROM blob_contents").fetchone()[0] == 1
    # Hidden is not forgotten: the bytes are intact for an owner who shows
    # evidence of a decision, and gone the moment the erasure is confirmed.
    with pytest.raises(BlobError, match="not retrievable"):
        blobs.read(written.id)


def test_damaged_bytes_are_reported_rather_than_handed_out(store, blobs, record):
    written = attach(store, blobs, record, [{"data": PDF, "filename": "invoice.pdf",
                                             "mime": "application/pdf"}])[0]
    with sqlite3.connect(store.db.execute("PRAGMA database_list").fetchone()[2]) as raw:
        raw.execute("UPDATE blob_chunks SET data = ? WHERE chunk_index = 0",
                    (b"%PDF-1.4 tampered",))

    with pytest.raises(BlobError, match="damaged"):
        blobs.read(written.id)


def test_listing_never_carries_bytes_into_a_prompt(store, blobs, record):
    attach(store, blobs, record, [{"data": PDF, "filename": "invoice.pdf",
                                   "mime": "application/pdf"}])

    listed = [item.as_dict() for item in blobs.list_for(record)]

    assert set(listed[0]) == {"id", "record_id", "position", "filename", "mime", "size",
                              "sha256", "added_at"}
    assert PDF[:20] not in str(listed).encode()


def test_an_attachment_row_survives_being_reopened_from_disk(tmp_path):
    path = tmp_path / "canonical.db"
    with EvidenceStore(path) as store:
        first = store.commit(envelope(source_id="msg-1"))["id"]
        manager = BlobStore(store)
        store.db.execute("BEGIN IMMEDIATE")
        written = manager.attach(first, [{"data": PDF, "filename": "invoice.pdf",
                                          "mime": "application/pdf"}], db=store.db)
        store.db.execute("COMMIT")

    with EvidenceStore(path) as reopened:
        again = BlobStore(reopened)
        assert isinstance(again.list_for(first)[0], Attachment)
        assert again.read(written[0].id) == PDF
