"""C4 forgetting: preview, owner-only confirmation, obligations and restore safety."""
from __future__ import annotations

import json

import pytest

from hermes_memory.backend.document_map import DocumentMap
from hermes_memory.lifecycle.erasure import AWAITING, COMPLETE, PENDING, ErasureManager
from hermes_memory.storage.evidence import EvidenceError

from conftest import envelope

OWNER = "owner-principal"


@pytest.fixture()
def forget(store):
    return ErasureManager(store, owner_principal=OWNER)


@pytest.fixture()
def projected(store):
    """Evidence that has already been mirrored into the derived backend."""

    def _project(record_id, revision):
        DocumentMap(store).begin(record_id, revision)
    return _project


def committed(store, **overrides):
    return store.commit(envelope(**overrides))["id"]


def attached(store, record_pk, payload=b"the invoice the owner attached\n",
             filename="invoice.pdf"):
    """One attachment written the way a source commit writes it: inside the transaction."""
    from hermes_memory.storage.blobs import BlobStore

    blobs = BlobStore(store)
    store.db.execute("BEGIN IMMEDIATE")
    try:
        blobs.attach(record_pk, [{"data": payload, "filename": filename,
                                  "mime": "application/pdf"}], db=store.db)
        store.db.execute("COMMIT")
    except BaseException:
        store.db.execute("ROLLBACK")
        raise
    return blobs


def text_of(store, match):
    return [item.text for item in store.search(match)]


# -- the bytes an attachment holds -------------------------------------------

def test_a_preview_counts_the_attachment_bytes_it_would_destroy(store, forget):
    """A preview that counted the text and not the file would understate the radius.

    The digest signs the numbers as well as the ids, so an owner confirming a preview is
    confirming the byte total the blob ledger reported — which means that total has to come
    from the ledger rather than from a second query that can drift away from it.
    """
    from hermes_memory.storage.blobs import BlobStore

    record = committed(store, source_id="msg-1")
    payload = b"the invoice the owner attached\n"
    blobs = attached(store, record, payload)
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    assert preview["attachments"] == {"files": 1, "bytes": len(payload)}
    assert blobs.totals_for([record]) == preview["attachments"]


def test_forgetting_a_record_destroys_its_bytes_and_keeps_a_shared_one(store, forget):
    """The same file in two messages is one byte string: forgetting one keeps the other.

    And the reverse has to hold too — a reference count that never reaches zero would leave
    private bytes on disk under a record the owner was told was forgotten.
    """
    from hermes_memory.storage.blobs import BlobStore

    mine = committed(store, source_id="msg-1")
    theirs = committed(store, source_id="msg-2", text="A different message, same file.")
    payload = b"a statement both messages carry\n"
    attached(store, mine, payload)
    attached(store, theirs, payload)
    first = forget.preview(record_ids=[mine], actor=OWNER, reason="withdrawn")
    forget.confirm(intent_id=first["intent_id"], preview_digest=first["preview_digest"],
                   actor=OWNER)
    keeper = BlobStore(store)
    assert keeper.totals_for([theirs]) == {"files": 1, "bytes": len(payload)}
    assert keeper.read(keeper.list_for(theirs)[0].id) == payload, \
        "the surviving message still has the file it also held"

    second = forget.preview(record_ids=[theirs], actor=OWNER, reason="withdrawn too")
    released = forget.confirm(intent_id=second["intent_id"],
                              preview_digest=second["preview_digest"], actor=OWNER)
    assert released["attachments"] == {"attachments": 1, "contents": 1,
                                       "bytes": len(payload)}, \
        "one reference, one content: the byte string is shared, the row is not"
    assert keeper.totals_for([mine, theirs]) == {"files": 0, "bytes": 0}
    assert store.db.execute("SELECT count(*) FROM blob_contents").fetchone()[0] == 0, \
        "the last reference went, so the bytes go with it"


# -- phase one ---------------------------------------------------------------

def test_a_preview_changes_nothing(store, forget):
    record = committed(store, source_id="msg-1")
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="mentioned in error")
    assert preview["records"] == [record]
    assert len(preview["preview_digest"]) == 64
    assert text_of(store, "meeting") == [store.get(record).text], "nothing is hidden yet"
    assert forget.status(preview["intent_id"])["state"] == AWAITING
    assert preview["confirmable_by"] == OWNER


def test_a_preview_names_everything_that_will_disappear(store, forget, projected):
    parent = committed(store, source_id="msg-1")
    projected(parent, "1")
    child = store.commit(envelope(source_id="summary-1", text="Summary of the meeting.",
                                  parent_record_ids=[parent]))["id"]
    grandchild = store.commit(envelope(source_id="lesson-1", text="Meetings move on Thursdays.",
                                        parent_record_ids=[child]))["id"]
    preview = forget.preview(record_ids=[parent], actor=OWNER, reason="withdrawn")
    assert sorted(preview["dependent_artifacts"]) == sorted([child, grandchild])
    kinds = {item["kind"] for item in preview["obligations"]}
    assert kinds == {"backend_document", "backend_derived"}


def test_an_unknown_or_already_forgotten_id_is_refused_up_front(store, forget):
    record = committed(store, source_id="msg-1")
    with pytest.raises(EvidenceError, match="unknown record"):
        forget.preview(record_ids=["rec_" + "a" * 32], actor=OWNER, reason="oops")
    with pytest.raises(EvidenceError, match="between 1 and 500"):
        forget.preview(record_ids=[], actor=OWNER, reason="nothing")
    confirm = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    forget.confirm(intent_id=confirm["intent_id"], preview_digest=confirm["preview_digest"],
                   actor=OWNER)
    with pytest.raises(EvidenceError, match="already forgotten"):
        forget.preview(record_ids=[record], actor=OWNER, reason="again")


# -- phase two ---------------------------------------------------------------

def test_confirmation_needs_the_owner_principal_configured_at_all(store, forget):
    record = committed(store, source_id="msg-1")
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    unconfigured = ErasureManager(store, owner_principal=None)
    with pytest.raises(EvidenceError, match="no owner principal"):
        unconfigured.confirm(intent_id=preview["intent_id"],
                             preview_digest=preview["preview_digest"], actor=OWNER)
    assert store.live_and_visible(record)


def test_an_agent_principal_cannot_confirm(store, forget):
    """The agent may ask; it may not decide."""
    record = committed(store, source_id="msg-1")
    preview = forget.preview(record_ids=[record], actor="hermes-agent", reason="asked",
                             actor_kind="agent")
    with pytest.raises(EvidenceError, match="owner principal"):
        forget.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                       actor="hermes-agent")
    assert store.live_and_visible(record)


def test_confirming_a_different_digest_than_was_shown_is_refused(store, forget):
    record = committed(store, source_id="msg-1")
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    with pytest.raises(EvidenceError, match="digest does not match"):
        forget.confirm(intent_id=preview["intent_id"], preview_digest="0" * 64, actor=OWNER)
    assert store.live_and_visible(record)


def test_confirmation_hides_and_tombstones_but_does_not_destroy_the_revision(store, forget):
    record = committed(store, source_id="msg-1")
    original = store.get(record, include_hidden=True).text
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    forget.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                   actor=OWNER)
    assert store.get(record) is None
    assert text_of(store, "meeting") == []
    assert forget.forgotten(record) is True
    # The row still exists for the erasure audit; it is flagged, not blanked.
    row = store.db.execute("SELECT text, deleted FROM records WHERE id=?", (record,)).fetchone()
    assert row["deleted"] == 1 and row["text"] == original


def test_dependents_are_invalidated_not_deleted(store, forget):
    parent = committed(store, source_id="msg-1")
    child = store.commit(envelope(source_id="summary-1", text="Summary of the meeting.",
                                  parent_record_ids=[parent]))["id"]
    preview = forget.preview(record_ids=[parent], actor=OWNER, reason="withdrawn")
    forget.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                   actor=OWNER)
    assert store.get(child) is None, "a summary must not outlive its evidence"
    kept = store.db.execute("SELECT deleted FROM records WHERE id=?", (child,)).fetchone()
    assert kept["deleted"] == 0, "invalidation is not forgetting"


def test_an_intent_is_replayable_through_the_journal(store, forget):
    record = committed(store, source_id="msg-1")
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    forget.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                   actor=OWNER)
    changes = store.db.execute(
        "SELECT change FROM change_journal WHERE record_id=?", (record,)).fetchall()
    assert [row["change"] for row in changes] == ["add", "erase"]


# -- obligations -------------------------------------------------------------

def test_an_unreachable_backend_leaves_the_intent_pending_not_complete(store, forget, projected):
    record = committed(store, source_id="msg-1")
    projected(record, "1")
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    outcome = forget.confirm(intent_id=preview["intent_id"],
                             preview_digest=preview["preview_digest"], actor=OWNER)
    assert outcome["state"] == PENDING
    assert outcome["obligations_outstanding"] == 2
    assert forget.status(preview["intent_id"])["state"] == PENDING


def test_verification_of_every_obligation_is_what_completes_an_intent(store, forget, projected):
    record = committed(store, source_id="msg-1")
    projected(record, "1")
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    forget.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                   actor=OWNER)
    outstanding = forget.pending()
    assert {item["kind"] for item in outstanding} == {"backend_document", "backend_derived"}
    for item in outstanding[:-1]:
        forget.verify(intent_id=item["intent_id"], kind=item["kind"], reference=item["reference"])
        assert forget.status(item["intent_id"])["state"] == PENDING
    last = outstanding[-1]
    done = forget.verify(intent_id=last["intent_id"], kind=last["kind"], reference=last["reference"])
    assert done["state"] == COMPLETE
    assert forget.pending() == []


def test_a_verified_document_stops_being_counted_as_coverage(store, forget, projected):
    """The mapping ledger is what the status report reads, so it has to settle too.

    An obligation marked verified while its document row still says ``queued`` would have
    the installation keep reporting derived coverage it has just proved was deleted.
    """
    record = committed(store, source_id="msg-1")
    projected(record, "1")
    document = store.db.execute("SELECT document_id FROM backend_documents").fetchone()[0]
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    forget.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                   actor=OWNER)
    item = [entry for entry in forget.pending() if entry["kind"] == "backend_document"][0]
    assert item["reference"] == f"hindsight:hermes:{document}", \
        "the obligation names the backend's own address for the document"
    before = dict(store.db.execute("SELECT state, count(*) FROM backend_documents "
                                   "GROUP BY state").fetchall())
    assert before == {"queued": 1}

    forget.verify(intent_id=item["intent_id"], kind=item["kind"], reference=item["reference"])

    rows = store.db.execute("SELECT state, document_id FROM backend_documents").fetchall()
    assert [(row["state"], row["document_id"]) for row in rows] == [("absent", document)], \
        "the one mapping that was there is marked gone, not replaced by a second row"
    assert store.db.in_transaction is False, "the settle and the verification are one fact"


def test_a_failed_cleanup_attempt_counts_up_and_stays_pending(store, forget, projected):
    record = committed(store, source_id="msg-1")
    projected(record, "1")
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    forget.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                   actor=OWNER)
    item = forget.pending()[0]
    outcome = forget.fail(intent_id=item["intent_id"], kind=item["kind"],
                          reference=item["reference"], error="backend offline")
    assert outcome["state"] == PENDING
    matched = [row for row in forget.status(item["intent_id"])["obligations"]
               if row["reference"] == item["reference"]]
    assert matched[0]["attempts"] == 1 and matched[0]["state"] == "pending"
    assert forget.pending() != []


def test_confirming_twice_does_not_re_run_the_erasure(store, forget):
    record = committed(store, source_id="msg-1")
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    again = forget.confirm(intent_id=preview["intent_id"],
                           preview_digest=preview["preview_digest"], actor=OWNER)
    assert again["state"] == COMPLETE
    with pytest.raises(EvidenceError, match="already"):
        forget.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                       actor=OWNER)


# -- restore safety ----------------------------------------------------------

def test_a_merged_ledger_keeps_the_original_request_date(store, forget):
    """The retired framework rewrote requested_at to the merge time."""
    record = committed(store, source_id="msg-1")
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    forget.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                   actor=OWNER)
    before = forget.status(preview["intent_id"])["requested_at"]
    outcome = forget.absorb([{
        "id": preview["intent_id"], "state": PENDING, "source": "gmail",
        "requested_at": before, "requested_by": OWNER,
        "tombstones": {record: "0" * 64},
    }])
    after = forget.status(preview["intent_id"])
    assert outcome["merged"] == 1
    assert after["requested_at"] == before
    assert after["state"] == COMPLETE, "a completed local intent is not reopened by a merge"


def test_absorb_imports_an_intent_that_local_history_lost(store, forget):
    record = committed(store, source_id="msg-1")
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    forget.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                   actor=OWNER)
    stored = forget.status(preview["intent_id"])
    # Simulate a node whose local ledger was lost: the dependents go first, or
    # the tombstone->ledger foreign key (which is what keeps the ledger
    # unprunable) correctly refuses the deletion.
    store.db.execute("DELETE FROM tombstones")
    store.db.execute("DELETE FROM erasure_ledger")
    store.db.execute("UPDATE records SET deleted=0 WHERE id=?", (record,))
    outcome = forget.absorb([{
        "id": stored["id"], "state": stored["state"], "source": "gmail",
        "requested_at": stored["requested_at"], "requested_by": stored["requested_by"],
        "reason": "withdrawn", "preview": "{}", "preview_digest": "0" * 64,
        "epoch": stored["epoch"], "tombstones": {record: "0" * 64},
    }])
    assert outcome["inserted"] == 1
    assert forget.reapply()["reapplied"] == 1
    assert store.get(record) is None, "a restore must not resurrect forgotten evidence"


def test_reapply_is_a_no_op_when_nothing_conflicts(store, forget):
    committed(store, source_id="msg-1")
    assert forget.reapply() == {"reapplied": 0}


def test_a_malformed_external_entry_is_refused_rather_than_trusted(store, forget):
    with pytest.raises(EvidenceError, match="erase_ id"):
        forget.absorb([{"id": "intent-1", "state": "complete", "requested_at": "2026-01-01T00:00:00+00:00"}])
    with pytest.raises(EvidenceError, match="unknown erasure state"):
        forget.absorb([{"id": "erase_x", "state": "deleted-i-think",
                        "requested_at": "2026-01-01T00:00:00+00:00"}])
    with pytest.raises(EvidenceError, match="requested_at"):
        forget.absorb([{"id": "erase_x", "state": "complete", "requested_at": "last tuesday"}])
    assert store.db.execute("SELECT count(*) FROM erasure_ledger").fetchone()[0] == 0


# -- transaction hygiene -----------------------------------------------------

def test_no_erasure_path_leaves_a_transaction_open(store, forget, projected):
    record = committed(store, source_id="msg-1")
    projected(record, "1")
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    intent = preview["intent_id"]
    forget.confirm(intent_id=intent, preview_digest=preview["preview_digest"], actor=OWNER)
    item = forget.pending()[0]
    steps = [
        lambda: forget.fail(intent_id=intent, kind=item["kind"], reference=item["reference"],
                            error="offline"),
        lambda: forget.verify(intent_id=intent, kind=item["kind"], reference=item["reference"]),
        lambda: forget.pending(),
        lambda: forget.status(intent),
        lambda: forget.reapply(),
    ]
    for step in steps:
        step()
        assert store.db.in_transaction is False
    with pytest.raises(EvidenceError):
        forget.verify(intent_id="erase_missing", kind="blob", reference="x")
    assert store.db.in_transaction is False


def test_the_ledger_survives_a_reopen(tmp_path):
    from hermes_memory.storage.evidence import EvidenceStore

    path = tmp_path / "canonical.db"
    with EvidenceStore(path) as store:
        manager = ErasureManager(store, owner_principal=OWNER)
        record = store.commit(envelope(source_id="msg-1"))["id"]
        preview = manager.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
        manager.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor=OWNER)
        assert json.dumps(manager.status(preview["intent_id"]))
    with EvidenceStore(path) as reopened:
        manager = ErasureManager(reopened, owner_principal=OWNER)
        assert manager.status(preview["intent_id"])["state"] == COMPLETE
        assert reopened.get(record) is None


# -- reset -------------------------------------------------------------------

@pytest.fixture()
def reset(store):
    from hermes_memory.lifecycle.reset import ResetController

    return ResetController(store, owner_principal=OWNER)


def _seed(store, count=3):
    return [committed(store, source_id=f"msg-{i}", text=f"note number {i}")
            for i in range(count)]


def test_a_reset_preview_names_the_banks_it_will_owe(store, reset, projected):
    records = _seed(store)
    projected(records[0], "1")
    preview = reset.preview(actor=OWNER, reason="leaving this account")
    assert preview["records"] == len(records)
    assert preview["banks_to_clear"] == ["hindsight:hermes"]
    assert store.db.execute("SELECT count(*) FROM records WHERE deleted=0").fetchone()[0] == 3


def test_reset_wipes_local_evidence_and_opens_the_debt_in_one_transaction(store, reset,
                                                                          projected):
    """The legacy crash gap: local rows gone, nobody obliged to finish."""
    records = _seed(store)
    projected(records[0], "1")
    preview = reset.preview(actor=OWNER, reason="leaving this account")
    outcome = reset.confirm(intent_id=preview["intent_id"],
                            preview_digest=preview["preview_digest"], actor=OWNER)
    assert outcome["state"] == PENDING and outcome["obligations_outstanding"] == 1
    assert store.db.execute("SELECT count(*) FROM records WHERE deleted=0").fetchone()[0] == 0
    assert store.db.execute(
        "SELECT count(*) FROM erasure_targets WHERE intent_id=? AND state='pending'",
        (preview["intent_id"],)).fetchone()[0] == 1
    # Payloads are retired, not blanked: the tombstone must still prove *what*
    # was forgotten to the ledger that says when.
    kept = store.db.execute("SELECT text, deleted FROM records WHERE id=?",
                            (records[0],)).fetchone()
    assert kept["deleted"] == 1 and kept["text"]


def test_reset_invalidates_every_lease_and_restartpoint(store, reset):
    from hermes_memory.sources.sync import StaleFence, SyncController

    _seed(store)
    sync = SyncController(store)
    sync.register("gmail", policy_version="private-api")
    fence = sync.acquire("gmail", holder="connector-a", ttl=3600)
    sync.publish(fence, "page-1", [envelope(source_id="msg-9")], next_cursor="tok-2")

    preview = reset.preview(actor=OWNER, reason="trust revoked")
    outcome = reset.confirm(intent_id=preview["intent_id"],
                            preview_digest=preview["preview_digest"], actor=OWNER)
    assert outcome["epoch"] == fence.epoch + 1
    with pytest.raises(StaleFence, match="epoch advanced"):
        sync.publish(fence, "page-2", [envelope(source_id="msg-10")])
    state = sync.state("gmail")
    assert state["cursor"] is None and state["lease_active"] is False


def test_confirming_a_stale_reset_preview_is_refused(store, reset):
    _seed(store)
    preview = reset.preview(actor=OWNER, reason="leaving")
    committed(store, source_id="arrived-after-the-look")
    with pytest.raises(EvidenceError, match="store changed after this preview"):
        reset.confirm(intent_id=preview["intent_id"],
                      preview_digest=preview["preview_digest"], actor=OWNER)
    assert store.db.execute("SELECT count(*) FROM records WHERE deleted=0").fetchone()[0] == 4


def test_a_reset_needs_an_owner_and_cannot_borrow_an_erasure_intent(store, forget, reset):
    _seed(store)
    unowned = type(reset)(store, owner_principal=None)
    preview = reset.preview(actor=OWNER, reason="leaving")
    with pytest.raises(EvidenceError, match="no owner principal"):
        unowned.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor=OWNER)
    erasure = forget.preview(record_ids=[store.db.execute(
        "SELECT id FROM records WHERE deleted=0 LIMIT 1").fetchone()[0]],
        actor=OWNER, reason="one item only")
    with pytest.raises(EvidenceError, match="not a reset"):
        reset.confirm(intent_id=erasure["intent_id"],
                      preview_digest=erasure["preview_digest"], actor=OWNER)


def test_an_empty_store_resets_straight_to_complete(store, reset):
    preview = reset.preview(actor=OWNER, reason="fresh start")
    assert preview["banks_to_clear"] == []
    outcome = reset.confirm(intent_id=preview["intent_id"],
                            preview_digest=preview["preview_digest"], actor=OWNER)
    assert outcome["state"] == COMPLETE and outcome["records_retired"] == 0
    assert outcome["obligations_outstanding"] == 0


def test_verifying_the_bank_clear_completes_the_reset(store, reset, projected):
    records = _seed(store, count=1)
    projected(records[0], "1")
    preview = reset.preview(actor=OWNER, reason="leaving")
    reset.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                  actor=OWNER)
    obligation = forget_targets(store, preview["intent_id"])[0]
    done = reset.verify(intent_id=preview["intent_id"], kind=obligation["kind"],
                        reference=obligation["reference"]) if hasattr(reset, "verify") else None
    assert done is None, "bank verification belongs to the erasure manager, not the reset"
    manager = ErasureManager(store, owner_principal=OWNER)
    settled = manager.verify(intent_id=preview["intent_id"], kind=obligation["kind"],
                             reference=obligation["reference"])
    assert settled["state"] == COMPLETE


def forget_targets(store, intent_id):
    return [dict(row) for row in store.db.execute(
        "SELECT kind, reference FROM erasure_targets WHERE intent_id=?", (intent_id,))]


def test_an_erasure_preview_goes_stale_if_derived_copies_appear(store, forget, projected):
    """The debt grows when a projection lands, so the shown digest must not pass."""
    record = committed(store, source_id="msg-1")
    preview = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    assert preview["obligations"] == []
    projected(record, "1")
    with pytest.raises(EvidenceError, match="changed after this preview"):
        forget.confirm(intent_id=preview["intent_id"],
                       preview_digest=preview["preview_digest"], actor=OWNER)
    assert store.live_and_visible(record)
    fresh = forget.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
    assert {item["kind"] for item in fresh["obligations"]} == {"backend_document",
                                                               "backend_derived"}
    outcome = forget.confirm(intent_id=fresh["intent_id"],
                             preview_digest=fresh["preview_digest"], actor=OWNER)
    assert outcome["state"] == PENDING
