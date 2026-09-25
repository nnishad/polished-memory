"""C5 document map: canonical revision to backend document, with durable intent.

The mapping is recorded *before* anything is sent, along with the submission
identity, so a process that dies mid-request leaves a row that says what was
attempted and lets a later run reconcile it through the operations API. A
mapping written only after a successful response would forget every request
whose acknowledgement was lost — which is precisely the case that matters.

Document IDs come from ``ids.backend_document_id``, which guarantees no ``_``
or ``~``: Hindsight escapes those when composing chunk IDs, and an ambiguous
chunk ID cannot be resolved back to the revision that produced it.
"""
from __future__ import annotations

import uuid
from typing import Any

from ..ids import backend_document_id, now
from ..storage.evidence import EvidenceError

__all__ = ["DocumentMap", "QUEUED", "SUBMITTED", "VERIFIED", "FAILED", "ABSENT"]

QUEUED = "queued"        # intent recorded, nothing sent yet
SUBMITTED = "submitted"  # request went out; outcome unconfirmed
VERIFIED = "verified"    # backend confirms this revision is present
FAILED = "failed"        # backend reported it could not
ABSENT = "absent"        # erased and verified gone


class DocumentMap:
    def __init__(self, store, *, backend: str = "hindsight", bank_id: str = "hermes"):
        self.store = store
        self.db = store.db
        self.backend = backend
        self.bank_id = bank_id

    def document_id(self, record_id: str, revision: str) -> str:
        return backend_document_id(record_id, revision)

    def begin(self, record_id: str, revision: str, *, async_submission: bool = False) -> dict:
        """Record intent and hand back the identity to submit under.

        ``submission_id`` is only meaningful for an async retain; a synchronous
        call is its own confirmation, so minting an id for it would suggest a
        reconciliation path that does not exist.
        """
        if not self.store.get(record_id, include_hidden=True):
            raise EvidenceError(f"cannot project unknown record {record_id!r}")
        document_id = self.document_id(record_id, revision)
        submission_id = str(uuid.uuid4()) if async_submission else None
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute(
                "SELECT state, operation_id FROM backend_documents "
                "WHERE record_id=? AND revision=? AND backend=? AND bank_id=?",
                (record_id, revision, self.backend, self.bank_id)).fetchone()
            if existing and existing["state"] == VERIFIED:
                self.db.execute("COMMIT")
                return {"document_id": document_id, "submission_id": None,
                        "state": VERIFIED, "already_projected": True}
            if existing:
                if existing["operation_id"]:
                    # Reuse the identity already on record: minting a new one
                    # would abandon the operation the backend may be running.
                    submission_id = existing["operation_id"]
                self.db.execute(
                    "UPDATE backend_documents SET state=?, operation_id=?, error=NULL "
                    "WHERE record_id=? AND revision=? AND backend=? AND bank_id=?",
                    (SUBMITTED if submission_id else QUEUED, submission_id,
                     record_id, revision, self.backend, self.bank_id))
            else:
                self.db.execute(
                    "INSERT INTO backend_documents(record_id, revision, backend, bank_id, "
                    "document_id, desired_epoch, state, operation_id, confirmed_at) "
                    "VALUES(?,?,?,?,?,?,?,?,NULL)",
                    (record_id, revision, self.backend, self.bank_id, document_id,
                     self.store.epoch(), SUBMITTED if submission_id else QUEUED, submission_id))
            self.store._audit("projection_begin", record_id,
                              {"revision": revision, "state": self.state(record_id, revision),
                               "document_id": document_id})
            outcome = self.state(record_id, revision)
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"document_id": document_id, "submission_id": submission_id, "state": outcome,
                "already_projected": False}

    def state(self, record_id: str, revision: str) -> str:
        row = self.db.execute(
            "SELECT state FROM backend_documents WHERE record_id=? AND revision=? AND backend=?"
            " AND bank_id=?", (record_id, revision, self.backend, self.bank_id)).fetchone()
        return row["state"] if row else "unmapped"

    def confirm(self, record_id: str, revision: str) -> None:
        self._set(record_id, revision, VERIFIED, error=None)

    def mark_absent(self, record_id: str, revision: str) -> None:
        self._set(record_id, revision, ABSENT, error=None)

    def fail(self, record_id: str, revision: str, *, error: str) -> None:
        self._set(record_id, revision, FAILED, error=error[:500])

    def record_submission(self, record_id: str, revision: str, submission_id: str) -> None:
        self._set(record_id, revision, SUBMITTED, operation_id=submission_id)

    def _set(self, record_id: str, revision: str, state: str, *, error: str | None,
             operation_id: str | None = None) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.db.execute(
                "UPDATE backend_documents SET state=?, error=?, "
                "operation_id=COALESCE(?, operation_id), confirmed_at=? "
                "WHERE record_id=? AND revision=? AND backend=? AND bank_id=?",
                (state, error, operation_id, now() if state == VERIFIED else None,
                 record_id, revision, self.backend, self.bank_id))
            if not cursor.rowcount:
                raise EvidenceError(f"no mapping for {record_id}@{revision}")
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    # -- reconciliation ------------------------------------------------------

    def outstanding(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Projections that never reached a terminal state, oldest first."""
        rows = self.db.execute(
            "SELECT record_id, revision, document_id, state, operation_id, error "
            "FROM backend_documents WHERE backend=? AND bank_id=? "
            "AND state IN ('queued','submitted','failed') ORDER BY rowid LIMIT ?",
            (self.backend, self.bank_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def reconcile(self, *, client, limit: int = 20) -> dict[str, int]:
        """Ask the backend what happened to submissions we cannot account for.

        This is the recovery path for a lost acknowledgement: the operation may
        have completed, failed, or never arrived, and only the backend knows.
        Guessing either way produces a store that disagrees with itself.
        """
        settled = verified = still_pending = unknown = 0
        for row in self.outstanding(limit=limit):
            submission = row["operation_id"]
            if not submission:
                still_pending += 1
                continue
            try:
                operation = client.operation(submission)
            except Exception as error:  # unreachable backend is not an outcome
                unknown += 1
                self.db.execute(
                    "UPDATE backend_documents SET error=? WHERE record_id=? AND revision=?"
                    " AND backend=? AND bank_id=?",
                    (str(error)[:500], row["record_id"], row["revision"], self.backend,
                     self.bank_id))
                continue
            state = str(operation.get("state") or operation.get("status") or "")
            settled += 1
            if state in {"completed", "succeeded"}:
                self.confirm(row["record_id"], row["revision"])
                verified += 1
            elif state in {"failed", "cancelled"}:
                self.fail(row["record_id"], row["revision"],
                          error=f"backend operation {state}")
            else:
                still_pending += 1
        return {"settled": settled, "verified": verified, "pending": still_pending,
                "unreachable": unknown}

    def as_dict(self) -> dict[str, Any]:
        rows = self.db.execute(
            "SELECT state, count(*) AS n FROM backend_documents WHERE backend=? AND bank_id=?"
            " GROUP BY state", (self.backend, self.bank_id)).fetchall()
        return {row["state"]: row["n"] for row in rows}
