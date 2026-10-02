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

from ..ids import backend_document_id, digest, now
from ..storage.evidence import EvidenceError
from .capabilities import (OPERATION_ABANDONED, OPERATION_DONE, OPERATION_RUNNING,
                           OPERATION_STOPPED)
from .hindsight_client import operation_reason, operation_state

__all__ = ["DocumentMap", "QUEUED", "SUBMITTED", "VERIFIED", "FAILED", "ABSENT"]

QUEUED = "queued"        # intent recorded, nothing sent yet
SUBMITTED = "submitted"  # request went out; outcome unconfirmed
VERIFIED = "verified"    # backend confirms this revision is present
FAILED = "failed"        # backend reported it could not
ABSENT = "absent"        # erased and verified gone


class DocumentMap:
    def __init__(self, store, *, backend: str = "hindsight", bank_id: str = "hermes",
                 generation_id: str | None = None):
        self.store = store
        self.db = store.db
        self.backend = backend
        self.bank_id = bank_id
        self.generation_id = generation_id
        if generation_id is not None:
            row = self.db.execute("SELECT bank_id,backend,epoch,state FROM projection_generations "
                                  "WHERE id=?", (generation_id,)).fetchone()
            if (row is None or row["bank_id"] != bank_id or row["backend"] != backend
                    or row["epoch"] != self.store.epoch() or row["state"] == "retired"):
                raise EvidenceError("mapping generation is not a current writable bank binding")

    def document_id(self, record_id: str, revision: str) -> str:
        return backend_document_id(record_id, revision)

    def _writable_generation(self):
        if self.generation_id is None:
            return
        row = self.db.execute("SELECT epoch,state FROM projection_generations WHERE id=?",
                              (self.generation_id,)).fetchone()
        if row is None or row["epoch"] != self.store.epoch() or row["state"] == "retired":
            raise EvidenceError("projection generation is stale or retired")

    def begin(self, record_id: str, revision: str, *, async_submission: bool = False,
              context_version: int = 2) -> dict:
        """Record intent and hand back the identity to submit under.

        ``submission_id`` is only meaningful for an async retain; a synchronous
        call is its own confirmation, so minting an id for it would suggest a
        reconciliation path that does not exist.
        """
        record = self.store.get(record_id, include_hidden=True)
        if not record:
            raise EvidenceError(f"cannot project unknown record {record_id!r}")
        if record.revision != revision:
            raise EvidenceError("projection revision differs from the canonical input revision")
        if type(context_version) is not int or context_version not in {1, 2}:
            raise EvidenceError("unsupported retention context contract")
        document_id = self.document_id(record_id, revision)
        submission_id = str(uuid.uuid4()) if async_submission else None
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._writable_generation()
            existing = self.db.execute(
                "SELECT state, operation_id, retain_context_version,desired_epoch,generation_id "
                "FROM backend_documents "
                "WHERE record_id=? AND revision=? AND backend=? AND bank_id=?",
                (record_id, revision, self.backend, self.bank_id)).fetchone()
            if existing and (existing["desired_epoch"] != self.store.epoch()
                             or (self.generation_id is not None
                                 and existing["generation_id"] != self.generation_id)):
                raise EvidenceError("mapping has a stale epoch/generation; rebuild in a shadow bank")
            if existing and existing["state"] == VERIFIED:
                self.db.execute("COMMIT")
                return {"document_id": document_id, "submission_id": None,
                        "state": VERIFIED, "already_projected": True}
            if existing:
                context_version = existing["retain_context_version"]
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
                    "document_id, desired_epoch, state, operation_id, confirmed_at, retain_context_version,"
                    "generation_id,input_manifest) VALUES(?,?,?,?,?,?,?,?,NULL,?,?,?)",
                    (record_id, revision, self.backend, self.bank_id, document_id,
                     self.store.epoch(), SUBMITTED if submission_id else QUEUED, submission_id,
                     context_version, self.generation_id, self._input_manifest(record_id, revision)))
            self.store._audit("projection_begin", record_id,
                              {"revision": revision, "state": self.state(record_id, revision),
                               "document_id": document_id})
            outcome = self.state(record_id, revision)
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"document_id": document_id, "submission_id": submission_id, "state": outcome,
                "already_projected": False, "context_version": context_version}

    def _input_manifest(self, record_id, revision):
        import json
        from .generations import input_manifest
        return json.dumps(input_manifest(self.store, record_id, revision), sort_keys=True)

    def pin_payload(self, record_id: str, revision: str, payload: dict) -> None:
        """A retry cannot reuse an operation UUID for different retained bytes.

        Persist only the digest: canonical text is immutable, and the versioned
        formatter reconstructs it without storing another private-text copy.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._writable_generation()
            row = self.db.execute(
                "SELECT retain_payload_digest, document_id, desired_epoch FROM backend_documents "
                "WHERE record_id=? AND revision=? AND backend=? AND bank_id=?",
                (record_id, revision, self.backend, self.bank_id)).fetchone()
            if (row is None or not self.store.live_and_visible(record_id)
                    or row["desired_epoch"] != self.store.epoch()
                    or payload.get("document_id") != row["document_id"]):
                raise EvidenceError("retention payload has no current live mapping")
            fingerprint = digest(payload)
            if row["retain_payload_digest"] and row["retain_payload_digest"] != fingerprint:
                raise EvidenceError("retention payload changed; reconcile the existing operation")
            if not row["retain_payload_digest"]:
                self.db.execute(
                    "UPDATE backend_documents SET retain_payload_digest=? "
                    "WHERE record_id=? AND revision=? AND backend=? AND bank_id=?",
                    (fingerprint, record_id, revision, self.backend, self.bank_id))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def state(self, record_id: str, revision: str) -> str:
        row = self.db.execute(
            "SELECT state FROM backend_documents WHERE record_id=? AND revision=? AND backend=?"
            " AND bank_id=?", (record_id, revision, self.backend, self.bank_id)).fetchone()
        return row["state"] if row else "unmapped"

    def confirm(self, record_id: str, revision: str) -> None:
        self._set(record_id, revision, VERIFIED, error=None)

    def mark_absent(self, record_id: str, revision: str, *, db=None) -> None:
        """The backend says this document is gone, so stop counting it as coverage.

        Pass *db* to write inside a transaction the caller already holds, which is how an
        erasure verification and the mapping it settles become one atomic fact.
        """
        self._set(record_id, revision, ABSENT, error=None, db=db)

    def fail(self, record_id: str, revision: str, *, error: str) -> None:
        self._set(record_id, revision, FAILED, error=error[:500])

    def _set(self, record_id: str, revision: str, state: str, *, error: str | None,
             operation_id: str | None = None, db=None) -> None:
        # Confirmation cannot promote a pre-reset submission into a new epoch.
        # Acknowledgements describe the intent persisted before that request.
        epoch = self.store.epoch() if state == VERIFIED else None
        if db is not None and not db.in_transaction:
            raise EvidenceError("a mapping written inside a transaction needs one open")
        connection = db if db is not None else self.db
        owns = db is None
        if owns:
            connection.execute("BEGIN IMMEDIATE")
        try:
            if state == VERIFIED:
                mapping = connection.execute(
                    "SELECT desired_epoch FROM backend_documents WHERE record_id=? AND revision=? "
                    "AND backend=? AND bank_id=?", (record_id, revision, self.backend,
                                                  self.bank_id)).fetchone()
                record = self.store.get(record_id)
                if (mapping is None or mapping["desired_epoch"] != self.store.epoch()
                        or record is None or record.revision != revision
                        or not self.store.live_and_visible(record_id)):
                    raise EvidenceError("cannot confirm stale or withdrawn projection input")
            cursor = connection.execute(
                "UPDATE backend_documents SET state=?, error=?, "
                "desired_epoch=COALESCE(?, desired_epoch), "
                "operation_id=COALESCE(?, operation_id), confirmed_at=? "
                "WHERE record_id=? AND revision=? AND backend=? AND bank_id=?",
                (state, error, epoch, operation_id,
                 now() if state == VERIFIED else None,
                 record_id, revision, self.backend, self.bank_id))
            if not cursor.rowcount:
                raise EvidenceError(f"no mapping for {record_id}@{revision}")
            if owns:
                connection.execute("COMMIT")
        except BaseException:
            if owns:
                connection.execute("ROLLBACK")
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
        settled = verified = still_pending = unknown = absent = 0
        stopped: list[str] = []
        for row in self.outstanding(limit=limit):
            submission = row["operation_id"]
            if not submission:
                # No operation id means a synchronous retain whose answer never reached
                # us. The document itself can still be asked about, and only the backend
                # can say whether it is there.
                answer = self._probe(client, row["document_id"])
                state = str(answer.get("state") or "")
                if state in {"present", "absent"}:
                    settled += 1
                    if state == "present":
                        try:
                            self.confirm(row["record_id"], row["revision"])
                            verified += 1
                        except EvidenceError:
                            self.fail(row["record_id"], row["revision"],
                                      error="present document has stale or withdrawn canonical input")
                    else:
                        self.mark_absent(row["record_id"], row["revision"])
                        absent += 1
                else:
                    unknown += 1
                    self._note(row, answer.get("reason") or "the backend could not be asked")
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
            state = operation_state(operation)
            settled += 1
            if state in OPERATION_DONE:
                try:
                    self.confirm(row["record_id"], row["revision"])
                    verified += 1
                except EvidenceError:
                    self.fail(row["record_id"], row["revision"],
                              error="completed operation has stale or withdrawn canonical input")
            elif state in OPERATION_ABANDONED | OPERATION_STOPPED:
                # `not_found` belongs here: an operation the backend has no record of is a
                # question that can never be answered, and leaving the row open would ask it
                # every pass forever while the coverage claim sat in limbo. Closing it as
                # failed makes the record unprojected again, so the next pass forms it
                # deliberately instead of waiting for a reply that will not come.
                reason = operation_reason(operation)
                self.fail(row["record_id"], row["revision"],
                          error=f"backend operation {state}"
                          + (f": {reason[:300]}" if reason else ""))
                if state in OPERATION_STOPPED:
                    # Named here and nowhere else: a stop and a failure leave the same row
                    # behind, and the projection ledger cannot afterwards tell them apart. The
                    # difference matters to the queue, where one is owed again and the other was
                    # ended by a person.
                    stopped.append(str(submission))
            elif state in OPERATION_RUNNING:
                still_pending += 1
            else:
                still_pending += 1
                self._note(row, f"the backend answered {state!r} about the operation, which "
                                "is neither finished nor one of the states it documents")
        return {"settled": settled, "verified": verified, "pending": still_pending,
                "absent": absent, "unreachable": unknown,
                # The identities somebody stopped. Every other verdict this pass collected is
                # already in the projection ledger and is read from there, so the queue sees
                # the older answers as plainly as the fresh ones.
                "stopped_operations": stopped}

    def answers(self, *, limit: int = 200) -> dict[str, list[str]]:
        """Every answer the backend has given about a submission, whenever it gave it.

        ``reconcile`` reports what it asked about today, and a queue row has no interest in
        when its answer arrived. A job left ``uncertain`` by an older run of this door carries
        a submission whose projection row is already settled, and ``outstanding`` no longer
        asks about those — so reporting only today's collection would leave the remedy the
        doctor names unable to reach the backlog it names.
        """
        established: list[str] = []
        did_not_land: list[str] = []
        forgotten: list[str] = []
        rows = self.db.execute(
            "SELECT record_id, document_id, operation_id, state FROM backend_documents "
            "WHERE backend=? AND bank_id=? AND state IN (?,?,?) ORDER BY rowid LIMIT ?",
            (self.backend, self.bank_id, VERIFIED, FAILED, ABSENT, limit)).fetchall()
        for row in rows:
            # The identity the queue carries: the operation id the backend named for an async
            # retain, or the document id a synchronous one agreed in advance and never got.
            identity = str(row["operation_id"] or row["document_id"] or "")
            if not identity:
                continue
            if row["state"] == VERIFIED:
                established.append(identity)
            elif self._tombstoned(str(row["record_id"])):
                forgotten.append(identity)
            else:
                did_not_land.append(identity)
        return {"established": established, "did_not_land": did_not_land,
                "forgotten": forgotten}

    def _tombstoned(self, record_id: str) -> bool:
        """Whether the owner forgot this record — the only fact that tells the two absences apart.

        A projection can be missing because a submission never arrived, or because the erasure
        path removed it on purpose in the same transaction as the tombstone. The first wants to
        be tried again; the second must never be re-driven, because that would put text the
        owner deleted back into the engine.
        """
        return bool(self.db.execute("SELECT 1 FROM tombstones WHERE record_id=?",
                                    (record_id,)).fetchone())

    def _probe(self, client, document_id: str) -> dict[str, Any]:
        """What the backend says about one document, with the reason when it cannot say.

        ``unknown`` and ``absent`` are deliberately different answers: the first leaves the
        mapping outstanding, the second is a fact about the derived copy that has to be
        recorded rather than retried forever.
        """
        if not hasattr(client, "document_state"):
            return {"state": "unknown", "reason": "this backend answers no document probe"}
        try:
            answer = client.document_state(document_id)
        except Exception as error:  # noqa: BLE001 - an unreachable backend is not an outcome
            return {"state": "unknown", "reason": str(error)[:500]}
        state = str(answer.get("state") or "")
        if state not in {"present", "absent", "unknown"}:
            return {"state": "unknown",
                    "reason": f"an answer nobody recognises: {state[:80]}"}
        return {"state": state, "reason": (str(answer.get("reason") or "")[:500] or None)}

    def _note(self, row: dict[str, Any], reason: str) -> None:
        """Record why a question went unanswered, without changing what is claimed."""
        self.db.execute(
            "UPDATE backend_documents SET error=? WHERE record_id=? AND revision=?"
            " AND backend=? AND bank_id=?",
            (str(reason)[:500], row["record_id"], row["revision"], self.backend,
             self.bank_id))

    def as_dict(self) -> dict[str, Any]:
        rows = self.db.execute(
            "SELECT state, count(*) AS n FROM backend_documents WHERE backend=? AND bank_id=?"
            " GROUP BY state", (self.backend, self.bank_id)).fetchall()
        return {row["state"]: row["n"] for row in rows}
