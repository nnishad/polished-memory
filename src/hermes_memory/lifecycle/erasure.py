"""C4 — forgetting: preview, owner confirmation, tombstone and derived cleanup."""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from ..ids import digest, new_id, now, timestamp
from ..storage.blobs import BlobStore
from ..storage.evidence import EvidenceError, journal
from ..storage.lineage import Lineage

__all__ = ["ErasureManager", "AWAITING", "PENDING", "COMPLETE"]

AWAITING = "awaiting_confirmation"
FENCED = "local_fence_set"
PENDING = "erasure_pending"
COMPLETE = "complete"


class ErasureManager:
    """Two-phase forgetting with a durable obligation list.

    The preview is persisted rather than handed back as a throwaway hash, so an
    owner can confirm it after a restart and an auditor can see what was shown
    to them. Confirmation is refused without the exact digest: 'yes, delete the
    invoice stuff' is not a boundary anyone can check afterwards.
    """

    def __init__(self, store, *, owner_principal: str | None = None,
                 backend: str = "hindsight"):
        self.store = store
        self.db = store.db
        self.owner_principal = owner_principal
        self.backend = backend
        self.blobs = BlobStore(store)
        self.lineage = Lineage(store)

    # -- phase one: preview --------------------------------------------------

    def preview(self, *, record_ids: Sequence[str], actor: str, reason: str,
                actor_kind: str = "owner") -> dict[str, Any]:
        """Show the blast radius and open an intent. Writes no evidence changes."""
        targets = self._resolve(record_ids)
        if not targets:
            raise EvidenceError("nothing to forget: no live evidence matched")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise EvidenceError("reason must be nonempty text of at most 1000 characters")
        rows, dependents, obligations, attachments, artifacts, fingerprint = self._radius(
            [row["id"] for row in targets])
        intent_id = new_id("erase")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "INSERT INTO erasure_ledger(id, source, requested_at, requested_by, "
                "requester_kind, reason, preview, preview_digest, state, epoch) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (intent_id, ",".join(sorted({row["source"] for row in rows}))[:500],
                 now(), actor, actor_kind, reason,
                 json.dumps({"records": [row["id"] for row in rows],
                             "dependents": dependents,
                             "obligations": obligations,
                             "artifacts": artifacts,
                             "attachments": attachments}, sort_keys=True),
                 fingerprint, AWAITING, self.store.epoch()),
            )
            self.store._audit("erasure_preview", intent_id,
                              {"actor": actor, "actor_kind": actor_kind,
                               "records": len(targets), "obligations": len(obligations)})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {
            "intent_id": intent_id,
            "preview_digest": fingerprint,
            "records": [row["id"] for row in rows],
            "dependent_artifacts": dependents,
            "derived_products": artifacts,
            "obligations": obligations,
            "attachments": attachments,
            "confirmable_by": self.owner_principal,
            "note": ("Nothing has been deleted. Confirmation must come from the owner with "
                     "this exact digest."),
        }

    # -- phase two: confirm --------------------------------------------------

    def confirm(self, *, intent_id: str, preview_digest: str, actor: str) -> dict[str, Any]:
        """Apply the fence. Only the owner principal may reach this."""
        if not isinstance(preview_digest, str) or len(preview_digest) != 64:
            raise EvidenceError("preview_digest must be the 64 character digest shown")
        if self.owner_principal is None:
            raise EvidenceError(
                "no owner principal is configured, so forgetting cannot be confirmed; "
                "set HERMES_MEMORY_OWNER_PRINCIPAL"
            )
        if actor != self.owner_principal:
            raise EvidenceError(
                f"confirmation requires the owner principal, not {actor!r}"
            )
        self.db.execute("BEGIN IMMEDIATE")
        try:
            intent = self.db.execute(
                "SELECT * FROM erasure_ledger WHERE id=?", (intent_id,)).fetchone()
            if intent is None:
                raise EvidenceError(f"unknown erasure intent {intent_id!r}")
            if intent["state"] != AWAITING:
                raise EvidenceError(f"intent is already {intent['state']}")
            if intent["preview_digest"] != preview_digest:
                raise EvidenceError(
                    "the preview digest does not match this intent; the evidence set changed "
                    "or a different preview was confirmed"
                )
            payload = json.loads(intent["preview"])
            record_ids = list(payload["records"])
            # Re-measure the blast radius: the caller's digest only proves they
            # saw *a* preview, not that the store still matches it. New evidence
            # or a projection added since the preview changes the debt.
            rows, _dependents, _obligations, _attachments, artifacts, current = \
                self._radius(record_ids)
            if len(rows) != len(record_ids):
                raise EvidenceError(
                    "part of the previewed set no longer exists; re-run the preview")
            if current != intent["preview_digest"]:
                raise EvidenceError(
                    "the evidence set, its derived copies or its attachments changed after "
                    "this preview; confirm the new preview instead")

            destroyed = self._erase(record_ids, intent_id, payload["dependents"])
            for item in payload["obligations"]:
                self.db.execute(
                    "INSERT INTO erasure_targets(intent_id, kind, reference, state) "
                    "VALUES(?,?,?,'pending') ON CONFLICT(intent_id, kind, reference) DO NOTHING",
                    (intent_id, item["kind"], item["reference"]),
                )
            outstanding = self.db.execute(
                "SELECT count(*) FROM erasure_targets WHERE intent_id=? AND state!='verified'",
                (intent_id,)).fetchone()[0]
            state = PENDING if outstanding else COMPLETE
            self.db.execute(
                "UPDATE erasure_ledger SET state=?, confirmed_at=?, confirmed_by=?, "
                "completed_at=? WHERE id=?",
                (state, now(), actor, None if outstanding else now(), intent_id),
            )
            self.store._audit("erasure_confirm", intent_id,
                              {"actor": actor, "records": len(record_ids),
                               "outstanding": outstanding, "state": state})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"intent_id": intent_id, "state": state, "erased": len(record_ids),
                "obligations_outstanding": outstanding, "attachments": destroyed}

    def _erase(self, record_ids: Iterable[str], intent_id: str,
               dependents: Sequence[str]) -> dict[str, int]:
        """Tombstone the evidence, destroy its bytes, invalidate what leaned on it."""
        released = {"attachments": 0, "contents": 0, "bytes": 0}
        for record_pk in record_ids:
            row = self.db.execute(
                "SELECT source, fingerprint FROM records WHERE id=?", (record_pk,)).fetchone()
            self.db.execute("UPDATE records SET deleted=1 WHERE id=?", (record_pk,))
            self.db.execute("DELETE FROM record_fts WHERE id=?", (record_pk,))
            self.db.execute(
                "INSERT OR REPLACE INTO tombstones(record_id, intent_id, fingerprint, deleted_at) "
                "VALUES(?,?,?,?)",
                (record_pk, intent_id, row["fingerprint"], now()),
            )
            # The tombstone and the bytes go in the same transaction as the
            # intent, so there is no window where forgotten evidence is still
            # readable through its attachments.
            for key, value in self.blobs.release([record_pk], db=self.db).items():
                released[key] = released.get(key, 0) + value
            journal(self.db, record_pk, "erase")
        # A summary or lesson built on forgotten evidence must not stay on file
        # as if it were still supported. It is hidden, not deleted: the artifact
        # can be recomputed from what remains.
        for record_pk in dependents:
            if self.db.execute("SELECT deleted FROM records WHERE id=?", (record_pk,)).fetchone() is None:
                continue
            self.db.execute(
                "INSERT INTO record_visibility(record_id, hidden, replacement_id, reason, changed_at) "
                "VALUES(?, 1, NULL, 'awaiting recomputation after erasure', ?) "
                "ON CONFLICT(record_id) DO UPDATE SET hidden=1, replacement_id=NULL, "
                "reason=excluded.reason, changed_at=excluded.changed_at",
                (record_pk, now()),
            )
            self.db.execute("DELETE FROM record_fts WHERE id=?", (record_pk,))
            journal(self.db, record_pk, "invalidate")
        return released

    # -- obligations ---------------------------------------------------------

    def pending(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Cleanup work a backend call still owes, oldest intent first."""
        rows = self.db.execute(
            """
            SELECT t.intent_id, t.kind, t.reference, t.attempts, l.state, l.requested_at
            FROM erasure_targets t JOIN erasure_ledger l ON l.id=t.intent_id
            WHERE t.state='pending' ORDER BY l.requested_at, t.rowid LIMIT ?
            """, (_bounded(limit),)).fetchall()
        return [dict(row) for row in rows]

    def verify(self, *, intent_id: str, kind: str, reference: str) -> dict[str, Any]:
        """Mark one obligation verified. The intent completes only when all do."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT 1 FROM erasure_targets WHERE intent_id=? AND kind=? AND reference=?",
                (intent_id, kind, reference)).fetchone()
            if row is None:
                raise EvidenceError(f"no such obligation {intent_id}/{kind}/{reference}")
            self.db.execute(
                "UPDATE erasure_targets SET state='verified', verified_at=?, error=NULL "
                "WHERE intent_id=? AND kind=? AND reference=?",
                (now(), intent_id, kind, reference),
            )
            outcome = self._retire(intent_id)
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return outcome

    def fail(self, *, intent_id: str, kind: str, reference: str, error: str) -> dict[str, Any]:
        """Record a failed attempt without completing the erasure.

        An unreachable backend leaves the intent 'erasure_pending' forever. That
        is the honest state: the local copy is gone, the derived one is not.
        """
        _check_reason(error)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT attempts FROM erasure_targets WHERE intent_id=? AND kind=? AND reference=?",
                (intent_id, kind, reference)).fetchone()
            if row is None:
                raise EvidenceError(f"no such obligation {intent_id}/{kind}/{reference}")
            self.db.execute(
                "UPDATE erasure_targets SET state='pending', attempts=?, error=? "
                "WHERE intent_id=? AND kind=? AND reference=?",
                (row["attempts"] + 1, error[:500], intent_id, kind, reference),
            )
            outcome = self._retire(intent_id)
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return outcome

    def _retire(self, intent_id: str) -> dict[str, Any]:
        """Recompute the intent's state. Settling one debt never fails for another's sake."""
        outstanding = self.db.execute(
            "SELECT count(*) FROM erasure_targets WHERE intent_id=? AND state!='verified'",
            (intent_id,)).fetchone()[0]
        if outstanding:
            return {"intent_id": intent_id, "state": PENDING, "outstanding": outstanding}
        self.db.execute(
            "UPDATE erasure_ledger SET state=?, completed_at=? WHERE id=? AND state!='complete'",
            (COMPLETE, now(), intent_id),
        )
        return {"intent_id": intent_id, "state": COMPLETE, "outstanding": 0}

    def status(self, intent_id: str) -> dict[str, Any]:
        intent = self.db.execute(
            "SELECT id, state, requested_at, requested_by, confirmed_at, completed_at, epoch "
            "FROM erasure_ledger WHERE id=?", (intent_id,)).fetchone()
        if intent is None:
            raise EvidenceError(f"unknown erasure intent {intent_id!r}")
        targets = self.db.execute(
            "SELECT kind, reference, state, attempts FROM erasure_targets "
            "WHERE intent_id=? ORDER BY rowid", (intent_id,)).fetchall()
        return {**dict(intent), "obligations": [dict(row) for row in targets]}

    def forgotten(self, record_pk: str) -> bool:
        return bool(self.db.execute("SELECT 1 FROM tombstones WHERE record_id=?",
                                    (record_pk,)).fetchone())

    # -- internals -----------------------------------------------------------

    def _resolve(self, record_ids: Sequence[str]) -> list[Any]:
        if isinstance(record_ids, (str, bytes)) or not isinstance(record_ids, Sequence):
            raise EvidenceError("record_ids must be a sequence of ids")
        if not 1 <= len(record_ids) <= 500:
            raise EvidenceError("a preview covers between 1 and 500 records")
        found = []
        for record_pk in record_ids:
            row = self.db.execute(
                "SELECT id, source, fingerprint, deleted FROM records WHERE id=?",
                (record_pk,)).fetchone()
            if row is None:
                raise EvidenceError(f"unknown record {record_pk!r}")
            if row["deleted"]:
                existing = self.db.execute(
                    "SELECT intent_id FROM tombstones WHERE record_id=?", (record_pk,)).fetchone()
                if existing:
                    raise EvidenceError(
                        f"record {record_pk!r} was already forgotten by intent "
                        f"{existing['intent_id']!r}"
                    )
                raise EvidenceError(f"record {record_pk!r} is already deleted")
            found.append(row)
        return found

    def _radius(self, record_ids: Sequence[str]):
        """Recompute (rows, dependents, obligations, attachments, artifacts, digest).

        Both the preview and the confirmation go through here, so a confirm
        that no longer matches what was shown cannot slip past the digest.
        """
        placeholders = ",".join("?" * len(record_ids))
        rows = self.db.execute(
            f"SELECT id, source, fingerprint FROM records WHERE id IN ({placeholders}) "
            "AND deleted=0 ORDER BY id", list(record_ids)).fetchall()
        live = [row["id"] for row in rows]
        obligations = self._obligations(live) if live else []
        dependents = self._transitive_dependents(live) if live else []
        attachments = self._attachments(live)
        # The summaries and typed claims quoting this evidence are part of what the
        # owner is agreeing to lose. They are read live rather than stored, so this is
        # the moment their absence would otherwise be silent.
        artifacts = self.lineage.artifacts(live) if live else []
        fingerprint = digest([
            [[row["id"], row["fingerprint"]] for row in rows],
            [[item["kind"], item["reference"]] for item in obligations],
            [[item["kind"], item["id"]] for item in artifacts],
            [attachments["files"], attachments["bytes"]],
        ])
        return rows, dependents, obligations, attachments, artifacts, fingerprint

    def _attachments(self, record_ids: Sequence[str]) -> dict[str, int]:
        """How much attachment material this radius would destroy."""
        if not record_ids:
            return {"files": 0, "bytes": 0}
        placeholders = ",".join("?" * len(record_ids))
        row = self.db.execute(
            f"SELECT count(*) AS files, COALESCE(sum(size), 0) AS bytes FROM attachments "
            f"WHERE record_id IN ({placeholders})", list(record_ids)).fetchone()
        return {"files": int(row["files"] or 0), "bytes": int(row["bytes"] or 0)}

    def _transitive_dependents(self, record_ids: Sequence[str], *, cap: int = 2000) -> list[str]:
        """Everything that cites the targets, directly or through other artifacts.

        Traversal lives in one place: an erasure that missed an edge the broker could
        follow would leave evidence reachable through a path nobody had to confirm.
        """
        targets = set(record_ids)
        closure, _truncated = self.lineage.closure(record_ids, cap=cap)
        return [item for item in closure if item not in targets]

    def _obligations(self, record_ids: Sequence[str]) -> list[dict[str, str]]:
        rows = self.db.execute(
            "SELECT backend, bank_id, document_id, revision FROM backend_documents "
            "WHERE record_id IN (%s)" % ",".join("?" * len(record_ids)),
            record_ids).fetchall()
        obligations = [
            {"kind": "backend_document",
             "reference": f"{row['backend']}:{row['bank_id']}:{row['document_id']}"}
            for row in rows
        ]
        # Derived facts and observations live behind the document; clearing the
        # document is not proof that its derivatives are gone, so each bank gets
        # its own verification obligation.
        for bank in sorted({f"{row['backend']}:{row['bank_id']}" for row in rows}):
            obligations.append({"kind": "backend_derived", "reference": bank})
        return obligations

    # -- restore safety ------------------------------------------------------

    def absorb(self, entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Merge an externally held ledger without rewriting history.

        The retired framework stamped a merged intent with the merge time, so a
        forgotten-on-Tuesday item looked forgotten-on-the-day-of-the-restore and
        the original request date was gone. Here ``requested_at`` keeps the
        earlier of the two, and only a terminal local state is allowed to win.
        """
        merged = inserted = 0
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for entry in entries:
                intent_id = _check_intent_id(entry.get("id"))
                existing = self.db.execute(
                    "SELECT state, requested_at FROM erasure_ledger WHERE id=?",
                    (intent_id,)).fetchone()
                incoming_state = _state_of(entry.get("state"))
                if existing is None:
                    self.db.execute(
                        "INSERT INTO erasure_ledger(id, source, requested_at, requested_by, "
                        "requester_kind, reason, preview, preview_digest, state, confirmed_at, "
                        "completed_at, epoch) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (intent_id, str(entry.get("source", ""))[:500],
                         _as_utc(entry.get("requested_at")), str(entry.get("requested_by"))[:200],
                         str(entry.get("requester_kind", "owner"))[:40],
                         str(entry.get("reason", "merged from an external ledger"))[:1000],
                         str(entry.get("preview", "{}"))[:200000],
                         str(entry.get("preview_digest", ""))[:64], incoming_state,
                         entry.get("confirmed_at"), entry.get("completed_at"),
                         int(entry.get("epoch", 0))),
                    )
                    inserted += 1
                else:
                    keep = existing["state"] if _RANK[existing["state"]] >= _RANK[incoming_state] \
                        else incoming_state
                    self.db.execute(
                        "UPDATE erasure_ledger SET state=?, requested_at=?, "
                        "completed_at=COALESCE(completed_at, ?) WHERE id=?",
                        (keep, min(existing["requested_at"], _as_utc(entry.get("requested_at"))),
                         entry.get("completed_at"), intent_id),
                    )
                    merged += 1
                for record_pk, fingerprint in (entry.get("tombstones") or {}).items():
                    self.db.execute(
                        "INSERT OR IGNORE INTO tombstones(record_id, intent_id, fingerprint, "
                        "deleted_at) VALUES(?,?,?,?)",
                        (_as_record_id(record_pk), intent_id, str(fingerprint)[:64],
                         _as_utc(entry.get("requested_at"))),
                    )
            self.store._audit("erasure_ledger_absorb", "erasure_ledger",
                              {"inserted": inserted, "merged": merged})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"inserted": inserted, "merged": merged}

    def reapply(self) -> dict[str, Any]:
        """Re-forget anything a restore brought back. Run this before exposing data.

        A backup is a snapshot of a moment that already excludes forgotten
        items; restoring it first and honoring the ledger second is the only
        ordering that does not silently undo a deletion.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            rows = self.db.execute(
                """
                SELECT t.record_id, t.intent_id FROM tombstones t
                JOIN records r ON r.id=t.record_id WHERE r.deleted=0
                """
            ).fetchall()
            for row in rows:
                self.db.execute("UPDATE records SET deleted=1 WHERE id=?", (row["record_id"],))
                self.db.execute("DELETE FROM record_fts WHERE id=?", (row["record_id"],))
                journal(self.db, row["record_id"], "erase_reapplied")
            self.store._audit("erasure_reapply", "erasure_ledger", {"records": len(rows)})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"reapplied": len(rows)}


_RANK = {AWAITING: 0, PENDING: 1, COMPLETE: 2}


def _state_of(value: Any) -> str:
    state = str(value or AWAITING)
    if state not in _RANK:
        raise EvidenceError(f"unknown erasure state {value!r}")
    return state


def _check_intent_id(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("erase_") or len(value) > 100:
        raise EvidenceError("an absorbed intent needs its original erase_ id")
    return value


def _as_record_id(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("rec_") or len(value) > 100:
        raise EvidenceError("an absorbed tombstone needs its original rec_ id")
    return value


def _as_utc(value: Any) -> str:
    # An external ledger is input, not trusted state: a bad timestamp is a
    # refused entry, not a crash partway through a merge.
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError("an absorbed entry needs its original requested_at timestamp")
    try:
        return timestamp(value)
    except (ValueError, TypeError):
        raise EvidenceError(f"absorbed requested_at is not a timezone-aware timestamp: {value!r}")



def _bounded(value: int, *, maximum: int = 200) -> int:
    if not isinstance(value, int) or not 1 <= value <= maximum:
        raise EvidenceError(f"limit must be an integer between 1 and {maximum}")
    return value


def _check_reason(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise EvidenceError("text must be nonempty and at most 1000 characters")
