"""C4 reset — wipe local evidence and open the remote cleanup debt in one transaction.

A reset that clears local rows first and *then* remembers to clear the backend
has a crash window in which the local copy is gone and no one is obliged to
finish the job. Here the obligation rows commit together with the local fence,
so a crash leaves work outstanding rather than data silently retained.
"""
from __future__ import annotations

import json
from typing import Any

from ..ids import digest, new_id, now
from ..storage.evidence import EvidenceError, journal
from .erasure import COMPLETE, PENDING, AWAITING

__all__ = ["ResetController"]


class ResetController:
    def __init__(self, store, *, owner_principal: str | None = None):
        self.store = store
        self.db = store.db
        self.owner_principal = owner_principal

    def _radius(self):
        """(live rows, obligations, digest) measured from current state."""
        live = self.db.execute(
            "SELECT id, fingerprint FROM records WHERE deleted=0 ORDER BY id").fetchall()
        banks = self.db.execute(
            "SELECT DISTINCT backend, bank_id FROM backend_documents ORDER BY backend, bank_id"
        ).fetchall()
        sources = self.db.execute(
            "SELECT source, generation FROM connectors ORDER BY source").fetchall()
        obligations = [{"kind": "backend_bank", "reference": f"{row['backend']}:{row['bank_id']}"}
                       for row in banks]
        return live, obligations, digest([
            [[row["id"], row["fingerprint"]] for row in live],
            [item["reference"] for item in obligations],
            [[row["source"], row["generation"]] for row in sources],
        ])

    def preview(self, *, actor: str, reason: str) -> dict[str, Any]:
        """Everything a reset would destroy and every bank it would owe."""
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise EvidenceError("reason must be nonempty text of at most 1000 characters")
        live, obligations, digest_value = self._radius()
        sources = self.db.execute(
            "SELECT source FROM connectors ORDER BY source").fetchall()
        intent_id = new_id("erase")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "INSERT INTO erasure_ledger(id, source, requested_at, requested_by, "
                "requester_kind, reason, preview, preview_digest, state, epoch) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (intent_id, "*", now(), actor, "owner", reason,
                 json.dumps({"records": [row["id"] for row in live], "dependents": [],
                             "obligations": obligations, "reset": True}, sort_keys=True),
                 digest_value, AWAITING, self.store.epoch()),
            )
            self.store._audit("reset_preview", intent_id,
                              {"actor": actor, "records": len(live), "banks": len(obligations)})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {
            "intent_id": intent_id,
            "preview_digest": digest_value,
            "records": len(live),
            "banks_to_clear": [item["reference"] for item in obligations],
            "connectors": [row["source"] for row in sources],
            "note": ("Every source cursor restarts from nothing and every connector lease is "
                     "voided. Confirmation must come from the owner with this exact digest."),
        }

    def confirm(self, *, intent_id: str, preview_digest: str, actor: str) -> dict[str, Any]:
        if self.owner_principal is None:
            raise EvidenceError(
                "no owner principal is configured, so a reset cannot be confirmed; "
                "set HERMES_MEMORY_OWNER_PRINCIPAL"
            )
        if actor != self.owner_principal:
            raise EvidenceError(f"confirmation requires the owner principal, not {actor!r}")
        if not isinstance(preview_digest, str) or len(preview_digest) != 64:
            raise EvidenceError("preview_digest must be the 64 character digest shown")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            intent = self.db.execute(
                "SELECT * FROM erasure_ledger WHERE id=?", (intent_id,)).fetchone()
            if intent is None:
                raise EvidenceError(f"unknown reset intent {intent_id!r}")
            if json.loads(intent["preview"]).get("reset") is not True:
                raise EvidenceError(f"intent {intent_id!r} is not a reset")
            if intent["state"] != AWAITING:
                raise EvidenceError(f"intent is already {intent['state']}")
            if intent["preview_digest"] != preview_digest:
                raise EvidenceError(
                    "the preview digest does not match this intent; a different preview was "
                    "confirmed")

            payload = json.loads(intent["preview"])
            # Re-measure rather than trust the caller's copy: a reset planned
            # before new evidence arrived is a smaller reset than the one being
            # confirmed now, and the difference is exactly what would survive it.
            _live, _obligations, current = self._radius()
            if current != intent["preview_digest"] or current != preview_digest:
                raise EvidenceError(
                    "the store changed after this preview; confirm the new preview instead")
            record_ids = list(payload["records"])
            for item in payload["obligations"]:
                self.db.execute(
                    "INSERT INTO erasure_targets(intent_id, kind, reference, state) "
                    "VALUES(?,?,?,'pending') ON CONFLICT(intent_id, kind, reference) DO NOTHING",
                    (intent_id, item["kind"], item["reference"]),
                )
            for record_pk in record_ids:
                row = self.db.execute(
                    "SELECT fingerprint FROM records WHERE id=?", (record_pk,)).fetchone()
                if row is None:
                    continue
                self.db.execute("UPDATE records SET deleted=1 WHERE id=?", (record_pk,))
                self.db.execute("DELETE FROM record_fts WHERE id=?", (record_pk,))
                self.db.execute(
                    "INSERT OR REPLACE INTO tombstones(record_id, intent_id, fingerprint, "
                    "deleted_at) VALUES(?,?,?,?)",
                    (record_pk, intent_id, row["fingerprint"], now()),
                )
                journal(self.db, record_pk, "reset")
            # The epoch bump and the obligations are in the same transaction as
            # the wipe: after this commit no writer holding an old lease can
            # land, and every bank that still holds a copy is on record.
            self.db.execute("UPDATE memory_epoch SET value=value+1 WHERE id=1")
            epoch = self.store.epoch()
            self.db.execute("UPDATE connectors SET lease=NULL, lease_until=NULL, holder=NULL, "
                            "cursor=NULL, coverage_state='unknown', last_success_at=NULL")
            outstanding = self.db.execute(
                "SELECT count(*) FROM erasure_targets WHERE intent_id=? AND state!='verified'",
                (intent_id,)).fetchone()[0]
            state = PENDING if outstanding else COMPLETE
            self.db.execute(
                "UPDATE erasure_ledger SET state=?, confirmed_at=?, confirmed_by=?, "
                "completed_at=?, epoch=? WHERE id=?",
                (state, now(), actor, None if outstanding else now(), epoch, intent_id),
            )
            self.store._audit("reset_confirm", intent_id,
                              {"actor": actor, "records": len(record_ids), "epoch": epoch,
                               "outstanding": outstanding})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"intent_id": intent_id, "state": state, "epoch": epoch,
                "records_retired": len(record_ids), "obligations_outstanding": outstanding}
