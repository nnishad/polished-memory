"""C4 recovery: bring the store back up without undoing anything it decided.

Two jobs live here. The first is the unclean start — a worker that died holding a
claim, an artifact handed to the host and never acknowledged, a connector lease whose
holder never came back. Each is reconciled into the state that is *true*, which is
usually ``uncertain`` rather than the retryable state it would have been convenient to
restore it to.

The second is rollback, and it has one hard rule. A snapshot is a copy of a moment
from before things were forgotten, and forgetting is not something a recovery gets to
reverse. So the erasure ledger, its outstanding obligations and the tombstones are
lifted out of the live database, carried forward over the restored copy, and applied
before that copy answers a single question. Restoring an old file and re-exposing
evidence the owner erased is not a restore; it is a leak that leaves a receipt.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..ids import now
from ..storage.evidence import EvidenceError, journal
from ..storage.migrations import apply_migrations
from .snapshots import Snapshots, _head, _seal, _sealed, open_read_only

__all__ = ["Recovery", "Ledger", "LEDGER_TABLES"]

# The tables a restore must never roll back. Everything else may legitimately be
# older than it was; these record decisions that stay made. Listed parent first:
# the rows go back in this order and come off in the reverse, or the foreign keys
# the restored copy still holds trip on the way through.
LEDGER_TABLES = ("erasure_ledger", "erasure_targets", "tombstones")
PRE_RESTORE = "pre-restore"


@dataclass
class Ledger:
    """What the live store had decided, taken out of the file before it is replaced."""

    epoch: int = 1
    checkpoints: dict[str, int] = field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    banks: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "consumers": len(self.checkpoints),
                "intents": len(self.tables.get("erasure_ledger") or []),
                "obligations": len(self.tables.get("erasure_targets") or []),
                "tombstones": len(self.tables.get("tombstones") or []),
                "banks": list(self.banks)}


class Recovery:
    """Startup reconciliation and ledger-preserving rollback.

    A restore writes through the store's existing connection, so managers built before
    it keep working afterwards; it takes the database lock while doing so, so nothing
    may be mid-transaction.
    """

    def __init__(self, store, *, snapshots: Snapshots, owner_principal: str | None = None,
                 events=None, outbox=None, jobs=None):
        self.store = store
        self.db = store.db
        self.snapshots = snapshots
        self.owner_principal = owner_principal
        self.events = events
        self.outbox = outbox
        self.jobs = jobs

    # -- unclean start -------------------------------------------------------

    def reconcile(self, *, at: float | None = None) -> dict[str, Any]:
        """Put every half-finished thing into the state that is actually true.

        Safe to run twice: each step narrows a set of rows into a terminal state, so
        the second pass has nothing left to report.
        """
        report: dict[str, Any] = {"uncertain": {}}
        if self.events is not None:
            # A lapsed due-event claim cannot be re-queued: the holder may have handed
            # it off and died before writing that down.
            expired = self.events.expire_claims(at=at)
            if expired:
                report["uncertain"]["due_events"] = len(expired)
        if self.outbox is not None:
            # The same argument one step further along: an artifact in ``leased`` or
            # ``attempted`` whose holder vanished may already have reached the owner.
            stranded = self.outbox.recover(at=at)
            if stranded:
                report["uncertain"]["outbox"] = len(stranded)
        if self.jobs is not None:
            report["uncertain"].update(self.jobs.reconcile(at=at) or {})
        report["stale_leases"] = self._stale_leases(at=at)
        self.store._audit("recovery_reconcile", "startup",
                          {"uncertain": report["uncertain"],
                           "stale_leases": len(report["stale_leases"])})
        return report

    def _stale_leases(self, *, at: float | None = None) -> list[str]:
        """Release a connector lease whose holder is not coming back.

        The cursor is left where it was: a takeover re-reads from the last committed
        page, and page-level idempotency makes the repeat harmless.
        """
        moment = time.time() if at is None else float(at)
        rows = self.db.execute(
            "SELECT source, holder FROM connectors WHERE lease IS NOT NULL AND "
            "lease_until < ?", (moment,)).fetchall()
        released = []
        for row in rows:
            self.db.execute(
                "UPDATE connectors SET lease=NULL, holder=NULL, lease_until=NULL, "
                "updated_at=? WHERE source=? AND lease IS NOT NULL", (now(), row["source"]))
            released.append(f"{row['source']}:{row['holder']}")
            self.store._audit("lease_stale", str(row["source"]), {"holder": row["holder"]})
        return released

    def integrity(self) -> dict[str, Any]:
        """What the store looks like from the outside, without changing it.

        Damage is reported rather than raised: this is the call an operator makes
        *because* something looks broken, and a traceback instead of a report leaves
        them with no idea which of the two is broken.
        """
        problems: list[str] = []
        head: dict[str, Any] = {}
        unfinished: dict[str, int] = {}
        try:
            verdict = str(self.db.execute("PRAGMA integrity_check").fetchone()[0])
            if verdict != "ok":
                problems.append(f"sqlite reports an integrity failure in the live store: "
                                f"{verdict[:120]}")
            foreign = self.db.execute("PRAGMA foreign_key_check").fetchall()
            if foreign:
                problems.append(f"{len(foreign)} foreign-key violation(s) present")
            head = _head(self.db)
            unfinished = {
                "outbox_unsettled": int(self.db.execute(
                    "SELECT count(*) FROM outbox WHERE state IN ('leased','attempted',"
                    "'uncertain')").fetchone()[0]),
                "intents_awaiting": int(self.db.execute(
                    "SELECT count(*) FROM decision_intents WHERE state='awaiting_analysis'"
                ).fetchone()[0]),
                "events_uncertain": int(self.db.execute(
                    "SELECT count(*) FROM due_events WHERE state='uncertain'").fetchone()[0]),
                "erasure_pending": int(self.db.execute(
                    "SELECT count(*) FROM erasure_targets WHERE state!='verified'"
                ).fetchone()[0]),
            }
        except sqlite3.Error as error:
            problems.append(f"the live store cannot be inspected: {str(error)[:160]}")
        return {"ok": not problems, "problems": problems, **head, **unfinished}

    # -- rollback ------------------------------------------------------------

    def restore(self, snapshot_id: str, *, actor: str, bank_id: str | None = None) \
            -> dict[str, Any]:
        """Take the store back to a snapshot, keeping every decision made since.

        Refused unless the snapshot verifies and the caller is the owner principal: a
        rollback destroys everything written after the snapshot, which is not
        something to do by accident or by inference.
        """
        if self.owner_principal is None:
            raise EvidenceError(
                "no owner principal is configured, so a restore cannot be confirmed; "
                "set HERMES_MEMORY_OWNER_PRINCIPAL")
        if actor != self.owner_principal:
            raise EvidenceError(f"a restore requires the owner principal, not {actor!r}")
        checked = self.snapshots.verify(snapshot_id)
        if not checked["ok"]:
            raise EvidenceError(
                f"snapshot {snapshot_id} does not verify: " + "; ".join(checked["problems"]))
        item = self.snapshots.resolve(snapshot_id)

        carried = self.carry()
        guard = self._copy_aside(item.id)
        try:
            self._install(item.database)
            applied = self.apply(carried, bank_id=bank_id)
        except BaseException as failure:
            # A failed restore leaves a store that is neither the old one nor the
            # restored one. Put back what was there before anything was promised.
            try:
                self._install(guard)
                self.apply(carried)
            except BaseException as repair:
                raise EvidenceError(
                    f"the restore failed ({str(failure)[:120]}) and the pre-restore "
                    f"guard {guard.name} could not be reinstalled ({str(repair)[:120]}); "
                    "the store needs a manual repair") from None
            raise
        self.store._audit("restore_complete", snapshot_id,
                          {"actor": actor, "guard": guard.name, **applied})
        note = ("Evidence forgotten since this snapshot stays forgotten: "
                f"{applied['reapplied']} tombstoned record(s) were taken out of "
                "the restored copy again before it answered a single read.")
        if applied["tombstones_without_a_record"]:
            # Said out loud rather than left as a count: these are the owner's decisions
            # about evidence this snapshot never held, so their obligations travel while
            # their markers have nothing to mark.
            note += (f" {applied['tombstones_without_a_record']} erasure intent(s) name "
                     "records this snapshot does not contain; their deletion obligations "
                     "are carried and their tombstones are not.")
        return {"restored": snapshot_id, "epoch": carried.epoch, **applied,
                "pre_restore_backup": str(guard), "note": note}

    # -- the ledger that does not roll back ----------------------------------

    def carry(self) -> Ledger:
        """Copy the durable forgetting record out of the live database."""
        carried = Ledger(epoch=self.store.epoch(),
                         banks=tuple(sorted({
                             str(row["bank_id"]) for row in self.db.execute(
                                 "SELECT DISTINCT bank_id FROM backend_documents")})))
        for table in LEDGER_TABLES:
            carried.tables[table] = [dict(row) for row in
                                     self.db.execute(f"SELECT * FROM {table}")]
        for row in self.db.execute("SELECT consumer, seq FROM consumer_checkpoints"):
            carried.checkpoints[str(row["consumer"])] = int(row["seq"])
        return carried

    def apply(self, carried: Ledger, *, bank_id: str | None = None) -> dict[str, Any]:
        """Write the kept decisions into the restored copy and honour the tombstones.

        All of it in one transaction, before the file is used: there is no moment in
        which the restored store can be read with the ledger still missing.
        """
        self.db.execute("BEGIN IMMEDIATE")
        orphans: list[str] = []
        try:
            # Empty the ledger before refilling it, children first: the copy that
            # just landed has its own intents, and clearing the parent table under
            # them is a foreign-key failure rather than a restore.
            for table in reversed(LEDGER_TABLES):
                self.db.execute(f"DELETE FROM {table}")
            for table in LEDGER_TABLES:
                rows = carried.tables.get(table) or []
                if not rows:
                    continue
                if table == "tombstones":
                    # A tombstone names the record it buries, and the record table belongs to
                    # the snapshot rather than to the ledger. Evidence that arrived *after* this
                    # snapshot and was forgotten after that has no row here to hide, and
                    # inserting its grave marker anyway is a foreign-key failure that aborts the
                    # whole restore — which is the opposite of what a rollback is for. The
                    # intent and its obligations are carried regardless, so the owner's decision
                    # survives; only the marker for a body this copy never held is left out.
                    present = {str(row["id"]) for row in
                               self.db.execute("SELECT id FROM records")}
                    unmatched = [str(row["record_id"]) for row in rows
                                 if str(row["record_id"]) not in present]
                    rows = [row for row in rows if str(row["record_id"]) in present]
                    orphans.extend(unmatched)
                    if not rows:
                        continue
                columns = sorted({key for row in rows for key in row})
                self.db.executemany(
                    f"INSERT INTO {table}({', '.join(columns)}) "
                    f"VALUES({', '.join('?' * len(columns))})",
                    [[row.get(column) for column in columns] for row in rows])
            for consumer, seq in sorted(carried.checkpoints.items()):
                # A consumer that has read past this file's end must not be walked
                # backwards, or it re-emits changes it already delivered as new.
                self.db.execute(
                    "INSERT INTO consumer_checkpoints(consumer, seq, updated_at) "
                    "VALUES(?,?,?) ON CONFLICT(consumer) DO UPDATE SET "
                    "seq=MAX(consumer_checkpoints.seq, excluded.seq), "
                    "updated_at=excluded.updated_at", (consumer, seq, now()))
            reapplied = self._honour_tombstones()
            epoch = self._keep_epoch_monotonic(carried.epoch)
            dropped = self._refuse_old_bank(bank_id)
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"reapplied": reapplied, "epoch": epoch, "obligations":
                len(carried.tables.get("erasure_targets") or []),
                "intents": len(carried.tables.get("erasure_ledger") or []),
                "tombstones": len(carried.tables.get("tombstones") or []),
                "tombstones_without_a_record": len(orphans),
                "orphaned_tombstones": orphans[:20],
                "stale_projections_dropped": dropped}

    def _honour_tombstones(self) -> int:
        """Take back anything the snapshot still has that has since been forgotten."""
        rows = self.db.execute(
            "SELECT t.record_id, t.fingerprint FROM tombstones t JOIN records r "
            "ON r.id = t.record_id WHERE r.deleted=0").fetchall()
        for row in rows:
            standing = self.db.execute("SELECT fingerprint FROM records WHERE id=?",
                                       (row["record_id"],)).fetchone()
            if str(standing["fingerprint"]) != str(row["fingerprint"] or ""):
                # The same id carrying different bytes is not the record that was
                # forgotten. Refuse rather than guess which of them is real.
                raise EvidenceError(
                    f"{row['record_id']} came back with different content than its "
                    "tombstone records; refusing to guess which is the erased one")
            self.db.execute("UPDATE records SET deleted=1 WHERE id=?", (row["record_id"],))
            self.db.execute("DELETE FROM record_fts WHERE id=?", (row["record_id"],))
            self.db.execute(
                "INSERT INTO record_visibility(record_id, hidden, replacement_id, reason, "
                "changed_at) VALUES(?,1,NULL,'erased after this snapshot',?) "
                "ON CONFLICT(record_id) DO UPDATE SET hidden=1, replacement_id=NULL, "
                "reason=excluded.reason, changed_at=excluded.changed_at",
                (row["record_id"], now()))
            journal(self.db, str(row["record_id"]), "restore_erase")
        return len(rows)

    def _keep_epoch_monotonic(self, kept: int) -> int:
        """The epoch never goes backwards, or a writer with an expired fence could write."""
        restored = int(self.db.execute("SELECT value FROM memory_epoch WHERE id=1")
                       .fetchone()[0])
        if kept > restored:
            self.db.execute("UPDATE memory_epoch SET value=? WHERE id=1", (kept,))
        return max(kept, restored)

    def _refuse_old_bank(self, bank_id: str | None) -> int:
        """Do not let a restore re-adopt projections into a bank that no longer exists.

        After a reset the old bank is cleared, so a snapshot from before it still
        believes its documents are projected there. Nothing from those rolled-back rows
        is trusted: dropping them is what makes the document map ask the question
        again of the bank that is actually configured.
        """
        if bank_id is None:
            return 0
        stale = int(self.db.execute(
            "SELECT count(*) FROM backend_documents WHERE bank_id != ?",
            (bank_id,)).fetchone()[0])
        if stale:
            self.db.execute("DELETE FROM backend_documents WHERE bank_id != ?", (bank_id,))
        return stale

    # -- file handling -------------------------------------------------------

    def _copy_aside(self, snapshot_id: str) -> Path:
        """Back up what is live right now, so a wrong restore is itself reversible."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        guard = self.snapshots.root / PRE_RESTORE / f"{stamp}-{snapshot_id}.db"
        guard.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = _sealed(guard)
        try:
            self.db.backup(target)
            _seal(target)
        finally:
            target.close()
        guard.chmod(0o600)
        return guard

    def _install(self, source: Path) -> None:
        """Write *source* over the live database, leaving the connection open.

        Replacing the file and reopening would rebind one handle and strand the
        twenty-odd managers that each captured ``store.db`` when they were built. The
        backup API puts the snapshot's pages through the live connection instead, so
        every handle keeps pointing at the database it is actually reading.

        The donor is opened read-only because an ordinary store connection would set
        the snapshot's journal mode and rewrite its header, altering a file whose
        digest is on record.

        A snapshot of an older build is a valid thing to restore, so the schema is
        brought forward here — before the ledger is re-applied and before the restored
        copy answers a single read.
        """
        donor = open_read_only(source)
        try:
            donor.backup(self.db)
        except sqlite3.Error as error:
            raise EvidenceError(
                f"the snapshot could not be put in place: {str(error)[:160]}") from None
        finally:
            donor.close()
        try:
            apply_migrations(self.db, at=now)
        except RuntimeError as error:
            raise EvidenceError(str(error)) from None
