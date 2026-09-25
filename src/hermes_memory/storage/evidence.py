"""C1 — canonical evidence store.

Immutable revisions, explicit visibility, dependency edges, receipts and the
backend document projection ledger. No inference, no network and no blob
download may happen while holding a write transaction here.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..ids import digest, now, record_id as make_record_id, timestamp
from .migrations import apply_migrations, connect

__all__ = ["EvidenceStore", "EvidenceError", "Evidence", "Prepared", "prepare_envelope", "journal"]

_MAXIMUM_TEXT = 4_000_000
_KNOWN_OCCURRED_PRECISION = {"second", "minute", "hour", "day", "week", "month", "year", "unknown"}


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class Evidence:
    id: str
    source: str
    source_id: str
    revision: str
    occurred_at: str | None
    occurred_precision: str
    observed_at: str
    ingested_at: str
    kind: str
    text: str
    metadata: dict[str, Any]
    fingerprint: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Evidence":
        return cls(
            id=row["id"],
            source=row["source"],
            source_id=row["source_id"],
            revision=row["revision"],
            occurred_at=row["occurred_at"],
            occurred_precision=row["occurred_precision"],
            observed_at=row["observed_at"],
            ingested_at=row["ingested_at"],
            kind=row["kind"],
            text=row["text"],
            metadata=json.loads(row["metadata"]),
            fingerprint=row["fingerprint"],
        )


def _required_text(value: Any, label: str, maximum: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise EvidenceError(f"{label} must be nonempty text of at most {maximum} characters")
    return value


@dataclass(frozen=True)
class Prepared:
    """A validated envelope with every derived column already computed.

    Validation is separated from writing so the sync layer can prepare a whole
    page before opening a transaction and keep the lock held for writes only.
    """

    id: str
    source: str
    source_id: str
    revision: str
    occurred_at: str | None
    occurred_precision: str
    observed_at: str
    ingested_at: str
    kind: str
    text: str
    metadata: str
    fingerprint: str
    parents: tuple[str, ...]
    receipt: str
    receipt_id: str


def prepare_envelope(envelope: dict[str, Any]) -> Prepared:
    if not isinstance(envelope, dict):
        raise EvidenceError("envelope must be an object")

    source = _required_text(envelope.get("source"), "source", 500)
    source_id = _required_text(envelope.get("source_id"), "source_id", 500)
    revision = _required_text(envelope.get("revision", "1"), "revision", 500)
    kind = _required_text(envelope.get("kind", "message"), "kind", 100)
    text = envelope.get("text")
    if not isinstance(text, str) or len(text) > _MAXIMUM_TEXT:
        raise EvidenceError(f"text must be a string of at most {_MAXIMUM_TEXT} characters")
    metadata = envelope.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise EvidenceError("metadata must be an object")

    observed_at = timestamp(_required_text(envelope.get("observed_at"), "observed_at", 100))
    # An unknown event time stays unknown. Filling it from ingestion time
    # would silently turn "we received this on the 25th" into "this
    # happened on the 25th".
    occurred_raw = envelope.get("occurred_at")
    precision = envelope.get("occurred_precision", "unknown")
    if precision not in _KNOWN_OCCURRED_PRECISION:
        raise EvidenceError(f"unknown occurred_precision {precision!r}")
    if occurred_raw in (None, ""):
        if precision != "unknown":
            raise EvidenceError("occurred_precision requires an occurred_at value")
        occurred_at: str | None = None
    else:
        occurred_at = timestamp(_required_text(occurred_raw, "occurred_at", 100))

    parents = envelope.get("parent_record_ids") or []
    if not isinstance(parents, list) or len(parents) > 100:
        raise EvidenceError("parent_record_ids must be a list of at most 100 IDs")
    parents = tuple(_required_text(parent, "parent_record_id", 100) for parent in parents)

    fingerprint = digest([occurred_at, kind, text, metadata])
    primary_key = make_record_id(source, source_id, revision)
    receipt = {k: v for k, v in envelope.items() if k != "_contract"}
    return Prepared(
        id=primary_key, source=source, source_id=source_id, revision=revision,
        occurred_at=occurred_at, occurred_precision=precision, observed_at=observed_at,
        ingested_at=now(), kind=kind, text=text,
        metadata=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        fingerprint=fingerprint, parents=parents,
        receipt=json.dumps(receipt, ensure_ascii=False),
        receipt_id=digest([primary_key, fingerprint]),
    )


def journal(db: sqlite3.Connection, record_pk: str, change: str,
            *, source: str | None = None, generation: int = 0) -> int:
    """Append one change to the journal. Requires an ambient transaction.

    Generation 0 means "not written by a connector" — a local tool call or an
    operator action. Journal rows are only ever written inside the transaction
    that commits the change, so a rolled back write leaves nothing to replay.
    """
    if source is None:
        row = db.execute("SELECT source FROM records WHERE id=?", (record_pk,)).fetchone()
        if row is None:
            raise EvidenceError(f"cannot journal {change} for unknown record {record_pk!r}")
        source = row["source"]
    epoch = db.execute("SELECT value FROM memory_epoch WHERE id=1").fetchone()[0]
    cursor = db.execute(
        "INSERT INTO change_journal(source, generation, epoch, record_id, change, committed_at) "
        "VALUES(?,?,?,?,?,?)",
        (source, generation, epoch, record_pk, change, now()),
    )
    return int(cursor.lastrowid)


class EvidenceStore:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.path = path
        self.db = connect(path)
        self.db.execute("PRAGMA user_version=%d" % apply_migrations(self.db, at=now))

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *excinfo) -> None:
        self.close()

    # -- reads ---------------------------------------------------------------

    def epoch(self) -> int:
        return int(self.db.execute("SELECT value FROM memory_epoch WHERE id=1").fetchone()[0])

    def bump_epoch(self, *, reason: str, actor: str) -> int:
        """Advance the global fence and invalidate every outstanding lease.

        A reset or trust-revoking change must make in-flight writers fail
        *forward*, not merely discard their output: a page already fetched under
        the old epoch would otherwise be committed as though it were current.
        """
        _required_text(reason, "reason", 500)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("UPDATE memory_epoch SET value=value+1 WHERE id=1")
            epoch = self.epoch()
            self.db.execute("UPDATE connectors SET lease=NULL, lease_until=NULL, holder=NULL")
            self._audit("epoch_bump", "memory_epoch",
                        {"actor": actor, "reason": reason, "epoch": epoch})
            self.db.execute("COMMIT")
            return epoch
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def get(self, record_pk: str, *, include_hidden: bool = False) -> Evidence | None:
        row = self.db.execute("SELECT * FROM records WHERE id=?", (record_pk,)).fetchone()
        if row is None:
            return None
        if not include_hidden and not self._visible(row["id"]):
            return None
        return Evidence.from_row(row)

    def _visible(self, record_pk: str) -> bool:
        if self.db.execute("SELECT deleted FROM records WHERE id=?", (record_pk,)).fetchone()[0]:
            return False
        row = self.db.execute(
            "SELECT hidden FROM record_visibility WHERE record_id=?", (record_pk,)
        ).fetchone()
        return row is None or not row[0]

    def live_and_visible(self, record_pk: str) -> bool:
        return self._visible(record_pk)

    def search(self, match: str, *, limit: int = 20) -> list[Evidence]:
        """Lexical search over committed, visible evidence.

        This path deliberately never touches the model or the network, so it
        still answers while inference is paused or Hindsight is down.
        """
        limit = _bounded_limit(limit)
        # COALESCE must wrap the whole scalar subquery: applied inside it, an
        # absent visibility row yields an empty result set and the comparison
        # becomes NULL = 0, which silently filters out every fresh record.
        rows = self.db.execute(
            """
            SELECT r.* FROM record_fts f JOIN records r ON r.id = f.id
            WHERE record_fts MATCH ? AND r.deleted = 0
              AND COALESCE((SELECT v.hidden FROM record_visibility v WHERE v.record_id = r.id), 0) = 0
            ORDER BY rank LIMIT ?
            """,
            (match, limit),
        ).fetchall()
        return [Evidence.from_row(row) for row in rows]

    def parents(self, record_pk: str) -> list[str]:
        return [
            row[0]
            for row in self.db.execute(
                "SELECT parent_id FROM record_dependencies WHERE child_id=?", (record_pk,)
            )
        ]

    def dependents(self, record_pk: str) -> list[str]:
        return [
            row[0]
            for row in self.db.execute(
                "SELECT child_id FROM record_dependencies WHERE parent_id=?", (record_pk,)
            )
        ]

    # -- writes --------------------------------------------------------------

    def commit(self, envelope: dict[str, Any], *, fence: Any = None) -> dict[str, Any]:
        """Commit one canonical evidence item. Idempotent per content fingerprint.

        Returns ``{"id", "duplicate"}``. A duplicate revision is acknowledged
        without rewriting, so replaying a page never forks two records.

        *fence*, when supplied, is validated inside the same transaction: a
        connector whose lease, generation or epoch went stale cannot interleave
        a check-then-write.
        """
        prepared = prepare_envelope(envelope)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if fence is not None:
                fence.validate(self.db)
            record_pk, duplicate = self.write_prepared(self.db, prepared)
            self.db.execute("COMMIT")
            return {"id": record_pk, "duplicate": duplicate}
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def write_prepared(self, db: sqlite3.Connection, prepared: Prepared, *, generation: int = 0) -> tuple[str, bool]:
        """Write one prepared envelope. Requires an ambient transaction."""
        existing = db.execute(
            "SELECT id, fingerprint FROM records WHERE source=? AND source_id=? AND revision=?",
            (prepared.source, prepared.source_id, prepared.revision),
        ).fetchone()
        if existing and existing["fingerprint"] == prepared.fingerprint:
            db.execute(
                "INSERT OR IGNORE INTO ingestion_receipts(id, record_id, observed_at, envelope) "
                "VALUES(?, ?, ?, ?)",
                (prepared.receipt_id, existing["id"], prepared.observed_at, prepared.receipt),
            )
            return existing["id"], True
        if existing:
            raise EvidenceError(
                "revision content conflict: the same (source, source_id, revision) "
                "already exists with different bytes; use a new revision"
            )

        for parent in prepared.parents:
            if not db.execute(
                "SELECT 1 FROM records WHERE id=? AND deleted=0", (parent,)
            ).fetchone():
                raise EvidenceError(
                    f"parent_record_id {parent!r} does not resolve to live evidence"
                )

        db.execute(
            """
            INSERT INTO records(id, source, source_id, revision, occurred_at, occurred_precision,
                                observed_at, ingested_at, kind, text, metadata, fingerprint, deleted)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,0)
            """,
            (
                prepared.id, prepared.source, prepared.source_id, prepared.revision,
                prepared.occurred_at, prepared.occurred_precision, prepared.observed_at,
                prepared.ingested_at, prepared.kind, prepared.text, prepared.metadata,
                prepared.fingerprint,
            ),
        )
        for parent in prepared.parents:
            db.execute(
                "INSERT OR IGNORE INTO record_dependencies(child_id, parent_id) VALUES(?,?)",
                (prepared.id, parent),
            )
        db.execute("INSERT INTO record_fts(id, text) VALUES(?, ?)", (prepared.id, prepared.text))
        db.execute(
            "INSERT INTO ingestion_receipts(id, record_id, observed_at, envelope) VALUES(?,?,?,?)",
            (prepared.receipt_id, prepared.id, prepared.observed_at, prepared.receipt),
        )
        self._audit("evidence_commit", prepared.id, {"source": prepared.source})
        journal(db, prepared.id, "add", source=prepared.source, generation=generation)
        return prepared.id, False

    def supersede(self, old_id: str, new_id: str, *, reason: str, actor: str) -> None:
        """Point a superseded revision at its replacement and drop it from search.

        The old row is preserved: 'what did we believe at the time' is a valid
        query and must remain answerable.
        """
        _required_text(reason, "reason", 500)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for candidate in (old_id, new_id):
                if not self.db.execute("SELECT 1 FROM records WHERE id=?", (candidate,)).fetchone():
                    raise EvidenceError(f"unknown record {candidate!r}")
            if old_id == new_id:
                raise EvidenceError("a record cannot supersede itself")
            self.db.execute(
                """
                INSERT INTO record_visibility(record_id, hidden, replacement_id, reason, changed_at)
                VALUES(?, 1, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    hidden=1, replacement_id=excluded.replacement_id,
                    reason=excluded.reason, changed_at=excluded.changed_at
                """,
                (old_id, new_id, reason, now()),
            )
            self.db.execute("DELETE FROM record_fts WHERE id=?", (old_id,))
            self._audit("evidence_supersede", old_id, {"actor": actor, "replacement": new_id})
            journal(self.db, old_id, "supersede")
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def hide(self, record_pk: str, *, reason: str, actor: str) -> None:
        """Remove evidence from retrieval without deleting the revision."""
        _required_text(reason, "reason", 500)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if not self.db.execute("SELECT 1 FROM records WHERE id=?", (record_pk,)).fetchone():
                raise EvidenceError(f"unknown record {record_pk!r}")
            self.db.execute(
                """
                INSERT INTO record_visibility(record_id, hidden, replacement_id, reason, changed_at)
                VALUES(?, 1, NULL, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    hidden=1, replacement_id=NULL, reason=excluded.reason, changed_at=excluded.changed_at
                """,
                (record_pk, reason, now()),
            )
            self.db.execute("DELETE FROM record_fts WHERE id=?", (record_pk,))
            self._audit("evidence_hide", record_pk, {"actor": actor})
            journal(self.db, record_pk, "hide")
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def show(self, record_pk: str, *, reason: str, actor: str) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM records WHERE id=?", (record_pk,)).fetchone()
            if row is None or row["deleted"]:
                raise EvidenceError(f"unknown or forgotten record {record_pk!r}")
            self.db.execute("DELETE FROM record_visibility WHERE record_id=?", (record_pk,))
            self.db.execute("DELETE FROM record_fts WHERE id=?", (record_pk,))
            self.db.execute("INSERT INTO record_fts(id, text) VALUES(?, ?)", (record_pk, row["text"]))
            self._audit("evidence_show", record_pk, {"actor": actor, "reason": reason})
            journal(self.db, record_pk, "show")
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def project(self, record_pk: str, revision: str, *, backend: str, bank_id: str, document_id: str) -> None:
        """Record that a canonical revision has a derived backend document.

        The mapping is explicit rather than derived by string parsing, and the
        backend document ID is underscore-free so chunk IDs stay reversible.
        """
        _required_text(backend, "backend", 100)
        _required_text(bank_id, "bank_id", 500)
        _required_text(document_id, "document_id", 200)
        if "_" in document_id or "~" in document_id:
            raise EvidenceError(
                "backend document_id must not contain '_' or '~': Hindsight's chunk-id "
                "escaping makes such ids ambiguous"
            )
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if not self.db.execute("SELECT 1 FROM records WHERE id=?", (record_pk,)).fetchone():
                raise EvidenceError(f"unknown record {record_pk!r}")
            epoch = self.epoch()
            self.db.execute(
                """
                INSERT INTO backend_documents(record_id, revision, backend, bank_id, document_id,
                                              desired_epoch, state)
                VALUES(?,?,?,?,?,?, 'pending')
                ON CONFLICT(record_id, revision, backend, bank_id) DO UPDATE SET
                    desired_epoch=excluded.desired_epoch, state='pending',
                    operation_id=NULL, confirmed_at=NULL, error=NULL
                """,
                (record_pk, revision, backend, bank_id, document_id, epoch),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def pending_projections(self, *, backend: str, bank_id: str, limit: int = 50) -> list[sqlite3.Row]:
        limit = _bounded_limit(limit, maximum=500)
        return self.db.execute(
            """
            SELECT * FROM backend_documents
            WHERE backend=? AND bank_id=? AND state IN ('pending', 'retry')
            ORDER BY desired_epoch, record_id LIMIT ?
            """,
            (backend, bank_id, limit),
        ).fetchall()

    def set_control(self, scope: str, stage: str, state: str, *, actor: str, reason: str, policy_version: str) -> None:
        """Persist an operator pause/resume with its authority and reason."""
        if state not in {"active", "paused"}:
            raise EvidenceError("state must be 'active' or 'paused'")
        _required_text(actor, "actor", 500)
        _required_text(reason, "reason", 1000)
        self.db.execute(
            """
            INSERT INTO runtime_controls(scope, stage, state, actor, reason, policy_version, changed_at)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(scope, stage) DO UPDATE SET
                state=excluded.state, actor=excluded.actor, reason=excluded.reason,
                policy_version=excluded.policy_version, changed_at=excluded.changed_at
            """,
            (scope, stage, state, actor, reason, policy_version, now()),
        )

    def stage_is_paused(self, scope: str, stage: str) -> bool:
        row = self.db.execute(
            "SELECT state FROM runtime_controls WHERE scope=? AND stage=?", (scope, stage)
        ).fetchone()
        return bool(row and row[0] == "paused")

    def _audit(self, action: str, object_id: str, metadata: dict[str, Any]) -> None:
        # Never log raw evidence bodies; ids and fingerprints are enough to diagnose.
        self.db.execute(
            "INSERT INTO audit(action, object_id, created_at, metadata) VALUES(?,?,?,?)",
            (action, object_id, now(), json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        )


def _bounded_limit(value: int, *, maximum: int = 200) -> int:
    if not isinstance(value, int) or not 1 <= value <= maximum:
        raise EvidenceError(f"limit must be an integer between 1 and {maximum}")
    return value


def visible_ids(store: EvidenceStore, candidates: Iterable[str]) -> list[str]:
    return [candidate for candidate in candidates if store.live_and_visible(candidate)]
