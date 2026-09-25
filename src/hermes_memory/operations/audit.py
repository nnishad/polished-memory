"""C14 audit: who did what to this evidence, in an order that can be checked.

The store already writes an audit row beside every consequential change — that is
half of why the framework bothers with explicit transactions. This is the other half:
reading it back so an operator can answer "when did this stop being private?" or
"which of these confirmations was really the owner?" without spelunking SQL.

Two rules shape everything here. Nothing is rendered that could carry a secret:
metadata is parsed, bounded and passed through the same redaction a record gets on
the way in, and a record's own text is never included unless someone asks for it by
name. And a filter that matches nothing returns nothing — a report that quietly
ignores the filter it was handed and prints everything is worse than no report,
because it looks like an answer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..ids import now
from ..sources.base import redact_secrets
from ..storage.evidence import EvidenceError

__all__ = ["AuditTrail", "MAX_DETAIL_CHARS", "MAX_METADATA_FIELDS"]

MAX_DETAIL_CHARS = 400
MAX_METADATA_FIELDS = 12
DEFAULT_LIMIT = 50
MAX_LIMIT = 1000


@dataclass(frozen=True)
class Entry:
    """One audit row, rendered for a human without adding anything to it."""

    id: int
    action: str
    object_id: str
    at: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "action": self.action, "object_id": self.object_id,
                "at": self.at, **self.metadata}


class AuditTrail:
    """Read-only view of the audit ledger and the change journal."""

    def __init__(self, store):
        self.store = store
        self.db = store.db

    # -- the ledger ----------------------------------------------------------

    def recent(self, *, action: str | None = None, object_id: str | None = None,
               actor: str | None = None, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
        """Newest first. Every filter that is given is applied; none of them is optional."""
        _check_limit(limit)
        clauses, params = [], []
        if action is not None:
            clauses.append("action=?")
            params.append(_text(action, "action"))
        if object_id is not None:
            clauses.append("object_id=?")
            params.append(_text(object_id, "object id"))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.execute(
            f"SELECT * FROM audit{where} ORDER BY id DESC LIMIT ?",
            [*params, limit]).fetchall()
        entries = [_entry(row) for row in rows]
        if actor is not None:
            wanted = _text(actor, "actor")
            entries = [item for item in entries if item.metadata.get("actor") == wanted]
        return [item.as_dict() for item in entries]

    def for_object(self, object_id: str, *, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
        return self.recent(object_id=object_id, limit=limit)

    def actions(self) -> dict[str, int]:
        """What kinds of thing have happened here at all, and how often."""
        return {row[0]: int(row[1]) for row in self.db.execute(
            "SELECT action, count(*) FROM audit GROUP BY action ORDER BY action")}

    def by_actor(self) -> dict[str, int]:
        """Who is acting in this system, counted from the rows that say so.

        An owner-only decision and an agent's proposal are different kinds of fact, and
        a mix of the two in one column is the thing an auditor is asked about first.
        """
        rows = self.db.execute("SELECT metadata FROM audit ORDER BY id DESC LIMIT ?",
                               (MAX_LIMIT,)).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            actor = _metadata(row["metadata"]).get("actor")
            if actor:
                counts[str(actor)] = counts.get(str(actor), 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def decided_by(self, kind: str) -> list[dict[str, Any]]:
        """Owner-only decisions — confirmations, revocations, erasures — by category.

        ``kind`` is one of the owner-only tables, not a free-form word: an answer that
        could be asked about anything would have to guess at what it was shown.
        """
        table, column = _OWNER_ONLY.get(kind or "", ("", ""))
        if not table:
            raise EvidenceError(f"unknown decision category {kind!r}; ask about one of "
                                f"{sorted(_OWNER_ONLY)}")
        rows = self.db.execute(
            f"SELECT {column} AS decided_by, count(*) AS n FROM {table} "
            f"WHERE {column} IS NOT NULL GROUP BY {column} ORDER BY count(*) DESC, "
            f"{column}").fetchall()
        return [{"actor": row["decided_by"], "decisions": int(row["n"])} for row in rows]

    # -- one thing's history -------------------------------------------------

    def timeline(self, reference: str, *, include_text: bool = False,
                 limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
        """Everything the store can say about one record, assembled from its own rows.

        The record's text is left out unless *include_text* is set, and even then it
        is redacted: a timeline is something an operator pastes into a message, and
        the evidence itself is the part that must not travel.
        """
        record_id = _text(reference, "record reference")
        row = self.db.execute("SELECT * FROM records WHERE id=?", (record_id,)).fetchone()
        if row is None:
            raise EvidenceError(f"no record {record_id!r} in this store")
        visibility = self.db.execute(
            "SELECT * FROM record_visibility WHERE record_id=?", (record_id,)).fetchone()
        tombstone = self.db.execute(
            "SELECT * FROM tombstones WHERE record_id=?", (record_id,)).fetchone()
        indexed = self.db.execute(
            "SELECT 1 FROM record_fts WHERE id=?", (record_id,)).fetchone()
        out: dict[str, Any] = {
            "record_id": record_id,
            "assembled_at": now(),
            "evidence": {
                "source": row["source"], "source_id": row["source_id"],
                "revision": row["revision"], "kind": row["kind"],
                "occurred_at": row["occurred_at"],
                "occurred_precision": row["occurred_precision"],
                "observed_at": row["observed_at"], "ingested_at": row["ingested_at"],
                "fingerprint": row["fingerprint"], "forgotten": bool(row["deleted"]),
                "metadata": _metadata(row["metadata"])},
            "retrievable": {
                "in_search_index": bool(indexed),
                "hidden": bool(visibility and visibility["hidden"]),
                "hidden_reason": visibility["reason"] if visibility else None,
                "replaced_by": visibility["replacement_id"] if visibility else None,
                "tombstoned_at": tombstone["deleted_at"] if tombstone else None},
            "supported_by": [dict(item) for item in self.db.execute(
                "SELECT a.id, a.subject, a.predicate, a.value, a.status, a.evidence_kind, "
                "a.quote_start, a.quote_end FROM assertions a WHERE a.record_id=? "
                "ORDER BY a.created_at, a.id", (record_id,)).fetchall()],
            "built_into": [dict(item) for item in self.db.execute(
                "SELECT c.artifact_id, c.kind, c.coverage, c.quote_start, c.quote_end, "
                "c.added_at FROM derived_citations c WHERE c.record_id=? "
                "ORDER BY c.added_at, c.artifact_id", (record_id,)).fetchall()],
            "changes": [dict(item) for item in self.db.execute(
                "SELECT seq, source, generation, epoch, change, committed_at "
                "FROM change_journal WHERE record_id=? ORDER BY seq", (record_id,)).fetchall()],
            "receipts": [{"observed_at": item["observed_at"]} for item in self.db.execute(
                "SELECT observed_at FROM ingestion_receipts WHERE record_id=? "
                "ORDER BY observed_at, id LIMIT ?", (record_id, limit)).fetchall()],
            "audit": self.recent(object_id=record_id, limit=limit),
        }
        if include_text:
            out["text"] = redact_secrets(str(row["text"]))[:MAX_DETAIL_CHARS]
        return out

    def source_history(self, source: str, *, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
        """What a connector has done: pages committed, gaps recorded, controls applied.

        Page rows are the only account of what was *offered*, as opposed to what was
        accepted, so a source that keeps replaying the same page shows up here rather
        than looking idle.
        """
        name = _text(source, "source")
        connector = self.db.execute("SELECT * FROM connectors WHERE source=?",
                                    (name,)).fetchone()
        if connector is None:
            raise EvidenceError(f"no connector named {name!r} is registered")
        return {
            "source": name,
            "assembled_at": now(),
            "connector": {key: connector[key] for key in connector.keys()},
            "pages": [{"page_token": row["page_token"], "generation": row["generation"],
                       "records": int(row["record_count"]),
                       "committed_at": row["committed_at"]}
                      for row in self.db.execute(
                          "SELECT * FROM source_pages WHERE source=? ORDER BY committed_at "
                          "DESC, page_token LIMIT ?", (name, limit)).fetchall()],
            "gaps": [dict(row) for row in self.db.execute(
                "SELECT generation, ref, reason, first_seen_at, last_seen_at, cleared_at "
                "FROM source_gaps WHERE source=? ORDER BY first_seen_at, ref LIMIT ?",
                (name, limit)).fetchall()],
            "controls": [dict(row) for row in self.db.execute(
                "SELECT stage, state, actor, reason, policy_version, changed_at "
                "FROM runtime_controls WHERE scope=? ORDER BY changed_at, stage",
                (name,)).fetchall()],
        }

    def journal(self, *, after: int = 0, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
        """Changes since a sequence number. ``after`` is inclusive of nothing: it is
        the point the caller last saw, so the answer starts one past it."""
        _check_limit(limit)
        if not isinstance(after, int) or after < 0:
            raise EvidenceError("after must be a nonnegative journal sequence number")
        return [dict(row) for row in self.db.execute(
            "SELECT seq, source, generation, epoch, record_id, change, committed_at "
            "FROM change_journal WHERE seq > ? ORDER BY seq LIMIT ?",
            (after, limit)).fetchall()]


_OWNER_ONLY = {
    "identity": ("identity_edges", "confirmed_by"),
    "identity-decisions": ("identity_candidates", "decided_by"),
    "goals": ("goals", "decided_by"),
    "lessons": ("lessons", "decided_by"),
    "erasure": ("erasure_ledger", "confirmed_by"),
}


def _entry(row) -> Entry:
    return Entry(id=int(row["id"]), action=str(row["action"]),
                 object_id=str(row["object_id"]), at=str(row["created_at"]),
                 metadata=_metadata(row["metadata"]))


def _metadata(raw: Any) -> dict[str, Any]:
    """Parse, bound and redact one metadata blob. It never carries more than names."""
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return {"unparseable": True}
    if not isinstance(parsed, dict):
        return {"unexpected_shape": str(type(parsed).__name__)}
    out: dict[str, Any] = {}
    for index, (key, value) in enumerate(sorted(parsed.items(), key=lambda item: str(item[0]))):
        if index >= MAX_METADATA_FIELDS:
            out["additional_fields"] = len(parsed) - MAX_METADATA_FIELDS
            break
        out[str(key)] = _scalar(value)
    return out


def _scalar(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return redact_secrets(str(value))[:MAX_DETAIL_CHARS]
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{label} must be nonempty text")
    return value.strip()[:200]


def _check_limit(limit: Any) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
        raise EvidenceError(f"limit must be between 1 and {MAX_LIMIT} rows")
