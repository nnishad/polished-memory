"""C2 consumer side: the plugin's durable capture spool, read as a source of events.

``sources.sdk.HermesEvents`` deliberately takes a drain rather than a path: the spool file
belongs to the plugin, and one component's durability boundary must not become another's
private schema. This module is where that boundary is honoured — it is the only place in the
package that knows the spool's column names, and it knows them as a contract it checks rather
than a layout it assumes.

Position is the cursor the connector stores, and a cursor is opaque: rows are ordered by the
spool's own insertion order, never by comparing event ids, which are the host's strings. The
spool's ``state`` column is the plugin's retry bookkeeping, and this reader answers in the
vocabulary it finds there rather than inventing its own: a row is retired as ``settled``, the
same word the plugin's own settle uses, because that is the state its compaction reclaims and
the state its ``counts()`` already knows how to name. A private word here would keep every row
forever and report a state no component understands.

A row is retired only once a *later* drain has been called past it, because the connector
advances its cursor after committing and a cursor that never moved must still be able to re-read
the rows. Only rows the plugin is still holding are touched: a row another worker has claimed is
left alone. A read with no position at all starts at the beginning of the file, retired rows
included — the connector clearing a cursor is it saying it does not want anything skipped.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

__all__ = ["SPOOL_SOURCE", "spool_beside", "spool_drain", "spool_backlog", "SpoolError"]

_COLUMNS = ("event_id", "session_id", "payload", "created_at")
_PENDING = "pending"
# The plugin's own word for "this event is done with", not this reader's.
_SETTLED = "settled"
_SPOOL_DIRECTORY = "hermes-memory"
_SPOOL_NAME = "capture-spool.db"
# The name the connector files these events under. It is the source the SDK adapter declares,
# and readers that want to ask "has anything ever drained this spool" ask it of that one
# connector rather of whichever ones happen to be registered.
SPOOL_SOURCE = "hermes"


class SpoolError(RuntimeError):
    """The file is not a capture spool this reader may safely walk."""


def spool_beside(store_path: Path | str) -> Path:
    """Where the plugin writing for this profile puts its spool.

    Derived from the canonical store rather than from a setting, because the two are already
    fixed relative to each other by the plugin: a configured path would be a second place to
    keep that fact, and the first one to drift.
    """
    return Path(store_path).parent / _SPOOL_DIRECTORY / _SPOOL_NAME


def _open_existing(path: Path | str) -> sqlite3.Connection:
    """Open without creating: an absent spool is an absence, not a file to make.

    A reader that created its own empty database would report a drained stream at exactly the
    moment the plugin had never written one, and the install would look healthy. The connection
    is read-write because the spool is a WAL database — a read-only handle cannot always open
    one — and nothing here writes except the marking of rows a later drain has passed.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise SpoolError(f"no capture spool at {resolved}")
    uri = f"file:{resolved.resolve()}?mode=rw"
    try:
        db = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError as error:
        raise SpoolError(f"the capture spool at {resolved} cannot be opened: {error}") from error
    db.row_factory = sqlite3.Row
    return db


def _require_spool(db: sqlite3.Connection, path: Path) -> None:
    """Refuse a file that is not a spool, before any of it is read as one.

    The caller owns closing the connection; this only says when the layout is not the one the
    contract claims, which is a different fault from an empty stream.
    """
    table = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='spool'"
                       ).fetchone()
    if table is None:
        raise SpoolError(f"{path} has no 'spool' table; it is not a capture spool")
    names = {row[1] for row in db.execute("PRAGMA table_info(spool)")}
    missing = [name for name in _COLUMNS if name not in names]
    if missing:
        raise SpoolError(f"the capture spool at {path} is missing column(s): "
                         f"{', '.join(missing)}")


def spool_drain(path: Path | str, *, limit_default: int = 200) -> Callable[..., list[dict]]:
    """A ``drain(after, limit)`` over the spool at ``path``, oldest row first.

    The returned callable is what ``HermesEvents`` expects. Each event carries the columns the
    adapter needs plus nothing else: payloads are parsed, because the adapter reads a mapping
    and the spool stores JSON text it never re-reads.
    """
    resolved = Path(path)
    if not 1 <= int(limit_default) <= 1000:
        raise ValueError("limit_default must be between 1 and 1000")

    def drain(after: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        limit = limit_default if limit is None else int(limit)
        if limit < 1:
            return []
        db = _open_existing(resolved)
        try:
            _require_spool(db, resolved)
            if after:
                anchor = db.execute("SELECT rowid AS r FROM spool WHERE event_id=?",
                                    (after,)).fetchone()
                if anchor is None:
                    # The cursor names an event the spool no longer holds. Saying so is the
                    # connector's cue to start from the beginning of what remains, not to
                    # assume an empty answer.
                    from .base import CursorExpired
                    raise CursorExpired(f"the capture spool no longer holds {after!r}")
                db.execute("UPDATE spool SET state=?, last_error=NULL "
                           "WHERE state=? AND rowid <= ?",
                           (_SETTLED, _PENDING, anchor["r"]))
                rows = db.execute(
                    "SELECT event_id, session_id, payload, created_at FROM spool "
                    "WHERE rowid > ? ORDER BY rowid LIMIT ?", (anchor["r"], limit)).fetchall()
            else:
                # No position means the connector has none to keep: a first read, or a
                # reconfigure that cleared one on purpose. Both are answered from the start of
                # the file, retired rows included, because `reconfigure` promises that a stale
                # cursor will not silently skip evidence and the store's dedupe makes re-reading
                # what is already there cost nothing but a page.
                rows = db.execute(
                    "SELECT event_id, session_id, payload, created_at FROM spool "
                    "ORDER BY rowid LIMIT ?", (limit,)).fetchall()
            db.commit()
        finally:
            db.close()
        events = []
        for row in rows:
            try:
                payload = json.loads(row["payload"] or "{}")
            except ValueError:
                payload = {"kind": "undecodable", "raw": str(row["payload"])[:2000]}
            events.append({"event_id": row["event_id"], "session_id": row["session_id"],
                           "created_at": row["created_at"], "payload": payload})
        return events

    return drain


def spool_backlog(path: Path | str) -> dict[str, Any]:
    """What the spool is holding, without reading a byte of conversation content.

    A reading that has to say whether capture is working can ask this: pending rows with no
    consumer are exactly the turns an installation would otherwise lose silently.
    """
    resolved = Path(path)
    if not resolved.exists():
        return {"spool": str(resolved), "present": False, "pending": 0, "settled": 0,
                "states": {}, "oldest_at": None}
    db = _open_existing(resolved)
    try:
        _require_spool(db, resolved)
        rows = db.execute("SELECT state, count(*) AS n, min(created_at) AS oldest "
                          "FROM spool GROUP BY state").fetchall()
    finally:
        db.close()
    counts = {row["state"]: row["n"] for row in rows}
    oldest = {row["state"]: row["oldest"] for row in rows if row["oldest"]}
    return {"spool": str(resolved), "present": True,
            "pending": int(counts.get(_PENDING, 0)),
            "settled": int(counts.get(_SETTLED, 0)),
            "states": counts,
            "oldest_at": oldest.get(_PENDING) or None}
