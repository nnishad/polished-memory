"""Durable capture spool.

Hermes calls ``sync_turn`` asynchronously and best-effort, and drains for only
a few seconds at shutdown: a returned chat answer is not proof the provider
committed anything. This spool is the durability boundary — once an event is
fsynced here, capture succeeded, and the backend projection can happen later
or not at all.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS spool(
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS spool_pending ON spool(state, created_at, event_id);
"""

_MAXIMUM_BATCH = 500


class CaptureSpool:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        # FULL, not NORMAL: WAL with NORMAL does not fsync on commit, so a
        # power loss could drop accepted turns.
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def append(self, *, event_id: str, session_id: str, payload: dict[str, Any], created_at: str) -> bool:
        """Record one capture event. Returns False if it was already present."""
        if not event_id or len(event_id) > 200:
            raise ValueError("event_id must be 1..200 characters")
        cursor = self.db.execute(
            "INSERT OR IGNORE INTO spool(event_id, session_id, payload, created_at) VALUES(?,?,?,?)",
            (event_id, session_id, json.dumps(payload, ensure_ascii=False, sort_keys=True), created_at),
        )
        self.db.commit()
        return cursor.rowcount == 1

    def pending(self, *, limit: int = 50) -> list[sqlite3.Row]:
        if not 1 <= limit <= _MAXIMUM_BATCH:
            raise ValueError(f"limit must be between 1 and {_MAXIMUM_BATCH}")
        return self.db.execute(
            "SELECT * FROM spool WHERE state='pending' ORDER BY created_at, event_id LIMIT ?",
            (limit,),
        ).fetchall()

    def claim(self, event_id: str) -> bool:
        """Move pending -> in_flight so a second worker cannot double-submit."""
        cursor = self.db.execute(
            "UPDATE spool SET state='in_flight', attempts=attempts+1 WHERE event_id=? AND state='pending'",
            (event_id,),
        )
        self.db.commit()
        return cursor.rowcount == 1

    def settle(self, event_id: str, *, ok: bool, error: str | None = None) -> None:
        state = "settled" if ok else "pending"
        self.db.execute(
            "UPDATE spool SET state=?, last_error=? WHERE event_id=?",
            (state, None if ok else (error or "")[:500], event_id),
        )
        self.db.commit()

    def forget_settled_before(self, cutoff: str) -> int:
        cursor = self.db.execute("DELETE FROM spool WHERE state='settled' AND created_at < ?", (cutoff,))
        self.db.commit()
        return cursor.rowcount

    def counts(self) -> dict[str, int]:
        return {
            row[0]: row[1]
            for row in self.db.execute("SELECT state, count(*) FROM spool GROUP BY state")
        }

    def iter_all(self) -> Iterator[sqlite3.Row]:
        yield from self.db.execute("SELECT * FROM spool ORDER BY created_at, event_id")
