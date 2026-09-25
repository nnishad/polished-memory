"""Explicit schema migrations.

The retired framework called ``executescript`` from inside runtime write paths,
so a processing write could carry DDL and inherit its caller's transaction.
Here schema changes happen only in ``apply_migrations``, before any writer runs.

Each migration is a tuple of individual statements rather than one script
string: ``sqlite3.Connection.executescript`` issues an implicit COMMIT for any
pending transaction, which would silently dismantle the ``BEGIN IMMEDIATE``
that wraps the migration and make a partial failure non-atomic.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable, Sequence

__all__ = ["Migration", "MIGRATIONS", "apply_migrations", "current_version", "connect"]

MIGRATION_LEDGER = "CREATE TABLE schema_migrations(name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"

EVIDENCE_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE records(
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        source_id TEXT NOT NULL,
        revision TEXT NOT NULL,
        occurred_at TEXT,
        occurred_precision TEXT NOT NULL DEFAULT 'unknown',
        observed_at TEXT NOT NULL,
        ingested_at TEXT NOT NULL,
        kind TEXT NOT NULL,
        text TEXT NOT NULL,
        metadata TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        deleted INTEGER NOT NULL DEFAULT 0,
        UNIQUE(source, source_id, revision)
    )""",
    "CREATE INDEX records_source_time ON records(source, occurred_at, id)",
    "CREATE INDEX records_ingested ON records(deleted, ingested_at, id)",
    """
    CREATE TABLE record_visibility(
        record_id TEXT PRIMARY KEY REFERENCES records(id),
        hidden INTEGER NOT NULL,
        replacement_id TEXT REFERENCES records(id),
        reason TEXT NOT NULL,
        changed_at TEXT NOT NULL
    )""",
    """
    CREATE TABLE record_dependencies(
        child_id TEXT NOT NULL REFERENCES records(id),
        parent_id TEXT NOT NULL REFERENCES records(id),
        PRIMARY KEY(child_id, parent_id)
    )""",
    "CREATE INDEX dependencies_parent ON record_dependencies(parent_id)",
    """
    CREATE TABLE ingestion_receipts(
        id TEXT PRIMARY KEY,
        record_id TEXT NOT NULL REFERENCES records(id),
        observed_at TEXT NOT NULL,
        envelope TEXT NOT NULL
    )""",
    "CREATE INDEX receipts_record_time ON ingestion_receipts(record_id, observed_at)",
    """
    CREATE TABLE audit(
        id INTEGER PRIMARY KEY,
        action TEXT NOT NULL,
        object_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        metadata TEXT NOT NULL
    )""",
    "CREATE VIRTUAL TABLE record_fts USING fts5(id UNINDEXED, text)",
    """
    CREATE TABLE backend_documents(
        record_id TEXT NOT NULL REFERENCES records(id),
        revision TEXT NOT NULL,
        backend TEXT NOT NULL,
        bank_id TEXT NOT NULL,
        document_id TEXT NOT NULL,
        desired_epoch INTEGER NOT NULL,
        state TEXT NOT NULL,
        operation_id TEXT,
        confirmed_at TEXT,
        error TEXT
    )""",
    "CREATE UNIQUE INDEX backend_documents_key ON backend_documents(record_id, revision, backend, bank_id)",
    "CREATE INDEX backend_documents_state ON backend_documents(backend, state, desired_epoch)",
    "CREATE UNIQUE INDEX backend_documents_docid ON backend_documents(backend, bank_id, document_id)",
    "CREATE TABLE memory_epoch(id INTEGER PRIMARY KEY CHECK(id = 1), value INTEGER NOT NULL)",
    "INSERT INTO memory_epoch VALUES(1, 1)",
    """
    CREATE TABLE runtime_controls(
        scope TEXT NOT NULL,
        stage TEXT NOT NULL,
        state TEXT NOT NULL,
        actor TEXT NOT NULL,
        reason TEXT NOT NULL,
        policy_version TEXT NOT NULL,
        changed_at TEXT NOT NULL,
        PRIMARY KEY(scope, stage)
    )""",
)


SYNC_STATEMENTS: tuple[str, ...] = (
    """
    -- One row per connector. ``generation`` is bumped by reconfigure/reset so a
    -- connector process still holding an old lease can never write into the new
    -- source's identity.
    CREATE TABLE connectors(
        source TEXT PRIMARY KEY,
        generation INTEGER NOT NULL DEFAULT 1,
        policy_version TEXT NOT NULL,
        lease TEXT,
        lease_until REAL,
        holder TEXT,
        cursor TEXT,
        cursor_kind TEXT NOT NULL DEFAULT 'opaque',
        coverage_state TEXT NOT NULL DEFAULT 'unknown',
        last_success_at TEXT,
        updated_at TEXT NOT NULL
    )""",
    """
    -- Page-level idempotency: a retried page is recognised before any write,
    -- so a crash between COMMIT and acknowledgement replays as a no-op.
    CREATE TABLE source_pages(
        source TEXT NOT NULL,
        generation INTEGER NOT NULL,
        page_token TEXT NOT NULL,
        record_count INTEGER NOT NULL,
        fingerprint TEXT NOT NULL,
        committed_at TEXT NOT NULL,
        PRIMARY KEY(source, generation, page_token)
    )""",
    """
    CREATE TABLE change_journal(
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        generation INTEGER NOT NULL,
        epoch INTEGER NOT NULL,
        record_id TEXT NOT NULL,
        change TEXT NOT NULL,
        committed_at TEXT NOT NULL
    )""",
    "CREATE INDEX journal_record ON change_journal(record_id, seq)",
    """
    -- Consumers checkpoint independently and replay idempotently, so one slow
    -- derived index cannot hold the upstream cursor back.
    CREATE TABLE consumer_checkpoints(
        consumer TEXT PRIMARY KEY,
        seq INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    )""",
)


@dataclass(frozen=True)
class Migration:
    name: str
    statements: tuple[str, ...]


MIGRATIONS: Sequence[Migration] = (
    Migration("0001_evidence", EVIDENCE_STATEMENTS),
    Migration("0002_sync", SYNC_STATEMENTS),
)


def connect(path) -> sqlite3.Connection:
    """Open a canonical store connection.

    ``isolation_level=None`` gives us explicit transaction control; the default
    implicit-begin mode would let a DDL statement commit a caller's partially
    written transaction.
    """
    db = sqlite3.connect(path, timeout=10, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    return db


def current_version(db: sqlite3.Connection) -> int:
    exists = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    if not exists:
        return 0
    return int(db.execute("SELECT count(*) FROM schema_migrations").fetchone()[0])


def apply_migrations(db: sqlite3.Connection, *, at: Callable[[], str]) -> int:
    """Bring *db* to head. Idempotent; refuses an unknown future version."""
    db.execute("BEGIN IMMEDIATE")
    try:
        if not db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        ).fetchone():
            db.execute(MIGRATION_LEDGER)
        applied = {row[0] for row in db.execute("SELECT name FROM schema_migrations")}
        known = {migration.name for migration in MIGRATIONS}
        for name in sorted(applied - known):
            raise RuntimeError(
                f"database carries migration {name!r} unknown to this build; refusing to "
                "run an older application against a newer schema"
            )
        for migration in MIGRATIONS:
            if migration.name in applied:
                continue
            for statement in migration.statements:
                db.execute(statement)
            db.execute(
                "INSERT INTO schema_migrations(name, applied_at) VALUES(?, ?)",
                (migration.name, at()),
            )
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")
    return len(MIGRATIONS)
