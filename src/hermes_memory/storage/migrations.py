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


ERASURE_STATEMENTS: tuple[str, ...] = (
    """
    -- The durable erasure ledger. It sits outside ordinary snapshot restore
    -- scope: a restore that replays old records must apply this ledger's newest
    -- state afterwards, or forgetting an item would be undone by a backup.
    CREATE TABLE erasure_ledger(
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        requested_at TEXT NOT NULL,
        requested_by TEXT NOT NULL,
        requester_kind TEXT NOT NULL,
        reason TEXT NOT NULL,
        preview TEXT NOT NULL,
        preview_digest TEXT NOT NULL,
        state TEXT NOT NULL,
        confirmed_at TEXT,
        confirmed_by TEXT,
        completed_at TEXT,
        epoch INTEGER NOT NULL
    )""",
    "CREATE INDEX erasure_ledger_state ON erasure_ledger(state, requested_at)",
    """
    -- One row per object that has to physically disappear, including the
    -- derived copies in the backend. 'complete' means every row verified.
    CREATE TABLE erasure_targets(
        intent_id TEXT NOT NULL REFERENCES erasure_ledger(id),
        kind TEXT NOT NULL,
        reference TEXT NOT NULL,
        state TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        verified_at TEXT,
        error TEXT,
        PRIMARY KEY(intent_id, kind, reference)
    )""",
    "CREATE INDEX erasure_targets_state ON erasure_targets(state, intent_id)",
    """
    -- A tombstone outlives the record it hides so a restore can tell 'never
    -- existed' apart from 'was forgotten and must not come back'.
    CREATE TABLE tombstones(
        record_id TEXT PRIMARY KEY REFERENCES records(id),
        intent_id TEXT NOT NULL REFERENCES erasure_ledger(id),
        fingerprint TEXT NOT NULL,
        deleted_at TEXT NOT NULL
    )""",
    "CREATE INDEX tombstones_intent ON tombstones(intent_id)",
)


IDENTITY_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE identity_accounts(
        id TEXT PRIMARY KEY,
        namespace TEXT NOT NULL,
        identifier TEXT NOT NULL,
        normalized TEXT NOT NULL,
        label TEXT,
        state TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        UNIQUE(namespace, identifier)
    )""",
    "CREATE UNIQUE INDEX identity_accounts_norm ON identity_accounts(namespace, normalized)",
    """
    -- A proposal is not an identity. It records the deterministic rule that
    -- produced it and the evidence, so an owner can judge it later.
    CREATE TABLE identity_candidates(
        id TEXT PRIMARY KEY,
        account_a TEXT NOT NULL REFERENCES identity_accounts(id),
        account_b TEXT NOT NULL REFERENCES identity_accounts(id),
        rule TEXT NOT NULL,
        rule_version TEXT NOT NULL,
        basis TEXT NOT NULL,
        evidence TEXT NOT NULL,
        proposed_by TEXT NOT NULL,
        proposed_kind TEXT NOT NULL,
        proposed_at TEXT NOT NULL,
        state TEXT NOT NULL,
        decided_by TEXT,
        decided_at TEXT,
        decision_reason TEXT,
        UNIQUE(account_a, account_b)
    )""",
    "CREATE INDEX identity_candidates_state ON identity_candidates(state, proposed_at)",
    """
    CREATE TABLE identity_edges(
        id TEXT PRIMARY KEY,
        account_a TEXT NOT NULL REFERENCES identity_accounts(id),
        account_b TEXT NOT NULL REFERENCES identity_accounts(id),
        candidate_id TEXT NOT NULL REFERENCES identity_candidates(id),
        valid_from TEXT,
        valid_until TEXT,
        state TEXT NOT NULL,
        confirmed_by TEXT NOT NULL,
        confirmed_at TEXT NOT NULL,
        revoked_at TEXT,
        revoked_by TEXT,
        revocation_reason TEXT,
        CHECK(valid_until IS NULL OR valid_from IS NOT NULL)
    )""",
    "CREATE INDEX identity_edges_pair ON identity_edges(account_a, account_b, state)",
    """
    -- Structural topics and model labels live together but are never
    -- interchangeable: kind distinguishes them and only structural rows may
    -- feed a deterministic rule.
    CREATE TABLE topic_links(
        id TEXT PRIMARY KEY,
        account_id TEXT NOT NULL REFERENCES identity_accounts(id),
        topic TEXT NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('structural', 'label')),
        source_record_id TEXT,
        observed_at TEXT NOT NULL,
        UNIQUE(account_id, topic, kind)
    )""",
)


PROCESSING_STATEMENTS: tuple[str, ...] = (
    """
    -- One row per physical resource in use. The partial unique index is the
    -- whole design: SQLite itself refuses a second concurrent holder, so the
    -- single-slot rule survives multiple processes and profiles without any
    -- in-memory lock that a crash could orphan.
    CREATE TABLE gate_reservations(
        id TEXT PRIMARY KEY,
        resource TEXT NOT NULL,
        route TEXT NOT NULL,
        holder TEXT NOT NULL,
        priority INTEGER NOT NULL,
        state TEXT NOT NULL,
        acquired_at REAL NOT NULL,
        lease_until REAL NOT NULL,
        job_id TEXT,
        released_at REAL,
        outcome TEXT
    )""",
    """
    CREATE UNIQUE INDEX gate_single_slot ON gate_reservations(resource)
        WHERE state IN ('held', 'uncertain')""",
    "CREATE INDEX gate_lease ON gate_reservations(state, lease_until)",
    """
    CREATE TABLE gate_ledger(
        id INTEGER PRIMARY KEY,
        reservation_id TEXT NOT NULL,
        resource TEXT NOT NULL,
        route TEXT NOT NULL,
        holder TEXT NOT NULL,
        event TEXT NOT NULL,
        at REAL NOT NULL,
        detail TEXT NOT NULL
    )""",
    "CREATE INDEX gate_ledger_reservation ON gate_ledger(reservation_id)",
    """
    CREATE TABLE processing_jobs(
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        state TEXT NOT NULL,
        priority INTEGER NOT NULL,
        resource TEXT NOT NULL,
        route TEXT NOT NULL,
        epoch INTEGER NOT NULL,
        processor_fingerprint TEXT NOT NULL,
        inputs TEXT NOT NULL,
        input_revision TEXT NOT NULL,
        submission_id TEXT,
        backend_operation_id TEXT,
        backend_state TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL,
        tokens_used INTEGER NOT NULL DEFAULT 0,
        token_budget INTEGER NOT NULL,
        deadline REAL,
        not_before REAL,
        lease TEXT,
        lease_until REAL,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT
    )""",
    "CREATE INDEX jobs_dispatch ON processing_jobs(state, priority, created_at)",
    "CREATE INDEX jobs_submission ON processing_jobs(submission_id)",
    """
    -- Budget consumption is measured, including internal retries, so a job that
    -- keeps producing truncated output runs out rather than looping.
    CREATE TABLE budget_usage(
        scope TEXT NOT NULL,
        period TEXT NOT NULL,
        resource TEXT NOT NULL,
        tokens INTEGER NOT NULL DEFAULT 0,
        calls INTEGER NOT NULL DEFAULT 0,
        seconds REAL NOT NULL DEFAULT 0,
        PRIMARY KEY(scope, period, resource)
    )""",
)


BLOB_STATEMENTS: tuple[str, ...] = (
    """
    -- One row per distinct byte string, referenced by every attachment that
    -- points at it. Chunks belong to the content, not to a record, so a
    -- forwarded attachment cannot be destroyed by forgetting one of its
    -- carriers — and cannot be kept alive by accident either, because refs is
    -- the only thing standing between an orphan and the vacuum.
    CREATE TABLE blob_contents(
        sha256 TEXT PRIMARY KEY,
        size INTEGER NOT NULL CHECK(size > 0),
        refs INTEGER NOT NULL DEFAULT 0 CHECK(refs >= 0),
        first_seen TEXT NOT NULL
    )""",
    """
    CREATE TABLE blob_chunks(
        sha256 TEXT NOT NULL REFERENCES blob_contents(sha256),
        chunk_index INTEGER NOT NULL,
        data BLOB NOT NULL,
        PRIMARY KEY(sha256, chunk_index)
    )""",
    """
    CREATE TABLE attachments(
        id TEXT PRIMARY KEY,
        record_id TEXT NOT NULL REFERENCES records(id),
        position INTEGER NOT NULL,
        filename TEXT NOT NULL,
        mime TEXT NOT NULL,
        size INTEGER NOT NULL CHECK(size > 0),
        sha256 TEXT NOT NULL REFERENCES blob_contents(sha256),
        added_at TEXT NOT NULL,
        UNIQUE(record_id, position)
    )""",
    "CREATE INDEX attachments_record ON attachments(record_id, position)",
    "CREATE INDEX attachments_content ON attachments(sha256)",
    """
    -- A crash can leave bytes with no reference holding them; this is the only
    -- shape that makes such an orphan findable without a full scan.
    CREATE INDEX contents_unreferenced ON blob_contents(sha256) WHERE refs = 0""",
)


KNOWLEDGE_STATEMENTS: tuple[str, ...] = (
    """
    -- One typed thing the archive asserts about a subject, over an interval, with
    -- the exact span of the evidence behind it. The quote offsets matter as much
    -- as the text: a claim that cannot point at what it was quoted from is a
    -- hypothesis about what the archive says, not a fact in it.
    CREATE TABLE assertions(
        id TEXT PRIMARY KEY,
        subject TEXT NOT NULL,
        predicate TEXT NOT NULL,
        value TEXT NOT NULL,
        category TEXT NOT NULL,
        unit TEXT,
        evidence_kind TEXT NOT NULL CHECK(evidence_kind IN ('owner_declared',
            'explicit_statement', 'observed_pattern', 'derived')),
        record_id TEXT NOT NULL REFERENCES records(id),
        quote_start INTEGER NOT NULL,
        quote_end INTEGER NOT NULL,
        quote TEXT NOT NULL,
        valid_from TEXT,
        valid_to TEXT,
        status TEXT NOT NULL CHECK(status IN ('candidate', 'confirmed', 'superseded',
            'retracted')),
        created_by TEXT NOT NULL,
        confirmed_by TEXT,
        created_at TEXT NOT NULL,
        confirmed_at TEXT,
        supersedes TEXT REFERENCES assertions(id),
        revision INTEGER NOT NULL DEFAULT 1
    )""",
    "CREATE INDEX assertions_subject ON assertions(subject, predicate, status, valid_from)",
    "CREATE INDEX assertions_record ON assertions(record_id, status)",
    """
    -- Derived artifacts declare what they were built from before anyone asks.
    -- The verdict is computed on read and never stored: a summary does not stop
    -- being supported because the row said so. The quoted text is kept, because
    -- drift can only be noticed against something to compare with.
    CREATE TABLE derived_citations(
        artifact_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        coverage TEXT NOT NULL CHECK(coverage IN ('full', 'truncated')),
        record_id TEXT NOT NULL REFERENCES records(id),
        quote TEXT,
        quote_start INTEGER,
        quote_end INTEGER,
        added_at TEXT NOT NULL,
        PRIMARY KEY(artifact_id, record_id)
    )""",
    "CREATE INDEX citations_record ON derived_citations(record_id, artifact_id)",
)


@dataclass(frozen=True)
class Migration:
    name: str
    statements: tuple[str, ...]



MIGRATIONS: Sequence[Migration] = (
    Migration("0001_evidence", EVIDENCE_STATEMENTS),
    Migration("0002_sync", SYNC_STATEMENTS),
    Migration("0003_erasure", ERASURE_STATEMENTS),
    Migration("0004_identity", IDENTITY_STATEMENTS),
    Migration("0005_processing", PROCESSING_STATEMENTS),
    Migration("0006_blobs", BLOB_STATEMENTS),
    Migration("0007_knowledge", KNOWLEDGE_STATEMENTS),
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
