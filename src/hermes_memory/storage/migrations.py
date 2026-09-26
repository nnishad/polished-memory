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

__all__ = ["Migration", "MIGRATIONS", "MIGRATION_LEDGER", "apply_migrations",
           "current_version", "connect"]

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


SUMMARY_STATEMENTS: tuple[str, ...] = (
    """
    -- A summary is a claim about a window of evidence, so it carries the window,
    -- the revision, and who built it. It is stored beside the evidence and never
    -- inside it: a summary that entered the record set would be cited by the
    -- next summary and quietly become a source.
    CREATE TABLE summaries(
        id TEXT PRIMARY KEY,
        scope TEXT NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('thread', 'day', 'week', 'project', 'mental_model')),
        title TEXT NOT NULL,
        body TEXT NOT NULL,
        account_id TEXT,
        window_from TEXT,
        window_to TEXT,
        revision INTEGER NOT NULL DEFAULT 1,
        supersedes TEXT REFERENCES summaries(id),
        status TEXT NOT NULL CHECK(status IN ('published', 'withdrawn')),
        processor_fingerprint TEXT NOT NULL,
        epoch INTEGER NOT NULL,
        budget_tokens INTEGER NOT NULL,
        refresh_after TEXT,
        created_at TEXT NOT NULL,
        published_at TEXT NOT NULL,
        withdrawn_at TEXT
    )""",
    "CREATE INDEX summaries_scope ON summaries(scope, kind, status, published_at)",
    """
    -- One refresh attempt per scope per change set, so a burst of arrivals
    -- coalesces instead of re-summarizing the same afternoon fifty times.
    CREATE TABLE summary_refreshes(
        scope TEXT NOT NULL,
        kind TEXT NOT NULL,
        through_at TEXT NOT NULL,
        requested_at TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('pending', 'running', 'done', 'failed')),
        detail TEXT,
        PRIMARY KEY(scope, kind, through_at)
    )""",
)


PROSPECTIVE_STATEMENTS: tuple[str, ...] = (
    """
    -- A goal is durable intent with a version. Every change appends a history row
    -- and re-publishes the due events, because an acknowledgment of revision 3
    -- must never be able to satisfy revision 4.
    CREATE TABLE goals(
        id TEXT PRIMARY KEY,
        owner_account TEXT REFERENCES identity_accounts(id),
        title TEXT NOT NULL,
        statement TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('candidate', 'active', 'completed',
            'cancelled', 'expired')),
        revision INTEGER NOT NULL DEFAULT 1,
        timezone TEXT NOT NULL,
        due_at TEXT,
        due_precision TEXT NOT NULL DEFAULT 'none' CHECK(due_precision IN
            ('none', 'minute', 'hour', 'day', 'week', 'month', 'year')),
        created_by TEXT NOT NULL,
        created_kind TEXT NOT NULL CHECK(created_kind IN ('owner', 'agent')),
        confirmed_by TEXT,
        source_record_id TEXT REFERENCES records(id),
        snoozed_until TEXT,
        expires_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        decided_at TEXT,
        decided_by TEXT
    )""",
    "CREATE INDEX goals_status_due ON goals(status, due_at, id)",
    """
    CREATE TABLE goal_history(
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        goal_id TEXT NOT NULL REFERENCES goals(id),
        revision INTEGER NOT NULL,
        status TEXT NOT NULL,
        due_at TEXT,
        due_precision TEXT NOT NULL,
        timezone TEXT NOT NULL,
        statement TEXT NOT NULL,
        changed_by TEXT NOT NULL,
        changed_at TEXT NOT NULL,
        reason TEXT NOT NULL
    )""",
    # Every decision appends: a settled goal writes at its current revision too, so
    # a (goal, revision) key would overwrite the change that made it worth settling.
    "CREATE INDEX goal_history_goal ON goal_history(goal_id, revision, seq)",
    """
    -- The bounded predicate vocabulary. A condition is data, not code: nothing
    -- here can express a rule the framework has not agreed to evaluate.
    CREATE TABLE goal_predicates(
        id TEXT PRIMARY KEY,
        goal_id TEXT NOT NULL REFERENCES goals(id),
        revision INTEGER NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('due_at', 'new_message_from',
            'source_item_update', 'measured_threshold', 'waiting_for')),
        params TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('pending', 'satisfied', 'failed', 'unknown')),
        detail TEXT,
        evidence_record_id TEXT REFERENCES records(id),
        evaluated_at TEXT,
        UNIQUE(goal_id, revision, kind, params)
    )""",
    """
    CREATE TABLE due_events(
        id TEXT PRIMARY KEY,
        goal_id TEXT NOT NULL REFERENCES goals(id),
        revision INTEGER NOT NULL,
        fire_at TEXT NOT NULL,
        reason TEXT NOT NULL,
        timezone TEXT NOT NULL,
        precision TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('pending', 'claimed', 'handed_off',
            'suppressed', 'cancelled', 'uncertain')),
        claim_token TEXT,
        claim_until REAL,
        claimed_by TEXT,
        created_at TEXT NOT NULL,
        decided_at TEXT,
        decision_kind TEXT,
        policy_version TEXT
    )""",
    "CREATE UNIQUE INDEX due_event_once ON due_events(goal_id, revision, fire_at, reason)",
    "CREATE INDEX due_event_state ON due_events(state, fire_at)",
    """
    -- The handoff seam: responsibility for a due event transfers into a durable
    -- decision intent. This is not a delivery claim, and the state names that.
    CREATE TABLE decision_intents(
        id TEXT PRIMARY KEY,
        event_id TEXT NOT NULL REFERENCES due_events(id),
        goal_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('silent', 'next_turn', 'digest', 'notify_owner',
            'draft', 'awaiting_analysis')),
        policy_version TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('prepared', 'awaiting_analysis', 'delivered',
            'suppressed', 'uncertain')),
        payload_digest TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(event_id, revision, kind, policy_version)
    )""",
)


PROACTIVITY_STATEMENTS: tuple[str, ...] = (
    """
    -- What the owner allows, per topic, with the clock they live on. Defaults are
    -- the conservative end of the range and only the owner may move them.
    CREATE TABLE topic_policies(
        topic TEXT PRIMARY KEY,
        state TEXT NOT NULL CHECK(state IN ('allowed', 'opted_out')),
        shadow INTEGER NOT NULL DEFAULT 1,
        quiet_from TEXT,
        quiet_until TEXT,
        timezone TEXT NOT NULL DEFAULT 'UTC',
        digest_per_day INTEGER NOT NULL DEFAULT 1,
        max_immediate INTEGER NOT NULL DEFAULT 2,
        cooldown_minutes INTEGER NOT NULL DEFAULT 240,
        updated_at TEXT NOT NULL,
        updated_by TEXT NOT NULL
    )""",
    """
    -- A digest is a slot, not a message: items coalesce into the one window that
    -- closes next outside quiet hours. The id is derived from the slot so a
    -- retried write cannot fork a second window for the same boundary.
    CREATE TABLE attention_windows(
        id TEXT PRIMARY KEY,
        scope TEXT NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('digest', 'immediate')),
        opens_at TEXT NOT NULL,
        closes_at TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('open', 'closed')),
        items INTEGER NOT NULL DEFAULT 0,
        closed_at TEXT,
        UNIQUE(scope, kind, closes_at)
    )""",
    """
    -- One policy outcome per handed-off intention. The intention itself belongs to
    -- C8; this row records what was decided about it and whether a model was even
    -- allowed to look.
    CREATE TABLE proactive_decisions(
        id TEXT PRIMARY KEY,
        intent_id TEXT NOT NULL UNIQUE REFERENCES decision_intents(id),
        goal_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        topic TEXT NOT NULL,
        window_id TEXT REFERENCES attention_windows(id),
        action TEXT NOT NULL CHECK(action IN ('silent', 'next_turn', 'digest',
            'notify_owner', 'draft')),
        reason TEXT NOT NULL,
        policy_version TEXT NOT NULL,
        shadow INTEGER NOT NULL DEFAULT 1,
        model_used INTEGER NOT NULL DEFAULT 0,
        packet_id TEXT,
        citations TEXT,
        decided_at TEXT NOT NULL
    )""",
    "CREATE INDEX decision_topic_time ON proactive_decisions(topic, decided_at)",
    """
    -- The framework makes artifacts; the host delivers them. Every state here is
    -- about the artifact, and only 'confirmed' has a receipt from outside.
    -- The revalidation inputs are stored *with* the artifact rather than looked up
    -- at delivery time only: a digest prepared on Tuesday and delivered on Friday
    -- must be able to say what it was true about, and whether it still is.
    CREATE TABLE outbox(
        id TEXT PRIMARY KEY,
        decision_id TEXT NOT NULL REFERENCES proactive_decisions(id),
        kind TEXT NOT NULL CHECK(kind IN ('next_turn', 'digest', 'notify_owner', 'draft')),
        topic TEXT NOT NULL,
        recipient TEXT NOT NULL,
        payload TEXT NOT NULL,
        payload_digest TEXT NOT NULL,
        evidence TEXT NOT NULL DEFAULT '[]',
        policy_version TEXT NOT NULL,
        expires_at TEXT,
        state TEXT NOT NULL CHECK(state IN ('prepared', 'leased', 'attempted', 'confirmed',
            'accepted_unverified', 'uncertain', 'suppressed', 'expired')),
        reason TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        lease_token TEXT,
        lease_until REAL,
        held_by TEXT,
        proof TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        delivered_at TEXT,
        UNIQUE(decision_id, kind)
    )""",
    "CREATE INDEX outbox_ready ON outbox(state, kind, created_at)",
)


LEARNING_STATEMENTS: tuple[str, ...] = (
    """
    -- One row per version of one lesson. A lesson is never edited in place: the
    -- version that was evaluated is the version that earned its status, and a
    -- reworded lesson is a new claim that has to be evaluated again.
    CREATE TABLE lessons(
        id TEXT NOT NULL,
        version INTEGER NOT NULL,
        text TEXT NOT NULL,
        applicability TEXT NOT NULL,
        prerequisites TEXT NOT NULL DEFAULT '[]',
        exceptions TEXT NOT NULL DEFAULT '[]',
        contrary_cases TEXT NOT NULL DEFAULT '[]',
        evidence TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL CHECK(status IN ('candidate', 'active', 'retracted')),
        created_by TEXT NOT NULL,
        created_kind TEXT NOT NULL CHECK(created_kind IN ('owner', 'agent', 'evaluation')),
        created_at TEXT NOT NULL,
        evaluation_id TEXT REFERENCES evaluations(id),
        decided_by TEXT,
        decided_at TEXT,
        retraction_reason TEXT,
        PRIMARY KEY(id, version)
    )""",
    "CREATE INDEX lessons_status ON lessons(status, id, version)",
    """
    -- Who reported what, kept apart on purpose. An assistant that announced success
    -- is a fact about a message, not a fact about the world, and the three kinds are
    -- never allowed to add up to the same thing.
    CREATE TABLE outcomes(
        id TEXT PRIMARY KEY,
        subject_kind TEXT NOT NULL CHECK(subject_kind IN ('lesson', 'goal', 'artifact',
            'task')),
        subject_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('host_receipt', 'owner_report',
            'assistant_claim')),
        valence TEXT NOT NULL CHECK(valence IN ('success', 'failure', 'unknown')),
        note TEXT NOT NULL,
        evidence TEXT NOT NULL DEFAULT '[]',
        actor TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )""",
    "CREATE INDEX outcome_subject ON outcomes(subject_kind, subject_id, kind)",
    """
    -- An evaluation is a claim about a run: which lesson version, which fixtures,
    -- which baseline, and which code and model actually executed. Change any of the
    -- five and the verdict stops being about the thing in front of you.
    CREATE TABLE evaluations(
        id TEXT PRIMARY KEY,
        lesson_id TEXT NOT NULL,
        lesson_version INTEGER NOT NULL,
        lesson_digest TEXT NOT NULL,
        runner TEXT NOT NULL,
        fixture_digest TEXT NOT NULL,
        baseline_digest TEXT NOT NULL,
        code_version TEXT NOT NULL,
        model_version TEXT NOT NULL,
        verdict TEXT NOT NULL CHECK(verdict IN ('running', 'passed', 'failed',
            'incomplete', 'stale')),
        detail TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        UNIQUE(lesson_id, lesson_version, fixture_digest, baseline_digest, code_version,
               model_version)
    )""",
    """
    -- Per case, because "the suite passed" is not evidence unless the cases it
    -- passed are. Targeted cases show the lesson does its job; regression cases
    -- show it did not break the ones that already did.
    CREATE TABLE evaluation_cases(
        evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
        case_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('targeted', 'regression')),
        outcome TEXT NOT NULL CHECK(outcome IN ('pass', 'fail', 'error', 'skipped')),
        detail TEXT NOT NULL DEFAULT '',
        PRIMARY KEY(evaluation_id, case_id)
    )""",
)


SOURCE_GAP_STATEMENTS: tuple[str, ...] = (
    """
    -- What a source could not give us, kept per connector generation. A gap that
    -- is only logged is a gap nobody can close: this row is what the next run
    -- checks itself against, and what a later arrival resolves.
    CREATE TABLE source_gaps(
        source TEXT NOT NULL,
        generation INTEGER NOT NULL,
        ref TEXT NOT NULL,
        reason TEXT NOT NULL,
        first_seen_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        cleared_at TEXT,
        PRIMARY KEY(source, generation, ref)
    )""",
    "CREATE INDEX source_gaps_open ON source_gaps(source, generation, cleared_at)",
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
    Migration("0008_summaries", SUMMARY_STATEMENTS),
    Migration("0009_prospective", PROSPECTIVE_STATEMENTS),
    Migration("0010_proactivity", PROACTIVITY_STATEMENTS),
    Migration("0011_learning", LEARNING_STATEMENTS),
    Migration("0012_source_gaps", SOURCE_GAP_STATEMENTS),
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
