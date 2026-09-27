"""C12 jobs: a durable state machine, fenced against epoch and lease.

Every job records the processor fingerprint, the memory epoch and the input
revision it was created for. That triple is what makes a stale result
recognisable: after a reset or an extractor change, a job that completed under
the old fingerprint is not evidence that the new one has run, and reporting it
as covered would be a lie about coverage rather than about data.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ..ids import digest, now
from ..storage.evidence import EvidenceError
from .routes import PRIORITY, Route

__all__ = ["JobQueue", "Job", "QUEUED", "LEASED", "SUBMITTING", "RUNNING", "SUCCEEDED",
           "RETRY_WAIT", "UNCERTAIN", "QUARANTINED", "CANCELLED", "TERMINAL"]

QUEUED = "queued"
LEASED = "leased"
SUBMITTING = "submitting"
RUNNING = "running"
SUCCEEDED = "succeeded"
RETRY_WAIT = "retry_wait"
UNCERTAIN = "uncertain"
QUARANTINED = "quarantined"
CANCELLED = "cancelled"

# A terminal job is never picked up again by a dispatcher.
TERMINAL = frozenset({SUCCEEDED, CANCELLED, QUARANTINED})
# States that hold an in-flight claim on a physical resource.
_INFLIGHT = frozenset({LEASED, SUBMITTING, RUNNING, UNCERTAIN})
# States a holder's lease is still worth something in: the row moves LEASED → SUBMITTING →
# RUNNING under one claim, so the token and its expiry have to survive those changes. The
# only thing that ends the claim is the work ending, or the expiry passing unnoticed.
_HELD = frozenset({LEASED, SUBMITTING, RUNNING})


@dataclass(frozen=True)
class Job:
    id: str
    kind: str
    state: str
    priority: int
    resource: str
    route: str
    epoch: int
    processor_fingerprint: str
    inputs: tuple[str, ...]
    input_revision: str
    submission_id: str | None
    backend_operation_id: str | None
    attempts: int
    max_attempts: int
    tokens_used: int
    token_budget: int
    deadline: float | None
    last_error: str | None = None
    # A lease is a holder's claim, and `lease_until` is its ticking promise to keep saying so.
    # A reader has to be able to tell "running, and somebody answers for it" from "running,
    # and the process that said so is gone" — the second is unreconciled work.
    lease: str | None = None
    lease_until: float | None = None


# Whether a job is overdue is asked of the queue, not of the row: `claim` compares a deadline
# against the queue's own (often injected) clock, and a property reading wall time would
# disagree with it under a fake clock. The attempt ceiling lives in `retry()` for the same
# reason — one place decides when work stops being retried.


class JobQueue:
    """Durable dispatch work, one writer at a time per job via a lease."""

    #: How long a renewal vouches for a job. A pass over one input is far shorter, so a
    #: worker that is still writing has plenty of room to say so again.
    DEFAULT_RENEWAL = 300.0

    def __init__(self, store, *, clock: Callable[[], float] = time.time):
        self.store = store
        self.db = store.db
        self.clock = clock

    # -- creation ------------------------------------------------------------

    def enqueue(self, *, kind: str, inputs: list[str], input_revision: str, route: Route,
                processor_fingerprint: str, priority: str = "maintenance",
                max_attempts: int = 3, token_budget: int = 50_000,
                deadline: float | None = None) -> dict[str, Any]:
        if not inputs or len(inputs) > 500:
            raise EvidenceError("a job covers between 1 and 500 inputs")
        if priority not in PRIORITY:
            raise EvidenceError(f"priority must be one of {sorted(PRIORITY)}")
        if not 1 <= max_attempts <= 10:
            raise EvidenceError("max_attempts must be between 1 and 10 (finite, deliberately)")
        if not 1 <= token_budget <= 10_000_000:
            raise EvidenceError("token_budget must be finite and positive")
        if not isinstance(processor_fingerprint, str) or not processor_fingerprint.strip():
            raise EvidenceError("processor_fingerprint is required; coverage is claimed "
                                "per processor, not per kind")
        # Identity is derived from the work, not from a counter, so re-enqueueing
        # the same revision under the same fingerprint is one job rather than two.
        job_id = "job_" + digest([kind, route.name, sorted(inputs), input_revision,
                                  processor_fingerprint])[:32]
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute("SELECT id, state FROM processing_jobs WHERE id=?",
                                       (job_id,)).fetchone()
            if existing and existing["state"] not in (QUARANTINED, CANCELLED):
                self.db.execute("COMMIT")
                return {"job_id": job_id, "state": existing["state"], "created": False}
            self.db.execute(
                "INSERT INTO processing_jobs(id, kind, state, priority, resource, route, epoch, "
                "processor_fingerprint, inputs, input_revision, max_attempts, token_budget, "
                "deadline, created_at, updated_at) "
                "VALUES(?,?,?, ?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at",
                (job_id, kind, QUEUED, PRIORITY[priority], route.resource, route.name,
                 self.store.epoch(), processor_fingerprint,
                 json.dumps(sorted(inputs), sort_keys=True), input_revision, max_attempts,
                 token_budget, deadline, now(), now()),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"job_id": job_id, "state": QUEUED, "created": True}

    def get(self, job_id: str) -> Job | None:
        row = self.db.execute("SELECT * FROM processing_jobs WHERE id=?", (job_id,)).fetchone()
        return _job(row) if row else None

    # -- dispatch ------------------------------------------------------------

    def claim(self, *, worker: str, ttl: float = 120.0) -> Job | None:
        """Lease the most urgent eligible job, honouring priority then age.

        An overdue job is not claimed: handing out work that has already blown
        its own deadline would spend a physical slot on a result nobody wanted.

        A job waiting out its backoff *is* claimable, once the backoff has passed. Leaving
        ``retry_wait`` out of this query made a retry a silent ending: the row sat in a state
        the status report called "waiting", the plan's bounded retry never happened, and
        nothing said why. It is the same rule `ready_for_retry()` reports, so the two agree by
        construction rather than by anybody remembering to update both.
        """
        if not 1 <= ttl <= 3600:
            raise EvidenceError("ttl must be between 1 and 3600 seconds")
        now_epoch = self.store.epoch()
        moment = self.clock()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._reclaim_expired_leases()
            row = self.db.execute(
                """
                SELECT * FROM processing_jobs
                WHERE (state=? OR (state=? AND (not_before IS NULL OR not_before <= ?)))
                  AND epoch=? AND attempts < max_attempts
                  AND (deadline IS NULL OR deadline > ?)
                ORDER BY priority, created_at, id LIMIT 1
                """, (QUEUED, RETRY_WAIT, moment, now_epoch, moment)).fetchone()
            if row is None:
                self.db.execute("COMMIT")
                return None
            lease = uuid.uuid4().hex
            self.db.execute(
                "UPDATE processing_jobs SET state=?, lease=?, lease_until=?, updated_at=? "
                "WHERE id=?",
                (LEASED, lease, self.clock() + ttl, now(), row["id"]))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.get(row["id"])

    def release(self, job: Job) -> str:
        """Put a leased job back without spending an attempt.

        Failing to obtain a resource slot is not the job's fault, so counting it
        would let contention alone quarantine work that never ran.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT state FROM processing_jobs WHERE id=?",
                                  (job.id,)).fetchone()
            if row is None:
                raise EvidenceError(f"unknown job {job.id!r}")
            if row["state"] != LEASED:
                self.db.execute("COMMIT")
                return row["state"]
            self.db.execute(
                "UPDATE processing_jobs SET state=?, lease=NULL, lease_until=NULL, updated_at=?"
                " WHERE id=?", (QUEUED, now(), job.id))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return QUEUED

    def begin_submission(self, job: Job, *, submission_id: str,
                         operation_id: str | None = None) -> None:
        """Record the submission identity *before* the request leaves the process."""
        self._transition(job.id, SUBMITTING, submission_id=submission_id,
                         operation_id=operation_id)

    def mark_running(self, job: Job, *, operation_id: str | None = None) -> None:
        """The work is with the backend now.

        ``operation_id`` is recorded only when the backend named one. A synchronous retain
        answers with content and no operation identity, and inventing an id here would give
        ``cancel --job`` a thing to ask a backend about that the backend never heard of.
        """
        self._transition(job.id, RUNNING, operation_id=operation_id)

    def complete(self, job: Job, *, covered: list[str], tokens: int = 0,
                 seconds: float = 0.0) -> dict[str, Any]:
        """Succeed only against actual coverage, charged to the job's own budget."""
        if not isinstance(covered, list):
            raise EvidenceError("covered must be the list of inputs actually processed")
        claimed = set(json.loads(self._inputs_json(job.id)))
        if not set(covered) <= claimed:
            raise EvidenceError("a job may only claim coverage for its own inputs")
        missing = sorted(claimed - set(covered))
        if tokens > job.token_budget - job.tokens_used:
            # Over budget is partial, not done: pretending otherwise would let a
            # job that spent its allowance still report success.
            self.retry(job, error="token budget exhausted before completion", backoff=60.0)
            return {"state": RETRY_WAIT, "covered": len(covered), "uncovered": missing,
                    "complete": False}
        if missing:
            self.partial(job, covered=len(covered), reason=f"{len(missing)} input(s) uncovered")
            return {"state": RETRY_WAIT, "covered": len(covered), "uncovered": missing,
                    "complete": False}
        self._transition(job.id, SUCCEEDED, tokens=tokens, seconds=seconds)
        return {"state": SUCCEEDED, "covered": len(covered), "uncovered": [], "complete": True}

    def retry(self, job: Job, *, error: str, backoff: float = 30.0) -> str:
        """Back off, and quarantine rather than loop when attempts run out."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            fresh = self.get(job.id)
            if fresh is None or fresh.state in TERMINAL:
                self.db.execute("COMMIT")
                return fresh.state if fresh else CANCELLED
            attempts = fresh.attempts + 1
            state = QUARANTINED if attempts >= fresh.max_attempts else RETRY_WAIT
            self.db.execute(
                "UPDATE processing_jobs SET state=?, attempts=?, last_error=?, "
                "not_before=?, lease=NULL, lease_until=NULL, updated_at=? WHERE id=?",
                (state, attempts, error[:500], self.clock() + backoff if state == RETRY_WAIT
                 else None, now(), job.id))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return state

    def partial(self, job: Job, *, covered: int, reason: str) -> str:
        """A malformed or truncated result is recorded, not retried forever."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            fresh = self.get(job.id)
            if fresh is None:
                raise EvidenceError(f"unknown job {job.id!r}")
            if fresh.state in TERMINAL:
                # A result arriving for work already quarantined or cancelled is
                # reported, not absorbed: re-counting it would quietly restart a
                # decision the attempt budget already closed.
                self.db.execute("COMMIT")
                return fresh.state
            state = QUARANTINED if fresh.attempts + 1 >= fresh.max_attempts else RETRY_WAIT
            self.db.execute(
                "UPDATE processing_jobs SET state=?, attempts=?, last_error=?, updated_at=? "
                "WHERE id=?",
                (state, fresh.attempts + 1, f"partial: {reason}"[:500], now(), job.id))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return state

    def uncertain(self, job: Job, *, reason: str) -> None:
        """We do not know whether the backend did it. The slot stays claimed."""
        self._transition(job.id, UNCERTAIN, error=reason, release_lease=True)

    def cancel(self, job_id: str, *, actor: str, reason: str) -> str:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT state FROM processing_jobs WHERE id=?",
                                  (job_id,)).fetchone()
            if row is None:
                raise EvidenceError(f"unknown job {job_id!r}")
            if row["state"] == SUCCEEDED:
                raise EvidenceError("a succeeded job cannot be cancelled; retract its output")
            self.db.execute(
                "UPDATE processing_jobs SET state=?, last_error=?, lease=NULL, "
                "lease_until=NULL, updated_at=? WHERE id=?",
                (CANCELLED, f"{actor}: {reason}"[:500], now(), job_id))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return CANCELLED

    def reap_overdue(self, *, at: float | None = None, limit: int = 50) -> list[str]:
        """Give an overdue job an ending instead of leaving it unclaimable and unseen.

        ``claim()`` refuses work past its deadline, which is right: a result nobody wanted by
        then is not a result. But the row then stops appearing as anything at all.
        Quarantining it is the same decision written where a person and a report can find it.
        Only work that never reached a backend is reaped — a job that is submitting, running
        or uncertain has an outcome nobody knows yet, and that is a different list.
        """
        moment = self.clock() if at is None else float(at)
        rows = self.db.execute(
            "SELECT id FROM processing_jobs WHERE deadline IS NOT NULL AND deadline <= ? "
            "AND state IN (?,?,?) ORDER BY deadline LIMIT ?",
            (moment, QUEUED, LEASED, RETRY_WAIT, limit)).fetchall()
        reaped = []
        for row in rows:
            self._transition(row["id"], QUARANTINED,
                             error="deadline passed before a worker took it")
            reaped.append(row["id"])
        return reaped

    def ready_for_retry(self, *, limit: int = 50) -> list[Job]:
        rows = self.db.execute(
            "SELECT * FROM processing_jobs WHERE state=? AND (not_before IS NULL OR "
            "not_before <= ?) ORDER BY priority, not_before, id LIMIT ?",
            (RETRY_WAIT, self.clock(), limit)).fetchall()
        return [_job(row) for row in rows]

    def unresolved(self, *, limit: int = 50) -> list[Job]:
        """Work whose outcome we never established — the reconciliation backlog."""
        rows = self.db.execute(
            "SELECT * FROM processing_jobs WHERE state IN (?,?) ORDER BY updated_at LIMIT ?",
            (UNCERTAIN, SUBMITTING, limit)).fetchall()
        return [_job(row) for row in rows]

    def quarantine(self, *, limit: int = 50) -> list[Job]:
        rows = self.db.execute(
            "SELECT * FROM processing_jobs WHERE state=? ORDER BY updated_at LIMIT ?",
            (QUARANTINED, limit)).fetchall()
        return [_job(row) for row in rows]

    def counts(self) -> dict[str, int]:
        rows = self.db.execute(
            "SELECT state, count(*) AS n FROM processing_jobs GROUP BY state").fetchall()
        return {row["state"]: row["n"] for row in rows}

    def abandon_stale_epoch(self, *, actor: str = "system") -> int:
        """Quarantine jobs created under a superseded epoch.

        Their inputs were fetched before a reset, so a success would describe
        coverage the current store does not have.
        """
        current = self.store.epoch()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.db.execute(
                "UPDATE processing_jobs SET state=?, last_error=?, lease=NULL, "
                "lease_until=NULL, updated_at=? WHERE epoch < ? AND state NOT IN (?,?)",
                (QUARANTINED, f"epoch superseded (now {current})", now(), current,
                 SUCCEEDED, QUARANTINED))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return cursor.rowcount

    # -- internals -----------------------------------------------------------

    def _transition(self, job_id: str, state: str, *, submission_id: str | None = None,
                    operation_id: str | None = None, tokens: int = 0, seconds: float = 0.0,
                    error: str | None = None, release_lease: bool = False) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT state, attempts FROM processing_jobs WHERE id=?",
                                  (job_id,)).fetchone()
            if row is None:
                raise EvidenceError(f"unknown job {job_id!r}")
            if row["state"] == CANCELLED and state not in {CANCELLED, QUARANTINED}:
                # A cancelled job that later reports success would resurrect work
                # the operator explicitly stopped.
                raise EvidenceError(f"job {job_id} is cancelled; refusing {state}")
            keep = state in _HELD and not release_lease
            self.db.execute(
                "UPDATE processing_jobs SET state=?, "
                "submission_id=COALESCE(?, submission_id), "
                "backend_operation_id=COALESCE(?, backend_operation_id), "
                "tokens_used=tokens_used+?, "
                "lease=CASE WHEN ? THEN lease ELSE NULL END, "
                "lease_until=CASE WHEN ? THEN lease_until ELSE NULL END, "
                "last_error=?, updated_at=?, completed_at=? WHERE id=?",
                (state, submission_id, operation_id, tokens, keep, keep, error, now(),
                 now() if state == SUCCEEDED else None, job_id))
            # tokens_used on the row is bookkeeping for the job. The shared
            # budget_usage ledger is written by the gate, which is the physical
            # boundary: charging it here as well would double-count every call.
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def _reclaim_expired_leases(self, *, at: float | None = None) -> int:
        """An expired lease becomes uncertain work, never a free retry.

        This covers a job that had already reached the backend, not only one that was
        claimed and abandoned: a worker that dies between the request and the answer leaves a
        row saying SUBMITTING or RUNNING, and the honest ending for that is uncertain work
        somebody has to reconcile — not a requeue that sends the same private text twice.
        """
        moment = self.clock() if at is None else float(at)
        return self.db.execute(
            "UPDATE processing_jobs SET state=?, last_error='lease expired', lease=NULL, "
            "lease_until=NULL, updated_at=? WHERE state IN (?,?,?) AND lease_until IS NOT NULL "
            "AND lease_until < ?",
            (UNCERTAIN, now(), LEASED, SUBMITTING, RUNNING, moment)).rowcount

    def renew(self, job: Job, *, ttl: float | None = None) -> bool:
        """Say, while the work is still in flight, that this holder still holds it.

        Refuses quietly when the row has moved on — released, re-claimed under another
        lease, or finished. A dying worker's last write must not extend a lease somebody
        else now owns.
        """
        window = self.DEFAULT_RENEWAL if ttl is None else ttl
        if not 1 <= window <= 3600:
            raise EvidenceError("ttl must be between 1 and 3600 seconds")
        if not job.lease:
            return False
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.db.execute(
                "UPDATE processing_jobs SET lease_until=?, updated_at=? WHERE id=? AND lease=?"
                " AND state IN (?,?,?)",
                (self.clock() + window, now(), job.id, job.lease, LEASED, SUBMITTING, RUNNING))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return int(cursor.rowcount or 0) > 0

    def reconcile(self, *, at: float | None = None) -> dict[str, int]:
        """Startup pass over the queue, before any worker claims.

        Two things cannot be inferred from a crash, so both are recorded as what is
        known rather than as what would be convenient: a job whose lease lapsed may
        already have reached the backend, and a job queued under a superseded epoch
        describes coverage the store no longer has. Each step is a narrowing update,
        so running this twice reports nothing the second time.
        """
        report: dict[str, int] = {}
        expired = self._reclaim_expired_leases(at=at)
        if expired:
            report["lease_expired"] = expired
        # A job whose deadline passed while the worker was down will never be claimed again,
        # so it is closed here rather than left to look like a queue with nothing in it.
        overdue = self.reap_overdue(at=at)
        if overdue:
            report["overdue"] = len(overdue)
        superseded = self.abandon_stale_epoch()
        if superseded:
            report["stale_epoch"] = superseded
        unresolved = len(self.unresolved())
        if unresolved:
            report["unresolved"] = unresolved
        return report

    def _inputs_json(self, job_id: str) -> str:
        row = self.db.execute("SELECT inputs FROM processing_jobs WHERE id=?",
                              (job_id,)).fetchone()
        if row is None:
            raise EvidenceError(f"unknown job {job_id!r}")
        return row["inputs"]


def _job(row) -> Job:
    return Job(id=row["id"], kind=row["kind"], state=row["state"], priority=row["priority"],
               resource=row["resource"], route=row["route"], epoch=row["epoch"],
               processor_fingerprint=row["processor_fingerprint"],
               inputs=tuple(json.loads(row["inputs"])), input_revision=row["input_revision"],
               submission_id=row["submission_id"],
               backend_operation_id=row["backend_operation_id"], attempts=row["attempts"],
               max_attempts=row["max_attempts"], tokens_used=row["tokens_used"],
               token_budget=row["token_budget"], deadline=row["deadline"],
               last_error=row["last_error"], lease=row["lease"],
               lease_until=row["lease_until"])
