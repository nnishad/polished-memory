"""C12 worker: one slot, one job, and honest accounting for both.

The composition order matters. A job is claimed, then the budget is checked,
then the physical slot is taken — so nothing occupies a device while we are
still deciding whether we can afford to use it. The submission identity is
written durably before the request leaves, and the slot is released as
'uncertain' on any transport failure, because we genuinely do not know whether
the backend is still running it.

A single slot is the design, not a limitation to work around: the local 4B and
the embedding model share one GPU, and the remote 9B serves one request best.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from ..backend.capabilities import (OPERATION_ABANDONED, OPERATION_DONE,
                                    OPERATION_RUNNING, OPERATION_STOPPED)
from ..backend.document_map import DocumentMap
from ..backend.hindsight_client import (HindsightError, HindsightUnavailable, SubmissionConflict,
                                        operation_reason, operation_state)
from .budgets import BudgetExhausted, Budgets
from .jobs import CANCELLED, JobQueue, QUEUED
from .resource_gate import GateClosed, GatePaused, ResourceGate
from .routes import RouteTable

__all__ = ["FormationWorker", "AttemptOutcome", "OperationAbandoned", "OperationStopped"]

# A rough ceiling on what one retain will cost, used only to refuse a request
# that cannot possibly fit today's allowance. Actual usage is charged from what
# the backend reports.
ESTIMATED_TOKENS_PER_ITEM = 2_000

# How a job waits for the operation it started. The engine runs a consolidation long
# after the submission has been answered, so the wait is the work, not a delay to
# optimise away; it is bounded because a job that waits forever holds a physical
# device forever, and every round renews the lease because an operation outlives any
# ttl fixed when the slot was claimed.
FOLLOW_POLL_S = 5.0
FOLLOW_MAX_POLLS = 720


class OperationAbandoned(HindsightError):
    """The backend said this operation will not produce the projection."""


class OperationStopped(Exception):
    """The operation was cancelled — an ending somebody asked for, not a failure.

    Deliberately not an ``HindsightError``: every backend failure this worker handles
    is retried, and resubmitting work the operator stopped would undo their decision.
    """

    def __init__(self, message: str, *, operation_id: str):
        super().__init__(message)
        self.operation_id = operation_id


@dataclass(frozen=True)
class AttemptOutcome:
    job_id: str
    state: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"job_id": self.job_id, "state": self.state, "detail": self.detail}


class FormationWorker:
    def __init__(self, *, store, jobs: JobQueue, gate: ResourceGate, budgets: Budgets,
                 documents: DocumentMap, client, routes: RouteTable, worker_id: str,
                 sleeper: Callable[[float], None] = time.sleep,
                 follow_poll_s: float = FOLLOW_POLL_S,
                 follow_max_polls: int = FOLLOW_MAX_POLLS):
        self.store = store
        self.jobs = jobs
        self.gate = gate
        self.budgets = budgets
        self.documents = documents
        self.client = client
        self.routes = routes
        self.worker_id = worker_id
        self.sleeper = sleeper
        self.follow_poll_s = follow_poll_s
        self.follow_max_polls = follow_max_polls

    def drain(self, *, max_jobs: int = 5, idle_wait: float = 0.0) -> dict[str, Any]:
        """Process until the queue is empty or the budget or pause says stop."""
        outcomes: list[AttemptOutcome] = []
        for _ in range(max_jobs):
            outcome = self.run_once(idle_wait=idle_wait)
            if outcome is None:
                break
            outcomes.append(outcome)
        return {"attempted": len(outcomes), "outcomes": [item.as_dict() for item in outcomes],
                "counts": self.jobs.counts(), "budget": self.budgets.report()}

    def run_once(self, *, idle_wait: float = 0.0) -> AttemptOutcome | None:
        if self.gate.paused:
            return AttemptOutcome("-", "paused", "all inference is paused by the operator")
        job = self.jobs.claim(worker=self.worker_id)
        if job is None:
            return None
        try:
            route = self.routes.by_name(job.route)
        except Exception as error:
            self.jobs.release(job)
            return AttemptOutcome(job.id, QUEUED, f"unknown route: {error}")

        estimate = len(job.inputs) * ESTIMATED_TOKENS_PER_ITEM
        try:
            self.budgets.admit(route.resource, estimated_tokens=estimate)
        except BudgetExhausted as error:
            self.jobs.release(job)
            return AttemptOutcome(job.id, "budget_exhausted", str(error))

        reservation = self.gate.try_acquire(route=route.name, holder=self.worker_id,
                                           resource=route.resource, priority=route.priority_rank(),
                                           job_id=job.id, ttl=max(60.0, self.gate.default_ttl))
        if reservation is None:
            # Contention is not failure. The job returns to the queue with its
            # attempt count untouched and retries once the slot frees.
            self.jobs.release(job)
            if idle_wait:
                self.sleeper(idle_wait)
            return AttemptOutcome(job.id, "slot_busy", f"{route.resource} is occupied")

        started = time.monotonic()
        return self._execute(job, route, reservation, started)

    # -- the guarded request -------------------------------------------------

    def _execute(self, job, route, reservation, started):
        """Submit while the slot is ours, then wait for the engine without holding it.

        Two things own the release. The first is the ``finally`` below: a success path that
        returned from inside a ``try`` would leave the physical slot claimed forever, which is
        the failure mode the gate exists to prevent. The second is the release inside the body,
        taken the moment the last submission is answered — because the model work that follows
        is done by the engine's own worker, which asks this same gate for the same device.
        Holding a single slot across a run that needs admission on that slot is not caution, it
        is a deadlock: the pass watches an operation that can only progress once the pass stops
        occupying the resource. The live machine proved it — four timed-out extraction attempts
        and a task rescheduled while the submitter politely renewed a lease nothing else could
        use. So the device is handed back before the wait, and what the wait is for is stated
        by the answer the engine gives rather than by a claim this process can no longer justify.
        """
        tokens = 0
        gate_outcome = "succeeded"
        released = False
        try:
            try:
                mappings = [self.documents.begin(record_id, job.input_revision,
                                                async_submission=True)
                            for record_id in job.inputs]
            except Exception as error:
                gate_outcome = "failed"
                state = self.jobs.retry(job, error=f"mapping failed: {error}"[:400],
                                        backoff=60.0)
                return AttemptOutcome(job.id, state, f"mapping failed: {error}")
            self.jobs.begin_submission(job, submission_id=next(
                (m["submission_id"] for m in mappings if m["submission_id"]), ""))
            covered: list[str] = []
            submitted: list[tuple[str, str, dict[str, Any]]] = []
            running = False
            for record_id, mapping in zip(job.inputs, mappings):
                # live_and_visible, not include_hidden: a record deleted by an
                # erasure or hidden by a supersession must never be re-projected
                # into the derived backend, and it is not coverage.
                if not self.store.live_and_visible(record_id):
                    continue
                if mapping.get("already_projected"):
                    # A previous attempt confirmed this one. Re-sending it would spend the
                    # device to ask a question that is already answered, and the job that
                    # covers it stays coverage either way.
                    covered.append(record_id)
                    continue
                evidence = self.store.get(record_id)
                if evidence is None:
                    continue
                # The lease is a promise that keeps being made, not a one-time gift: a pass
                # over several inputs outlives the ttl it was claimed under, and a row that
                # stops being vouched for is what reconciliation reads as abandoned work.
                self.jobs.renew(job)
                if not running:
                    # The job is with the backend from the first request onwards, which is
                    # the one state a reader cannot recover afterwards: a worker that died
                    # here leaves a row saying `submitting`, and `running` says the slot was
                    # paid for.
                    self.jobs.mark_running(job)
                    running = True
                body = self.client.retain_async(
                    [{"content": evidence.text, "document_id": mapping["document_id"],
                      "timestamp": evidence.occurred_at,
                      "metadata": {"source": evidence.source, "record_id": record_id}}],
                    submission_id=mapping["submission_id"])
                operation = str(body.get("operation_id") or mapping["submission_id"])
                self.jobs.mark_running(job, operation_id=operation)
                submitted.append((record_id, operation, body))
            # The submissions are the only part of this job that used the device through us.
            self.gate.release(reservation, outcome="succeeded",
                              tokens=sum(_tokens_from(body) for _, _, body in submitted),
                              seconds=time.monotonic() - started)
            released = True
            return self._await(job, submitted, covered)
        except OperationStopped as error:
            # The operator's decision to stop this work, reported by the backend. What
            # earlier inputs of this job already spent is still charged.
            gate_outcome = "cancelled"
            self.jobs.cancel(job.id, actor="backend", reason=str(error)[:400])
            return AttemptOutcome(job.id, CANCELLED, str(error))
        except SubmissionConflict as error:
            # The backend already holds this submission; reconcile, never resend.
            self.documents.reconcile(client=self.client, limit=max(1, len(job.inputs)))
            self._quarantine(job, reservation, error)
            tokens = None  # slot is not being released; it is being quarantined
            return AttemptOutcome(job.id, "uncertain", str(error))
        except HindsightUnavailable as error:
            # We cannot claim the backend did not run this. Holding the slot as
            # uncertain is the honest state; a retry would double-spend the device.
            self._quarantine(job, reservation, error)
            tokens = None
            return AttemptOutcome(job.id, "uncertain", str(error))
        except HindsightError as error:
            gate_outcome = "failed"
            state = self.jobs.retry(job, error=str(error)[:400], backoff=60.0)
            return AttemptOutcome(job.id, state, str(error))
        except Exception as error:  # a defect must not be reported as a backend failure
            gate_outcome = "failed"
            state = self.jobs.retry(job, error=f"worker error: {error}"[:400], backoff=300.0)
            return AttemptOutcome(job.id, state, f"worker error: {error}")
        finally:
            # None means the slot was quarantined as uncertain instead of freed;
            # releasing it there would hand out a device that may still be busy.
            if tokens is not None and not released:
                self.gate.release(reservation, outcome=gate_outcome, tokens=tokens,
                                  seconds=time.monotonic() - started)

    def _quarantine(self, job, reservation, error: Exception) -> None:
        """Write down that this work is unresolved, and hold the device if it is ours.

        The slot may have been settled from outside while we waited — an operator's
        ``gate --resolve`` says the device is free again. Re-blocking it here would
        overrule that answer, so the refusal is absorbed; the job stays uncertain
        either way, because nothing about the reservation says the model read anything.
        """
        self.jobs.uncertain(job, reason=str(error)[:400])
        try:
            self.gate.mark_uncertain(reservation, reason=str(error)[:400])
        except GateClosed:
            pass

    # -- waiting for the operation the submission started ---------------------

    def _await(self, job, submitted: list[tuple[str, str, dict[str, Any]]],
               covered: list[str]) -> AttemptOutcome:
        """Turn each answered operation into coverage, one honest outcome at a time.

        A submission is not a projection: the document is confirmed only once the operation the
        backend named has finished, because "queued" and "the model has read it" are different
        claims and coverage is the second one. Nothing here holds a device — the engine is using
        it under its own admission — so the outcomes are written straight to the job: a stop ends
        it, an operation that ended badly is the ordinary failed attempt, and an answer that
        never came leaves the row uncertain for `form --reconcile`, which is the door built for
        exactly that question.
        """
        tokens = 0
        waiting = time.monotonic()
        try:
            for record_id, operation, body in submitted:
                try:
                    finished = self._follow(operation)
                except OperationStopped as error:
                    self.jobs.cancel(job.id, actor="backend", reason=str(error)[:400])
                    return AttemptOutcome(job.id, CANCELLED, str(error))
                except OperationAbandoned as error:
                    state = self.jobs.retry(job, error=str(error)[:400], backoff=60.0)
                    return AttemptOutcome(job.id, state, str(error))
                except HindsightUnavailable as error:
                    self.jobs.uncertain(job, reason=str(error)[:400])
                    return AttemptOutcome(job.id, "uncertain", str(error))
                self.documents.confirm(record_id, job.input_revision)
                covered.append(record_id)
                tokens += _tokens_from(body, finished)
            outcome = self.jobs.complete(job, covered=covered, tokens=tokens)
            return AttemptOutcome(job.id, outcome["state"],
                                  f"covered {len(covered)}/{len(job.inputs)}")
        finally:
            if tokens:
                # The device spent what it spent, whatever happened to the rest of the job. An
                # ending — a stop, an abandoned operation, a wait that ran out — is not a
                # rebate, and a daily budget blind to it would refuse nothing when it should.
                self.gate.charge(job.resource, tokens=tokens,
                                 seconds=time.monotonic() - waiting,
                                 note=f"operation(s) of job {job.id}")

    def _follow(self, operation: str) -> dict[str, Any]:
        """Ask the engine, on a bounded schedule, whether its operation finished.

        A status read is not a use of the device: it is the engine's own worker that runs the
        model, and it asks the gate for that. So this waits without holding anything, and the
        wait still has an end — a worker that polled forever would report a job as live long
        after anybody who could answer had stopped listening. When the end comes the honest
        state is uncertainty, and the identity on the job row is what settles it later.
        """
        for _ in range(self.follow_max_polls):
            answer = self.client.operation(operation)
            state = operation_state(answer)
            if state in OPERATION_DONE:
                return answer
            if state in OPERATION_STOPPED:
                raise OperationStopped(f"operation {operation} was cancelled"
                                       f"{_because(answer)}", operation_id=operation)
            if state in OPERATION_ABANDONED:
                raise OperationAbandoned(
                    f"operation {operation} ended {state}{_because(answer)}; nothing of this "
                    "submission is still running on the device")
            if state not in OPERATION_RUNNING:
                raise OperationAbandoned(
                    f"operation {operation} answered {state!r}, which the pinned backend does "
                    "not document. An answer nobody recognises is neither evidence that the "
                    "projection happened nor evidence that it will")
            self.sleeper(self.follow_poll_s)
        raise HindsightUnavailable(
            f"operation {operation} was still unfinished after {self.follow_max_polls} looks "
            f"({self.follow_max_polls * self.follow_poll_s:g}s). This process holds no slot "
            "for it — the engine runs the operation under its own admission — so the job waits "
            "as uncertain work, and the identity on the row is what `hermes-memory form "
            "--reconcile` asks the backend about")


def _tokens_from(*bodies: dict[str, Any]) -> int:
    """The first answer that names a cost.

    An async submission reports none at submission time and the completed operation may
    report the whole run, so both are asked and the one that knows is charged — never both.
    """
    for body in bodies:
        usage = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(usage, dict):
            continue
        total = usage.get("total_tokens") or usage.get("tokens") or 0
        try:
            tokens = max(0, int(total))
        except (TypeError, ValueError):
            continue
        if tokens:
            return tokens
    return 0


def _because(answer: dict[str, Any]) -> str:
    """What the backend said why, when it said any why at all."""
    reason = operation_reason(answer)
    return f": {reason[:300]}" if reason else ""
