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

from ..backend.document_map import DocumentMap
from ..backend.hindsight_client import HindsightError, HindsightUnavailable, SubmissionConflict
from .budgets import BudgetExhausted, Budgets
from .jobs import JobQueue, QUEUED
from .resource_gate import GatePaused, ResourceGate
from .routes import RouteTable

__all__ = ["FormationWorker", "AttemptOutcome"]

# A rough ceiling on what one retain will cost, used only to refuse a request
# that cannot possibly fit today's allowance. Actual usage is charged from what
# the backend reports.
ESTIMATED_TOKENS_PER_ITEM = 2_000


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
                 sleeper: Callable[[float], None] = time.sleep):
        self.store = store
        self.jobs = jobs
        self.gate = gate
        self.budgets = budgets
        self.documents = documents
        self.client = client
        self.routes = routes
        self.worker_id = worker_id
        self.sleeper = sleeper

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
        """Run one job, then always give the slot back or mark it uncertain.

        The release lives in ``finally`` rather than at the end of the body:
        a success path that returns from inside a ``try`` would otherwise leave
        the physical slot claimed forever, which is exactly the failure mode the
        gate exists to prevent.
        """
        tokens = 0
        gate_outcome = "succeeded"
        try:
            try:
                mappings = [self.documents.begin(record_id, job.input_revision,
                                                async_submission=False)
                            for record_id in job.inputs]
            except Exception as error:
                gate_outcome = "failed"
                state = self.jobs.retry(job, error=f"mapping failed: {error}"[:400],
                                        backoff=60.0)
                return AttemptOutcome(job.id, state, f"mapping failed: {error}")
            self.jobs.begin_submission(job, submission_id=mappings[0]["document_id"])
            covered: list[str] = []
            running = False
            for record_id, mapping in zip(job.inputs, mappings):
                # live_and_visible, not include_hidden: a record deleted by an
                # erasure or hidden by a supersession must never be re-projected
                # into the derived backend, and it is not coverage.
                if not self.store.live_and_visible(record_id):
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
                    # paid for. The operation identity stays empty unless this backend names
                    # one — a synchronous retain answers with content and no name, and
                    # inventing an id would hand `cancel --job` an operation the backend
                    # never heard of.
                    self.jobs.mark_running(job)
                    running = True
                body = self.client.retain(document_id=mapping["document_id"],
                                          content=evidence.text,
                                          timestamp=evidence.occurred_at,
                                          metadata={"source": evidence.source,
                                                    "record_id": record_id})
                operation = body.get("operation_id") if isinstance(body, dict) else None
                if operation and not job.backend_operation_id:
                    self.jobs.mark_running(job, operation_id=str(operation))
                self.documents.confirm(record_id, job.input_revision)
                covered.append(record_id)
                tokens += _tokens_from(body)
            outcome = self.jobs.complete(job, covered=covered, tokens=tokens)
            return AttemptOutcome(job.id, outcome["state"],
                                  f"covered {len(covered)}/{len(job.inputs)}")
        except SubmissionConflict as error:
            # The backend already holds this submission; reconcile, never resend.
            self.documents.reconcile(client=self.client, limit=max(1, len(job.inputs)))
            self.jobs.uncertain(job, reason=str(error)[:400])
            self.gate.mark_uncertain(reservation, reason=str(error)[:400])
            tokens = None  # slot is not being released; it is being quarantined
            return AttemptOutcome(job.id, "uncertain", str(error))
        except HindsightUnavailable as error:
            # We cannot claim the backend did not run this. Holding the slot as
            # uncertain is the honest state; a retry would double-spend the device.
            self.jobs.uncertain(job, reason=str(error)[:400])
            self.gate.mark_uncertain(reservation, reason=str(error)[:400])
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
            if tokens is not None:
                self.gate.release(reservation, outcome=gate_outcome, tokens=tokens,
                                  seconds=time.monotonic() - started)


def _tokens_from(body: dict[str, Any]) -> int:
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens") or usage.get("tokens") or 0
    try:
        return max(0, int(total))
    except (TypeError, ValueError):
        return 0
