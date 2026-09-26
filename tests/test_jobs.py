"""C12 job state machine: leases, epoch fencing, budgets and honest coverage."""
from __future__ import annotations

import pytest

from hermes_memory.processing.jobs import (CANCELLED, LEASED, QUARANTINED, QUEUED, RETRY_WAIT,
                                           RUNNING, SUCCEEDED, SUBMITTING, UNCERTAIN, JobQueue)
from hermes_memory.processing.routes import PRIORITY, Route
from hermes_memory.storage.evidence import EvidenceError

from conftest import envelope

REMOTE = Route("retain", "remote-9b", "chat", "http://127.0.0.1:8080/v1", "cred", "freshness",
               2048)
MAINT = Route("consolidate", "remote-9b", "chat", "http://127.0.0.1:8080/v1", "cred2",
              "maintenance", 2048)
FINGERPRINT = "extractor-v3"


@pytest.fixture()
def jobs(store):
    return JobQueue(store)


def enqueued(jobs, *, inputs=("rec_a",), route=REMOTE, priority="maintenance",
             fingerprint=FINGERPRINT, **kwargs):
    return jobs.enqueue(kind="retain", inputs=list(inputs), input_revision="1", route=route,
                        processor_fingerprint=fingerprint, priority=priority, **kwargs)


# -- identity and dedup ------------------------------------------------------

def test_the_same_work_enqueued_twice_is_one_job(jobs):
    first = enqueued(jobs, inputs=["rec_a", "rec_b"])
    second = enqueued(jobs, inputs=["rec_b", "rec_a"])
    assert first["job_id"] == second["job_id"]
    assert second["created"] is False


def test_a_different_processor_version_is_different_work(jobs):
    a = enqueued(jobs, fingerprint="extractor-v3")
    b = enqueued(jobs, fingerprint="extractor-v4")
    assert a["job_id"] != b["job_id"], "coverage is per processor, not per kind"


def test_a_new_input_revision_resumes_the_original_row_without_restarting_attempts(jobs):
    a = enqueued(jobs, inputs=["rec_a"])
    job = jobs.get(a["job_id"])
    jobs.partial(job, covered=0, reason="truncated")
    b = enqueued(jobs, inputs=["rec_a"])
    assert b["job_id"] == a["job_id"]
    assert jobs.get(a["job_id"]).attempts == 1, "a re-enqueue must not reset the attempt counter"


@pytest.mark.parametrize("overrides, message", [
    ({"inputs": []}, "between 1 and 500"),
    ({"inputs": [f"rec_{i}" for i in range(501)]}, "between 1 and 500"),
    ({"max_attempts": 0}, "between 1 and 10"),
    ({"max_attempts": 99}, "between 1 and 10"),
    ({"token_budget": 0}, "finite and positive"),
])
def test_unbounded_or_empty_jobs_are_refused(jobs, overrides, message):
    arguments = {"kind": "retain", "inputs": ["rec_a"], "input_revision": "1", "route": REMOTE,
                 "processor_fingerprint": FINGERPRINT, "max_attempts": 3, "token_budget": 1000}
    arguments.update(overrides)
    with pytest.raises(EvidenceError, match=message):
        jobs.enqueue(**arguments)


def test_an_empty_processor_fingerprint_is_refused(jobs):
    with pytest.raises(EvidenceError, match="processor_fingerprint is required"):
        jobs.enqueue(kind="retain", inputs=["rec_a"], input_revision="1", route=REMOTE,
                     processor_fingerprint="  ")


# -- dispatch ----------------------------------------------------------------

def test_priority_beats_arrival_order(jobs):
    old_maintenance = enqueued(jobs, inputs=["rec_m"], priority="maintenance")
    newer_urgent = enqueued(jobs, inputs=["rec_u"], priority="interactive")
    claimed = jobs.claim(worker="w1")
    assert claimed.id == newer_urgent["job_id"]
    assert claimed.priority == PRIORITY["interactive"]
    assert jobs.get(old_maintenance["job_id"]).state == QUEUED


def test_a_claimed_job_is_not_handed_to_a_second_worker(jobs):
    enqueued(jobs)
    assert jobs.claim(worker="w1") is not None
    assert jobs.claim(worker="w2") is None


def test_an_expired_lease_becomes_uncertain_not_available(jobs, store):
    clock = tests_clock = _Clock()
    jobs.clock = clock
    enqueued(jobs)
    first = jobs.claim(worker="w1", ttl=10)
    clock.advance(50)
    assert jobs.claim(worker="w2") is None, "an expired lease does not prove the work stopped"
    assert jobs.get(first.id).state == UNCERTAIN


class _Clock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_the_submission_identity_is_recorded_before_the_request(jobs):
    enqueued(jobs)
    job = jobs.claim(worker="w1")
    jobs.begin_submission(job, submission_id="sub-123")
    fresh = jobs.get(job.id)
    assert fresh.state == SUBMITTING and fresh.submission_id == "sub-123"
    jobs.mark_running(job, operation_id="op-9")
    assert jobs.get(job.id).backend_operation_id == "op-9"


def test_an_overdue_job_is_not_dispatched(jobs):
    enqueued(jobs, deadline=1.0)
    assert jobs.claim(worker="w1") is None
    assert jobs.counts().get(QUEUED) == 1


# -- completion --------------------------------------------------------------

def test_success_requires_coverage_of_every_input(jobs):
    created = enqueued(jobs, inputs=["rec_a", "rec_b"])
    job = jobs.claim(worker="w1")
    outcome = jobs.complete(job, covered=["rec_a"])
    assert outcome["complete"] is False and outcome["uncovered"] == ["rec_b"]
    assert jobs.get(created["job_id"]).state == RETRY_WAIT


def test_a_job_may_not_claim_coverage_it_was_not_given(jobs):
    enqueued(jobs, inputs=["rec_a"])
    job = jobs.claim(worker="w1")
    with pytest.raises(EvidenceError, match="its own inputs"):
        jobs.complete(job, covered=["rec_a", "rec_invented"])


def test_full_coverage_succeeds_once_and_settles(jobs):
    created = enqueued(jobs, inputs=["rec_a", "rec_b"])
    job = jobs.claim(worker="w1")
    outcome = jobs.complete(job, covered=["rec_a", "rec_b"], tokens=500, seconds=2.0)
    assert outcome["complete"] is True
    settled = jobs.get(created["job_id"])
    assert settled.state == SUCCEEDED and settled.tokens_used == 500


def test_finishing_over_budget_is_partial_not_success(jobs):
    created = enqueued(jobs, inputs=["rec_a"], token_budget=100)
    job = jobs.get(created["job_id"])
    outcome = jobs.complete(job, covered=["rec_a"], tokens=5_000)
    assert outcome["complete"] is False
    assert "budget" in (jobs.get(created["job_id"]).last_error or "")


def test_repeated_malformed_output_is_quarantined_rather_than_retried_forever(jobs):
    created = enqueued(jobs, max_attempts=3)
    job = jobs.get(created["job_id"])
    states = []
    for _ in range(4):
        states.append(jobs.partial(job, covered=0, reason="truncated json"))
    assert states[-1] == QUARANTINED
    assert jobs.get(created["job_id"]).attempts == 3
    assert jobs.claim(worker="w") is None, "quarantined work is not silently re-dispatched"


def test_a_quarantined_job_can_be_requeued_after_a_fix(jobs):
    created = enqueued(jobs, max_attempts=1)
    job = jobs.get(created["job_id"])
    jobs.retry(job, error="backend 500", backoff=0)
    assert jobs.get(created["job_id"]).state == QUARANTINED
    again = enqueued(jobs)
    assert again["created"] is True, "a quarantined row is replaced rather than resurrected"


# -- epoch and cancellation --------------------------------------------------

def test_a_reset_quarantines_work_created_under_the_old_epoch(jobs, store):
    enqueued(jobs)
    before = jobs.counts()[QUEUED]
    store.bump_epoch(reason="owner reset", actor="owner")
    assert jobs.abandon_stale_epoch() == before
    assert jobs.claim(worker="w") is None
    assert jobs.counts().get(QUARANTINED) == before


def test_a_cancelled_job_cannot_report_success_late(jobs):
    created = enqueued(jobs)
    job = jobs.get(created["job_id"])
    jobs.claim(worker="w1")
    jobs.cancel(created["job_id"], actor="owner", reason="not needed")
    assert jobs.get(created["job_id"]).state == CANCELLED
    with pytest.raises(EvidenceError, match="cancelled"):
        jobs._transition(created["job_id"], SUCCEEDED)


def test_a_succeeded_job_cannot_be_cancelled_after_the_fact(jobs):
    created = enqueued(jobs)
    job = jobs.claim(worker="w1")
    jobs.complete(job, covered=list(job.inputs))
    with pytest.raises(EvidenceError, match="retract its output"):
        jobs.cancel(created["job_id"], actor="owner", reason="changed my mind")


def test_the_reconciliation_backlog_is_queryable(jobs):
    enqueued(jobs, inputs=["rec_a"])
    job = jobs.claim(worker="w1")
    jobs.begin_submission(job, submission_id="sub-1")
    jobs.uncertain(job, reason="timed out waiting for acknowledgement")
    unresolved = jobs.unresolved()
    assert [item.id for item in unresolved] == [job.id]
    assert unresolved[0].state == UNCERTAIN


def test_a_job_past_its_attempt_ceiling_is_not_claimed_even_if_the_row_drifted(jobs):
    """The ceiling is enforced where the work is handed out, not only where it is recorded.

    `retry()` quarantines at the ceiling, so a queued row past `max_attempts` should not exist —
    and the claim does not assume it, because a queue whose guard depends on every writer
    having behaved is a queue that runs the extra attempt.
    """
    created = enqueued(jobs, max_attempts=2)
    jobs.db.execute("UPDATE processing_jobs SET attempts=2 WHERE id=?", (created["job_id"],))

    assert jobs.claim(worker="w1") is None
    assert jobs.get(created["job_id"]).state == QUEUED, "refusing to claim changes nothing"


def test_a_backed_off_retry_is_the_next_thing_a_worker_claims(jobs):
    """A retry that only a report can see is not a retry.

    Before this, ``retry()`` moved the row to ``retry_wait`` and ``claim()`` looked only at
    ``queued`` — so the job waited forever, the status report called it waiting, and the
    plan's bounded retry never happened.
    """
    clock = _Clock()
    jobs.clock = clock
    created = enqueued(jobs, max_attempts=5)
    first = jobs.claim(worker="w1")
    jobs.retry(first, error="503 saturated", backoff=60.0)

    assert jobs.claim(worker="w2") is None, "the backoff has not passed yet"

    clock.advance(61)
    again = jobs.claim(worker="w2")

    assert again is not None and again.id == created["job_id"]
    assert again.attempts == 1, "the retry is the second attempt of one job, not a new one"
    assert again.state == LEASED
    assert jobs.counts().get(RETRY_WAIT, 0) == 0


def test_retry_waits_its_backoff_before_becoming_eligible(jobs):
    clock = _Clock()
    jobs.clock = clock
    created = enqueued(jobs, max_attempts=5)
    job = jobs.claim(worker="w1")
    jobs.retry(job, error="503 saturated", backoff=60.0)
    assert jobs.ready_for_retry() == []
    clock.advance(61)
    assert [item.id for item in jobs.ready_for_retry()] == [job.id]


def test_no_job_path_leaves_a_transaction_open(jobs):
    enqueued(jobs)
    job = jobs.claim(worker="w1")
    for step in (lambda: jobs.counts(), lambda: jobs.unresolved(), lambda: jobs.quarantine(),
                 lambda: jobs.ready_for_retry(), lambda: jobs.begin_submission(job,
                                                                              submission_id="s"),
                 lambda: jobs.complete(job, covered=list(job.inputs))):
        step()
        assert jobs.db.in_transaction is False
    with pytest.raises(EvidenceError):
        jobs._transition("job_missing", SUCCEEDED)
    assert jobs.db.in_transaction is False
