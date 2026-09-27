"""C12 worker and budgets: the guarded formation path, offline."""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from hermes_memory.backend.document_map import VERIFIED, DocumentMap
from hermes_memory.backend.hindsight_client import (HindsightClient, HindsightUnavailable,
                                                   TransportResult)
from hermes_memory.processing.budgets import Budget, BudgetExhausted, Budgets
from hermes_memory.processing.jobs import QUEUED, UNCERTAIN, JobQueue
from hermes_memory.processing.resource_gate import ResourceGate
from hermes_memory.processing.routes import PRIORITY, RouteTable, Route
from hermes_memory.processing.worker import FormationWorker

from conftest import envelope

BANK = "hermes"
GPU = "local-gpu"
REMOTE = "remote-9b"

RETAIN = Route("retain", REMOTE, "chat", "http://127.0.0.1:8813/v1", "cred-retain", "freshness",
               2048)
ROUTES = RouteTable({"retain": RETAIN})
FINGERPRINT = "extractor-v3"


@pytest.fixture()
def transport():
    class T:
        def __init__(self):
            self.calls = []
            self.reply = TransportResult(200, {"ok": True, "usage": {"total_tokens": 321}})

        def __call__(self, method, url, payload, headers):
            self.calls.append((method, url, payload))
            return self.reply

    return T()


@pytest.fixture()
def harness(store, transport):
    """Everything a worker needs, with no backend and no network."""
    jobs = JobQueue(store)
    gate = ResourceGate(store)
    budgets = Budgets(store, daily={REMOTE: Budget(tokens=100_000)})
    docs = DocumentMap(store, bank_id=BANK)
    client = HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK, transport=transport)
    worker = FormationWorker(store=store, jobs=jobs, gate=gate, budgets=budgets, documents=docs,
                            client=client, routes=ROUTES, worker_id="w1")

    counter = iter(range(1, 10_000))

    def record(text="The invoice was paid."):
        # A distinct source_id per call: the store refuses two different bodies
        # under one revision, and this helper is used to make several records.
        return store.commit(envelope(source_id=f"msg-{next(counter)}", text=text))["id"]

    def enqueue(*inputs, priority="maintenance"):
        return jobs.enqueue(kind="retain", inputs=list(inputs), input_revision="1",
                            route=RETAIN, processor_fingerprint=FINGERPRINT,
                            priority=priority)["job_id"]

    return SimpleNamespace(store=store, jobs=jobs, gate=gate, budgets=budgets, docs=docs,
                           client=client, worker=worker, record=record, enqueue=enqueue,
                           transport=transport)


# -- the happy path ----------------------------------------------------------

def test_a_job_runs_and_frees_its_slot(harness):
    record = harness.record()
    job_id = harness.enqueue(record)
    outcome = harness.worker.run_once()
    assert outcome.state == "succeeded" and "1/1" in outcome.detail
    assert harness.gate.blocked_resources() == [], "the slot must not stay claimed"
    assert harness.jobs.get(job_id).tokens_used == 321


class Watches:
    """The transport, asked to report what the queue said at the moment it was called."""

    def __init__(self, inner, jobs, job_id):
        self.inner, self.jobs, self.job_id = inner, jobs, job_id
        self.states: list[str] = []

    def __call__(self, *arguments):
        self.states.append(self.jobs.get(self.job_id).state)
        return self.inner(*arguments)


def test_a_job_says_it_is_running_before_the_request_leaves(harness):
    """A worker that died here otherwise leaves a row claiming nothing was spent."""
    record = harness.record()
    job_id = harness.enqueue(record)
    harness.client.transport = Watches(harness.client.transport, harness.jobs, job_id)
    harness.worker.run_once()
    assert harness.client.transport.states == ["running"], \
        "the state is written once, before the first retain, not after the answer"


def test_an_operation_identity_is_recorded_only_when_the_backend_names_one(harness):
    record = harness.record()
    job_id = harness.enqueue(record)
    assert harness.jobs.get(job_id).backend_operation_id is None
    harness.worker.run_once()
    assert harness.jobs.get(job_id).backend_operation_id is None, \
        "a synchronous retain answers with content and no name; inventing one would give " \
        "cancellation an operation the backend never heard of"


def test_an_async_answer_correlates_the_job_with_the_operation(harness):
    record = harness.record()
    job_id = harness.enqueue(record)
    harness.transport.reply = TransportResult(
        200, {"ok": True, "usage": {"total_tokens": 12}, "operation_id": "op-42"})
    harness.worker.run_once()
    assert harness.jobs.get(job_id).backend_operation_id == "op-42"


def test_measured_usage_is_charged_to_the_resource_that_earned_it(harness):
    record = harness.record()
    harness.enqueue(record)
    harness.worker.run_once()
    used = harness.budgets.used(REMOTE)
    assert used["tokens"] == 321 and used["calls"] == 1


def test_the_backend_document_is_verified_after_a_successful_retain(harness):
    record = harness.record()
    harness.enqueue(record)
    harness.worker.run_once()
    assert harness.docs.state(record, "1") == VERIFIED


def test_the_retain_carries_the_real_event_time_and_source(harness):
    record = harness.record()
    harness.enqueue(record)
    harness.worker.run_once()
    payload = harness.transport.calls[-1][2]
    item = payload["items"][0]
    assert item["timestamp"] == "2026-09-24T07:15:00+00:00", "the occurred time, not ingest time"
    assert item["metadata"]["source"] == "gmail" and item["metadata"]["record_id"] == record


# -- contention and refusal --------------------------------------------------

def test_a_busy_slot_returns_the_job_without_spending_an_attempt(harness):
    record = harness.record()
    job_id = harness.enqueue(record)
    held = harness.gate.try_acquire(route="foreground", holder="other", resource=REMOTE,
                                    priority=PRIORITY["interactive"])
    outcome = harness.worker.run_once()
    assert outcome.state == "slot_busy"
    job = harness.jobs.get(job_id)
    assert job.state == QUEUED and job.attempts == 0, "contention is not the job's fault"
    assert held.holder == "other"
    harness.gate.release(held, outcome="succeeded")
    assert harness.worker.run_once().state == "succeeded"


def test_an_exhausted_budget_stops_dispatch_before_the_slot_is_taken(harness):
    record = harness.record()
    harness.enqueue(record)
    # A budget one request cannot fit is a misconfiguration, not exhaustion;
    # this spends a workable allowance instead.
    harness.budgets.daily = {REMOTE: Budget(tokens=2_500)}
    harness.budgets.charge(REMOTE, tokens=2_400)
    outcome = harness.worker.run_once()
    assert outcome.state == "budget_exhausted"
    assert harness.jobs.get(outcome.job_id).state == QUEUED
    assert harness.gate.blocked_resources() == []
    assert harness.transport.calls == [], "no request may leave while over budget"


def test_a_resource_with_no_declared_budget_is_never_used(harness):
    record = harness.record()
    harness.enqueue(record)
    harness.budgets.daily = {}
    assert harness.worker.run_once().state == "budget_exhausted"
    assert harness.transport.calls == []


def test_an_operator_pause_stops_the_worker_immediately(harness):
    harness.enqueue(harness.record())
    harness.gate.pause(actor="owner", reason="gpu busy")
    outcome = harness.worker.run_once()
    assert outcome.state == "paused"
    assert harness.transport.calls == []
    assert harness.jobs.counts().get(QUEUED) == 1


def test_an_empty_queue_is_not_an_error(harness):
    assert harness.worker.run_once() is None
    assert harness.worker.drain()["attempted"] == 0


# -- failure semantics -------------------------------------------------------

def test_an_unreachable_backend_holds_the_slot_as_uncertain(harness):
    """Freeing it would let the next job onto a device still running the last one."""
    record = harness.record()
    job_id = harness.enqueue(record)
    harness.transport.reply = TransportResult(0, transport_error="connection reset")
    outcome = harness.worker.run_once()
    assert outcome.state == "uncertain"
    assert harness.jobs.get(job_id).state == UNCERTAIN
    assert harness.gate.blocked_resources() == [REMOTE]
    assert harness.transport.calls and len(harness.transport.calls) == 1, "no retry storm"


def test_a_500_retries_and_frees_the_slot(harness):
    record = harness.record()
    job_id = harness.enqueue(record)
    harness.transport.reply = TransportResult(500, {"detail": "extraction failed"})
    outcome = harness.worker.run_once()
    assert outcome.state == "retry_wait"
    assert harness.jobs.get(job_id).attempts == 1
    assert harness.gate.blocked_resources() == [], "the request completed, badly"


def test_evidence_forged_mid_flight_is_not_counted_as_coverage(harness, store):
    from hermes_memory.lifecycle.erasure import ErasureManager

    record = harness.record()
    job_id = harness.enqueue(record)
    manager = ErasureManager(store, owner_principal="owner")
    preview = manager.preview(record_ids=[record], actor="owner", reason="withdrawn")
    manager.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                    actor="owner")
    outcome = harness.worker.run_once()
    assert "0/1" in outcome.detail
    assert harness.transport.calls == [], "nothing was sent for forgotten evidence"


def test_a_worker_defect_is_labelled_as_one(harness, monkeypatch):
    record = harness.record()
    harness.enqueue(record)
    monkeypatch.setattr(harness.client, "retain",
                        lambda **kwargs: (_ for _ in ()).throw(ZeroDivisionError("bug")))
    outcome = harness.worker.run_once()
    assert "worker error" in outcome.detail
    assert harness.gate.blocked_resources() == []


# -- budgets -----------------------------------------------------------------

def test_a_request_larger_than_the_whole_day_is_refused(store):
    budgets = Budgets(store, daily={REMOTE: Budget(tokens=1000)})
    with pytest.raises(Exception, match="may not be larger than the whole day"):
        budgets.admit(REMOTE, estimated_tokens=50_000)


def test_a_zero_allowance_disables_rather_than_unlimits(store):
    budgets = Budgets(store, daily={REMOTE: Budget(tokens=0)})
    with pytest.raises(BudgetExhausted):
        budgets.admit(REMOTE, estimated_tokens=1)
    assert budgets.report()["resources"][REMOTE]["enabled"] is False


def test_negative_usage_is_refused(store):
    budgets = Budgets(store, daily={REMOTE: Budget(tokens=100)})
    with pytest.raises(Exception, match="negative"):
        budgets.charge(REMOTE, tokens=-1)


def test_the_call_ceiling_binds_even_with_tokens_left(store):
    budgets = Budgets(store, daily={REMOTE: Budget(tokens=100_000, calls=2)})
    budgets.charge(REMOTE, tokens=5)
    budgets.charge(REMOTE, tokens=5)
    with pytest.raises(BudgetExhausted):
        budgets.admit(REMOTE, estimated_tokens=5)


def test_usage_rolls_over_across_periods(store):
    clock = {"day": "2026-09-25"}
    budgets = Budgets(store, daily={REMOTE: Budget(tokens=100)},
                      clock=lambda: _Day(clock["day"]))
    budgets.charge(REMOTE, tokens=90)
    with pytest.raises(BudgetExhausted):
        budgets.admit(REMOTE, estimated_tokens=20)
    clock["day"] = "2026-09-26"
    assert budgets.used(REMOTE)["tokens"] == 0
    budgets.admit(REMOTE, estimated_tokens=20)


class _Day:
    def __init__(self, value):
        self.value = value

    def strftime(self, fmt):
        return self.value


# -- drain -------------------------------------------------------------------

def test_drain_processes_until_the_queue_is_empty(harness):
    for index in range(3):
        harness.enqueue(harness.record(text=f"note {index}"))
    report = harness.worker.drain(max_jobs=10)
    assert report["attempted"] == 3
    assert report["counts"].get("succeeded") == 3
    assert report["budget"]["resources"][REMOTE]["tokens_used"] == 963


def test_a_worker_vouches_for_every_item_of_a_long_pass(harness):
    """A pass over several inputs outlives the ttl it was claimed under."""
    seen = []
    real = harness.jobs.renew

    def renew(job, **kwargs):
        answer = real(job, **kwargs)
        seen.append(answer)
        return answer

    harness.jobs.renew = renew
    records = [harness.record() for _ in range(3)]
    job_id = harness.enqueue(*records)
    outcome = harness.worker.run_once()
    assert outcome.state == "succeeded"
    assert len(seen) == 3, "one promise per input, not one at the start and hope"
    assert all(seen), "the row still carried this worker's lease at every one of them"
    assert harness.jobs.get(job_id).lease is None, "success gives the claim back"
