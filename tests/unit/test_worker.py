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
from hermes_memory.storage.evidence import EvidenceError

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
    """A backend that answers like the pinned release.

    A submission reports no cost — nothing has been read yet, and it may name no operation
    of its own. The operation is the answer that says the model ran, and usage appears there.
    """
    class T:
        def __init__(self):
            self.calls = []
            self.reply = TransportResult(200, {"ok": True})
            self.operation = TransportResult(200, {"status": "completed",
                                                   "usage": {"total_tokens": 321}})

        def __call__(self, method, url, payload, headers):
            self.calls.append((method, url, payload))
            if "/operations/" in url:
                return self.operation
            return self.reply

    return T()


def submissions(transport):
    """The retains that left the process, ignoring the questions asked about them."""
    return [call for call in transport.calls if "/operations/" not in call[1]]


@pytest.fixture()
def harness(store, transport):
    """Everything a worker needs, with no backend and no network."""
    jobs = JobQueue(store)
    gate = ResourceGate(store)
    budgets = Budgets(store, daily={REMOTE: Budget(tokens=100_000)})
    docs = DocumentMap(store, bank_id=BANK)
    client = HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK, transport=transport)
    waits: list[float] = []
    worker = FormationWorker(store=store, jobs=jobs, gate=gate, budgets=budgets, documents=docs,
                             client=client, routes=ROUTES, worker_id="w1",
                             sleeper=waits.append, follow_poll_s=0.5, follow_max_polls=3)

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
                           transport=transport, waits=waits)


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
    assert harness.client.transport.states[0] == "running", \
        "the state is written before the first request leaves, not after the answer"
    assert set(harness.client.transport.states) == {"running"}, \
        "waiting for the operation is still the same running job, not a new one"


def test_the_operation_is_asked_about_under_the_identity_the_submission_carried(harness):
    """A name we invented would be a question about nothing; theirs is a receipt."""
    record = harness.record()
    harness.enqueue(record)
    harness.worker.run_once()
    submission = submissions(harness.transport)[-1][2]["operation_id"]
    asked = [call for call in harness.transport.calls if "/operations/" in call[1]]
    assert len(asked) == 1 and asked[0][1].endswith(f"/operations/{submission}"), \
        "the durable submission id is the operation id the backend dedupes on"
    assert submissions(harness.transport)[-1][2]["async"] is True, \
        "the slot is not held open by an http request that would time out first"


def test_an_async_answer_correlates_the_job_with_the_operation(harness):
    record = harness.record()
    job_id = harness.enqueue(record)
    harness.transport.reply = TransportResult(200, {"ok": True, "operation_id": "op-42"})
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


def test_a_charge_records_spend_without_claiming_another_admission(harness):
    """The engine's run cost tokens after our slot was released; it was not a second request.

    `calls` is how many times this installation was admitted to the device. Counting the
    follow-up charge as one more would make the day's busiest-looking number a fiction, and the
    budget that refuses work on it would be refusing on a count nobody made.
    """
    harness.gate.charge(REMOTE, tokens=120, seconds=1.5, note="operation(s) of job_x")
    used = harness.budgets.used(REMOTE)
    assert used["tokens"] == 120
    assert used["calls"] == 0, "spend arrived without a second admission"
    entry = harness.gate.db.execute("SELECT event, detail FROM gate_ledger WHERE event='charged'"
                                    ).fetchone()
    assert "job_x" in entry["detail"], "the ledger says what the charge was for"


def test_a_charge_cannot_refund_the_day_or_name_nothing(harness):
    """A primitive that writes usage must refuse an input that would unwrite it."""
    with pytest.raises(EvidenceError):
        harness.gate.charge(REMOTE, tokens=-5, seconds=0.0)
    with pytest.raises(EvidenceError):
        harness.gate.charge("", tokens=5, seconds=0.0)
    with pytest.raises(EvidenceError):
        harness.gate.charge(REMOTE, tokens="many", seconds=0.0)
    assert harness.budgets.used(REMOTE)["tokens"] == 0, "and none of it reached the ledger"


def test_the_retain_carries_the_real_event_time_and_source(harness):
    record = harness.record()
    harness.enqueue(record)
    harness.worker.run_once()
    payload = submissions(harness.transport)[-1][2]
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


# -- waiting for the operation ------------------------------------------------

class OperationScript:
    """A prepared answer for each question asked about an operation.

    Anything else the client asks goes to the transport underneath, so a test that
    scripts the waiting still sees the submissions that caused it.
    """

    def __init__(self, inner, answers):
        self.inner, self.answers = inner, list(answers)
        self.asks: list[str] = []

    def __call__(self, method, url, payload, headers):
        if "/operations/" not in url:
            return self.inner(method, url, payload, headers)
        self.asks.append(url)
        if self.answers:
            return self.answers.pop(0)
        return TransportResult(200, {"status": "completed",
                                     "usage": {"total_tokens": 321}})


def waiting_for(harness, *answers) -> OperationScript:
    script = OperationScript(harness.client.transport, list(answers))
    harness.client.transport = script
    return script


def test_a_still_running_operation_is_awaited_not_guessed(harness):
    record = harness.record()
    harness.enqueue(record)
    script = waiting_for(harness, TransportResult(200, {"status": "pending"}),
                         TransportResult(200, {"status": "processing"}))
    assert harness.worker.run_once().state == "succeeded"
    assert len(script.asks) == 3, "two looks that said not yet, a third that said done"
    assert harness.waits == [0.5, 0.5], "each unfinished answer costs one bounded wait"
    assert len(submissions(harness.transport)) == 1, "waiting for the work is not resending it"


def test_the_device_is_handed_back_before_the_engine_is_waited_for(harness):
    """Holding the slot across the engine's run is a deadlock, not caution.

    On the live machine this was the whole failure: the pass kept renewing a lease on
    `remote-9b` while it polled, and the engine's own worker — which asks that same gate for
    the same single slot in order to extract the facts — timed out four times and rescheduled
    the task. The thing being waited for can only finish if the waiter is not standing on it.
    """
    record = harness.record()
    harness.enqueue(record)
    waiting_for(harness, TransportResult(200, {"status": "processing"}),
                TransportResult(200, {"status": "completed",
                                      "usage": {"total_tokens": 321}}))
    seen: list[list[str]] = []
    renewals: list[object] = []

    def watch(seconds):
        seen.append(harness.gate.blocked_resources())
        harness.waits.append(seconds)

    real_renew = harness.gate.renew
    harness.gate.renew = lambda reservation, **kwargs: (
        renewals.append(1), real_renew(reservation, **kwargs))[1]
    harness.worker.sleeper = watch
    assert harness.worker.run_once().state == "succeeded"
    assert seen and all(blocked == [] for blocked in seen), \
        "every round of the wait finds the device free for the engine to admit itself"
    assert not renewals, "a wait that renews a lease it no longer needs is the bug, not the fix"


def test_a_second_job_is_admitted_while_the_first_operation_is_still_running(harness):
    """One device, one in-flight inference — and the inference is the engine's, not ours."""
    first, second = harness.record(), harness.record()
    harness.enqueue(first)
    harness.enqueue(second)
    waiting_for(harness,
                TransportResult(200, {"status": "processing"}),
                TransportResult(200, {"status": "completed", "usage": {"total_tokens": 321}}),
                TransportResult(200, {"status": "completed", "usage": {"total_tokens": 321}}))

    def admit_the_next_while_waiting(seconds):
        harness.waits.append(seconds)
        if harness.gate.occupancy():
            return
        granted = harness.gate.try_acquire(route="retain", holder="somebody-else",
                                           resource=REMOTE, priority=1)
        assert granted is not None, "the slot is not ours to reserve while we only poll"
        harness.gate.release(granted, outcome="succeeded", tokens=0, seconds=0.0)

    harness.worker.sleeper = admit_the_next_while_waiting
    assert harness.worker.run_once().state == "succeeded"


def test_a_wait_that_runs_out_leaves_the_job_uncertain_and_frees_everything(harness):
    """An answer that never came is uncertainty about the job, not a claim on the device.

    The old code held the slot as uncertain on the reasoning that the model might still be
    reading the document. That reasoning belongs to the engine, which holds its own admission
    for exactly as long as it is running; a second claim from the submitter would be the
    double-booking that starved it.
    """
    record = harness.record()
    job_id = harness.enqueue(record)
    script = waiting_for(harness, *[TransportResult(200, {"status": "processing"})] * 5)
    outcome = harness.worker.run_once()
    assert outcome.state == "uncertain"
    assert len(script.asks) == 3 == harness.worker.follow_max_polls, "bounded, not forever"
    assert "reconcile" in outcome.detail
    assert harness.jobs.get(job_id).state == UNCERTAIN
    assert harness.gate.blocked_resources() == [], "the engine owns its own admission now"
    assert harness.jobs.get(job_id).backend_operation_id, \
        "the identity left on the row is what a later pass asks the backend about"


def test_an_operation_that_never_arrived_is_not_waited_for(harness):
    record = harness.record()
    job_id = harness.enqueue(record)
    waiting_for(harness, TransportResult(404, {"detail": "no such operation"}))
    outcome = harness.worker.run_once()
    assert outcome.state == "retry_wait", "an operation that never existed may be submitted again"
    assert harness.jobs.get(job_id).attempts == 1
    assert harness.gate.blocked_resources() == [], "the device is provably not busy with it"


def test_an_operation_the_backend_failed_says_why(harness):
    record = harness.record()
    job_id = harness.enqueue(record)
    waiting_for(harness, TransportResult(
        200, {"status": "failed", "error_message": "extraction produced no facts"}))
    outcome = harness.worker.run_once()
    assert "extraction produced no facts" in outcome.detail, "the reason, not just the state"
    assert harness.jobs.get(job_id).last_error == outcome.detail
    assert harness.gate.blocked_resources() == []


def test_an_operation_the_operator_cancelled_ends_the_job_instead_of_resubmitting(harness):
    """A retry would undo their decision, so this answer is not a backend failure."""
    record = harness.record()
    job_id = harness.enqueue(record)
    waiting_for(harness, TransportResult(200, {"status": "cancelled"}))
    outcome = harness.worker.run_once()
    assert outcome.state == "cancelled"
    assert harness.jobs.get(job_id).state == "cancelled"
    assert len(submissions(harness.transport)) == 1, "stopped work is not work to retry"


def test_work_the_backend_already_did_is_still_paid_for_when_the_rest_is_stopped(harness):
    """An end to the job is not an erasure of the spend inside it."""
    first, second = harness.record(), harness.record()
    harness.enqueue(first, second)
    waiting_for(harness,
                TransportResult(200, {"status": "completed", "usage": {"total_tokens": 321}}),
                TransportResult(200, {"status": "cancelled"}))
    outcome = harness.worker.run_once()
    assert outcome.state == "cancelled"
    assert harness.budgets.used(REMOTE)["tokens"] == 321, \
        "the device spent what it spent, whatever happened to the rest"
    assert harness.docs.state(first, "1") == VERIFIED
    assert harness.docs.state(second, "1") == "submitted", "stopped work is not coverage"


def test_a_state_the_pinned_backend_does_not_document_is_not_guessed(harness):
    """`succeeded` is not in the operation enum; reading it as `completed` would be a lie."""
    record = harness.record()
    job_id = harness.enqueue(record)
    waiting_for(harness, TransportResult(200, {"status": "succeeded"}))
    outcome = harness.worker.run_once()
    assert outcome.state == "retry_wait" and "succeeded" in outcome.detail
    assert harness.docs.state(record, "1") == "submitted", "no coverage claimed for it"


def test_a_job_that_partially_landed_does_not_resend_what_is_already_verified(harness):
    """A second submission of a document already projected would spend the device twice."""
    first, second = harness.record(), harness.record()
    harness.docs.begin(first, "1", async_submission=True)
    harness.docs.confirm(first, "1")
    harness.enqueue(first, second)
    assert harness.worker.run_once().state == "succeeded"
    assert len(submissions(harness.transport)) == 1, "one submission for the one open question"
    assert harness.docs.state(second, "1") == VERIFIED

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
    monkeypatch.setattr(harness.client, "retain_async",
                        lambda *a, **k: (_ for _ in ()).throw(ZeroDivisionError("bug")))
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
