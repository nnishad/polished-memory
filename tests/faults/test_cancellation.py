"""Stopping work: the queue half, the ledger half, and what neither is allowed to claim.

The three pieces existed and never met. `JobQueue.cancel` closed a row,
`OperationLedger.request_cancellation` wrote an intent the pinned worker honours before it
starts an operation, and `HindsightClient.cancel_operation` was a telephone nobody rang — so
an operation already handed to the backend carried on after the job behind it was stopped,
and the `cancellation='confirmed'` the schema permits was never written by anybody. What this
composes is deliberately narrow: stop the part that is ours, ask about the part that is not,
and record which of those two happened. The honest residue of a lost connection is
`uncertain` with the intent still on file, because a wish is not an answer.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_memory.backend.capabilities import UnsupportedCapability
from hermes_memory.backend.hindsight_client import HindsightError, HindsightUnavailable
from hermes_memory.backend.worker_launcher import (OPERATION_KEY, LauncherRefused,
                                                   OperationLedger)
from hermes_memory.processing.cancellation import (BACKEND_OUTCOMES, Canceller, outstanding,
                                                   run)
from hermes_memory.processing.instance_gate import GATE_FILENAME, GateStore, gate_path
from hermes_memory.processing.jobs import CANCELLED, JobQueue, RUNNING
from hermes_memory.processing.routes import Route
from hermes_memory.storage.evidence import EvidenceError

OWNER = "owner-principal"
BANK = "p-personal"
RESOURCE = "remote-9b"
REMOTE = Route("retain", RESOURCE, "chat", "http://127.0.0.1:8080/v1", "cred", "freshness",
               2048)


class Settings:
    """The handful of attributes the doors read, and nothing else."""

    def __init__(self, root):
        self.db_path = Path(root) / "canonical.db"
        self.home = Path(root)
        self.owner_principal = OWNER
        self.profile = "test"
        self.hindsight_url = ""
        self.hindsight_api_key_env = "HERMES_MEMORY_TEST_KEY"


class Client:
    """A backend that answers in one fixed way, and remembers being asked."""

    def __init__(self, outcome=None):
        self.outcome = {"state": "cancelled"} if outcome is None else outcome
        self.calls: list[str] = []

    def cancel_operation(self, operation_id):
        self.calls.append(operation_id)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class WatchesIntent:
    """A backend that looks over its own shoulder: what did the ledger say at call time?"""

    def __init__(self, ledger):
        self.ledger = ledger
        self.seen: list[str] = []

    def cancel_operation(self, operation_id):
        self.seen.append(self.ledger.get(operation_id)["cancellation"])
        return {"state": "cancelled"}


@pytest.fixture()
def dispatched(store):
    """A job leased, submitted and running, with its operation recorded in the gate ledger.

    The gate is a separate file from the store on purpose, and the fixture keeps both: the
    interesting part of a cancellation is what crosses that boundary.
    """
    gate = GateStore(":memory:")
    jobs = JobQueue(store)
    ledger = OperationLedger(gate)

    def build(operation_id="op-1", *, record="rec_a", state=RUNNING, tokens=0):
        job_id = jobs.enqueue(kind="retain", inputs=[record], input_revision="1", route=REMOTE,
                              processor_fingerprint="extractor-v3")["job_id"]
        claimed = jobs.claim(worker="w1")
        jobs.mark_running(claimed, operation_id=operation_id)
        ledger.record({OPERATION_KEY: operation_id, "bank_id": BANK,
                       "operation_type": "retain"}, worker_id="w1", bank_id=BANK,
                      resource=RESOURCE)
        ledger.mark(operation_id, "running", tokens=tokens)
        if state != "running":
            ledger.mark(operation_id, state)
        return job_id

    def canceller(client=None, **kwargs):
        return Canceller(store, jobs=jobs, ledger=ledger,
                         client=client if client is not None else Client(), **kwargs)

    build.canceller = canceller
    build.jobs = jobs
    build.ledger = ledger
    yield build
    gate.db.close()


@pytest.fixture()
def ledger_free():
    """A bare admission ledger, for the two transitions the Canceller does not make."""
    gate = GateStore(":memory:")
    yield OperationLedger(gate)
    gate.db.close()


def one(report):
    """The single operation inside a job report — the shape the job door returns."""
    assert len(report["operations"]) == 1
    return report["operations"][0]


# -- the two halves in order ---------------------------------------------------

def test_the_intent_is_on_file_before_the_request_leaves_the_process(dispatched):
    """A crash mid-call must leave "somebody asked" behind, not an un-cancellable operation."""
    watcher = WatchesIntent(dispatched.ledger)
    dispatched("op-1")
    dispatched.canceller(watcher).cancel_operation("op-1", actor=OWNER, reason="wrong bank")
    assert watcher.seen == ["requested"], "the backend was asked before the intent was written"
    assert dispatched.ledger.get("op-1")["cancellation"] == "confirmed"


def test_an_acknowledged_cancellation_closes_the_job_and_the_operation(dispatched):
    client = Client()
    job_id = dispatched("op-1")
    report = dispatched.canceller(client).cancel_job(job_id, actor=OWNER, reason="wrong bank")
    assert report["job_state"] == CANCELLED
    assert one(report)["backend"] == "confirmed"
    assert client.calls == ["op-1"], "the job door rang the telephone the queue could not"
    row = dispatched.ledger.get("op-1")
    assert (row["state"], row["cancellation"]) == ("cancelled", "confirmed")


def test_a_cancelled_job_is_never_offered_to_a_worker_again(dispatched):
    job_id = dispatched("op-1")
    dispatched.canceller().cancel_job(job_id, actor=OWNER, reason="superseded")
    assert dispatched.jobs.claim(worker="w2") is None
    assert dispatched.jobs.get(job_id).state == CANCELLED


def test_a_job_that_never_reached_the_backend_reports_no_operation(dispatched):
    """A queued job has nothing to ask about, and saying so is the whole answer."""
    job_id = dispatched.jobs.enqueue(
        kind="retain", inputs=["rec_q"], input_revision="1", route=REMOTE,
        processor_fingerprint="extractor-v3")["job_id"]
    report = dispatched.canceller().cancel_job(job_id, actor=OWNER, reason="mistake")
    assert report["operations"] == [] and report["job_state"] == CANCELLED
    assert dispatched.ledger.report()["by_state"] == {}


def test_the_spent_allowance_is_reported_and_not_invented(dispatched):
    job_id = dispatched("op-1")
    dispatched.jobs.retry(dispatched.jobs.get(job_id), error="provider reset", backoff=0.0)
    report = dispatched.canceller().cancel_job(job_id, actor=OWNER, reason="superseded")
    assert report["spent"] == {"tokens": 0, "attempts": 1}


# -- what the backend can say ---------------------------------------------------

def test_an_unreachable_backend_leaves_the_operation_uncertain_never_cancelled(dispatched):
    dispatched("op-1")
    report = dispatched.canceller(Client(HindsightUnavailable("connection reset"))
                                 ).cancel_operation("op-1", actor=OWNER, reason="owner asked")
    assert report["backend"] == "unreachable"
    assert report["state_after"] == "uncertain"
    assert report["cancellation"] == "requested", "the report says what is still owed"
    row = dispatched.ledger.get("op-1")
    assert (row["state"], row["cancellation"]) == ("uncertain", "requested")
    assert "uncertain rather than as cancelled" in report["note"]
    assert "cancel could not be confirmed" in row["error"]


def test_a_refused_cancellation_changes_nothing_about_the_operation(dispatched):
    dispatched("op-1")
    report = dispatched.canceller(Client(HindsightError("HTTP 409", status=409))
                                 ).cancel_operation("op-1", actor=OWNER, reason="too late")
    assert report["backend"] == "refused"
    row = dispatched.ledger.get("op-1")
    assert (row["state"], row["cancellation"]) == (RUNNING, "requested")
    assert "nothing about the operation changed" in report["note"]


def test_a_revision_without_the_endpoint_is_an_answer_not_a_failure(dispatched):
    """The pinned revision may simply not have the route. The intent stands either way."""
    dispatched("op-1")
    report = dispatched.canceller(Client(UnsupportedCapability("not available on 0.9.9"))
                                 ).cancel_operation("op-1", actor=OWNER, reason="owner asked")
    assert report["backend"] == "unsupported"
    assert dispatched.ledger.get("op-1")["state"] == RUNNING
    assert "no cancellation operation" in report["note"]


def test_an_installation_with_no_route_still_files_the_intent(dispatched):
    dispatched("op-1")
    report = Canceller(None, ledger=dispatched.ledger).cancel_operation(
        "op-1", actor=OWNER, reason="owner asked")
    assert report["backend"] == "not_attempted"
    assert dispatched.ledger.get("op-1")["cancellation"] == "requested"
    assert "no backend route is configured" in report["note"]


def test_every_reported_outcome_is_one_the_backend_can_have(dispatched):
    dispatched("op-1")
    answer = dispatched.canceller().cancel_operation("op-1", actor=OWNER, reason="x")
    assert answer["backend"] in BACKEND_OUTCOMES


# -- the past is not cancellable --------------------------------------------------

@pytest.mark.parametrize("settled", ["finished", "cancelled", "refused"])
def test_a_settled_operation_is_never_made_true_by_a_later_request(dispatched, settled):
    client = Client()
    dispatched("op-1", state=settled)
    report = dispatched.canceller(client).cancel_operation("op-1", actor=OWNER,
                                                           reason="forgot earlier")
    assert report["backend"] == "not_attempted"
    assert client.calls == [], "no request, so no claim about a response"
    row = dispatched.ledger.get("op-1")
    assert row["state"] == settled
    assert row["cancellation"] == "none", "an intent nobody could act on is not filed"


def test_an_acknowledgement_that_arrives_after_the_work_finished_reports_both(dispatched):
    """The worker can finish while this request is in flight. Neither fact buries the other."""
    ledger = dispatched.ledger

    class Settles:
        def cancel_operation(self, operation_id):
            ledger.mark(operation_id, "finished", tokens=40)
            return {"state": "cancelled"}

    dispatched("op-1")
    report = dispatched.canceller(Settles()).cancel_operation("op-1", actor=OWNER,
                                                              reason="owner asked")
    assert report["backend"] == "confirmed"
    assert report["state_after"] == "finished"
    assert report["refused_transition"] == "finished -> cancelled"
    assert "already settled" in report["note"], "the note says both, not the convenient one"
    assert report["cancellation"] == "confirmed"
    assert ledger.get("op-1")["tokens"] == 40, "the charge the worker reported survives"


# -- the reading ----------------------------------------------------------------

def test_the_owed_list_is_the_intents_without_an_answer(dispatched):
    dispatched("op-1")
    dispatched("op-2", record="rec_b")
    canceller = dispatched.canceller(Client(HindsightUnavailable("reset")))
    canceller.cancel_operation("op-1", actor=OWNER, reason="owner asked")
    assert [row["operation_id"] for row in canceller.owed()] == ["op-1"]
    dispatched.canceller().cancel_operation("op-2", actor=OWNER, reason="owner asked")
    assert [row["operation_id"] for row in canceller.owed()] == ["op-1"], \
        "a confirmed stop leaves the queue of things still owed"


def test_the_owed_list_survives_the_process_that_asked(dispatched):
    """Nothing is remembered in memory: the intent is the durable record of the request."""
    dispatched("op-1")
    dispatched.canceller(Client(HindsightUnavailable("reset"))).cancel_operation(
        "op-1", actor=OWNER, reason="owner asked")
    reopened = Canceller(None, ledger=dispatched.ledger)
    assert [row["operation_id"] for row in reopened.owed()] == ["op-1"]
    assert len(reopened.owed(limit="nonsense")) == 1, \
        "an absurd bound falls back to the named default rather than reading the table"


def test_an_unwired_ledger_owes_nothing_and_says_so(store):
    assert Canceller(store, jobs=JobQueue(store)).owed() == []


# -- refusals -------------------------------------------------------------------

def test_a_succeeded_job_is_retracted_rather_than_cancelled(dispatched):
    job_id = dispatched.jobs.enqueue(
        kind="retain", inputs=["rec_done"], input_revision="1", route=REMOTE,
        processor_fingerprint="extractor-v3")["job_id"]
    job = dispatched.jobs.claim(worker="w1")
    assert job.id == job_id
    dispatched.jobs.mark_running(job, operation_id="op-9")
    dispatched.jobs.complete(dispatched.jobs.get(job_id), covered=["rec_done"], tokens=11)
    with pytest.raises(EvidenceError, match="retract"):
        dispatched.canceller().cancel_job(job_id, actor=OWNER, reason="too late")


@pytest.mark.parametrize("kwargs", [
    {"actor": "", "reason": "a reason"},
    {"actor": "   ", "reason": "a reason"},
    {"actor": OWNER, "reason": ""},
    {"actor": OWNER, "reason": "  "},
])
def test_a_cancellation_has_to_say_who_and_why(dispatched, kwargs):
    dispatched("op-1")
    canceller = dispatched.canceller()
    with pytest.raises(EvidenceError, match="must be named"):
        canceller.cancel_operation("op-1", **kwargs)
    with pytest.raises(EvidenceError, match="must be named"):
        canceller.cancel_job("nope", **kwargs)


def test_an_unknown_operation_or_job_is_refused_not_reported_as_stopped(dispatched):
    canceller = dispatched.canceller()
    with pytest.raises(EvidenceError, match="no record of operation"):
        canceller.cancel_operation("op-nope", actor=OWNER, reason="x")
    with pytest.raises(EvidenceError, match="no queued job"):
        canceller.cancel_job("job-nope", actor=OWNER, reason="x")
    assert canceller.client.calls == []


def test_a_canceller_with_no_ledger_says_nothing_is_ours_to_stop(store):
    """An operation the admission ledger never recorded cannot be attributed to us."""
    jobs = JobQueue(store)
    job_id = jobs.enqueue(kind="retain", inputs=["rec_x"], input_revision="1", route=REMOTE,
                          processor_fingerprint="extractor-v3")["job_id"]
    jobs.mark_running(jobs.claim(worker="w1"), operation_id="op-1")
    report = Canceller(store, jobs=jobs).cancel_job(job_id, actor=OWNER, reason="mistake")
    assert one(report)["backend"] == "not_attempted"
    assert one(report)["cancellation"] == "absent"


def test_a_record_only_canceller_owes_nothing(store):
    assert Canceller(store).owed() == []


# -- the ledger's own door --------------------------------------------------------

def test_confirming_a_cancellation_nobody_asked_for_is_refused(ledger_free):
    ledger_free.record({OPERATION_KEY: "op-1", "bank_id": BANK, "operation_type": "retain"},
                       worker_id="w1", bank_id=BANK, resource=RESOURCE)
    with pytest.raises(LauncherRefused, match="never asked to stop"):
        ledger_free.confirm_cancellation("op-1")
    assert ledger_free.get("op-1")["cancellation"] == "none"


def test_an_already_confirmed_cancellation_answers_with_itself(ledger_free):
    ledger_free.record({OPERATION_KEY: "op-1", "bank_id": BANK, "operation_type": "retain"},
                       worker_id="w1", bank_id=BANK, resource=RESOURCE)
    ledger_free.request_cancellation("op-1", actor=OWNER)
    first = ledger_free.confirm_cancellation("op-1")
    assert first["cancellation"] == "confirmed"
    assert ledger_free.confirm_cancellation("op-1") == first


def test_the_ledger_refuses_to_confirm_an_operation_it_has_no_record_of(ledger_free):
    with pytest.raises(LauncherRefused, match="no record of operation"):
        ledger_free.confirm_cancellation("op-nope")


# -- the doors --------------------------------------------------------------------

def test_the_door_refuses_an_installation_with_no_store(tmp_path):
    report = run(Settings(tmp_path), job="job-1", actor=OWNER, reason="mistake")
    assert report["ok"] is False and "no canonical store" in report["refused"]


def test_the_door_refuses_an_operation_no_ledger_ever_recorded(tmp_path):
    from hermes_memory.storage.evidence import EvidenceStore

    settings = Settings(tmp_path)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    EvidenceStore(settings.db_path).close()
    report = run(settings, operation="op-1", actor=OWNER, reason="mistake")
    assert report["ok"] is False and "nothing has been dispatched" in report["refused"]
    assert not gate_path(settings).exists(), "a refusal must not create the ledger it refused"


def test_the_door_closes_only_the_store_it_opened(tmp_path):
    """A caller handing in its own archive keeps it: the door borrows, it does not finish."""
    from hermes_memory.storage.evidence import EvidenceStore

    settings = Settings(tmp_path)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    with EvidenceStore(settings.db_path) as opened:
        jobs = JobQueue(opened)
        job_id = jobs.enqueue(kind="retain", inputs=["rec_a"], input_revision="1",
                              route=REMOTE, processor_fingerprint="extractor-v3")["job_id"]
        report = run(settings, job=job_id, actor=OWNER, reason="superseded", store=opened)
        assert report["ok"] is True
        assert jobs.get(job_id).state == CANCELLED, "the caller's handle is still usable"


def test_the_door_stops_a_queued_job_with_no_admission_ledger_at_all(tmp_path):
    """The queue is ours even on a machine that has never dispatched: no gate is needed."""
    from hermes_memory.storage.evidence import EvidenceStore

    settings = Settings(tmp_path)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    with EvidenceStore(settings.db_path) as opened:
        jobs = JobQueue(opened)
        job_id = jobs.enqueue(kind="retain", inputs=["rec_a"], input_revision="1",
                              route=REMOTE, processor_fingerprint="extractor-v3")["job_id"]
    report = run(settings, job=job_id, actor=OWNER, reason="superseded")
    assert report["ok"] is True and report["job_state"] == CANCELLED
    assert report["operations"] == []
    assert not gate_path(settings).exists()


def test_the_door_needs_exactly_one_target(tmp_path):
    settings = Settings(tmp_path)
    with pytest.raises(EvidenceError, match="exactly one"):
        run(settings, actor=OWNER, reason="mistake")
    with pytest.raises(EvidenceError, match="exactly one"):
        run(settings, job="job-1", operation="op-1", actor=OWNER, reason="mistake")


def test_the_door_cancels_over_the_installation_it_names(tmp_path):
    """The same composition the unit cases drive, reached through a real pair of files."""
    from hermes_memory.storage.evidence import EvidenceStore

    settings = Settings(tmp_path)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    with EvidenceStore(settings.db_path) as opened:
        gate = GateStore(gate_path(settings))
        jobs = JobQueue(opened)
        ledger = OperationLedger(gate)
        job_id = jobs.enqueue(kind="retain", inputs=["rec_a"], input_revision="1",
                              route=REMOTE, processor_fingerprint="extractor-v3")["job_id"]
        claimed = jobs.claim(worker="w1")
        jobs.mark_running(claimed, operation_id="op-1")
        ledger.record({OPERATION_KEY: "op-1", "bank_id": BANK, "operation_type": "retain"},
                      worker_id="w1", bank_id=BANK, resource=RESOURCE)
        ledger.mark("op-1", "running")
        gate.db.close()
    client = Client()
    report = run(settings, job=job_id, actor=OWNER, reason="wrong bank", client=client)
    assert report["ok"] is True and report["profile"] == "test"
    assert client.calls == ["op-1"], "the door used the installation, not a stub"
    assert report["operations"][0]["cancellation"] == "confirmed"
    assert outstanding(settings) == {"ledger": "present", "owed": []}


def test_the_door_without_a_configured_backend_files_the_intent_only(tmp_path):
    from hermes_memory.storage.evidence import EvidenceStore

    settings = Settings(tmp_path)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    with EvidenceStore(settings.db_path) as opened:
        gate = GateStore(gate_path(settings))
        OperationLedger(gate).record(
            {OPERATION_KEY: "op-1", "bank_id": BANK, "operation_type": "retain"},
            worker_id="w1", bank_id=BANK, resource=RESOURCE)
        gate.db.close()
    report = run(settings, operation="op-1", actor=OWNER, reason="owner asked")
    assert report["ok"] is True
    assert report["backend"] == "not_attempted", "no endpoint, so nothing was told"
    owed = outstanding(settings)["owed"]
    assert [row["operation_id"] for row in owed] == ["op-1"]
    assert owed[0]["cancellation"] == "requested" and owed[0]["state"] == "recorded"


def test_the_reading_reports_a_ledger_that_predates_operations(tmp_path):
    settings = Settings(tmp_path)
    assert outstanding(settings) == {
        "ledger": "absent", "at": str(gate_path(settings)), "owed": [],
        "note": "nothing has been queued from this installation yet"}
    gate = GateStore(gate_path(settings))
    gate.db.execute("DROP TABLE backend_operations")
    gate.db.close()
    report = outstanding(settings)
    assert report["ledger"] == "predates operations" and report["owed"] == []
    assert "operation table" in report["note"], "the reading names what it could not find"


def test_the_reading_does_not_bring_a_ledger_into_being(tmp_path):
    settings = Settings(tmp_path)
    outstanding(settings)
    assert not Path(tmp_path / GATE_FILENAME).exists()
