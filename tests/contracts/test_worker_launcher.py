"""§8.1.2: the pinned worker launcher, tested against fakes that keep their signatures.

Hindsight is not installed in this environment and must not be — so the parts under test
are the three things this adapter actually owns: it refuses a revision it did not read, it
refuses a configuration that could take two slots at once, and it writes down which
operation is about to run *before* the call goes out. The fake ``MemoryEngine`` and
``WorkerPoller`` declare the same keyword names as the pinned tag, which is what lets
``_accepts`` be checked for drift rather than waved through.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from hermes_memory.backend.worker_launcher import (HOLD_POLL_SECONDS, LAUNCHER_VERSION,
                                                   OPERATION_KEY, RETRY_KEY,
                                                   LauncherRefused, OperationLedger,
                                                   attribute_tasks, bring_up,
                                                   check, compose, database_not_listening,
                                                   inference_hold, main,
                                                   memory_arguments, poller_arguments,
                                                   slot_contract, version_contract,
                                                   wait_for_inference)
from hermes_memory.config import load_settings
from hermes_memory.processing.instance_gate import (GATE_SCHEMA_VERSION, GateStore,
                                                   gate_path)
from hermes_memory.processing.resource_gate import ResourceGate
from hermes_memory.processing.routes import Route, RouteTable

BANK = "hermes"
REMOTE = "remote-9b"
PINS = {"retain": Route("retain", REMOTE, "chat", "http://127.0.0.1:11434/v1", "cred",
                        "freshness", 2048)}
ROUTES = RouteTable(PINS)
TASK = {OPERATION_KEY: "op-1", "operation_type": "retain", "bank_id": BANK,
        RETRY_KEY: 0, "contents": [{"content": "note"}]}


class Config:
    """The §8.1.1 values a compliant installation is started with."""

    def __init__(self, **overrides):
        base = {"worker_max_slots": 1, "worker_slot_reservations": {},
                "retain_batch_enabled": False, "retain_max_completion_tokens": 2048,
                "consolidation_max_completion_tokens": 2048,
                "reflect_max_completion_tokens": 1024, "worker_poll_interval_ms": 500,
                "worker_max_retries": 3, "worker_consolidation_bank_priority": None,
                "database_schema": "public", "database_url":
                    "postgresql://hindsight:s3cr3t@127.0.0.1:5433/hindsight"}
        base.update(overrides)
        for key, value in base.items():
            setattr(self, key, value)


class Backend:
    supports_worker_poller = True


class Engine:
    """``MemoryEngine`` at the tag: these keywords, and nothing else, are what we pass.

    The method names are the pinned engine's own, because a double that invents one keeps the
    suite green while the call fails on a real machine.
    """

    def __init__(self, *, run_migrations, task_backend, tenant_extension,
                 operation_validator):
        self.run_migrations = run_migrations
        self.task_backend = task_backend
        self.tenant_extension = tenant_extension
        self.operation_validator = operation_validator
        self._backend = Backend()
        self.executed = []
        self.initialized = False
        self.closed = False

    async def execute_task(self, task):
        self.executed.append(task)

    async def initialize(self):
        self.initialized = True

    async def on_task_wall_timeout(self, task):
        return "timed out"

    async def close(self):
        self.closed = True


class Poller:
    def __init__(self, *, backend, worker_id, executor, poll_interval_ms, schema,
                 tenant_extension, max_slots, slot_reservations,
                 consolidation_bank_priority, max_retries, on_wall_timeout):
        self.arguments = {"backend": backend, "worker_id": worker_id, "executor": executor,
                          "poll_interval_ms": poll_interval_ms, "schema": schema,
                          "tenant_extension": tenant_extension, "max_slots": max_slots,
                          "slot_reservations": slot_reservations,
                          "consolidation_bank_priority": consolidation_bank_priority,
                          "max_retries": max_retries, "on_wall_timeout": on_wall_timeout}
        self.ran = False

    async def run(self):
        self.ran = True


class TaskBackend:
    pass


MODULES = {"MemoryEngine": Engine, "WorkerTaskBackend": TaskBackend, "WorkerPoller": Poller,
           "config": type("ConfigModule", (), {"DEFAULT_DATABASE_SCHEMA": "public"})}


@pytest.fixture()
def ledger(tmp_path):
    store = GateStore(tmp_path / "gate.db")
    yield OperationLedger(store)
    store.close()


def run(coroutine):
    return asyncio.run(coroutine)


def a_ledger():
    """A ledger over an in-memory gate database, for the composition tests."""
    return OperationLedger(GateStore(":memory:"))


# -- the two contracts -------------------------------------------------------

def test_a_revision_that_was_not_read_is_a_refusal_not_a_warning():
    assert version_contract(declared="0.10.1")["matches"] is True
    refusal = version_contract(declared="0.11.0")
    assert refusal["matches"] is False and refusal["declared"] == "0.11.0"
    assert refusal["pinned"] == "0.10.1", "the report says which of the two is stale"


def test_a_version_nobody_told_us_is_reported_as_unknown_rather_than_blank():
    assert version_contract(declared=None)["declared"] == "unknown"


def test_the_pinned_environment_satisfies_the_slot_contract():
    assert slot_contract(Config())["ok"] is True


@pytest.mark.parametrize(("override", "phrase"), [
    ({"worker_max_slots": 2}, "one slot per physical resource"),
    ({"worker_slot_reservations": {"consolidation": 2}}, "floor"),
    ({"retain_batch_enabled": True}, "outside the admission gate"),
    ({"retain_max_completion_tokens": None}, "explicit output cap"),
    ({"reflect_max_completion_tokens": 0}, "explicit output cap"),
])
def test_a_configuration_that_could_overrun_the_machine_never_starts(override, phrase):
    verdict = slot_contract(Config(**override))
    assert verdict["ok"] is False
    assert any(phrase in line for line in verdict["violations"]), verdict["violations"]


def test_the_slot_report_shows_what_was_observed_without_the_connection_string():
    verdict = json.dumps(slot_contract(Config(worker_max_slots=4)), sort_keys=True)
    assert "worker_max_slots" in verdict and "s3cr3t" not in verdict


# -- the ledger --------------------------------------------------------------

def test_an_operation_is_recorded_under_the_identity_the_poller_claimed(ledger):
    row = ledger.record(TASK, worker_id="w1", bank_id=BANK, resource=REMOTE)
    assert row["operation_id"] == "op-1" and row["kind"] == "retain"
    assert row["bank_id"] == BANK and row["resource"] == REMOTE
    assert row["state"] == "recorded" and row["attempts"] == 1


def test_a_retry_charges_the_same_operation_rather_than_becoming_new_work(ledger):
    ledger.record(TASK, worker_id="w1", bank_id=BANK, resource=REMOTE)
    second = ledger.record({**TASK, RETRY_KEY: 1}, worker_id="w2", bank_id=BANK,
                           resource=REMOTE)
    assert second["attempts"] == 2
    assert second["worker_id"] == "w2", "the row follows the holder, not the first writer"
    assert len(ledger.db.execute("SELECT 1 FROM backend_operations").fetchall()) == 1


def test_a_folded_execution_counts_the_operations_it_is_responsible_for(ledger):
    folded = {**TASK, "_fold_members": [{"operation_id": "op-2", "items_count": 1},
                                        {"operation_id": "op-3", "items_count": 2}]}
    assert ledger.record(folded, worker_id="w1", bank_id=BANK,
                         resource=REMOTE)["folded"] == 2


def test_a_finished_operation_is_not_reopened_by_a_late_answer(ledger):
    ledger.record(TASK, worker_id="w1", bank_id=BANK, resource=REMOTE)
    ledger.mark("op-1", "finished", tokens=321)
    late = ledger.mark("op-1", "running")
    assert late["state"] == "finished"
    assert late["refused_transition"] == "finished -> running"
    assert ledger.get("op-1")["tokens"] == 321, "the charge is not doubled by the lie"


def test_an_unresolved_operation_is_the_one_a_restart_has_to_ask_about(ledger):
    ledger.record(TASK, worker_id="w1", bank_id=BANK, resource=REMOTE)
    ledger.mark("op-1", "running")
    assert [row["operation_id"] for row in ledger.unresolved()] == ["op-1"]
    assert ledger.report()["by_state"]["running"] == 1


def test_an_operation_that_cannot_be_placed_on_a_resource_is_counted_as_a_defect(ledger):
    ledger.record(TASK, worker_id="w1", bank_id=BANK, resource=None)
    assert ledger.unattributed() == 1
    assert ledger.report()["unattributed"] == 1


def test_marking_an_operation_that_was_never_recorded_is_refused(ledger):
    with pytest.raises(LauncherRefused, match="no record"):
        ledger.mark("op-nothing", "finished")
    with pytest.raises(LauncherRefused, match="unknown operation state"):
        ledger.mark("op-1", "mislaid")


# -- the wrapped executor ----------------------------------------------------

def test_the_record_is_written_before_the_engine_is_asked_for_anything(ledger):
    seen = []

    class Watching(Engine):
        async def execute_task(self, task):
            seen.append(dict(ledger.get("op-1")))
            await super().execute_task(task)

    engine = Watching(**memory_arguments(MODULES))
    run(attribute_tasks(engine, ledger, worker_id="w1", routes=ROUTES)(TASK))
    assert seen and seen[0]["state"] == "running", "the row exists before the call is made"
    assert ledger.get("op-1")["state"] == "finished"
    assert engine.executed == [TASK]


def test_a_lost_connection_leaves_the_operation_uncertain_not_failed(ledger):
    class Dying(Engine):
        async def execute_task(self, task):
            raise ConnectionResetError("the socket went away mid-request")

    engine = Dying(**memory_arguments(MODULES))
    with pytest.raises(ConnectionResetError):
        run(attribute_tasks(engine, ledger, worker_id="w1", routes=ROUTES)(TASK))
    row = ledger.get("op-1")
    assert row["state"] == "uncertain" and "socket" in row["error"]
    assert engine.executed == [], "the wrapper does not swallow the failure from the poller"


def test_a_cancelled_operation_is_never_started(ledger):
    engine = Engine(**memory_arguments(MODULES))
    ledger.record(TASK, worker_id="w1", bank_id=BANK, resource=REMOTE)
    ledger.request_cancellation("op-1", actor="jugaadu")
    with pytest.raises(LauncherRefused, match="was not started"):
        run(attribute_tasks(engine, ledger, worker_id="w1", routes=ROUTES)(TASK))
    assert engine.executed == []
    assert ledger.get("op-1")["state"] == "cancelled"


def test_a_cancellation_has_to_be_attributed_to_someone(ledger):
    ledger.record(TASK, worker_id="w1", bank_id=BANK, resource=REMOTE)
    with pytest.raises(LauncherRefused, match="attributed"):
        ledger.request_cancellation("op-1", actor="  ")


def test_the_owed_count_and_the_owed_list_answer_the_same_question(ledger):
    """One predicate, two readings. If they drift, the report has started lying.

    The count is what makes a status stage degraded and the list is what an operator works
    from, so they must be the same set of rows — not two SQL clauses that happen to match
    today.
    """
    for index in ("op-1", "op-2"):
        ledger.record({**TASK, OPERATION_KEY: index}, worker_id="w1", bank_id=BANK,
                      resource=REMOTE)
    ledger.request_cancellation("op-1", actor="jugaadu")
    ledger.request_cancellation("op-2", actor="jugaadu")
    ledger.confirm_cancellation("op-2")
    listed = ledger.cancellation_owed()
    assert [row["operation_id"] for row in listed] == ["op-1"]
    assert ledger.report()["cancellations_owed"] == len(listed)


def test_a_settled_operation_never_accrues_an_intent_nothing_can_answer(ledger):
    """``refused`` is as terminal as ``finished``: an owed cancellation must stay answerable."""
    for index, state in (("op-1", "finished"), ("op-2", "cancelled"), ("op-3", "refused")):
        ledger.record({**TASK, OPERATION_KEY: index}, worker_id="w1", bank_id=BANK,
                      resource=REMOTE)
        ledger.mark(index, state)
        assert ledger.request_cancellation(index, actor="jugaadu")["cancellation"] == "none"
        assert ledger.get(index)["cancellation"] == "none"
    assert ledger.cancellation_owed() == []
    assert ledger.report()["cancellations_owed"] == 0


def test_an_operation_that_matches_no_route_is_refused_rather_than_charged_to_another(
        ledger):
    engine = Engine(**memory_arguments(MODULES))
    task = {**TASK, "operation_type": "consolidate"}
    with pytest.raises(LauncherRefused, match="matches no configured route"):
        run(attribute_tasks(engine, ledger, worker_id="w1", routes=ROUTES)(task))
    assert engine.executed == [], "an unchargeable request is not somebody else's budget"
    row = ledger.get("op-1")
    assert row["resource"] is None and row["state"] == "uncertain"


def test_a_task_that_names_no_bank_is_refused_before_it_is_recorded(ledger):
    engine = Engine(**memory_arguments(MODULES))
    with pytest.raises(LauncherRefused, match="names no bank"):
        run(attribute_tasks(engine, ledger, worker_id="w1", routes=ROUTES)(
            {k: v for k, v in TASK.items() if k != "bank_id"}))
    assert ledger.report()["by_state"] == {}


def test_a_claim_without_an_operation_identity_cannot_be_accounted_for(ledger):
    with pytest.raises(LauncherRefused, match="no operation id"):
        ledger.record({"operation_type": "retain", "bank_id": BANK}, worker_id="w1",
                      bank_id=BANK, resource=REMOTE)


def test_a_payload_that_is_not_a_mapping_is_refused_rather_than_executed(ledger):
    engine = Engine(**memory_arguments(MODULES))
    with pytest.raises(LauncherRefused, match="must be a mapping"):
        run(attribute_tasks(engine, ledger, worker_id="w1", routes=ROUTES)(["not", "a",
                                                                           "task"]))
    assert engine.executed == []


# -- the composition ---------------------------------------------------------

def test_the_engine_is_constructed_the_way_the_pinned_worker_builds_it():
    arguments = memory_arguments(MODULES)
    assert arguments["run_migrations"] is False, "a worker that migrates races the API"
    assert isinstance(arguments["task_backend"], TaskBackend)


def test_the_poller_is_handed_our_executor_and_the_engines_own_hooks():
    built = compose(modules=MODULES, config=Config(), worker_id="w1", ledger=a_ledger(),
                    routes=ROUTES)
    arguments = built["poller_arguments"]
    assert arguments["max_slots"] == 1 and arguments["worker_id"] == "w1"
    assert arguments["backend"] is built["memory"]._backend
    assert arguments["on_wall_timeout"] == built["memory"].on_task_wall_timeout
    assert callable(arguments["executor"])


def test_the_default_schema_is_passed_as_no_prefix_rather_than_as_a_name():
    built = compose(modules=MODULES, config=Config(), worker_id="w1", ledger=a_ledger(),
                    routes=ROUTES)
    assert built["poller_arguments"]["schema"] is None
    named = compose(modules=MODULES, config=Config(database_schema="tenant_7"),
                    worker_id="w1", ledger=a_ledger(), routes=ROUTES)
    assert named["poller_arguments"]["schema"] == "tenant_7"


def test_a_signature_that_no_longer_matches_is_a_refusal_not_a_guess():
    class DriftedPoller(Poller):
        """``slot_reservations`` gone from the signature: the call would raise mid-launch."""

        def __init__(self, *, backend, worker_id, executor, poll_interval_ms, schema,
                     tenant_extension, max_slots, consolidation_bank_priority,
                     max_retries, on_wall_timeout):
            super().__init__(backend=backend, worker_id=worker_id, executor=executor,
                             poll_interval_ms=poll_interval_ms, schema=schema,
                             tenant_extension=tenant_extension, max_slots=max_slots,
                             slot_reservations={},
                             consolidation_bank_priority=consolidation_bank_priority,
                             max_retries=max_retries, on_wall_timeout=on_wall_timeout)

    with pytest.raises(LauncherRefused, match="no longer accepts slot_reservations"):
        compose(modules={**MODULES, "WorkerPoller": DriftedPoller}, config=Config(),
                worker_id="w1", ledger=a_ledger(), routes=ROUTES)


def test_a_signature_that_cannot_be_inspected_is_not_claimed_as_verified():
    """``**kwargs`` gives nothing to compare, so the check stays quiet rather than lying.

    The strict case is the one that matters and is covered above; this one records what the
    adapter is *not* promised, which is what the version contract is for.
    """
    from hermes_memory.backend.worker_launcher import _accepts

    class TakesAnything:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    _accepts(TakesAnything, {"run_migrations": False, "something_new": True},
             what="MemoryEngine")
    assert TakesAnything(**{"run_migrations": False, "something_new": True}).kwargs == {
        "run_migrations": False, "something_new": True}


# -- the process -------------------------------------------------------------

def test_an_environment_without_the_backend_reports_rather_than_tracebacks(tmp_path,
                                                                           monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path))
    verdict = check(load_settings(), modules=None)
    assert verdict["ok"] is False
    assert any("not importable" in line for line in verdict["blocking"])
    assert verdict["launcher_version"] == LAUNCHER_VERSION
    assert "not_performed" in verdict


def test_a_backend_of_another_version_is_named_and_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path))
    modules = {"hindsight_api": type("M", (), {"__version__": "0.9.4"})}
    verdict = check(load_settings(), modules=modules)
    assert verdict["ok"] is False
    assert "0.9.4" in verdict["blocking"][0] and "0.10.1" in verdict["blocking"][0]


def test_check_prints_no_connection_string_even_though_the_config_holds_one(capsys,
                                                                           tmp_path,
                                                                           monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path))
    assert main(["--check"]) == 2
    printed = capsys.readouterr().out
    assert "postgresql" not in printed and "s3cr3t" not in printed
    assert json.loads(printed)["ok"] is False


def test_launching_refuses_before_touching_a_model_here(capsys, tmp_path, monkeypatch):
    """This environment has no Hindsight, and that is an answer rather than a crash."""
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path))
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "refused" in captured.err and captured.out == ""


def test_the_unit_verb_runs_the_module_the_unit_names(tmp_path):
    """``ExecStart`` is ``python -m hermes_memory.backend.worker_launcher``, in a process.

    Every other test here calls ``main()`` directly, which cannot see the difference between
    a module with an entry point and one without: the latter imports, runs nothing and exits
    ``0``, and a unit with ``Restart=on-failure`` reads that as a service that stopped
    cleanly. The worker would be dead, the journal quiet, and the archive unprojected.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "hermes_memory.backend.worker_launcher", "--check"],
        env={**os.environ, "HERMES_MEMORY_HOME": str(tmp_path)},
        capture_output=True, text=True)
    verdict = json.loads(completed.stdout)
    assert verdict["launcher_version"] == LAUNCHER_VERSION
    assert completed.returncode == (0 if verdict["ok"] else 2)
    assert "postgresql://" not in completed.stdout + completed.stderr


def test_a_present_backend_missing_a_pinned_symbol_is_refused_by_name(monkeypatch):
    """Half a distribution is worse than none: the composition would fail mid-launch."""
    from hermes_memory.backend import worker_launcher

    modules = {"hindsight_api": type("M", (), {"MemoryEngine": Engine})(),
               "hindsight_api.config": type("C", (), {"get_config": None})(),
               "hindsight_api.engine.task_backend": type("T", (), {})(),
               "hindsight_api.worker.poller": type("P", (), {})()}

    def fake_import(name):
        return modules[name]

    monkeypatch.setattr("importlib.import_module", fake_import)
    with pytest.raises(LauncherRefused, match="hindsight_api is missing __version__"):
        worker_launcher._import_backend()


def a_worker_installation(tmp_path, monkeypatch, *, engine=Engine, poller=Poller):
    """An owned installation whose backend is a fake with the pinned keywords.

    The env file is part of it because a worker that cannot place an operation on a route
    refuses that operation, so an installation under test has to name its routes the way an
    owned env file does.
    """
    from hermes_memory.backend import worker_launcher

    (tmp_path / "data").mkdir()
    (tmp_path / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={tmp_path / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS=200000\n"
        "HERMES_MEMORY_HINDSIGHT_URL=http://127.0.0.1:8123\n"
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS=127.0.0.1\n"
        f"HERMES_MEMORY_TEXT_BASE_URL={PINS['retain'].upstream}\n"
        "HERMES_MEMORY_EMBEDDINGS_BASE_URL=http://127.0.0.1:11435/v1\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_RETAIN=cred\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_EMBEDDINGS=cred-embed\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_CONSOLIDATE=cred-c\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_REFLECT=cred-r\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_FOREGROUND=cred-f\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path))
    config = Config()
    modules = {
        "hindsight_api": type("M", (), {"__version__": "0.10.1", "MemoryEngine": engine})(),
        "hindsight_api.config": type("C", (), {
            "get_config": staticmethod(lambda: config),
            "DEFAULT_DATABASE_SCHEMA": "public",
            "load_dotenv_for_entrypoint": staticmethod(lambda: None)})(),
        "hindsight_api.engine.task_backend": type("T", (),
                                                  {"WorkerTaskBackend": TaskBackend})(),
        "hindsight_api.worker.poller": type("P", (), {"WorkerPoller": poller})(),
    }
    monkeypatch.setattr(worker_launcher, "_import_backend", lambda: modules)
    return config


def an_engine(cls=Engine):
    """One fake ``MemoryEngine``, built the way the pinned launcher builds it."""
    return cls(run_migrations=False, task_backend=None, tenant_extension=None,
               operation_validator=None)


def a_build(engines, *, first=None):
    """A ``bring_up`` factory that records every engine it makes.

    ``first`` is the class used for the first attempt only, so a test can have one startup
    fail and the next behave.
    """
    def build():
        engine = an_engine(first if (first and not engines) else Engine)
        engines.append(engine)
        return {"memory": engine, "poller_factory": Poller, "poller_arguments": {}}
    return build


def test_the_real_launch_path_composes_the_pinned_construction_end_to_end(tmp_path,
                                                                         monkeypatch):
    """``main([])`` with a backend present: engine built, poller started, our wrapper in it.

    This is the only place the whole path is exercised, and it is the path the worker unit
    runs, so the assertion that matters is that the executor the poller was handed is the
    one that writes the ledger before the engine is asked for anything.
    """
    built = {"engine": None, "poller": None}

    class RecordingEngine(Engine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            built["engine"] = self

    class RecordingPoller(Poller):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            built["poller"] = self

        async def run(self):
            # What the real poller does with a claimed task: hand it to the executor.
            self.ran = True
            await self.arguments["executor"](TASK)

    a_worker_installation(tmp_path, monkeypatch, engine=RecordingEngine,
                          poller=RecordingPoller)
    assert main([]) == 0
    assert built["poller"].arguments["max_slots"] == 1
    assert built["poller"].ran and built["engine"].initialized
    assert built["engine"].closed is True
    assert built["engine"].executed == [TASK], "the wrapper passed the task through"
    with GateStore(gate_path(load_settings())) as store:
        assert OperationLedger(store).get("op-1")["state"] == "finished"


def test_a_backend_without_a_worker_poller_refuses_rather_than_running_inline(
        tmp_path, monkeypatch, capsys):
    """The pinned main exits here; running operations in the API process is another thing."""
    class Standalone(Backend):
        supports_worker_poller = False

    class EngineWithoutPoller(Engine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._backend = Standalone()

    a_worker_installation(tmp_path, monkeypatch, engine=EngineWithoutPoller)
    assert main([]) == 2
    printed = json.loads(capsys.readouterr().err)
    assert "no worker poller" in printed["refused"]


# -- the operator's hold ------------------------------------------------------

def a_hold(store, *, state="paused", actor="jugaadu"):
    """The owner's decision, written where every process that spends a model reads it."""
    store.set_control("global", "inference", state, actor=actor,
                      reason="the owner stopped the models", policy_version="gate-v1")


def a_ticker(store, slept, *, lift_after=None):
    """A sleep that does not sleep, optionally lifting the hold on the n-th tick."""
    async def sleep(seconds):
        slept.append(seconds)
        if lift_after is not None and len(slept) >= lift_after:
            a_hold(store, state="active")
    return sleep


def test_the_hold_is_read_from_the_row_the_owner_wrote(ledger):
    assert inference_hold(ledger.store) is None
    a_hold(ledger.store)
    held = inference_hold(ledger.store)
    assert held["state"] == "paused" and held["actor"] == "jugaadu"
    a_hold(ledger.store, state="active")
    assert inference_hold(ledger.store) is None


def test_a_held_installation_makes_the_worker_wait_rather_than_exit(ledger):
    """§10.5: a hold survives a restart on purpose, so the worker has to survive waiting.

    The alternative is the failure this launcher used to have: ``hermes-memory stop`` holds
    inference, ``start`` does not lift it, the worker's startup probe gets a 503, the unit
    exits, and its restart limit eventually silences a process nobody will start again —
    leaving operations pending after the hold lifted, with nothing left to drain them.
    """
    a_hold(ledger.store)
    slept, engines = [], []
    built = run(bring_up(a_build(engines), ledger.store,
                         sleep=a_ticker(ledger.store, slept, lift_after=2)))
    assert slept == [HOLD_POLL_SECONDS, HOLD_POLL_SECONDS]
    assert len(engines) == 1, "one hold is one wait, not one build per poll"
    assert built["memory"] is engines[0] and engines[0].initialized is True


def test_nothing_is_composed_while_the_owner_holds_inference(ledger):
    """The wait comes before the build, and that order is the whole point.

    ``compose`` hands the backend a task backend and builds a ``MemoryEngine``, which at the
    pinned tag is a database connection pool. Asking for either in a machine whose owner
    wrote "do not dispatch" is what this launcher exists to avoid — and a test that only
    counted how many engines were made would pass with the two statements swapped.
    """
    a_hold(ledger.store)
    composed_under_hold: list[bool] = []

    def build():
        composed_under_hold.append(inference_hold(ledger.store) is not None)
        return {"memory": an_engine(), "poller_factory": Poller, "poller_arguments": {}}

    run(bring_up(build, ledger.store, sleep=a_ticker(ledger.store, [], lift_after=1)))
    assert composed_under_hold == [False], \
        "the engine was composed only after the hold lifted, never while it stood"


def test_the_wait_is_announced_with_who_is_holding_it(ledger, capsys):
    """A worker that goes quiet for an hour looks dead; the reason has to be in the journal."""
    a_hold(ledger.store)
    slept = []
    waited = run(wait_for_inference(ledger.store, sleep=a_ticker(ledger.store, slept,
                                                                 lift_after=1)))
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert waited == HOLD_POLL_SECONDS
    assert lines[0]["worker"] == "waiting" and lines[0]["actor"] == "jugaadu"
    assert lines[0]["since"] and lines[-1]["worker"] == "resumed"


def test_an_attempt_that_races_a_hold_is_torn_down_and_retried_with_a_fresh_engine(ledger):
    """The hold may arrive mid-startup; the half-built engine is shut down, not reused.

    Asking one engine to ``initialize()`` twice is not something this launcher will assume
    is safe, so the attempt is discarded and the composition is performed again.
    """
    class HeldDuringStartup(Engine):
        async def initialize(self):
            a_hold(ledger.store)  # the owner stops the models during the startup probe
            raise RuntimeError("503: all inference is paused by the operator")

    engines = []
    slept = []
    built = run(bring_up(a_build(engines, first=HeldDuringStartup), ledger.store,
                         sleep=a_ticker(ledger.store, slept, lift_after=1)))
    assert len(engines) == 2
    assert engines[0].closed is True and engines[0].initialized is False
    assert built["memory"] is engines[1] and engines[1].initialized is True


def test_a_startup_failure_that_is_not_the_hold_still_exits(ledger):
    """A real misconfiguration stays a real failure: retried forever, it would look healthy."""
    a_hold(ledger.store, state="active")
    engines = []

    class Broken(Engine):
        async def initialize(self):
            raise RuntimeError("no such table: banks")

    slept = []
    with pytest.raises(RuntimeError, match="no such table"):
        run(bring_up(a_build(engines, first=Broken), ledger.store,
                     sleep=a_ticker(ledger.store, slept)))
    assert slept == [], "a broken installation is not waited out; waiting is for a race"
    assert len(engines) == 1
    assert engines[0].closed is False, "nothing was torn down behind a retry loop"


# -- the engine's own database ------------------------------------------------
# The database on the engine's port is the backend's child, and a unit order says only that
# the backend was *started* first. Restarting all three units together put this exact refusal
# in the journal three times before systemd's own restart caught up with the database, and a
# worker that exits over a race it was never going to win is a machine that looks broken
# every time it is booted.

REFUSED = OSError("Multiple exceptions: [Errno 111] Connect call failed ('::1', 5432, 0, 0), "
                  "[Errno 111] Connect call failed ('127.0.0.1', 5432)")


def an_always_build(engines, cls):
    """Like ``a_build``, except every attempt gets the same class, for a fault that persists."""
    def build():
        engine = an_engine(cls)
        engines.append(engine)
        return {"memory": engine, "poller_factory": Poller, "poller_arguments": {}}
    return build


def test_a_refusal_to_connect_is_told_apart_from_a_broken_installation():
    assert database_not_listening(REFUSED)
    assert database_not_listening(ConnectionRefusedError(111, "Connection refused"))
    assert database_not_listening(RuntimeError("connection failed: the database system is "
                                               "starting up"))
    assert not database_not_listening(RuntimeError("no such table: banks"))
    assert not database_not_listening(RuntimeError("password authentication failed for "
                                                   "user \"hindsight\"")), (
        "a wrong secret is not something to wait out")


def test_a_worker_that_starts_before_its_database_waits_for_it(ledger, capsys):
    engines = []
    ticks = 0

    class NotUpYet(Engine):
        async def initialize(self):
            nonlocal ticks
            ticks += 1
            if ticks <= 2:
                raise REFUSED
            self.initialized = True

    def build():
        engine = an_engine(NotUpYet)
        engines.append(engine)
        return {"memory": engine, "poller_factory": Poller, "poller_arguments": {}}

    slept = []
    built = run(bring_up(build, ledger.store, sleep=a_ticker(ledger.store, slept),
                         db_wait_s=20.0, db_poll_s=5.0))
    assert slept == [5.0, 5.0], "the wait is the launcher's own, not systemd's restart timer"
    assert len(engines) == 3, "a fresh engine per attempt, as with a hold"
    assert [engine.closed for engine in engines[:2]] == [True, True]
    assert built["memory"] is engines[2] and engines[2].initialized is True
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    waits = [line for line in lines
             if line["worker"] == "database not accepting connections yet"]
    assert len(waits) == 2, "each failed attempt is said, not sat on"
    assert "Connect call failed" in waits[0]["error"]


def test_a_database_that_never_answers_is_still_a_failure(ledger):
    """The wait has an end, because a host that never answers is a broken installation."""
    engines = []

    class NeverThere(Engine):
        async def initialize(self):
            raise ConnectionRefusedError(111, "Connection refused")

    slept = []
    with pytest.raises(ConnectionRefusedError):
        run(bring_up(an_always_build(engines, NeverThere), ledger.store,
                     sleep=a_ticker(ledger.store, slept), db_wait_s=10.0, db_poll_s=5.0))
    assert slept == [5.0, 5.0], "two waits at this ceiling, and then the refusal is the answer"
    assert len(engines) == 3
    assert engines[-1].closed is False, "the last attempt is not torn down twice"


def test_an_unclean_teardown_of_a_half_started_engine_is_said_not_swallowed(ledger, capsys):
    """The hold is the reason to wait, but a shutdown that also failed is a second fact."""
    class HalfBuilt(Engine):
        async def initialize(self):
            a_hold(ledger.store)
            raise RuntimeError("503: paused")

        async def close(self):
            raise RuntimeError("connection already closed")

    engines = []
    run(bring_up(a_build(engines, first=HalfBuilt), ledger.store,
                 sleep=a_ticker(ledger.store, [], lift_after=1)))
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    torn = [line for line in lines if line.get("worker") == "held during startup"]
    assert len(torn) == 1
    assert "paused" in torn[0]["error"]
    assert "connection already closed" in torn[0]["shutdown_error"]


def test_the_unit_waits_out_a_hold_and_drains_when_the_owner_resumes(tmp_path, monkeypatch,
                                                                    capsys):
    """The whole unit path, because this is what ``Restart=on-failure`` used to end.

    The hold is in the installation's own admission ledger, where ``hermes-memory pause
    --scope inference`` put it and where a restart cannot lose it.
    """
    real_sleep = asyncio.sleep
    store = GateStore(tmp_path / "gate.db")
    a_hold(store)
    store.close()

    slept = []

    async def waiting_sleep(seconds):
        slept.append(seconds)
        if len(slept) >= 2:
            lifted = GateStore(tmp_path / "gate.db")
            a_hold(lifted, state="active")
            lifted.close()
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", waiting_sleep)
    a_worker_installation(tmp_path, monkeypatch)
    assert main([]) == 0
    assert len(slept) >= 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()
             if line.startswith("{")]
    assert [line["worker"] for line in lines] == ["waiting", "resumed"]
    assert lines[0]["actor"] == "jugaadu"


# -- the schema it lives in --------------------------------------------------

def test_an_older_ledger_gains_the_operation_table_without_losing_reservations(tmp_path):
    path = tmp_path / "gate.db"
    store = GateStore(path)
    gate = ResourceGate(store)
    assert gate.try_acquire(route="retain", holder="generation-1", resource=REMOTE,
                            priority=1) is not None
    # Stand in for a ledger written by the previous build: the new table is absent and
    # the file says so.
    store.db.execute("DROP TABLE backend_operations")  # its indexes go with it
    store.db.execute("PRAGMA user_version=1")
    store.close()

    reopened = GateStore(path)
    assert [row["holder"] for row in reopened.db.execute(
        "SELECT holder FROM gate_reservations")] == ["generation-1"], \
        "an upgrade that dropped a live reservation would free a busy device"
    assert reopened.db.execute("SELECT count(*) FROM backend_operations").fetchone()[0] == 0
    assert int(reopened.db.execute("PRAGMA user_version").fetchone()[0]) == GATE_SCHEMA_VERSION


def test_an_upgrade_torn_halfway_finishes_rather_than_failing_on_itself(tmp_path):
    """Objects already made are skipped; a second attempt must not jam the admission gate."""
    path = tmp_path / "gate.db"
    store = GateStore(path)
    store.db.execute("PRAGMA user_version=1")  # the table is there, the version says not
    store.close()

    reopened = GateStore(path)
    assert reopened.db.execute("SELECT count(*) FROM backend_operations").fetchone()[0] == 0
    assert int(reopened.db.execute("PRAGMA user_version").fetchone()[0]) == GATE_SCHEMA_VERSION
    reopened.close()


def test_a_ledger_from_a_newer_build_is_still_read_only_and_refused(tmp_path):
    path = tmp_path / "gate.db"
    GateStore(path).close()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version=99")
    connection.commit()
    connection.close()
    from hermes_memory.processing.instance_gate import GateError

    with pytest.raises(GateError, match="newer admission ledger"):
        GateStore(path)


def test_a_fresh_ledger_carries_both_tables(tmp_path):
    store = GateStore(tmp_path / "gate.db")
    tables = {row[0] for row in store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"gate_reservations", "budget_usage", "runtime_controls",
            "backend_operations"} <= tables
    store.close()


def test_a_ledger_written_before_the_table_exists_is_reported_as_absent(tmp_path):
    """A reading cannot upgrade the file it is reading, and must not crash for it.

    ``present`` is a property, not a method: a truthy bound method would make every
    ledger look like it had the table and hand the next query back as an OperationalError
    inside a status report.
    """
    store = GateStore(tmp_path / "gate.db")
    store.db.execute("DROP TABLE backend_operations")
    ledger = OperationLedger(store)
    assert ledger.present is False
    assert ledger.report() == {"ledger": "absent", "by_state": {}, "unattributed": 0,
                               "unresolved": 0, "cancellations_owed": 0}
    store.close()
    reopened = GateStore(tmp_path / "gate.db")
    assert OperationLedger(reopened).present is False, \
        "a reading that repaired the ledger would have written while promising not to"
    reopened.close()
    assert OperationLedger(GateStore(tmp_path / "second.db")).present is True
