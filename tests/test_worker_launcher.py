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
import sqlite3

import pytest

from hermes_memory.backend.worker_launcher import (LAUNCHER_VERSION, OPERATION_KEY,
                                                   RETRY_KEY, LauncherRefused,
                                                   OperationLedger, attribute_tasks, check,
                                                   compose, main, memory_arguments,
                                                   poller_arguments, slot_contract,
                                                   version_contract)
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
    """``MemoryEngine`` at the tag: these keywords, and nothing else, are what we pass."""

    def __init__(self, *, run_migrations, task_backend, tenant_extension,
                 operation_validator):
        self.run_migrations = run_migrations
        self.task_backend = task_backend
        self.tenant_extension = tenant_extension
        self.operation_validator = operation_validator
        self._backend = Backend()
        self.executed = []
        self.initialized = False
        self.shutdown_called = False

    async def execute_task(self, task):
        self.executed.append(task)

    async def initialize(self):
        self.initialized = True

    async def on_task_wall_timeout(self, task):
        return "timed out"

    async def shutdown(self):
        self.shutdown_called = True


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


def test_the_real_launch_path_composes_the_pinned_construction_end_to_end(tmp_path,
                                                                         monkeypatch):
    """``main([])`` with a backend present: engine built, poller started, our wrapper in it.

    This is the only place the whole path is exercised, and it is the path the worker unit
    runs, so the assertion that matters is that the executor the poller was handed is the
    one that writes the ledger before the engine is asked for anything.
    """
    from hermes_memory.backend import worker_launcher

    (tmp_path / "data").mkdir()
    # A worker that cannot place an operation on a route refuses it, so this installation
    # has to name the routes the way an owned env file would.
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

    config = Config()
    modules = {
        "hindsight_api": type("M", (), {"__version__": "0.10.1",
                                        "MemoryEngine": RecordingEngine})(),
        "hindsight_api.config": type("C", (), {
            "get_config": staticmethod(lambda: config),
            "DEFAULT_DATABASE_SCHEMA": "public",
            "load_dotenv_for_entrypoint": staticmethod(lambda: None)})(),
        "hindsight_api.engine.task_backend": type("T", (),
                                                  {"WorkerTaskBackend": TaskBackend})(),
        "hindsight_api.worker.poller": type("P", (), {"WorkerPoller": RecordingPoller})(),
    }
    monkeypatch.setattr(worker_launcher, "_import_backend", lambda: modules)
    assert main([]) == 0
    assert built["poller"].arguments["max_slots"] == 1
    assert built["poller"].ran and built["engine"].initialized
    assert built["engine"].shutdown_called is True
    assert built["engine"].executed == [TASK], "the wrapper passed the task through"
    with GateStore(gate_path(load_settings())) as store:
        assert OperationLedger(store).get("op-1")["state"] == "finished"


def test_a_backend_without_a_worker_poller_refuses_rather_than_running_inline(
        tmp_path, monkeypatch, capsys):
    """The pinned main exits here; running operations in the API process is another thing."""
    from hermes_memory.backend import worker_launcher

    (tmp_path / "data").mkdir()
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path))

    class Standalone(Backend):
        supports_worker_poller = False

    class EngineWithoutPoller(Engine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._backend = Standalone()

    config = Config()
    modules = {
        "hindsight_api": type("M", (), {"__version__": "0.10.1",
                                        "MemoryEngine": EngineWithoutPoller})(),
        "hindsight_api.config": type("C", (), {
            "get_config": staticmethod(lambda: config),
            "DEFAULT_DATABASE_SCHEMA": "public"})(),
        "hindsight_api.engine.task_backend": type("T", (),
                                                  {"WorkerTaskBackend": TaskBackend})(),
        "hindsight_api.worker.poller": type("P", (), {"WorkerPoller": Poller})(),
    }
    monkeypatch.setattr(worker_launcher, "_import_backend", lambda: modules)
    assert main([]) == 2
    printed = json.loads(capsys.readouterr().err)
    assert "no worker poller" in printed["refused"]


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
                               "unresolved": 0}
    store.close()
    reopened = GateStore(tmp_path / "gate.db")
    assert OperationLedger(reopened).present is False, \
        "a reading that repaired the ledger would have written while promising not to"
    reopened.close()
    assert OperationLedger(GateStore(tmp_path / "second.db")).present is True
