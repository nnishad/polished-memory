"""§8.1.2: the pinned backend worker launcher, and the accounting wrapped around it.

Hindsight's own worker is the process that spends a model on consolidation, refresh and
async retain. It has a native semaphore per process and no view of the GPU next door, so
this launcher composes it the way ``hindsight_api/worker/main.py`` does at the pinned tag
and wraps the executor with one job of our own: write down which operation is about to
run, for whose bank, under whose budget, and whether the owner has asked for it not to
start — *before* the call goes out. A retry then charges to the same operation instead of
looking like new work, and a lost outcome stays recorded as uncertain rather than being
silently replayed.

Nothing here patches an installed source file. Every upstream symbol arrives through
``modules``, so the composition is testable against fakes, and the price of that honesty
is stated plainly: the fake proves *our* shape, and only the version contract below proves
that the real thing still has it. An unverified revision refuses to start rather than
guessing at a construction that would fail mid-request.

An operator's hold on inference is waited out rather than treated as a startup failure.
§10.5 keeps a pause durable across a restart on purpose, so a worker that exited whenever
the models were stopped would be restarted into the same exit until its start limit
silenced the unit — and the operations it exists to drain would stay pending after the
hold lifted, with nothing left to run them.

The resolved database DSN is never printed. It carries credentials, and a worker that logs
its own connection string leaks the whole machine's memory into a journal.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from typing import Any, Awaitable, Callable, Mapping

from ..config import SettingError, load_settings
from ..ids import now
from ..processing.instance_gate import GateStore, gate_path
from ..processing.routes import build_routes
from .capabilities import PINNED_VERSION

__all__ = ["LAUNCHER_VERSION", "REQUIRED_IMPORTS", "HOLD_POLL_SECONDS", "DB_WAIT_SECONDS",
           "DB_POLL_SECONDS", "LauncherRefused",
           "version_contract", "slot_contract", "inference_hold", "wait_for_inference",
           "database_not_listening",
           "bring_up", "OperationLedger", "attribute_tasks", "compose", "check", "main"]

LAUNCHER_VERSION = "worker-launcher-v1"

# The symbols this construction depends on, by import path. Renames upstream are a
# different release, not a detail to paper over.
REQUIRED_IMPORTS: Mapping[str, tuple[str, ...]] = {
    "hindsight_api": ("MemoryEngine", "__version__"),
    "hindsight_api.config": ("get_config", "load_dotenv_for_entrypoint",
                             "DEFAULT_DATABASE_SCHEMA"),
    "hindsight_api.engine.task_backend": ("WorkerTaskBackend",),
    "hindsight_api.worker.poller": ("WorkerPoller",),
}

# What the pinned poller injects into the task payload before it calls the executor
# (``worker/poller.py`` at the tag): the operation it claimed, how many times it has
# tried, and the retain peers folded into this one execution.
OPERATION_KEY = "_operation_id"
RETRY_KEY = "_retry_count"
FOLD_KEY = "_fold_members"
BANK_KEYS = ("bank_id", "_bank_id")
TYPE_KEYS = ("operation_type", "type")

# Database operation names and executor payload discriminators are distinct at
# the pinned revision. These policies attribute work; they do not grant inference
# or turn on automatic consolidation. Each model hop still uses the owned gate.
TASK_POLICIES = {
    "retain": ("retain", frozenset({"batch_retain", "retain"})),
    "consolidation": ("consolidate", frozenset({"consolidation"})),
    "refresh_mental_model": ("reflect", frozenset({"refresh_mental_model"})),
    "graph_maintenance": (None, frozenset({"graph_maintenance"})),
    "vector_index_maintenance": (None, frozenset({"vector_index_maintenance"})),
}
DB_TASK_TIMEOUT_S = 300.0  # native graph passes have a 240-second budget

TERMINAL = frozenset({"finished", "cancelled", "refused"})

# One predicate, so the count a report gives and the list an operator works from cannot
# quietly become two different questions. A cancellation still in this state was asked for
# and never answered.
CANCELLATION_OWED = "cancellation='requested'"

# How often a worker that is waiting out an operator hold looks again. It spends nothing
# while it waits, so the only cost of asking often is a wakeup.
HOLD_POLL_SECONDS = 15.0

# The engine's database belongs to the backend process, and a unit order says only that the
# backend was started before this one, not that its database is accepting connections yet.
# Restarting into the same refusal costs a start limit, so the wait happens here.
DB_WAIT_SECONDS = 60.0
DB_POLL_SECONDS = 5.0

# What "not listening yet" looks like, from the shapes the pinned stack really raises:
# asyncio's multi-address connect, a plain refusal, and a database that answers on the port
# while it is still coming up. Anything else propagates — a wrong host or a missing table is
# not something to retry until it looks healthy.
_NOT_LISTENING = ("connect call failed", "connection refused", "connection failed",
                  "network is unreachable", "no route to host",
                  "the database system is starting up")

# The hold is one row in the admission ledger, for the whole machine.
INFERENCE_STAGE = ("global", "inference")


class LauncherRefused(Exception):
    """The composition this launcher would perform is not safe to perform."""


def _announce(payload: dict[str, Any]) -> None:
    """One journal line about what this process is doing and why.

    A worker that goes quiet for an hour looks dead to whoever runs ``status``, and the
    honest answer — "I am waiting because you told me to" — has to be in the journal or it
    is not an answer at all.
    """
    print(json.dumps(payload, sort_keys=True), flush=True)


# -- the operator's hold ------------------------------------------------------

def inference_hold(store) -> dict[str, Any] | None:
    """What the owner wrote down about this machine's models, or None when nothing is held.

    Read from the admission ledger rather than asked across the gate's HTTP port: the
    worker already has that ledger open, and whether the owner said "stop" is answered by
    the row the owner wrote, not by a dispatch that would itself be refused. One query
    decides it, because a reading that asked two questions of the same row would have to
    decide what to do when they disagreed.
    """
    control = store.control(*INFERENCE_STAGE)
    if not control or control.get("state") != "paused":
        return None
    return control


async def wait_for_inference(store, *, sleep: Callable[[float], Any],
                             poll_s: float = HOLD_POLL_SECONDS,
                             announce: Callable[[dict], Any] = _announce) -> float:
    """Sleep while inference is held, saying so once on each side of the wait.

    Returns the seconds waited. This is the difference between a worker the owner stopped
    and a worker that crashed: both leave operations pending, but only the second one
    leaves an exit status, and a manager that restarts on failure will eventually refuse to
    start the process again over a hold that is still in force.
    """
    held = inference_hold(store)
    if held is None:
        return 0.0
    announce({"worker": "waiting", "reason": "inference is paused by the operator",
              "actor": held.get("actor"), "since": held.get("changed_at"),
              "poll_seconds": poll_s})
    waited = 0.0
    while inference_hold(store) is not None:
        await sleep(poll_s)
        waited += poll_s
    announce({"worker": "resumed", "waited_seconds": round(waited, 1)})
    return waited


async def bring_up(build: Callable[[], Mapping[str, Any]], store, *,
                   sleep: Callable[[float], Any],
                   poll_s: float = HOLD_POLL_SECONDS,
                   db_wait_s: float = DB_WAIT_SECONDS,
                   db_poll_s: float = DB_POLL_SECONDS,
                   announce: Callable[[dict], Any] = _announce) -> dict[str, Any]:
    """Build and initialize the engine, but never against a hold and never fatally because of one.

    The engine's own startup probe asks for a single embedding, and while inference is held
    the gate answers 503. That is an operator's decision rather than a broken environment,
    so the wait happens first and an attempt that raced a hold arriving mid-startup is torn
    down and retried with a *fresh* engine — the half-initialized one has already opened
    whatever its failing task managed to open, and asking it to initialize twice is not a
    thing this launcher is willing to assume is safe.

    A database that is not listening yet is waited out the same bounded way. It is the
    backend's own child, and the unit order only says the backend was started first, so every
    cold start of all three units races it; exiting over that is a journal full of failures
    that mean nothing. The wait has an end, though: a host that never answers is a broken
    installation, and it is reported as one.

    Any other startup failure propagates untouched: a real misconfiguration still exits.
    """
    waiting_for_db = 0.0
    while True:
        await wait_for_inference(store, sleep=sleep, poll_s=poll_s, announce=announce)
        built = build()
        try:
            await built["memory"].initialize()
        except Exception as error:
            if inference_hold(store) is not None:
                await _discard(built["memory"], error=error, announce=announce)
                continue
            if waiting_for_db < db_wait_s and database_not_listening(error):
                await _discard(built["memory"], error=error, announce=announce,
                               worker="database not accepting connections yet")
                await sleep(db_poll_s)
                waiting_for_db += db_poll_s
                continue
            raise
        return dict(built)


async def _discard(memory: Any, *, error: BaseException,
                   announce: Callable[[dict], Any],
                   worker: str = "held during startup") -> None:
    """Tear down an engine a wait stopped half-started. Its shutdown gets no veto."""
    announcement = {"worker": worker, "attempt": 1,
                    "error": f"{type(error).__name__}: {str(error)[:200]}",
                    "next": "waiting, then building a fresh engine"}
    try:
        await memory.close()
    except Exception as shutdown_error:
        announcement["shutdown_error"] = (
            f"{type(shutdown_error).__name__}: {str(shutdown_error)[:200]}")
    announce(announcement)


def database_not_listening(error: BaseException) -> bool:
    """Is this the engine's database being up still, rather than the installation being wrong?

    Matched on what the pinned stack puts in the message rather than on the exception type:
    the same refusal arrives as ``ConnectionRefusedError`` from one address and a bare
    ``OSError`` listing several, and neither is distinguishable from a real fault by class.
    """
    message = f"{type(error).__name__}: {error}".lower()
    return any(marker in message for marker in _NOT_LISTENING)


# -- the two contracts -------------------------------------------------------

def version_contract(*, declared: Any, pinned: str = PINNED_VERSION) -> dict[str, Any]:
    """Refuse to run a backend whose version we did not compose against.

    A mismatch is not a warning: the whole value of a revision-specific adapter is that it
    was read at that revision. The observed value is reported so the owner knows which of
    the two is stale.
    """
    observed = str(declared or "").strip()
    return {"pinned": pinned, "declared": observed or "unknown",
            "matches": observed == pinned,
            "note": "the worker composition is read at one revision and refuses another"}


def slot_contract(config: Any) -> dict[str, Any]:
    """The one-slot rule, checked against the configuration the worker will actually use.

    Native reservations are floors, not ceilings: two reserved slots against one total slot
    does not halve the concurrency, it deadlocks retain behind consolidation. Batch retain
    execution bypasses the gate entirely, and an unset completion cap lets a provider
    answer at length nobody budgeted for.
    """
    observed = {
        "worker_max_slots": _field(config, "worker_max_slots"),
        "worker_slot_reservations": dict(_field(config, "worker_slot_reservations") or {}),
        "retain_batch_enabled": _field(config, "retain_batch_enabled"),
        "retain_max_completion_tokens": _field(config, "retain_max_completion_tokens"),
        "consolidation_max_completion_tokens": _field(config,
                                                      "consolidation_max_completion_tokens"),
        "reflect_max_completion_tokens": _field(config, "reflect_max_completion_tokens"),
    }
    violations = []
    if observed["worker_max_slots"] != 1:
        violations.append(f"worker_max_slots is {observed['worker_max_slots']!r}; one slot "
                          "per physical resource is the design")
    reserved = sum(int(value) for value in observed["worker_slot_reservations"].values())
    if reserved:
        violations.append(f"{reserved} slot(s) are natively reserved; a reservation is a "
                          "floor and would deadlock the single slot")
    if observed["retain_batch_enabled"]:
        violations.append("native batch retain is enabled; it executes outside the "
                          "admission gate")
    for key in ("retain_max_completion_tokens", "consolidation_max_completion_tokens",
                "reflect_max_completion_tokens"):
        value = observed[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            violations.append(f"{key} is {value!r}; every generation path needs an explicit "
                              "output cap")
    return {"ok": not violations, "observed": observed, "violations": violations}


# -- the ledger --------------------------------------------------------------

class OperationLedger:
    """What the backend worker was running, in the instance ledger.

    Deliberately not the framework's job queue: these operations exist only inside
    Hindsight's own task table. This is the accounting side of §8.1.2 — who was standing
    where, and what we still do not know about it.
    """

    def __init__(self, store):
        self.store = store
        self.db = store.db

    def record(self, task: Mapping[str, Any], *, worker_id: str, bank_id: str,
               resource: str | None, budget_scope: str = "global") -> dict[str, Any]:
        """Write the operation down before it runs. A retry grows the same row.

        ``attempts`` counts dispatches of this identity rather than inserting a row per
        restart, so a crashed worker returning to the same operation cannot double-spend
        the device unnoticed.
        """
        operation_id = _operation_id(task)
        folded = len(task.get(FOLD_KEY) or []) if isinstance(task, Mapping) else 0
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "INSERT INTO backend_operations(operation_id, kind, bank_id, resource, "
                "budget_scope, worker_id, state, attempts, folded, first_seen_at, updated_at) "
                "VALUES(?,?,?,?,?,?,'recorded',?,?,?,?) "
                "ON CONFLICT(operation_id) DO UPDATE SET attempts=attempts+1, folded=?, "
                "worker_id=excluded.worker_id, updated_at=excluded.updated_at, error=NULL",
                (operation_id, _kind(task), bank_id, resource, budget_scope, worker_id,
                 1 + _retry_count(task), folded, now(), now(), folded),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.get(operation_id)

    def get(self, operation_id: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM backend_operations WHERE operation_id=?",
                              (operation_id,)).fetchone()
        return dict(row) if row else {}

    def mark(self, operation_id: str, state: str, *, tokens: int = 0,
             error: str | None = None) -> dict[str, Any]:
        """Move an operation forward. A settled one is not reopened by a late answer.

        ``uncertain`` is the honest residue of a lost connection: the request may still be
        running upstream, and only the backend's own operation record can say otherwise.
        """
        if state not in {"recorded", "running", "finished", "uncertain", "refused",
                         "cancelled"}:
            raise LauncherRefused(f"unknown operation state {state!r}")
        current = self.get(operation_id)
        if not current:
            raise LauncherRefused(f"no record of operation {operation_id!r}")
        if current["state"] in TERMINAL and state != current["state"]:
            return {**current, "refused_transition": f"{current['state']} -> {state}"}
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE backend_operations SET state=?, tokens=tokens+?, error=?, "
                "updated_at=? WHERE operation_id=?",
                (state, max(0, int(tokens)), error and str(error)[:500], now(),
                 operation_id))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.get(operation_id)

    # -- cancellation --------------------------------------------------------

    def request_cancellation(self, operation_id: str, *, actor: str) -> dict[str, Any]:
        """Ask that an operation not start. Already-running work is never killed here.

        A settled row keeps ``cancellation='none'`` rather than accruing an intent nothing
        can ever answer, so the same :data:`TERMINAL` set that closes ``mark()`` closes the
        request: an operator's list of owed cancellations cannot hold a question with no
        possible answer.
        """
        if not isinstance(actor, str) or not actor.strip():
            raise LauncherRefused("a cancellation has to be attributed")
        current = self.get(operation_id)
        if not current:
            raise LauncherRefused(f"no record of operation {operation_id!r}")
        if current["state"] in TERMINAL:
            return current
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE backend_operations SET cancellation='requested', updated_at=? "
                "WHERE operation_id=?", (now(), operation_id))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.get(operation_id)

    def confirm_cancellation(self, operation_id: str) -> dict[str, Any]:
        """Mark an asked-for stop as acknowledged by the backend, and only then.

        ``cancellation`` is the durable statement of somebody's intent. Confirming one that
        was never requested would invent an intent nobody wrote, so the transition is
        one-way and refuses from ``none``; an already-confirmed row answers with itself.
        """
        current = self.get(operation_id)
        if not current:
            raise LauncherRefused(f"no record of operation {operation_id!r}")
        if current["cancellation"] == "confirmed":
            return current
        if current["cancellation"] != "requested":
            raise LauncherRefused(
                f"operation {operation_id!r} was never asked to stop; refusing to confirm a "
                "cancellation nobody requested")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE backend_operations SET cancellation='confirmed', updated_at=? "
                "WHERE operation_id=?", (now(), operation_id))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.get(operation_id)

    # -- reporting -----------------------------------------------------------

    def cancellation_owed(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Operations somebody asked to stop with no answer yet.

        Durable on purpose: the intent survives the process that wrote it, so this is the
        queue an operator works from after a restart rather than a memory of a request.
        """
        rows = self.db.execute(
            "SELECT operation_id, kind, bank_id, resource, state, cancellation, updated_at "
            "FROM backend_operations WHERE " + CANCELLATION_OWED +
            " ORDER BY updated_at LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def unresolved(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM backend_operations WHERE state IN ('running', 'uncertain', "
            "'recorded') ORDER BY updated_at LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def unattributed(self) -> int:
        """Operations we could not place on a resource — a defect, and never a silent free pass."""
        return int(self.db.execute(
            "SELECT count(*) FROM backend_operations WHERE resource IS NULL").fetchone()[0])

    @property
    def present(self) -> bool:
        """Whether this ledger has the table at all.

        A reading cannot upgrade the file it is reading, and a status report that raised on
        a ledger written by the previous build would be worse than one that said the table
        is not there yet. Opening the gate for any real dispatch brings it up.
        """
        return bool(self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='backend_operations'"
        ).fetchone())

    def report(self) -> dict[str, Any]:
        if not self.present:
            return {"ledger": "absent", "by_state": {}, "unattributed": 0, "unresolved": 0,
                    "cancellations_owed": 0}
        rows = self.db.execute(
            "SELECT state, count(*) AS n FROM backend_operations GROUP BY state").fetchall()
        return {"ledger": "present", "by_state": {row[0]: int(row[1]) for row in rows},
                "unattributed": self.unattributed(),
                "unresolved": len(self.unresolved()),
                "cancellations_owed": int(self.db.execute(
                    "SELECT count(*) FROM backend_operations WHERE "
                    + CANCELLATION_OWED).fetchone()[0])}


# -- the wrapped executor ----------------------------------------------------

def attribute_tasks(memory: Any, ledger: OperationLedger, *, worker_id: str,
                    routes: Any = None) -> Callable[[Mapping[str, Any]], Awaitable[None]]:
    """``memory.execute_task`` with the record written first and the outcome told honestly.

    The wrapper never decides whether the operation *should* have run — the poller owns
    that. It owns three narrower things: an operation the owner asked not to start is
    refused before the payload is executed, an operation whose bank or type cannot be read
    fails closed rather than running unattributed, and an exception from the engine is
    recorded as uncertain because a model call may have gone out and not come back.
    """
    async def execute(task: Mapping[str, Any]) -> None:
        if not isinstance(task, Mapping):
            raise LauncherRefused("a task payload must be a mapping to be attributed")
        bank = _bank(task)
        if not bank:
            raise LauncherRefused(
                f"operation {_operation_id(task)} names no bank; refusing to run it "
                "against somebody else's budget")
        resource, reason = _resource_for(task, routes)
        ledger.record(task, worker_id=worker_id, bank_id=bank, resource=resource)
        row = ledger.get(_operation_id(task))
        if row.get("cancellation") == "requested":
            ledger.mark(row["operation_id"], "cancelled",
                        error="cancellation was requested before the operation started")
            raise LauncherRefused(
                f"operation {row['operation_id']} was not started: the owner asked for it "
                "to be cancelled")
        if reason:
            ledger.mark(row["operation_id"], "uncertain", error=reason)
            raise LauncherRefused(f"operation {row['operation_id']}: {reason}")
        ledger.mark(row["operation_id"], "running")
        try:
            if resource == "backend-db":
                await asyncio.wait_for(memory.execute_task(task), timeout=DB_TASK_TIMEOUT_S)
            else:
                await memory.execute_task(task)
        except Exception as error:
            # Not "failed": the request may be sitting in a model server's queue still.
            ledger.mark(row["operation_id"], "uncertain", error=error)
            raise
        ledger.mark(row["operation_id"], "finished")
    return execute


def memory_arguments(modules: Mapping[str, Any], *, tenant_extension: Any = None,
                     operation_validator: Any = None, db_url: str | None = None) -> dict[str, Any]:
    """The engine arguments the pinned worker main passes: no migrations, worker backend.

    Workers do not run migrations — the API process owns the schema — and a
    ``WorkerTaskBackend`` makes submission a no-op because the row already exists. Handing
    the engine the plain task backend instead would have the worker enqueue work it is
    supposed to be executing.
    """
    arguments = {"run_migrations": False,
                 "task_backend": modules["WorkerTaskBackend"](),
                 "tenant_extension": tenant_extension,
                 "operation_validator": operation_validator}
    if db_url is not None:
        arguments["db_url"] = db_url
    _accepts(modules["MemoryEngine"], arguments, what="MemoryEngine")
    return arguments


def poller_arguments(memory: Any, modules: Mapping[str, Any], *, config: Any,
                     worker_id: str, ledger: OperationLedger, routes: Any = None,
                     tenant_extension: Any = None) -> dict[str, Any]:
    """The poller construction, with our executor wrapped around the engine's.

    ``backend`` and ``on_wall_timeout`` are read off the engine exactly as the pinned
    ``main`` reads them. The default database schema means "no prefix", which the poller
    wants as ``None`` rather than as a name it would qualify tables with.
    """
    default_schema = getattr(modules["config"], "DEFAULT_DATABASE_SCHEMA", None)
    schema = getattr(config, "database_schema", None)
    arguments = {
        "backend": memory._backend,
        "worker_id": worker_id,
        "executor": attribute_tasks(memory, ledger, worker_id=worker_id, routes=routes),
        "poll_interval_ms": _field(config, "worker_poll_interval_ms"),
        "schema": None if schema == default_schema else schema,
        "tenant_extension": tenant_extension,
        "max_slots": _field(config, "worker_max_slots"),
        "slot_reservations": _field(config, "worker_slot_reservations"),
        "consolidation_bank_priority": _field(config,
                                              "worker_consolidation_bank_priority") or None,
        "max_retries": _field(config, "worker_max_retries"),
        "on_wall_timeout": memory.on_task_wall_timeout,
    }
    _accepts(modules["WorkerPoller"], arguments, what="WorkerPoller")
    return arguments


def compose(*, modules: Mapping[str, Any], config: Any, worker_id: str,
            ledger: OperationLedger, routes: Any = None,
            tenant_extension: Any = None, operation_validator: Any = None,
            db_url: str | None = None) -> dict[str, Any]:
    """Perform the pinned construction against whatever ``modules`` actually holds.

    The order is the upstream one — engine first, then the poller that reads its private
    backend and its wall-timeout hook. Keyword names are checked against the real
    signatures before anything is called, so a drifted signature is a refusal here rather
    than an argument silently ignored at three in the morning.
    """
    engine_arguments = memory_arguments(modules, tenant_extension=tenant_extension,
                                        operation_validator=operation_validator, db_url=db_url)
    memory = modules["MemoryEngine"](**engine_arguments)
    poller = poller_arguments(memory, modules, config=config, worker_id=worker_id,
                              ledger=ledger, routes=routes,
                              tenant_extension=tenant_extension)
    return {"memory": memory, "memory_arguments": engine_arguments,
            "poller_arguments": poller, "poller_factory": modules["WorkerPoller"]}


# -- the process -------------------------------------------------------------

def check(settings=None, *, modules: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Everything a launch would depend on, reported without launching anything."""
    settings = settings or load_settings()
    declared = None
    if isinstance(modules, Mapping) and "hindsight_api" in modules:
        declared = getattr(modules["hindsight_api"], "__version__", None)
    version = version_contract(declared=declared)
    blockers: list[str] = []
    if modules is None:
        blockers.append("hindsight_api is not importable in this environment; the worker "
                        "runs in the backend environment, which is the one that ships it")
    elif not version["matches"]:
        blockers.append(f"this launcher was composed against hindsight-api-slim "
                        f"{version['pinned']} and found {version['declared']}")
    return {"ok": not blockers, "launcher_version": LAUNCHER_VERSION,
            "version": version, "blocking": blockers,
            "gate_ledger": str(gate_path(settings)),
            "profile": settings.profile, "bank_id": settings.bank_id,
            "capture_only": settings.capture_only,
            "not_performed": ["no operation was claimed, executed or cancelled",
                              "no database connection string is printed here"]}


def main(argv: list[str] | None = None) -> int:
    """``python -m hermes_memory.backend.worker_launcher`` — the unit's ExecStart.

    ``--check`` answers the questions preflight asks and starts nothing. A real launch
    refuses on any contract failure with one machine-readable line and a non-zero exit, so
    a manager that restarts on failure keeps saying why.
    """
    parser = argparse.ArgumentParser(prog="hermes-memory-worker-launcher")
    parser.add_argument("--check", action="store_true",
                        help="report the version and slot contracts and start nothing")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--env-file", default=None)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.env_file)
    except SettingError as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2
    modules = _import_backend()
    verdict = check(settings, modules=modules)
    if args.check:
        print(json.dumps(verdict, sort_keys=True))
        return 0 if verdict["ok"] else 2
    if not verdict["ok"]:
        print(json.dumps({**verdict, "refused": "launch"}, sort_keys=True), file=sys.stderr)
        return 2
    config = modules["hindsight_api.config"].get_config()
    slots = slot_contract(config)
    if not slots["ok"]:
        print(json.dumps({"ok": False, "slot_contract": slots}, sort_keys=True),
              file=sys.stderr)
        return 2
    try:
        return _launch(settings, modules=modules, config=config,
                       worker_id=args.worker_id or f"hermes-memory-worker-{settings.profile}")
    except LauncherRefused as error:
        # A refusal is the answer, not a stack trace: the manager will start us again in
        # fifteen seconds and the reason has to still be readable in the journal.
        print(json.dumps({"ok": False, "refused": str(error)[:400]}, sort_keys=True),
              file=sys.stderr)
        return 2


async def worker_database(settings, config, *, sleep, ready=None, inspect_instance=None,
                          wait_s: float = DB_WAIT_SECONDS) -> str:
    """The API owns pg0 startup/migrations; the worker only reads its resolved DSN."""
    import asyncio
    import urllib.error
    import urllib.request
    from urllib.parse import urlsplit

    configured = str(_field(config, "database_url") or "")
    if configured != "pg0" and not configured.startswith("pg0://"):
        return configured

    if ready is None:
        def ready():
            try:
                with urllib.request.urlopen(settings.hindsight_url.rstrip("/") + "/health",
                                            timeout=3) as response:
                    return response.status == 200
            except (urllib.error.URLError, TimeoutError, OSError):
                return False

    waited = 0.0
    while not await asyncio.to_thread(ready):
        if waited >= wait_s:
            raise LauncherRefused("API-owned embedded database is not ready; worker did not start it")
        await sleep(DB_POLL_SECONDS)
        waited += DB_POLL_SECONDS

    if inspect_instance is None:
        from hindsight_api.pg0 import parse_pg0_url
        from pg0 import Pg0
        parsed = parse_pg0_url(configured)
        inspect_instance = Pg0(name=parsed.instance_name).info
    info = await asyncio.to_thread(inspect_instance)
    uri = str(info.uri or "")
    if not info.running or urlsplit(uri).scheme not in {"postgres", "postgresql"}:
        raise LauncherRefused("API answered but its embedded database has no running PostgreSQL DSN")
    return uri


def _launch(settings, *, modules: Mapping[str, Any], config: Any,
            worker_id: str) -> int:
    """Compose, then hand the poller to the event loop. One refusal above, one path here."""
    import asyncio

    with GateStore(gate_path(settings)) as ledger_store:
        ledger = OperationLedger(ledger_store)
        database_url = None

        def build() -> dict[str, Any]:
            built = compose(
                modules={"MemoryEngine": modules["hindsight_api"].MemoryEngine,
                         "WorkerTaskBackend":
                             modules["hindsight_api.engine.task_backend"].WorkerTaskBackend,
                         "WorkerPoller": modules["hindsight_api.worker.poller"].WorkerPoller,
                         "config": modules["hindsight_api.config"]},
                config=config, worker_id=worker_id, ledger=ledger,
                routes=_routes(settings), db_url=database_url)
            if not getattr(built["memory"]._backend, "supports_worker_poller", False):
                # The pinned main exits here rather than running operations inline in a
                # process that was not asked to run them.
                raise LauncherRefused("this database backend has no worker poller; the API "
                                      "process runs operations itself")
            return dict(built)

        async def run() -> None:
            nonlocal database_url
            # The hold is asked of the ledger before the engine is asked for anything: an
            # installation whose owner stopped the models has a queue nobody may drain, and
            # a worker that exited over that would be restarted into the same exit until its
            # start limit silenced the unit for the rest of the hold.
            await wait_for_inference(ledger_store, sleep=asyncio.sleep)
            database_url = await worker_database(settings, config, sleep=asyncio.sleep)
            built = await bring_up(build, ledger_store, sleep=asyncio.sleep)
            memory = built["memory"]
            poller = built["poller_factory"](**built["poller_arguments"])
            try:
                await poller.run()
            finally:
                await memory.close()

        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            pass
    return 0


# -- internals ---------------------------------------------------------------

def _import_backend() -> Mapping[str, Any] | None:
    """The pinned modules, or None if this environment does not carry the backend.

    Import failures are turned into an answer rather than a traceback: the launcher runs in
    the framework's own environment often enough that "not here" is the useful report.
    """
    import importlib

    found: dict[str, Any] = {}
    for path, names in REQUIRED_IMPORTS.items():
        try:
            module = importlib.import_module(path)
        except ImportError:
            return None
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            raise LauncherRefused(f"{path} is missing {', '.join(missing)}; this launcher "
                                  f"was written against hindsight-api-slim "
                                  f"{PINNED_VERSION}")
        found[path] = module
    return found


def _routes(settings):
    if not settings.hindsight_url:
        return None
    try:
        return build_routes(settings, credentials=settings.route_credentials)
    except SettingError:
        return None


def _field(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _operation_id(task: Any) -> str:
    value = task.get(OPERATION_KEY) if isinstance(task, Mapping) else None
    text = str(value or "").strip()
    if not text:
        raise LauncherRefused("the claimed task carries no operation id; an unattributable "
                              "execution cannot be accounted for")
    return text[:200]


def _kind(task: Any) -> str:
    for key in TYPE_KEYS:
        value = task.get(key)
        if value:
            return str(value)[:60]
    return "unknown"


def _bank(task: Any) -> str:
    for key in BANK_KEYS:
        value = task.get(key)
        if value:
            return str(value)[:200]
    return ""


def _retry_count(task: Any) -> int:
    try:
        return max(0, int(task.get(RETRY_KEY) or 0))
    except (TypeError, ValueError):
        return 0


def _resource_for(task: Any, routes: Any) -> tuple[str | None, str | None]:
    """Which physical device this operation is about to occupy, or why we cannot say.

    An unknown operation type is recorded without a resource and counted as a defect. The
    alternative — borrowing a route that happens to exist — is how one profile's work ends
    up charged to another's allowance.
    """
    kind = _kind(task)
    # Without poller metadata, only an explicitly known executor discriminator
    # may select a policy; a mismatched claimed operation/payload is refused.
    if kind == "batch_retain" and not task.get("operation_type"):
        kind = "retain"
    policy = TASK_POLICIES.get(kind)
    if policy is None:
        return None, f"operation type {kind!r} matches no configured route or task policy"
    wanted, payload_types = policy
    payload_type = task.get("type")
    if payload_type is not None and (not isinstance(payload_type, str)
                                      or payload_type not in payload_types):
        return None, "operation type and executor payload type disagree"
    if wanted is None:
        # Pinned graph maintenance only drains/relinks/prunes DB queues; it has
        # its own deadline and one poller slot, not an imaginary inference route.
        return "backend-db", None
    if routes is None:
        return None, "no route table is configured, so this operation cannot be charged " \
                     "to a physical resource"
    try:
        route = routes.by_name(wanted)
    except Exception:
        return None, f"operation type {wanted!r} matches no configured route"
    return route.resource, None


def _accepts(factory: Any, arguments: Mapping[str, Any], *, what: str) -> None:
    """Refuse a construction whose signature no longer matches the one we read."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # a C-level or exotic callable: nothing to compare
        return
    parameters = signature.parameters
    if any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values()):
        return
    unexpected = sorted(key for key in arguments if key not in parameters)
    if unexpected:
        raise LauncherRefused(
            f"{what} no longer accepts {', '.join(unexpected)}; this launcher is pinned to "
            f"hindsight-api-slim {PINNED_VERSION} and will not guess at a newer signature")


if __name__ == "__main__":
    # The unit's ExecStart is `python -m hermes_memory.backend.worker_launcher`, so this
    # guard is not a convenience: without it the module imports, runs nothing, and exits 0
    # — which `Restart=on-failure` reads as a service that stopped cleanly, on a machine
    # whose owner believes the worker is draining.
    raise SystemExit(main())
