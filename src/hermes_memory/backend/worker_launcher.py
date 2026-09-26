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

The resolved database DSN is never printed. It carries credentials, and a worker that logs
its own connection string leaks the whole machine's memory into a journal.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from typing import Any, Awaitable, Callable, Mapping

from ..config import SettingError, load_settings
from ..ids import now
from ..processing.instance_gate import GateStore, gate_path
from ..processing.routes import build_routes
from .capabilities import PINNED_VERSION

__all__ = ["LAUNCHER_VERSION", "REQUIRED_IMPORTS", "LauncherRefused", "version_contract",
           "slot_contract", "OperationLedger", "attribute_tasks", "compose", "check", "main"]

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

TERMINAL = frozenset({"finished", "cancelled", "refused"})


class LauncherRefused(Exception):
    """The composition this launcher would perform is not safe to perform."""


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
        """Ask that an operation not start. Already-running work is never killed here."""
        if not isinstance(actor, str) or not actor.strip():
            raise LauncherRefused("a cancellation has to be attributed")
        if not self.get(operation_id):
            raise LauncherRefused(f"no record of operation {operation_id!r}")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE backend_operations SET cancellation='requested', updated_at=? "
                "WHERE operation_id=? AND state NOT IN ('finished', 'cancelled')",
                (now(), operation_id))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.get(operation_id)

    # -- reporting -----------------------------------------------------------

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
            return {"ledger": "absent", "by_state": {}, "unattributed": 0, "unresolved": 0}
        rows = self.db.execute(
            "SELECT state, count(*) AS n FROM backend_operations GROUP BY state").fetchall()
        return {"ledger": "present", "by_state": {row[0]: int(row[1]) for row in rows},
                "unattributed": self.unattributed(),
                "unresolved": len(self.unresolved())}


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
            await memory.execute_task(task)
        except Exception as error:
            # Not "failed": the request may be sitting in a model server's queue still.
            ledger.mark(row["operation_id"], "uncertain", error=error)
            raise
        ledger.mark(row["operation_id"], "finished")
    return execute


def memory_arguments(modules: Mapping[str, Any], *, tenant_extension: Any = None,
                     operation_validator: Any = None) -> dict[str, Any]:
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
            tenant_extension: Any = None, operation_validator: Any = None) -> dict[str, Any]:
    """Perform the pinned construction against whatever ``modules`` actually holds.

    The order is the upstream one — engine first, then the poller that reads its private
    backend and its wall-timeout hook. Keyword names are checked against the real
    signatures before anything is called, so a drifted signature is a refusal here rather
    than an argument silently ignored at three in the morning.
    """
    engine_arguments = memory_arguments(modules, tenant_extension=tenant_extension,
                                        operation_validator=operation_validator)
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


def _launch(settings, *, modules: Mapping[str, Any], config: Any,
            worker_id: str) -> int:
    """Compose, then hand the poller to the event loop. One refusal above, one path here."""
    import asyncio

    with GateStore(gate_path(settings)) as ledger_store:
        ledger = OperationLedger(ledger_store)
        built = compose(
            modules={"MemoryEngine": modules["hindsight_api"].MemoryEngine,
                     "WorkerTaskBackend":
                         modules["hindsight_api.engine.task_backend"].WorkerTaskBackend,
                     "WorkerPoller": modules["hindsight_api.worker.poller"].WorkerPoller,
                     "config": modules["hindsight_api.config"]},
            config=config, worker_id=worker_id, ledger=ledger,
            routes=_routes(settings))
        memory = built["memory"]
        if not getattr(memory._backend, "supports_worker_poller", False):
            # The pinned main exits here rather than running operations inline in a
            # process that was not asked to run them.
            raise LauncherRefused("this database backend has no worker poller; the API "
                                  "process runs operations itself")
        poller = built["poller_factory"](**built["poller_arguments"])

        async def run() -> None:
            await memory.initialize()
            try:
                await poller.run()
            finally:
                await memory.shutdown()

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
    if routes is None:
        return None, "no route table is configured, so this operation cannot be charged " \
                     "to a physical resource"
    wanted = _kind(task)
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
