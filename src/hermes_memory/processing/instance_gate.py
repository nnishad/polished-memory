"""The one resource gate: every profile, the same physical models, one slot each.

§10.5 counts the enrollment of a second person as an increase in demand on the same
GPU and the same remote model server, and forbids the obvious wrong answer — a gateway
per profile, each sure that it serialised its own work. So the reservation table, the
accounting ledger beside it and the hold on ``inference`` all live in one database under
the instance home, which every profile's process opens and none of them owns.

The single-slot rule is still enforced the same way it always was: a partial unique index
in SQLite, not an in-memory lock. Moving the table from one file to another is what turns
"one holder per profile store" into "one holder per resource", because now there is one
store to conflict against.

The canonical per-profile databases are unchanged in every other respect. Evidence,
outboxes, source policies, journals and the delivery hold all stay where they were; only
admission for a *physical* device is shared, because that is the only thing the two
profiles genuinely have in common.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..ids import now
from ..storage.migrations import (BUDGET_STATEMENTS, CONTROLS_STATEMENTS,
                                  GATE_STATEMENTS, connect)
from .resource_gate import ResourceGate

__all__ = ["GATE_FILENAME", "GATE_SCHEMA_VERSION", "GateStore", "GateError",
           "gate_path", "instance_gate", "status_gate"]

GATE_FILENAME = "gate.db"
GATE_SCHEMA_VERSION = 1
# The whole of it: reservations, their accounting, the measured consumption the gate
# charges when it releases a slot, and the hold an operator puts on inference.
GATE_SCHEMA = (*GATE_STATEMENTS, *CONTROLS_STATEMENTS, *BUDGET_STATEMENTS)


class GateError(sqlite3.Error):
    """The gate database is not one this build can use."""


def gate_path(settings) -> Path:
    return Path(settings.home) / GATE_FILENAME


class GateStore:
    """The instance admission ledger. One file, opened by whoever has to queue.

    Deliberately small: it holds reservations, their accounting and the inference hold,
    and nothing that belongs to a person. ``closes_with_gate`` is how :class:`ResourceGate`
    knows this database was opened by the gate and not handed in by a caller.
    """

    closes_with_gate = True

    def __init__(self, path: str | Path, *, create: bool = True):
        self.path = Path(path)
        if not create:
            # A reading does not bring a ledger into being. `status` asks how busy the
            # models are; it has no business creating the file that answers.
            if not self.path.is_file():
                raise GateError(f"no admission ledger at {self.path}; nothing has been "
                                "queued from this installation yet")
            self.db = sqlite3.connect(self.path.absolute().as_uri() + "?mode=ro",
                                      uri=True, timeout=10, isolation_level=None)
            self.db.row_factory = sqlite3.Row
            return
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.db = connect(self.path)
        self._bring_up()
        if self.path.is_file():
            # Who is running what is not public, even on a shared account.
            self.path.chmod(0o600)

    def _bring_up(self) -> None:
        version = int(self.db.execute("PRAGMA user_version").fetchone()[0])
        if version > GATE_SCHEMA_VERSION:
            raise GateError(
                f"the gate database at {self.path} is at schema {version} and this build "
                f"understands {GATE_SCHEMA_VERSION}; upgrade the framework rather than "
                "reading a newer admission ledger with an older one")
        if version == GATE_SCHEMA_VERSION:
            return
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for statement in GATE_SCHEMA:
                self.db.execute(statement)
            self.db.execute(f"PRAGMA user_version={GATE_SCHEMA_VERSION}")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

    # -- the hold that belongs to the machine --------------------------------

    def set_control(self, scope: str, stage: str, state: str, *, actor: str,
                    reason: str, policy_version: str) -> None:
        if state not in {"active", "paused"}:
            raise GateError("state must be 'active' or 'paused'")
        if not isinstance(actor, str) or not actor.strip():
            raise GateError("a hold on the shared models has to be attributed")
        self.db.execute(
            "INSERT INTO runtime_controls(scope, stage, state, actor, reason, "
            "policy_version, changed_at) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(scope, stage) DO UPDATE SET state=excluded.state, "
            "actor=excluded.actor, reason=excluded.reason, "
            "policy_version=excluded.policy_version, changed_at=excluded.changed_at",
            (scope, stage, state, actor.strip(), reason, policy_version, now()))

    def stage_is_paused(self, scope: str, stage: str) -> bool:
        row = self.db.execute("SELECT state FROM runtime_controls WHERE scope=? AND "
                              "stage=?", (scope, stage)).fetchone()
        return bool(row and row[0] == "paused")

    def control(self, scope: str, stage: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT state, actor, reason, policy_version, changed_at "
                              "FROM runtime_controls WHERE scope=? AND stage=?",
                              (scope, stage)).fetchone()
        return dict(row) if row else None

    # -- lifetime ------------------------------------------------------------

    def close(self) -> None:
        try:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        self.db.close()

    def __enter__(self) -> "GateStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def instance_gate(settings, *, clock=None, default_ttl: float = 300.0,
                  create: bool = True) -> ResourceGate:
    """A gate over this installation's shared admission ledger.

    The caller may ``close()`` it, which closes the database the gate opened; a gate built
    on somebody's evidence store never closes that store.
    """
    arguments: dict[str, Any] = {"create": create}
    if clock is not None:
        arguments["clock"] = clock
    return ResourceGate(GateStore(gate_path(settings), **arguments),
                        default_ttl=default_ttl)


def status_gate(settings) -> ResourceGate | None:
    """The gate, for a reading that must not create anything.

    ``None`` when there is no ledger yet. That is an answer — nothing has been queued
    from here — and inventing an empty database to give it would make every status
    command the first writer in a machine it promised only to look at.
    """
    try:
        return instance_gate(settings, create=False)
    except GateError:
        return None
