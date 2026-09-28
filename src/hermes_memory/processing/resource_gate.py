"""C12 physical admission: one slot per device, across every process and profile.

Hindsight's own concurrency caps are kept as defense in depth, but they cannot
be the boundary: its semaphores are process-local, the single-batch embedding
path bypasses its request semaphore entirely, and the vision and embedding
pipelines share a GPU with no common lock. So admission is enforced here, in a
table whose partial unique index makes a second concurrent holder impossible
rather than merely unlikely.

A timeout is not evidence that an upstream slot is free. An expired lease
therefore becomes 'uncertain' and keeps the resource blocked until completion or
abort is established — guessing would let two requests hit one model server.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ..ids import now
from ..storage.evidence import EvidenceError

__all__ = ["Reservation", "ResourceGate", "GateBusy", "GatePaused", "GateClosed"]

HELD = "held"
UNCERTAIN = "uncertain"
WAITING = "waiting"
RELEASED = "released"


class GateError(EvidenceError):
    pass


class GateBusy(GateError):
    """The resource is occupied. Waiting is the caller's decision."""


class GatePaused(GateError):
    """An operator paused all inference. New dispatch is denied immediately."""


class GateClosed(GateError):
    """The reservation is no longer usable."""


@dataclass(frozen=True)
class Reservation:
    id: str
    resource: str
    route: str
    holder: str
    priority: int
    lease_until: float


class ResourceGate:
    def __init__(self, store, *, clock: Callable[[], float] = time.time,
                 default_ttl: float = 300.0):
        self.store = store
        self.db = store.db
        self.clock = clock
        self.default_ttl = default_ttl

    # -- dispatch ------------------------------------------------------------

    def try_acquire(self, *, route: str, holder: str, resource: str, priority: int,
                    job_id: str | None = None, ttl: float | None = None) -> Reservation | None:
        """One admission attempt. Returns None when the resource is occupied."""
        self._refuse_if_paused()
        ttl = self.default_ttl if ttl is None else ttl
        if not 1 <= ttl <= 3600:
            raise EvidenceError("ttl must be between 1 and 3600 seconds")
        if priority not in (0, 1, 2, 3):
            raise EvidenceError("priority must be one of 0..3 (interactive..maintenance)")
        waiter = f"wait_{uuid.uuid4().hex}"
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._reap_locked()
            self.db.execute(
                "INSERT INTO gate_reservations(id, resource, route, holder, priority, state, "
                "acquired_at, lease_until, job_id) VALUES(?,?,?,?,?,?,?,0,?)",
                (waiter, resource, route, holder, priority, WAITING, self.clock(), job_id),
            )
            promoted = self._promote_locked(waiter, resource, priority)
            if not promoted:
                self.db.execute("DELETE FROM gate_reservations WHERE id=?", (waiter,))
                self.db.execute("COMMIT")
                return None
            lease_until = self.clock() + ttl
            self.db.execute(
                "UPDATE gate_reservations SET lease_until=? WHERE id=?", (lease_until, waiter))
            self._log(waiter, resource, route, holder, "acquired",
                      {"priority": priority, "job_id": job_id, "ttl": ttl})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return Reservation(waiter, resource, route, holder, priority, lease_until)

    def acquire(self, *, route: str, holder: str, resource: str, priority: int,
                job_id: str | None = None, ttl: float | None = None,
                timeout: float = 0.0, poll: float = 0.05) -> Reservation | None:
        """Wait for the resource, keeping a visible queue entry the whole time.

        Registering for the duration is what makes priority meaningful: two
        blocked callers can see each other, so the more urgent one wins instead
        of whichever happened to poll first after the slot freed.

        Waiting is only offered to a caller that can be served by it. A held slot belongs to a
        request that will end; an unresolved one belongs to a request nobody can answer for, and
        it keeps the device blocked until that outcome is established from outside — by a
        reconciliation that asks the backend, or by an operator's written settlement. Standing in
        line behind that pays a full timeout for an answer no amount of waiting produces, so the
        standing ends with it. The question is asked after the first attempt, not before it, so
        that a lease which only just expired is reaped into uncertainty and named as such, rather
        than being waited on as though it were still live work.
        """
        deadline = self.clock() + timeout
        while True:
            attempt = self.try_acquire(route=route, holder=holder, resource=resource,
                                       priority=priority, job_id=job_id, ttl=ttl)
            if attempt is not None or self.clock() >= deadline:
                return attempt
            if self.unresolved_for(resource):
                return None
            time.sleep(max(poll, 0.001))

    def _promote_locked(self, waiter: str, resource: str, priority: int) -> bool:
        """Claim the slot only if nobody holds it and nobody more urgent waits."""
        acquired_at = self.db.execute(
            "SELECT acquired_at FROM gate_reservations WHERE id=?", (waiter,)).fetchone()[0]
        cursor = self.db.execute(
            """
            UPDATE gate_reservations SET state=?
            WHERE id=?
              AND NOT EXISTS(SELECT 1 FROM gate_reservations
                             WHERE resource=? AND state IN (?,?))
              AND NOT EXISTS(SELECT 1 FROM gate_reservations
                             WHERE resource=? AND state=? AND id != ?
                               AND (priority < ? OR (priority = ? AND acquired_at < ?)))
            """,
            (HELD, waiter, resource, HELD, UNCERTAIN, resource, WAITING, waiter,
             priority, priority, acquired_at),
        )
        return cursor.rowcount == 1

    # -- completion ----------------------------------------------------------

    def release(self, reservation: Reservation, *, outcome: str, tokens: int = 0,
                seconds: float = 0.0) -> None:
        """Free the slot and charge measured usage, including internal retries."""
        if outcome not in {"succeeded", "failed", "cancelled", "malformed", "truncated"}:
            raise EvidenceError(f"unknown outcome {outcome!r}")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self._row_or_raise(reservation.id)
            if row["state"] == RELEASED:
                self.db.execute("COMMIT")
                return
            if row["state"] == UNCERTAIN:
                # Resolving an uncertain reservation is the point: this is how a
                # timed-out request is finally established as finished or dead.
                pass
            self.db.execute(
                "UPDATE gate_reservations SET state=?, released_at=?, outcome=? WHERE id=?",
                (RELEASED, self.clock(), outcome, reservation.id))
            self._charge(row["resource"], tokens=tokens, seconds=seconds)
            self._log(reservation.id, row["resource"], row["route"], row["holder"], "released",
                      {"outcome": outcome, "tokens": tokens, "seconds": round(seconds, 3)})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def charge(self, resource: str, *, tokens: int, seconds: float,
               reservation: Reservation | None = None, note: str = "") -> None:
        """Record measured device use that had no admission of its own to release.

        A submission handed the slot back and the engine then ran the model under its own
        admission; the cost of that run is still this installation's spending, and a daily
        budget that only counts what a caller held the slot for would understate the device
        and let a pass refuse nothing when it should. This is an accounting entry, not a key:
        it grants no access, holds no slot and cannot make a resource busy — which is why it
        takes a resource name where every other method takes a reservation.
        """
        if not isinstance(resource, str) or not resource.strip():
            raise EvidenceError("a charge names the resource that earned it")
        if not isinstance(tokens, int) or isinstance(tokens, bool):
            raise EvidenceError("tokens must be an integer")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            # No call is counted here: `calls` is how many times this installation was admitted
            # to the device, and the admission that covered this work was already counted when
            # the submission released its slot. Only the spend arrives after the fact.
            self._charge(resource, tokens=tokens, seconds=seconds, calls=0)
            self._log(reservation.id if reservation else "-", resource, "-", "system",
                      "charged", {"tokens": tokens, "seconds": round(seconds, 3),
                                  "note": note[:300]})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def mark_uncertain(self, reservation: Reservation, *, reason: str) -> None:
        """Record that we do not know whether the upstream finished.

        The slot stays occupied. Killing or reconfiguring the user's model
        servers to find out is not an option this framework takes.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise EvidenceError("reason must be nonempty text")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self._row_or_raise(reservation.id)
            if row["state"] == RELEASED:
                raise GateClosed(f"reservation {reservation.id} is already released")
            self.db.execute(
                "UPDATE gate_reservations SET state=?, outcome=? WHERE id=?",
                (UNCERTAIN, reason[:500], reservation.id))
            self._log(reservation.id, row["resource"], row["route"], row["holder"],
                      "uncertain", {"reason": reason})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def renew(self, reservation: Reservation, *, ttl: float | None = None) -> float | None:
        """Vouch for a slot that is still being used, and say whether it is still ours.

        A long operation outlives any lease fixed at its start, and a lease that cannot be
        renewed forces a choice between two false reports: that the device is free when it
        is busy, or that the worker died when it did not. The answer comes back rather than
        being assumed, because the row may have been settled from outside in the meantime —
        an operator, or the expiry of a lease this worker stopped vouching for.

        A promise that has run out is not renewed into existence either. The state alone
        can still say ``held`` for as long as nothing has reaped it, and another admission
        may already have been promoted over the top of it at the next reaping, so the clock
        is part of the question.
        """
        ttl = self.default_ttl if ttl is None else ttl
        if not 1 <= ttl <= 3600:
            raise EvidenceError("ttl must be between 1 and 3600 seconds")
        until = self.clock() + ttl
        self.db.execute("BEGIN IMMEDIATE")
        try:
            moved = self.db.execute(
                "UPDATE gate_reservations SET lease_until=? WHERE id=? AND state=? "
                "AND lease_until>=?", (until, reservation.id, HELD, self.clock()))
            held = moved.rowcount == 1
            if held:
                self._log(reservation.id, reservation.resource, reservation.route,
                          reservation.holder, "renewed", {"lease_until": until, "ttl": ttl})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return until if held else None

    def resolve(self, reservation_id: str, *, outcome: str, actor: str,
                reason: str) -> dict[str, Any]:
        """Establish what happened to a slot this process could not prove, and free it.

        The holder cannot call this: a request whose connection was lost is exactly the
        case where the caller has no right to claim the upstream finished. So the answer
        comes from outside — a cancellation the backend acknowledged, a reconciled record,
        an operator who looked — and this writes down who gave it and when.

        Nothing is charged to the budget. The token count of a request nobody saw the end
        of is not a number that can be honestly added to a total, and a ledger that
        invented one would make the day's spend a guess.
        """
        if outcome not in {"succeeded", "failed", "cancelled"}:
            raise EvidenceError(
                f"unknown outcome {outcome!r}; an unresolved reservation stays unresolved "
                "until somebody establishes that it finished, failed or was stopped")
        for name, value in (("actor", actor), ("reason", reason)):
            if not isinstance(value, str) or not value.strip():
                raise EvidenceError(f"{name} must be nonempty text: a settled slot without a "
                                    "name on it is indistinguishable from a lost one")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self._row_or_raise(reservation_id)
            if row["state"] == RELEASED:
                raise GateClosed(f"reservation {reservation_id} was already released by its "
                                 "holder; there is nothing here for an operator to settle")
            if row["state"] != UNCERTAIN:
                raise GateClosed(
                    f"reservation {reservation_id} is still a live lease held by "
                    f"{row['holder']!r}; an operator settles a device nobody can answer for, "
                    "and a live lease becomes that when it expires")
            self.db.execute(
                "UPDATE gate_reservations SET state=?, released_at=?, outcome=? WHERE id=?",
                (RELEASED, self.clock(), outcome[:500], reservation_id))
            self._log(reservation_id, row["resource"], row["route"], row["holder"], "resolved",
                      {"outcome": outcome, "actor": actor[:200], "reason": reason[:400]})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"reservation": reservation_id, "resource": row["resource"],
                "route": row["route"], "holder": row["holder"], "state": RELEASED,
                "outcome": outcome, "settled_by": actor}

    def reap_expired(self) -> list[str]:
        """Expired leases become uncertain; they never become free."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            reaped = self._reap_locked()
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return reaped

    def _reap_locked(self) -> list[str]:
        rows = self.db.execute(
            "SELECT id, resource, route, holder FROM gate_reservations "
            "WHERE state=? AND lease_until < ?", (HELD, self.clock())).fetchall()
        for row in rows:
            self.db.execute(
                "UPDATE gate_reservations SET state=?, outcome='lease expired' WHERE id=?",
                (UNCERTAIN, row["id"]))
            self._log(row["id"], row["resource"], row["route"], row["holder"], "uncertain",
                      {"reason": "lease expired"})
        return [row["id"] for row in rows]

    # -- operator controls ---------------------------------------------------

    def pause(self, *, actor: str, reason: str) -> None:
        """All-inference pause. Already-started requests are reported, not killed."""
        self.store.set_control("global", "inference", "paused", actor=actor, reason=reason,
                               policy_version="gate-v1")

    def resume(self, *, actor: str, reason: str) -> None:
        self.store.set_control("global", "inference", "active", actor=actor, reason=reason,
                               policy_version="gate-v1")

    @property
    def paused(self) -> bool:
        return self.store.stage_is_paused("global", "inference")

    def hold(self) -> dict[str, Any] | None:
        """Who is holding inference, and since when. None if nobody wrote it down.

        ``paused`` answers "may I run"; this answers "who decided that", which is the
        question worth asking of a hold that stops inference for every profile sharing
        this machine.
        """
        return self.store.control("global", "inference")

    def close(self) -> None:
        """Release the database this gate opened, if it opened one.

        A gate built on somebody's evidence store leaves that store alone: closing it
        here would take the archive's connection out from under its owner.
        """
        if getattr(self.store, "closes_with_gate", False):
            self.store.close()

    def __enter__(self) -> "ResourceGate":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _refuse_if_paused(self) -> None:
        if self.paused:
            raise GatePaused("all inference is paused by the operator; new dispatch is denied")

    # -- reporting -----------------------------------------------------------

    def occupancy(self) -> dict[str, Any]:
        rows = self.db.execute(
            "SELECT resource, state, count(*) AS n FROM gate_reservations "
            "WHERE state IN (?,?,?) GROUP BY resource, state",
            (HELD, UNCERTAIN, WAITING)).fetchall()
        out: dict[str, Any] = {}
        for row in rows:
            out.setdefault(row["resource"], {})[row["state"]] = row["n"]
        return out

    def held(self) -> list[dict[str, Any]]:
        """Who is standing in a slot right now, with the clocks left as stored.

        Reported rather than recomputed: the age of a reservation is the first thing an
        operator asks, and a reader that rounds it here would hide how stale the row was
        when it was read.
        """
        rows = self.db.execute(
            "SELECT resource, route, holder, priority, acquired_at, lease_until "
            "FROM gate_reservations WHERE state=? ORDER BY resource", (HELD,)).fetchall()
        return [dict(row) for row in rows]

    def unresolved(self) -> list[dict[str, Any]]:
        """Reservations that keep a device blocked with nobody left to answer for them.

        ``held`` is the live queue; this is the residue — a request whose connection went
        away and whose end nobody has established. Each row here is a device nothing may
        use, which is why the doctor calls it degraded and why settling one is a written
        decision rather than a delete.
        """
        rows = self.db.execute(
            "SELECT id, resource, route, holder, job_id, acquired_at, outcome "
            "FROM gate_reservations WHERE state=? ORDER BY acquired_at", (UNCERTAIN,)).fetchall()
        return [dict(row) for row in rows]

    def last_outcomes(self) -> dict[str, dict[str, Any]]:
        """The newest settled answer on each route, as this installation got it.

        A pin says which routes the backend serves; this says what came back when one was
        asked for work. The two are different claims and they disagree: a route can be
        routed and still be answered by a model that cannot do the thing it was asked to
        do, which is exactly the gap between "supported" and "usable here".
        """
        rows = self.db.execute(
            "SELECT route, resource, state, outcome, "
            "coalesce(released_at, acquired_at) AS settled_at FROM gate_reservations "
            "WHERE outcome IS NOT NULL ORDER BY settled_at DESC, id")
        newest: dict[str, dict[str, Any]] = {}
        for row in rows:
            newest.setdefault(row["route"], dict(row))
        return newest

    def unresolved_for(self, resource: str) -> bool:
        """Whether this device is blocked by a request nobody can answer for.

        The question a caller asks before it decides to wait. A held slot is a queue this caller
        can join and be served from; an unresolved row is a device that stays blocked until
        somebody establishes what happened outside this process, and waiting is not somebody.
        """
        return bool(self.db.execute(
            "SELECT 1 FROM gate_reservations WHERE resource=? AND state=?",
            (resource, UNCERTAIN)).fetchone())

    def ever_used(self) -> bool:
        """Whether this gate has ever admitted anything. An empty table is a fact."""
        return bool(self.db.execute("SELECT 1 FROM gate_reservations LIMIT 1").fetchone())

    def blocked_resources(self) -> list[str]:
        """Resources that cannot accept work, including ones stuck uncertain."""
        rows = self.db.execute(
            "SELECT DISTINCT resource FROM gate_reservations WHERE state IN (?,?)",
            (HELD, UNCERTAIN)).fetchall()
        return sorted(row["resource"] for row in rows)

    def usage(self, *, scope: str = "global") -> dict[str, Any]:
        rows = self.db.execute(
            "SELECT resource, tokens, calls, seconds FROM budget_usage WHERE scope=?",
            (scope,)).fetchall()
        return {row["resource"]: {"tokens": row["tokens"], "calls": row["calls"],
                                  "seconds": round(row["seconds"], 3)} for row in rows}

    # -- internals -----------------------------------------------------------

    def _row_or_raise(self, reservation_id: str):
        row = self.db.execute(
            "SELECT * FROM gate_reservations WHERE id=?", (reservation_id,)).fetchone()
        if row is None:
            raise GateClosed(f"unknown reservation {reservation_id!r}")
        return row

    def _charge(self, resource: str, *, tokens: int, seconds: float, calls: int = 1) -> None:
        if tokens < 0 or seconds < 0 or calls < 0:
            raise EvidenceError("usage cannot be negative")
        period = now()[:10]
        self.db.execute(
            "INSERT INTO budget_usage(scope, period, resource, tokens, calls, seconds) "
            "VALUES('global',?,?,?,?,?) "
            "ON CONFLICT(scope, period, resource) DO UPDATE SET "
            "tokens=budget_usage.tokens+excluded.tokens, calls=budget_usage.calls+excluded.calls, "
            "seconds=budget_usage.seconds+excluded.seconds",
            (period, resource, tokens, calls, seconds),
        )

    def _log(self, reservation_id: str, resource: str, route: str, holder: str,
             event: str, detail: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO gate_ledger(reservation_id, resource, route, holder, event, at, detail) "
            "VALUES(?,?,?,?,?,?,?)",
            (reservation_id, resource, route, holder, event, self.clock(),
             json.dumps(detail, sort_keys=True)),
        )
