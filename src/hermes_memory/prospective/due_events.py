"""C8 due events: a durable handoff of responsibility, never a claim of delivery.

The old behaviour this replaces marked a reminder "delivered" as soon as a worker
picked it up. That is a statement about a transport nobody had touched. Here a
claim says *someone is holding it*, an acknowledgment says *responsibility has
moved into a durable decision*, and only the outbox — which is not in this file —
can later say anything about delivery.

The lease is the interesting part. A claim whose lease lapses becomes
``uncertain``, not ``pending``: the holder may have handed it off and died before
writing that down, and re-queueing it would produce a second artifact for one
intention.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ..ids import digest, new_id, now, timestamp
from ..storage.evidence import EvidenceError

__all__ = ["DueEventLog", "DueEvent", "Claim", "PENDING", "CLAIMED", "HANDED_OFF",
           "SUPPRESSED", "CANCELLED", "UNCERTAIN", "DECISION_KINDS"]

PENDING = "pending"
CLAIMED = "claimed"
HANDED_OFF = "handed_off"
SUPPRESSED = "suppressed"
CANCELLED = "cancelled"
UNCERTAIN = "uncertain"
TERMINAL = {HANDED_OFF, SUPPRESSED, CANCELLED}

DECISION_KINDS = ("silent", "next_turn", "digest", "notify_owner", "draft",
                  "awaiting_analysis")
DEFAULT_LEASE_S = 120.0


@dataclass(frozen=True)
class Claim:
    """One held event and the token that proves who holds it.

    The token is handed back here and nowhere else: it is not stored on the
    event's public shape, so a listing of due events cannot leak the capability
    to speak for them.
    """

    event: DueEvent
    token: str

    def as_dict(self) -> dict[str, Any]:
        return {**self.event.as_dict()}


@dataclass(frozen=True)
class DueEvent:
    id: str
    goal_id: str
    revision: int
    fire_at: str
    reason: str
    timezone: str
    precision: str
    state: str
    claimed_by: str | None
    claim_until: float | None
    created_at: str
    decided_at: str | None

    @classmethod
    def from_row(cls, row) -> "DueEvent":
        return cls(id=row["id"], goal_id=row["goal_id"], revision=row["revision"],
                   fire_at=row["fire_at"], reason=row["reason"], timezone=row["timezone"],
                   precision=row["precision"], state=row["state"],
                   claimed_by=row["claimed_by"], claim_until=row["claim_until"],
                   created_at=row["created_at"], decided_at=row["decided_at"])

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "goal": self.goal_id, "revision": self.revision,
                "fire_at": self.fire_at, "reason": self.reason, "state": self.state,
                "claimed_by": self.claimed_by, "claim_until": self.claim_until,
                "decided_at": self.decided_at, "timezone": self.timezone,
                "precision": self.precision}


class DueEventLog:
    def __init__(self, store, *, clock=None):
        self.store = store
        self.db = store.db
        self._clock = clock

    def _now(self) -> float:
        return time.time() if self._clock is None else float(self._clock())

    @contextmanager
    def _writing(self, db: sqlite3.Connection | None):
        """Run in the caller's transaction, or in one of our own.

        The handoff is two statements — move the event, create the intent — and
        the crash window that used to lose a reminder sat exactly between them.
        """
        connection = db or self.db
        if db is not None and not db.in_transaction:
            raise EvidenceError("this write needs an ambient transaction")
        owns = db is None
        if owns:
            connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            if owns:
                connection.execute("ROLLBACK")
            raise
        else:
            if owns:
                connection.execute("COMMIT")

    # -- scheduling ----------------------------------------------------------

    def schedule(self, *, goal_id: str, revision: int, fire_at: str, timezone: str,
                 precision: str, reason: str = "due",
                 db: sqlite3.Connection | None = None) -> dict[str, Any]:
        """Queue one occurrence. Idempotent per (goal, revision, instant, reason).

        A new revision is a new identity, which is what lets an uncertain send be
        left uncertain: the fix is a different event, not the same one retried.
        """
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise EvidenceError("a due event belongs to a numbered goal revision")
        moment = timestamp(fire_at)
        event_id = "due_" + digest([goal_id, revision, moment, reason])[:32]
        with self._writing(db) as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO due_events(id, goal_id, revision, fire_at, reason, "
                "timezone, precision, state, created_at) VALUES(?,?,?,?,?,?,?, 'pending', ?)",
                (event_id, goal_id, revision, moment, reason, timezone, precision, now()))
        return {"id": event_id, "scheduled": int(cursor.rowcount or 0) > 0}

    def cancel(self, goal_id: str, *, revision: int | None = None, reason: str,
               db: sqlite3.Connection | None = None) -> int:
        """Take back everything not yet handed off. A committed handoff stays."""
        clause = "goal_id=?" + (" AND revision=?" if revision else "")
        params = [goal_id] + ([revision] if revision else [])
        with self._writing(db) as connection:
            cursor = connection.execute(
                f"UPDATE due_events SET state=?, decided_at=?, claim_token=NULL, "
                f"claim_until=NULL, claimed_by=NULL WHERE {clause} AND state IN (?, ?)",
                [CANCELLED, now(), *params, PENDING, CLAIMED])
            connection.execute(
                "INSERT INTO audit(action, object_id, created_at, metadata) "
                "VALUES('due_event_cancelled', ?, ?, ?)",
                (goal_id, now(), f'{{"count": {int(cursor.rowcount or 0)}, '
                                 f'"reason": "{reason[:120]}"}}'))
        return int(cursor.rowcount or 0)

    # -- claims --------------------------------------------------------------

    def due(self, now_iso: str, *, limit: int = 25) -> list[DueEvent]:
        """Events whose time has come on goals that are still asking for them."""
        rows = self.db.execute(
            """
            SELECT e.* FROM due_events e JOIN goals g ON g.id = e.goal_id
            WHERE e.state = 'pending' AND e.fire_at <= ? AND g.status = 'active'
              AND (g.snoozed_until IS NULL OR g.snoozed_until <= ?)
            ORDER BY e.fire_at, e.id LIMIT ?
            """, (timestamp(now_iso), timestamp(now_iso), _bounded(limit))).fetchall()
        return [DueEvent.from_row(row) for row in rows]

    def claim(self, event_id: str, *, holder: str, lease_s: float = DEFAULT_LEASE_S,
              at: float | None = None) -> Claim | None:
        """Take responsibility for one event, for a while.

        Only a fresh ``pending`` row can be claimed. ``uncertain`` cannot: there
        is no way to tell a lost handoff from a completed one from this side of
        the fence, and a second holder would make the difference a duplicate.
        """
        if not isinstance(holder, str) or not holder.strip():
            raise EvidenceError("a claim names who holds it")
        if not isinstance(lease_s, (int, float)) or not 1 <= lease_s <= 3600:
            raise EvidenceError("lease_s must be between 1 and 3600 seconds")
        moment = self._now() if at is None else float(at)
        self.expire_claims(at=moment)
        token = new_id("claim")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.db.execute(
                "UPDATE due_events SET state=?, claim_token=?, claim_until=?, claimed_by=? "
                "WHERE id=? AND state=?", (CLAIMED, token, moment + float(lease_s), holder,
                                           event_id, PENDING))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        if not int(cursor.rowcount or 0):
            return None
        row = self.db.execute("SELECT * FROM due_events WHERE id=?", (event_id,)).fetchone()
        return Claim(event=DueEvent.from_row(row), token=token)

    def release(self, *, event_id: str, token: str, reason: str = "") -> dict[str, Any]:
        """Give an unacknowledged claim back. Only the holder may do this."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.db.execute(
                "UPDATE due_events SET state=?, claim_token=NULL, claim_until=NULL, "
                "claimed_by=NULL WHERE id=? AND claim_token=? AND state=?",
                (PENDING, event_id, token, CLAIMED))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        if not int(cursor.rowcount or 0):
            raise EvidenceError(
                f"no live claim {token!r} on {event_id!r} to release; it was handed off, "
                "cancelled, or taken over when its lease expired")
        return {"id": event_id, "state": PENDING, "reason": reason[:200]}

    def expire_claims(self, *, at: float | None = None) -> list[str]:
        """A lapsed lease is uncertain, never free."""
        moment = self._now() if at is None else float(at)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            rows = self.db.execute(
                "SELECT id, claimed_by FROM due_events WHERE state=? AND claim_until < ?",
                (CLAIMED, moment)).fetchall()
            for row in rows:
                self.db.execute(
                    "UPDATE due_events SET state=?, decided_at=? WHERE id=? AND state=?",
                    (UNCERTAIN, now(), row["id"], CLAIMED))
                self.store._audit("due_claim_expired", row["id"],
                                  {"holder": row["claimed_by"]})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return [row["id"] for row in rows]

    # -- handoff -------------------------------------------------------------

    def ack(self, *, event_id: str, token: str, decision: str, policy_version: str,
            payload_digest: str | None = None,
            db: sqlite3.Connection | None = None) -> dict[str, Any]:
        """Move responsibility into a durable decision, in one transaction.

        The event and the intent are written together or not at all: the crash
        window that used to lose a reminder sat exactly between those two writes.
        Replaying the same acknowledgment returns the intent that already exists
        instead of creating a second artifact for one promise.
        """
        if decision not in DECISION_KINDS:
            raise EvidenceError(
                f"decision {decision!r} is not one of {DECISION_KINDS}")
        if not isinstance(policy_version, str) or not policy_version.strip():
            raise EvidenceError("a decision names the policy version that made it")
        with self._writing(db) as connection:
            return self._ack(connection, event_id, token, decision, policy_version,
                             payload_digest)

    def _ack(self, connection, event_id, token, decision, policy_version, payload_digest):
        row = connection.execute("SELECT * FROM due_events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise EvidenceError(f"unknown due event {event_id!r}")
        intent_id = "dec_" + digest([event_id, row["revision"], decision,
                                     policy_version])[:32]
        existing = connection.execute("SELECT * FROM decision_intents WHERE id=?",
                                      (intent_id,)).fetchone()
        if existing is not None:
            return {"intent": existing["id"], "state": existing["state"],
                    "replayed": True, "event": event_id,
                    "note": "responsibility for this event already moved; no second "
                            "artifact was created"}
        if row["state"] == UNCERTAIN:
            raise EvidenceError(
                f"{event_id} is uncertain: its holder's lease lapsed before it said what "
                "it decided. Acknowledging it now would be a guess about a send")
        if row["state"] != CLAIMED:
            raise EvidenceError(
                f"{event_id} is {row['state']}; only a live claim can hand off")
        if row["claim_token"] != token:
            raise EvidenceError(
                "this claim belongs to someone else, or expired and was re-taken")
        stamp = now()
        state = "awaiting_analysis" if decision == "awaiting_analysis" else (
            "suppressed" if decision == "silent" else "prepared")
        connection.execute(
            "UPDATE due_events SET state=?, decided_at=?, decision_kind=?, policy_version=?, "
            "claim_token=NULL, claim_until=NULL WHERE id=? AND state=?",
            (HANDED_OFF, stamp, decision, policy_version, event_id, CLAIMED))
        connection.execute(
            "INSERT INTO decision_intents(id, event_id, goal_id, revision, kind, "
            "policy_version, state, payload_digest, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (intent_id, event_id, row["goal_id"], row["revision"], decision, policy_version,
             state, payload_digest, stamp, stamp))
        return {"intent": intent_id, "state": state, "replayed": False, "event": event_id}

    def intents(self, *, state: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        clause = " WHERE state=?" if state else ""
        rows = self.db.execute(
            f"SELECT * FROM decision_intents{clause} ORDER BY created_at, id LIMIT ?",
            ([state] if state else []) + [_bounded(limit)]).fetchall()
        return [dict(row) for row in rows]

    def get(self, event_id: str) -> DueEvent | None:
        row = self.db.execute("SELECT * FROM due_events WHERE id=?", (event_id,)).fetchone()
        return DueEvent.from_row(row) if row else None

    def status(self) -> dict[str, int]:
        rows = self.db.execute("SELECT state, count(*) AS n FROM due_events "
                               "GROUP BY state").fetchall()
        return {row["state"]: int(row["n"]) for row in rows}


def _bounded(value: int) -> int:
    return value if isinstance(value, int) and 1 <= value <= 200 else 20
