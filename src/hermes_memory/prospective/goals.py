"""C8 goals: durable intent with a version, an owner, and a clock.

Two rules do most of the work here. A candidate cannot create an obligation:
what an agent noticed as a possible task is not something the archive will remind
anybody about until the owner activates it. And every change is a new revision:
an acknowledgment of revision three must never be able to satisfy revision four,
so cancelling and re-scheduling happen on the same clock tick as the change.

Time is stored as an instant and understood as a wall time. "Call the plumber on
Monday" means Monday in the owner's zone, with the precision they said it at, and
a due time that the zone does not contain is refused rather than quietly shifted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from ..ids import digest, new_id, now, timestamp
from ..storage.evidence import EvidenceError, EvidenceStore
from . import predicates as P
from .due_events import DueEventLog

__all__ = ["GoalStore", "Goal", "ACTIVE", "CANDIDATE", "COMPLETED", "CANCELLED", "EXPIRED"]

CANDIDATE = "candidate"
ACTIVE = "active"
COMPLETED = "completed"
CANCELLED = "cancelled"
EXPIRED = "expired"
SETTLED = {COMPLETED, CANCELLED, EXPIRED}
MAX_TITLE = 200
MAX_STATEMENT = 4000


@dataclass(frozen=True)
class Goal:
    id: str
    title: str
    statement: str
    status: str
    revision: int
    timezone: str
    due_at: str | None
    due_precision: str
    owner_account: str | None
    created_by: str
    created_kind: str
    confirmed_by: str | None
    source_record_id: str | None
    snoozed_until: str | None
    expires_at: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row) -> "Goal":
        return cls(id=row["id"], title=row["title"], statement=row["statement"],
                   status=row["status"], revision=row["revision"], timezone=row["timezone"],
                   due_at=row["due_at"], due_precision=row["due_precision"],
                   owner_account=row["owner_account"], created_by=row["created_by"],
                   created_kind=row["created_kind"], confirmed_by=row["confirmed_by"],
                   source_record_id=row["source_record_id"],
                   snoozed_until=row["snoozed_until"], expires_at=row["expires_at"],
                   created_at=row["created_at"], updated_at=row["updated_at"])

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "statement": self.statement,
                "status": self.status, "revision": self.revision,
                "timezone": self.timezone, "due_at": self.due_at,
                "due_precision": self.due_precision, "owner_account": self.owner_account,
                "created_by": self.created_by, "created_kind": self.created_kind,
                "confirmed_by": self.confirmed_by, "snoozed_until": self.snoozed_until,
                "expires_at": self.expires_at, "source_record_id": self.source_record_id}


class GoalStore:
    """Goals, their conditions, and the due events they generate."""

    def __init__(self, store: EvidenceStore, *, events: DueEventLog | None = None,
                 owner_principal: str | None = None,
                 clock=lambda: datetime.now(timezone.utc)):
        self.store = store
        self.db = store.db
        self.events = events or DueEventLog(store)
        self.owner_principal = owner_principal
        self.clock = clock

    # -- authoring -----------------------------------------------------------

    def propose(self, *, title: str, statement: str, timezone_name: str = "UTC",
                due: str | None = None, due_precision: str = "minute",
                proposed_by: str, proposed_kind: str = "agent",
                conditions: Sequence[dict] = (), expires_at: str | None = None,
                owner_account: str | None = None,
                source_record_id: str | None = None,
                goal_id: str | None = None) -> dict[str, Any]:
        """Open a goal. An agent's proposal starts as a candidate and stays there.

        ``due`` is a wall time in ``timezone_name`` unless it arrives with an
        offset, in which case it is an instant and the zone only labels it.
        """
        title, statement = _text(title, "title", MAX_TITLE), _text(statement, "statement",
                                                                   MAX_STATEMENT)
        zone = _zone(timezone_name)
        instant, precision = _due(due, zone, due_precision)
        expiry = _instant(expires_at, zone) if expires_at else None
        if expiry and instant and expiry < instant:
            raise EvidenceError("the goal expires before it is due")
        if proposed_kind not in ("owner", "agent"):
            raise EvidenceError("proposed_kind is 'owner' or 'agent'")
        if source_record_id and not self.store.live_and_visible(source_record_id):
            raise EvidenceError("a goal must cite evidence that can still be read")
        if owner_account and self.db.execute(
                "SELECT 1 FROM identity_accounts WHERE id=? AND state='active'",
                (owner_account,)).fetchone() is None:
            raise EvidenceError(f"unknown owner account {owner_account!r}")
        checked = [_condition(item) for item in conditions]
        goal_id = goal_id or new_id("gol")
        stamp = now()
        status = ACTIVE if proposed_kind == "owner" else CANDIDATE

        self.db.execute("BEGIN IMMEDIATE")
        try:
            if self.db.execute("SELECT 1 FROM goals WHERE id=?", (goal_id,)).fetchone():
                raise EvidenceError(f"goal {goal_id!r} already exists")
            self.db.execute(
                "INSERT INTO goals(id, owner_account, title, statement, status, revision, "
                "timezone, due_at, due_precision, created_by, created_kind, confirmed_by, "
                "source_record_id, expires_at, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (goal_id, owner_account, title, statement, status, 1, zone, instant,
                 precision, proposed_by, proposed_kind,
                 proposed_by if status == ACTIVE else None, source_record_id, expiry,
                 stamp, stamp),
            )
            self._history(goal_id, revision=1, status=status, due_at=instant,
                          precision=precision, zone=zone, statement=statement,
                          by=proposed_by, reason="proposed")
            self._conditions(goal_id, 1, checked)
            if status == ACTIVE:
                self._schedule(goal_id, revision=1, instant=instant, zone=zone,
                               precision=precision, checked=checked)
            self.store._audit("goal_propose", goal_id,
                              {"status": status, "by": proposed_by, "due_at": instant,
                               "conditions": len(checked)})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": goal_id, "status": status, "revision": 1, "due_at": instant,
                "note": (None if status == ACTIVE else
                         "a proposal from outside the owner is a candidate: it schedules "
                         "nothing until the owner activates it")}

    def activate(self, *, goal_id: str, actor: str, reason: str) -> dict[str, Any]:
        """Owner-only. A candidate becomes an obligation with a real clock."""
        self._require_owner(actor)
        goal = self._row(goal_id)
        if goal["status"] != CANDIDATE:
            return {"id": goal_id, "status": goal["status"], "changed": False}
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE goals SET status=?, confirmed_by=?, updated_at=? WHERE id=?",
                (ACTIVE, actor, now(), goal_id))
            self._history(goal_id, revision=goal["revision"], status=ACTIVE,
                          due_at=goal["due_at"], precision=goal["due_precision"],
                          zone=goal["timezone"], statement=goal["statement"], by=actor,
                          reason=_text(reason, "reason", 1000))
            self._schedule(goal_id, revision=goal["revision"], instant=goal["due_at"],
                           zone=goal["timezone"], precision=goal["due_precision"],
                           checked=self._conditions_of(goal_id, goal["revision"]))
            self.store._audit("goal_activate", goal_id, {"actor": actor})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": goal_id, "status": ACTIVE, "changed": True,
                "revision": goal["revision"]}

    def complete(self, *, goal_id: str, actor: str, reason: str,
                 at: str | None = None) -> dict[str, Any]:
        return self._settle(goal_id, COMPLETED, actor, reason, at=at)

    def cancel(self, *, goal_id: str, actor: str, reason: str,
               at: str | None = None) -> dict[str, Any]:
        return self._settle(goal_id, CANCELLED, actor, reason, at=at)

    def _settle(self, goal_id: str, status: str, actor: str, reason: str,
                *, at: str | None) -> dict[str, Any]:
        """Finish or drop a goal, and take back everything it still had queued.

        The suppression is part of the same transaction as the decision: a crash
        between them is how a completed task keeps reminding people.
        """
        self._require_owner(actor)
        reason = _text(reason, "reason", 1000)
        goal = self._row(goal_id)
        if goal["status"] in SETTLED:
            return {"id": goal_id, "status": goal["status"], "changed": False,
                    "note": "already settled; nothing was re-decided"}
        moment = timestamp(at) if at else now()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE goals SET status=?, decided_at=?, decided_by=?, updated_at=? "
                "WHERE id=?", (status, moment, actor, moment, goal_id))
            self._history(goal_id, revision=goal["revision"], status=status,
                          due_at=goal["due_at"], precision=goal["due_precision"],
                          zone=goal["timezone"], statement=goal["statement"], by=actor,
                          reason=reason)
            suppressed = self.events.cancel(goal_id, reason=reason, db=self.db)
            self.store._audit(f"goal_{status}", goal_id,
                              {"actor": actor, "suppressed_events": suppressed})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": goal_id, "status": status, "changed": True,
                "suppressed_events": suppressed}

    def revise(self, *, goal_id: str, actor: str, reason: str, due: str | None = None,
               due_precision: str | None = None, statement: str | None = None,
               conditions: Sequence[dict] | None = None,
               timezone_name: str | None = None) -> dict[str, Any]:
        """One new revision. Never edits a due time in place.

        The revision number is what makes an in-flight reminder safe to ignore:
        a handoff against the old revision refers to an event that no longer
        describes the goal.
        """
        self._require_owner(actor)
        goal = self._row(goal_id)
        if goal["status"] in SETTLED:
            raise EvidenceError(
                f"goal {goal_id!r} is {goal['status']}; reopen it as a new goal rather "
                "than editing a finished one")
        zone = _zone(timezone_name) if timezone_name else goal["timezone"]
        precision = due_precision or goal["due_precision"]
        instant = None
        if due is not None:
            instant, precision = _due(due, zone, precision)
        elif due_precision:
            instant = _reanchor(goal["due_at"], zone, precision)
        else:
            instant = goal["due_at"]
        statement = goal["statement"] if statement is None else _text(statement, "statement",
                                                                     MAX_STATEMENT)
        checked = [_condition(item) for item in conditions] if conditions is not None \
            else self._conditions_of(goal_id, goal["revision"])
        revision = int(goal["revision"]) + 1

        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE goals SET revision=?, due_at=?, due_precision=?, timezone=?, "
                "statement=?, updated_at=? WHERE id=?",
                (revision, instant, precision, zone, statement, now(), goal_id))
            self._history(goal_id, revision=revision, status=goal["status"], due_at=instant,
                          precision=precision, zone=zone, statement=statement, by=actor,
                          reason=_text(reason, "reason", 1000))
            self._conditions(goal_id, revision, checked)
            self.events.cancel(goal_id, revision=goal["revision"], reason=reason,
                               db=self.db)
            if goal["status"] == ACTIVE:
                self._schedule(goal_id, revision=revision, instant=instant, zone=zone,
                               precision=precision, checked=checked)
            self.store._audit("goal_revise", goal_id,
                              {"actor": actor, "revision": revision, "due_at": instant})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": goal_id, "revision": revision, "due_at": instant, "status": goal["status"],
                "superseded_events": self.db.execute(
                    "SELECT count(*) FROM due_events WHERE goal_id=? AND state=? "
                    "AND revision=?", (goal_id, "cancelled", goal["revision"])).fetchone()[0]}

    def snooze(self, *, goal_id: str, actor: str, until: str, reason: str) -> dict[str, Any]:
        """Push the clock without creating an obligation revision.

        Nothing queued before ``until`` will surface while the snooze holds, and
        no already-handed-off artifact is recalled — that ship has literally
        sailed, and pretending otherwise would produce a second one.
        """
        self._require_owner(actor)
        goal = self._row(goal_id)
        moment = _instant(until, goal["timezone"])
        if goal["due_at"] and moment <= goal["due_at"] and goal["status"] == ACTIVE:
            raise EvidenceError(
                "this snooze ends before the goal is due, so it would suppress nothing; "
                "revise the due time instead")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("UPDATE goals SET snoozed_until=?, updated_at=? WHERE id=?",
                            (moment, now(), goal_id))
            self._history(goal_id, revision=goal["revision"], status=goal["status"],
                          due_at=goal["due_at"], precision=goal["due_precision"],
                          zone=goal["timezone"], statement=goal["statement"], by=actor,
                          reason=f"snoozed until {moment}: {_text(reason, 'reason', 500)}")
            self.store._audit("goal_snooze", goal_id, {"actor": actor, "until": moment})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": goal_id, "snoozed_until": moment}

    # -- reading -------------------------------------------------------------

    def get(self, goal_id: str) -> Goal | None:
        row = self.db.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
        return Goal.from_row(row) if row else None

    def open(self, *, now_iso: str | None = None, include_candidates: bool = False,
             limit: int = 25) -> list[Goal]:
        """Goals still live, with anything whose clock ran out marked expired."""
        moment = timestamp(now_iso) if now_iso else self._now()
        expired = self.db.execute(
            "UPDATE goals SET status=?, updated_at=? WHERE status=? AND expires_at IS NOT "
            "NULL AND expires_at <= ?", (EXPIRED, moment, ACTIVE, moment))
        rows = self.db.execute(
            "SELECT * FROM goals WHERE status IN ('active','candidate') "
            + ("" if include_candidates else "AND status='active'")
            + " ORDER BY CASE WHEN due_at IS NULL THEN 1 ELSE 0 END, due_at, id LIMIT ?",
            (_bounded(limit),)).fetchall()
        goals = [Goal.from_row(row) for row in rows]
        if int(expired.rowcount or 0):
            self.store._audit("goal_expired", "goals", {"count": int(expired.rowcount)})
        return goals

    def history(self, goal_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM goal_history WHERE goal_id=? "
                               "ORDER BY revision", (goal_id,)).fetchall()
        return [dict(row) for row in rows]

    def conditions(self, goal_id: str, *, revision: int | None = None,
                   now_iso: str | None = None) -> list[dict[str, Any]]:
        """Evaluate every condition of one revision, and store what was found."""
        goal = self._row(goal_id)
        wanted = revision or goal["revision"]
        rows = self.db.execute(
            "SELECT * FROM goal_predicates WHERE goal_id=? AND revision=? "
            "ORDER BY kind, id", (goal_id, wanted)).fetchall()
        moment = timestamp(now_iso) if now_iso else self._now()
        out = []
        for row in rows:
            verdict = P.evaluate(self.db, row["kind"], row["params"], now_iso=moment)
            self.db.execute(
                "UPDATE goal_predicates SET state=?, detail=?, evidence_record_id=?, "
                "evaluated_at=? WHERE id=?",
                (verdict.state, verdict.detail[:300], verdict.evidence_record_id, moment,
                 row["id"]))
            out.append({"kind": row["kind"], "params": json.loads(row["params"]),
                        **verdict.as_dict()})
        return out

    def due_now(self, now_iso: str | None = None, *, limit: int = 25) -> list[dict[str, Any]]:
        """What is due, with each event's conditions as part of the answer.

        An event whose conditions cannot be checked is reported as unknown rather
        than dropped or fired: the owner decides what an unanswerable "did nobody
        reply" is worth.
        """
        moment = timestamp(now_iso) if now_iso else self._now()
        out = []
        for event in self.events.due(moment, limit=limit):
            goal = self.get(event.goal_id)
            if goal is None or goal.snoozed_until and goal.snoozed_until > moment:
                continue
            checks = self.conditions(event.goal_id, revision=event.revision, now_iso=moment)
            out.append({**event.as_dict(), "title": goal.title, "conditions": checks,
                        "ready": all(item["state"] == P.SATISFIED for item in checks),
                        "unknown": [item["kind"] for item in checks
                                    if item["state"] == P.UNKNOWN]})
        return out

    # -- internals -----------------------------------------------------------

    def _schedule(self, goal_id, *, revision, instant, zone, precision,
                  checked: Sequence[dict]) -> None:
        if instant:
            self.events.schedule(goal_id=goal_id, revision=revision, fire_at=instant,
                                 timezone=zone, precision=precision, reason="due",
                                 db=self.db)
        for item in checked:
            if item["kind"] == "due_at":
                at = json.loads(item["params"])["at"]
                if at != instant:
                    self.events.schedule(goal_id=goal_id, revision=revision,
                                         fire_at=timestamp(at), timezone=zone,
                                         precision="minute", reason=f"due_at:{item['id']}",
                                         db=self.db)

    def _conditions(self, goal_id: str, revision: int,
                    checked: Sequence[dict[str, str]]) -> None:
        for item in checked:
            self.db.execute(
                "INSERT INTO goal_predicates(id, goal_id, revision, kind, params, state) "
                "VALUES(?,?,?,?,?, 'pending') ON CONFLICT(goal_id, revision, kind, params) "
                "DO NOTHING",
                (item["id"], goal_id, revision, item["kind"], item["params"]))

    def _conditions_of(self, goal_id: str, revision: int) -> list[dict[str, str]]:
        rows = self.db.execute("SELECT id, kind, params FROM goal_predicates WHERE goal_id=? "
                               "AND revision=?", (goal_id, revision)).fetchall()
        return [{"id": row["id"], "kind": row["kind"], "params": row["params"]}
                for row in rows]

    def _history(self, goal_id, *, revision, status, due_at, precision, zone, statement,
                 by, reason) -> None:
        self.db.execute(
            "INSERT INTO goal_history(goal_id, revision, status, due_at, "
            "due_precision, timezone, statement, changed_by, changed_at, reason) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (goal_id, revision, status, due_at, precision, zone, statement, by, now(),
             reason[:500]))

    def _row(self, goal_id: str):
        row = self.db.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
        if row is None:
            raise EvidenceError(f"unknown goal {goal_id!r}")
        return row

    def _require_owner(self, actor: str) -> None:
        if self.owner_principal is None:
            raise EvidenceError(
                "no owner principal is configured, so no goal can be activated or settled; "
                "set HERMES_MEMORY_OWNER_PRINCIPAL")
        if actor != self.owner_principal:
            raise EvidenceError(f"this decision belongs to the owner, not to {actor!r}")

    def _now(self) -> str:
        return self.clock().astimezone(timezone.utc).isoformat()


def _condition(item: dict) -> dict[str, str]:
    if not isinstance(item, dict):
        raise EvidenceError("each condition is {kind, params}")
    kind = item.get("kind")
    params = item.get("params") or {}
    if not isinstance(kind, str) or not isinstance(params, dict):
        raise EvidenceError("each condition is {kind, params}")
    normalized = P.validate(kind, params)
    return {"id": "cnd_" + digest([kind, normalized])[:24], "kind": kind,
            "params": normalized}


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise EvidenceError(f"{label} must be nonempty text of at most {maximum} characters")
    return " ".join(value.split())


def _zone(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        raise EvidenceError("timezone must be a named zone such as 'Europe/Amsterdam'")
    name = value.strip()
    try:
        P.resolve_wall_time("2026-01-01T00:00", name, "minute")
    except EvidenceError as error:
        raise EvidenceError(f"timezone {name!r} is not usable: {error}") from None
    return name


def _due(due: Any, zone: str, precision: str) -> tuple[str | None, str]:
    if due in (None, ""):
        return None, "none" if precision == "minute" else precision
    text = str(due)
    if _has_offset(text):
        if precision not in P.PRECISIONS:
            raise EvidenceError(f"due precision must be one of {P.PRECISIONS}")
        return timestamp(text), precision
    naive = text.replace("Z", "+00:00")
    if len(naive) == 10:  # a date is a day, whatever the caller guessed
        precision = "day"
    elif len(naive) == 7:
        precision = "month"
    moment = P.resolve_wall_time(naive, zone, precision)
    return moment.astimezone(timezone.utc).isoformat(), precision


def _instant(value: Any, zone: str) -> str:
    """An instant given directly, or a wall time in the zone the goal lives in."""
    text = str(value)
    if _has_offset(text):
        return timestamp(text)
    return P.resolve_wall_time(text, zone, "day" if len(text) == 10 else "minute") \
        .astimezone(timezone.utc).isoformat()


def _reanchor(value: str, zone: str, precision: str) -> str:
    """Re-express a stored instant in the zone it was meant for."""
    stored = timestamp(value)
    local = datetime.fromisoformat(stored)
    if precision in ("day", "week", "month", "year"):
        naive = local.strftime("%Y-%m-%dT%H:%M" if precision == "day" else "%Y-%m-%d")
        return P.resolve_wall_time(naive, zone, precision).astimezone(
            timezone.utc).isoformat()
    return stored


def _has_offset(text: str) -> bool:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        raise EvidenceError(f"due time {text!r} is not an ISO date or datetime") from None


def _bounded(value: int) -> int:
    return value if isinstance(value, int) and 1 <= value <= 200 else 20
