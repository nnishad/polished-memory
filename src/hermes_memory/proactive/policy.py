"""C10 attention policy: what may be said, when, and how often.

Defaults sit at the conservative end and every number in here is the owner's to
move: shadow mode on, one digest a day, at most two immediate notifications a day,
a quiet window, and a per-topic cooldown. The reasoning is deterministic on
purpose — a model decides *what* a change is worth saying, this decides whether it
gets said at all, and only the owner decides what either of them may spend.

Quiet hours are understood in the owner's zone, so a window that spans a DST
change is still one window of the intended length rather than an hour short or
long. ``decide`` only reads: it is called on every candidate moment, and a
decision that wrote nothing to disk would be indistinguishable from one that did.
The durable half of the story is ``record``, which runs once per intention.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from ..ids import digest, now, timestamp
from ..storage.evidence import EvidenceError

__all__ = ["AttentionPolicy", "Decision", "ACTIONS", "URGENCIES", "POLICY_VERSION",
           "DEFAULTS"]

ACTIONS = ("silent", "next_turn", "digest", "notify_owner", "draft")
URGENCIES = ("interactive", "freshness", "proactive", "maintenance")
POLICY_VERSION = "attention-1"
DEFAULTS: dict[str, Any] = {
    "shadow": True,
    "digest_per_day": 1,
    "max_immediate_per_day": 2,
    "cooldown_minutes": 240,
    "quiet_from": "21:00",
    "quiet_until": "07:30",
    "timezone": "UTC",
}
BOUNDS = {"digest_per_day": (0, 12), "max_immediate_per_day": (0, 10),
          "cooldown_minutes": (0, 24 * 60)}
# A slot search that walks this many local days has met a policy that cannot be
# satisfied; report it rather than returning a boundary nobody could deliver on.
MAX_SLOT_SEARCH = 7


@dataclass(frozen=True)
class Decision:
    """One gate outcome. ``downgrades`` is the story, not decoration: C14 has to
    answer "why was I not told?" without re-running the decision."""

    action: str
    reason: str
    topic: str = "general"
    shadow: bool = True
    at: str = ""
    policy_version: str = POLICY_VERSION
    digest_closes_at: str | None = None
    model_allowed: bool = False
    downgrades: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reason": self.reason, "topic": self.topic,
                "shadow": self.shadow, "at": self.at, "policy": self.policy_version,
                "digest_closes_at": self.digest_closes_at,
                "model_allowed": self.model_allowed, "downgrades": list(self.downgrades)}


class AttentionPolicy:
    """Deterministic gates between "something happened" and "Hermes may say so"."""

    def __init__(self, store, *, owner_principal: str | None = None):
        self.store = store
        self.db = store.db
        self.owner_principal = owner_principal

    # -- owner settings ------------------------------------------------------

    def configure(self, *, actor: str, topic: str = "general", shadow: bool | None = None,
                  digest_per_day: int | None = None, max_immediate_per_day: int | None = None,
                  cooldown_minutes: int | None = None, quiet_from: str | None = None,
                  quiet_until: str | None = None, timezone_name: str | None = None,
                  opted_out: bool | None = None) -> dict[str, Any]:
        """Owner-only. An agent that could widen its own attention budget is not gated."""
        if not isinstance(actor, str) or self.owner_principal is None or \
                actor != self.owner_principal:
            raise EvidenceError(
                "the attention policy belongs to the owner principal; nobody else may "
                "change it, and an agent asking for a wider budget is not the owner")
        name = _text(topic, "topic", 80)
        numbers = {"digest_per_day": digest_per_day,
                   "max_immediate_per_day": max_immediate_per_day,
                   "cooldown_minutes": cooldown_minutes}
        for key, value in numbers.items():
            low, high = BOUNDS[key]
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)
                                      or not low <= value <= high):
                raise EvidenceError(f"{key} must be an integer between {low} and {high}")
        zone = _zone(timezone_name or DEFAULTS["timezone"])
        current = self.settings(name)
        merged = dict(current)
        for key, value in numbers.items():
            if value is not None:
                merged[key] = value
        if shadow is not None:
            merged["shadow"] = bool(shadow)
        if quiet_from is not None:
            merged["quiet_from"] = _clock_time(quiet_from, "quiet_from")
        if quiet_until is not None:
            merged["quiet_until"] = _clock_time(quiet_until, "quiet_until")
        if timezone_name is not None:
            merged["timezone"] = zone.key
        if opted_out is not None:
            merged["state"] = "opted_out" if opted_out else "allowed"
        _validate_quiet_span(str(merged["quiet_from"]), str(merged["quiet_until"]))

        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                """
                INSERT INTO topic_policies(topic, state, shadow, quiet_from, quiet_until,
                    timezone, digest_per_day, max_immediate, cooldown_minutes, updated_at,
                    updated_by)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(topic) DO UPDATE SET
                    state=excluded.state, shadow=excluded.shadow,
                    quiet_from=excluded.quiet_from, quiet_until=excluded.quiet_until,
                    timezone=excluded.timezone, digest_per_day=excluded.digest_per_day,
                    max_immediate=excluded.max_immediate,
                    cooldown_minutes=excluded.cooldown_minutes,
                    updated_at=excluded.updated_at, updated_by=excluded.updated_by
                """,
                (name, merged["state"], int(merged["shadow"]), merged["quiet_from"],
                 merged["quiet_until"], merged["timezone"], merged["digest_per_day"],
                 merged["max_immediate_per_day"], merged["cooldown_minutes"], now(), actor))
            self.store._audit("attention_configure", name, {
                "actor": actor, "shadow": merged["shadow"],
                "digest_per_day": merged["digest_per_day"],
                "max_immediate": merged["max_immediate_per_day"],
                "cooldown_minutes": merged["cooldown_minutes"], "state": merged["state"]})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"topic": name, **merged}

    def settings(self, topic: str = "general") -> dict[str, Any]:
        """The effective policy: stored rows win, everything else is the default."""
        merged = dict(DEFAULTS)
        merged["state"] = "allowed"
        row = self.db.execute("SELECT * FROM topic_policies WHERE topic=?",
                              (topic,)).fetchone()
        if row is None:
            return merged
        merged.update({"shadow": bool(row["shadow"]), "state": row["state"],
                       "quiet_from": row["quiet_from"] or DEFAULTS["quiet_from"],
                       "quiet_until": row["quiet_until"] or DEFAULTS["quiet_until"],
                       "timezone": row["timezone"],
                       "digest_per_day": int(row["digest_per_day"]),
                       "max_immediate_per_day": int(row["max_immediate"]),
                       "cooldown_minutes": int(row["cooldown_minutes"])})
        return merged

    def next_waking(self, topic: str, at: Any) -> str | None:
        """When this moment's quiet hours end, if it is inside them at all.

        Delivery asks this before handing anything over: a notification that
        arrives at 02:00 has not informed the owner, it has woken them.
        """
        settings = self.settings(topic)
        zone = _zone(str(settings["timezone"]))
        moment = _parse(at)
        if not _in_quiet(moment, zone, str(settings["quiet_from"]),
                         str(settings["quiet_until"])):
            return None
        return _quiet_bounds(_parse(moment).astimezone(zone), str(settings["quiet_from"]),
                             str(settings["quiet_until"]))[1].isoformat()

    def topics(self) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT topic FROM topic_policies ORDER BY topic").fetchall()
        return [self.settings(row["topic"]) | {"topic": row["topic"]} for row in rows]

    # -- the decision --------------------------------------------------------

    def decide(self, *, topic: str = "general", urgency: str = "proactive",
               at: str | None = None, already_silent: bool = False,
               source_connected: bool = True) -> Decision:
        """One action for one candidate moment. Never a judgement about truth.

        The ladder is one rung deep per urgency and every rung has a named reason
        for the rung below it: an interruption the owner has not bought falls back
        to the digest, and a digest the owner has already spent falls back to
        "when they next ask". Nothing here is ever discarded on the way down.
        """
        moment = timestamp(at) if at else now()
        if urgency not in URGENCIES:
            raise EvidenceError(f"unknown urgency {urgency!r}; expected one of {URGENCIES}")
        settings = self.settings(topic)
        if settings["state"] == "opted_out":
            return Decision("silent", "the owner opted this topic out", topic=topic,
                            at=moment, model_allowed=False)
        if not source_connected:
            return Decision("silent", "the source is disconnected, so this cannot be "
                                      "re-checked before delivery and a stale alert is "
                                      "worse than a missing one", topic=topic, at=moment)
        if already_silent:
            return Decision("silent", "nothing in the change needed the owner",
                            topic=topic, at=moment)
        if self.store.stage_is_paused(f"topic:{topic}", "attention"):
            return Decision("silent", "an operator paused this topic's attention",
                            topic=topic, at=moment)
        if urgency == "maintenance":
            # Background work has nothing owner-facing to say; recording a decision
            # for it would fill the log with entries nobody asked about.
            return Decision("silent", "maintenance runs whether or not the owner is "
                                      "watching, so it asks for nothing",
                            topic=topic, at=moment, model_allowed=False)
        if urgency == "interactive":
            # The owner is already here; the answer belongs to this turn, and a
            # notification on top of it would be two messages about one thing.
            return Decision("next_turn", "an interactive turn already has the owner's "
                                         "attention", topic=topic, at=moment,
                            model_allowed=False)

        zone = _zone(str(settings["timezone"]))
        shadow = bool(settings["shadow"])
        downgrades: list[str] = []
        chosen = "notify_owner"
        closes: str | None = None
        if _in_quiet(moment, zone, str(settings["quiet_from"]),
                     str(settings["quiet_until"])):
            chosen = "digest"
            downgrades.append("notify_owner: quiet hours on the owner's own clock")
        else:
            spent, why = self._immediate_spent(topic, moment, zone, settings, shadow)
            if spent:
                chosen = "digest"
                downgrades.append(f"notify_owner: {why}")
        if chosen == "digest":
            closes = self._digest_slot(moment, zone, settings, topic)
            if closes is None:
                chosen = "next_turn"
                downgrades.append(
                    "digest: " + ("the owner disabled digests for this topic"
                                  if int(settings["digest_per_day"]) <= 0
                                  else "the digest windows this day are already spoken for"))
        reason = f"policy {POLICY_VERSION} chose {chosen}"
        if downgrades:
            reason += "; " + "; ".join(downgrades)
        return Decision(chosen, reason, topic=topic, shadow=shadow, at=moment,
                        digest_closes_at=closes, model_allowed=not shadow,
                        downgrades=tuple(downgrades))

    # -- the record ----------------------------------------------------------

    def record(self, decision: Decision, *, intent_id: str, goal_id: str, revision: int,
               packet_id: str | None = None, citations: Any = None,
               model_used: bool = False, db=None) -> dict[str, Any]:
        """Write one decision against the intention it answers.

        Idempotent by construction: the id comes from the intention, so a retry
        after a half-written transaction rewrites the same row and adds nothing to
        the digest window. That is the only thing standing between a crash and a
        double-counted day.
        """
        if decision.action not in ACTIONS:
            raise EvidenceError(f"unknown action {decision.action!r}")
        if not isinstance(intent_id, str) or not intent_id.strip():
            raise EvidenceError("a decision answers one intention")
        connection = db if db is not None else self.db
        if db is not None and not db.in_transaction:
            raise EvidenceError("recording a decision needs an ambient transaction")
        decision_id = "dec_" + digest(["proactive", intent_id])[:32]
        if decision.action == "digest" and not decision.digest_closes_at:
            raise EvidenceError("a digest decision must name the window it joins")
        citation_text = json.dumps(list(citations or []), ensure_ascii=False)[:4000]
        # Bucketed by the moment the gate was asked about, not by the wall clock of
        # whatever is recording it: a retried or batched decision belongs to the
        # owner's day it was meant for, or a backlog could spend a whole budget.
        stamp = decision.at or now()
        try:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO proactive_decisions(id, intent_id, goal_id, revision,
                    topic, window_id, action, reason, policy_version, shadow, model_used,
                    packet_id, citations, decided_at)
                VALUES(?,?,?,?,?,NULL,?,?,?,?,?,?,?,?)
                """,
                (decision_id, intent_id, goal_id, int(revision), decision.topic,
                 decision.action, decision.reason[:500], decision.policy_version,
                 int(decision.shadow), int(model_used), packet_id, citation_text, stamp))
        except sqlite3.IntegrityError as error:
            raise EvidenceError(
                f"this decision answers intention {intent_id} which does not exist, or "
                f"belongs to another goal: {str(error)[:120]}") from None
        recorded = int(cursor.rowcount or 0) > 0
        window_id = None
        if recorded and decision.action == "digest":
            # Only a decision that was actually written may occupy a window, or a
            # crash between the two halves of a retry would count the same item
            # twice against the day.
            window_id = self._open_window(decision, connection)
            connection.execute("UPDATE proactive_decisions SET window_id=? WHERE id=?",
                               (window_id, decision_id))
        if recorded:
            self.store._audit("proactive_decide", decision_id, {
                "intent": intent_id, "action": decision.action, "topic": decision.topic,
                "shadow": decision.shadow, "model_used": model_used,
                "window": window_id})
        return {"id": decision_id, "recorded": recorded, "window_id": window_id,
                "decision": decision.as_dict()}

    def counts(self, topic: str, *, at: str | None = None) -> dict[str, Any]:
        """What the owner's attention has already been asked for today."""
        moment = timestamp(at) if at else now()
        settings = self.settings(topic)
        zone = _zone(str(settings["timezone"]))
        shadow = bool(settings["shadow"])
        day_start, day_end = _local_day(moment, zone)
        immediate = self.db.execute(
            "SELECT count(*) FROM proactive_decisions WHERE topic=? AND action=? "
            "AND shadow=? AND decided_at>=? AND decided_at<?",
            (topic, "notify_owner", int(shadow), day_start, day_end)).fetchone()[0]
        digests = self.db.execute(
            "SELECT count(*) FROM attention_windows WHERE scope=? AND kind='digest' "
            "AND closes_at>=? AND closes_at<?",
            (f"topic:{topic}", day_start, day_end)).fetchone()[0]
        return {"immediate_today": int(immediate), "digests_closing_today": int(digests),
                "shadow": shadow, "local_day": _parse(moment).astimezone(zone)
                .date().isoformat()}

    def decisions(self, *, topic: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        clause = " WHERE topic=?" if topic else ""
        rows = self.db.execute(
            f"SELECT * FROM proactive_decisions{clause} ORDER BY decided_at DESC, id "
            "LIMIT ?", ([topic] if topic else []) + [_bounded(limit)]).fetchall()
        return [dict(row) for row in rows]

    # -- internals -----------------------------------------------------------

    def _immediate_spent(self, topic: str, moment: str, zone, settings,
                         shadow: bool) -> tuple[bool, str]:
        limit = int(settings["max_immediate_per_day"])
        day_start, day_end = _local_day(moment, zone)
        if limit <= 0:
            return True, "the owner allows no immediate notifications"
        used = int(self.db.execute(
            "SELECT count(*) FROM proactive_decisions WHERE topic=? AND action=? AND "
            "shadow=? AND decided_at>=? AND decided_at<?",
            (topic, "notify_owner", int(shadow), day_start, day_end)).fetchone()[0])
        if used >= limit:
            return True, f"the immediate budget for this day is spent ({used}/{limit})"
        minutes = int(settings["cooldown_minutes"])
        if minutes > 0:
            row = self.db.execute(
                "SELECT decided_at FROM proactive_decisions WHERE topic=? AND action=? "
                "AND shadow=? ORDER BY decided_at DESC, id LIMIT 1",
                (topic, "notify_owner", int(shadow))).fetchone()
            if row is not None:
                gap = (_parse(moment) - _parse(str(row["decided_at"]))).total_seconds() / 60
                if gap < minutes:
                    return True, (f"this topic spoke {int(gap)} minutes ago and waits "
                                  f"{minutes}")
        return False, ""

    def _digest_slot(self, moment: str, zone, settings, topic: str) -> str | None:
        """When the next digest that may hold this item closes.

        Slots are wall-clock boundaries in the owner's zone, and a boundary inside
        quiet hours slips to the end of quiet: a digest delivered at midnight is a
        digest that woke somebody. A day whose slots are all spoken for returns
        ``None``, because the windows are exactly what the owner capped.
        """
        per_day = int(settings["digest_per_day"])
        if per_day <= 0:
            return None
        quiet_from = str(settings["quiet_from"])
        quiet_until = str(settings["quiet_until"])
        local = _parse(moment).astimezone(zone)
        step = timedelta(days=1) / per_day
        day = local.replace(hour=0, minute=0, second=0, microsecond=0)
        for offset in range(MAX_SLOT_SEARCH):
            for index in range(1, per_day + 1):
                closes = day + timedelta(days=offset) + step * index
                if closes <= local:
                    continue
                if _in_quiet(closes, zone, quiet_from, quiet_until):
                    closes = _quiet_bounds(closes, quiet_from, quiet_until)[1] \
                        + timedelta(minutes=1)
                    if closes <= local:
                        continue
                iso = closes.astimezone(timezone.utc).isoformat()
                if not self._window_full(iso, settings, topic):
                    return iso
            day = day + timedelta(days=1)
        return None

    def _window_full(self, closes_at: str, settings, topic: str) -> bool:
        """May this topic still put an item in the window that closes then?

        Counted per topic, and by when the window closes rather than when the item
        was decided: a digest queued this evening and delivered tomorrow morning
        spends tomorrow's allowance, which is the one the owner actually sees.
        """
        scope = f"topic:{topic}"
        existing = self.db.execute("SELECT state FROM attention_windows WHERE scope=? "
                                   "AND kind='digest' AND closes_at=?",
                                   (scope, closes_at)).fetchone()
        if existing is not None:
            # A window that has already been delivered cannot take another item:
            # the owner has read that digest, and a late arrival in it would be
            # either missed or repeated.
            return existing["state"] != "open"
        day_start, day_end = _local_day(closes_at, _zone(str(settings["timezone"])))
        opened = int(self.db.execute(
            "SELECT count(*) FROM attention_windows WHERE scope=? AND kind='digest' "
            "AND closes_at>=? AND closes_at<?",
            (scope, day_start, day_end)).fetchone()[0])
        return opened >= int(settings["digest_per_day"])

    def _open_window(self, decision: Decision, connection) -> str:
        """Attach the item to its slot, creating the slot on first use."""
        closes_at = timestamp(str(decision.digest_closes_at))
        window_id = "win_" + digest(["digest", decision.topic, closes_at])[:32]
        connection.execute(
            "INSERT OR IGNORE INTO attention_windows(id, scope, kind, opens_at, closes_at, "
            "state, items) VALUES(?,?, 'digest', ?, ?, 'open', 0)",
            (window_id, f"topic:{decision.topic}", decision.at or now(), closes_at))
        connection.execute("UPDATE attention_windows SET items=items+1 WHERE id=?",
                           (window_id,))
        return window_id


def _in_quiet(value: Any, zone, quiet_from: str, quiet_until: str) -> bool:
    """Is this instant inside the owner's quiet window?"""
    local = _parse(value).astimezone(zone)
    start, end = _quiet_bounds(local, quiet_from, quiet_until)
    return start <= local < end


def _quiet_bounds(local: datetime, quiet_from: str, quiet_until: str) -> tuple[datetime, datetime]:
    """Tonight's window on the owner's wall clock, spanning midnight if it must."""
    start = _at(local, quiet_from)
    end = _at(local, quiet_until)
    if end <= start:
        end += timedelta(days=1)
    if local < start:
        # Before tonight's window: the one that can still cover this moment is
        # the window that opened yesterday evening.
        start -= timedelta(days=1)
        end -= timedelta(days=1)
    return start, end


def _local_day(moment: str, zone) -> tuple[str, str]:
    """The owner's calendar day containing *moment*, as UTC-bounded instants."""
    local = _parse(moment).astimezone(zone)
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return (start.astimezone(timezone.utc).isoformat(),
            end.astimezone(timezone.utc).isoformat())


def _at(anchor: datetime, clock: str) -> datetime:
    hours, minutes = (int(part) for part in clock.split(":", 1))
    return anchor.replace(hour=hours, minute=minutes, second=0, microsecond=0)


def _clock_time(value: Any, label: str) -> str:
    text = str(value or "").strip()
    parts = text.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise EvidenceError(f"{label} must be HH:MM, got {value!r}")
    hours, minutes = int(parts[0]), int(parts[1])
    if not 0 <= hours <= 23 or not 0 <= minutes <= 59:
        raise EvidenceError(f"{label} {value!r} is not a time of day")
    return f"{hours:02d}:{minutes:02d}"


def _validate_quiet_span(quiet_from: str, quiet_until: str) -> None:
    if quiet_from == quiet_until:
        raise EvidenceError(
            "quiet hours that start and end at the same time cover the whole day; "
            "use an opt-out if that is what you meant")


def _zone(value: Any) -> ZoneInfo:
    try:
        return ZoneInfo(str(value).strip())
    except Exception as error:
        raise EvidenceError(f"unknown timezone {value!r}: {str(error)[:120]}") from None


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise EvidenceError(f"{label} must be nonempty text of at most {maximum} characters")
    return " ".join(value.split())


def _parse(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise EvidenceError(f"{value!r} is not an ISO-8601 instant") from None
    if parsed.tzinfo is None:
        raise EvidenceError("attention decisions need a timezone-aware instant")
    return parsed


def _bounded(value: int) -> int:
    return value if isinstance(value, int) and 1 <= value <= 200 else 50
