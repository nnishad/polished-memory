"""C8 predicates: the whole vocabulary of conditions the framework will evaluate.

A condition is data, not code. That limit is the point: nothing here can express
a rule the framework has not agreed to check, and a goal cannot quietly grow into
"ping me when things look bad".

Every evaluation answers in three states, not two. ``unknown`` is the one that
matters: an inbox we have not read since Tuesday cannot tell us that nobody
replied, and reporting that as ``failed`` turns an absence of coverage into a
statement about the world.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any

from ..ids import timestamp
from ..storage.evidence import EvidenceError

__all__ = ["KINDS", "PRECISIONS", "Verdict", "evaluate", "validate", "resolve_wall_time",
           "UNKNOWN", "SATISFIED", "FAILED", "PENDING"]

KINDS = ("due_at", "new_message_from", "source_item_update", "measured_threshold",
         "waiting_for")

SATISFIED = "satisfied"
FAILED = "failed"
PENDING = "pending"
UNKNOWN = "unknown"

# A connector that has not reported since this long is not evidence of silence.
COVERAGE_STALE_AFTER_S = 6 * 3600
PRECISIONS = ("minute", "hour", "day", "week", "month", "year")
_PERIODS = ("hour", "day", "week", "month", "year")
OPERATORS = ("<", "<=", ">", ">=", "==")


@dataclass(frozen=True)
class Verdict:
    state: str
    detail: str = ""
    evidence_record_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state, "detail": self.detail,
                "evidence_record_id": self.evidence_record_id}


def validate(kind: str, params: Any) -> str:
    """Check the shape at write time, so a bad condition never reaches an evaluation."""
    if kind not in KINDS:
        raise EvidenceError(f"unknown predicate {kind!r}; the admissible set is {KINDS}")
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            raise EvidenceError("predicate params must be a JSON object") from None
    if not isinstance(params, dict):
        raise EvidenceError("predicate params must be an object")
    required = {"due_at": ("at",), "new_message_from": ("account_id", "since"),
                "source_item_update": ("source", "source_id", "since"),
                "measured_threshold": ("subject", "predicate", "operator", "value", "unit"),
                "waiting_for": ("goal_id",)}[kind]
    missing = [name for name in required if name not in params]
    if missing:
        raise EvidenceError(f"{kind} is missing params {missing}")
    if kind == "due_at":
        _moment(str(params["at"]))
    if kind == "measured_threshold":
        if params["operator"] not in OPERATORS:
            raise EvidenceError(f"operator must be one of {OPERATORS}")
        try:
            float(params["value"])
        except (TypeError, ValueError):
            raise EvidenceError("measured_threshold needs a numeric value") from None
    for key in ("since",):
        if key in params:
            _moment(str(params[key]))
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def evaluate(db, kind: str, params: Any, *, now_iso: str) -> Verdict:
    """One condition, answered honestly. Never raises for a data problem."""
    if isinstance(params, str):
        params = json.loads(params)
    moment = timestamp(now_iso)
    if kind == "due_at":
        at = timestamp(str(params["at"]))
        return Verdict(SATISFIED if moment >= at else PENDING,
                       f"due {at}" if moment < at else f"{at} has passed")
    if kind == "new_message_from":
        return _new_message(db, params, moment)
    if kind == "source_item_update":
        return _source_update(db, params, moment)
    if kind == "measured_threshold":
        return _threshold(db, params, moment)
    if kind == "waiting_for":
        return _waiting(db, params)
    return Verdict(UNKNOWN, f"no evaluator for {kind!r}")


def _moment(value: str) -> str:
    try:
        return timestamp(value)
    except ValueError as error:
        raise EvidenceError(f"{value!r} is not an ISO instant: {error}") from None


def _new_message(db, params, moment: str) -> Verdict:
    account = str(params["account_id"])
    since = timestamp(str(params["since"]))
    if not account.startswith("acct_"):
        return Verdict(UNKNOWN, "new_message_from needs a confirmed account id, not "
                                "an address")
    row = db.execute(
        """
        -- Typed author matching only for the existence check. Whether that message is
        -- retrievable by this caller is decided at delivery by the same identity
        -- rules every other read goes through; here we only answer "did something
        -- arrive", and a hidden record did arrive.
        SELECT r.id FROM records r
        WHERE r.deleted = 0 AND r.ingested_at > ?
          AND (json_extract(r.metadata, '$.author_account_id') = ?
               OR json_extract(r.metadata, '$.sender_account_id') = ?
               OR EXISTS (SELECT 1 FROM identity_accounts a
                          WHERE a.id = ? AND a.identifier IN (
                            json_extract(r.metadata, '$.author'),
                            json_extract(r.metadata, '$.sender'),
                            json_extract(r.metadata, '$.from.address'))))
        ORDER BY r.ingested_at DESC LIMIT 1
        """, (since, account, account, account)).fetchone()
    if row:
        return Verdict(SATISFIED, "a message authored by that account was ingested", row["id"])
    gap = _coverage_gap(db, since, moment)
    if gap:
        return Verdict(UNKNOWN, gap)
    return Verdict(FAILED, f"no message from that account since {since}")


def _source_update(db, params, moment: str) -> Verdict:
    since = timestamp(str(params["since"]))
    row = db.execute(
        "SELECT id FROM records WHERE source=? AND source_id=? AND deleted=0 "
        "AND ingested_at > ? ORDER BY ingested_at DESC LIMIT 1",
        (str(params["source"]), str(params["source_id"]), since)).fetchone()
    if row:
        return Verdict(SATISFIED, "the source item was updated", row["id"])
    gap = _coverage_gap(db, since, moment, source=str(params["source"]))
    if gap:
        return Verdict(UNKNOWN, gap)
    return Verdict(FAILED, "no newer revision of that source item")


def _threshold(db, params, moment: str) -> Verdict:
    row = db.execute(
        "SELECT a.id, a.value, a.unit FROM assertions a JOIN records r ON r.id=a.record_id "
        "WHERE a.subject=? AND a.predicate=? AND a.category='measurement' "
        "AND a.status='confirmed' AND r.deleted=0 "
        "AND NOT EXISTS (SELECT 1 FROM record_visibility v WHERE v.record_id=r.id AND v.hidden=1) "
        "AND (a.valid_from IS NULL OR a.valid_from<=?) AND (a.valid_to IS NULL OR a.valid_to>=?) "
        "AND substr(r.text,a.quote_start+1,a.quote_end-a.quote_start)=a.quote "
        "ORDER BY a.confirmed_at DESC",
        (str(params["subject"]), str(params["predicate"]), moment, moment) ).fetchall()
    if not row:
        return Verdict(UNKNOWN, "nothing has been measured for that subject")
    if len({(item["value"], item["unit"]) for item in row}) > 1:
        return Verdict(UNKNOWN, "current measurements conflict; threshold cannot be established")
    row = row[0]
    if str(row["unit"] or "") != str(params["unit"]):
        return Verdict(UNKNOWN,
                       f"the only measurement is in {row['unit']!r}, not {params['unit']!r}")
    try:
        actual = float(row["value"])
    except (TypeError, ValueError):
        return Verdict(UNKNOWN, "the stored measurement is not a number")
    wanted = float(params["value"])
    holds = {"<": actual < wanted, "<=": actual <= wanted, ">": actual > wanted,
             ">=": actual >= wanted, "==": actual == wanted}[str(params["operator"])]
    return Verdict(SATISFIED if holds else FAILED,
                   f"{actual} {params['operator']} {wanted} is {holds}")


def _waiting(db, params) -> Verdict:
    """One level only. A chain of waiting-for goals is evaluated as it is read."""
    row = db.execute("SELECT status, title FROM goals WHERE id=?",
                     (str(params["goal_id"]),)).fetchone()
    if row is None:
        return Verdict(UNKNOWN, "the goal being waited on does not exist")
    if row["status"] == "completed":
        return Verdict(SATISFIED, f"{row['title']} completed")
    if row["status"] in ("cancelled", "expired"):
        return Verdict(FAILED, f"{row['title']} was {row['status']}")
    if row["status"] == "candidate":
        return Verdict(UNKNOWN, "the goal being waited on is not active yet")
    return Verdict(PENDING, f"{row['title']} is still {row['status']}")


def _coverage_gap(db, since: str, moment: str, *, source: str | None = None) -> str | None:
    """Say so when the silence is ours rather than the world's."""
    row = db.execute(
        "SELECT source, last_success_at, coverage_state FROM connectors "
        + ("WHERE source=?" if source else "WHERE last_success_at IS NOT NULL")
        + " ORDER BY last_success_at LIMIT 1", (source,) if source else ()).fetchone()
    if row is None:
        return "no source has ever reported coverage, so an absence cannot be concluded"
    if str(row["coverage_state"]) != "complete":
        return (f"coverage of {row['source']} is {row['coverage_state']}, so nothing "
                "arriving is not evidence that nothing arrived")
    last = row["last_success_at"]
    if last is None or str(last) < since:
        return (f"{row['source']} was last read at {last}, before the window opened; "
                "a gap is not an absence")
    try:
        age = (_parse(moment) - _parse(str(last))).total_seconds()
    except ValueError:
        return f"coverage of {row['source']} could not be timed"
    if age > COVERAGE_STALE_AFTER_S:
        return (f"{row['source']} has not reported in {int(age // 60)} minutes, so a reply "
                "we cannot see is not a reply that did not happen")
    return None


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def resolve_wall_time(wall: str, zone_name: str, precision: str) -> datetime:
    """Turn '09:00 on Thursday, in this zone' into exactly one instant.

    Refuses the two cases a local clock cannot answer: a spring-forward gap that
    never existed, and an autumn repeated hour. Guessing either one schedules a
    reminder an hour out, and the owner has no way to see that it did.
    """
    try:
        zone = ZoneInfo(zone_name)
    except Exception as error:
        raise EvidenceError(f"unknown timezone {zone_name!r}: {str(error)[:120]}") from None
    try:
        naive = datetime.fromisoformat(str(wall).replace("Z", "+00:00"))
    except ValueError:
        raise EvidenceError(f"due time {wall!r} is not an ISO date or datetime") from None
    if naive.tzinfo is not None:
        raise EvidenceError(
            f"due time {wall!r} already carries an offset; give a wall time and a "
            "timezone, or an instant and no timezone, and not both")
    if precision not in PRECISIONS:
        raise EvidenceError(f"due precision must be one of {PRECISIONS}, not {precision!r}")
    moment = naive.replace(tzinfo=zone)
    if precision in _PERIODS:
        # A coarse promise is anchored at the start of its period, in local time.
        # In a zone with a midnight transition that hour can be missing or
        # repeated, which is why the ambiguity checks below still run on the
        # anchored instant and not on what the caller typed.
        moment = moment.replace(hour=0, minute=0, second=0, microsecond=0) \
            if precision != "hour" else moment.replace(minute=0, second=0, microsecond=0)
        if precision == "week":
            moment = moment - timedelta(days=moment.weekday())
        elif precision == "month":
            moment = moment.replace(day=1)
        elif precision == "year":
            moment = moment.replace(month=1, day=1)
    gap = _roundtrip(moment, zone)
    if gap:
        raise EvidenceError(gap)
    if moment.utcoffset() != moment.replace(fold=1).utcoffset():
        raise EvidenceError(
            f"{wall} in {zone_name} happens twice when the clocks go back; pick the "
            "instant you mean instead of the wall time")
    return moment


def _roundtrip(moment: datetime, zone: ZoneInfo) -> str | None:
    """A nonexistent wall time round-trips to a different one."""
    instantic = moment.astimezone(timezone.utc).astimezone(zone)
    if instantic.replace(tzinfo=None) != moment.replace(tzinfo=None):
        return (f"{moment.replace(tzinfo=None)} does not exist in {zone.key}: the clocks "
                f"skip to {instantic.replace(tzinfo=None)}; name the instant you mean")
    return None
