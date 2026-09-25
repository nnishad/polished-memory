"""C10 engine: the one caller that turns a due event into a deliverable artifact.

Policy, eligibility, the analyst and the outbox are each refuse-only on their own.
This is where they run in order, and the order matters: the handoff is recorded
before anything is decided, so a crash leaves an intention that says *this was
promised and not finished* rather than an event that looks unclaimed to everybody
who looks next.

A reminder needs no model. The goal already carries the sentence the owner asked to
be told, so with inference switched off the engine still delivers — it only stops
rewording. That is deliberate: turning memory inference off must not turn
proactivity off as a side effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..ids import timestamp
from ..storage.evidence import EvidenceError

__all__ = ["ProactiveEngine", "Sweep"]

# An intention in this state was handed over but never decided: the process that
# owned it died between the two writes.
AWAITING = "awaiting_analysis"


@dataclass(frozen=True)
class Sweep:
    decided: int = 0
    prepared: int = 0
    suppressed: int = 0
    analysed: int = 0
    refused: int = 0
    expired: int = 0
    recovered: int = 0
    notes: Sequence[str] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"decided": self.decided, "prepared": self.prepared,
                "suppressed": self.suppressed, "analysed": self.analysed,
                "refused": self.refused, "expired": self.expired,
                "recovered": self.recovered, "notes": list(self.notes)[:12]}


class ProactiveEngine:
    """Runs one pass of the proactive path. Safe to call on a timer, or not at all."""

    def __init__(self, store, *, events, policy, eligibility, outbox, analyst=None,
                 broker=None, holder: str = "proactive-worker", lease_s: float = 120.0):
        self.store = store
        self.db = store.db
        self.events = events
        self.policy = policy
        self.eligibility = eligibility
        self.outbox = outbox
        self.analyst = analyst
        self.broker = broker
        self.holder = holder
        if not isinstance(lease_s, (int, float)) or not 5 <= lease_s <= 3600:
            raise EvidenceError("lease_s must be between 5 and 3600 seconds")
        self.lease_s = float(lease_s)

    def sweep(self, *, at: str | None = None, limit: int = 25,
              topic_for=None) -> dict[str, Any]:
        """One pass: expire, recover, finish, then take on what is due now."""
        moment = timestamp(at) if at else None
        notes: list[str] = []
        expired = self.outbox.expire(at=moment)
        recovered = self.outbox.recover(at=None if moment is None else _epoch(moment))
        finished = self._finish_awaiting(moment=moment, topic_for=topic_for,
                                         limit=limit, notes=notes)
        fresh = self._take_due(moment=moment, limit=limit, topic_for=topic_for,
                               notes=notes)
        outcomes = [self._one(goal_id, revision, event_id, intent_id, moment, topic_for,
                              notes)
                    for goal_id, revision, event_id, intent_id in finished + fresh]
        return Sweep(decided=sum(1 for item in outcomes if item["decided"]),
                     prepared=sum(1 for item in outcomes if item["prepared"]),
                     suppressed=sum(1 for item in outcomes if item["state"] == "suppressed"),
                     analysed=sum(1 for item in outcomes if item["analysed"]),
                     refused=sum(1 for item in outcomes if not item["decided"]),
                     expired=len(expired), recovered=len(recovered),
                     notes=tuple(notes)).as_dict()

    # -- the pass ------------------------------------------------------------

    def _take_due(self, *, moment, limit: int, topic_for, notes: list[str]) -> list[tuple]:
        found: list[tuple] = []
        if moment is None:
            notes.append("no instant was given, so nothing is due yet as far as this pass "
                         "is concerned")
            return found
        for event in self.events.due(moment, limit=limit):
            claim = self.events.claim(event.id, holder=self.holder, lease_s=self.lease_s)
            if claim is None:
                notes.append(f"{event.id} was taken by another worker")
                continue
            topic = (topic_for(event.as_dict()) if topic_for else None) or "general"
            # Hand off first. From here the event cannot be re-claimed, so a crash
            # leaves an unfinished intention rather than a second reminder.
            handed = self.events.ack(event_id=event.id, token=claim.token,
                                    decision=AWAITING, policy_version=self.policy_version())
            found.append((event.goal_id, event.revision, event.id, handed["intent"]))
        return found

    def _finish_awaiting(self, *, moment, topic_for, limit: int,
                         notes: list[str]) -> list[tuple]:
        """Find intentions a previous pass handed over but did not finish.

        They go through the same eligibility and policy reads as a fresh event,
        because the minutes in between are exactly when the owner may have completed
        the goal or opted the topic out.
        """
        rows = self.db.execute(
            "SELECT i.id, i.event_id, i.goal_id, i.revision FROM decision_intents i "
            "WHERE i.state=? ORDER BY i.created_at, i.id LIMIT ?",
            (AWAITING, _bounded(limit))).fetchall()
        found = [(str(row["goal_id"]), int(row["revision"]), str(row["event_id"]),
                  str(row["id"])) for row in rows]
        if found:
            notes.append(f"{len(found)} intention(s) were left unfinished by an earlier "
                         "pass and are being finished now")
        return found

    def _one(self, goal_id: str, revision: int, event_id: str, intent_id: str,
             moment, topic_for, notes: list[str]) -> dict[str, Any]:
        topic = ((topic_for({"goal": goal_id}) if topic_for else None) or "general")
        outcome = {"decided": False, "prepared": 0, "analysed": False,
                   "state": "refused", "intent": intent_id}
        verdict = self.eligibility.for_event(goal_id=goal_id, revision=revision,
                                            topic=topic, intent_id=intent_id, at=moment)
        if not verdict.eligible:
            self.events.suppress(event_id, reason=f"eligibility: {verdict.reason}")
            outcome["state"] = "suppressed"
            notes.append(f"{event_id}: {verdict.stage}: {verdict.reason}")
            return outcome
        decision = self.policy.decide(topic=topic, urgency="proactive", at=moment)
        goal = self.goal_text(goal_id)
        analysis = self._analyse(topic=topic, gate=decision.action, goal=goal)
        outcome["analysed"] = bool(analysis is not None and analysis.source == "analysed")
        if analysis is not None and analysis.outcome == "silent":
            # The model's no is honoured; its yes is only ever a demotion.
            self.events.suppress(event_id, reason=analysis.reason)
            outcome["state"] = "suppressed"
            notes.append(f"{event_id}: the analyst stayed silent: {analysis.reason}")
            return outcome
        # Decision, artifact and settlement in one transaction, with the slow part —
        # the model call — safely outside it. Split across three commits, a crash
        # between them used to leave a recorded decision with no artifact, and the
        # duplicate guard then made the reminder permanently undeliverable.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            recorded = self.policy.record(decision, intent_id=intent_id, goal_id=goal_id,
                                          revision=revision, db=self.db)
            prepared = self.outbox.prepare(
                decision_id=recorded["id"], kind=decision.action, topic=topic,
                payload=_text_of(analysis, goal), evidence=_citations(analysis),
                db=self.db)
            settled = self.events.settle(event_id=event_id, intent_id=intent_id,
                                         decision=str(prepared["kind"]),
                                         payload_digest=str(prepared["payload_digest"]),
                                         db=self.db)
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        outcome.update({"decided": bool(recorded["recorded"]),
                        "prepared": int(bool(prepared.get("prepared", True))),
                        "state": str(prepared["kind"])})
        if settled.get("replayed"):
            notes.append(f"{event_id}: this intention was already settled on "
                         f"{settled['kind']}")
        if not prepared.get("prepared", True):
            notes.append(f"{event_id}: the artifact already existed; nothing was queued "
                         "twice")
        return outcome

    def _analyse(self, *, topic: str, gate: str, goal: dict[str, Any]):
        if self.analyst is None or gate == "next_turn":
            return None
        packet = None
        if self.broker is not None and goal.get("statement"):
            packet = self.broker.assemble(str(goal["statement"]), limit=6)
        if packet is None:
            return None
        return self.analyst.analyse(packet=packet, topic=topic, gate=gate,
                                    question=str(goal.get("title") or "") or None)

    # -- small reads ---------------------------------------------------------

    def policy_version(self) -> str:
        from .policy import POLICY_VERSION
        return POLICY_VERSION

    def goal_text(self, goal_id: str) -> dict[str, Any]:
        row = self.db.execute("SELECT title, statement, status FROM goals WHERE id=?",
                              (goal_id,)).fetchone()
        return dict(row) if row else {}


def _text_of(analysis, goal: dict[str, Any]) -> str:
    if analysis is not None and analysis.worth_sending:
        return analysis.message
    # No model, or a model with nothing to add: the owner's own words are the
    # reminder. A rewrite is never required to keep a promise.
    return str(goal.get("title") or "Something you asked to be told about")


def _citations(analysis) -> list[str]:
    return list(analysis.citations) if analysis is not None else []


def _epoch(moment: str) -> float:
    from datetime import datetime, timezone
    return datetime.fromisoformat(str(moment)).astimezone(timezone.utc).timestamp()


def _bounded(value: int) -> int:
    return value if isinstance(value, int) and 1 <= value <= 200 else 20
