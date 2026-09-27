"""C10 eligibility: the cheap deterministic no, before anything spends a model.

The retired system asked a model whether an event was interesting and then decided
whether it was allowed to ask. That order spends money on a foregone conclusion and
lets an inference decide whether to bother. Here every filter is a fact about the
store: the source's coverage, whether the evidence is still there, whether the goal
still wants to fire, what the owner's attention budget has left, and whether the
model is permitted to look at all.

Every refusal names the stage that refused it, because C14 has to answer "why was I
not told?" from the record rather than from a re-run. And every age is taken from a
date the record carries — an occurrence time, or failing that the moment its source
said the thing arrived — rather than from the moment this call happened to look.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..ids import timestamp
from ..prospective import predicates as P
from ..storage.evidence import EvidenceError

__all__ = ["Eligibility", "Verdict", "CONSUMER"]

# The proactive engine is an ordinary journal consumer, so duplicate suppression is
# the checkpoint machinery rather than a second ledger that can disagree with the
# first.
CONSUMER = "proactive"
# Older than this at trigger time is a historical import, not a live change, no
# matter what the connector's coverage says.
DEFAULT_MAX_AGE_DAYS = 14
COVERAGE_LIVE = "current"
# Where an adapter records that its source said the payload arrived. Undated
# evidence is aged by this instead of by the moment it happened to be noticed.
ARRIVAL_FIELD = "spooled_at"


@dataclass(frozen=True)
class Verdict:
    eligible: bool
    stage: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"eligible": self.eligible, "stage": self.stage, "reason": self.reason}


class Eligibility:
    """Deterministic filters between a committed change and any analysis of it."""

    def __init__(self, store, *, policy, sync=None, goals=None, budgets=None,
                 max_age_days: int = DEFAULT_MAX_AGE_DAYS,
                 model_budget_resource: str | None = None):
        self.store = store
        self.db = store.db
        self.policy = policy
        self.sync = sync
        self.goals = goals
        self.budgets = budgets
        self.model_budget_resource = model_budget_resource
        if not isinstance(max_age_days, int) or not 1 <= max_age_days <= 365:
            raise EvidenceError("max_age_days must be between 1 and 365")
        self.max_age = timedelta(days=max_age_days)

    # -- triggers ------------------------------------------------------------

    def for_change(self, *, source: str, record_id: str, topic: str = "general",
                   at: str | None = None) -> Verdict:
        """A committed live change: is it worth asking anything about?"""
        moment = timestamp(at) if at else None
        coverage = self._coverage(source)
        if not coverage.eligible:
            return coverage
        evidence = self.store.get(record_id)
        if evidence is None:
            return Verdict(False, "evidence",
                           "the record this change names is not in the store, so there is "
                           "nothing to describe")
        stale = self._stale(evidence, moment)
        if not stale.eligible:
            return stale
        decision = self.policy.decide(topic=topic, urgency="freshness", at=moment,
                                      source_connected=True)
        return self._gate(decision, topic)

    def for_event(self, *, goal_id: str, revision: int, topic: str = "general",
                  intent_id: str | None = None, at: str | None = None) -> Verdict:
        """A due event that C8 handed over: does anything still want it analysed?"""
        goal = self.db.execute("SELECT status, snoozed_until, expires_at FROM goals "
                               "WHERE id=? AND revision=?", (goal_id, int(revision))).fetchone()
        if goal is None:
            return Verdict(False, "goal",
                           f"revision {revision} of {goal_id} is not on file; a due event "
                           "for a version that does not exist cannot be analysed")
        if goal["status"] != "active":
            return Verdict(False, "goal", f"the goal is {goal['status']}, so its reminder is "
                                          "no longer something the owner asked for")
        moment = timestamp(at) if at else None
        # Both sides normalised before comparing: an offset-bearing string is not
        # orderable against one in another zone, and a snooze that reads as expired
        # because of its suffix would deliver something the owner put off.
        snoozed = timestamp(goal["snoozed_until"]) if goal["snoozed_until"] else None
        expiry = timestamp(goal["expires_at"]) if goal["expires_at"] else None
        if snoozed and moment and snoozed > moment:
            return Verdict(False, "snooze", f"the owner snoozed this goal until {snoozed}")
        if expiry and moment and expiry < moment:
            return Verdict(False, "expiry", "the goal expired before its event was reached")
        if intent_id is not None and self.db.execute(
                "SELECT 1 FROM proactive_decisions WHERE intent_id=?", (intent_id,)).fetchone():
            # Not a suppression: the intention already has its answer, and a second
            # analysis of one promise is how duplicates reach an inbox.
            return Verdict(False, "duplicate", "this intention already has a recorded decision")
        conditions = self._conditions(goal_id, int(revision), moment)
        if not conditions.eligible:
            return conditions
        decision = self.policy.decide(topic=topic, urgency="proactive", at=moment)
        return self._gate(decision, topic)

    # -- the model's own budget ----------------------------------------------

    def model_budget(self) -> Verdict:
        """Can a packet analysis be paid for at all?

        Checked separately from the attention gate because it fails for the whole
        system at once rather than per topic, and because an exhausted budget must
        not be reported as "nothing needed the owner".
        """
        if self.budgets is None or self.model_budget_resource is None:
            return Verdict(True, "budget", "no budget is configured, so nothing reserves one")
        try:
            self.budgets.admit(self.model_budget_resource, estimated_tokens=1)
        except Exception as error:
            return Verdict(False, "budget", f"the analysis budget is spent: {str(error)[:160]}")
        return Verdict(True, "budget", "the analysis budget has room")

    # -- internals -----------------------------------------------------------

    def _conditions(self, goal_id: str, revision: int, moment: str | None) -> Verdict:
        """Every condition of the promise, answered against the store before anything is said.

        "At nine o'clock" and "if nobody replies by Friday" are not the same kind of promise,
        and only the first one is settled by a clock. A predicate has to be evaluated at the
        moment the event fires, or a reminder goes out for a thing that already happened — or
        never could.

        The states split the way the rest of this module splits everything: *not now* holds
        the event, *not at all* consumes it. `pending` and `unknown` are both not-now — the
        first because the instant has not arrived, the second because the store cannot say,
        and an unanswerable question is not a licence to act — while `failed` is the durable
        no: the check was made against current coverage and the premise is simply not true,
        which is the case an owner would otherwise be reminded about forever.
        """
        if self.goals is None:
            return Verdict(True, "conditions", "no goal ledger is wired here, so a promise's "
                                               "conditions are checked where they are owned")
        checks = self.goals.conditions(goal_id, revision=revision, now_iso=moment)
        if not checks:
            return Verdict(True, "conditions", "this promise carries no conditions")
        failed = [item for item in checks if item["state"] == P.FAILED]
        if failed:
            return Verdict(False, "predicate",
                           f"the condition did not hold: {failed[0]['detail']}"[:400])
        waiting = [item for item in checks if item["state"] in (P.PENDING, P.UNKNOWN)]
        if waiting:
            kind = "cannot be checked" if waiting[0]["state"] == P.UNKNOWN else "not reached"
            return Verdict(False, "condition",
                           f"{waiting[0]['kind']} {kind}: {waiting[0]['detail']}"[:400])
        return Verdict(True, "conditions", "every condition of this promise is satisfied")

    def _gate(self, decision, topic: str) -> Verdict:
        """The attention gate is also the model's permission slip."""
        settings = self.policy.settings(topic)
        if settings["state"] == "opted_out":
            return Verdict(False, "opt_out", "the owner opted this topic out")
        if self.store.stage_is_paused(f"topic:{topic}", "attention"):
            # An operator pause is a *not now* with an expiry the operator controls: the
            # moment it is lifted, what was due is owed again. Under the generic policy
            # stage a caller reading the verdict cannot tell the two apart, and a no that
            # consumes the promise is a very expensive kind of caution.
            return Verdict(False, "paused", "an operator paused this topic's attention")
        if decision.action == "silent":
            return Verdict(False, "policy", f"the policy stayed silent: {decision.reason}")
        if decision.shadow:
            # The expensive question here is not "is this interesting?" but "is it
            # worth a call at all": a shadowed run has no audience, so analysing it
            # would spend the budget twice and deliver neither. This is a *not yet*,
            # and the engine treats it as one — an installation defaults to shadow, and
            # consuming its owner's reminders to prove a point nobody has tested yet is
            # data loss, not caution.
            return Verdict(False, "shadow", "shadow mode spends no model call on an "
                                            "undeliverable message")
        budget = self.model_budget()
        if not budget.eligible:
            return budget
        return Verdict(True, "allowed", "the change is live, on file, and inside policy")

    def _coverage(self, source: str) -> Verdict:
        if self.sync is None:
            return Verdict(True, "coverage", "no connector registry is wired, so coverage "
                                             "is asserted elsewhere rather than checked here")
        try:
            state = str(self.sync.state(source).get("coverage_state"))
        except EvidenceError:
            return Verdict(False, "coverage", f"source {source!r} is not registered, so its "
                                              "changes have no coverage claim to trust")
        if state != COVERAGE_LIVE:
            return Verdict(False, "coverage",
                           f"{source} reports coverage {state!r}; a backfill, a gap or an "
                           "unreachable source is not a live change, and a gap is not an "
                           "absence")
        return Verdict(True, "coverage", f"{source} is current")

    def _stale(self, evidence, moment: str | None) -> Verdict:
        """Age a change by the latest moment anything about it claims to be true.

        An undated arrival is the hole in 'this happened on ...': a capture spool
        replayed after three weeks carries no occurrence time to age, and would be
        read as news the moment it was noticed. When the source said the thing
        arrived, that is the date a live-versus-backfill verdict rests on, and it is
        in the record rather than in this call's clock.
        """
        when = evidence.occurred_at or _arrival_of(evidence.metadata)
        if when is None or moment is None:
            return Verdict(True, "freshness", "the change carries no occurrence time to age")
        then = _parse(when)
        now = _parse(moment)
        if now - then > self.max_age:
            return Verdict(False, "freshness",
                           f"this happened on {when[:10]}, which is older than the "
                           f"{self.max_age.days} days a late arrival can still be news for")
        return Verdict(True, "freshness", "the change is recent enough to be worth attention")


def _arrival_of(metadata: Any) -> str | None:
    """When the source says its own payload arrived, if it said so at all."""
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(ARRIVAL_FIELD)
    return value if isinstance(value, str) and value.strip() else None


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise EvidenceError("eligibility compares timezone-aware instants only")
    return parsed
