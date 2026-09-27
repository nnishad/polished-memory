"""C12 budgets: finite, measured, and enforced before dispatch rather than after.

Usage is charged from what the backend actually reported, not from an estimate
of what the prompt looked like, and a reservation is refused before it occupies
a physical slot. A budget that only stops work after it has been spent is a
meter, not a control.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..storage.evidence import EvidenceError

__all__ = ["Budgets", "BudgetExhausted", "Budget"]


class BudgetExhausted(Exception):
    def __init__(self, scope: str, resource: str, used: int, limit: int):
        super().__init__(
            f"{scope} budget for {resource} is spent: {used:,} of {limit:,} tokens")
        self.scope, self.resource, self.used, self.limit = scope, resource, used, limit


@dataclass(frozen=True)
class Budget:
    """Daily allowance for one resource, in tokens, with a call ceiling."""

    tokens: int
    calls: int = 2000
    seconds: float = 3600.0

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise EvidenceError("a token budget cannot be negative; use 0 to disable a resource")


class Budgets:
    def __init__(self, store, *, daily: dict[str, Budget], scope: str = "global",
                 clock=None):
        self.store = store
        self.db = store.db
        self.daily = dict(daily)
        self.scope = scope
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def period(self) -> str:
        return self.clock().strftime("%Y-%m-%d")

    def used(self, resource: str) -> dict[str, float]:
        row = self.db.execute(
            "SELECT tokens, calls, seconds FROM budget_usage WHERE scope=? AND period=? "
            "AND resource=?", (self.scope, self.period, resource)).fetchone()
        if row is None:
            return {"tokens": 0, "calls": 0, "seconds": 0.0}
        return {"tokens": row["tokens"], "calls": row["calls"], "seconds": row["seconds"]}

    def remaining(self, resource: str) -> int:
        budget = self.daily.get(resource)
        if budget is None:
            return 0
        return max(0, budget.tokens - int(self.used(resource)["tokens"]))

    def admit(self, resource: str, *, estimated_tokens: int) -> None:
        """Refuse dispatch that cannot fit. Zero allowance means disabled, not unlimited."""
        budget = self.daily.get(resource)
        if budget is None:
            raise BudgetExhausted(self.scope, resource, 0, 0)
        if budget.tokens == 0:
            raise BudgetExhausted(self.scope, resource, int(self.used(resource)["tokens"]), 0)
        if not 0 <= estimated_tokens <= budget.tokens:
            raise EvidenceError(
                f"estimated {estimated_tokens} tokens is outside 0..{budget.tokens} for "
                f"{resource}; a single request may not be larger than the whole day")
        spent = self.used(resource)
        if spent["tokens"] + estimated_tokens > budget.tokens:
            raise BudgetExhausted(self.scope, resource, int(spent["tokens"]), budget.tokens)
        if spent["calls"] >= budget.calls:
            raise BudgetExhausted(self.scope, resource, int(spent["calls"]), budget.calls)

    def charge(self, resource: str, *, tokens: int, seconds: float = 0.0,
               calls: int = 1) -> dict[str, Any]:
        """Record measured consumption, including internal retries."""
        if tokens < 0 or seconds < 0 or calls < 0:
            raise EvidenceError("usage cannot be negative")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "INSERT INTO budget_usage(scope, period, resource, tokens, calls, seconds) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(scope, period, resource) DO UPDATE SET "
                "tokens=budget_usage.tokens+excluded.tokens, "
                "calls=budget_usage.calls+excluded.calls, "
                "seconds=budget_usage.seconds+excluded.seconds",
                (self.scope, self.period, resource, tokens, calls, seconds))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.used(resource)

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {"period": self.period, "scope": self.scope, "resources": {}}
        for resource, budget in sorted(self.daily.items()):
            spent = self.used(resource)
            out["resources"][resource] = {
                "tokens_used": int(spent["tokens"]), "token_budget": budget.tokens,
                "calls_used": int(spent["calls"]), "call_budget": budget.calls,
                "seconds_used": round(spent["seconds"], 2),
                "enabled": budget.tokens > 0,
                "headroom": self.remaining(resource)}
        return out
