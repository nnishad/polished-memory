"""A standing, bounded permission to spend the machine's models without a fresh read.

Formation has always refused until somebody read the list and approved its digest. That is the
right rule for an installation trusted with someone's conversations, and it is also the rule
that stops a memory from ever forming anything while its owner sleeps — so the plan's shape of
an *owner-approved budget* for background work is this: grant it once, with a record cap, a
token cap and an expiry, and every pass under it is counted against those numbers in the same
instance ledger that already holds the holds.

Three things this deliberately is not.

* **Not a widening of the daily budget.** :mod:`processing.budgets` stays the hard ceiling, and
  a pass that cannot fit the day is refused by that ceiling even with an allowance outstanding.
  Two controls where one can overrule the other is how a budget becomes advice.
* **Not reachable by an agent.** Granting and revoking require the owner principal, checked the
  way identity and goal decisions check it. A memory that could authorize its own inference
  would never have to ask.
* **Not open-ended.** An expiry is mandatory. The failure guarded against here is not one bad
  pass; it is a machine still spending a GPU on a decision nobody remembers making.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Callable

from ..ids import digest

__all__ = ["AllowanceError", "Allowances", "STAGE", "ANY_RESOURCE", "MAX_DURATION_S",
           "MAX_RECORDS"]

STAGE = "formation"
STAGES = frozenset({"formation", "consolidation", "synthesis", "assertions", "media"})
# The grant names a resource ("remote-9b") or this, meaning whatever the retain route points
# at today. A wildcard is allowed because a route's upstream changes more often than an
# owner's willingness does — and because a *moved* endpoint must be re-decided by hand.
ANY_RESOURCE = "*"
MAX_DURATION_S = 7 * 86_400
MAX_RECORDS = 10_000
INSTANCE_SCOPE = "global"


class AllowanceError(ValueError):
    """The grant is not one this ledger may record, or not one that covers this pass."""


def _instant(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _epoch(value: str) -> float:
    """An absolute instant, or a refusal. A naive one is a different instant per machine."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("an allowance instant must carry a timezone")
    return parsed.timestamp()


class Allowances:
    """The owner's standing decisions about bounded inference, and their accounting."""

    def __init__(self, store, *, owner_principal: str | None = None,
                 clock: Callable[[], float] = time.time, scope: str = INSTANCE_SCOPE,
                 stage: str = STAGE):
        # A reading has no owner in the room — `status` reports who holds the models without
        # naming anybody — so the principal is not required to build one. It is required to
        # grant, and an unset principal matches no actor, which fails closed rather than open.
        self.store = store
        self.db = store.db
        self.owner_principal = owner_principal.strip() if isinstance(owner_principal, str) \
            else None
        self.clock = clock
        self.scope = scope
        if stage not in STAGES:
            raise AllowanceError("unsupported inference allowance stage")
        self.stage = stage

    # -- the grant -----------------------------------------------------------

    def grant(self, *, actor: str, reason: str, records: int, tokens: int,
              duration_s: float | None = None, expires_at: str | None = None,
              resource: str = ANY_RESOURCE) -> dict[str, Any]:
        """Authorize bounded passes until the cap or the clock, whichever comes first.

        Exactly one live allowance may cover a (scope, stage, resource); a second grant does
        not add to the first, it is refused, because two standing permissions for the same
        device would have to be summed to be read and nobody sums a permission.
        """
        self._require_owner(actor)
        _checked_reason(reason)
        if not isinstance(records, int) or isinstance(records, bool) or not 1 <= records \
                <= MAX_RECORDS:
            raise AllowanceError(f"records must be between 1 and {MAX_RECORDS}")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
            raise AllowanceError("tokens must be a positive integer; a zero budget is no "
                                 "permission, and `owner --grant-allowance` is not how a "
                                 "resource gets disabled")
        if not isinstance(resource, str) or not resource.strip():
            raise AllowanceError("resource must name a device or '*' for whatever is "
                                 "configured now")
        moment = self.clock()
        end = self._expiry(moment=moment, duration_s=duration_s, expires_at=expires_at)
        # Retire on the way in rather than letting the unique index answer: a grant whose
        # expiry has passed is no permission, but its row still says `active`, and the owner
        # deserves to be told which grant was replaced rather than to read a constraint name.
        self.retire_expired()
        wanted = resource.strip()
        # A wildcard grant would sit over any live named one and be shadowed by it, so the
        # question a `*` grant has to answer is whether anything is already granted at all.
        live = self.current() if wanted == ANY_RESOURCE else self.current(resource=wanted)
        if live is not None:
            raise AllowanceError(
                f"allowance {live['id']} already covers {self.scope}/{self.stage}/"
                f"{live['resource']} until {live['expires_at']}; revoke it first — a second "
                "grant for one device would have to be added up to be read, and a permission "
                "that needs arithmetic is not a permission")
        identifier = "alw_" + digest([self.scope, self.stage, wanted, actor.strip(),
                                      reason, _instant(moment), records, tokens,
                                      _instant(end)])[:24]
        self.db.execute("BEGIN IMMEDIATE")
        try:
            # The wildcard conflict check must also run under the write lock;
            # the per-resource unique index alone cannot arbitrate '*' vs named.
            if self.current(resource=None if wanted == ANY_RESOURCE else wanted) is not None:
                raise AllowanceError("a live allowance already covers this stage/resource")
            self.db.execute(
                "INSERT INTO allowances(id, scope, stage, resource, actor, reason, "
                "granted_at, expires_at, max_records, token_budget, state) "
                "VALUES(?,?,?,?,?,?,?,?,?,?, 'active')",
                (identifier, self.scope, self.stage, wanted, actor.strip(), reason,
                 _instant(moment), _instant(end), records, tokens))
            self.db.execute("COMMIT")
        except sqlite3.IntegrityError:
            # Two processes granting at the same instant: the index is the arbiter, and the
            # row it collided with is the answer the owner should get.
            self.db.execute("ROLLBACK")
            live = self.current(resource=wanted)
            if live is None:
                raise AllowanceError(f"a grant for {wanted!r} is already live in this ledger")
            raise AllowanceError(f"allowance {live['id']} for {live['resource']} was recorded "
                                 "first; it is the standing permission this device has")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.state(identifier)

    def revoke(self, identifier: str, *, actor: str, reason: str) -> dict[str, Any]:
        """Withdraw the standing permission. What has already been spent stays spent."""
        self._require_owner(actor)
        _checked_reason(reason)
        row = self._row(identifier)
        if row is None:
            raise AllowanceError(f"no allowance {identifier!r} in this ledger")
        if row["state"] != "active":
            raise AllowanceError(f"allowance {identifier!r} is already {row['state']}; "
                                 "revoking it again would report an act that happened twice")
        self.db.execute("UPDATE allowances SET state='revoked' WHERE id=?", (identifier,))
        return self.state(identifier)

    # -- the reading a pass needs -------------------------------------------

    def current(self, *, resource: str | None = None) -> dict[str, Any] | None:
        """The live allowance for this scope and stage, or None.

        This reads and writes nothing: a status report that had to mark a row expired would
        need a writable ledger, and the codebase's readings are not allowed to become writers.
        An expired row is therefore *answered* as no allowance here, and
        :meth:`retire_expired` moves the state on a ledger the caller may write to.

        At most one row can answer, so no precedence rule is applied: a `*` grant is refused
        while anything is live and a named grant is refused while a `*` or a grant for that
        same device is live, which is what makes "the live allowance for this device" a
        singular question rather than a sum.
        """
        rows = self.db.execute(
            "SELECT id, expires_at, resource FROM allowances WHERE scope=? AND stage=? "
            "AND state='active' ORDER BY granted_at", (self.scope, self.stage)).fetchall()
        moment = self.clock()
        wanted = resource.strip() if isinstance(resource, str) and resource.strip() else None
        chosen = None
        for row in rows:
            if _epoch(str(row["expires_at"])) <= moment:
                continue
            if wanted is not None and row["resource"] not in (wanted, ANY_RESOURCE):
                continue
            chosen = row
        return self.state(str(chosen["id"])) if chosen is not None else None

    def retire_expired(self) -> list[str]:
        """Move every past-expiry grant to `expired`, and say which ones.

        A row that reads as no permission but still says `active` in the ledger is how a
        machine ends up with a pile of permissions nobody can tell apart from live ones.

        The instant is compared by :func:`_epoch` rather than in SQL: ISO strings of different
        precision do not order the way the moments they name do, and a grant that would not
        retire on a string is one that would keep being offered as live.
        """
        rows = self.db.execute(
            "SELECT id, expires_at FROM allowances WHERE scope=? AND stage=? AND "
            "state='active'", (self.scope, self.stage)).fetchall()
        moment = self.clock()
        retired = [str(row["id"]) for row in rows if _epoch(str(row["expires_at"])) <= moment]
        for identifier in retired:
            self.db.execute("UPDATE allowances SET state='expired' WHERE id=?", (identifier,))
        return retired

    def authorize(self, *, records: int, tokens: int, resource: str,
                  allowance: str | None = None) -> tuple[dict[str, Any] | None, str]:
        """Can this pass run under a standing permission? Answer with the reason either way.

        The reason is the point: a refusal that only says "no" pushes the operator to the
        settings file, while one that says *which bound was reached* says what to grant next.

        ``allowance`` may be the literal ``*``, which means "whichever grant covers this
        device" — the same symbol a grant is named with, so a scheduled pass can be written
        once and survive the rotating ids of the decisions behind it.
        """
        if records < 1:
            return None, "nothing was selected, so there is no pass to authorize"
        if allowance is not None and allowance.strip() != ANY_RESOURCE:
            raw = self._row(allowance)
            if raw is None:
                return None, f"no allowance {allowance!r} is recorded in this ledger"
            if raw["state"] != "active":
                return None, f"allowance {raw['id']!r} is {raw['state']}"
            row = self.state(str(raw["id"]))
        else:
            row = self.current(resource=resource)
            if row is None:
                return None, (f"no active allowance covers {self.scope}/{self.stage}/{resource}; "
                              "run `hermes-memory owner --grant-allowance`, or approve a "
                              "pass one at a time with `form --review <digest>`")
        if row["resource"] not in (ANY_RESOURCE, resource):
            return None, (f"allowance {row['id']} was granted for "
                          f"{row['resource']!r}, not for {resource!r}; a different device is a "
                          "different decision")
        if _epoch(row["expires_at"]) <= self.clock():
            return None, f"allowance {row['id']} expired at {row['expires_at']}"
        left_records = row["records"]["left"]
        left_tokens = row["tokens"]["left"]
        if records > left_records:
            return None, (f"allowance {row['id']} has {left_records} of {row['records']['cap']} "
                          f"record(s) left and this pass would take {records}")
        if tokens > left_tokens:
            return None, (f"allowance {row['id']} has {left_tokens:,} of "
                          f"{row['tokens']['cap']:,} token(s) left and this pass would take "
                          f"{tokens:,}")
        return row, ""

    def consume(self, identifier: str, *, records: int, tokens: int) -> dict[str, Any]:
        """Write down what a pass actually used, and close the grant when it is used up.

        Tokens are what the pass measured, not what it estimated, for the same reason the
        daily ledger charges measured usage: a cap that only ever sees estimates is a cap on
        an optimist's guess.
        """
        if records < 0 or tokens < 0:
            raise AllowanceError("consumption cannot be negative")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self._row(identifier)
            if row is None:
                raise AllowanceError(f"no allowance {identifier!r} to charge")
            used_records = int(row["used_records"]) + int(records)
            used_tokens = int(row["used_tokens"]) + int(tokens)
            exhausted = (used_records >= int(row["max_records"])
                         or used_tokens >= int(row["token_budget"]))
            self.db.execute(
                "UPDATE allowances SET used_records=?, used_tokens=?, passes=passes+1, "
                "state=? WHERE id=?",
                (used_records, used_tokens, "spent" if exhausted else row["state"],
                 identifier))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.state(identifier)

    def ledger(self, *, include_closed: bool = False) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id FROM allowances WHERE scope=? AND stage=? ORDER BY granted_at",
            (self.scope, self.stage)).fetchall()
        out = [self.state(str(row["id"])) for row in rows]
        return out if include_closed else [entry for entry in out
                                           if entry["state"] == "active"]

    # -- internals -----------------------------------------------------------

    def state(self, identifier: str) -> dict[str, Any]:
        """What a grant covers, what it has cost and what is left — the numbers a refusal quotes.

        ``left`` is floored at zero: a grant charged past its cap has nothing left, and a
        negative balance reads as though spending more would put it back.
        """
        row = self._row(identifier)
        if row is None:
            raise AllowanceError(f"no allowance {identifier!r} in this ledger")
        records = {"used": int(row["used_records"]), "cap": int(row["max_records"])}
        records["left"] = max(0, records["cap"] - records["used"])
        tokens = {"used": int(row["used_tokens"]), "cap": int(row["token_budget"])}
        tokens["left"] = max(0, tokens["cap"] - tokens["used"])
        return {
            "id": row["id"], "scope": row["scope"], "stage": row["stage"],
            "resource": row["resource"], "actor": row["actor"], "reason": row["reason"],
            "granted_at": row["granted_at"], "expires_at": row["expires_at"],
            "records": records, "tokens": tokens,
            "passes": int(row["passes"]), "state": row["state"],
        }

    def _row(self, identifier: str) -> sqlite3.Row | None:
        if not isinstance(identifier, str) or not identifier.strip():
            raise AllowanceError("an allowance is named by its id")
        return self.db.execute("SELECT * FROM allowances WHERE id=? AND scope=? AND stage=?",
                               (identifier.strip(), self.scope, self.stage)).fetchone()

    def _expiry(self, *, moment: float, duration_s: float | None,
                expires_at: str | None) -> float:
        """When the grant ends — and it always ends."""
        if (duration_s is None) == (expires_at is None):
            raise AllowanceError("an allowance is bounded by --hours or --until, and by "
                                 "exactly one of them")
        if expires_at is not None:
            try:
                end = _epoch(expires_at)
            except (ValueError, TypeError):
                raise AllowanceError("--until must be an instant with a timezone") from None
        else:
            if not isinstance(duration_s, (int, float)) or isinstance(duration_s, bool):
                raise AllowanceError("--hours must be a number of hours")
            end = moment + float(duration_s) * 3600.0
        if end <= moment:
            raise AllowanceError("an allowance that has already expired authorizes nothing; "
                                 "give it a moment in the future or do not grant it")
        if end - moment > MAX_DURATION_S:
            raise AllowanceError(
                f"an allowance may run for at most {MAX_DURATION_S // 86_400} days; a longer "
                "permission is a configuration change, and should be re-read as one")
        return end

    def _require_owner(self, actor: str) -> None:
        if self.owner_principal is None:
            raise AllowanceError("no owner principal is configured, so a standing permission "
                                 "could not be attributed to anybody; set "
                                 "HERMES_MEMORY_OWNER_PRINCIPAL first")
        if not isinstance(actor, str) or actor.strip() != self.owner_principal:
            raise AllowanceError(
                f"only the owner may grant or revoke a standing permission to spend the "
                f"models, not {actor!r}")


def _checked_reason(reason: str) -> None:
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
        raise AllowanceError("a standing permission has to say why, in 1..500 characters")
