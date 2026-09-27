"""The background pass: the work that has to happen when nobody is asking.

Four things were built and left with no way to reach them. A due reminder was never taken,
because nothing constructed the proactive engine. A summary invalidated by a correction was
reported as stale and never queued for the refresh that would clear it. An identity edge whose
evidence had been forgotten stayed pending forever. And a job that had backed off was visible
in the queue and claimable at the same time, which is to say not claimable at all.

This module is the seam that runs them on a timer, or once from a shell.

**It never spends a model call.** That single rule is what lets it run unattended: `form`
requires the digest of a list somebody actually read, because dispatching inference costs
tokens, a device slot and a spend from a daily budget. Nothing here does any of those, and
the rule is structural rather than a promise about arguments: the proactive engine is built
without an analyst or a broker, so there is no model to reach even by mistake. A reminder is
delivered in the owner's own words with inference off — the engine deliberately works that
way, so switching memory inference off must not switch proactivity off as a side effect — and
every other section is either a local durable write or a read that ends in a report.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from ..storage.evidence import EvidenceError, EvidenceStore

__all__ = ["Maintenance", "Ticker", "DEFAULT_LIMIT", "MAX_LIMIT", "SECTIONS", "run"]

DEFAULT_LIMIT = 25
MAX_LIMIT = 200
SECTIONS = ("proactive", "summaries", "identity", "queue", "erasure")


class Maintenance:
    """One bounded pass over the durable bookkeeping, safe to call on a timer."""

    def __init__(self, store, *, owner_principal: str | None = None, sync=None,
                 gate=None, clock: Callable[[], float] = time.time):
        from ..backend.provenance import ProvenanceLedger
        from ..knowledge.summaries import SummaryStore
        from ..processing.jobs import JobQueue
        from ..proactive.eligibility import Eligibility
        from ..proactive.engine import ProactiveEngine
        from ..proactive.outbox import Outbox
        from ..proactive.policy import AttentionPolicy
        from ..prospective.due_events import DueEventLog
        from ..prospective.goals import GoalStore
        from ..storage.identity import IdentityStore

        self.store = store
        self.db = store.db
        self.owner_principal = owner_principal
        self.clock = clock
        self.gate = gate
        self.jobs = JobQueue(store, clock=clock)
        self.summaries = SummaryStore(store, ledger=ProvenanceLedger(store),
                                     owner_principal=owner_principal)
        self.identity = IdentityStore(store, owner_principal=owner_principal)
        # The proactive stack, composed in the order it runs: a claimed due event becomes a
        # durable intention, the intention becomes a decision, the decision becomes one
        # artifact. Each of those parts refuses on its own; the engine is where they agree.
        self.events = DueEventLog(store, clock=clock)
        self.goals = GoalStore(store, events=self.events, owner_principal=owner_principal)
        self.policy = AttentionPolicy(store, owner_principal=owner_principal)
        self.outbox = Outbox(store, policy=self.policy, owner_principal=owner_principal,
                            clock=clock, goals=self.goals, events=self.events)
        self.eligibility = Eligibility(store, policy=self.policy, sync=sync,
                                       goals=self.goals)
        self.engine = ProactiveEngine(store, events=self.events, policy=self.policy,
                                     eligibility=self.eligibility, outbox=self.outbox)

    # -- the pass ------------------------------------------------------------

    def pass_now(self, *, at: str | None = None, limit: int = DEFAULT_LIMIT,
                 sections: tuple[str, ...] = ()) -> dict[str, Any]:
        """Run the named sections, or all of them. Every section is bounded and idempotent.

        One instant is computed and passed down rather than read per section: a pass that
        started at 09:00 and noticed a reminder due at 08:59 must not disagree with the report
        it writes at 09:04 about what was due when.
        """
        wanted = tuple(sections or SECTIONS)
        unknown = [name for name in wanted if name not in SECTIONS]
        if unknown:
            raise EvidenceError(f"unknown maintenance section(s) {unknown}; this pass runs "
                                f"{list(SECTIONS)}")
        moment = self._moment(at)
        report: dict[str, Any] = {"at": moment, "sections": list(wanted),
                                  "inference_paused": bool(self.gate is not None
                                                           and self.gate.paused)}
        for name in wanted:
            try:
                report[name] = getattr(self, f"_{name}")(moment=moment,
                                                         limit=_bounded(limit))
            except Exception as error:
                # The sections run in order, so a raise halfway through means this pass would
                # record nothing at all — and a scheduler that is alive but failing every
                # period would report exactly like one that stopped. Name what broke, then let
                # it propagate: the caller still gets the exception, and the next healthy pass
                # writes a better line over it.
                report["failed_section"] = name
                report["error"] = f"{type(error).__name__}: {str(error)[:200]}"
                self._heartbeat(report)
                raise
        # The pass writes down that it happened. `status` and `doctor` run in other
        # processes, so the loop's own thread cannot be their evidence: without this line
        # an installation whose scheduler died would report exactly what one that is
        # healthy reports.
        self._heartbeat(report)
        return report

    def _heartbeat(self, report: dict[str, Any]) -> None:
        proactive = report.get("proactive") or {}
        summaries = report.get("summaries") or {}
        line = {"at": report["at"], "sections": report["sections"],
                "prepared": proactive.get("prepared"), "deferred": proactive.get("deferred"),
                "suppressed": proactive.get("suppressed"),
                "refreshes_promised": summaries.get("promised"),
                "inference_paused": report["inference_paused"]}
        if "failed_section" in report:
            # Absent on a pass that ran to the end. Recorded on one that did not, because
            # a failed pass and a skipped period otherwise look identical to every reading
            # that exists to notice the difference.
            line["failed_section"] = report["failed_section"]
            line["error"] = report["error"]
        self.store._audit("maintenance_pass", "maintenance", line)

    @classmethod
    def last_pass(cls, store) -> dict[str, Any] | None:
        """The most recent pass this archive recorded, or None when it never recorded one."""
        row = store.db.execute(
            "SELECT created_at, metadata FROM audit WHERE action=? ORDER BY created_at DESC, "
            "rowid DESC LIMIT 1", ("maintenance_pass",)).fetchone()
        if row is None:
            return None
        return {"created_at": str(row["created_at"]),
                **json.loads(str(row["metadata"] or "{}"))}

    # -- the sections --------------------------------------------------------

    def _proactive(self, *, moment: str, limit: int) -> dict[str, Any]:
        swept = self.engine.sweep(at=moment, limit=limit)
        swept["open_intents"] = len(self.events.intents(state="awaiting_analysis"))
        return swept

    def _summaries(self, *, moment: str, limit: int) -> dict[str, Any]:
        stale = self.summaries.needs_refresh(limit=limit)
        already = {(str(entry["scope"]), str(entry["kind"]))
                   for entry in self.summaries.pending_refreshes(limit=limit)}
        booked: list[str] = []
        for entry in stale:
            key = (str(entry["scope"]), str(entry["kind"]))
            if key in already:
                continue
            # The promise is the durable part: without it a correction stays a complaint in
            # a report. One window per scope, because a refresh that has not happened yet
            # covers whatever arrives while it waits — otherwise a pass on a timer would
            # book a new promise every few minutes and never be catchable.
            self.summaries.request_refresh(key[0], kind=key[1], through_at=moment)
            booked.append(key[0])
        return {"stale": len(stale), "promised": len(booked),
                "already_promised": len(stale) - len(booked),
                "pending": len(self.summaries.pending_refreshes(limit=limit)),
                "scopes": booked[:8]}

    def _identity(self, *, moment: str, limit: int) -> dict[str, Any]:
        dropped = self.identity.invalidate_stale()
        return {"candidates_expired": len(dropped["stale"]),
                "confirmed_needing_review": len(dropped["confirmed_needing_review"])}

    def _queue(self, *, moment: str, limit: int) -> dict[str, Any]:
        waiting = self.jobs.ready_for_retry(limit=limit)
        stuck = self.jobs.quarantine(limit=limit)
        # A job whose deadline passed is invisible in exactly the way a scheduler fears:
        # `claim()` refuses it, so it is neither worked nor waiting nor failed. It gets its
        # ending here, in the pass that runs when nobody is watching.
        reaped = self.jobs.reap_overdue(limit=limit)
        # A device slot is the same kind of debt as an overdue job. A holder that died
        # leaves the lease standing until somebody new asks for the resource, so on a
        # quiet installation an expired claim keeps its device occupied forever — and
        # `status` reports a worker that is gone as if it were still running.
        uncertain = list(self.gate.reap_expired()) if self.gate is not None else []
        return {"counts": self.jobs.counts(),
                "ready_for_retry": [item.id for item in waiting][:8],
                "quarantined": [item.id for item in stuck][:8],
                "overdue_reaped": reaped[:8],
                "leases_uncertain": len(uncertain),
                "uncertain_ids": uncertain[:8],
                "note": "a backed-off job is claimed by the next worker that looks; nothing "
                        "here spends anything to move it, and an expired lease becomes "
                        "uncertain rather than free"}

    def _erasure(self, *, moment: str, limit: int) -> dict[str, Any]:
        from ..lifecycle.erasure import ErasureManager

        manager = ErasureManager(self.store, owner_principal=self.owner_principal)
        pending = manager.pending(limit=limit)
        awaiting = manager.awaiting(limit=limit)
        return {"backend_owed": len(pending), "waiting_for_owner": len(awaiting),
                "oldest_owed": pending[0]["requested_at"] if pending else None,
                "note": "reported, not retried: an obligation to a backend nobody is "
                        "authorized to call stays owed and says so"}

    # -- internals -----------------------------------------------------------

    def _moment(self, at: str | None) -> str:
        """The instant this pass speaks for, in UTC.

        An explicit `at` is a test or a backfill saying "pretend it is then"; without one the
        pass uses the clock it was given, so a scheduled run and a manual one disagree only
        when they really are at different times.
        """
        if at is not None:
            parsed = datetime.fromisoformat(str(at).replace("z", "Z").replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("a maintenance instant must include a timezone")
            return parsed.astimezone(timezone.utc).isoformat()
        return datetime.fromtimestamp(self.clock(), tz=timezone.utc).isoformat()


class Ticker:
    """The pass on a period, in the process that owns the memory.

    Two rules hold it together. It never takes the process down: a pass that raises is
    counted, kept for whoever asks, and the next one is attempted on schedule, because a
    scheduler that dies quietly is worse than a reminder that arrives a quarter late. And
    it stops when told: the sleep is interruptible, so a unit being reloaded does not wait
    on a period it has already been asked to leave.
    """

    # A restart should not have to wait a full period to deliver what came due while the
    # process was down, and should not race the startup either.
    FIRST_DELAY_S = 15.0

    def __init__(self, *, runner: Callable[[], dict[str, Any]], interval_s: float,
                 first_delay_s: float | None = None):
        if not isinstance(interval_s, (int, float)) or not 0 <= float(interval_s) <= 86_400:
            raise EvidenceError("the maintenance interval must be between 0 and 86400 seconds")
        self.runner = runner
        self.interval_s = float(interval_s)
        self.first_delay_s = float(first_delay_s if first_delay_s is not None else
                                   min(Ticker.FIRST_DELAY_S, self.interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.passes = 0
        self.failures = 0
        self.last: dict[str, Any] = {}

    def start(self) -> "Ticker":
        """Begin the loop, or decline to: a zero interval is the owner's answer."""
        if self.interval_s <= 0:
            return self
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run,
                                        name="hermes-memory-maintenance", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        wait = self.first_delay_s
        while not self._stop.is_set():
            if self._stop.wait(wait):
                break
            self.tick()
            wait = self.interval_s

    def tick(self) -> dict[str, Any]:
        """One pass, kept for the report. Public so a test can drive it without a period."""
        try:
            report = self.runner()
        except Exception as error:
            self.failures += 1
            self.last = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
            return self.last
        self.passes += 1
        proactive = report.get("proactive") or {}
        self.last = {"ok": bool(report.get("ok")), "at": report.get("at"),
                     "prepared": proactive.get("prepared"),
                     "deferred": proactive.get("deferred"),
                     "suppressed": proactive.get("suppressed")}
        return self.last

    def stop(self, *, timeout: float = 5.0) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        return self.state()

    def state(self) -> dict[str, Any]:
        return {"interval_s": self.interval_s,
                "running": bool(self._thread is not None and self._thread.is_alive()),
                "passes": self.passes, "failures": self.failures, "last": self.last,
                "never": self.last == {} and self.interval_s <= 0}


def run(settings, *, limit: int = DEFAULT_LIMIT, at: str | None = None,
        sections: tuple[str, ...] = (), store: EvidenceStore | None = None) -> dict[str, Any]:
    """The door: one pass over the installation this settings object names."""
    from ..processing.instance_gate import writable_gate
    from ..sources.sync import SyncController

    if not store and not settings.db_path.exists():
        return {"ok": False, "refused": f"no canonical store at {settings.db_path}; run "
                                        "`hermes-memory init` or `hermes-memory setup` first"}
    opened = store or EvidenceStore(settings.db_path)
    # Not ``status_gate``: this pass reaps expired leases, which is a write to the
    # admission ledger, and a read-only connection would raise partway through — losing
    # the heartbeat and making a live-but-failing scheduler look dead.
    gate = writable_gate(settings)
    try:
        report = Maintenance(opened, owner_principal=settings.owner_principal,
                             sync=SyncController(opened), gate=gate).pass_now(
                                 at=at, limit=limit, sections=sections)
    finally:
        if store is None:
            opened.close()
    report["ok"] = True
    report["profile"] = getattr(settings, "profile", None)
    return report


def _bounded(value: int) -> int:
    return value if isinstance(value, int) and 1 <= value <= MAX_LIMIT else DEFAULT_LIMIT
