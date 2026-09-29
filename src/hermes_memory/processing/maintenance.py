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

One section does reach outside the process, and it is the exception rather than a hole in the
rule: a confirmed forgetting owed the derived backend a deletion, and the owner's confirmation
is the authorization for that call. It asks for no model, takes no slot and charges no budget —
it is a cleanup call made because somebody already agreed to the consequence.
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
SECTIONS = ("capture", "proactive", "questions", "summaries", "identity", "queue", "erasure")
#: Every question the pass opens lives under this one topic. Per-kind topics would give a
#: backlog of proposals one interruption budget each, and the owner's limit on being
#: interrupted is a limit on being interrupted, not on the kind of thing asked.
QUESTION_TOPIC = "decisions"


class Maintenance:
    """One bounded pass over the durable bookkeeping, safe to call on a timer."""

    def __init__(self, store, *, owner_principal: str | None = None, sync=None,
                 gate=None, clock: Callable[[], float] = time.time,
                 backend=None, backend_unavailable: str = ""):
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
        # The backend this pass may settle a debt with. It is handed in rather than built here
        # because the credential and the endpoint belong to the installation's configuration,
        # and a pass that could not reach one says why instead of pretending it had nothing owed.
        self.backend = backend
        self.backend_unavailable = backend_unavailable
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

    def _capture(self, *, moment: str, limit: int) -> dict[str, Any]:
        """Drain the host's capture spool into the record set. The consumer half of C2.

        The plugin writes every turn it is allowed to keep into a spool beside the profile's
        own store, and the adapter that turns those events into records takes a drain rather
        than a path on purpose. Nothing here is optional about that: an installation whose
        spool is never read has recorded every conversation it will ever show, in a file no
        component looks at, while its status answers that capture is operational.
        """
        from ..sources.capture_spool import (SPOOL_SOURCE, spool_backlog, spool_beside,
                                             spool_drain)
        from ..sources.sdk import HermesEvents
        from ..sources.runtime import ConnectorRuntime
        from ..sources.sync import SyncController

        spool = spool_beside(self.store.path)
        before = spool_backlog(spool)
        if not before["present"]:
            return {"spool": str(spool), "present": False, "records": 0, "pending": 0,
                    "note": "no capture spool for this profile; nothing to drain"}

        sync = SyncController(self.store, clock=self.clock)
        try:
            sync.state(SPOOL_SOURCE)
        except EvidenceError:
            # Registration is this component's to do, not the plugin's: the policy says these
            # events are read where they already are and never leave the machine.
            sync.register(SPOOL_SOURCE, policy_version="local-only")
        runtime = ConnectorRuntime(self.store, sync, holder=f"maintenance@{moment}",
                                   clock=self.clock)
        run = runtime.run(HermesEvents(spool_drain(spool), records_per_page=limit))
        after = spool_backlog(spool)
        return {"spool": str(spool), "present": True, "records": run.records,
                "repeats": run.repeats, "gaps": run.gaps, "pages": run.pages,
                "stopped": run.stopped, "cursor": run.cursor,
                "coverage_state": run.coverage_state, "pending": after["pending"],
                "pending_before": before["pending"],
                "note": run.note or "the spool is the source; the cursor is the position in it"}

    def _proactive(self, *, moment: str, limit: int) -> dict[str, Any]:
        swept = self.engine.sweep(at=moment, limit=limit)
        swept["open_intents"] = len(self.events.intents(state="awaiting_analysis"))
        return swept

    def _questions(self, *, moment: str, limit: int) -> dict[str, Any]:
        """Open a question for everything this archive is holding up on the owner.

        Asking is not sending. This section writes rows and touches no transport — the drain
        that owns the sink is what reaches the owner, on their budget and inside their quiet
        hours — so the pass keeps the rule that it spends no model call and opens no channel.
        Only what an agent proposed is asked about: an owner who typed a proposal themselves
        does not need to be asked whether they meant it.
        """
        from ..proactive.inquiries import InquiryStore

        inquiries = InquiryStore(self.store, owner_principal=self.owner_principal)
        expired = inquiries.expire_due()["expired"]
        allowed, who = inquiries.replies_allowed()
        if not allowed:
            return {"expired": expired, "asked": 0, "open": 0,
                    "note": f"nothing is asked while replies are switched off: {who}"}
        opened: list[str] = []
        declined: list[str] = []
        for decision, subject, question in _awaiting_owner(self.store, limit=limit):
            made = inquiries.ask(decision=decision, subject_id=subject, question=question,
                                 topic=QUESTION_TOPIC,
                                 reason="opened by the maintenance pass")
            if made.get("asked"):
                opened.append(f"{decision}:{subject}")
            elif made.get("note", "").startswith("asked again"):
                # It was void or expired and the state still awaits the owner, so the same
                # question goes out again rather than the row that recorded the failure
                # quietly ending the matter.
                opened.append(f"{decision}:{subject}")
            elif not made.get("ok", True) or "nothing is awaited" in str(made.get("reason")):
                declined.append(f"{decision}:{subject}")
        waiting = len(inquiries.list(states=("open", "sent")))
        return {"expired": expired, "asked": len(opened), "subjects": opened[:8],
                "stale": declined[:8], "open": waiting,
                "note": "the questions are queued, not sent: the delivery loop is what asks, "
                        "and the owner's daily limit is what decides when"}

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
        # A standing grant whose clock has passed authorizes nothing, but its row keeps saying
        # `active` until something writes it, and a quiet installation never does. Retiring it
        # is the same housekeeping as reaping a lease: a ledger whose rows disagree with the
        # clock is one nobody can read.
        grants = _retire_grants(self.gate, clock=self.clock)
        return {"counts": self.jobs.counts(),
                "ready_for_retry": [item.id for item in waiting][:8],
                "quarantined": [item.id for item in stuck][:8],
                "overdue_reaped": reaped[:8],
                "leases_uncertain": len(uncertain),
                "uncertain_ids": uncertain[:8],
                "grants_expired": grants[:8],
                "note": "a backed-off job is claimed by the next worker that looks; nothing "
                        "here spends anything to move it, and an expired lease becomes "
                        "uncertain rather than free"}

    def _erasure(self, *, moment: str, limit: int) -> dict[str, Any]:
        from ..lifecycle.erasure import ErasureManager

        manager = ErasureManager(self.store, owner_principal=self.owner_principal)
        owed_before = manager.pending(limit=limit)
        awaiting = manager.awaiting(limit=limit)
        if self.backend is None:
            return {"backend_owed": len(owed_before), "dispatched": 0, "cleared": 0,
                    "waiting_for_owner": len(awaiting),
                    "oldest_owed": owed_before[0]["requested_at"] if owed_before else None,
                    "note": self.backend_unavailable or (
                        "no backend client was handed to this pass, so a debt to the derived "
                        "copy is reported rather than paid")}
        outcome = manager.discharge(client=self.backend, limit=limit)
        owed_after = manager.pending(limit=limit)
        return {"backend_owed": len(owed_after), "dispatched": outcome["attempted"],
                "cleared": len(outcome["settled"]), "waiting_for_owner": len(awaiting),
                "oldest_owed": owed_after[0]["requested_at"] if owed_after else None,
                "still_owed": outcome["owed"][:5],
                "note": (f"{len(outcome['settled'])} obligation(s) dispatched and read back as "
                         f"gone, {len(outcome['owed'])} still answered for"
                         if outcome["attempted"] else "nothing was owed to the backend")}

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


def _maintenance_line(report: dict[str, Any]) -> dict[str, Any]:
    """What a reader of `state()` needs from a pass: that it ran, and what it did."""
    proactive = report.get("proactive") or {}
    return {"ok": bool(report.get("ok")), "at": report.get("at"),
            "prepared": proactive.get("prepared"),
            "deferred": proactive.get("deferred"),
            "suppressed": proactive.get("suppressed")}


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
                 first_delay_s: float | None = None, name: str = "hermes-memory-maintenance",
                 summarize: Callable[[dict[str, Any]], dict[str, Any]] | None = None):
        """A loop that is not the maintenance pass names itself and how to describe a run.

        ``name`` matters because a thread that lies in a stack trace is worse than no
        thread: a delivery loop shown as `maintenance` sends people looking at the wrong
        code. ``summarize`` is the same problem one level down — the line kept for whoever
        asks is shaped by what the run returns, and only the pass knows that.
        """
        if not isinstance(interval_s, (int, float)) or not 0 <= float(interval_s) <= 86_400:
            raise EvidenceError("the maintenance interval must be between 0 and 86400 seconds")
        self.runner = runner
        self.interval_s = float(interval_s)
        self.name = str(name)
        self._summarize = summarize or _maintenance_line
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
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
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
        self.last = self._summarize(report)
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
    backend, unavailable = _backend_for(settings)
    try:
        report = Maintenance(opened, owner_principal=settings.owner_principal,
                             sync=SyncController(opened), gate=gate, backend=backend,
                             backend_unavailable=unavailable).pass_now(
                                 at=at, limit=limit, sections=sections)
    finally:
        if store is None:
            opened.close()
    report["ok"] = True
    report["profile"] = getattr(settings, "profile", None)
    return report


def _backend_for(settings) -> tuple[Any, str]:
    """The client a pass may settle erasure debts with, or the reason it has none.

    An installation with no derived backend is the ordinary case rather than a fault, so this
    is a pair and not an exception: the report says which situation it is in.
    """
    from ..processing.formation import backend_client

    if not getattr(settings, "hindsight_url", ""):
        return None, "no derived backend is configured for this installation"
    try:
        return backend_client(settings), ""
    except Exception as error:
        return None, (f"the configured backend could not be addressed: "
                      f"{type(error).__name__}: {str(error)[:200]}")


def _bounded(value: int) -> int:
    return value if isinstance(value, int) and 1 <= value <= MAX_LIMIT else DEFAULT_LIMIT


def _retire_grants(gate, *, clock) -> list[str]:
    """Close the standing grants whose clock has passed, and name them.

    A reading answers an expired grant as no permission and must not write; this is the pass
    that does. It is asked against the pass's own clock, so a run speaking for one instant does
    not retire a grant against a different one. Nothing here sends a request, so the module's
    one rule holds.
    """
    import sqlite3

    from .allowance import Allowances

    if gate is None:
        return []
    try:
        return list(Allowances(gate.store, clock=clock).retire_expired())
    except sqlite3.OperationalError:
        # A gate database older than standing grants has none to retire.
        return []


def _awaiting_owner(store, *, limit: int) -> list[tuple[str, str, str]]:
    """``(decision, subject, question)`` for each state an agent put in front of the owner.

    Only what an *agent* proposed is asked about. An owner who typed a proposal at the door
    does not need to be asked by message whether they meant it, and a question that repeats a
    decision they made in person is noise that spends their budget.

    Oldest first and bounded per kind, so a week of proposals being read cannot push out the
    one candidate that has waited longest — and whatever already carries a live question is
    left out of the search rather than skipped after it. A bound spent re-reading the oldest
    twenty-five of a longer backlog would never reach the rest, which is the difference
    between a bounded pass and a stalled one. A `void` or `expired` question is deliberately
    still offered: those are endings where the asking failed, and `ask` reopens them. A
    withdrawn one is the owner's own "stop", and an answered one is a decision.

    The question quotes the subject's own words rather than describing them: for a reminder
    and a learned practice the sentence *is* the decision, and an owner cannot adopt what they
    were not shown.
    """
    from ..lifecycle.erasure import AWAITING

    bound = max(1, min(_bounded(limit), 25))
    found: list[tuple[str, str, str]] = []
    for row in store.db.execute(
            "SELECT g.id, g.title FROM goals g WHERE g.status='candidate' AND "
            "g.created_kind='agent' AND NOT EXISTS (SELECT 1 FROM inquiries q WHERE "
            "q.decision='goal-activation' AND q.subject_id=g.id AND q.state IN "
            "('open','sent','answered','withdrawn')) "
            "ORDER BY g.created_at, g.id LIMIT ?", (bound,)).fetchall():
        found.append(("goal-activation", str(row["id"]),
                      f"May I put this on your list: “{str(row['title'])[:160]}”? It reminds "
                      "for nothing until you say yes, and saying no decides nothing."))
    for row in store.db.execute(
            "SELECT l.id, l.version, l.text FROM lessons l WHERE l.status='candidate' AND "
            "l.created_kind='agent' AND NOT EXISTS (SELECT 1 FROM inquiries q WHERE "
            "q.decision='lesson-activation' AND q.subject_id=l.id || '@' || l.version AND "
            "q.state IN ('open','sent','answered','withdrawn')) "
            "ORDER BY l.created_at, l.id LIMIT ?", (bound,)).fetchall():
        found.append(("lesson-activation", f"{row['id']}@{row['version']}",
                      f"May I start following this: “{str(row['text'])[:220]}”? It is a "
                      "candidate until you say so."))
    for row in store.db.execute(
            "SELECT c.id, c.rule, c.basis, a.identifier AS account_a, "
            "b.identifier AS account_b FROM identity_candidates c "
            "JOIN identity_accounts a ON a.id=c.account_a "
            "JOIN identity_accounts b ON b.id=c.account_b "
            "WHERE c.state='pending' AND c.proposed_kind='agent' AND NOT EXISTS ("
            "SELECT 1 FROM inquiries q WHERE q.decision='identity' AND q.subject_id=c.id "
            "AND q.state IN ('open','sent','answered','withdrawn')) "
            "ORDER BY c.proposed_at, c.id LIMIT ?", (bound,)).fetchall():
        found.append(("identity", str(row["id"]),
                      f"Are {str(row['account_a'])[:80]} and {str(row['account_b'])[:80]} the "
                      f"same person? The rule proposed was {str(row['rule'])[:80]}, because "
                      f"{str(row['basis'])[:200]}"))
    for row in store.db.execute(
            "SELECT l.id, l.preview, l.preview_digest FROM erasure_ledger l WHERE "
            "l.state=? AND l.requester_kind='agent' AND NOT EXISTS (SELECT 1 FROM inquiries q "
            "WHERE q.decision='forgetting' AND q.subject_id=l.id AND q.state IN "
            "('open','sent','answered','withdrawn')) "
            "ORDER BY l.requested_at, l.id LIMIT ?", (AWAITING, bound)).fetchall():
        # The preview's own counts, read off the JSON the intent was priced with — the same
        # numbers `owner --list` prints, so the message and the door cannot disagree about how
        # big a deletion is.
        payload = json.loads(str(row["preview"] or "{}"))
        found.append(("forgetting", str(row["id"]),
                      f"Confirm this forgetting: {len(payload.get('records', []))} record(s), "
                      f"{len(payload.get('dependents', []))} derived thing(s), and "
                      f"{len(payload.get('obligations', []))} copies the backend has to be "
                      f"told. The preview you would be signing is "
                      f"{str(row['preview_digest'])[:12]}."))
    return found
