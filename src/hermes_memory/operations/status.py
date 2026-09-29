"""C14 status: what is running, stage by stage, without asking a model anything.

A single "healthy" flag is the thing that made the old installation impossible to
diagnose: capture can be durable while formation is paused, formation can be running
while the backend is unreachable, and the resource gate can be holding a slot while
nothing is queued. Each stage therefore reports what it actually observed, with
``configured``, ``operational``, ``degraded`` and ``paused`` kept apart — an
unconfigured stage is not a broken one, and a paused stage is somebody's decision
with a name and a timestamp attached to it.

Nothing here writes, opens a socket, or composes prose. Every figure comes from one
read transaction over one connection, so no two numbers in a report are drawn from
different moments, and asking for status costs a running system nothing.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from ..backend.worker_launcher import OperationLedger
from ..ids import now, timestamp
from ..processing.allowance import Allowances
from ..processing.instance_gate import instance_gate, status_gate
from ..processing.resource_gate import UNCERTAIN, WAITING, ResourceGate
from ..sources.capture_spool import (
    SPOOL_SOURCE, SpoolError, spool_backlog, spool_beside)
from ..sources.sync import COVERAGE_STATES
from ..storage.evidence import EvidenceError

__all__ = ["StatusReporter", "StageReport", "STATUS_STATES", "REPORTED_STAGES",
           "formation_unattended", "snapshot"]

CONFIGURED = "configured"
OPERATIONAL = "operational"
DEGRADED = "degraded"
PAUSED = "paused"
UNCONFIGURED = "unconfigured"
DISABLED = "disabled"
STATUS_STATES = (CONFIGURED, OPERATIONAL, DEGRADED, PAUSED, UNCONFIGURED, DISABLED)

# Coverage the runtime reports when the last honest look at a source went badly.
# ``unknown`` and ``partial`` are deliberately absent: a connector nobody has run
# yet, and one still working through a backfill, are neither of them a fault.
UNHEALTHY_COVERAGE = frozenset({"stale", "unreachable", "revoked"})

# A queue this old has been retried, quarantined, or forgotten about. It is not "busy".
STALE_QUEUE_MINUTES = 60

REPORTED_STAGES = ("capture", "raw_indexing", "observations", "summaries", "goals",
                   "analysis", "delivery", "backend", "resource_gate")

# Outbox rows a transport is still responsible for, and rows whose only outstanding
# question is whether the outside world accepted them.
IN_FLIGHT = ("prepared", "leased", "attempted")
# A send the framework cannot witness is not one failure but two, and they want different
# answers. `uncertain` means the handover began and no answer came back: the owner may have
# been messaged or may not have, and somebody has to reconcile it. `accepted_unverified`
# means the carrier said it sent and gave nothing to check that against — a delivery that
# finished, with a transport that does not read its own wire back. Holding that open as a
# degradation asks every day's reminders to be permanently red, which is how a report stops
# being read at all; what it deserves is to be counted, named, and marked as awaiting a
# transport that can echo the correlation back.
AWAITING_RECONCILIATION = ("uncertain",)
UNPROVEN = ("accepted_unverified",)
# Job states that mean work is waiting rather than finished, and states that mean the
# queue is not merely slow but stuck.
BUSY_JOBS = ("queued", "leased", "submitting", "running", "retry_wait")
STUCK_JOBS = ("uncertain", "quarantined")
# The two halves of ``BUSY_JOBS``, kept apart because they say different things: a lease
# has a worker's name on it, and an unclaimed row only has a place in line.
RUNNING_JOBS = ("leased", "submitting", "running")
WAITING_JOBS = ("queued", "retry_wait")


@dataclass(frozen=True)
class StageReport:
    """One component's own account of itself."""

    name: str
    state: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in STATUS_STATES:
            raise EvidenceError(f"unknown status state {self.state!r}")

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "state": self.state, "detail": self.detail,
                **self.evidence}


class StatusReporter:
    """Read-only status over the canonical store."""

    def __init__(self, store, *, settings: Any = None, gate: ResourceGate | None = None,
                 stale_queue_minutes: int = STALE_QUEUE_MINUTES):
        self.store = store
        self.db = store.db
        self.settings = settings
        # One gate for the whole installation, not one per profile: two people using
        # the same GPU are one queue, and the reading that reports it has to look at the
        # ledger where both of them actually stand. Read-only, because a status report
        # that created the file it was about to measure would not be a reading.
        self.gate = gate or (status_gate(settings) if settings is not None
                             else ResourceGate(store))
        if not 1 <= int(stale_queue_minutes) <= 10_080:
            raise EvidenceError("stale_queue_minutes must be between 1 minute and a week")
        self.stale_queue_minutes = int(stale_queue_minutes)
        self._by_name: dict[str, Callable[[], StageReport]] = {
            "capture": self.capture, "raw_indexing": self.raw_indexing,
            "observations": self.observations, "summaries": self.summaries,
            "goals": self.goals, "analysis": self.analysis, "delivery": self.delivery,
            "backend": self.backend, "resource_gate": self.resource_gate,
        }

    def release_staged_at(self) -> str | None:
        """When the release this installation runs was staged, or nothing when it says less.

        Read from the tree the units point at rather than from this process's own import path:
        the question is about the machine that is answering, and a staged release that
        predates this field is allowed to stay silent.
        """
        if self.settings is None:
            return None
        from ..install.release import running_staged_at

        return running_staged_at(self.settings)

    # -- report --------------------------------------------------------------

    def report(self, *, profile: str | None = None) -> dict[str, Any]:
        """Everything, as one snapshot of one connection."""
        with snapshot(self.db):
            stages = [self._by_name[name]() for name in REPORTED_STAGES]
            erasure = self.erasure_backlog()
            waiting = self.pending_confirmations()
            queue = self.queue_age()
            background = self.background_pass()
            sending = self.delivery_loop()
            asking = self.questions()
            epoch = self.store.epoch()
        states = {stage.name: stage.state for stage in stages}
        return {
            "observed_at": now(),
            "profile": profile or "default",
            "epoch": epoch,
            "stages": [stage.as_dict() for stage in stages],
            "states": states,
            # One word for a human, derived from parts that stay in the report.
            "overall": _overall(states),
            "erasure_backlog": erasure,
            "awaiting_owner": waiting,
            "questions": asking,
            "queue": queue,
            "background_pass": background,
            "delivery_loop": sending,
            "notes": _notes(stages, erasure=erasure, waiting=waiting, queue=queue,
                            background=background, delivery_loop=sending, questions=asking),
        }

    def stage(self, name: str) -> StageReport:
        """One stage on its own, for a watcher that only cares about that part."""
        if name not in self._by_name:
            raise EvidenceError(f"unknown stage {name!r}; reportable stages are "
                                f"{list(REPORTED_STAGES)}")
        with snapshot(self.db):
            return self._by_name[name]()

    # -- the stages ----------------------------------------------------------

    def capture(self) -> StageReport:
        """Ingestion: connectors, their fences, and what they could not give us."""
        sources = self.db.execute("SELECT * FROM connectors ORDER BY source").fetchall()
        paused = self._pauses("capture")
        gaps = self.db.execute(
            "SELECT g.source, count(*) AS n FROM source_gaps g "
            "JOIN connectors c ON c.source = g.source AND c.generation = g.generation "
            "WHERE g.cleared_at IS NULL GROUP BY g.source ORDER BY g.source").fetchall()
        open_gaps = {row["source"]: int(row["n"]) for row in gaps}
        # Only the generation that is still reading counts as debt. A gap is a promise that the
        # thing is missing, and a reconfigured connector re-reads from the beginning — so the
        # retired generation's rows are history (still `sources --gaps`'s to show), while
        # counting them here would keep a stage open on a failure it has already answered.
        unhealthy = sorted(row["source"] for row in sources
                           if row["coverage_state"] in UNHEALTHY_COVERAGE)
        records = int(self.db.execute(
            "SELECT count(*) FROM records WHERE deleted=0").fetchone()[0])
        spool = self._spool()
        # Turns the host handed over and no connector has ever committed are invisible here:
        # they are not in this database, so the stage's own ledger says nothing about them. A
        # spool with rows waiting and no successful drain behind them is the stage not working,
        # whichever green number the connectors table happens to hold.
        drained = any(row["source"] == SPOOL_SOURCE and row["last_success_at"]
                      for row in sources)
        unread = bool(spool.get("error")) or (spool["pending"] > 0 and not drained)
        if self._all_stopped("capture", paused):
            # Somebody's decision, with a name and a timestamp: turns waiting behind a pause
            # are what the pause asked for.
            state = PAUSED
        elif unread:
            # Ahead of the unconfigured answer on purpose. The host has handed over a source
            # with work in it, so "nothing is configured here" would be the same lie the
            # degraded table told, only in a different set of clothes.
            state = DEGRADED
        elif not sources:
            state = UNCONFIGURED
        elif unhealthy or open_gaps:
            state = DEGRADED
        else:
            state = OPERATIONAL
        detail = (f"{len(sources)} source(s) registered, "
                  f"{len([row for row in sources if row['source'] in paused])} paused, "
                  f"{sum(open_gaps.values())} open gap(s), {records} live record(s)")
        if spool["present"]:
            detail += f"; {spool['pending']} captured event(s) in the host spool"
        if spool.get("error"):
            detail += f"; the host spool cannot be read: {spool['error']}"
        return StageReport(
            "capture", state, detail,
            {"sources": [{"source": row["source"], "generation": row["generation"],
                          "coverage_state": row["coverage_state"],
                          "paused": row["source"] in paused,
                          "pause_reason": (paused.get(row["source"]) or {}).get("reason"),
                          "cursor": row["cursor"],
                          "open_gaps": open_gaps.get(row["source"], 0),
                          "last_success_at": row["last_success_at"]}
                         for row in sources],
             "unhealthy": unhealthy,
             "records": records,
             "open_gaps": sum(open_gaps.values()),
             "spool": spool,
             "spool_unread": unread,
             "coverage_states": sorted(COVERAGE_STATES)})

    def raw_indexing(self) -> StageReport:
        """Canonical evidence and its local index — the channel that never needs a model."""
        live, visible, searchable = (int(value) for value in self.db.execute(
            """
            SELECT (SELECT count(*) FROM records WHERE deleted=0) AS live,
                   (SELECT count(*) FROM records r
                     WHERE r.deleted=0 AND NOT EXISTS (
                         SELECT 1 FROM record_visibility v
                          WHERE v.record_id=r.id AND v.hidden=1)) AS visible,
                   (SELECT count(*) FROM records r
                     WHERE r.deleted=0 AND NOT EXISTS (
                         SELECT 1 FROM record_visibility v
                          WHERE v.record_id=r.id AND v.hidden=1)
                       AND EXISTS (SELECT 1 FROM record_fts f WHERE f.id=r.id)) AS searchable
            """).fetchone())
        leaks = int(self.db.execute(
            "SELECT count(*) FROM record_fts f WHERE NOT EXISTS ("
            "  SELECT 1 FROM records r WHERE r.id = f.id AND r.deleted = 0 AND NOT EXISTS ("
            "    SELECT 1 FROM record_visibility v WHERE v.record_id = r.id AND v.hidden = 1))"
        ).fetchone()[0])
        # Every figure is a count over *records*, never one table total minus another:
        # that arithmetic hides an orphaned search entry behind a missing one, and a
        # hidden record absent from the index is correct rather than a defect.
        unindexed = visible - searchable
        # A leak outranks "nothing is configured here": an entry that outlived its
        # evidence is exactly the shape a restore would resurrect, and it is most
        # dangerous in a store that has since gone quiet.
        state = (DEGRADED if leaks else
                 UNCONFIGURED if not live else
                 DEGRADED if unindexed else OPERATIONAL)
        return StageReport(
            "raw_indexing", state,
            f"{live} live record(s), {visible} eligible for retrieval, "
            f"{searchable} in the index",
            {"live": live, "hidden": live - visible, "visible": visible,
             "searchable": searchable, "unindexed": unindexed, "index_leaks": leaks})

    def observations(self) -> StageReport:
        """Assertions formed from evidence, and the queue that forms them."""
        counts = self._grouped("SELECT status, count(*) AS n FROM assertions GROUP BY status")
        queue = self._job_counts()
        if self.capture_only:
            return StageReport("observations", DISABLED,
                               "no inference route or budget is configured, so nothing is "
                               "formed and the canonical record is the answer",
                               {"reason": "capture-only", "assertions": dict(counts),
                                "queue": dict(queue), "journal": self._journal()})
        paused = self._pauses("formation")
        # The hold on a *physical* model is not a per-source policy: it is the owner's
        # decision over hardware every profile shares, and it lives in the instance
        # admission ledger. Nothing can be formed while it stands, whatever the
        # connector-level fences and the queue say.
        hold = self.gate.hold() if self.gate is not None else None
        held = bool(hold and hold["state"] == "paused")
        stuck = {state: int(queue.get(state, 0)) for state in STUCK_JOBS
                 if queue.get(state)}
        # Two different things a queue can say. ``running`` means a lease has a name on
        # it, so work is happening now; ``waiting`` means rows exist that nobody has
        # claimed, which is a stage that is set up rather than one that is operating.
        # Calling the second one operational is the false readiness this report exists to
        # avoid, because nothing in this installation claims a job by itself.
        waiting = _any(queue, WAITING_JOBS) and not _any(queue, RUNNING_JOBS)
        running = _any(queue, RUNNING_JOBS)
        # The local `assertions` table is what this stage has *confirmed*, not what it is owed:
        # a record can be captured, live, and never asked of the backend, and the queue can be
        # empty because formation is an approved act rather than because there is nothing to
        # form. Measuring only the confirmed rows called a working pipeline unconfigured on a
        # machine with five verified documents and seven records nobody has offered yet.
        unprojected = self._unprojected()
        standing = self._standing()
        state = (DEGRADED if stuck else
                 PAUSED if held or self._all_stopped("formation", paused) else
                 OPERATIONAL if running else
                 CONFIGURED if (unprojected or waiting) else
                 OPERATIONAL if counts else UNCONFIGURED)
        detail = (f"{counts.get('confirmed', 0)} confirmed / "
                  f"{counts.get('candidate', 0)} candidate assertion(s); "
                  f"queue {_flatten(queue)}")
        if unprojected:
            detail += (f"; {unprojected} live record(s) have no backend projection and nothing "
                       "is queued to form them — `hermes-memory form` shows the list and prices "
                       "the approval, which is the part nothing does for itself")
        if standing is not None:
            # Said even when there is no debt: a permission the machine is holding is a fact
            # about what will happen next, and the owner needs it named to remember it exists.
            detail += (f"; a standing grant by {standing['actor']} ({standing['id']}) covers "
                       f"{standing['resource']} until {standing['expires_at']} with "
                       f"{standing['records']['left']} record(s) and "
                       f"{standing['tokens']['left']:,} token(s) left")
        if held:
            detail += _hold_note(hold, "inference", self.release_staged_at())
        elif waiting:
            detail += ("; only `hermes-memory form` works this queue — nothing drains it "
                       "by itself")
        return StageReport(
            "observations", state, detail,
            {"assertions": dict(counts), "queue": dict(queue), "stuck": stuck,
             "unprojected": unprojected,
             # Asked of the queue as it stands, not as a row count: five finished jobs are a
             # busy-looking queue and an unformable record is still unformed.
             "formation_owed": bool(unprojected) and not waiting and not running,
             "paused_sources": sorted(paused), "instance_hold": held,
             "instance_hold_by": (hold or {}).get("actor"),
             "instance_hold_reason": (hold or {}).get("reason"),
             # Whether a pass could run without anybody reading this list first — a live
             # grant, not a wish. The resource it names is shown so a reader can see whether
             # it covers the device the route currently points at.
             "formation_unattended": standing is not None, "allowance": standing,
             "draining": running,
             "journal": self._journal()})

    def summaries(self) -> StageReport:
        rows = self.db.execute(
            "SELECT kind, status, count(*) AS n, max(published_at) AS newest FROM summaries "
            "GROUP BY kind, status ORDER BY kind, status").fetchall()
        by_kind: dict[str, dict[str, int]] = {}
        for row in rows:
            by_kind.setdefault(row["kind"], {})[row["status"]] = int(row["n"])
        refreshes = self._grouped(
            "SELECT state, count(*) AS n FROM summary_refreshes GROUP BY state")
        unsupported = self.unsupported_artifacts()
        total = sum(sum(kinds.values()) for kinds in by_kind.values())
        # A withdrawn summary is not a fault; a summary citing evidence that no longer
        # resolves is, because it is still handed out as though it were supported.
        state = (DEGRADED if unsupported["artifacts"] or refreshes.get("failed")
                 else OPERATIONAL if total else UNCONFIGURED)
        return StageReport(
            "summaries", state,
            f"{total} summary row(s) across {len(by_kind)} kind(s), "
            f"{unsupported['artifacts']} with a citation that no longer resolves",
            {"by_kind": by_kind, "refreshes": dict(refreshes),
             "newest": max((row["newest"] for row in rows if row["newest"]), default=None),
             "unsupported": unsupported})

    def goals(self) -> StageReport:
        """Prospective memory: intent on file, and whether its clocks are being kept."""
        counts = self._grouped("SELECT status, count(*) AS n FROM goals GROUP BY status")
        events = self._grouped("SELECT state, count(*) AS n FROM due_events GROUP BY state")
        intents = self._grouped(
            "SELECT state, count(*) AS n FROM decision_intents GROUP BY state")
        overdue = int(self.db.execute(
            "SELECT count(*) FROM due_events WHERE state='pending' AND fire_at <= ?",
            (now(),)).fetchone()[0])
        expired_claims = int(self.db.execute(
            "SELECT count(*) FROM due_events WHERE state IN ('claimed','handed_off') "
            "AND claim_until IS NOT NULL AND claim_until < ?", (_clock(),)).fetchone()[0])
        # An uncertain due event is a handoff whose outcome nobody can attest to; an
        # expired claim is one whose holder stopped answering; a pending event past its
        # moment is a clock nobody is watching. All three are open debts.
        unresolved = {"due_events": int(events.get("uncertain", 0)),
                      "intents": int(intents.get("uncertain", 0)),
                      "overdue": overdue,
                      "expired_claims": expired_claims}
        state = (UNCONFIGURED if not counts else
                 DEGRADED if any(unresolved.values()) else OPERATIONAL)
        return StageReport(
            "goals", state,
            f"{counts.get('active', 0)} active goal(s), "
            f"{events.get('pending', 0)} due event(s) pending, {overdue} overdue",
            {"by_status": dict(counts), "due_events": dict(events),
             "intents": dict(intents), "overdue": overdue, "unresolved": unresolved})

    def analysis(self) -> StageReport:
        """What the shadow analyst decided, as opposed to what it was asked."""
        if self.capture_only:
            return StageReport("analysis", DISABLED,
                               "no inference route is configured, so no intent is analysed "
                               "and policy decides alone",
                               {"reason": "capture-only"})
        rows = self.db.execute(
            "SELECT action, shadow, model_used, count(*) AS n FROM proactive_decisions "
            "GROUP BY action, shadow, model_used").fetchall()
        by_action: dict[str, int] = {}
        for row in rows:
            by_action[row["action"]] = by_action.get(row["action"], 0) + int(row["n"])
        modelled = sum(int(row["n"]) for row in rows if row["model_used"])
        shadowed = sum(int(row["n"]) for row in rows if row["shadow"])
        total = sum(by_action.values())
        last = self.db.execute(
            "SELECT decided_at FROM proactive_decisions ORDER BY decided_at DESC LIMIT 1"
        ).fetchone()
        paused = self._pauses("proactivity")
        # A stopped stage is reported as stopped even where it left decisions behind:
        # the history is in the evidence, and the headline has to say what happens next.
        state = (PAUSED if self._all_stopped("proactivity", paused) else
                 OPERATIONAL if total else UNCONFIGURED)
        return StageReport(
            "analysis", state,
            f"{total} decision(s) on file, {modelled} with a model in the loop, "
            f"{shadowed} still shadowed",
            {"by_action": by_action, "model_used": modelled, "shadowed": shadowed,
             "last_decided_at": last["decided_at"] if last else None,
             "paused_sources": sorted(paused)})

    def delivery(self) -> StageReport:
        """The framework makes artifacts; a transport sends them. This says which is stuck."""
        counts = self._grouped("SELECT state, count(*) AS n FROM outbox GROUP BY state")
        in_flight = sum(int(counts.get(state, 0)) for state in IN_FLIGHT)
        unproven = sum(int(counts.get(state, 0)) for state in UNPROVEN)
        awaiting = sum(int(counts.get(state, 0)) for state in AWAITING_RECONCILIATION)
        overdue = int(self.db.execute(
            "SELECT count(*) FROM outbox WHERE state IN ({}) AND expires_at IS NOT NULL "
            "AND expires_at <= ?".format(",".join("?" * len(IN_FLIGHT))),
            [*IN_FLIGHT, now()]).fetchone()[0])
        paused = self._pauses("proactivity")
        hold = self.store.control("global", "delivery")
        held = bool(hold and hold["state"] == "paused")
        # An artifact past its own expiry is not "in flight": it is a transport that
        # stopped coming, and the report says so rather than counting it as busy work.
        state = (DEGRADED if awaiting or overdue else
                 PAUSED if held or self._all_stopped("proactivity", paused) else
                 OPERATIONAL if in_flight or counts.get("confirmed") or unproven
                 else UNCONFIGURED)
        shadowed = self._shadowed()
        # An empty outbox does not say *why* it is empty, and "unconfigured" sent people to the
        # model config to look for a switch that lives in the attention policy. The state stays
        # as it was — nothing was configured into a working pipeline — but the reason is said
        # where it is read, next to the reminder the pass is deferring for exactly this.
        quiet = (state == UNCONFIGURED and shadowed)
        return StageReport(
            "delivery", state,
            f"{in_flight} artifact(s) waiting on the transport, {unproven} sent without a "
            f"proof this framework can check, {awaiting} awaiting reconciliation, "
            f"{counts.get('confirmed', 0)} confirmed"
            + ("; the owner has not switched delivery on yet, so a due reminder is kept and "
               "deferred (`hermes-memory owner --switch-delivery on --timezone …`)"
               if quiet else "")
            + (_hold_note(hold, "delivery", self.release_staged_at()) if held else ""),
            {"by_state": dict(counts), "in_flight": in_flight, "unproven": unproven,
             "awaiting_reconciliation": awaiting,
             "past_expiry": overdue, "paused_sources": sorted(paused),
             "shadow": shadowed,
             "instance_hold": held, "instance_hold_by": (hold or {}).get("actor"),
             "owner_principal": getattr(self.settings, "owner_principal", None)})

    def _shadowed(self) -> bool:
        """Whether the owner has yet agreed to be interrupted at all."""
        from ..proactive.policy import AttentionPolicy

        try:
            return bool(AttentionPolicy(self.store).settings("general")["shadow"])
        except Exception:
            # A reading that cannot be taken is not a decision about delivery; the stage says
            # what it knows from the outbox instead of guessing at the policy.
            return False

    def backend(self) -> StageReport:
        """Derived backend: what the projection ledger says, never what we hope."""
        url = getattr(self.settings, "hindsight_url", None) if self.settings else None
        counts = self._grouped("SELECT state, count(*) AS n FROM backend_documents "
                               "GROUP BY state")
        banks = [row["bank_id"] for row in self.db.execute(
            "SELECT DISTINCT bank_id FROM backend_documents ORDER BY bank_id")]
        total = sum(counts.values())
        if not url:
            return StageReport("backend", UNCONFIGURED,
                               "no backend is configured; the local channels still answer",
                               {"reason": "unconfigured", "documents": dict(counts),
                                "banks": banks})
        stale_epoch = int(self.db.execute(
            "SELECT count(*) FROM backend_documents WHERE state IN ('queued','submitted') "
            "AND desired_epoch < ?", (self.store.epoch(),)).fetchone()[0])
        # Mappings under a superseded epoch can never be acknowledged by the current
        # backend: the reset that advanced the epoch invalidated them.
        state = (DEGRADED if counts.get("failed") or stale_epoch else
                 OPERATIONAL if total else CONFIGURED)
        return StageReport(
            "backend", state,
            f"{total} document(s) mapped across {len(banks)} bank(s), "
            f"{counts.get('queued', 0) + counts.get('submitted', 0)} unacknowledged",
            {"url": url, "banks": banks, "by_state": dict(counts),
             "superseded_epoch": stale_epoch,
             "reachable": "not probed; status never opens a socket"})

    def resource_gate(self) -> StageReport:
        """One slot per physical resource. This says who is standing in it."""
        if self.gate is None:
            return StageReport("resource_gate", UNCONFIGURED,
                               "no admission ledger exists, so nothing has been queued "
                               "against a physical model from here",
                               {"occupancy": {}, "held": [], "uncertain": 0,
                                "waiting": 0, "blocked": [], "usage": None,
                                "operations": {"by_state": {}, "unattributed": 0,
                                               "unresolved": 0, "cancellations_owed": 0}})
        held = self.gate.held()
        occupancy = {resource: dict(states)
                     for resource, states in self.gate.occupancy().items()}
        uncertain = sum(int(states.get(UNCERTAIN, 0)) for states in occupancy.values())
        waiting = sum(int(states.get(WAITING, 0)) for states in occupancy.values())
        taken = sum(sum(states.values()) for states in occupancy.values())
        # The backend's own worker writes its operation identities into this same ledger,
        # because the thing being attributed is a claim on one machine's models. Operations
        # that could not be placed on a resource are an accounting fault, not a curiosity:
        # the alternative reading is that somebody else's allowance paid for them.
        operations = OperationLedger(self.gate.store).report()
        # A pause on a gate nobody has used yet is still somebody's decision, so it is
        # reported before the "never reserved" case rather than hidden behind it.
        owed = int(operations.get("cancellations_owed", 0))
        state = (DEGRADED if uncertain or operations["unattributed"] or owed else
                 PAUSED if self.gate.paused else
                 OPERATIONAL if taken or self.gate.ever_used() else UNCONFIGURED)
        return StageReport(
            "resource_gate", state,
            f"{taken} slot(s) occupied, {waiting} waiting"
            + (f", {uncertain} unresolved reservation(s)" if uncertain else "")
            + (f", {operations['unresolved']} backend operation(s) unaccounted"
               if operations["unresolved"] or operations["unattributed"] else "")
            # An asked-for stop with no answer is not a detail about the past: the operation
            # may still be spending a slot, and `hermes-memory cancel --list` is the queue
            # that says which ones.
            + (f", {owed} cancellation(s) asked for with no answer" if owed else ""),
            {"occupancy": occupancy,
             "held": [{"resource": row["resource"], "route": row["route"],
                       "holder": row["holder"], "priority": int(row["priority"]),
                       "held_for_seconds": round(_age_since(row["acquired_at"]), 1),
                       "lease_seconds_left": round(_seconds_until(row["lease_until"]), 1)}
                      for row in held],
             "uncertain": uncertain, "waiting": waiting,
             "blocked": self.gate.blocked_resources(),
             "operations": operations,
             "usage": self.gate.usage()})

    # -- supporting reads ----------------------------------------------------

    def erasure_backlog(self) -> dict[str, Any]:
        """Forgotten locally is not the same as forgotten everywhere."""
        intents = self._grouped(
            "SELECT state, count(*) AS n FROM erasure_ledger GROUP BY state")
        return {
            "intents": dict(intents),
            "awaiting_owner": int(intents.get("awaiting_confirmation", 0)),
            "obligations_open": int(self.db.execute(
                "SELECT count(*) FROM erasure_targets WHERE state!='verified'").fetchone()[0]),
            "tombstones": int(self.db.execute(
                "SELECT count(*) FROM tombstones").fetchone()[0]),
        }

    def pending_confirmations(self) -> dict[str, Any]:
        """Decisions that are only ever the owner's to make, and are still waiting."""
        from ..learning.lessons import LessonStore
        from ..learning.outcomes import OutcomeLog

        lessons = LessonStore(self.store,
                              outcomes=OutcomeLog(self.store, owner_principal=getattr(
                                  self.settings, "owner_principal", None)),
                              owner_principal=getattr(self.settings, "owner_principal", None))
        return {
            "identity_candidates": int(self.db.execute(
                "SELECT count(*) FROM identity_candidates WHERE state='pending'"
            ).fetchone()[0]),
            "candidate_assertions": int(self.db.execute(
                "SELECT count(*) FROM assertions WHERE status='candidate'").fetchone()[0]),
            "erasure_intents": int(self.db.execute(
                "SELECT count(*) FROM erasure_ledger WHERE state='awaiting_confirmation'"
            ).fetchone()[0]),
            # A habit the system proposes is not a rule until a person says so, and a rule
            # that stopped holding up is withdrawn by the same kind of act. Retrieval
            # already stops teaching either one; this is the half that says so out loud.
            "lesson_candidates": int(self.db.execute(
                "SELECT count(*) FROM lessons WHERE status='candidate'").fetchone()[0]),
            "lessons_for_review": len(lessons.needs_review()),
            "lessons_awaiting_promotion": [
                {"id": f"{row['id']}@{row['version']}", "proposed_by": row["created_by"],
                 "proposed_kind": row["created_kind"], "proposed_at": row["created_at"]}
                for row in self.db.execute(
                    "SELECT id, version, created_by, created_kind, created_at FROM lessons "
                    "WHERE status='candidate' ORDER BY created_at, id LIMIT 8").fetchall()],
            # A reminder an agent proposed schedules nothing until the owner adopts it, so
            # the waiting list is the only place its existence is visible. Counted, not
            # quoted: the title is the owner's business and `goal --list` reads it.
            "goal_candidates": int(self.db.execute(
                "SELECT count(*) FROM goals WHERE status='candidate'").fetchone()[0]),
            "goals_awaiting_adoption": [
                {"goal": row["id"], "proposed_by": row["created_by"],
                 "proposed_kind": row["created_kind"], "proposed_at": row["created_at"],
                 "wants_a_due_time": row["due_at"] is not None}
                for row in self.db.execute(
                    "SELECT id, created_by, created_kind, created_at, due_at FROM goals "
                    "WHERE status='candidate' ORDER BY created_at, id LIMIT 8").fetchall()],
        }

    def unsupported_artifacts(self) -> dict[str, Any]:
        """Derived artifacts whose quoted evidence no longer resolves to a live record.

        The verdict is computed on read and never stored, so a summary does not stop
        being supported because a row said so — but it does stop being supportable
        once the record behind it is gone or hidden, and that is what this counts.
        """
        rows = self.db.execute(
            """
            SELECT c.artifact_id, c.kind,
                   sum(CASE WHEN r.id IS NULL OR r.deleted=1 OR v.hidden=1 THEN 1 ELSE 0 END)
                       AS broken,
                   count(*) AS cited
            FROM derived_citations c
            LEFT JOIN records r ON r.id = c.record_id
            LEFT JOIN record_visibility v ON v.record_id = c.record_id
            GROUP BY c.artifact_id, c.kind ORDER BY c.artifact_id""").fetchall()
        bad = [{"artifact_id": row["artifact_id"], "kind": row["kind"],
                "broken_citations": int(row["broken"]), "cited": int(row["cited"])}
               for row in rows if int(row["broken"])]
        return {"artifacts": len(bad), "details": bad[:50]}

    def queue_age(self) -> dict[str, Any]:
        """How long the oldest waiting job has been waiting, and whether that is odd.

        Ordered by arrival, not by priority: a maintenance job that has sat for hours
        is the evidence that nothing is consuming the queue, and sorting by urgency
        first would hide it behind a job that arrived a second ago.
        """
        row = self.db.execute(
            "SELECT created_at FROM processing_jobs WHERE state IN ({}) "
            "ORDER BY created_at LIMIT 1".format(",".join("?" * len(BUSY_JOBS))),
            list(BUSY_JOBS)).fetchone()
        age = None if row is None else _age_from_text(row["created_at"])
        limit = self.stale_queue_minutes * 60
        return {"oldest_seconds": None if age is None else round(age, 1),
                "stale": age is not None and age > limit, "limit_seconds": limit}

    def background_pass(self) -> dict[str, Any]:
        """The zero-inference pass: whether it is scheduled, and when it last said so.

        The scheduler is a thread in another process, so the only evidence available to a
        reading is the heartbeat the pass itself recorded. Absence is reported as absence
        rather than as health: three days of due reminders and no heartbeat is the failure
        this line exists to make visible, and it looks exactly like a healthy report
        without it.
        """
        from ..processing.maintenance import Maintenance

        last = Maintenance.last_pass(self.store)
        interval = int(getattr(self.settings, "maintenance_interval_s", 0) or 0)
        waiting = int(self.db.execute(
            "SELECT count(*) AS n FROM due_events WHERE state='pending' AND fire_at<=?",
            (now(),)).fetchone()["n"])
        age = None if last is None else _age_from_text(last["created_at"])
        # Three periods: one missed pass is a restart, two is a busy machine, three is a
        # loop that is not running.
        behind = age is not None and interval and age > interval * 3
        never = last is None and interval > 0
        failed = last.get("failed_section") if last else None
        return {"scheduled": interval > 0, "interval_seconds": interval,
                "last_pass_at": None if last is None else last["created_at"],
                "seconds_since_last_pass": None if age is None else round(age, 1),
                "last_report": None if last is None else {
                    key: value for key, value in last.items() if key != "created_at"},
                "failed_section": failed,
                "error": last.get("error") if failed else None,
                "waiting": waiting, "behind": bool(behind or never or failed),
                "note": ("nothing is scheduled to run by itself; `hermes-memory maintain` "
                         "is the pass, and the owner decides when" if interval <= 0 else
                         f"the last pass raised in section {failed!r}: {last['error']}; the "
                         "scheduler is running but the pass is not finishing" if failed else
                         "no pass has been recorded since this installation started"
                         if never else
                         "the last pass is older than three periods, so the runtime unit's "
                         "scheduler is not running" if behind else
                         "the background pass is running on schedule")}

    def delivery_loop(self) -> dict[str, Any]:
        """Whether anything on this machine sends what the pass prepared, and what it saw.

        The drain is a thread in another process, so its heartbeat is not evidence here. Two
        readings carry it: the line the drain wrote when it had something to say, and the
        backlog. A drain that finds nothing to send writes nothing — recording 1440 identical
        no-ops a day would bury the audit and prove nothing more — so an old line is not a
        dead loop. An artifact that stays *prepared* past three periods is, and telling those
        two apart is the whole of what this exists to do.
        """
        from ..proactive.delivery import last_drain

        last = last_drain(self.store)
        poll = int(getattr(self.settings, "delivery_poll_s", 0) or 0)
        enabled = bool(getattr(self.settings, "delivery_enabled", False))
        row = self.db.execute("SELECT count(*) AS n, min(created_at) AS oldest FROM outbox "
                              "WHERE state='prepared'").fetchone()
        waiting = int(row["n"] or 0)
        age = None if not row["oldest"] else _age_from_text(str(row["oldest"]))
        behind = bool(waiting and poll and age is not None and age > poll * 3)
        return {"scheduled": bool(enabled and poll > 0), "interval_seconds": poll,
                "enabled": enabled, "waiting": waiting,
                "oldest_seconds": None if age is None else round(age, 1),
                "last_drain_at": None if last is None else last["created_at"],
                "last_report": None if last is None else {
                    key: value for key, value in last.items() if key != "created_at"},
                "behind": behind,
                "note": ("delivery is switched off, so prepared artifacts wait for "
                         "`hermes-memory deliver`" if not enabled else
                         "no drain is scheduled (HERMES_MEMORY_DELIVERY_POLL_S=0), so the "
                         "owner sends by hand" if poll <= 0 else
                         f"{waiting} artifact(s) have been prepared for longer than three "
                         "periods: the runtime's drain loop is not sending them" if behind
                         else "the runtime drains the outbox on its own, on schedule")}

    def questions(self) -> dict[str, Any]:
        """What the archive is asking the owner, and whether the asking is switched on.

        A question is the only way a decision reaches the owner without them typing a command,
        and it is off by default — so the commonest real state of a fresh installation is
        "everything awaits, nothing is asked", and a report that could not say so would leave
        the owner reading a quiet archive as an idle one.
        """
        from ..proactive.inquiries import REPLIES_STAGE, InquiryStore

        inquiries = InquiryStore(self.store,
                                 owner_principal=getattr(self.settings, "owner_principal", None))
        allowed, who = inquiries.replies_allowed()
        counts = self._grouped("SELECT state, count(*) AS n FROM inquiries GROUP BY state")
        # Held, not refused: a question inside quiet hours or behind the daily limit has a
        # time on it, and one that has been waiting past its own window is the same fact at a
        # different temperature — which is what the doctor reads off this pair.
        held = int(self.db.execute(
            "SELECT count(*) FROM inquiries WHERE state='open' AND next_try_at IS NOT NULL "
            "AND next_try_at>?", (now(),)).fetchone()[0])
        open_rows = self.db.execute(
            "SELECT id, decision, subject_id, question, asked_at, expires_at, state, "
            "next_try_at FROM inquiries WHERE state IN ('open','sent') ORDER BY asked_at, id "
            "LIMIT 8").fetchall()
        # A question still being asked about a state that has already moved: the decision was
        # taken somewhere the asking never saw it, which is the one way this ledger can end up
        # disagreeing with the archive it is supposed to be a reading of.
        elsewhere = inquiries.settled_elsewhere()
        return {"replies_enabled": allowed, "why": who,
                "stage": str((self.store.control("global", REPLIES_STAGE) or {}).get("state")
                             or "unset"),
                "counts": {key: int(counts.get(key, 0)) for key in
                           ("open", "sent", "answered", "expired", "withdrawn", "void")},
                "awaiting_answer": int(counts.get("sent", 0)),
                "queued": int(counts.get("open", 0)),
                "answered": int(counts.get("answered", 0)),
                "expired": int(counts.get("expired", 0)),
                "decided_elsewhere": [{"inquiry": str(item.id),
                                       "decision": str(item.decision),
                                       "subject": str(item.subject_id),
                                       "state": str(item.state)} for item in elsewhere],
                "live": [{"id": str(row["id"]), "decision": str(row["decision"]),
                          "subject": str(row["subject_id"]), "state": str(row["state"]),
                          "asked_at": str(row["asked_at"]),
                          "expires_at": str(row["expires_at"])} for row in open_rows],
                "held": held, "waiting": int(counts.get("open", 0)) + int(
                    counts.get("sent", 0)),
                "note": (f"{int(counts.get('open', 0))} question(s) are queued and "
                         f"{int(counts.get('sent', 0))} are with the owner"
                         if allowed else
                         "nothing is being asked: " + who)}

    # -- primitives ----------------------------------------------------------

    @property
    def capture_only(self) -> bool:
        return bool(getattr(self.settings, "capture_only", False))

    def _unprojected(self) -> int | None:
        """Live records the derived backend has never been asked about.

        None when this installation has no backend to project to — a reading that invented a
        debt against an absent endpoint would be worse than one that says nothing. Counted here
        because this is the stage that has to say whether formation is acting, and formation
        answers either to a digest somebody read or to a standing grant the owner issued,
        neither of which a reading can give itself.
        """
        from ..processing.formation import count_unprojected

        bank_id = getattr(self.settings, "bank_id", None)
        if not bank_id or not getattr(self.settings, "hindsight_url", None):
            return None
        return count_unprojected(self.store, bank_id=bank_id)

    def _standing(self) -> dict[str, Any] | None:
        """A live standing grant over the shared models, or None.

        Read from the instance ledger on the read-only gate this reporter already holds, and
        never retired here: a report that had to mark a grant expired would have to create or
        write the file it promised only to read.
        """
        return _live_grant(self.gate)

    def _spool(self) -> dict[str, Any]:
        """What the host's capture spool holds beside this store — counts, never text.

        A file at that path which is not a spool is reported rather than raised: an unreadable
        spool is a finding about capture, not a reason this reading cannot answer.
        """
        path = spool_beside(self.store.path)
        try:
            return spool_backlog(path)
        except SpoolError as error:
            return {"spool": str(path), "present": True, "pending": 0, "settled": 0,
                    "states": {}, "oldest_at": None, "error": str(error)}

    def _pauses(self, stage: str) -> dict[str, dict[str, Any]]:
        rows = self.db.execute(
            "SELECT scope, actor, reason, changed_at FROM runtime_controls "
            "WHERE stage=? AND state='paused' ORDER BY scope", (stage,)).fetchall()
        return {row["scope"]: {"actor": row["actor"], "reason": row["reason"],
                               "changed_at": row["changed_at"]} for row in rows}

    def _all_stopped(self, stage: str, paused: dict[str, Any]) -> bool:
        """Is every registered source held at this stage?

        One paused source among several is a narrowed pipeline rather than a stopped
        stage: reporting the whole stage as paused would hide the work still expected
        of it, and a stop nobody asked for must never be reported as one.
        """
        sources = [row["source"] for row in self.db.execute("SELECT source FROM connectors")]
        return bool(sources) and all(source in paused for source in sources)

    def _journal(self) -> dict[str, Any]:
        """How far behind each consumer is, in changes rather than in guesses.

        Consumers name themselves as they advance, so this reports the ones that
        exist rather than a fixed list this component invented.
        """
        head = int(self.db.execute(
            "SELECT COALESCE(max(seq), 0) FROM change_journal").fetchone()[0])
        rows = self.db.execute(
            "SELECT consumer, seq, updated_at FROM consumer_checkpoints ORDER BY consumer"
        ).fetchall()
        return {"head": head,
                "consumers": [{"consumer": row["consumer"], "seq": int(row["seq"]),
                               "behind": head - int(row["seq"]),
                               "updated_at": row["updated_at"]} for row in rows]}

    def _grouped(self, sql: str, params: Sequence[Any] = ()) -> dict[str, int]:
        return {row[0]: int(row[1]) for row in self.db.execute(sql, list(params))}

    def _job_counts(self) -> dict[str, int]:
        return self._grouped("SELECT state, count(*) AS n FROM processing_jobs GROUP BY state")


class _Snapshot:
    """The transaction behind :func:`snapshot`."""

    def __init__(self, db):
        self.db = db
        self.owned = False

    def __enter__(self):
        if not self.db.in_transaction:
            self.db.execute("BEGIN")
            self.owned = True
        return self

    def __exit__(self, *excinfo):
        if self.owned:
            self.db.execute("ROLLBACK" if any(excinfo) else "COMMIT")
        return False


def snapshot(db):
    """A deferred read transaction that leaves an outer one alone.

    Status asked from inside a worker must not become the reason that worker failed,
    and a nested ``BEGIN`` would raise.
    """
    return _Snapshot(db)


def _clock() -> float:
    return datetime.now(timezone.utc).timestamp()


def _live_grant(reading) -> dict[str, Any] | None:
    """The owner's live standing grant, from an admission ledger that may not be open.

    Read-only and write-free by construction: the callers are readings. A ledger from before
    standing grants existed answers as what it is — no permission — rather than failing the
    report that only went looking.
    """
    if reading is None:
        return None
    try:
        return Allowances(reading.store).current()
    except sqlite3.OperationalError:
        return None


def formation_unattended(settings) -> bool:
    """Whether a bounded pass could be performed here without somebody reading the list first.

    A capability of the *installation*, so it is answered without opening anybody's archive:
    the admission ledger is the whole of what it asks.
    """
    reading = status_gate(settings)
    if reading is None:
        return False
    try:
        return _live_grant(reading) is not None
    finally:
        reading.close()


def _age_since(at: Any) -> float:
    return max(0.0, _clock() - float(at or 0))


def _seconds_until(at: Any) -> float:
    return float(at or 0) - _clock()


def _age_from_text(value: Any) -> float | None:
    try:
        parsed = datetime.fromisoformat(timestamp(str(value)))
    except (TypeError, ValueError):
        return None
    return max(0.0, _clock() - parsed.timestamp())


def _any(counts: dict[str, int], states: Sequence[str]) -> bool:
    return any(counts.get(state) for state in states)


def _hold_note(hold: dict[str, Any] | None, stage: str,
               staged_at: str | None = None) -> str:
    """Name whoever is holding a stage, rather than saying that something is.

    Both holds are owner decisions by construction — no other credential can write
    them — but the reading that says "the owner" without saying which one, or why, is
    the one an operator cannot act on. And a reason is free text about the machine that
    was running when it was written: when the release now running was staged after the
    hold, the reading says so, because the decision may be about a machine that is gone.
    """
    if not hold:
        return f", and the owner is holding {stage} for this installation"
    note = (f", and the owner is holding {stage} for this installation"
            f" ({hold['actor']}: {hold['reason']})")
    if staged_at and _before(hold.get("changed_at"), staged_at):
        note += (f"; it was set before this release was staged ({staged_at}), so it may be "
                 "a decision about the machine this release replaced")
    return note


def _before(earlier: Any, later: Any) -> bool:
    """True when both instants parse and the first is the older. Silence is not a comparison.

    A release record that predates this field, or a hold written without a zone, answers no
    question here; inventing an order would date an owner's decision on no evidence.
    """
    try:
        first = datetime.fromisoformat(str(earlier))
        second = datetime.fromisoformat(str(later))
    except (TypeError, ValueError):
        return False
    if first.tzinfo is None or second.tzinfo is None:
        return False
    return first < second


def _flatten(counts: dict[str, int]) -> str:
    if not counts:
        return "nothing queued"
    return ", ".join(f"{value} {state}" for state, value in sorted(counts.items()))


def _overall(states: dict[str, str]) -> str:
    """One word, derived in public. A degraded part says so here."""
    if DEGRADED in states.values():
        return DEGRADED
    if OPERATIONAL in states.values():
        return OPERATIONAL
    if PAUSED in states.values():
        return PAUSED
    if CONFIGURED in states.values():
        return CONFIGURED
    return UNCONFIGURED


def _notes(stages: list[StageReport], *, erasure: dict[str, Any],
           waiting: dict[str, Any], queue: dict[str, Any],
           background: dict[str, Any] | None = None,
           delivery_loop: dict[str, Any] | None = None,
           questions: dict[str, Any] | None = None) -> list[str]:
    """What an operator would ask next, from the stages that already answered."""
    notes = [f"{stage.name}: {stage.detail}" for stage in stages if stage.state == DEGRADED]
    if queue.get("stale"):
        notes.append(f"queue: a job has been waiting {queue['oldest_seconds']}s, past the "
                     f"{queue['limit_seconds']}s allowance — nothing is consuming it")
    if erasure["obligations_open"]:
        notes.append(f"erasure: {erasure['obligations_open']} obligation(s) are not "
                     "verified gone everywhere")
    if waiting["identity_candidates"]:
        notes.append(f"identity: {waiting['identity_candidates']} candidate(s) awaiting the "
                     "owner; no agent may confirm them")
    if waiting.get("lessons_for_review"):
        notes.append(f"learning: {waiting['lessons_for_review']} lesson(s) the archive no "
                     "longer stands behind are still marked active; they are taught to "
                     "nothing, and only the owner can withdraw or re-promote them")
    if waiting.get("goal_candidates"):
        notes.append(f"prospective: {waiting['goal_candidates']} reminder(s) an agent "
                     "proposed are waiting to be adopted; they schedule nothing until the "
                     "owner says so with `hermes-memory goal --activate`")
    deciding = (waiting["identity_candidates"] + waiting["candidate_assertions"]
                + waiting["erasure_intents"] + waiting["lesson_candidates"]
                + waiting["lessons_for_review"] + waiting["goal_candidates"])
    if deciding:
        # Named as a command, because a count with no door behind it is how a queue of
        # decisions ends up waited on by nobody.
        notes.append(f"owner: {deciding} decision(s) are yours alone; "
                     "`hermes-memory owner --list` shows them")
    if questions is not None and deciding and not questions["replies_enabled"]:
        # The distinct complaint: the archive is not merely waiting, it is waiting without
        # saying so. A decision nobody is asked about is a decision nobody knows to make.
        notes.append(f"questions: {deciding} decision(s) await the owner and nothing is asked "
                     f"on their channel — {questions['why']}; "
                     "`hermes-memory owner --switch-replies on --reason ...` makes the "
                     "archive ask instead of wait")
    if questions is not None and questions["expired"]:
        notes.append(f"questions: {questions['expired']} question(s) expired unanswered; the "
                     "decisions behind them are still open at `hermes-memory owner --list`")
    if questions is not None and questions["decided_elsewhere"]:
        moved = questions["decided_elsewhere"]
        names = ", ".join(sorted({str(item["decision"]) for item in moved}))
        notes.append(f"questions: {len(moved)} question(s) are still being asked about a "
                     f"decision that has already moved ({names}) — something decided it "
                     "without waiting for the answer, and the audit says what")
    delivery = next((stage for stage in stages if stage.name == "delivery"), None)
    if delivery_loop is not None and delivery_loop.get("waiting") and (
            delivery_loop.get("behind") or not delivery_loop.get("scheduled")):
        # The artifact was prepared by the pass and nothing has claimed it. Saying "the host
        # transport has not claimed" would send the reader to another machine: on this one
        # the runtime drains its own outbox, so the question is whether that loop is running
        # at all, and the note answers it from the period the owner configured. An artifact
        # prepared a moment ago under a drain that is running is not a stall, and reporting
        # one every time would make the note worth ignoring.
        notes.append(f"delivery: {delivery_loop['waiting']} artifact(s) are prepared and "
                     f"unsent — {delivery_loop['note']}")
    elif delivery is not None and delivery.evidence.get("in_flight"):
        notes.append("delivery: the framework prepared artifacts that no drain has claimed")
    if background is not None and background.get("behind") and background.get("waiting"):
        # Only said when something is actually waiting: a loop that has never run on an
        # installation with nothing due is not a problem an operator has to fix tonight.
        notes.append(f"background: {background['waiting']} reminder(s) are due and the pass "
                     f"that takes them has not been recorded — {background['note']}")
    return notes
