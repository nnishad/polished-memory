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

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from ..ids import now, timestamp
from ..processing.instance_gate import instance_gate, status_gate
from ..processing.resource_gate import UNCERTAIN, WAITING, ResourceGate
from ..sources.sync import COVERAGE_STATES
from ..storage.evidence import EvidenceError

__all__ = ["StatusReporter", "StageReport", "STATUS_STATES", "REPORTED_STAGES",
           "snapshot"]

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
UNPROVEN = ("uncertain", "accepted_unverified")
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

    # -- report --------------------------------------------------------------

    def report(self, *, profile: str | None = None) -> dict[str, Any]:
        """Everything, as one snapshot of one connection."""
        with snapshot(self.db):
            stages = [self._by_name[name]() for name in REPORTED_STAGES]
            erasure = self.erasure_backlog()
            waiting = self.pending_confirmations()
            queue = self.queue_age()
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
            "queue": queue,
            "notes": _notes(stages, erasure=erasure, waiting=waiting, queue=queue),
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
            "SELECT source, count(*) AS n FROM source_gaps WHERE cleared_at IS NULL "
            "GROUP BY source ORDER BY source").fetchall()
        open_gaps = {row["source"]: int(row["n"]) for row in gaps}
        unhealthy = sorted(row["source"] for row in sources
                           if row["coverage_state"] in UNHEALTHY_COVERAGE)
        records = int(self.db.execute(
            "SELECT count(*) FROM records WHERE deleted=0").fetchone()[0])
        state = (UNCONFIGURED if not sources else
                 PAUSED if self._all_stopped("capture", paused) else
                 DEGRADED if unhealthy or open_gaps else OPERATIONAL)
        return StageReport(
            "capture", state,
            f"{len(sources)} source(s) registered, "
            f"{len([row for row in sources if row['source'] in paused])} paused, "
            f"{sum(open_gaps.values())} open gap(s), {records} live record(s)",
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
        held = bool(self.gate is not None and self.gate.paused)
        stuck = {state: int(queue.get(state, 0)) for state in STUCK_JOBS
                 if queue.get(state)}
        # Two different things a queue can say. ``running`` means a lease has a name on
        # it, so work is happening now; ``waiting`` means rows exist that nobody has
        # claimed, which is a stage that is set up rather than one that is operating.
        # Calling the second one operational is the false readiness this report exists to
        # avoid, because nothing in this installation claims a job by itself.
        waiting = _any(queue, WAITING_JOBS) and not _any(queue, RUNNING_JOBS)
        running = _any(queue, RUNNING_JOBS)
        state = (DEGRADED if stuck else
                 PAUSED if held or self._all_stopped("formation", paused) else
                 OPERATIONAL if counts or running else
                 CONFIGURED if waiting else UNCONFIGURED)
        detail = (f"{counts.get('confirmed', 0)} confirmed / "
                  f"{counts.get('candidate', 0)} candidate assertion(s); "
                  f"queue {_flatten(queue)}")
        if held:
            detail += ", and the owner is holding inference for this installation"
        elif waiting:
            detail += ("; only `hermes-memory form` works this queue — nothing drains it "
                       "by itself")
        return StageReport(
            "observations", state, detail,
            {"assertions": dict(counts), "queue": dict(queue), "stuck": stuck,
             "paused_sources": sorted(paused), "instance_hold": held,
             "unattended": False, "draining": running,
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
        """The framework makes artifacts; the host sends them. This says which is stuck."""
        counts = self._grouped("SELECT state, count(*) AS n FROM outbox GROUP BY state")
        in_flight = sum(int(counts.get(state, 0)) for state in IN_FLIGHT)
        unproven = sum(int(counts.get(state, 0)) for state in UNPROVEN)
        overdue = int(self.db.execute(
            "SELECT count(*) FROM outbox WHERE state IN ({}) AND expires_at IS NOT NULL "
            "AND expires_at <= ?".format(",".join("?" * len(IN_FLIGHT))),
            [*IN_FLIGHT, now()]).fetchone()[0])
        paused = self._pauses("proactivity")
        held = self.db.execute(
            "SELECT 1 FROM runtime_controls WHERE scope='global' AND stage='delivery' "
            "AND state='paused'").fetchone()
        # An artifact past its own expiry is not "in flight": it is a transport that
        # stopped coming, and the report says so rather than counting it as busy work.
        state = (DEGRADED if unproven or overdue else
                 PAUSED if held or self._all_stopped("proactivity", paused) else
                 OPERATIONAL if in_flight or counts.get("confirmed") else UNCONFIGURED)
        return StageReport(
            "delivery", state,
            f"{in_flight} artifact(s) waiting on the transport, {unproven} without a "
            f"delivery proof, {counts.get('confirmed', 0)} confirmed"
            + (", and the owner is holding delivery for this installation" if held else ""),
            {"by_state": dict(counts), "in_flight": in_flight, "unproven": unproven,
             "past_expiry": overdue, "paused_sources": sorted(paused),
             "instance_hold": bool(held),
             "owner_principal": getattr(self.settings, "owner_principal", None)})

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
                                "waiting": 0, "blocked": [], "usage": None})
        held = self.gate.held()
        occupancy = {resource: dict(states)
                     for resource, states in self.gate.occupancy().items()}
        uncertain = sum(int(states.get(UNCERTAIN, 0)) for states in occupancy.values())
        waiting = sum(int(states.get(WAITING, 0)) for states in occupancy.values())
        taken = sum(sum(states.values()) for states in occupancy.values())
        # A pause on a gate nobody has used yet is still somebody's decision, so it is
        # reported before the "never reserved" case rather than hidden behind it.
        state = (DEGRADED if uncertain else
                 PAUSED if self.gate.paused else
                 OPERATIONAL if taken or self.gate.ever_used() else UNCONFIGURED)
        return StageReport(
            "resource_gate", state,
            f"{taken} slot(s) occupied, {waiting} waiting"
            + (f", {uncertain} unresolved reservation(s)" if uncertain else ""),
            {"occupancy": occupancy,
             "held": [{"resource": row["resource"], "route": row["route"],
                       "holder": row["holder"], "priority": int(row["priority"]),
                       "held_for_seconds": round(_age_since(row["acquired_at"]), 1),
                       "lease_seconds_left": round(_seconds_until(row["lease_until"]), 1)}
                      for row in held],
             "uncertain": uncertain, "waiting": waiting,
             "blocked": self.gate.blocked_resources(),
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
        return {
            "identity_candidates": int(self.db.execute(
                "SELECT count(*) FROM identity_candidates WHERE state='pending'"
            ).fetchone()[0]),
            "candidate_assertions": int(self.db.execute(
                "SELECT count(*) FROM assertions WHERE status='candidate'").fetchone()[0]),
            "erasure_intents": int(self.db.execute(
                "SELECT count(*) FROM erasure_ledger WHERE state='awaiting_confirmation'"
            ).fetchone()[0]),
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

    # -- primitives ----------------------------------------------------------

    @property
    def capture_only(self) -> bool:
        return bool(getattr(self.settings, "capture_only", False))

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
           waiting: dict[str, Any], queue: dict[str, Any]) -> list[str]:
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
    delivery = next((stage for stage in stages if stage.name == "delivery"), None)
    if delivery is not None and delivery.evidence.get("in_flight"):
        notes.append("delivery: the framework prepared artifacts that the host transport "
                     "has not claimed")
    return notes
