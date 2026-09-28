"""C12 composition: controlled raw-fact formation, from archive to backend and back.

``jobs.py`` holds work and ``worker.py`` performs it, and neither of them is on its own a
reason for anything to happen. An installation that ships a queue with no producer and no
consumer fails the P2 gate in both directions: nothing would ever be asked of the model,
and a report could still say formation was available. This module is the seam where the
tested parts meet — the document map, the queue, the budgets, the one instance-wide gate
and the backend client.

It is deliberately operator-invoked. §8.2 begins with capture plus the local index and
refuses to enable an async worker before operation reconciliation and per-request bounds
have been proven, so a bounded pass happens when a named person asks for one, and the
queue is never reported as somebody else's job. The one way it can be performed without
somebody reading the list first is a standing, capped grant the owner issued
(:mod:`processing.allowance`) — which is still a named person's decision, recorded with a
record cap, a token cap and an expiry.

The plan is a reading and the apply is the same reading re-taken: the digest covers the
work itself — which records, which route, what ceiling, how many jobs — and not the
surrounding activity that a busy machine changes every second.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Sequence

from ..backend.capabilities import PINNED_VERSION
from ..backend.document_map import VERIFIED, DocumentMap
from ..config import scoped_secret
from ..ids import digest, now
from .allowance import Allowances
from .budgets import Budget, Budgets
from .instance_gate import instance_gate, status_gate
from .jobs import JobQueue
from .routes import build_routes
from .worker import ESTIMATED_TOKENS_PER_ITEM, FormationWorker

__all__ = ["KIND", "ROUTE_NAME", "DEFAULT_BATCH", "MAX_BATCH", "MAX_JOBS",
           "FormationError", "retain_route", "processor_fingerprint", "unprojected",
           "count_unprojected", "formation_plan", "formation_apply", "formation_reconcile",
           "backend_client"]

KIND = "raw-facts"
ROUTE_NAME = "retain"
PLAN_VERSION = "formation-plan-v1"
BUDGET_SCOPE = "global"
BACKEND = "hindsight"

# Records per pass, and per job. A pass is small because it is the unit an operator
# reviews, not because the queue could not hold more.
DEFAULT_BATCH = 20
MAX_BATCH = 500
MAX_JOBS = 50


class FormationError(ValueError):
    """Formation was asked for without the configuration, the approval or the budget."""


def retain_route(settings) -> Any:
    """The route this installation's own configuration says does raw-fact extraction.

    There is no fallback to a default model: an installation that cannot name a retain
    route cannot form observations, and saying so is the whole answer.
    """
    return build_routes(settings, credentials=settings.route_credentials).by_name(ROUTE_NAME)


def processor_fingerprint(settings, route) -> str:
    """Which processor produced a piece of coverage.

    The same text through a different server, a different output cap or a different
    pinned backend is different work, so a job that succeeded under another fingerprint
    proves nothing about this one. Credentials are deliberately absent: rotating a key
    does not invalidate a fact that was already extracted.
    """
    return digest([KIND, PINNED_VERSION, settings.bank_id, route.resource, route.operation,
                   route.upstream, int(route.max_output_tokens)])[:24]


def unprojected(store, *, bank_id: str, limit: int = DEFAULT_BATCH,
                epoch: int | None = None) -> list[dict[str, str]]:
    """Live, visible evidence with no projection the current epoch can rely on.

    Ordered by arrival, so a bounded pass works through the oldest gap first. Hidden and
    deleted records are absent rather than pending: a projection of evidence the owner
    withdrew must never be re-run.
    """
    _bounded(limit)
    rows = _unprojected_rows(store, bank_id=bank_id,
                             epoch=_epoch(store, epoch), limit=limit)
    return [{"record_id": row["record_id"], "revision": row["revision"]} for row in rows]


def count_unprojected(store, *, bank_id: str, epoch: int | None = None) -> int:
    """How far behind the archive is — the number a bounded pass is chosen against."""
    row = store.db.execute(_SELECT_COUNT, _params(bank_id, _epoch(store, epoch))).fetchone()
    return int(row[0])


def formation_plan(settings, *, limit: int = DEFAULT_BATCH,
                   gate: Any = None) -> dict[str, Any]:
    """What a bounded pass would do, computed without a connection to anything.

    No store is opened on a path that could migrate it, no socket is dialled and no
    credential is printed. Every entry in ``blocking`` is a reason ``formation_apply``
    would refuse, so approving a plan shown here is approving what that call actually
    does.
    """
    _bounded(limit)
    reading = gate if gate is not None else status_gate(settings)
    store = _read_store(settings)
    try:
        blocking: list[str] = []
        if store is None:
            blocking.append(f"no canonical store at {settings.db_path}; run "
                            "`hermes-memory init` first")
        if not settings.inference_enabled:
            blocking.append("inference is switched off (HERMES_MEMORY_INFERENCE_ENABLED)")
        if not settings.hindsight_url:
            blocking.append("no Hindsight endpoint is configured, so there is nothing to "
                            "project evidence into")
        if settings.background_budget_tokens <= 0:
            blocking.append("the daily background budget is 0, so no dispatch is affordable")

        route = None
        try:
            route = retain_route(settings)
        except Exception as error:
            blocking.append(str(error))
        resource = route.resource if route else "unset"
        fingerprint = processor_fingerprint(settings, route) if route else "unset"

        # One job per record, because a job carries a single ``input_revision`` and the
        # document map keys a projection on (record, revision): a job over several records
        # would confirm them under a revision none of them has. Each job is therefore
        # allowed the prompt estimate plus the route's own output ceiling, and no more of
        # the day than the day actually has.
        per_job_tokens = int(settings.background_budget_tokens)
        if route:
            per_job_tokens = min(per_job_tokens,
                                 ESTIMATED_TOKENS_PER_ITEM + int(route.max_output_tokens))
        budget = {"resource": resource, "ledger": "absent",
                  "token_budget": int(settings.background_budget_tokens),
                  "tokens_used": 0, "remaining": int(settings.background_budget_tokens),
                  "per_job_tokens": per_job_tokens, "planned": _planned_count(limit, 0)}
        if route and reading is not None:
            tracker = Budgets(reading.store, daily={resource: Budget(
                tokens=int(settings.background_budget_tokens))}, scope=BUDGET_SCOPE)
            spent = tracker.report()["resources"].get(resource, {})
            budget.update({"ledger": "instance", "period": tracker.period,
                           "tokens_used": int(spent.get("tokens_used", 0)),
                           "remaining": int(spent.get("headroom", 0))})
        # One request may not be larger than the whole day, so a small allowance is a
        # refusal here rather than an exception from inside the worker.
        if route and budget["remaining"] < ESTIMATED_TOKENS_PER_ITEM:
            blocking.append(f"the budget left for {resource} today ({budget['remaining']:,} "
                            f"token(s)) cannot fit one item (~{ESTIMATED_TOKENS_PER_ITEM:,} "
                            "estimated)")

        pending = 0
        selected: list[dict[str, str]] = []
        queue: dict[str, int] = {}
        if store is not None:
            epoch = store.epoch()
            pending = count_unprojected(store, bank_id=settings.bank_id, epoch=epoch)
            selected = unprojected(store, bank_id=settings.bank_id, limit=limit, epoch=epoch)
            queue = _queue_counts(store)
            if not pending:
                blocking.append("nothing is unprojected: every live record already has a "
                                "verified document for this epoch")
        budget["planned"] = _planned_count(limit, pending)

        admission = _admission(reading)
        if admission["paused"]:
            blocking.append("all inference is paused by the operator")
        standing = _standing(reading, resource)
        planned = budget["planned"]
        actionable = {
            "plan_version": PLAN_VERSION, "profile": settings.profile,
            "bank_id": settings.bank_id, "route": ROUTE_NAME if route else "unset",
            "resource": resource,
            "upstream": route.upstream if route else "unset",
            "priority": route.priority if route else "unset",
            "max_output_tokens": int(route.max_output_tokens) if route else 0,
            "processor_fingerprint": fingerprint,
            "selected": [[item["record_id"], item["revision"]] for item in selected],
            "planned_jobs": planned,
            "per_job_tokens": per_job_tokens,
            "token_budget": budget["token_budget"], "tokens_used": budget["tokens_used"],
            "blocking": blocking,
        }
        note = "planning only; run with --review <digest> to perform the pass"
        if standing is not None:
            note = (f"planning only; approve this list with --review <digest>, or perform it "
                    f"under allowance {standing['id']} — {standing['records']['left']} "
                    f"record(s) and {standing['tokens']['left']:,} token(s) left of it, until "
                    f"{standing['expires_at']}")
        return {
            "ok": not blocking,
            **actionable,
            "observed_at": now(),
            "limit": int(limit),
            "unprojected": pending,
            "selected": selected,
            "queue": queue,
            "budget": budget,
            "gate": admission,
            "allowance": standing,
            "not_performed": [
                "no model request was sent while this was planned",
                "no job was written to the queue",
                "no projection row was created or confirmed",
                "nothing paused by an operator was resumed",
                "no standing allowance was charged; planning a pass spends nothing",
            ],
            "review_digest": digest([PLAN_VERSION, actionable]),
            "note": note,
        }
    finally:
        if store is not None:
            store.close()
        if reading is not None and gate is None:
            reading.close()


def formation_apply(settings, *, actor: str, review: str | None = None,
                    under_allowance: str | None = None, limit: int = DEFAULT_BATCH,
                    max_jobs: int = MAX_JOBS, client: Any = None,
                    worker_id: str | None = None) -> dict[str, Any]:
    """Perform exactly the bounded pass that was shown, and account for every part of it.

    Two doors, one work. ``review`` is the digest of a list somebody read: it has to match the
    plan as it stands *now*, because records captured in the meantime that change the
    selection, a pause somebody set, or a budget spent while the plan was being read is a
    different pass, and this refuses rather than performing the old one under the new name.
    ``under_allowance`` names a standing grant the owner issued, with a record cap, a token cap
    and an expiry — the owner has already read and decided, so what bounds this pass is that
    grant: it may not take more records than the grant has left, and its measured spend is
    charged back to it.

    Either way every refusal the plan reported still refuses here. An allowance is a way to
    skip the re-read, not a way past the budget, the pause or the empty queue.
    """
    _bounded(limit)
    if (review is None) == (under_allowance is None):
        raise FormationError(
            "a bounded pass is authorized one of two ways: --review <digest> of the list that "
            "was actually shown, or --under-allowance <id> of a standing grant the owner "
            "issued — never both, because then which decision a spend answers to would depend "
            "on the order the flags were written in")
    if not isinstance(actor, str) or not actor.strip():
        raise FormationError("an actor must be named: formation spends a shared device")
    if review is not None and (not isinstance(review, str) or not review.strip()):
        raise FormationError("run `hermes-memory form` without --review first and approve "
                             "the digest it prints")
    if not isinstance(max_jobs, int) or not 1 <= max_jobs <= MAX_JOBS:
        raise FormationError(f"max_jobs must be between 1 and {MAX_JOBS}")

    from ..storage.evidence import EvidenceStore

    with instance_gate(settings) as gate:
        proposal = formation_plan(settings, limit=limit, gate=gate)
        if review is not None and review != proposal["review_digest"]:
            raise FormationError("the review digest does not match what would happen now. "
                                 "Run `hermes-memory form` again and approve the plan it "
                                 "prints")
        if proposal["blocking"]:
            raise FormationError("refused: " + "; ".join(proposal["blocking"]))
        route = retain_route(settings)

        grants = Allowances(gate.store, owner_principal=settings.owner_principal)
        allowance: dict[str, Any] | None = None
        if under_allowance is not None:
            # Expired grants are retired here rather than answered around: this is a pass that
            # writes to the ledger anyway, and a stale `active` row is a permission that reads
            # as live to every later report.
            grants.retire_expired()
            standing, refusal = grants.authorize(
                allowance=under_allowance, records=int(proposal["planned_jobs"]),
                tokens=int(proposal["planned_jobs"]) * int(proposal["per_job_tokens"]),
                resource=route.resource)
            if standing is None:
                raise FormationError(f"refused: {refusal}")
            # The grant bounds the drain, not only the selection: a permission for five records
            # must not attempt the fifty the queue happens to already be holding.
            max_jobs = min(max_jobs, standing["records"]["left"])
            allowance = standing

        with EvidenceStore(settings.db_path) as store:
            jobs = JobQueue(store)
            # A pass begins by settling what an earlier one lost: a lease that outlived
            # its holder and work queued under a superseded epoch are both recorded as
            # what is known before any new claim.
            reconciled = jobs.reconcile()
            queued = _enqueue(jobs, proposal, route=route)
            budgets = Budgets(gate.store, daily={route.resource: Budget(
                tokens=proposal["token_budget"])}, scope=BUDGET_SCOPE)
            holder = FormationWorker(
                store=store, jobs=jobs, gate=gate, budgets=budgets,
                documents=DocumentMap(store, bank_id=settings.bank_id),
                client=client or backend_client(settings),
                routes=build_routes(settings, credentials=settings.route_credentials),
                slot_queue_s=settings.gate_queue_s,
                worker_id=(worker_id or f"form-{actor.strip()}")[:120])
            spent_before = int(budgets.used(route.resource)["tokens"])
            drained = holder.drain(max_jobs=max_jobs)
            performed = [
                f"{queued['created']} job(s) queued and up to {max_jobs} attempted",
                "usage charged to the instance gate from what the backend reported",
            ]
            if allowance is not None:
                charged = {"records": int(drained["attempted"]),
                           "tokens": max(0, int(budgets.used(route.resource)["tokens"])
                                         - spent_before)}
                settled = grants.consume(allowance["id"], **charged)
                store._audit("formation_under_allowance", allowance["id"], {
                    "actor": actor.strip(), "resource": route.resource,
                    "route": ROUTE_NAME, "review_digest": proposal["review_digest"],
                    **charged, "state": settled["state"]})
                performed.append(
                    f"allowance {allowance['id']} was charged {charged['records']} record(s) "
                    f"and {charged['tokens']:,} measured token(s) and is now "
                    f"{settled['state']}")
                allowance = settled
            return {
                "ok": True,
                "performed_at": now(),
                "actor": actor.strip(),
                "review_digest": proposal["review_digest"],
                "profile": settings.profile,
                "route": proposal["route"],
                "resource": proposal["resource"],
                "processor_fingerprint": proposal["processor_fingerprint"],
                "reconciled": reconciled,
                "queued": queued,
                "drain": drained,
                "budget": budgets.report(),
                "allowance": allowance,
                "unprojected_after": count_unprojected(store, bank_id=settings.bank_id),
                "performed": performed,
            }


def formation_reconcile(settings, *, limit: int = DEFAULT_BATCH, client: Any = None) -> dict:
    """Ask the backend what became of submissions this machine cannot account for.

    A worker that stopped waiting leaves an operation identity on the row and nothing more;
    whether the engine then ran the work is a question only it can answer. This is the door
    that closes that loop, and the reason a bounded wait is allowed to end at all — without
    it, uncertainty would be permanent rather than deferred. No model request is sent and no
    budget is spent, so this is bounded but not gated, and asks about a countable list.
    """
    from ..storage.evidence import EvidenceStore

    _bounded(limit)
    if not settings.db_path.is_file():
        raise FormationError(f"no canonical store at {settings.db_path}; run "
                             "`hermes-memory init` first")
    empty = {"settled": 0, "verified": 0, "pending": 0, "absent": 0, "unreachable": 0,
             "stopped_operations": []}
    with EvidenceStore(settings.db_path) as store:
        docs = DocumentMap(store, bank_id=settings.bank_id)
        asked = docs.outstanding(limit=limit)
        outcome = docs.reconcile(client=client or backend_client(settings), limit=limit) \
            if asked else empty
        # The answers are read from the projection ledger rather than from what this pass
        # collected. A job left uncertain by an older run of this door carries a submission
        # whose projection has long since been answered, and `outstanding` stops asking about a
        # row once it has an answer — so a door that reported only its own collection left the
        # backlog it names untouched. A queue row has no view on when its answer arrived.
        held = docs.answers(limit=max(int(limit), 1) * 10)
        # A stop is the one verdict the ledger cannot remember: it leaves the same `failed` row
        # as a failure does, and re-driving somebody's stop would overrule them. So the fresh
        # answer carries it, and it outranks the ledger's blunter reading of that row.
        stopped = list(outcome["stopped_operations"])
        stopped_set = set(stopped)
        established = held["established"]
        refused = [item for item in held["did_not_land"] if item not in stopped_set]
        ended = stopped + held["forgotten"]
        actor = settings.owner_principal or "operator"
        queue = JobQueue(store)
        settled_jobs = queue.settle_established(established, actor=actor)
        retried_jobs = queue.settle_refused(refused, actor=actor,
                                            reason="reconciled: the backend answered that this "
                                                   "submission did not reach it, so the work is "
                                                   "owed again")
        ended_jobs = queue.settle_ended(
            ended, actor=actor,
            reason="reconciled: the answer is that this work ends — the operation was stopped "
                   "at the backend, or the record was forgotten and its projection must not be "
                   "formed again")
        still_open = outcome["pending"] + outcome["unreachable"]
        return {
            "ok": True,
            "performed_at": now(),
            "profile": settings.profile,
            "bank_id": settings.bank_id,
            "limit": int(limit),
            "asked": [{"record_id": row["record_id"], "state": row["state"],
                       "operation_id": row["operation_id"]} for row in asked],
            "jobs_settled": settled_jobs,
            "jobs_retried": retried_jobs,
            "jobs_ended": ended_jobs,
            **{key: value for key, value in outcome.items() if key != "stopped_operations"},
            "stopped_operations": stopped,
            "answers": {"established": established, "refused": refused, "ended": ended},
            "projections": docs.as_dict(),
            "performed": [
                f"the backend was asked about {len(asked)} submission(s)",
                "a projection it confirmed is written down as verified coverage",
                f"{settled_jobs} queue job(s) it answered for are closed rather than left "
                "uncertain",
                f"{retried_jobs} job(s) whose submission the backend denies are counted as one "
                "failed attempt and wait for the next claim, under the attempt budget",
                f"{ended_jobs} job(s) are ended rather than re-driven: their operation was "
                "stopped, or their record was forgotten",
            ],
            "not_performed": [
                "no model request was sent, so nothing new was formed",
                "no document the backend did not name was written as present",
                "no gate slot was claimed and no budget was charged",
                "nothing whose record carries a tombstone was queued again",
            ],
            "note": (f"{still_open} submission(s) still have no answer from the backend; run "
                     "this again" if still_open else
                     "every submission this machine could account for now has the backend's "
                     "answer, and the queue rows that carried them have moved"),
        }


def backend_client(settings) -> Any:
    """The configured backend, addressed by the installation rather than by a caller.

    The API key is read from scoped secret storage at the moment of use and appears in no
    plan, receipt or log line.
    """
    from ..backend.hindsight_client import HindsightClient

    if not settings.hindsight_url:
        raise FormationError("no backend endpoint is configured")
    return HindsightClient(base_url=settings.hindsight_url, bank_id=settings.bank_id,
                           api_key=scoped_secret(settings, settings.hindsight_api_key_env))


# -- internals ---------------------------------------------------------------

_SELECT = """
        SELECT r.id AS record_id, r.revision AS revision FROM records r
        WHERE r.deleted = 0
          AND NOT EXISTS (SELECT 1 FROM record_visibility v
                           WHERE v.record_id = r.id AND v.hidden = 1)
          AND NOT EXISTS (SELECT 1 FROM backend_documents b
                           WHERE b.record_id = r.id AND b.revision = r.revision
                             AND b.backend = ? AND b.bank_id = ? AND b.state = ?
                             AND b.desired_epoch >= ?)
        """
# Ingest order, not occurrence order: the oldest gap in the derived backend is the one a
# bounded pass is expected to close first.
_ORDER = " ORDER BY r.ingested_at, r.id"
_SELECT_COUNT = "SELECT count(*) FROM (" + _SELECT + ")"


def _epoch(store, epoch: int | None) -> int:
    wanted = store.epoch() if epoch is None else int(epoch)
    if wanted < 1:
        raise FormationError("epoch must be a positive integer")
    return wanted


def _params(bank_id: str, epoch: int, *, backend: str = BACKEND) -> list[Any]:
    return [backend, bank_id, VERIFIED, epoch]


def _unprojected_rows(store, *, bank_id: str, epoch: int, limit: int) -> Sequence[Any]:
    return store.db.execute(_SELECT + _ORDER + " LIMIT ?",
                            _params(bank_id, epoch) + [limit]).fetchall()


def _read_store(settings):
    """A connection that cannot change the archive, or None if there is no archive.

    ``ReadOnlyStore`` rather than ``EvidenceStore``: opening the canonical database the
    writable way applies its migrations, and a command that promised only to look would
    have performed a schema change on the machine it was measuring.
    """
    from ..storage.evidence import ReadOnlyStore

    if not settings.db_path.is_file():
        return None
    return ReadOnlyStore(settings.db_path)


def _enqueue(jobs: JobQueue, proposal: dict[str, Any], *, route) -> dict[str, Any]:
    """One job per selected record, at that record's own revision, oldest first."""
    created = existing = 0
    for item in proposal["selected"]:
        outcome = jobs.enqueue(
            kind=KIND, inputs=[item["record_id"]], input_revision=item["revision"],
            route=route, processor_fingerprint=proposal["processor_fingerprint"],
            priority=route.priority, token_budget=int(proposal["per_job_tokens"]))
        created += int(outcome["created"])
        existing += not int(outcome["created"])
    return {"created": created, "existing": existing, "records": created + existing,
            "route": route.name}


def _bounded(limit: Any) -> int:
    """A pass covers a countable number of records, and not more than ``MAX_BATCH``."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_BATCH:
        raise FormationError(f"limit must be an integer between 1 and {MAX_BATCH}")
    return int(limit)


def _planned_count(limit: int, pending: int) -> int:
    """How many jobs the pass would queue: one per selected record, and no more."""
    return min(int(limit), int(pending)) if pending else 0


def _queue_counts(store) -> dict[str, int]:
    rows = store.db.execute(
        "SELECT state, count(*) AS n FROM processing_jobs GROUP BY state").fetchall()
    return {row[0]: int(row[1]) for row in rows}


def _admission(reading) -> dict[str, Any]:
    """What the shared ledger says about the physical models right now."""
    if reading is None:
        return {"ledger": "absent", "paused": False, "held": [], "blocked": [], "waiting": 0}
    return {"ledger": "instance", "paused": bool(reading.paused),
            "held": [f"{row['resource']} for {row['holder']}" for row in reading.held()],
            "blocked": reading.blocked_resources(),
            "waiting": sum(sum(states.values()) for states in reading.occupancy().values())}


def _standing(reading, resource: str) -> dict[str, Any] | None:
    """The owner's live standing grant covering this device, or None.

    Reported outside the approved digest deliberately: whether a pass happens to need a fresh
    read is not part of the work being read, and an approval should not expire because
    somebody granted or revoked a permission elsewhere on the machine.
    """
    if reading is None:
        return None
    try:
        return Allowances(reading.store).current(resource=resource)
    except sqlite3.OperationalError:
        # A gate ledger from before allowances existed. Reading it is not the plan's job and
        # an older file answers as what it is: no grant.
        return None
