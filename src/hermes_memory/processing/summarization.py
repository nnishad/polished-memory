"""C6 composition: scoped canonical evidence to admitted synthesis and the store.

``SummaryStore`` knows how to hold a summary, ``ProvenanceLedger`` knows whether its
evidence still stands, and the broker knows how to read one. None of them knows how a
summary gets *written*, and a hierarchy with no producer is a table with no rows: the
status report would say summaries are supported while the archive produced none.

This module is the producer, and it is operator-invoked for the same reason formation is:
one reflection is a model call over private evidence, and §8.2 does not put a background
reflector on the machine before the request bounds have been shown to a person. So a pass
happens when somebody names the scope and approves the digest of what was read.

The model receives only the approved canonical input. Published citations are the
validated subset it actually used, not all planned records or an unrestricted bank
reflection. A concurrent archive revision invalidates publication.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence


from ..backend.capabilities import PINNED_VERSION
from ..ids import digest, now
from ..knowledge.summaries import KINDS, MAX_CITATIONS_PER_SUMMARY, SummaryStore
from .instance_gate import instance_gate, status_gate
from .routes import build_routes
from .synthesis import MAX_INPUT_BYTES, SYNTHESIS_CONTRACT
from .faithfulness import VERIFICATION_TOKENS, validate_publication

__all__ = ["ROUTE_NAME", "SUMMARY_ROUTE", "DEFAULT_BATCH", "MAX_BATCH", "SCOPE_FIELDS",
           "SummarizeError", "resolve_scope", "scope_records", "summary_fingerprint",
           "summarize_plan", "summarize_apply", "client_for"]

ROUTE_NAME = SUMMARY_ROUTE = "reflect"
PLAN_VERSION = "summary-plan-v3-entailment"
BUDGET_SCOPE = "global"
BACKEND = "hindsight"

# One reflection per scope per pass, and a scope is only worth reflecting over while it
# fits inside the citation ceiling the store enforces.
DEFAULT_BATCH = 25
MAX_BATCH = MAX_CITATIONS_PER_SUMMARY
# A day's summary is a claim about 24 hours of evidence; the ceiling is what stops it
# from becoming a claim about the archive.
_MAX_BODY_TOKENS = 1200
#: A reflection is a generation, not a lookup. The transport's read timeout is sized for
#: calls that answer in seconds — a health check, a submit that returns an operation id —
#: and a small model asked to write a page over the LAN is neither: one failing reflect
#: here took 12.7s before it answered at all, and a successful one is minutes of decoding.
#: Waiting past the short deadline reads as "the backend may still be running it", which
#: sends an operator to reconcile a slot for what is only a slow answer.
REFLECT_TIMEOUT_S = 180.0

SCOPE_FIELDS = ("project", "thread", "account", "source", "day", "week")
_KIND_OF = {"project": "project", "thread": "thread", "account": "day",
            "source": "day", "day": "day", "week": "week"}


class SummarizeError(ValueError):
    """A summary was asked for without the scope, the approval or the budget to write one."""


def summary_fingerprint(settings, route) -> str:
    """Which processor wrote a summary. See ``formation.processor_fingerprint``."""
    return digest(["summary", PINNED_VERSION, settings.bank_id, route.resource,
                   route.operation, route.upstream, int(route.max_output_tokens),
                   PLAN_VERSION, MAX_INPUT_BYTES, SYNTHESIS_CONTRACT,
                   settings.text_route.model if settings.text_route else None])[:24]


def resolve_scope(scope: str) -> tuple[str, str]:
    """Split ``field:value`` and refuse anything the store cannot resolve a window from."""
    if not isinstance(scope, str) or not scope.strip():
        raise SummarizeError("a summary needs a scope, e.g. --scope source:email or "
                             "--scope day:2026-09-26")
    field, _, value = scope.strip().partition(":")
    field = field.strip().casefold()
    if field not in SCOPE_FIELDS:
        raise SummarizeError(f"scope must be one of {', '.join(SCOPE_FIELDS)}; got {field!r}")
    if field in ("day", "week"):
        try:
            date.fromisoformat(value.strip())
        except ValueError:
            raise SummarizeError(f"a {field} scope is a real date, e.g. {field}:2026-09-26"
                                 f" (got {value!r})") from None
        return field, value.strip()
    if not value.strip() or len(value) > 120:
        raise SummarizeError(f"{field} needs a name of at most 120 characters")
    return field, " ".join(value.split())


def _interval(field: str, value: str) -> tuple[str, str]:
    day = date.fromisoformat(value)
    end = day + timedelta(days=6) if field == "week" else day
    # The scope names the first day of a rolling seven-day interval.
    start = day
    return (f"{start.isoformat()}T00:00:00+00:00", f"{end.isoformat()}T23:59:59+00:00")


def _where(field: str, value: str) -> tuple[str, list[Any]]:
    live = ("r.deleted = 0 AND NOT EXISTS (SELECT 1 FROM record_visibility v "
            "WHERE v.record_id = r.id AND v.hidden = 1)")
    if field == "source":
        return f"{live} AND r.source = ?", [value]
    if field in ("day", "week"):
        start, end = _interval(field, value)
        # Occurrence, not arrival: a week's summary is about the week, and a backfill
        # of last month's mail is not last month's news.
        return f"{live} AND r.occurred_at >= ? AND r.occurred_at <= ?", [start, end]
    key = {"project": "project", "thread": "thread", "account": "account"}[field]
    return (f"{live} AND COALESCE(json_extract(r.metadata, '$.{key}'), "
            f"json_extract(r.metadata, '$.tags.{key}')) = ? COLLATE NOCASE", [value])


def scope_records(store, *, scope: str, limit: int = DEFAULT_BATCH,
                  since: str | None = None) -> list[dict[str, Any]]:
    """The live, visible evidence a scope is a claim about, oldest occurrence first."""
    field, value = resolve_scope(scope)
    clause, params = _where(field, value)
    if since:
        clause += " AND COALESCE(r.occurred_at, r.ingested_at) >= ?"
        params.append(str(since))
    rows = store.db.execute(
        f"""
        SELECT r.id, r.source, r.source_id, r.occurred_at, r.revision FROM records r
        WHERE {clause}
        ORDER BY COALESCE(r.occurred_at, r.ingested_at), r.id LIMIT ?
        """, [*params, _bounded(limit)]).fetchall()
    return [{"record_id": row["id"], "source": row["source"], "source_id": row["source_id"],
             "occurred_at": row["occurred_at"], "revision": row["revision"]} for row in rows]


def _projected(store, record_ids: Sequence[str], *, bank_id: str) -> int:
    """How many of these records the backend could have been shown at all."""
    if not record_ids:
        return 0
    placeholders = ",".join("?" * len(record_ids))
    row = store.db.execute(
        f"SELECT count(*) FROM backend_documents WHERE backend = ? AND bank_id = ? "
        f"AND state = 'verified' AND record_id IN ({placeholders})",
        [BACKEND, bank_id, *record_ids]).fetchone()
    return int(row[0] or 0)


def summarize_plan(settings, *, scope: str, kind: str | None = None,
                   limit: int = DEFAULT_BATCH, since: str | None = None,
                   gate: Any = None) -> dict[str, Any]:
    """What one reflection would read and write, computed without dialling anything."""
    _bounded(limit)
    field, value = resolve_scope(scope)
    kind = kind or _KIND_OF[field]
    if kind not in KINDS:
        raise SummarizeError(f"unknown summary kind {kind!r}; admissible are {KINDS}")
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
                            "reflect over")
        if not settings.admission_url or not (settings.text_route and settings.text_route.model):
            blocking.append("scoped synthesis needs the owned admission URL and explicit text model")
        if settings.background_budget_tokens <= 0:
            blocking.append("the daily background budget is 0, so no reflection is affordable")

        route = None
        try:
            route = build_routes(settings,
                                 credentials=settings.route_credentials).by_name(ROUTE_NAME)
        except Exception as error:
            blocking.append(str(error))
        resource = route.resource if route else "unset"
        fingerprint = summary_fingerprint(settings, route) if route else "unset"

        token_budget = int(settings.background_budget_tokens)
        per_call_tokens = token_budget
        if route:
            per_call_tokens = min(token_budget, 600 + int(route.max_output_tokens))
        budget = {"resource": resource, "ledger": "absent", "token_budget": token_budget,
                  "tokens_used": 0, "remaining": token_budget,
                  "per_call_tokens": per_call_tokens}
        if route and reading is not None:
            from .budgets import Budget, Budgets

            tracker = Budgets(reading.store, daily={resource: Budget(
                tokens=token_budget)}, scope=BUDGET_SCOPE)
            spent = tracker.report()["resources"].get(resource, {})
            budget.update({"ledger": "instance",
                           "tokens_used": int(spent.get("tokens_used", 0)),
                           "remaining": int(spent.get("headroom", 0))})

        selected: list[dict[str, Any]] = []
        unprojected = 0
        refreshes: list[dict[str, Any]] = []
        window: tuple[str | None, str | None] = (None, None)
        if store is not None:
            selected = scope_records(store, scope=scope, limit=limit + 1, since=since)
            truncated = len(selected) > limit
            selected = selected[:limit]
            record_ids = [item["record_id"] for item in selected]
            unprojected = len(record_ids) - _projected(store, record_ids,
                                                       bank_id=settings.bank_id)
            stamps = [str(item["occurred_at"] or "") for item in selected if item["occurred_at"]]
            evidence_bytes = sum(len(store.get(item["record_id"]).text.encode()) + 500
                                 for item in selected)
            if evidence_bytes > MAX_INPUT_BYTES:
                blocking.append("canonical synthesis input exceeds its byte ceiling; narrow the scope")
            # Conservative byte-as-token estimate for generation AND verification;
            # the latter also carries the bounded draft. Each actual hop is gated.
            per_call_tokens = (2 * (evidence_bytes + 4096) + 16_000 + VERIFICATION_TOKENS
                               + (int(route.max_output_tokens) if route else 0))
            window = (stamps[0] if stamps else None, stamps[-1] if stamps else None)
            if not record_ids:
                blocking.append(f"nothing in this archive belongs to {scope!r}: a summary of "
                                "no evidence is a guess")
            if truncated:
                blocking.append(f"{scope!r} holds more than {limit} live record(s); raise "
                                "--limit or narrow the scope, do not summarize a prefix "
                                "and call it the scope")
            refreshes = _pending(store, scope=scope, kind=kind)
        budget["per_call_tokens"] = per_call_tokens
        if route and budget["remaining"] < per_call_tokens:
            blocking.append(f"the budget left for {resource} today ({budget['remaining']:,} "
                            f"token(s)) cannot fit synthesis and verification ({per_call_tokens:,} reserved)")
        admission = _admission(reading)
        if admission["paused"]:
            blocking.append("all inference is paused by the operator")

        actionable = {
            "plan_version": PLAN_VERSION, "profile": settings.profile,
            "bank_id": settings.bank_id, "scope": f"{field}:{value}", "kind": kind,
            "route": ROUTE_NAME if route else "unset", "resource": resource,
            "upstream": route.upstream if route else "unset",
            "max_output_tokens": int(route.max_output_tokens) if route else 0,
            "processor_fingerprint": fingerprint,
            "window": list(window),
            "selected": [[item["record_id"], item["revision"]] for item in selected],
            "unprojected": unprojected,
            "per_call_tokens": per_call_tokens, "token_budget": budget["token_budget"],
            "tokens_used": budget["tokens_used"], "blocking": blocking,
        }
        return {
            "ok": not blocking,
            **actionable,
            "observed_at": now(),
            "limit": int(limit),
            "records": len(selected),
            "selected": selected,
            "refreshes": refreshes,
            "budget": budget,
            "gate": admission,
            "coverage": "full" if selected and not truncated else "truncated",
            "not_performed": [
                "no model request was sent while this was planned",
                "no summary was written or superseded",
                "no refresh promise was settled",
                "nothing paused by an operator was resumed",
            ],
            "review_digest": digest([PLAN_VERSION, actionable]),
            "note": "planning only; run with --review <digest> to write this summary",
        }
    finally:
        if store is not None:
            store.close()
        if reading is not None and gate is None:
            reading.close()


def summarize_apply(settings, *, scope: str, kind: str | None = None, review: str,
                    actor: str, limit: int = DEFAULT_BATCH, since: str | None = None,
                    client: Any = None, title: str | None = None) -> dict[str, Any]:
    """Perform the one reflection that was shown, and file it with its window.

    The digest is re-taken against the archive as it stands now: evidence captured since
    the plan was read changes what the summary would be a claim about, and writing the old
    approval over a new window is exactly the thing the two-phase doors exist to prevent.
    """
    _bounded(limit)
    if not isinstance(actor, str) or not actor.strip():
        raise SummarizeError("an actor must be named: a summary is attributed to whoever "
                             "asked the model to write it")
    if not isinstance(review, str) or not review.strip():
        raise SummarizeError("run `hermes-memory summarize` without --review first and "
                             "approve the digest it prints")
    from ..storage.evidence import EvidenceStore

    with instance_gate(settings) as gate:
        proposal = summarize_plan(settings, scope=scope, kind=kind, limit=limit,
                                  since=since, gate=gate)
        if review != proposal["review_digest"]:
            raise SummarizeError("the review digest does not match what would happen now. "
                                 "Run `hermes-memory summarize` again and approve the plan "
                                 "it prints")
        if proposal["blocking"]:
            raise SummarizeError("refused: " + "; ".join(proposal["blocking"]))
        route = build_routes(settings,
                             credentials=settings.route_credentials).by_name(ROUTE_NAME)
        holder = client or client_for(settings)
        window_end = proposal["window"][1]

        with EvidenceStore(settings.db_path) as store:
            from .budgets import Budget, Budgets

            budgets = Budgets(gate.store, daily={route.resource: Budget(
                tokens=proposal["token_budget"])}, scope=BUDGET_SCOPE)
            habits = SummaryStore(store, owner_principal=settings.owner_principal)
            stamp = store.watermark()
            supplied = []
            for item in proposal["selected"]:
                record = store.get(item["record_id"])
                if record is None or record.revision != item["revision"]:
                    raise SummarizeError("approved input is no longer live at its reviewed revision")
                supplied.append({"record_id": record.id, "revision": record.revision,
                                 "source": record.source, "text": record.text,
                                 "role": record.metadata.get("role"),
                                 "occurred_at": record.occurred_at})
            outcome, failure = _reflect(gate, route, proposal, budgets=budgets, holder=holder,
                                         evidence=supplied)
            if store.watermark() != stamp:
                failure = "canonical authority changed during synthesis; output withheld"
            if failure:
                # The promise stays in the ledger as a failed refresh rather than
                # vanishing: the reason it did not happen is the thing an operator
                # reading `status` needs, and silence looks like there was nothing to do.
                settled = habits.settle_refresh(
                    scope=proposal["scope"], kind=proposal["kind"],
                    through_at=window_end or now(), ok=False, detail=failure)
                return {"ok": False, "performed_at": now(), "actor": actor.strip(),
                        "review_digest": review, "scope": proposal["scope"],
                        "kind": proposal["kind"], "refused": failure[:300],
                        "refreshes_settled": settled["settled"],
                        "budget": budgets.report()}
            body = str(outcome.get("text") or "").strip()
            if not body:
                raise SummarizeError("the backend returned no text, so there is nothing to "
                                     "file; the window is left unsummarized rather than "
                                     "filled with a placeholder")
            spent = _used(outcome)
            citations = outcome.get("record_ids")
            if (not isinstance(citations, list) or not citations
                    or any(identifier not in {item["record_id"] for item in supplied}
                           for identifier in citations)):
                raise SummarizeError("synthesis returned no valid actual input references")
            written = habits.publish(
                scope=proposal["scope"], kind=proposal["kind"],
                title=(title or _default_title(proposal))[:200], body=body[:8000],
                citations=[{"record_id": identifier} for identifier in sorted(set(citations))],
                processor_fingerprint=proposal["processor_fingerprint"],
                window=proposal["window"],
                budget_tokens=min(_MAX_BODY_TOKENS, int(route.max_output_tokens) or
                                  _MAX_BODY_TOKENS),
                refresh_after=_next_window(proposal), approved_by=actor.strip(),
                full_coverage=proposal["coverage"] == "full")
            settled = habits.settle_refresh(
                scope=proposal["scope"], kind=proposal["kind"],
                through_at=window_end or now(), ok=True,
                detail=f"rev {written['revision']}")
            return {
                "ok": True, "performed_at": now(), "actor": actor.strip(),
                "review_digest": review, "route": proposal["route"],
                "resource": proposal["resource"],
                "processor_fingerprint": proposal["processor_fingerprint"],
                "summary": written["id"], "published": written["published"],
                "already_on_file": not written["published"],
                "revision": written["revision"], "verdict": written["verdict"],
                "scope": proposal["scope"], "kind": proposal["kind"],
                "window": list(proposal["window"]),
                "citations": len(set(citations)),
                "planned_inputs": len(proposal["selected"]),
                "coverage": proposal["coverage"],
                "refreshes_settled": settled["settled"],
                "tokens_charged": spent,
                "budget": budgets.report(),
                "performed": [
                    "scoped generation and separate per-claim entailment verification",
                    f"{len(proposal['selected'])} canonical record(s) supplied as evidence",
                    f"{len(set(citations))} record(s) cited by semantically checked claims",
                    "usage accounted by the owned admission gate",
                ],
            }


def client_for(settings) -> Any:
    """Explicit synthesis through the installation's owned admission endpoint."""
    from .synthesis import ScopedSynthesizer

    return ScopedSynthesizer(base_url=settings.admission_url,
                             credential=settings.route_credentials.get(ROUTE_NAME),
                             model=settings.text_route.model if settings.text_route else None,
                             timeout=REFLECT_TIMEOUT_S)


# -- internals ---------------------------------------------------------------

def _reflect(gate, route, proposal, *, budgets, holder, evidence) -> tuple[dict[str, Any], str]:
    """One bounded reflection, asked without claiming the device it is answered on.

    A pause and an unaffordable budget are refusals rather than something to attempt anyway,
    and so is a device blocked by a request nobody has answered for: the physical device is
    shared, and a request that goes out while the operator has paused inference is not a
    small procedural slip. Occupation by a *live* request is not a refusal — see the note
    below on why the outer claim was the deadlock.
    """
    import time

    from ..backend.hindsight_client import HindsightError, HindsightUnavailable
    from .budgets import BudgetExhausted

    if gate.paused:
        return {}, "all inference is paused by the operator"
    try:
        budgets.admit(route.resource, estimated_tokens=proposal["per_call_tokens"])
    except BudgetExhausted as error:
        return {}, str(error)
    if gate.unresolved_for(route.resource):
        return {}, (f"{route.resource} is blocked by a request nobody has answered for, and "
                    "waiting would not free it — `hermes-memory gate --resolve` settles it")
    # The owned HTTP gate admits and accounts the model hop. Reserving its device
    # again in this outer coordinator would deadlock that admission.
    started = time.monotonic()
    outcome: dict[str, Any] = {}
    tokens, failure = 0, ""
    try:
        if not callable(getattr(holder, "synthesize", None)):
            raise SummarizeError("summary client must synthesize from explicit canonical input")
        outcome = holder.synthesize(_question(proposal), evidence=evidence,
                                    max_tokens=int(route.max_output_tokens), budget="low")
        tokens = _used(outcome)
        validate_publication(outcome, evidence)
    except HindsightUnavailable as error:
        # The backend's own admissions say whether the device is free; nothing here holds it.
        failure = str(error)
    except (HindsightError, SummarizeError) as error:
        failure = str(error)
    if tokens and not outcome.get("admission_accounted"):
        gate.charge(route.resource, tokens=tokens, seconds=time.monotonic() - started,
                    note=f"reflect {proposal['scope']}")
    return outcome, failure


def _question(proposal: dict[str, Any]) -> str:
    """What we ask the model, in the terms the plan allows it to answer in.

    It names the window and ceiling. The synthesis adapter separately supplies the
    approved canonical evidence; no backend retrieval may expand that input.
    """
    start, end = proposal["window"]
    span = f"between {start} and {end}" if start and end else "of no fixed date"
    return (f"Summarize what is known about {proposal['scope']} ({proposal['kind']}), "
            f"covering the {proposal['records']} archived item(s) {span}. "
            f"Say only what the evidence supports, keep it under "
            f"{proposal['max_output_tokens']} tokens, and list nothing that is not "
            "attributable.")


def _default_title(proposal: dict[str, Any]) -> str:
    start, _end = proposal["window"]
    return f"{proposal['kind']} of {proposal['scope']}" + (f" as of {start}" if start else "")


def _next_window(proposal: dict[str, Any]) -> str | None:
    """When this reading should be considered stale: the end of the window it covers.

    A summary of a week is out of date the moment the week is over and something new
    happened, so it asks to be re-read then rather than on an invented schedule.
    """
    end = proposal["window"][1]
    if not end:
        return None
    try:
        return (datetime.fromisoformat(str(end)) + timedelta(hours=1)).astimezone(
            timezone.utc).isoformat()
    except ValueError:
        return None


def _used(outcome: dict[str, Any]) -> int:
    return int(outcome.get("input_tokens") or 0) + int(outcome.get("output_tokens") or 0)


def _pending(store, *, scope: str, kind: str) -> list[dict[str, Any]]:
    rows = store.db.execute(
        "SELECT through_at, requested_at FROM summary_refreshes WHERE state='pending' "
        "AND scope=? AND kind=? ORDER BY through_at", (scope, kind)).fetchall()
    return [{"through_at": row["through_at"], "requested_at": row["requested_at"]}
            for row in rows]


def _read_store(settings):
    from ..storage.evidence import ReadOnlyStore

    if not settings.db_path.is_file():
        return None
    return ReadOnlyStore(settings.db_path)


def _admission(reading) -> dict[str, Any]:
    if reading is None:
        return {"ledger": "absent", "paused": False, "held": [], "blocked": [], "waiting": 0}
    return {"ledger": "instance", "paused": bool(reading.paused),
            "held": [f"{row['resource']} for {row['holder']}" for row in reading.held()],
            "blocked": reading.blocked_resources(),
            "waiting": sum(sum(states.values()) for states in reading.occupancy().values())}


def _bounded(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_BATCH:
        raise SummarizeError(f"limit must be an integer between 1 and {MAX_BATCH}")
    return int(value)
