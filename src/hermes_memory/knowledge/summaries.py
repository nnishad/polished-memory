"""C6 summaries: a controlled hierarchy over the evidence, never a source itself.

A summary is a claim about a window of records. It carries that window, the
citations it was built from, who built it, and a revision — and it lives in its
own table. That separation is the point: a summary that entered the record set
would be cited by the next summary, and after three rounds the archive would be
mostly about itself.

Whether a summary still stands is never stored. It is recomputed from the
provenance ledger every time one is read, so a forgotten email takes its summary
out of circulation without anyone remembering to invalidate it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from ..backend.provenance import COMPLETE, INVALID, PARTIAL, ProvenanceLedger
from ..ids import digest, now, timestamp
from ..storage.evidence import EvidenceError, EvidenceStore

__all__ = ["SummaryStore", "Summary", "KINDS", "MAX_BODY"]

KINDS = ("thread", "day", "week", "project", "mental_model")
# Only an owner-approved kind reaches past the evidence it cites. Everything else
# is a re-reading of it, and labelling that as insight would let the archive
# promote its own recollection into a fact.
OWNER_APPROVED = ("mental_model",)
MAX_BODY = 8000
MAX_TITLE = 200
MAX_CITATIONS_PER_SUMMARY = 60


@dataclass(frozen=True)
class Summary:
    id: str
    scope: str
    kind: str
    title: str
    body: str
    account_id: str | None
    window_from: str | None
    window_to: str | None
    revision: int
    status: str
    processor_fingerprint: str
    epoch: int
    budget_tokens: int
    refresh_after: str | None
    created_at: str
    published_at: str
    supersedes: str | None

    @classmethod
    def from_row(cls, row) -> "Summary":
        return cls(id=row["id"], scope=row["scope"], kind=row["kind"], title=row["title"],
                   body=row["body"], account_id=row["account_id"],
                   window_from=row["window_from"], window_to=row["window_to"],
                   revision=row["revision"], status=row["status"],
                   processor_fingerprint=row["processor_fingerprint"], epoch=row["epoch"],
                   budget_tokens=row["budget_tokens"], refresh_after=row["refresh_after"],
                   created_at=row["created_at"], published_at=row["published_at"],
                   supersedes=row["supersedes"])

    def as_dict(self) -> dict[str, Any]:
        payload = {"id": self.id, "scope": self.scope, "kind": self.kind,
                   "title": self.title, "revision": self.revision, "status": self.status,
                   "window_from": self.window_from, "window_to": self.window_to,
                   "processor": self.processor_fingerprint, "epoch": self.epoch,
                   "budget_tokens": self.budget_tokens, "refresh_after": self.refresh_after,
                   "published_at": self.published_at, "supersedes": self.supersedes}
        payload["body"] = self.body
        return payload


class SummaryStore:
    def __init__(self, store: EvidenceStore, *, ledger: ProvenanceLedger | None = None,
                 owner_principal: str | None = None,
                 clock: Callable[[], float] = time.time):
        self.store = store
        self.db = store.db
        self.ledger = ledger or ProvenanceLedger(store)
        self.owner_principal = owner_principal
        self.clock = clock

    # -- publishing ----------------------------------------------------------

    def publish(self, *, scope: str, kind: str, title: str, body: str,
                citations: Sequence[dict], processor_fingerprint: str,
                window: tuple[str | None, str | None] = (None, None),
                supersedes: str | None = None, account_id: str | None = None,
                budget_tokens: int = 1200, refresh_after: str | None = None,
                approved_by: str | None = None,
                full_coverage: bool = True) -> dict[str, Any]:
        """Write one revision and its coverage manifest in a single transaction.

        The manifest is declared inside the same transaction as the summary, so a
        crash cannot leave a body on file whose citations were never recorded —
        the shape a fabricated provenance looks like from the outside.
        """
        scope, kind, title, body = _checked(scope, kind, title, body)
        if not citations or len(citations) > MAX_CITATIONS_PER_SUMMARY:
            raise EvidenceError(
                f"a summary cites between 1 and {MAX_CITATIONS_PER_SUMMARY} records; "
                "a summary of nothing is a guess")
        if not isinstance(processor_fingerprint, str) or not processor_fingerprint.strip():
            raise EvidenceError("a summary must name the processor version that wrote it")
        if not isinstance(budget_tokens, int) or not 1 <= budget_tokens <= 32_000:
            raise EvidenceError("budget_tokens must be between 1 and 32000")
        if kind in OWNER_APPROVED and approved_by != self.owner_principal:
            raise EvidenceError(
                f"a {kind} reaches past the evidence, so it needs the owner's approval; "
                f"got {approved_by!r}")
        start, end = (timestamp(window[0]) if window[0] else None,
                      timestamp(window[1]) if window[1] else None)
        if start and end and end < start:
            raise EvidenceError("the summary window ends before it begins")
        when = timestamp(refresh_after) if refresh_after else None

        summary_id = "sum_" + digest([scope, kind, title, body, processor_fingerprint,
                                      start, end])[:32]
        previous = self._target(supersedes, scope=scope, kind=kind) if supersedes else None
        revision = 1 if previous is None else int(previous["revision"]) + 1

        self.db.execute("BEGIN IMMEDIATE")
        try:
            if self.db.execute("SELECT 1 FROM summaries WHERE id=?",
                               (summary_id,)).fetchone():
                self.db.execute("COMMIT")
                return {"id": summary_id, "published": False, "revision": revision,
                        "verdict": self.verdict(summary_id)}
            stamp = now()
            self.db.execute(
                "INSERT INTO summaries(id, scope, kind, title, body, account_id, window_from, "
                "window_to, revision, supersedes, status, processor_fingerprint, epoch, "
                "budget_tokens, refresh_after, created_at, published_at, withdrawn_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?, 'published', ?,?,?,?,?,?,NULL)",
                (summary_id, scope, kind, title, body, account_id, start, end, revision,
                 supersedes, processor_fingerprint, self.store.epoch(), budget_tokens, when,
                 stamp, stamp),
            )
            self.ledger.declare(summary_id, kind=f"summary:{kind}", citations=citations,
                                full_coverage=full_coverage, db=self.db)
            if previous is not None:
                self.db.execute("UPDATE summaries SET status='withdrawn', withdrawn_at=? "
                                "WHERE id=?", (stamp, supersedes))
            self.store._audit("summary_publish", summary_id,
                              {"scope": scope, "kind": kind, "revision": revision,
                               "citations": len(citations), "processor": processor_fingerprint,
                               "coverage": "full" if full_coverage else "truncated"})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": summary_id, "published": True, "revision": revision,
                "verdict": self.verdict(summary_id)}

    def withdraw(self, summary_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        """Take a summary out of circulation. Its evidence and its history stay."""
        if self.owner_principal is None or actor != self.owner_principal:
            raise EvidenceError(f"only the owner may withdraw a summary, not {actor!r}")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise EvidenceError("reason must be nonempty text of at most 1000 characters")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.db.execute(
                "UPDATE summaries SET status='withdrawn', withdrawn_at=? WHERE id=? "
                "AND status='published'", (now(), summary_id))
            self.store._audit("summary_withdraw", summary_id,
                              {"actor": actor, "reason": reason[:200]})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": summary_id, "withdrawn": int(cursor.rowcount or 0) > 0}

    # -- reading -------------------------------------------------------------

    def get(self, summary_id: str) -> Summary | None:
        row = self.db.execute("SELECT * FROM summaries WHERE id=?", (summary_id,)).fetchone()
        return Summary.from_row(row) if row else None

    def verdict(self, summary_id: str, *, account_id: str | None = None) -> str:
        return self.ledger.resolve(summary_id, account_id=account_id).verdict

    def read(self, summary_id: str, *, account_id: str | None = None) -> dict[str, Any]:
        """Return the summary, or why it is not being returned.

        A summary whose evidence went away is withheld rather than shown with a
        disclaimer: a caveat attached to an unsupported claim is still the claim.
        """
        row = self.db.execute("SELECT * FROM summaries WHERE id=?",
                              (summary_id,)).fetchone()
        if row is None:
            return {"id": summary_id, "found": False}
        summary = Summary.from_row(row)
        if summary.status != "published":
            return {"id": summary_id, "found": True, "available": False, "body": None,
                    "verdict": self.verdict(summary_id, account_id=account_id),
                    "reason": "withdrawn"}
        provenance = self.ledger.resolve(summary_id, account_id=account_id)
        if provenance.verdict == INVALID:
            return {"id": summary.id, "found": True, "available": False, "body": None,
                    "verdict": provenance.verdict, "problems": list(provenance.problems),
                    "reason": "its evidence is no longer complete",
                    "evidence": [item.id for item in provenance.evidence]}
        return {"id": summary.id, "found": True, "available": True,
                **summary.as_dict(), "verdict": provenance.verdict,
                "usable_for_decision": provenance.usable_for_decision,
                "citations": provenance.cited, "stale": self.is_stale(summary)}

    def latest(self, scope: str, *, kind: str | None = None) -> list[Summary]:
        """The newest published revision per kind, when it still stands.

        An unsupported summary is absent rather than replaced by the last one that
        was supported. Substituting an older reading would present yesterday's
        understanding as the current state of a scope that has moved.
        """
        clauses = ["scope=?", "status='published'"]
        params: list[Any] = [scope]
        if kind:
            clauses.append("kind=?")
            params.append(kind)
        rows = self.db.execute(
            f"SELECT * FROM summaries WHERE {' AND '.join(clauses)} "
            "ORDER BY kind, published_at DESC, revision DESC", params).fetchall()
        decided: set[str] = set()
        chosen: list[Summary] = []
        for row in rows:
            summary = Summary.from_row(row)
            if summary.kind in decided:
                continue
            decided.add(summary.kind)
            if self.ledger.resolve(summary.id).verdict != INVALID:
                chosen.append(summary)
        return chosen

    def is_stale(self, summary: Summary) -> bool:
        """Out of date by its own promise, or by anything it spoke about changing.

        Sibling revisions count: a correction to an email is a *new* record, so a
        check on cited ids alone would call a summary of a corrected afternoon
        perfectly current.
        """
        if summary.refresh_after and self._moment() > summary.refresh_after:
            return True
        row = self.db.execute(
            """
            SELECT 1 FROM change_journal j JOIN records r ON r.id = j.record_id
            WHERE j.committed_at > ?
              AND (r.source, r.source_id) IN (
                    SELECT s.source, s.source_id FROM derived_citations c
                      JOIN records s ON s.id = c.record_id
                    WHERE c.artifact_id = ?)
            LIMIT 1
            """, (summary.published_at, summary.id)).fetchone()
        return row is not None

    def needs_refresh(self, *, limit: int = 25) -> list[dict[str, Any]]:
        """Published summaries that no longer say what the window says.

        Coalesced by scope and kind: fifty corrections to one afternoon ask for
        one refresh of that afternoon, not fifty.
        """
        rows = self.db.execute(
            "SELECT * FROM summaries WHERE status='published' ORDER BY published_at, id "
            "LIMIT ?", (_bounded(limit),)).fetchall()
        wanted: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            summary = Summary.from_row(row)
            provenance = self.ledger.resolve(summary.id)
            reasons = []
            if provenance.verdict == INVALID:
                reasons.append("unsupported")
            elif provenance.verdict == PARTIAL:
                reasons.append("truncated manifest")
            elif provenance.verdict != COMPLETE:
                reasons.append("unresolved")
            if self.is_stale(summary):
                reasons.append("changed since publication")
            if reasons:
                wanted[(summary.scope, summary.kind)] = {
                    "scope": summary.scope, "kind": summary.kind, "summary": summary.id,
                    "reasons": reasons, "revisions": wanted.get((summary.scope, summary.kind),
                                                                {}).get("revisions", 0) + 1}
        return list(wanted.values())

    # -- refresh bookkeeping -------------------------------------------------

    def request_refresh(self, scope: str, *, kind: str, through_at: str) -> dict[str, Any]:
        """Queue one refresh for a scope up to a point. Repeated requests coalesce."""
        _checked_scope(scope)
        if kind not in KINDS:
            raise EvidenceError(f"unknown summary kind {kind!r}; admissible are {KINDS}")
        moment = timestamp(through_at)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "INSERT INTO summary_refreshes(scope, kind, through_at, requested_at, state) "
                "VALUES(?,?,?,?, 'pending') ON CONFLICT(scope, kind, through_at) DO NOTHING",
                (scope, kind, moment, now()))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"scope": scope, "kind": kind, "through_at": moment}

    def pending_refreshes(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM summary_refreshes WHERE state='pending' ORDER BY through_at, "
            "scope, kind LIMIT ?", (_bounded(limit),)).fetchall()
        return [dict(row) for row in rows]

    def settle_refresh(self, *, scope: str, kind: str, through_at: str, ok: bool,
                       detail: str = "") -> dict[str, Any]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.db.execute(
                "UPDATE summary_refreshes SET state=?, detail=? WHERE scope=? AND kind=? "
                "AND through_at=? AND state='pending'",
                ("done" if ok else "failed", detail[:200], scope, kind, timestamp(through_at)))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"settled": int(cursor.rowcount or 0)}

    def summarize(self) -> dict[str, Any]:
        rows = self.db.execute("SELECT kind, status, count(*) AS n FROM summaries "
                               "GROUP BY kind, status").fetchall()
        return {f"{row['kind']}.{row['status']}": int(row["n"]) for row in rows}

    # -- internals -----------------------------------------------------------

    def _moment(self) -> str:
        """Wall clock in the same form the journal stores, so the comparison is honest."""
        return datetime.fromtimestamp(self.clock(), tz=timezone.utc).isoformat()

    def _target(self, summary_id: str, *, scope: str, kind: str):
        row = self.db.execute("SELECT * FROM summaries WHERE id=?", (summary_id,)).fetchone()
        if row is None:
            raise EvidenceError(f"unknown summary {summary_id!r}")
        if (row["scope"], row["kind"]) != (scope, kind):
            raise EvidenceError(
                f"summary {summary_id!r} belongs to {row['scope']}/{row['kind']}, not "
                f"{scope}/{kind}: a revision cannot move between scopes")
        return row


def _checked(scope: str, kind: str, title: str,
             body: str) -> tuple[str, str, str, str]:
    _checked_scope(scope)
    if kind not in KINDS:
        raise EvidenceError(f"unknown summary kind {kind!r}; admissible are {KINDS}")
    if not isinstance(title, str) or not title.strip() or len(title) > MAX_TITLE:
        raise EvidenceError(f"title must be nonempty text of at most {MAX_TITLE} characters")
    if not isinstance(body, str) or not body.strip() or len(body) > MAX_BODY:
        raise EvidenceError(f"body must be nonempty text of at most {MAX_BODY} characters")
    return scope, kind, " ".join(title.split()), body.strip()


def _checked_scope(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise EvidenceError("scope must be nonempty text of at most 200 characters")
    return " ".join(value.split())


def _bounded(value: int) -> int:
    return value if isinstance(value, int) and 1 <= value <= 200 else 20
