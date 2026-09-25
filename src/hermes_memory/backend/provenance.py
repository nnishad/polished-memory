"""C6 provenance: whether a derived artifact can still be shown its evidence.

A summary, a mental model or a backend observation is a claim about a set of
records. The set changes underneath it — evidence is corrected, hidden,
forgotten, or turns out to belong to somebody else — and the artifact does not
notice. So nothing here is stored as a verdict: the answer is recomputed from
the canonical store every time it is asked, and the four possible answers are
deliberately different failures.

``complete`` may be relied on. ``partial`` says the producer admits it did not
record everything. ``unresolved`` says some citation points at nothing we can
check, which is not the same as checking it and finding it gone. ``invalid``
says a contributing dependency was forgotten, tampered with, or is not this
caller's to see — and then the synthesis is withheld and the evidence that is
still authorized is handed back in its place.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..ids import now
from ..storage.evidence import EvidenceError, EvidenceStore
from ..storage.identity import IdentityStore, evidence_accounts

__all__ = ["ProvenanceLedger", "Provenance", "COMPLETE", "PARTIAL", "UNRESOLVED", "INVALID"]

COMPLETE = "complete"
PARTIAL = "partial"
UNRESOLVED = "unresolved"
INVALID = "invalid"
_ORDER = {COMPLETE: 0, PARTIAL: 1, UNRESOLVED: 2, INVALID: 3}
MAX_CITATIONS = 500


@dataclass(frozen=True)
class Provenance:
    artifact_id: str
    kind: str
    verdict: str
    cited: int
    verified: int
    problems: tuple[str, ...] = ()
    evidence: tuple[Any, ...] = field(default_factory=tuple, compare=False)

    @property
    def usable_for_decision(self) -> bool:
        """Only a complete manifest backs a decision. Everything else is a hint."""
        return self.verdict == COMPLETE

    @property
    def withheld(self) -> bool:
        return self.verdict == INVALID

    def as_dict(self) -> dict[str, Any]:
        return {"artifact": self.artifact_id, "kind": self.kind, "verdict": self.verdict,
                "cited": self.cited, "verified": self.verified,
                "problems": list(self.problems),
                "usable_for_decision": self.usable_for_decision}


class ProvenanceLedger:
    def __init__(self, store: EvidenceStore, *, identity: IdentityStore | None = None):
        self.store = store
        self.db = store.db
        self.identity = identity

    # -- declaration ---------------------------------------------------------

    def declare(self, artifact_id: str, *, kind: str, citations: Sequence[dict],
                full_coverage: bool = True,
                db: sqlite3.Connection | None = None) -> dict[str, Any]:
        """Replace what *artifact* says it was built from.

        Replacing rather than appending: a refresh that leaves the old manifest
        in place would report provenance for a version nobody is reading. Pass
        *db* to write inside a transaction the caller already holds, which is how
        a summary and its citations become one atomic fact.
        """
        _text(artifact_id, "artifact_id", 100)
        _text(kind, "kind", 60)
        rows = self._normalize(citations)
        if len(rows) > MAX_CITATIONS:
            raise EvidenceError(
                f"{len(rows)} citations exceed the {MAX_CITATIONS} per artifact ceiling; "
                "narrow the scope instead of over-claiming support")
        connection = db or self.db
        if db is not None and not db.in_transaction:
            raise EvidenceError("provenance writes require an ambient transaction")
        owns_transaction = db is None
        if owns_transaction:
            connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("DELETE FROM derived_citations WHERE artifact_id=?",
                               (artifact_id,))
            coverage = "full" if full_coverage else "truncated"
            for record_id, quote, start, end in rows:
                connection.execute(
                    "INSERT INTO derived_citations(artifact_id, kind, coverage, record_id, "
                    "quote, quote_start, quote_end, added_at) VALUES(?,?,?,?,?,?,?,?)",
                    (artifact_id, kind, coverage, record_id, quote, start, end, now()))
            self.store._audit("provenance_declare", artifact_id,
                              {"kind": kind, "citations": len(rows),
                               "full_coverage": full_coverage})
            if owns_transaction:
                connection.execute("COMMIT")
        except BaseException:
            if owns_transaction:
                connection.execute("ROLLBACK")
            raise
        return {"artifact": artifact_id, "citations": len(rows),
                "coverage": "full" if full_coverage else "truncated"}

    def _normalize(self, citations: Sequence[dict]
                   ) -> list[tuple[str, str | None, int | None, int | None]]:
        if isinstance(citations, (str, bytes)) or not isinstance(citations, Sequence):
            raise EvidenceError("citations must be a list of {record_id, quote?} objects")
        seen: set[str] = set()
        rows: list[tuple[str, str | None, int | None, int | None]] = []
        for item in citations:
            if not isinstance(item, dict):
                raise EvidenceError("each citation is an object naming a record")
            record_id = item.get("record_id")
            if not isinstance(record_id, str) or not record_id.startswith("rec_"):
                raise EvidenceError(
                    f"citation must name a canonical record id, got {record_id!r}; a backend "
                    "document id is not evidence of anything here")
            quote = item.get("quote")
            span = self._offsets(record_id, quote) if quote is not None else (None, None)
            if record_id in seen:
                continue
            seen.add(record_id)
            rows.append((record_id, quote if span[0] is not None else None, *span))
        return rows

    def _offsets(self, record_id: str, quote: Any) -> tuple[int, int]:
        """Locate a quote now, so a later read can tell drift from presence."""
        if not isinstance(quote, str) or not quote.strip():
            raise EvidenceError("a cited quote must be nonempty text")
        record = self.store.get(record_id)
        if record is None:
            raise EvidenceError(
                f"cited evidence {record_id!r} is not retrievable, so it cannot support "
                "anything")
        hits = list(_find_all(record.text, quote))
        if len(hits) != 1:
            raise EvidenceError(
                f"the quote appears {len(hits)} times in {record_id}; cite a span, not a "
                "coincidence")
        return hits[0], hits[0] + len(quote)

    # -- resolution ----------------------------------------------------------

    def resolve(self, artifact_id: str, *, account_id: str | None = None) -> Provenance:
        """Recompute the verdict. Reads nothing that a stale row can assert.

        The worst finding wins: one forgotten dependency is not balanced out by
        twenty healthy ones, because the artifact is a single claim that all of
        them hold it up.
        """
        rows = self.db.execute(
            "SELECT * FROM derived_citations WHERE artifact_id=? ORDER BY record_id",
            (artifact_id,)).fetchall()
        if not rows:
            return Provenance(artifact_id=artifact_id, kind="unknown", verdict=UNRESOLVED,
                              cited=0, verified=0,
                              problems=("no one has declared what this was built from",))
        kind = rows[0]["kind"]
        # The producer's own admission. A manifest kept after budget truncation
        # lists what survived, not what was used, and that is a different claim.
        full_coverage = all(row["coverage"] == "full" for row in rows)
        problems: list[str] = []
        verified = 0
        allowed: list[Any] = []
        identifiers = set(self.identity.group(account_id)) if (account_id and self.identity) \
            else set()
        for row in rows:
            record = self.store.get(row["record_id"], include_hidden=True)
            if record is None:
                problems.append(f"cited evidence {row['record_id']} does not exist")
                continue
            if not self.store.live_and_visible(row["record_id"]):
                problems.append(f"cited evidence {row['record_id']} was forgotten or hidden")
                continue
            claims = evidence_accounts(self.identity, record)
            if claims and not (identifiers and claims & identifiers):
                problems.append(f"cited evidence {row['record_id']} is outside this scope")
                continue
            if row["quote"] is not None and record.text[
                    row["quote_start"]:row["quote_end"]] != row["quote"]:
                problems.append(
                    f"the quoted span in {row['record_id']} no longer matches the evidence")
                continue
            verified += 1
            allowed.append(record)
        verdict = _worst(problems, verified=verified, cited=len(rows),
                         full_coverage=full_coverage)
        return Provenance(artifact_id=artifact_id, kind=kind, verdict=verdict,
                          cited=len(rows), verified=verified, problems=tuple(problems),
                          evidence=tuple(allowed))

    def safe_evidence(self, artifact_id: str, *, account_id: str | None = None,
                      limit: int = 10) -> list[Any]:
        """What to hand back when the synthesis is withheld: the evidence itself."""
        resolved = self.resolve(artifact_id, account_id=account_id)
        return list(resolved.evidence)[:max(1, min(limit, 50))]

    def affected_by(self, record_ids: Iterable[str]) -> list[str]:
        """Artifacts that lean on any of these records, directly."""
        targets = [pk for pk in record_ids if pk]
        if not targets:
            return []
        placeholders = ",".join("?" * len(targets))
        rows = self.db.execute(
            f"SELECT DISTINCT artifact_id FROM derived_citations WHERE record_id IN ({placeholders})",
            targets).fetchall()
        return sorted(row["artifact_id"] for row in rows)

    def artifacts(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT artifact_id, kind, count(*) AS citations, max(added_at) AS declared_at "
            "FROM derived_citations GROUP BY artifact_id, kind ORDER BY artifact_id").fetchall()
        return [dict(row) for row in rows]

    def forget(self, artifact_id: str) -> int:
        """Drop a manifest. The artifact itself is someone else's row to remove."""
        _text(artifact_id, "artifact_id", 100)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.db.execute("DELETE FROM derived_citations WHERE artifact_id=?",
                                     (artifact_id,))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return int(cursor.rowcount or 0)


def _worst(problems: Sequence[str], *, verified: int, cited: int,
           full_coverage: bool = True) -> str:
    """The verdict is the worst thing found, never an average.

    An untraceable citation stays merely unresolved: we cannot say the artifact
    is wrong, only that we cannot tell. Evidence that was here and is gone, a
    span that no longer matches, or a citation that belongs to somebody else all
    say more than that — they say the synthesis is standing on nothing now.
    """
    verdict = (COMPLETE if verified == cited and verified else UNRESOLVED)
    if verdict == COMPLETE and not full_coverage:
        verdict = PARTIAL
    for problem in problems:
        if "does not exist" in problem:
            verdict = _take(verdict, UNRESOLVED)
        else:
            verdict = _take(verdict, INVALID)
    if verified < cited and verdict == COMPLETE:
        verdict = UNRESOLVED
    return verdict


def _take(current: str, candidate: str) -> str:
    return candidate if _ORDER[candidate] > _ORDER[current] else current


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise EvidenceError(f"{label} must be nonempty text of at most {maximum} characters")
    return value


def _find_all(text: str, needle: str) -> Iterable[int]:
    index = text.find(needle)
    while index != -1:
        yield index
        index = text.find(needle, index + 1)
