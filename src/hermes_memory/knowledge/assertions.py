"""C6 assertions: one typed claim, its interval, and the exact span behind it.

An assertion is the smallest unit of knowledge that can be checked afterwards.
Three properties make that possible and none of them are optional: a subject and
predicate that can be compared against another row, a validity interval (so
"moved to Thursday" and "moved to Friday" can both have been true), and a quote
with offsets into a canonical record (so the claim can be re-read at the source
rather than trusted).

A model may propose any of this. What it may not do is confirm: the only claims
that arrive already standing are the ones where a human said the thing and we
still hold the sentence they said it in, checked byte for byte.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from ..ids import digest, interval_contains, now, timestamp

if TYPE_CHECKING:                                  # pragma: no cover
    from .contradictions import Contradiction
from ..storage.evidence import EvidenceError, EvidenceStore

__all__ = ["AssertionStore", "Assertion", "KINDS", "EVIDENCE_KINDS",
           "CANDIDATE", "CONFIRMED", "SUPERSEDED", "RETRACTED"]

KINDS = ("belief", "preference", "fact", "measurement")
# Only these two may be created already standing, and only with a quote that
# matches the evidence byte for byte. The rest are hypotheses until the owner
# says otherwise.
EVIDENCE_KINDS = ("owner_declared", "explicit_statement", "observed_pattern", "derived")
CONFIRMABLE = ("owner_declared", "explicit_statement")
# Written into confirmed_by when the quote, not a person, is what carries the
# claim. Attributing that to whoever proposed it would be a forgery with good
# manners.
EVIDENCE_PROOF = "evidence-quote"

CANDIDATE = "candidate"
CONFIRMED = "confirmed"
SUPERSEDED = "superseded"
RETRACTED = "retracted"

_TERMS = re.compile(r"[^\W]+", re.UNICODE)

MAX_VALUE = 500
MAX_QUOTE = 2000
MAX_TERM = 100


@dataclass(frozen=True)
class Assertion:
    id: str
    subject: str
    predicate: str
    value: str
    kind: str
    unit: str | None
    evidence_kind: str
    record_id: str
    quote: str
    quote_start: int
    quote_end: int
    valid_from: str | None
    valid_to: str | None
    status: str
    created_by: str
    confirmed_by: str | None
    created_at: str
    confirmed_at: str | None
    supersedes: str | None
    revision: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Assertion":
        return cls(id=row["id"], subject=row["subject"], predicate=row["predicate"],
                   value=row["value"], kind=row["category"], unit=row["unit"],
                   evidence_kind=row["evidence_kind"], record_id=row["record_id"],
                   quote=row["quote"], quote_start=row["quote_start"],
                   quote_end=row["quote_end"], valid_from=row["valid_from"],
                   valid_to=row["valid_to"], status=row["status"],
                   created_by=row["created_by"], confirmed_by=row["confirmed_by"],
                   created_at=row["created_at"], confirmed_at=row["confirmed_at"],
                   supersedes=row["supersedes"], revision=row["revision"])

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "subject": self.subject, "predicate": self.predicate,
                "value": self.value, "kind": self.kind, "unit": self.unit,
                "evidence_kind": self.evidence_kind, "record_id": self.record_id,
                "quote": self.quote, "quote_start": self.quote_start,
                "quote_end": self.quote_end, "valid_from": self.valid_from,
                "valid_to": self.valid_to, "status": self.status,
                "confirmed_by": self.confirmed_by, "supersedes": self.supersedes,
                "revision": self.revision}


class AssertionStore:
    def __init__(self, store: EvidenceStore, *, owner_principal: str | None = None):
        self.store = store
        self.db = store.db
        self.owner_principal = owner_principal

    # -- authoring -----------------------------------------------------------

    def propose(self, *, subject: str, predicate: str, value: str, kind: str,
                evidence_kind: str, record_id: str, quote: str, proposed_by: str,
                unit: str | None = None, valid_from: str | None = None,
                valid_to: str | None = None, span: tuple[int, int] | None = None,
                supersedes: str | None = None) -> dict[str, Any]:
        """Record one claim against its evidence. Idempotent per content.

        Nothing is written until the quote has been found in the record it cites,
        so a paraphrase cannot enter the archive wearing a citation it does not
        have.
        """
        subject, predicate, value, unit = _terms(subject, predicate, value, unit)
        if kind not in KINDS:
            raise EvidenceError(f"unknown assertion kind {kind!r}; admissible are {KINDS}")
        if evidence_kind not in EVIDENCE_KINDS:
            raise EvidenceError(
                f"unknown evidence kind {evidence_kind!r}; admissible are {EVIDENCE_KINDS}")
        if kind == "measurement":
            _measurement(value, unit)
        elif unit:
            raise EvidenceError(f"only a measurement carries a unit, not a {kind}")
        start, end, quote = _span(self.store, record_id, quote, span=span)
        window = _window(valid_from, valid_to)
        superseded = self._target(supersedes, subject=subject,
                                 predicate=predicate) if supersedes else None

        assertion_id = "asr_" + digest([subject, predicate, value, unit, record_id,
                                        start, end, window[0], window[1]])[:32]
        status = CONFIRMED if evidence_kind in CONFIRMABLE else CANDIDATE
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute("SELECT * FROM assertions WHERE id=?",
                                       (assertion_id,)).fetchone()
            if existing:
                self.db.execute("COMMIT")
                return {"id": assertion_id, "status": existing["status"], "created": False,
                        "quote_start": start, "quote_end": end}
            stamp = now()
            if superseded is not None:
                self.db.execute("UPDATE assertions SET status=?, revision=revision+1 "
                                "WHERE id=?", (SUPERSEDED, supersedes))
            self.db.execute(
                "INSERT INTO assertions(id, subject, predicate, value, category, unit, "
                "evidence_kind, record_id, quote_start, quote_end, quote, valid_from, valid_to, "
                "status, created_by, confirmed_by, created_at, confirmed_at, supersedes, "
                "revision) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (assertion_id, subject, predicate, value, kind, unit, evidence_kind,
                 record_id, start, end, quote, window[0], window[1], status, proposed_by,
                 proposed_by if status == CONFIRMED else None, stamp,
                 stamp if status == CONFIRMED else None, supersedes,
                 1 + (superseded["revision"] if superseded is not None else 0)),
            )
            self.store._audit("assertion_propose", assertion_id,
                              {"kind": kind, "evidence_kind": evidence_kind,
                               "status": status, "by": proposed_by, "supersedes": supersedes})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": assertion_id, "status": status, "created": True,
                "quote_start": start, "quote_end": end,
                "confirmed_by": proposed_by if status == CONFIRMED else None,
                "note": (None if status == CONFIRMED else
                         "held as a candidate: a pattern someone noticed is not something "
                         "the archive asserts until the owner confirms it")}

    def confirm(self, *, assertion_id: str, actor: str, reason: str) -> dict[str, Any]:
        """Owner-only. A candidate becomes something the archive stands behind."""
        self._require_owner(actor)
        _reason(reason)
        row = self._target(assertion_id)
        if row["status"] == CONFIRMED:
            return {"id": assertion_id, "status": CONFIRMED, "changed": False,
                    "confirmed_by": row["confirmed_by"]}
        if row["status"] == RETRACTED:
            raise EvidenceError(
                f"assertion {assertion_id!r} was retracted; propose it again on current "
                "evidence rather than un-deleting a decision")
        supported, why = self.support(Assertion.from_row(row))
        if not supported:
            raise EvidenceError(
                f"cannot confirm an assertion its own evidence no longer supports: {why}")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE assertions SET status=?, confirmed_by=?, confirmed_at=?, "
                "revision=revision+1 WHERE id=?", (CONFIRMED, actor, now(), assertion_id))
            self.store._audit("assertion_confirm", assertion_id,
                              {"actor": actor, "reason": reason[:200]})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": assertion_id, "status": CONFIRMED, "changed": True,
                "confirmed_by": actor}

    def retract(self, *, assertion_id: str, actor: str, reason: str) -> dict[str, Any]:
        """Owner-only. The claim stops being asserted; the evidence is untouched."""
        self._require_owner(actor)
        _reason(reason)
        self._target(assertion_id)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE assertions SET status=?, confirmed_by=NULL, revision=revision+1 "
                "WHERE id=?", (RETRACTED, assertion_id))
            self.store._audit("assertion_retract", assertion_id,
                              {"actor": actor, "reason": reason[:200]})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": assertion_id, "status": RETRACTED, "reason": reason}

    # -- reading -------------------------------------------------------------

    def get(self, assertion_id: str) -> Assertion | None:
        row = self.db.execute("SELECT * FROM assertions WHERE id=?",
                              (assertion_id,)).fetchone()
        return Assertion.from_row(row) if row else None

    def support(self, assertion: Assertion) -> tuple[bool, str]:
        """Can this claim still be shown the sentence it came from?

        Status records what was decided. Support says whether that decision is
        still checkable, which is a different question and the one that has to be
        asked on every read.
        """
        if not self.store.live_and_visible(assertion.record_id):
            return False, "the cited evidence is hidden or forgotten"
        record = self.store.get(assertion.record_id)
        if record is None:
            return False, "the cited evidence could not be read"
        if record.text[assertion.quote_start:assertion.quote_end] != assertion.quote:
            return False, "the quoted span no longer matches the cited evidence"
        return True, ""

    def current(self, *, subject: str | None = None, predicate: str | None = None,
                at: str | None = None, include_candidates: bool = False) -> list[Assertion]:
        """What the archive asserts, with support checked on the way out.

        An assertion whose evidence has been forgotten is not returned here in
        any mode: that is the whole arrangement between knowledge and forgetting.
        """
        clauses, params = [], []
        if subject:
            clauses.append("subject=?")
            params.append(_term(subject, "subject"))
        if predicate:
            clauses.append("predicate=?")
            params.append(_term(predicate, "predicate"))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.execute(
            f"SELECT * FROM assertions{where} ORDER BY subject, predicate, valid_from, id",
            params).fetchall()
        wanted = {CONFIRMED} | ({CANDIDATE} if include_candidates else set())
        found = []
        for row in rows:
            assertion = Assertion.from_row(row)
            if assertion.status not in wanted:
                continue
            if at and not _contains(assertion, at):
                continue
            if not self.support(assertion)[0]:
                continue
            found.append(assertion)
        return found

    def needs_review(self) -> list[dict[str, Any]]:
        """Claims that still say they stand but can no longer show their evidence.

        Reported rather than silently dropped or deleted: a forgotten email does
        not change what the owner believed, it changes what we can prove.
        """
        rows = self.db.execute(
            "SELECT * FROM assertions WHERE status IN (?, ?) ORDER BY created_at, id",
            (CONFIRMED, CANDIDATE)).fetchall()
        review = []
        for row in rows:
            assertion = Assertion.from_row(row)
            supported, why = self.support(assertion)
            if not supported:
                review.append({"assertion": assertion.as_dict(), "reason": why})
        return review

    def contradictions(self, *, subject: str | None = None,
                       at: str | None = None) -> list["Contradiction"]:
        """Supported claims that cannot all be true for one subject at one time.

        The comparison itself lives in ``knowledge/contradictions.py``; what this method
        contributes is the input, which is the part with the archive in it. Candidates
        are already gone, because ``current`` returns what stands.
        """
        from .contradictions import find

        return find(self.current(subject=subject, at=at), at=at)

    def matching(self, query: str, *, limit: int = 10, at: str | None = None,
                include_candidates: bool = False) -> list[Assertion]:
        """Claims whose subject, predicate, value or quote mention the question.

        A substring scan over short columns rather than a second full-text index:
        there are few assertions, and an intersection of every word in a sentence
        would answer almost nothing. Ordering is by how many terms it took to
        reach a claim, which decides what fits in the packet first and vouches
        for none of them.
        """
        terms = [term.casefold() for term in _TERMS.findall(query or "") if len(term) >= 2][:8]
        if not terms:
            return []
        clauses = " OR ".join(["(subject LIKE ? OR predicate LIKE ? OR value LIKE ? "
                               "OR quote LIKE ?)"] * len(terms))
        params: list[Any] = []
        for term in terms:
            params += [f"%{term}%"] * 4
        rows = self.db.execute(
            f"SELECT * FROM assertions WHERE {clauses} ORDER BY created_at DESC, id",
            params).fetchall()
        wanted = {CONFIRMED} | ({CANDIDATE} if include_candidates else set())
        scored: list[tuple[int, int, int, Assertion]] = []
        for index, row in enumerate(rows):
            assertion = Assertion.from_row(row)
            if assertion.status not in wanted:
                continue
            if at and not _contains(assertion, at):
                continue
            if not self.support(assertion)[0]:
                continue
            scored.append((-_score(assertion, terms), index, 0, assertion))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [item[3] for item in scored[:max(1, min(limit, 100))]]

    # -- internals -----------------------------------------------------------

    def _target(self, assertion_id: str, *, subject: str | None = None,
                predicate: str | None = None) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM assertions WHERE id=?",
                              (assertion_id,)).fetchone()
        if row is None:
            raise EvidenceError(f"unknown assertion {assertion_id!r}")
        if subject and (row["subject"], row["predicate"]) != (subject, predicate):
            raise EvidenceError(
                f"assertion {assertion_id!r} is about {row['subject']}/{row['predicate']}, "
                f"not {subject}/{predicate}: supersession cannot cross the thing being "
                "talked about")
        return row

    def _require_owner(self, actor: str) -> None:
        if self.owner_principal is None:
            raise EvidenceError(
                "no owner principal is configured, so nothing can be confirmed; set "
                "HERMES_MEMORY_OWNER_PRINCIPAL")
        if actor != self.owner_principal:
            raise EvidenceError(f"this decision belongs to the owner, not to {actor!r}")


def _terms(subject: Any, predicate: Any, value: Any,
           unit: Any) -> tuple[str, str, str, str | None]:
    return (_term(subject, "subject"), _term(predicate, "predicate"),
            _term(value, "value", MAX_VALUE),
            None if unit in (None, "") else _term(unit, "unit", 40))


def _term(value: Any, label: str, maximum: int = MAX_TERM) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise EvidenceError(f"{label} must be nonempty text of at most {maximum} characters")
    # Collapsed so that the same claim written two ways compares as one claim.
    return " ".join(value.split())


def _measurement(value: str, unit: str | None) -> None:
    if not unit:
        raise EvidenceError(
            "a measurement needs a unit; a bare number cannot be compared against "
            "another one later")
    try:
        float(value)
    except ValueError:
        raise EvidenceError(f"measurement value {value!r} is not a number") from None


def _span(store: EvidenceStore, record_id: Any, quote: Any,
          *, span: tuple[int, int] | None) -> tuple[int, int, str]:
    if not isinstance(record_id, str) or not record_id.startswith("rec_"):
        raise EvidenceError(f"cited evidence must be a canonical record id, got {record_id!r}")
    if not store.live_and_visible(record_id):
        raise EvidenceError(
            f"evidence {record_id!r} is not retrievable, so nothing can be asserted from it")
    record = store.get(record_id)
    if record is None:
        raise EvidenceError(f"evidence {record_id!r} could not be read")
    text = record.text
    if not isinstance(quote, str) or not quote.strip():
        raise EvidenceError("quote must be the exact text being cited")
    if len(quote) > MAX_QUOTE:
        raise EvidenceError(
            f"quote is longer than {MAX_QUOTE} characters; cite a span, not the document")
    if span is not None:
        if (not isinstance(span, (tuple, list)) or len(span) != 2
                or not all(isinstance(item, int) and not isinstance(item, bool)
                           for item in span)):
            raise EvidenceError("an explicit span must be a (start, end) pair of integers")
        start, end = span
        found = text[start:end]
        if found != quote:
            raise EvidenceError(
                f"the claimed span holds {found!r}, not {quote!r}; a citation is checked "
                "against the evidence, not taken on trust")
        return start, end, quote
    occurrences = list(_find_all(text, quote))
    if not occurrences:
        raise EvidenceError(
            "the quote does not appear in the cited evidence; a claim cannot borrow a "
            "citation from a sentence it does not match")
    if len(occurrences) > 1:
        raise EvidenceError(
            f"the quote appears {len(occurrences)} times in the cited evidence; name the "
            "span explicitly rather than attaching the claim to an arbitrary one")
    return occurrences[0], occurrences[0] + len(quote), quote


def _score(assertion: Assertion, terms: Sequence[str]) -> int:
    """How many of the question's terms this claim mentions, counted once each."""
    haystack = " ".join([assertion.subject, assertion.predicate, assertion.value,
                         assertion.quote]).casefold()
    return sum(1 for term in dict.fromkeys(terms) if term in haystack)


def _find_all(text: str, needle: str) -> Iterable[int]:
    index = text.find(needle)
    while index != -1:
        yield index
        index = text.find(needle, index + 1)


def _window(valid_from: Any, valid_to: Any) -> tuple[str | None, str | None]:
    start = timestamp(valid_from) if valid_from else None
    end = timestamp(valid_to) if valid_to else None
    if start and end and end < start:
        raise EvidenceError("valid_to precedes valid_from")
    return start, end


def _contains(assertion: Assertion, at: str) -> bool:
    return interval_contains(assertion.valid_from, assertion.valid_to, timestamp(at))


def _reason(value: Any) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise EvidenceError("reason must be nonempty text of at most 1000 characters")
