"""C11 lessons: what worked, expressed so a machine can check it still applies.

A lesson is not a note with confidence attached. It is one versioned sentence plus
the conditions under which it applies, what has to be true first, when it does not
apply, the cases that contradicted it, and the evidence it came from. The
applicability rule is drawn from a closed vocabulary for the same reason C8's
predicates are: anything outside it cannot be evaluated, and a lesson nobody can
evaluate is a rumour with a database row.

Status is never trusted from the column alone. A lesson is retrieved only while its
rule matches, its evidence still exists, and its evaluation is still about this
version of itself — so a correction to the underlying evidence retires the lesson on
the next read rather than on the next review someone never schedules.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..ids import digest, now, record_pk
from ..storage.evidence import EvidenceError
from ..storage.identity import citations_in_scope
from .evaluation import PASSED

# How a decision on a lesson is attributed. A promotion names the program that scored the
# run rather than a person, because no person decided it; the bound is enforced where the
# attribution is written, so a name has to fit before it gets here.
ACTOR_LIMIT = 120

__all__ = ["Lesson", "LessonStore", "STATUSES", "APPLICABLE", "match"]

STATUSES = ("candidate", "active", "retracted")
# The whole vocabulary of a lesson's applicability. A lesson cannot express a test
# this file cannot run, which is what makes retrieval deterministic.
FIELDS = ("source", "kind", "topic", "tool", "host", "channel", "account")
OPERATORS = ("eq", "in", "any", "all")
MAX_LESSON_CHARS = 800


@dataclass(frozen=True)
class Lesson:
    id: str
    version: int
    text: str
    applicability: dict[str, Any]
    prerequisites: tuple[str, ...]
    exceptions: tuple[str, ...]
    contrary_cases: tuple[str, ...]
    evidence: tuple[str, ...]
    status: str
    created_by: str
    created_kind: str
    evaluation_id: str | None
    support: int = 0
    against: int = 0
    applicable: bool = True
    why: str = ""

    @classmethod
    def from_row(cls, row) -> "Lesson":
        return cls(id=row["id"], version=int(row["version"]), text=row["text"],
                   applicability=_json(row["applicability"], "{}"),
                   prerequisites=tuple(_json(row["prerequisites"], [])),
                   exceptions=tuple(_json(row["exceptions"], [])),
                   contrary_cases=tuple(_json(row["contrary_cases"], [])),
                   evidence=tuple(_json(row["evidence"], [])), status=row["status"],
                   created_by=row["created_by"], created_kind=row["created_kind"],
                   evaluation_id=row["evaluation_id"])

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "version": self.version, "text": self.text,
                "applicability": dict(self.applicability),
                "prerequisites": list(self.prerequisites),
                "exceptions": list(self.exceptions),
                "contrary_cases": list(self.contrary_cases),
                "evidence": list(self.evidence), "status": self.status,
                "support": self.support, "against": self.against,
                "applicable": self.applicable, "why": self.why,
                "evaluation": self.evaluation_id}


class LessonStore:
    def __init__(self, store, *, outcomes, evaluations=None, identities=None,
                 owner_principal: str | None = None):
        self.store = store
        self.db = store.db
        self.outcomes = outcomes
        self.evaluations = evaluations
        self.identities = identities
        self.owner_principal = owner_principal

    # -- writing -------------------------------------------------------------

    def propose(self, *, text: str, applicability: Mapping[str, Any], lesson_id: str,
                version: int | None = None, prerequisites: Iterable[str] = (),
                exceptions: Iterable[str] = (), contrary_cases: Iterable[str] = (),
                evidence: Iterable[str] = (), proposed_by: str,
                proposed_kind: str) -> dict[str, Any]:
        """Offer a lesson. Nothing here can make it applicable to anything.

        A proposal from the system about its own procedure starts as a candidate,
        because the alternative is an archive that promotes its own habits.
        """
        if proposed_kind not in ("owner", "agent", "evaluation"):
            raise EvidenceError(f"unknown proposer kind {proposed_kind!r}")
        body = _text(text, "text", MAX_LESSON_CHARS)
        rule = _validate_rule(applicability)
        span_list = [_check_citation(item) for item in evidence][:24]
        self._check_evidence(span_list)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            newest = self.db.execute("SELECT max(version) FROM lessons WHERE id=?",
                                     (lesson_id,)).fetchone()[0]
            number = int(version) if version else int(newest or 0) + 1
            if version and number <= int(newest or 0):
                raise EvidenceError(
                    f"version {number} of {lesson_id} is not newer than {newest}; a "
                    "lesson is replaced by a later version, never by rewriting one")
            clash = self.db.execute("SELECT 1 FROM lessons WHERE id=? AND version=?",
                                    (lesson_id, number)).fetchone()
            if clash:
                raise EvidenceError(f"version {number} of {lesson_id} is already on file")
            self.db.execute(
                "INSERT INTO lessons(id, version, text, applicability, prerequisites, "
                "exceptions, contrary_cases, evidence, status, created_by, created_kind, "
                "created_at) VALUES(?,?,?,?,?,?,?,?, 'candidate', ?,?,?)",
                (lesson_id, number, body, json.dumps(rule, sort_keys=True),
                 json.dumps(list(prerequisites)[:12], ensure_ascii=False),
                 json.dumps(list(exceptions)[:12], ensure_ascii=False),
                 json.dumps(list(contrary_cases)[:12], ensure_ascii=False),
                 json.dumps(span_list, ensure_ascii=False), _text(proposed_by,
                                                                  "proposed_by", 120),
                 proposed_kind, now()))
            self.store._audit("lesson_propose", f"{lesson_id}@{number}", {
                "by": proposed_by, "kind": proposed_kind, "evidence": len(span_list)})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": lesson_id, "version": number, "status": "candidate",
                "note": "a proposal is not a rule; it applies to nothing until an "
                        "evaluation or the owner says otherwise"}

    def activate(self, *, lesson_id: str, version: int, actor: str,
                 reason: str) -> dict[str, Any]:
        """The owner's own promotion. It needs no evaluation, and says who decided."""
        if self.owner_principal is None or actor != self.owner_principal:
            raise EvidenceError(
                "only the owner may promote a lesson without an evaluation; the system "
                "granting itself new rules is precisely the loop this closes")
        return self._set(lesson_id, version, "active", actor=actor, reason=reason)

    def retract(self, *, lesson_id: str, version: int | None = None, actor: str,
                reason: str) -> dict[str, Any]:
        """Immediate, and it does not wait for a review cycle.

        The owner named the actor; the system may also retract on its own when the
        evidence under a lesson disappears — that is not a promotion of anything, so
        it does not need the owner's permission to stop trusting a claim.
        """
        if actor != "system" and (self.owner_principal is None or actor !=
                                 self.owner_principal):
            raise EvidenceError("a lesson is retracted by the owner or by the store "
                                "itself, never by the agent that proposed it")
        number = version if version else self._current(lesson_id)
        if number is None:
            raise EvidenceError(f"{lesson_id} has no version on file")
        return self._set(lesson_id, int(number), "retracted", actor=actor, reason=reason)

    def activate_evaluation(self, *, lesson_id: str, version: int,
                            evaluation_id: str) -> dict[str, Any]:
        """Promote on the strength of a run. The proposer cannot do this to itself.

        The evaluation has to be current — same code, same model, same lesson text,
        same fixtures — and passed. Neither of those is taken on trust from the caller:
        both are read back from the run's own recorded cases.
        """
        if self.evaluations is None:
            raise EvidenceError(
                "no evaluation machinery is wired, so nothing can be promoted by a run")
        row = self.db.execute("SELECT * FROM lessons WHERE id=? AND version=?",
                              (lesson_id, int(version))).fetchone()
        if row is None:
            raise EvidenceError(f"no lesson {lesson_id} version {version} on file")
        evaluation = self.evaluations.get(evaluation_id)
        if str(evaluation.runner).strip() == str(row["created_by"]).strip():
            raise EvidenceError(
                f"{row['created_by']} proposed this lesson and also ran its evaluation; a "
                "candidate that promotes itself is not learning, it is agreeing with itself")
        lesson = Lesson.from_row(row)
        if not self.evaluations.is_current(evaluation_id, lesson):
            raise EvidenceError(
                f"{evaluation_id} is no longer a verdict about {lesson_id}@{version}: "
                "the code, model, fixtures or the lesson itself have moved since it ran")
        runner = str(evaluation.runner or "")
        actor = f"evaluation:{runner}"
        if len(actor) > ACTOR_LIMIT:
            # The run row keeps the whole path; an activation only has to name the program
            # unambiguously, and a deep install directory is not part of its identity.
            actor = f"evaluation:{Path(runner).name}:{digest([runner])[:12]}"
        return self._set(lesson_id, int(version), "active", actor=actor,
                         reason=f"promoted by {evaluation_id}", evaluation_id=evaluation_id)

    def record_contradiction(self, *, lesson_id: str, version: int, note: str,
                            actor: str, evidence: Iterable[str] = ()) -> dict[str, Any]:
        """A case the lesson got wrong. Filed against the lesson, and weighted."""
        return self.outcomes.record(subject_kind="lesson",
                                    subject_id=f"{lesson_id}@{version}",
                                    kind=_kind_for(actor, self.owner_principal),
                                    valence="failure", note=note, actor=actor,
                                    evidence=evidence)

    def record_confirmation(self, *, lesson_id: str, version: int, note: str,
                            actor: str, evidence: Iterable[str] = (),
                            kind: str | None = None) -> dict[str, Any]:
        """A case the lesson got right, from somebody who could know."""
        teller = kind or _kind_for(actor, self.owner_principal)
        return self.outcomes.record(subject_kind="lesson",
                                    subject_id=f"{lesson_id}@{version}",
                                    kind=teller, valence="success", note=note,
                                    actor=actor, evidence=evidence)

    # -- reading -------------------------------------------------------------

    def applicable(self, task: Mapping[str, Any], *, limit: int = 4,
                   account_id: str | None = None) -> list[Lesson]:
        """Active lessons whose rule this task satisfies, with the reason attached.

        Every candidate is re-checked rather than believed: the rule, the evidence
        under it and the evaluation that promoted it all have to still hold, so a
        lesson outlives its usefulness only as long as nothing has changed.
        """
        rows = self.db.execute(
            "SELECT * FROM lessons WHERE status='active' ORDER BY id, version DESC").fetchall()
        newest: dict[str, int] = {}
        found: list[Lesson] = []
        for row in rows:
            # One lesson contributes its highest active version, never a stack of
            # revisions that all say slightly different things about the same task.
            if str(row["id"]) in newest:
                continue
            newest[str(row["id"])] = int(row["version"])
            lesson = Lesson.from_row(row)
            ok, why = match(lesson.applicability, task)
            if not ok:
                continue
            if account_id is not None and not self._visible_to(lesson, account_id):
                continue
            alive, note = self._still_supported(lesson)
            if not alive:
                continue
            lesson = Lesson(**{**lesson.__dict__, "support": note[0], "against": note[1],
                               "applicable": True, "why": why})
            found.append(lesson)
            if len(found) >= _bounded(limit):
                break
        return found

    def get(self, lesson_id: str, version: int | None = None) -> Lesson | None:
        number = version if version else self._current(lesson_id)
        if number is None:
            return None
        row = self.db.execute("SELECT * FROM lessons WHERE id=? AND version=?",
                              (lesson_id, int(number))).fetchone()
        if row is None:
            return None
        lesson = Lesson.from_row(row)
        tally = self.outcomes.tally("lesson", f"{lesson.id}@{lesson.version}")
        return Lesson(**{**lesson.__dict__, "support": tally["support"],
                         "against": tally["against"]})

    def versions(self, lesson_id: str) -> list[Lesson]:
        rows = self.db.execute("SELECT * FROM lessons WHERE id=? ORDER BY version",
                               (lesson_id,)).fetchall()
        return [Lesson.from_row(row) for row in rows]

    def needs_review(self, *, limit: int = 25) -> list[dict[str, Any]]:
        """Active lessons that no longer hold up, for whatever reports status.

        Retrieval already drops these silently — a lesson that cannot prove itself is
        not taught to anything — but the owner should still see that it happened
        rather than discover a missing habit by its absence.
        """
        rows = self.db.execute(
            "SELECT id, version FROM lessons WHERE status='active' ORDER BY id, version"
        ).fetchall()
        review: list[dict[str, Any]] = []
        for row in rows:
            stored = self.db.execute("SELECT * FROM lessons WHERE id=? AND version=?",
                                     (row["id"], row["version"])).fetchone()
            lesson = Lesson.from_row(stored)
            alive, note = self._still_supported(lesson)
            if not alive:
                review.append({"id": f"{row['id']}@{row['version']}", "reason": note[2],
                               "support": note[0], "against": note[1]})
            if len(review) >= _bounded(limit):
                break
        return review

    # -- internals -----------------------------------------------------------

    def _still_supported(self, lesson: Lesson) -> tuple[bool, tuple[int, int, str]]:
        for span in lesson.evidence:
            if not self.store.live_and_visible(record_pk(span)):
                return False, (0, 0, "evidence gone")
        tally = self.outcomes.tally("lesson", f"{lesson.id}@{lesson.version}")
        bad, reason = self._support_state(lesson, tally)
        if bad:
            return False, (tally["support"], tally["against"], reason)
        if lesson.evaluation_id:
            # The verdict is read from the row, not from a caller's assurance: a run that
            # was invalidated — by an upgrade, a retraction or a fixture that turned out to
            # be wrong — stops licensing anything the moment it is marked, and that fact is
            # in the archive rather than in whoever is asking.
            verdict = self.db.execute("SELECT verdict FROM evaluations WHERE id=?",
                                      (lesson.evaluation_id,)).fetchone()
            if verdict is None or str(verdict["verdict"]) != PASSED:
                return False, (tally["support"], tally["against"], "evaluation withdrawn")
            if self.evaluations is not None and not self.evaluations.is_current(
                    str(lesson.evaluation_id), lesson):
                # The stricter question — is this verdict about *this* code, model and
                # lesson text — can only be asked by a caller that knows which versions it
                # is running. A status read does not, so it asks the one above and stops.
                return False, (tally["support"], tally["against"], "evaluation stale")
        return True, (tally["support"], tally["against"], "")

    def _support_state(self, lesson: Lesson, tally: Mapping[str, Any]) -> tuple[bool, str]:
        """Is this lesson still worth retrieving, on the evidence that can be checked?

        Assistant claims are absent from the arithmetic by construction: a system
        that said "done" forty times cannot manufacture support for its own habit.
        """
        support, against = int(tally["support"]), int(tally["against"])
        if lesson.contrary_cases and not support:
            return True, ("every confirmation this lesson had was contradicted by a "
                          "checked outcome")
        if against > support:
            return True, f"{against} checked failure(s) against {support} success(es)"
        return False, ""

    def _check_evidence(self, spans: Sequence[str]) -> None:
        if not spans:
            raise EvidenceError(
                "a lesson with no evidence is an opinion; say which records it came from")
        for span in spans:
            if span.startswith("sum_") or span.startswith("hdoc"):
                # The recursive-learning gate: summaries and derived documents are
                # this system's own readings. Learning from one would let an error
                # teach itself, which is how a confident archive drifts.
                raise EvidenceError(
                    f"{span} is a derived artifact; a lesson is learned from evidence, "
                    "not from this system's own summary of it")
            if not self.store.live_and_visible(record_pk(span)):
                raise EvidenceError(f"{span} is not live evidence this store can read")

    def _visible_to(self, lesson: Lesson, account_id: str) -> bool:
        """Practice drawn from someone's records is not portable to another person.

        """
        if self.identities is None or not lesson.evidence:
            return True
        return citations_in_scope(self.store, self.identities, lesson.evidence,
                                  identifiers=set(self.identities.group(account_id)))

    def _current(self, lesson_id: str) -> int | None:
        row = self.db.execute("SELECT max(version) FROM lessons WHERE id=?",
                              (lesson_id,)).fetchone()
        return row[0] if row and row[0] is not None else None

    def _set(self, lesson_id: str, version: int, status: str, *, actor: str,
             reason: str, evaluation_id: str | None = None) -> dict[str, Any]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.db.execute(
                "UPDATE lessons SET status=?, decided_by=?, decided_at=?, retraction_reason=?"
                + (", evaluation_id=?" if evaluation_id else "") +
                " WHERE id=? AND version=?",
                (status, _text(actor, "actor", ACTOR_LIMIT), now(),
                 _text(reason, "reason", 400),
                 *(([evaluation_id] if evaluation_id else [])), lesson_id, int(version)))
            if not int(cursor.rowcount or 0):
                raise EvidenceError(f"no lesson {lesson_id} version {version} on file")
            if status == "active":
                # Promoting one version retires the ones below it, whoever promoted.
                # Two active revisions of one lesson would both be retrieved, and the
                # older would keep teaching what the newer exists to correct.
                self.db.execute(
                    "UPDATE lessons SET status='retracted', retraction_reason=?, "
                    "decided_by=?, decided_at=? WHERE id=? AND version<? AND "
                    "status='active'",
                    (f"superseded by {lesson_id}@{version}", actor, now(), lesson_id,
                     int(version)))
            self.store._audit(f"lesson_{status}", f"{lesson_id}@{version}",
                              {"by": actor, "reason": reason[:300]})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": f"{lesson_id}@{version}", "status": status, "by": actor}


def match(rule: Mapping[str, Any], task: Mapping[str, Any]) -> tuple[bool, str]:
    """Does this task satisfy the lesson's applicability rule?"""
    combinator = str(rule.get("all") is not None and "all" or
                    rule.get("any") is not None and "any" or
                    rule.get("not") is not None and "not" or "")
    if combinator == "all":
        for part in rule.get("all") or []:
            ok, _ = match(part, task)
            if not ok:
                return False, "not every condition held"
        return True, "all conditions held"
    if combinator == "any":
        for part in rule.get("any") or []:
            ok, _ = match(part, task)
            if ok:
                return True, "one condition held"
        return False, "no alternative held"
    if combinator == "not":
        ok, _ = match(rule.get("not") or {}, task)
        return (not ok), "the excluded case was absent"
    return _leaf(rule, task)


def _leaf(rule: Mapping[str, Any], task: Mapping[str, Any]) -> tuple[bool, str]:
    field = str(rule.get("field") or "")
    operator = str(rule.get("op") or "eq")
    wanted = task.get(field)
    if wanted is None:
        return False, f"{field} was not part of this task"
    value = rule.get("value")
    if operator == "eq":
        return (str(wanted) == str(value)), f"{field}={wanted}"
    if operator == "in":
        return any(str(wanted) == str(item) for item in value or ()), f"{field}={wanted}"
    if operator in ("any", "all"):
        held = [str(item) for item in value or ()]
        have = {str(item) for item in (wanted if isinstance(wanted, (list, tuple, set))
                                       else [wanted])}
        if operator == "any":
            return bool(set(held) & have), f"{field} overlaps"
        return set(held) <= have, f"{field} covers all"
    return False, f"{operator} is not a test this store runs"


def _validate_rule(rule: Any, depth: int = 0) -> dict[str, Any]:
    if depth > 6:
        raise EvidenceError("an applicability rule may not nest past six levels")
    if not isinstance(rule, Mapping):
        raise EvidenceError("an applicability rule must be an object")
    for key in ("all", "any"):
        if key in rule:
            parts = rule.get(key)
            if not isinstance(parts, Sequence) or not parts:
                raise EvidenceError(f"{key} needs a nonempty list of conditions")
            return {key: [_validate_rule(part, depth + 1) for part in parts]}
    if "not" in rule:
        return {"not": _validate_rule(rule["not"], depth + 1)}
    field = str(rule.get("field") or "")
    if field not in FIELDS:
        raise EvidenceError(
            f"{field!r} is not something a lesson can condition on; a rule the store "
            f"cannot evaluate is a rumour. Known: {', '.join(FIELDS)}")
    operator = str(rule.get("op") or "eq")
    if operator not in OPERATORS:
        raise EvidenceError(f"unknown applicability operator {operator!r}")
    value = rule.get("value")
    if operator == "eq":
        if not isinstance(value, str) or not value.strip():
            raise EvidenceError("eq needs a text value")
        return {"field": field, "op": operator, "value": value.strip()[:120]}
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise EvidenceError(f"{operator} needs a nonempty list of values")
    if len(value) > 24:
        raise EvidenceError(f"{operator} takes at most 24 values")
    return {"field": field, "op": operator,
            "value": [str(item).strip()[:120] for item in value]}


def _kind_for(actor: str, owner_principal: str | None) -> str:
    if owner_principal is not None and actor == owner_principal:
        return "owner_report"
    return "assistant_claim"


def _check_citation(value: Any) -> str:
    """A lesson cites evidence by id, and an id has to be a usable one."""
    span = str(value or "").strip().split(" ", 1)[0]
    if not span or len(span) > 160 or not span.replace("_", "").replace("@", "").\
            replace("#", "").isalnum():
        raise EvidenceError(f"{value!r} is not a citation; a lesson names the record it "
                            "came from, not a feeling about it")
    return span


def _json(value: Any, default):
    try:
        return json.loads(value) if value else default
    except (TypeError, json.JSONDecodeError):
        return default


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise EvidenceError(f"{label} must be nonempty text of at most {maximum} "
                            "characters")
    return " ".join(value.split())


def _bounded(value: int) -> int:
    return value if isinstance(value, int) and 1 <= value <= 50 else 4
