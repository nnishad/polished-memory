"""C11 evaluation: a lesson is promoted by a run, never by a claim.

The shape of this file is the whole point. A caller declares the cases it intends to
run — which fixes the fixture digest — then reports each result one at a time under
a token, then asks for the verdict. Nothing anywhere accepts "it passed" as an input,
because the failure mode being closed is a system that files its own optimism as
evidence: an assistant says it succeeded, the archive promotes the habit, and the
next run inherits the claim as a prerequisite.

A verdict is only about the thing that was run. Which lesson version, which fixtures,
which baseline, which code and which model are all part of its identity, so an
evaluation from before an upgrade cannot license a lesson today — it reads as stale,
which is a different and honest answer from "failed".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..ids import digest, new_id, now
from ..storage.evidence import EvidenceError

__all__ = ["EvaluationLedger", "Evaluation", "ROLES", "OUTCOMES"]

ROLES = ("targeted", "regression")
OUTCOMES = ("pass", "fail", "error", "skipped")
PASSED = "passed"
MAX_CASES = 200


@dataclass(frozen=True)
class Evaluation:
    id: str
    lesson_id: str
    lesson_version: int
    lesson_digest: str
    runner: str
    fixture_digest: str
    baseline_digest: str
    code_version: str
    model_version: str
    verdict: str
    detail: str
    started_at: str
    finished_at: str | None
    cases: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_row(cls, row, cases: Sequence[Mapping[str, Any]] = ()) -> "Evaluation":
        return cls(id=row["id"], lesson_id=row["lesson_id"],
                   lesson_version=int(row["lesson_version"]),
                   lesson_digest=row["lesson_digest"], runner=row["runner"],
                   fixture_digest=row["fixture_digest"],
                   baseline_digest=row["baseline_digest"], code_version=row["code_version"],
                   model_version=row["model_version"], verdict=row["verdict"],
                   detail=str(row["detail"] or ""), started_at=row["started_at"],
                   finished_at=row["finished_at"],
                   cases=tuple(dict(item) for item in cases))

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "lesson": f"{self.lesson_id}@{self.lesson_version}",
                "lesson_digest": self.lesson_digest[:16],
                "runner": self.runner, "verdict": self.verdict, "detail": self.detail,
                "fixtures": self.fixture_digest[:16], "baseline": self.baseline_digest[:16],
                "code_version": self.code_version, "model_version": self.model_version,
                "cases": list(self.cases), "finished_at": self.finished_at}


class EvaluationLedger:
    """Runs candidates against fixtures. The runner is authorised separately."""

    def __init__(self, store, *, runner, code_version: str, model_version: str,
                 lessons=None, owner_principal: str | None = None):
        self.store = store
        self.db = store.db
        self.runner = runner
        self.lessons = lessons
        self.owner_principal = owner_principal
        self.code_version = _text(code_version, "code_version", 80)
        self.model_version = _text(model_version, "model_version", 80)
        self._tokens: dict[str, str] = {}

    @classmethod
    def reading(cls, store, *, owner_principal: str | None = None) -> "EvaluationLedger":
        """A ledger that can read every verdict and start none.

        Reports and housekeeping need to ask what a run said; only an authorized runner
        may say what a case scored. The distinction lives in this object rather than in
        whoever remembers to pass ``runner=None``.
        """
        return cls(store, runner=None, code_version="reading", model_version="reading",
                   owner_principal=owner_principal)

    # -- the protocol --------------------------------------------------------

    def begin(self, *, lesson_id: str, version: int, cases: Sequence[Mapping[str, Any]],
              baseline: Mapping[str, Any] | None = None,
              runner: str | None = None) -> dict[str, Any]:
        """Declare the exact suite that will be run. This is what fixes its identity."""
        if self.runner is None:
            raise EvidenceError(
                "this ledger can read evaluations and cannot start one: no runner is "
                "authorized here, and a promotion has to rest on a run somebody else "
                "agreed to pay for")
        declared = _check_cases(cases)
        lesson = self._lesson(lesson_id, version)
        if str(lesson["status"]) != "candidate":
            raise EvidenceError(
                f"{lesson_id}@{version} is {lesson['status']}; an evaluation licenses a "
                "promotion, so there is nothing to evaluate about a lesson already settled")
        lesson_digest = digest(_row_shape(lesson))
        fixture_digest = digest([{"case_id": item["case_id"], "role": item["role"],
                                 **{key: value for key, value in item.items()
                                    if key not in ("case_id", "role")}}
                                for item in declared])
        baseline_digest = digest(baseline or {})
        evaluation_id = "eval_" + digest([lesson_id, int(version), fixture_digest,
                                         baseline_digest, self.code_version,
                                         self.model_version])[:32]
        token = new_id("run")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute("SELECT * FROM evaluations WHERE id=?",
                                       (evaluation_id,)).fetchone()
            if existing is not None:
                raise EvidenceError(
                    f"{evaluation_id} already recorded this exact suite against this "
                    "version of the code and model; re-running it would overwrite an "
                    "outcome that happened")
            self.db.execute(
                "INSERT INTO evaluations(id, lesson_id, lesson_version, lesson_digest, "
                "runner, fixture_digest, baseline_digest, code_version, model_version, "
                "verdict, started_at) VALUES(?,?,?,?,?,?,?,?,?, 'running', ?)",
                (evaluation_id, lesson_id, int(version), lesson_digest,
                 _text(runner or _runner_name(self.runner), "runner", 120),
                 fixture_digest, baseline_digest, self.code_version, self.model_version,
                 now()))
            for item in declared:
                self.db.execute(
                    "INSERT INTO evaluation_cases(evaluation_id, case_id, role, outcome, "
                    "detail) VALUES(?,?,?, 'skipped', 'not run yet')",
                    (evaluation_id, item["case_id"], item["role"]))
            self.store._audit("evaluation_begin", evaluation_id,
                              {"lesson": f"{lesson_id}@{version}", "cases": len(declared),
                               "fixtures": fixture_digest[:16]})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self._tokens[evaluation_id] = token
        return {"id": evaluation_id, "token": token, "cases": len(declared),
                "fixtures": fixture_digest[:16]}

    def record(self, *, evaluation_id: str, token: str, case_id: str, outcome: str,
               detail: str = "") -> dict[str, Any]:
        """One case's result, into an evaluation that is still open."""
        self._check_token(evaluation_id, token)
        if outcome not in OUTCOMES:
            raise EvidenceError(f"unknown case outcome {outcome!r}")
        if self.db.execute("SELECT 1 FROM evaluations WHERE id=?",
                           (evaluation_id,)).fetchone() is None:
            raise EvidenceError(f"unknown evaluation {evaluation_id!r}")
        changed = self.db.execute(
            "UPDATE evaluation_cases SET outcome=?, detail=? WHERE evaluation_id=? AND "
            "case_id=?", (_text(outcome, "outcome", 20), _clip(detail, 400),
                          evaluation_id, _text(case_id, "case_id", 120))).rowcount
        if not int(changed or 0):
            raise EvidenceError(
                f"{case_id!r} was not in the suite declared for {evaluation_id}; results "
                "have to be about the cases that were actually run")
        return {"id": evaluation_id, "case": case_id, "outcome": outcome}

    def finish(self, *, evaluation_id: str, token: str) -> dict[str, Any]:
        """Derive the verdict. There is no path to 'passed' except through the cases."""
        self._check_token(evaluation_id, token)
        cases = self.cases(evaluation_id)
        verdict, detail = _verdict_of(cases)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE evaluations SET verdict=?, detail=?, finished_at=? WHERE id=? AND "
                "verdict='running'", (verdict, detail, now(), evaluation_id))
            self.store._audit("evaluation_finish", evaluation_id,
                              {"verdict": verdict, "detail": detail[:200]})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self._tokens.pop(evaluation_id, None)
        return {**self.get(evaluation_id).as_dict(), "promoted": False}

    def run(self, *, lesson_id: str, version: int, cases: Sequence[Mapping[str, Any]],
            baseline: Mapping[str, Any] | None = None, promote: bool = True) -> dict:
        """Drive the authorised runner through the whole protocol, and nothing else.

        This is the only way an evaluation normally exists. The runner returns each
        case's result; this file decides what the collection of results means.
        """
        opened = self.begin(lesson_id=lesson_id, version=version, cases=cases,
                            baseline=baseline)
        lesson = self._lesson(lesson_id, version)
        try:
            for item in _check_cases(cases):
                answer = self.runner.evaluate(lesson=dict(lesson), case=dict(item))
                outcome, detail = _result_of(answer)
                self.record(evaluation_id=opened["id"], token=opened["token"],
                            case_id=item["case_id"], outcome=outcome, detail=detail)
            report = self.finish(evaluation_id=opened["id"], token=opened["token"])
        except BaseException:
            # An abandoned run stays 'running' rather than being guessed at: the next
            # attempt has to declare a new suite, and an unfinished one promotes nothing.
            raise
        if promote and report["verdict"] == PASSED and self.lessons is not None:
            promoted = self.lessons.activate_evaluation(
                lesson_id=lesson_id, version=version, evaluation_id=opened["id"])
            report["promoted"] = bool(promoted.get("status") == "active")
            report["promoted_by"] = promoted.get("by")
        return report

    # -- reading -------------------------------------------------------------

    def get(self, evaluation_id: str) -> Evaluation:
        row = self.db.execute("SELECT * FROM evaluations WHERE id=?",
                              (evaluation_id,)).fetchone()
        if row is None:
            raise EvidenceError(f"unknown evaluation {evaluation_id!r}")
        return Evaluation.from_row(row, self.cases(evaluation_id))

    def cases(self, evaluation_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM evaluation_cases WHERE evaluation_id=? "
                               "ORDER BY role, case_id", (evaluation_id,)).fetchall()
        return [dict(row) for row in rows]

    def latest(self, lesson_id: str, version: int) -> Evaluation | None:
        row = self.db.execute(
            "SELECT * FROM evaluations WHERE lesson_id=? AND lesson_version=? AND "
            "verdict!='running' ORDER BY started_at DESC, id LIMIT 1",
            (lesson_id, int(version))).fetchone()
        return Evaluation.from_row(row, self.cases(str(row["id"]))) if row else None

    def is_current(self, evaluation_id: str, lesson: Any) -> bool:
        """Is this verdict still about the thing in front of us?

        Code, model, fixtures and the lesson's own content all count. A changed word
        in the lesson is a changed claim, and an old run says nothing about the new one.
        """
        evaluation = self.get(evaluation_id)
        if evaluation.verdict != PASSED:
            return False
        if evaluation.code_version != self.code_version or \
                evaluation.model_version != self.model_version:
            return False
        # The lesson itself, by content: identity is in the first two fields of the
        # shape, so a different lesson or a different version cannot collide here.
        return digest(_lesson_shape(lesson)) == evaluation.lesson_digest

    def invalidate(self, evaluation_id: str, *, reason: str,
                   db=None) -> dict[str, Any]:
        """Mark a run as no longer licensing anything, without deleting what happened.

        ``db`` lets an erasure stale the verdict in the same transaction as the tombstones,
        so there is no moment in which forgotten evidence still has a passed run behind a
        lesson that cites it.
        """
        _text(reason, "reason", 400)
        connection = db or self.db
        if db is not None and not db.in_transaction:
            raise EvidenceError("an invalidation writes need an ambient transaction")
        owns_transaction = db is None
        if owns_transaction:
            connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "UPDATE evaluations SET verdict='stale', detail=?, "
                "finished_at=COALESCE(finished_at, ?) WHERE id=? AND "
                "verdict='passed'", (_clip(reason, 400), now(), evaluation_id))
            self.store._audit("evaluation_invalidate", evaluation_id,
                              {"reason": reason[:200]})
            if owns_transaction:
                connection.execute("COMMIT")
        except BaseException:
            if owns_transaction:
                connection.execute("ROLLBACK")
            raise
        return {"id": evaluation_id, "verdict": self.get(evaluation_id).verdict}

    # -- internals -----------------------------------------------------------

    def _check_token(self, evaluation_id: str, token: str) -> None:
        held = self._tokens.get(evaluation_id)
        if held is None:
            raise EvidenceError(
                f"{evaluation_id} has no open run in this process; a result cannot be "
                "filed against a finished or never-declared evaluation")
        if token != held:
            raise EvidenceError("this is not the token that opened the run")

    def _lesson(self, lesson_id: str, version: int) -> Mapping[str, Any]:
        row = self.db.execute("SELECT * FROM lessons WHERE id=? AND version=?",
                              (lesson_id, int(version))).fetchone()
        if row is None:
            raise EvidenceError(f"no lesson {lesson_id} version {version} on file")
        return dict(row)


def _verdict_of(cases: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    outcomes = [str(item["outcome"]) for item in cases]
    targeted = [item for item in cases if item["role"] == "targeted"]
    regression = [item for item in cases if item["role"] == "regression"]
    if not targeted or not regression:
        return "incomplete", ("a lesson needs both a case it is supposed to fix and a "
                              "case that already worked; one alone proves nothing")
    if any(item["outcome"] in ("fail", "error") for item in regression):
        broken = ", ".join(str(item["case_id"]) for item in regression
                           if item["outcome"] in ("fail", "error"))
        return "failed", f"regression case(s) broke: {broken}"
    if any(item["outcome"] != "pass" for item in targeted):
        return "failed", "the targeted cases did not pass"
    if any(item == "skipped" for item in outcomes):
        return "incomplete", "some declared cases were never run"
    return PASSED, f"{len(targeted)} targeted and {len(regression)} regression case(s) passed"


def _result_of(answer: Any) -> tuple[str, str]:
    if isinstance(answer, str):
        answer = {"outcome": answer}
    if not isinstance(answer, Mapping):
        return "error", "the runner returned something that is not a result"
    outcome = str(answer.get("outcome") or answer.get("status") or "error")
    if outcome not in OUTCOMES:
        return "error", f"the runner said {outcome!r}, which is not an outcome"
    return outcome, str(answer.get("detail") or "")


def _check_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise EvidenceError("a suite is a list of cases")
    if not 1 <= len(cases) <= MAX_CASES:
        raise EvidenceError(f"a suite holds between 1 and {MAX_CASES} cases")
    seen: list[str] = []
    declared: list[dict[str, Any]] = []
    for item in cases:
        if not isinstance(item, Mapping):
            raise EvidenceError("every case is an object")
        case_id = _text(item.get("case_id"), "case_id", 120)
        if case_id in seen:
            raise EvidenceError(f"{case_id!r} appears twice in one suite")
        seen.append(case_id)
        role = str(item.get("role") or "")
        if role not in ROLES:
            raise EvidenceError(f"{case_id} has role {role!r}; expected {ROLES}")
        declared.append({**dict(item), "case_id": case_id, "role": role})
    return declared


def _lesson_shape(lesson: Any) -> list[Any]:
    return [lesson.id, lesson.version, lesson.text, lesson.applicability,
            list(lesson.prerequisites), list(lesson.exceptions),
            list(lesson.contrary_cases), list(lesson.evidence)]


def _row_shape(row: Mapping[str, Any]) -> list[Any]:
    """The same eight values a lesson is, from a stored row.

    Both shapes must agree byte for byte or staleness detection is fiction: the row
    keeps its lists as JSON text, so it is parsed back before hashing rather than
    digested in whatever spelling the writer happened to use.
    """
    return [row["id"], int(row["version"]), row["text"],
            json.loads(row["applicability"] or "{}"),
            list(json.loads(row["prerequisites"] or "[]")),
            list(json.loads(row["exceptions"] or "[]")),
            list(json.loads(row["contrary_cases"] or "[]")),
            list(json.loads(row["evidence"] or "[]"))]


def _runner_name(runner: Any) -> str:
    return str(getattr(runner, "name", None) or runner.__class__.__name__)


def _clip(value: Any, maximum: int) -> str:
    return str(value or "")[:maximum]


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise EvidenceError(f"{label} must be nonempty text of at most {maximum} "
                            "characters")
    return " ".join(value.split())
