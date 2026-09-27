"""C11 the authorized runner: something outside the archive decides what passed.

``EvaluationLedger`` refuses to start a run when nobody was authorized to answer the
cases, and that "nobody" is the state every installation starts in. This module is the
one way to stop it being that state: the owner names a program, and the program answers
each case. Nothing here decides what a good lesson looks like, and nothing here can be
asked by an agent — the door is an operator's command, the program behind it is configured
rather than passed in, and the answer that comes back is a claim about one case, which the
ledger then weighs against the rest of the suite.

The runner is executed directly, with no shell, on a scrubbed environment, with a
timeout and a bounded read. A broken runner, a timeout or an answer that is not JSON
becomes ``error`` for that case — never a pass, and never an exception that leaves the
evaluation looking unfinished by accident.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..ids import now
from ..storage.evidence import EvidenceError
from .evaluation import EvaluationLedger
from .lessons import LessonStore
from .outcomes import OutcomeLog

__all__ = ["EvaluationError", "CommandRunner", "load_suite", "evaluate", "report_on",
           "MAX_SUITE_BYTES"]

MAX_SUITE_BYTES = 1_000_000
MAX_ANSWER_BYTES = 200_000
# The subprocess gets these and nothing else. The store's secrets are read from scoped
# secret storage when needed, so a runner that inherits the whole environment would be
# handed more than an evaluation of one lesson requires. Anything past this is named in
# HERMES_MEMORY_EVALUATOR_ENV — the owner's own config file, not the caller's shell.
ENV_ALLOWANCE = ("PATH", "LANG", "LC_ALL", "HOME", "PYTHONIOENCODING")
OUTCOMES = ("pass", "fail", "error", "skipped")


class EvaluationError(ValueError):
    """Evaluation was asked for without a runner, a suite, or a version to be about."""


class CommandRunner:
    """Ask one named program, once per case, what that case scored."""

    def __init__(self, argv: Sequence[str], *, timeout_s: float = 120.0,
                 env: Mapping[str, str] | None = None, env_extra: Sequence[str] = (),
                 execute: Callable[..., Any] = subprocess.run):
        argv = [str(part) for part in (argv or ())]
        if not argv or not argv[0].startswith("/"):
            raise EvaluationError(
                "the evaluator must be an absolute path; an evaluation runs whatever it "
                "names, so a bare word is somebody else's executable")
        if not isinstance(timeout_s, (int, float)) or not 1 <= timeout_s <= 3600:
            raise EvaluationError("the evaluator timeout must be between 1 and 3600 seconds")
        self.argv = tuple(argv)
        # What the run was scored by has to be readable out of the verdict, not guessed
        # from whoever still has the object around.
        self.name = argv[0]
        self.timeout_s = float(timeout_s)
        self.env = dict(env if env is not None else _scrubbed_env(env_extra))
        self._execute = execute
        self.answers = 0

    def evaluate(self, *, lesson: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, str]:
        """One case, one question. The answer is trusted only as far as the ledger weighs it."""
        payload = json.dumps({"lesson": _jsonable(dict(lesson)), "case": _jsonable(dict(case))},
                            sort_keys=True, default=str)
        self.answers += 1
        try:
            finished = self._execute(list(self.argv), input=payload.encode("utf-8"),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=self.timeout_s, env=self.env)
        except subprocess.TimeoutExpired:
            return {"outcome": "error",
                    "detail": f"the evaluator did not answer within {self.timeout_s:g}s"}
        except OSError as error:
            return {"outcome": "error", "detail": f"the evaluator could not run: {error}"[:400]}
        out = bytes(finished.stdout or b"")[:MAX_ANSWER_BYTES]
        if getattr(finished, "returncode", 0) != 0:
            return {"outcome": "error",
                    "detail": f"the evaluator exited {finished.returncode}: "
                              f"{bytes(finished.stderr or b'')[:400].decode('utf-8', 'replace')}"}
        try:
            answer = json.loads(out.decode("utf-8", "replace").strip() or "{}")
        except ValueError:
            return {"outcome": "error", "detail": "the evaluator did not answer in JSON"}
        if not isinstance(answer, dict):
            return {"outcome": "error", "detail": "the evaluator answered a non-object"}
        outcome = str(answer.get("outcome") or answer.get("status") or "error")
        if outcome not in OUTCOMES:
            return {"outcome": "error",
                    "detail": f"the evaluator said {outcome!r}, which is not an outcome"}
        return {"outcome": outcome, "detail": str(answer.get("detail") or "")[:400]}


def load_suite(path: os.PathLike[str] | str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the fixture suite the owner wrote. Bounded, and never a directory."""
    wanted = Path(str(path))
    if not wanted.is_file():
        raise EvaluationError(f"no suite file at {wanted}")
    raw = wanted.read_bytes()
    if len(raw) > MAX_SUITE_BYTES:
        raise EvaluationError(f"a suite is at most {MAX_SUITE_BYTES:,} bytes, this is "
                             f"{len(raw):,}")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise EvaluationError(f"{wanted} is not readable JSON: {error}") from None
    baseline: dict[str, Any] = {}
    if isinstance(document, Mapping):
        baseline = dict(document.get("baseline") or {})
        cases = document.get("cases")
    else:
        cases = document
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise EvaluationError('a suite is a list of cases, or {"cases": [...], "baseline": {...}}')
    found = [dict(item) for item in cases]
    if not found:
        raise EvaluationError("an empty suite asks nothing, so it can answer nothing; "
                              "declare at least one case")
    return found, baseline


def evaluate(settings, *, lesson: str, suite_path: os.PathLike[str] | str,
             model_version: str, code_version: str | None = None, promote: bool = True,
             runner: CommandRunner | None = None) -> dict[str, Any]:
    """Run one lesson's suite and report what the run means.

    Promotion is the run's own consequence, never a separate assurance: the ledger reads
    the recorded cases back, and ``activate_evaluation`` refuses a verdict that was not
    current or was produced by the same hand that proposed the lesson.
    """
    if not isinstance(lesson, str) or "@" in lesson or not lesson.strip():
        raise EvaluationError("name the lesson to evaluate, e.g. --lesson chase-invoice "
                              "(a version is chosen for you)")
    if not isinstance(model_version, str) or not model_version.strip():
        raise EvaluationError(
            "an evaluation has to name the model it was about: a verdict with no version "
            "beside it can never be found stale, which is the hole this whole path closes")
    argv = runner.argv if runner is not None else settings.evaluator_command
    if not argv:
        raise EvaluationError(
            "no evaluator is authorized (HERMES_MEMORY_EVALUATOR_COMMAND), so a lesson can "
            "be proposed, contradicted and retracted but never promoted by a run")
    cases, baseline = load_suite(suite_path)
    holder = runner or CommandRunner(argv, timeout_s=settings.evaluator_timeout_s,
                                     env_extra=settings.evaluator_env)
    from .. import __version__
    from ..storage.evidence import EvidenceStore

    with EvidenceStore(settings.db_path) as store:
        habits = _lessons(store, settings)
        current = habits.get(lesson)
        if current is None:
            raise EvaluationError(f"{lesson} has no version on file to evaluate")
        version = current.version
        ledger = EvaluationLedger(store, runner=holder,
                                 code_version=code_version or f"hermes-memory/{__version__}",
                                 model_version=model_version.strip(), lessons=habits,
                                 owner_principal=settings.owner_principal)
        # The two refer to each other on purpose: a lesson is promoted by a run, and a run
        # can only promote the lesson it was declared against. The link is made here rather
        # than in a constructor because neither can exist without the other.
        habits.evaluations = ledger
        report = ledger.run(lesson_id=lesson, version=int(version), cases=cases,
                            baseline=baseline, promote=promote)
        report["lesson"] = f"{lesson}@{version}"
        report["evaluated_at"] = now()
        report["suite"] = {"cases": len(cases), "path": str(suite_path)}
        report["evaluator"] = getattr(holder, "argv", ("(in-process)",))[0]
        return report


def report_on(settings, *, lesson: str,
              version: int | None = None) -> dict[str, Any]:
    """What the archive remembers about the last run for one lesson. Opens no socket."""
    from ..storage.evidence import ReadOnlyStore

    if not settings.db_path.is_file():
        raise EvaluationError(f"no canonical store at {settings.db_path}")
    with ReadOnlyStore(settings.db_path) as store:
        habits = _lessons(store, settings)
        # A ledger with no runner can read every verdict and start none, which is exactly
        # what a report should be able to do without pretending to authorize a run.
        ledger = EvaluationLedger.reading(store)
        current = habits.get(lesson)
        if current is None:
            raise EvaluationError(f"{lesson} has no version on file")
        wanted = int(version) if version else int(current.version)
        found = ledger.latest(lesson, wanted)
        history = [{"lesson": f"{item.id}@{item.version}", "status": item.status,
                    "proposed_by": item.created_by, "proposed_kind": item.created_kind,
                    "evaluation": item.evaluation_id or ""}
                   for item in habits.versions(lesson)]
        return {"lesson": f"{lesson}@{wanted}",
                "evaluation": found.as_dict() if found is not None else None,
                "evaluated": found is not None, "versions": history}


# -- internals ---------------------------------------------------------------

def _lessons(store, settings) -> LessonStore:
    return LessonStore(store,
                       outcomes=OutcomeLog(store, owner_principal=settings.owner_principal),
                       owner_principal=settings.owner_principal)


def _scrubbed_env(extra: Sequence[str] = ()) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if key in ENV_ALLOWANCE and isinstance(value, str)}
    for name in (str(item).strip() for item in extra):
        if name and name.isidentifier() and isinstance(os.environ.get(name), str):
            env[name] = os.environ[name]
    return env


def _jsonable(value: dict[str, Any]) -> dict[str, Any]:
    """The lesson as stored, with anything that cannot be serialized left out.

    The runner is a separate program, so the case and the claim are the whole of what it
    gets: no store handle, no credentials, no path to the archive it is being asked about.
    """
    out: dict[str, Any] = {}
    for key, item in value.items():
        try:
            json.dumps(item)
        except (TypeError, ValueError):
            continue
        out[str(key)] = item
    return out
