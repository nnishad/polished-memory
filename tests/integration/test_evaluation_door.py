"""C11 the authorized evaluation door, and the forgetting that can end a verdict.

A promotion has to rest on a run, and a run has to be scored by something the owner
authorized rather than by the system being graded. Until now the ledger that enforces that
had no caller at all: an installation could propose, contradict and retract lessons but
could never evaluate one, which left the plan's "promoted by a run" path a test fixture
rather than a feature.

These tests walk the whole door — a named program answers each case over a pipe, the
ledger weighs the answers, the lesson is promoted or not — and the other half: forgetting
the evidence a lesson was built on takes the passed run out of circulation, so the old
score cannot be pointed at again tomorrow.
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
from types import SimpleNamespace

import pytest
from learning_cases import CASES, RULE, propose

from conftest import envelope
from hermes_memory.backend.capabilities import CAPABILITIES, PINNED_VERSION, capabilities_for
from hermes_memory.config import load_settings
from hermes_memory.learning.evaluation import EvaluationLedger
from hermes_memory.learning.evaluator import (CommandRunner, EvaluationError, evaluate,
                                              load_suite, report_on)
from hermes_memory.learning.lessons import LessonStore
from hermes_memory.learning.outcomes import OutcomeLog
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.operations.doctor import Doctor
from hermes_memory.storage.evidence import EvidenceStore, EvidenceError as StoreError

OWNER = "jugaadu"
SUITE = list(CASES)


class AllPass:
    """Separately authorized evaluation machinery, or an imitation of it."""

    name = "fixture-suite"

    def evaluate(self, *, lesson, case):
        return {"outcome": "pass", "detail": "it did"}


def finished(stdout=b'{"outcome": "pass", "detail": "it did"}', returncode=0, stderr=b""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def runner_for(*answers, timeout_s=5.0, program="/usr/local/bin/hermes-fixture-runner"):
    """A CommandRunner whose subprocess is a list of recorded answers."""
    asked = []

    def execute(argv, **kwargs):
        asked.append({"argv": list(argv), **kwargs})
        answer = answers[min(len(asked), len(answers)) - 1]
        if isinstance(answer, BaseException):
            raise answer
        return finished(stdout=json.dumps(answer).encode())

    return CommandRunner([program, "--strict"], timeout_s=timeout_s, execute=execute), asked


@pytest.fixture()
def installation(tmp_path, monkeypatch):
    home = tmp_path / "instance"
    (home / "data").mkdir(parents=True)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n", encoding="utf-8")
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        store.commit(envelope(source_id="playbook-1",
                              text="Chase the invoice twice before phoning."))
    return settings


def cited(store) -> str:
    return with_evidence(store)


def candidate(settings, *, proposed_by="agent-x") -> dict:
    with EvidenceStore(settings.db_path) as store:
        return propose(lessons(store), cited(store), proposed_by=proposed_by)


def suite_file(tmp_path, payload=None):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(SUITE if payload is None else payload), encoding="utf-8")
    return path


def lessons(store, ledger=None):
    return LessonStore(store, outcomes=OutcomeLog(store, owner_principal=OWNER),
                       evaluations=ledger, owner_principal=OWNER)


def with_evidence(store) -> str:
    """One record to stand under a lesson, committed once per store."""
    found = store.db.execute("SELECT id FROM records WHERE deleted=0").fetchone()
    return str(found["id"]) if found else str(
        store.commit(envelope(source_id="playbook-1",
                              text="Chase the invoice twice before phoning."))["id"])


def evaluated(store):
    """A candidate, a run against it, and the two objects that know about each other."""
    with_evidence(store)
    habits = lessons(store)
    ledger = EvaluationLedger(store, runner=AllPass(), code_version="hm-test",
                              model_version="remote-9b", lessons=habits,
                              owner_principal=OWNER)
    habits.evaluations = ledger
    made = propose(habits, cited(store), proposed_by="agent-x")
    report = ledger.run(lesson_id="chase-invoice", version=made["version"], cases=SUITE)
    return habits, ledger, report


# -- the runner is a program the owner named, not a string that gets evaluated --

def test_a_relative_program_is_refused_before_anything_is_run():
    with pytest.raises(EvaluationError, match="absolute path"):
        CommandRunner(["hermes-fixture-runner"])


def test_the_program_is_executed_without_a_shell_and_with_its_arguments():
    holder, asked = runner_for({"outcome": "pass"}, program="/bin/echo")
    holder.evaluate(lesson={"text": "t"}, case={"case_id": "c1", "role": "targeted"})
    assert asked[0]["argv"] == ["/bin/echo", "--strict"]
    assert "shell" not in asked[0], "an evaluation runs a program, not a command line"


def test_the_child_sees_a_scrubbed_environment(monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HINDSIGHT_API_KEY", "do-not-leak")
    monkeypatch.setenv("PATH", "/usr/bin")
    seen = {}

    def execute(argv, **kwargs):
        seen.update(kwargs["env"])
        return finished()

    CommandRunner(["/bin/true"], timeout_s=5, execute=execute).evaluate(
        lesson={}, case={"case_id": "c", "role": "targeted"})
    assert "PATH" in seen and "HERMES_MEMORY_HINDSIGHT_API_KEY" not in seen


def test_an_explicitly_allowed_variable_is_still_the_operators_decision(monkeypatch):
    """The addition comes from configuration, so it cannot be widened by the caller."""
    monkeypatch.setenv("FIXTURE_MODE", "strict")
    seen = {}

    def execute(argv, **kwargs):
        seen.update(kwargs["env"])
        return finished()

    CommandRunner(["/bin/true"], timeout_s=5, env_extra=("FIXTURE_MODE",),
                  execute=execute).evaluate(
        lesson={}, case={"case_id": "c", "role": "targeted"})
    assert seen.get("FIXTURE_MODE") == "strict"
    assert "HERMES_MEMORY_EVALUATOR_ENV" not in seen


def test_the_answer_is_only_ever_a_case_result():
    holder, asked = runner_for({"outcome": "pass"})
    holder.evaluate(lesson={"text": "Chase twice"}, case={"case_id": "c1", "role": "targeted"})
    sent = json.loads(asked[0]["input"].decode())
    assert set(sent) == {"lesson", "case"}, "no store handle, no credentials, no paths"


@pytest.mark.parametrize("answer, outcome", [
    ({"outcome": "pass"}, "pass"),
    ({"status": "fail", "detail": "it rang the wrong person"}, "fail"),
])
def test_an_admissible_answer_is_taken_at_face_value(answer, outcome):
    holder, _ = runner_for(answer)
    verdict = holder.evaluate(lesson={}, case={"case_id": "c1", "role": "targeted"})
    assert verdict["outcome"] == outcome and "detail" in verdict


@pytest.mark.parametrize("raw", [b'{"outcome": "magnificent"}', b"not json at all",
                                 b"[1, 2, 3]", b'{"outcome": null}'])
def test_an_unusable_answer_becomes_an_error_and_never_a_pass(raw):
    holder = CommandRunner(["/bin/true"], timeout_s=5,
                           execute=lambda argv, **kwargs: finished(stdout=raw))
    answer = holder.evaluate(lesson={}, case={"case_id": "c", "role": "targeted"})
    assert answer["outcome"] == "error"


def test_a_runner_that_exited_badly_did_not_pass():
    holder = CommandRunner(["/bin/false"], timeout_s=5,
                           execute=lambda argv, **kwargs: finished(stdout=b"", returncode=2,
                                                                   stderr=b"boom"))
    answer = holder.evaluate(lesson={}, case={"case_id": "c", "role": "targeted"})
    assert answer["outcome"] == "error" and "boom" in answer["detail"]


def test_a_hung_evaluation_is_cut_off_rather_than_waiting_forever():
    def hang(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])

    holder = CommandRunner(["/bin/sleep"], timeout_s=1, execute=hang)
    answer = holder.evaluate(lesson={}, case={"case_id": "c", "role": "targeted"})
    assert answer["outcome"] == "error" and "did not answer within" in answer["detail"]


def test_a_program_that_cannot_be_found_is_reported_as_a_failure_to_answer():
    def missing(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    holder = CommandRunner(["/nonexistent/runner"], timeout_s=5, execute=missing)
    answer = holder.evaluate(lesson={}, case={"case_id": "c", "role": "targeted"})
    assert answer["outcome"] == "error"


@pytest.mark.parametrize("timeout", [0, 0.5, 99_999])
def test_the_timeout_is_named_in_seconds_and_bounded(timeout):
    with pytest.raises(EvaluationError, match="between 1 and 3600"):
        CommandRunner(["/bin/true"], timeout_s=timeout)


# -- the suite is a file somebody wrote --------------------------------------

def test_a_suite_may_be_a_list_or_a_document_with_a_baseline(tmp_path):
    cases, baseline = load_suite(suite_file(tmp_path, SUITE))
    assert cases == SUITE and baseline == {}
    cases, baseline = load_suite(suite_file(tmp_path, {"cases": SUITE,
                                                      "baseline": {"model": "remote-9b"}}))
    assert cases == SUITE and baseline == {"model": "remote-9b"}


@pytest.mark.parametrize("payload", ['{"cases": "not a list"}', "[]", "not json",
                                     '"a string"'])
def test_a_suite_that_cannot_be_run_is_refused_before_the_store_is_opened(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(EvaluationError):
        load_suite(path)


def test_a_missing_suite_says_which_file_it_wanted(tmp_path):
    with pytest.raises(EvaluationError, match="no suite file"):
        load_suite(tmp_path / "nothing.json")


def test_an_oversized_suite_is_refused_rather_than_half_read(tmp_path):
    path = tmp_path / "big.json"
    path.write_text(json.dumps([{"case_id": f"c{i}", "role": "targeted", "pad": "x" * 400}
                                for i in range(4000)]), encoding="utf-8")
    with pytest.raises(EvaluationError, match="at most"):
        load_suite(path)


# -- the door ----------------------------------------------------------------

def test_nothing_can_be_promoted_without_an_authorized_evaluator(installation, tmp_path):
    candidate(installation)
    with pytest.raises(EvaluationError, match="no evaluator is authorized"):
        evaluate(installation, lesson="chase-invoice", suite_path=suite_file(tmp_path),
                 model_version="remote-9b")


def test_a_verdict_about_no_version_of_the_model_is_refused(installation, tmp_path):
    holder, _ = runner_for({"outcome": "pass"})
    candidate(installation)
    with pytest.raises(EvaluationError, match="can never be found stale"):
        evaluate(installation, lesson="chase-invoice", suite_path=suite_file(tmp_path),
                 model_version="  ", runner=holder)


def test_a_passing_run_promotes_the_lesson_and_says_who_did_it(installation, tmp_path):
    holder, asked = runner_for({"outcome": "pass"})
    made = candidate(installation)
    report = evaluate(installation, lesson="chase-invoice", suite_path=suite_file(tmp_path),
                     model_version="remote-9b", runner=holder)
    assert report["verdict"] == "passed" and report["promoted"] is True
    assert report["lesson"] == f"chase-invoice@{made['version']}"
    assert len(asked) == 2, "one question per declared case"
    assert report["evaluator"] == "/usr/local/bin/hermes-fixture-runner"
    with EvidenceStore(installation.db_path) as store:
        assert lessons(store).get("chase-invoice").status == "active"


def test_a_run_can_be_recorded_without_letting_it_promote_anything(installation, tmp_path):
    holder, _ = runner_for({"outcome": "pass"})
    candidate(installation)
    report = evaluate(installation, lesson="chase-invoice", suite_path=suite_file(tmp_path),
                     model_version="remote-9b", runner=holder, promote=False)
    assert report["verdict"] == "passed" and report["promoted"] is False
    with EvidenceStore(installation.db_path) as store:
        assert lessons(store).get("chase-invoice").status == "candidate"


def test_a_broken_regression_yields_no_promotion(installation, tmp_path):
    holder, _ = runner_for({"outcome": "pass"},
                           {"outcome": "fail", "detail": "the meeting moved"})
    candidate(installation)
    report = evaluate(installation, lesson="chase-invoice", suite_path=suite_file(tmp_path),
                     model_version="remote-9b", runner=holder)
    assert report["verdict"] == "failed" and report["promoted"] is False
    assert "regression case(s) broke" in report["detail"]


def test_the_proposer_cannot_score_its_own_lesson(installation, tmp_path):
    """The named program is the same hand that wrote the rule, so the run licenses nothing."""
    holder, _ = runner_for({"outcome": "pass"})
    holder.name = "fixture-suite"
    made = candidate(installation, proposed_by="fixture-suite")
    with pytest.raises(StoreError, match="also ran its evaluation"):
        evaluate(installation, lesson="chase-invoice", suite_path=suite_file(tmp_path),
                 model_version="remote-9b", runner=holder)
    with EvidenceStore(installation.db_path) as store:
        assert lessons(store).get("chase-invoice", version=made["version"]).status == \
            "candidate", "a refused promotion leaves the lesson where it was"


def test_a_lesson_that_does_not_exist_is_named_in_the_refusal(installation, tmp_path):
    holder, _ = runner_for({"outcome": "pass"})
    with pytest.raises(EvaluationError, match="no version on file"):
        evaluate(installation, lesson="never-proposed", suite_path=suite_file(tmp_path),
                 model_version="remote-9b", runner=holder)


def test_the_version_is_resolved_from_the_archive_not_from_whoever_asked(installation,
                                                                        tmp_path):
    holder, _ = runner_for({"outcome": "pass"})
    made = candidate(installation)
    report = evaluate(installation, lesson="chase-invoice", suite_path=suite_file(tmp_path),
                     model_version="remote-9b", runner=holder)
    assert report["lesson"].endswith(f"@{made['version']}")


@pytest.mark.parametrize("lesson", ["", "   ", "chase@3"])
def test_a_lesson_addressed_by_version_is_refused_and_asked_for_by_name(installation,
                                                                      tmp_path, lesson):
    holder, _ = runner_for({"outcome": "pass"})
    with pytest.raises(EvaluationError, match="name the lesson"):
        evaluate(installation, lesson=lesson, suite_path=suite_file(tmp_path),
                 model_version="remote-9b", runner=holder)


def configured_evaluator(tmp_path, monkeypatch, extra: str):
    """An installation whose owner named a program, and whatever else they allowed."""
    home = tmp_path / "instance"
    (home / "data").mkdir(parents=True)
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n" + extra, encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        with_evidence(store)
    candidate(settings)
    return settings


def answering_program(tmp_path, records_to: str):
    """A real program that answers every case and leaves its own environment on disk.

    The door's scrubbing is a claim about a child process, so it is checked against one
    rather than against a patched ``subprocess.run`` — which the door would not see anyway,
    since its default argument bound the real function at import.
    """
    script = tmp_path / "answer.sh"
    script.write_text('#!/bin/sh\nenv > "' + records_to + '"\n'
                      'printf \'%s\' \'{"outcome": "pass"}\'\n', encoding="utf-8")
    script.chmod(0o755)
    return str(script)


def test_the_environment_an_evaluator_sees_comes_from_the_owners_file(tmp_path, monkeypatch):
    """The allowance is configuration, so the door has to carry it to the child.

    Wired by hand at the call site instead, a caller could widen what a scored run reads —
    which is the one thing a scrubbed environment exists to prevent.
    """
    recorded = tmp_path / "child.env"
    settings = configured_evaluator(
        tmp_path, monkeypatch,
        f"HERMES_MEMORY_EVALUATOR_COMMAND={answering_program(tmp_path, str(recorded))}\n"
        "HERMES_MEMORY_EVALUATOR_ENV=FIXTURE_MODE\n")
    monkeypatch.setenv("FIXTURE_MODE", "strict")
    monkeypatch.setenv("HERMES_MEMORY_HINDSIGHT_API_KEY", "not-for-a-child-process")
    report = evaluate(settings, lesson="chase-invoice", suite_path=suite_file(tmp_path),
                      model_version="remote-9b")
    assert report["verdict"] == "passed"
    seen = recorded.read_text(encoding="utf-8")
    assert "FIXTURE_MODE=strict" in seen, "named by the owner, so it arrived"
    assert "not-for-a-child-process" not in seen
    with EvidenceStore(settings.db_path) as store:
        lesson = store.db.execute("SELECT decided_by, evaluation_id FROM lessons "
                                 "WHERE status='active'").fetchone()
        run = store.db.execute("SELECT runner FROM evaluations WHERE id=?",
                               (lesson["evaluation_id"],)).fetchone()
    assert run["runner"].endswith("answer.sh"), "the run keeps the whole path it was told"
    assert lesson["decided_by"].startswith("evaluation:answer.sh:"), \
        "and the promotion names the program that scored it, in the space an actor has"


def test_nothing_is_allowed_through_until_the_owner_names_it(tmp_path, monkeypatch):
    recorded = tmp_path / "child.env"
    settings = configured_evaluator(
        tmp_path, monkeypatch,
        f"HERMES_MEMORY_EVALUATOR_COMMAND={answering_program(tmp_path, str(recorded))}\n")
    monkeypatch.setenv("FIXTURE_MODE", "strict")
    evaluate(settings, lesson="chase-invoice", suite_path=suite_file(tmp_path),
             model_version="remote-9b")
    assert "FIXTURE_MODE" not in recorded.read_text(encoding="utf-8"), \
        "an allowance has to be asked for by name"


# -- the reading half ---------------------------------------------------------

def test_a_lesson_that_was_never_run_reports_that_instead_of_nothing_at_all(installation):
    made = candidate(installation)
    report = report_on(installation, lesson="chase-invoice")
    assert report["evaluated"] is False and report["evaluation"] is None
    assert report["lesson"] == f"chase-invoice@{made['version']}"
    statuses = [item["status"] for item in report["versions"]]
    assert statuses == ["candidate"], "the version history is what an owner reads first"


def test_the_last_run_can_be_read_without_an_evaluator(installation, tmp_path):
    holder, _ = runner_for({"outcome": "pass"})
    candidate(installation)
    evaluate(installation, lesson="chase-invoice", suite_path=suite_file(tmp_path),
             model_version="remote-9b", runner=holder)
    report = report_on(installation, lesson="chase-invoice")
    assert report["evaluated"] is True
    assert report["evaluation"]["verdict"] == "passed"
    assert report["evaluation"]["runner"] == "/usr/local/bin/hermes-fixture-runner"


def test_reading_never_authorizes_a_run(installation):
    """A report must not be able to start one, even by accident."""
    candidate(installation)
    with EvidenceStore(installation.db_path) as store:
        ledger = EvaluationLedger.reading(store)
        assert ledger.runner is None
        with pytest.raises(StoreError, match="cannot start one"):
            ledger.begin(lesson_id="chase-invoice", version=1, cases=SUITE)


def test_a_report_without_an_archive_says_which_file_is_missing(tmp_path, monkeypatch):
    home = tmp_path / "untouched"
    (home / "data").mkdir(parents=True)
    (home / "hermes-memory.env").write_text(f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n",
                                            encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(EvaluationError, match="no canonical store"):
        report_on(load_settings(), lesson="chase-invoice")


# -- forgetting ends a verdict ------------------------------------------------

def test_forgetting_the_evidence_stales_the_run_behind_the_lesson(installation):
    with EvidenceStore(installation.db_path) as store:
        habits, ledger, report = evaluated(store)
        assert report["promoted"] is True
        manager = ErasureManager(store, owner_principal=OWNER)
        preview = manager.preview(record_ids=[cited(store)], actor=OWNER, reason="not ours")
        outcome = manager.confirm(intent_id=preview["intent_id"],
                                 preview_digest=preview["preview_digest"], actor=OWNER)
        assert outcome["evaluations_staled"] == 1, (
            "the owner is told a licence went, not only a record")
        row = store.db.execute("SELECT verdict, detail FROM evaluations WHERE id=?",
                              (report["id"],)).fetchone()
        assert row["verdict"] == "stale" and "erasure intent" in str(row["detail"])
        assert habits.get("chase-invoice").evaluation_id == report["id"]


def test_a_staled_run_cannot_promote_a_lesson_again(store):
    habits, ledger, report = evaluated(store)
    ledger.invalidate(report["id"], reason="the fixture turned out to be wrong")
    assert ledger.get(report["id"]).verdict == "stale"
    with pytest.raises(StoreError, match="no longer a verdict about"):
        habits.activate_evaluation(lesson_id="chase-invoice", version=1,
                                   evaluation_id=report["id"])


def test_an_invalidation_needs_a_reason(store):
    _habits, ledger, report = evaluated(store)
    with pytest.raises(StoreError, match="reason"):
        ledger.invalidate(report["id"], reason="   ")


def test_a_lesson_whose_evidence_survives_keeps_its_verdict(installation):
    with EvidenceStore(installation.db_path) as store:
        _habits, ledger, report = evaluated(store)
        other = store.commit(envelope(source_id="elsewhere", text="an unrelated note"))["id"]
        manager = ErasureManager(store, owner_principal=OWNER)
        preview = manager.preview(record_ids=[other], actor=OWNER, reason="not ours")
        outcome = manager.confirm(intent_id=preview["intent_id"],
                                 preview_digest=preview["preview_digest"], actor=OWNER)
        assert outcome["evaluations_staled"] == 0
        assert ledger.get(report["id"]).verdict == "passed"


def test_a_refused_confirmation_leaves_the_licence_standing(installation):
    with EvidenceStore(installation.db_path) as store:
        _habits, ledger, report = evaluated(store)
        manager = ErasureManager(store, owner_principal=OWNER)
        preview = manager.preview(record_ids=[cited(store)], actor=OWNER, reason="gone")
        with pytest.raises(StoreError, match="does not match"):
            manager.confirm(intent_id=preview["intent_id"], preview_digest="0" * 64,
                            actor=OWNER)
        assert ledger.get(report["id"]).verdict == "passed"
        assert store.db.execute("SELECT count(*) FROM evaluations").fetchone()[0] == 1


def test_a_lesson_citing_other_evidence_keeps_its_verdict(store):
    with_evidence(store)
    habits = lessons(store)
    ledger = EvaluationLedger(store, runner=AllPass(), code_version="hm-test",
                              model_version="remote-9b", lessons=habits)
    habits.evaluations = ledger
    made = propose(habits, cited(store), lesson_id="other-lesson", proposed_by="agent-x")
    report = ledger.run(lesson_id="other-lesson", version=made["version"], cases=SUITE)
    assert report["verdict"] == "passed"

    other = store.commit(envelope(source_id="elsewhere", text="something to forget"))["id"]
    manager = ErasureManager(store, owner_principal=OWNER)
    preview = manager.preview(record_ids=[other], actor=OWNER, reason="not ours")
    outcome = manager.confirm(intent_id=preview["intent_id"],
                             preview_digest=preview["preview_digest"], actor=OWNER)
    assert outcome["evaluations_staled"] == 0
    assert ledger.get(report["id"]).verdict == "passed"


def test_a_cited_span_is_read_as_the_record_it_quoted(installation):
    """A citation may carry which revision or fragment it meant; that is still one record.

    Reading the suffix as part of the identity would stale a lesson's verdict because an
    unrelated forgetting happened, and then refuse a promotion the surviving evidence
    still supports.
    """
    with EvidenceStore(installation.db_path) as store:
        record = with_evidence(store)
        habits = lessons(store)
        made = habits.propose(lesson_id="spanned", text="A lesson quoted by span",
                              applicability=RULE, evidence=[f"{record}@1#head"],
                              proposed_by="agent-x", proposed_kind="agent")
        ledger = EvaluationLedger(store, runner=AllPass(), code_version="hm-test",
                                  model_version="remote-9b", lessons=habits,
                                  owner_principal=OWNER)
        habits.evaluations = ledger
        report = ledger.run(lesson_id="spanned", version=made["version"], cases=SUITE)
        assert report["promoted"] is True

        manager = ErasureManager(store, owner_principal=OWNER)
        other = store.commit(envelope(source_id="elsewhere", text="something to forget"))["id"]
        preview = manager.preview(record_ids=[other], actor=OWNER, reason="not ours")
        outcome = manager.confirm(intent_id=preview["intent_id"],
                                 preview_digest=preview["preview_digest"], actor=OWNER)
        assert outcome["evaluations_staled"] == 0
        assert ledger.get(report["id"]).verdict == "passed"

        preview = manager.preview(record_ids=[record], actor=OWNER, reason="the quoted part")
        outcome = manager.confirm(intent_id=preview["intent_id"],
                                 preview_digest=preview["preview_digest"], actor=OWNER)
        assert outcome["evaluations_staled"] == 1


# -- the doctor says which routes are actually served -------------------------

class Answers:
    def __init__(self, *, routed=None):
        self.routed = routed

    def health(self):
        return {"version": PINNED_VERSION}

    def negotiate(self, *, probe_routes=True):
        declared = {cap.name for cap in CAPABILITIES}
        return capabilities_for(PINNED_VERSION,
                                route_names=declared if self.routed is None else self.routed)


def test_a_backend_routing_everything_pinned_is_simply_available(store):
    finding = Doctor(store, backend=lambda: Answers()).backend_connectivity()
    assert finding.severity == "ok"
    assert finding.evidence["observed_via"] == "routes", "the stub says what it routed"
    assert finding.evidence["not_routed"] == []
    assert finding.evidence["pinned_to"] == PINNED_VERSION


def test_a_backend_that_does_not_route_a_pinned_capability_is_a_warning(store):
    routed = {cap.name for cap in CAPABILITIES} - {"reflect"}
    finding = Doctor(store, backend=lambda: Answers(routed=routed)).backend_connectivity()
    assert finding.severity == "warn"
    assert "reflect" in finding.evidence["not_routed"]
    assert "does not route" in finding.detail
    assert "tag" in finding.remedy


def test_an_unconfirmable_capability_set_is_reported_as_such(store):
    class Silent(Answers):
        def negotiate(self, *, probe_routes=True):
            raise RuntimeError("no route list")

    finding = Doctor(store, backend=lambda: Silent()).backend_connectivity()
    assert finding.severity == "warn" and "unconfirmed" in finding.detail


# -- the door speaks through the CLI too --------------------------------------

def run(*args):
    from hermes_memory.cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(args))
    text = out.getvalue()
    return code, (json.loads(text) if text.strip().startswith("{") else {}), err.getvalue()


def test_the_evaluate_door_reports_without_running_anything(installation):
    made = candidate(installation)
    code, report, _ = run("evaluate", "--lesson", "chase-invoice")
    assert code == 0 and report["lesson"] == f"chase-invoice@{made['version']}"
    assert report["evaluated"] is False


def test_the_door_refuses_to_invent_an_evaluator(installation, tmp_path):
    candidate(installation)
    code, report, refused = run("evaluate", "--lesson", "chase-invoice",
                                "--suite", str(suite_file(tmp_path)),
                                "--model-version", "remote-9b")
    assert code == 2 and report == {}
    assert "no evaluator is authorized" in refused
