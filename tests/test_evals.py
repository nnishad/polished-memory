"""The evaluation harness has to be able to fail, or it is a press release.

These tests run real parts of the harness over the real corpus — slow by unit-test standards,
and worth it, because the alternative is a suite that checks the plumbing of a report nobody
has produced. On top of the standing green baseline, each case breaks one property upstream
(duplicate a code, mislabel a scenario, silence the gate) and requires the report to notice.
A harness that cannot be made to say "no" is not measuring anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVALS = Path(__file__).resolve().parents[1] / "evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

import checks                                # noqa: E402
import corpus                                # noqa: E402
import run_synthetic                         # noqa: E402

# §12.4's table, in the plan's own words, plus the areas this harness owes its own
# correctness to: the corpus that feeds the numbers, and the lifecycle of a correction.
PLAN_ROWS = ("privacy/lifecycle", "capture/sync", "extraction", "retrieval", "proactivity",
             "compute", "context latency", "formation latency", "operations", "install",
             "recovery")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    """One full run, shared: it is the slowest thing in this suite by a wide margin."""
    return run_synthetic.evaluate(tmp_path_factory.mktemp("evals-clean"))


@pytest.fixture()
def env(tmp_path):
    built = checks.Environment(root=tmp_path)
    checks.ingest(built)
    try:
        yield built
    finally:
        built.close()


def verdict(name: str, value):
    return [checks.measured("synthetic", name, value=value, criterion="= 1",
                            passes=bool(value), denominator=1)]


def ingested(root: Path) -> checks.Environment:
    """A scratch installation with the corpus as it currently reads.

    Built after any patching, on purpose: a corpus sabotaged *after* ingestion would be
    measured against a store that never held the sabotage, and the row would go red for the
    wrong reason.
    """
    built = checks.Environment(root=root)
    checks.ingest(built)
    return built


# -- the shape of the report --------------------------------------------------

def test_every_row_of_the_metric_table_is_answered(report):
    areas = {row["row"] for row in report["results"]}
    assert set(PLAN_ROWS) <= areas, sorted(set(PLAN_ROWS) - areas)


def test_each_row_is_either_a_number_or_an_explicit_refusal(report):
    for row in report["results"]:
        assert row["status"] in (checks.MEASURED, checks.NOT_MEASURED), row
        if row["status"] == checks.MEASURED:
            assert row["criterion"] and row["denominator"] is not None, row
            assert isinstance(row["pass"], bool), row
            assert "detail" in row, row
        else:
            assert row["pass"] is None, "a thing not measured cannot vote on the release"
            assert row["because"] and row["requires"], row


def test_the_report_survives_being_written_down(report):
    assert json.loads(json.dumps(report, default=str))["summary"]["measured"] > 0


def test_a_clean_tree_leaves_nothing_failing(report):
    assert report["ok"] is True, json.dumps(report["failures"], indent=1)[:2000]


def test_every_measured_row_carries_the_denominator_it_was_scored_over(report):
    # The plan's gates are ratios, and a ratio without its denominator is how 1-of-1 becomes a
    # headline. Each check states what it counted; this asserts the numbers add up against the
    # corpus's own account of itself.
    totals = corpus.totals()
    by_name = {row["check"]: row for row in report["results"]}
    assert by_name["required-evidence recall (lexical route, top 8)"]["denominator"] == \
        totals["answerable"]
    assert by_name["abstention on questions with no answer"]["denominator"] == \
        totals["abstentions"]
    assert by_name["matched no-action cases stay quiet"]["denominator"] == \
        totals["no_action_scenarios"]
    assert by_name["eligible moments are actually surfaced (recall)"]["denominator"] == \
        totals["eligible_scenarios"]


def test_one_failing_row_makes_the_whole_run_fail(tmp_path, monkeypatch):
    """The exit status is the gate, so it has to be derived from every row."""
    monkeypatch.setattr(run_synthetic, "SUITE",
                        [lambda environment: verdict("nothing is wrong", 1),
                         lambda environment: verdict("something is wrong", 0)])
    broken = run_synthetic.evaluate(tmp_path)

    assert broken["ok"] is False
    assert [row["check"] for row in broken["failures"]] == ["something is wrong"]


def test_a_row_that_is_not_measured_neither_passes_nor_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(run_synthetic, "SUITE",
                        [lambda environment: [checks.not_measured(
                            "extraction", "quality", because="no model ran",
                            requires="the P0 gate")]])
    report = run_synthetic.evaluate(tmp_path)

    assert report["ok"] is True
    # One patched row plus the standing refusals: none of them measured, none of them a
    # failure, and the release still does not claim they passed.
    assert report["summary"]["measured"] == 0
    assert report["summary"]["failed"] == 0
    assert report["summary"]["not_measured"] == report["summary"]["checks"] == 5


# -- the harness can be made to say no ----------------------------------------

def test_a_duplicated_code_is_reported_as_a_broken_corpus(tmp_path, monkeypatch):
    """Recall is only meaningful if a code names one record; the harness must notice when it
    does not, rather than scoring a coin flip."""
    real = corpus.records()

    def records():
        twin = dict(real[0])
        twin["source_id"] = "duplicate-code"
        twin["revision"] = "9"
        twin["text"] = twin["text"].replace("[ref", "[twin ref")
        return [*real, twin]

    monkeypatch.setattr(corpus, "records", records)
    built = ingested(tmp_path)
    try:
        rows = checks.check_corpus(built)
    finally:
        built.close()

    assert [row["check"] for row in rows if not row["pass"]] == ["one distinct code per record"]


def test_a_missing_category_is_reported_rather_than_gone_quietly(tmp_path, monkeypatch):
    """The declared list is the claim; a template that stops being written is the drift."""
    real = corpus._templates

    def templates(person, index):
        return [row for row in real(person, index) if row[0] != "negation"]

    monkeypatch.setattr(corpus, "_templates", templates)
    built = ingested(tmp_path)
    try:
        rows = checks.check_corpus(built)
    finally:
        built.close()

    assert any(row["check"] == "every §12.3 category is represented" and not row["pass"]
               for row in rows), rows


def test_a_silence_that_is_not_for_the_named_reason_fails_the_mechanism_row(
        env, monkeypatch):
    """The precision gate is only worth having if the matched no-action cases are quiet *for
    the reason they were written*. Opt-out is scored here against the coverage stage: the
    framework still stays silent, and the report says that is not the same silence."""
    monkeypatch.setitem(checks.NO_ACTION_MECHANISMS, "opt-out", ("coverage", "not registered"))

    rows = checks.check_proactivity(env)

    row = next(item for item in rows if item["check"] == "every refusal is the mechanism it claims")
    assert row["pass"] is False
    assert row["detail"]["examples"], "the report has to say which case it means"


def test_a_gate_that_never_speaks_fails_the_recall_row(env, monkeypatch):
    """Permanent silence is how a proactivity gate cheats: precision 100%, no interruptions,
    and a memory that never tells the owner anything. Every answer is forced silent, and the
    recall row is what catches it."""
    from dataclasses import replace

    from hermes_memory.proactive.policy import AttentionPolicy

    real = AttentionPolicy.decide

    def mute(self, **kwargs):
        return replace(real(self, **kwargs), action="silent")

    monkeypatch.setattr(AttentionPolicy, "decide", mute)
    rows = checks.check_proactivity(env)

    row = next(item for item in rows
               if item["check"] == "eligible moments are actually surfaced (recall)")
    assert row["value"] == 0.0 and row["pass"] is False


def test_an_answer_where_none_should_exist_fails_the_abstention_row(env):
    """120 of the held-out questions have no answer in the corpus. Put an answer in for one of
    them and the abstention row has to go red — otherwise it is reporting the scorer's
    optimism rather than the archive's silence."""
    ghost = corpus.UNKNOWN_CODES[0]
    row = corpus.records()[0]
    env.corpus_store.commit({**row, "source_id": "fabricated-answer", "revision": "1",
                             "text": f"An answer that should not have been there [ref {ghost}]",
                             "metadata": {"code": ghost, "category": "unknown-answer"}})

    item = next(item for item in checks.check_retrieval(env)
                if item["check"] == "abstention on questions with no answer")

    assert item["pass"] is False
    assert item["value"] < 1.0


def test_a_replay_that_brings_a_new_row_is_reported(env, monkeypatch):
    """A replay is supposed to change nothing. Give the second pass a record the store has
    never seen and the capture row must say so rather than call it a harmless no-op."""
    assert all(row["pass"] for row in checks.check_capture(env))
    real = corpus.records()

    def records():
        extra = dict(real[0])
        extra["source_id"] = "arrives-on-the-second-pass"
        extra["text"] = extra["text"].replace("[ref", "[second ref")
        return [*real, extra]

    monkeypatch.setattr(corpus, "records", records)

    row = next(item for item in checks.check_capture(env)
               if item["check"] == "a replayed page forks no second record")

    assert row["pass"] is False
    assert row["detail"]["new_rows"] == 1


def test_the_tripwire_notices_the_reach_it_was_built_for():
    """The compute row claims zero model requests, and that claim is only as good as the wire
    it is measured with: touching it has to leave a mark and stop answering."""
    wire = checks.Tripwire()

    with pytest.raises(AssertionError, match="must not reach a model"):
        wire.retain(bank_id="hermes")

    assert wire.reached == ["retain"]


def test_a_record_left_visible_after_a_correction_fails_the_retrieval_row(env):
    """The corrected-answer row is the retrieval face of the same invariant the store
    enforces: undo the hiding, put the sentence back in the index, and the row goes red."""
    assert all(row["pass"] for row in checks.check_retrieval(env))
    store = env.corpus_store
    stale = store.db.execute(
        "SELECT record_id FROM record_visibility WHERE replacement_id IS NOT NULL LIMIT 1"
    ).fetchone()
    store.db.execute("DELETE FROM record_visibility WHERE record_id=?", (stale["record_id"],))
    store.db.execute("INSERT INTO record_fts(id, text) SELECT id, text FROM records WHERE id=?",
                     (stale["record_id"],))
    row = next(item for item in checks.check_retrieval(env)
               if item["check"] == "a corrected answer is not returned as if it still stood")

    assert row["pass"] is False


# -- the command --------------------------------------------------------------

def test_a_dirty_scratch_directory_is_refused_rather_than_wiped(tmp_path, capsys):
    (tmp_path / "someone-elses-work").write_text("keep me", encoding="utf-8")

    assert run_synthetic.main(["--root", str(tmp_path)]) == 2
    assert "not empty" in capsys.readouterr().err
    assert (tmp_path / "someone-elses-work").read_text(encoding="utf-8") == "keep me"


def test_the_json_report_is_the_same_answer_as_the_table(tmp_path, capsys):
    target = tmp_path / "report.json"
    code = run_synthetic.main(["--root", str(tmp_path / "run"), "--json", str(target),
                               "--quiet"])
    written = json.loads(target.read_text(encoding="utf-8"))

    assert code == 0
    assert written["ok"] is True
    assert written["summary"]["failed"] == 0
    assert len(written["results"]) == written["summary"]["checks"]
    assert "measured checks passed" in capsys.readouterr().out
