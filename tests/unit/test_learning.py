"""C11 evaluated procedural learning: a promotion has to come from a run.

The gate items here are the ones the plan named: forged pass claims, stale
evaluations, non-applicable tasks, regressions, retraction, and learning recursively
from this system's own unverified summaries.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from conftest import envelope
from learning_cases import AGENT, CASES, INVOICE_RULE, RULE, Runner, propose
from hermes_memory.learning.evaluation import EvaluationLedger
from hermes_memory.learning.lessons import (ACTOR_LIMIT, LessonStore, match,
                                           promotion_actor)
from hermes_memory.learning.outcomes import OutcomeLog
from hermes_memory.ids import digest
from hermes_memory.proactive.outbox import Outbox
from hermes_memory.proactive.policy import POLICY_VERSION, AttentionPolicy
from hermes_memory.storage.evidence import EvidenceError

OWNER = "owner-principal"
MORNING = "2026-09-15T09:00:00+00:00"
EPOCH = 1789462800.0


@pytest.fixture()
def outcomes(store):
    return OutcomeLog(store, owner_principal=OWNER)


@pytest.fixture()
def receipts(store, stack_outbox):
    """An outcome log that can check a host receipt against a real delivery."""
    return OutcomeLog(store, owner_principal=OWNER, outbox=stack_outbox[0])


@pytest.fixture()
def lessons(store, outcomes):
    return LessonStore(store, outcomes=outcomes, owner_principal=OWNER)


@pytest.fixture()
def ledger(store, lessons):
    runner = Runner()
    subject = EvaluationLedger(store, runner=runner, code_version="hm-0.4",
                              model_version="remote-9b", lessons=lessons)
    lessons.evaluations = subject
    subject.runner = runner
    return subject


@pytest.fixture()
def evidence(store):
    return str(store.commit(envelope(source_id="playbook-1",
                                     text="Chase the invoice twice before phoning."))["id"])


def activate(store, lessons, ledger, evidence, lesson_id="chase-invoice"):
    made = propose(lessons, evidence, lesson_id=lesson_id)
    ledger.run(lesson_id=lesson_id, version=made["version"], cases=list(CASES))
    return made


# -- outcomes: three kinds that must not add up ------------------------------

def test_the_system_only_offers_candidates(store, lessons, evidence):
    made = propose(lessons, evidence)
    assert made["status"] == "candidate"
    assert lessons.applicable({"source": "gmail"}) == [], \
        "a proposal must not start out teaching anything"


def test_a_lesson_without_evidence_is_an_opinion(store, lessons):
    with pytest.raises(EvidenceError, match="no evidence"):
        propose(lessons, "", evidence=[])


@pytest.mark.parametrize("span", ["sum_8891", "hdoc0abc"])
def test_a_lesson_cannot_be_learned_from_its_own_summary(store, lessons, span):
    """The recursive-learning gate: a derived reading must not teach the deriver."""
    with pytest.raises(EvidenceError, match="derived artifact"):
        propose(lessons, "", evidence=[span])


def test_a_citation_has_to_be_live_evidence(store, lessons, evidence):
    with pytest.raises(EvidenceError, match="not live evidence"):
        propose(lessons, evidence, evidence=["rec_does_not_exist"])


@pytest.mark.parametrize("rule", [
    {"all": [{"field": "mood", "op": "eq", "value": "sunny"}]},
    {"all": []}, {"any": "not a list"}, {"all": [{"field": "source", "op": "sounds_like",
                                                 "value": "gmail"}]},
    {"all": [{"field": "source", "op": "eq", "value": " "}]},
    {},
])
def test_a_rule_the_store_cannot_evaluate_is_not_a_lesson(store, lessons, evidence, rule):
    with pytest.raises(EvidenceError):
        propose(lessons, evidence, applicability=rule)


def test_an_assistant_claim_is_filed_and_never_counted(store, outcomes):
    outcomes.record(subject_kind="lesson", subject_id="chase-invoice@1",
                    kind="assistant_claim", valence="success", note="it worked fine",
                    actor=AGENT)
    tally = outcomes.tally("lesson", "chase-invoice@1")
    assert tally["claimed"] == 1 and tally["support"] == 0 and tally["net"] == 0
    assert "never counted" not in tally["note"] and "excluded" in tally["note"]


def test_a_host_receipt_is_checked_against_the_delivery_it_names(store, receipts,
                                                           stack_outbox):
    artifact = stack_outbox[1]
    receipts.record(subject_kind="artifact", subject_id=artifact, kind="host_receipt",
                    valence="success", note="the host echoed the digest", actor="hermes",
                    evidence=[artifact])
    assert receipts.tally("artifact", artifact)["support"] == 1


def test_a_receipt_cannot_name_an_artifact_that_never_existed(store, receipts,
                                                             stack_outbox):
    with pytest.raises(EvidenceError, match="not an artifact this store ever prepared"):
        receipts.record(subject_kind="artifact", subject_id=stack_outbox[1],
                        kind="host_receipt", valence="success", note="sent",
                        actor="hermes", evidence=["out_never_existed"])


def test_a_receipt_for_an_unsent_artifact_is_not_a_receipt(store, receipts, stack_outbox):
    """An artifact that is on file and never went out is not evidence of a send."""
    outbox, artifact_id = stack_outbox
    store.db.execute("UPDATE outbox SET state='prepared', delivered_at=NULL WHERE id=?",
                     (artifact_id,))
    assert outbox.get(artifact_id).state == "prepared"
    with pytest.raises(EvidenceError, match="nothing reached anybody"):
        receipts.record(subject_kind="artifact", subject_id=artifact_id,
                        kind="host_receipt", valence="success", note="sent, probably",
                        actor="hermes", evidence=[artifact_id])


def test_an_agent_cannot_report_what_the_owner_experienced(store, outcomes):
    with pytest.raises(EvidenceError, match="speaking for them"):
        outcomes.record(subject_kind="lesson", subject_id="x@1", kind="owner_report",
                        valence="success", note="the owner liked it", actor=AGENT)


def test_the_owner_can_report_and_it_counts(store, outcomes):
    outcomes.record(subject_kind="lesson", subject_id="x@1", kind="owner_report",
                    valence="success", note="that fixed it", actor=OWNER)
    assert outcomes.tally("lesson", "x@1")["support"] == 1


def test_a_retracted_report_takes_its_weight_off_again(store, outcomes):
    written = outcomes.record(subject_kind="lesson", subject_id="x@1", kind="owner_report",
                             valence="success", note="that fixed it", actor=OWNER)
    outcomes.retract(written["id"], actor=OWNER, reason="the owner meant a different case")
    assert outcomes.tally("lesson", "x@1")["support"] == 0


def test_only_the_owner_can_take_a_report_off_the_record(store, outcomes):
    written = outcomes.record(subject_kind="lesson", subject_id="x@1", kind="owner_report",
                             valence="success", note="ok", actor=OWNER)
    with pytest.raises(EvidenceError, match="only the owner"):
        outcomes.retract(written["id"], actor=AGENT, reason="inconvenient")


def test_a_receipt_needs_something_to_be_a_receipt_about(store, evidence):
    """No outbox wired means no delivery record to check against."""
    bare = OutcomeLog(store, owner_principal=OWNER)
    with pytest.raises(EvidenceError, match="without a delivery record"):
        bare.record(subject_kind="artifact", subject_id="out_whatever",
                    kind="host_receipt", valence="success", note="the host said so",
                    actor="hermes", evidence=["out_whatever"])


@pytest.mark.parametrize("kind, valence", [
    ("vibes", "success"), ("host_receipt", "probably"), ("owner_report", "great"),
])
def test_an_outcome_that_is_not_a_known_kind_or_valence_is_refused(store, outcomes, kind,
                                                                   valence):
    with pytest.raises(EvidenceError):
        outcomes.record(subject_kind="task", subject_id="t-1", kind=kind,
                        valence=valence, note="x", actor=OWNER)


def test_an_assistant_cannot_demote_a_lesson_it_disagrees_with(store, outcomes, lessons,
                                                              ledger, evidence):
    """Both directions: a claim about a message carries no weight either way."""
    activate(store, lessons, ledger, evidence)
    outcomes.record(subject_kind="lesson", subject_id="chase-invoice@1",
                    kind="assistant_claim", valence="failure",
                    note="the assistant said it did not work", actor=AGENT)
    tally = outcomes.tally("lesson", "chase-invoice@1")
    assert tally["claimed"] == 1 and tally["against"] == 0
    assert lessons.applicable({"source": "gmail"}), \
        "an unverified self-report is not a checked failure"


def test_the_same_report_twice_is_one_report(store, outcomes):
    first = outcomes.record(subject_kind="lesson", subject_id="x@1", kind="owner_report",
                           valence="success", note="that worked", actor=OWNER)
    again = outcomes.record(subject_kind="lesson", subject_id="x@1", kind="owner_report",
                           valence="success", note="that worked", actor=OWNER)
    assert again["recorded"] is False and again["id"] == first["id"]


def test_every_kind_says_what_it_is_evidence_of(store, receipts, stack_outbox):
    """A caller must never have to guess whether a row on file can be relied on."""
    artifact = stack_outbox[1]
    receipts.record(subject_kind="artifact", subject_id=artifact, kind="host_receipt",
                    valence="success", note="the host echoed the digest", actor="hermes",
                    evidence=[artifact])
    claim = receipts.record(subject_kind="artifact", subject_id=artifact,
                            kind="assistant_claim", valence="success", note="it worked",
                            actor=AGENT)
    report = receipts.record(subject_kind="artifact", subject_id=artifact,
                             kind="owner_report", valence="success", note="it worked",
                             actor=OWNER)
    assert claim["counts_as_support"] is False
    assert "statement about a message" in claim["meaning"]
    assert report["counts_as_support"] is True
    assert report["meaning"] == "this is evidence about the world"


def test_an_operator_the_store_does_not_run_is_refused_by_name(store, lessons, evidence):
    with pytest.raises(EvidenceError, match="unknown applicability operator"):
        propose(lessons, evidence,
                applicability={"all": [{"field": "source", "op": "sounds_like",
                                       "value": "gmail"}]})


def test_a_rule_that_nests_for_ever_is_not_a_rule(store, lessons, evidence):
    rule: dict = {"field": "source", "op": "eq", "value": "gmail"}
    for _ in range(9):
        rule = {"all": [rule]}
    with pytest.raises(EvidenceError, match="six levels"):
        propose(lessons, evidence, applicability=rule)


def test_a_lesson_is_replaced_by_a_newer_version_never_rewritten(store, lessons, evidence):
    propose(lessons, evidence, version=3, text="A future revision.")
    with pytest.raises(EvidenceError, match="not newer"):
        propose(lessons, evidence, version=2, text="Going back to an old revision.")


def test_an_un_current_evaluation_cannot_promote(store, lessons, ledger, evidence):
    made = propose(lessons, evidence)
    report = ledger.run(lesson_id="chase-invoice", version=made["version"],
                        cases=list(CASES), promote=False)
    lessons.db.execute("UPDATE lessons SET text='Chase three times instead.' WHERE "
                       "id='chase-invoice'")
    reworded = lessons.get("chase-invoice")
    assert ledger.is_current(report["id"], reworded) is False, \
        "a reworded sentence is a new claim; the old run was about other words"
    with pytest.raises(EvidenceError, match="have moved since it ran"):
        lessons.activate_evaluation(lesson_id="chase-invoice", version=made["version"],
                                    evaluation_id=report["id"])


def test_an_evaluation_about_another_lesson_is_not_about_this_one(store, lessons, ledger,
                                                                 evidence):
    activate(store, lessons, ledger, evidence)
    other = propose(lessons, evidence, lesson_id="different-lesson")
    latest = ledger.latest("chase-invoice", 1)
    assert ledger.is_current(latest.id, lessons.get("different-lesson",
                                                   version=other["version"])) is False


# -- the suite itself --------------------------------------------------------

def test_a_case_left_unrun_makes_the_verdict_incomplete(store, lessons, ledger, evidence):
    made = propose(lessons, evidence)
    opened = ledger.begin(lesson_id="chase-invoice", version=made["version"],
                          cases=[*CASES, {"case_id": "third-one", "role": "regression"}])
    for item in CASES:
        ledger.record(evaluation_id=opened["id"], token=opened["token"],
                      case_id=item["case_id"], outcome="pass")
    report = ledger.finish(evaluation_id=opened["id"], token=opened["token"])
    assert report["verdict"] == "incomplete"
    assert "never run" in report["detail"], \
        "a regression case that was never run is not a case that did not break"


def test_a_case_cannot_appear_twice_in_one_suite(store, lessons, ledger, evidence):
    made = propose(lessons, evidence)
    with pytest.raises(EvidenceError, match="appears twice"):
        ledger.begin(lesson_id="chase-invoice", version=made["version"],
                     cases=[dict(CASES[0]), dict(CASES[0])])


def test_a_case_has_to_have_a_role_the_verdict_understands(store, lessons, ledger,
                                                           evidence):
    made = propose(lessons, evidence)
    with pytest.raises(EvidenceError, match="targeted"):
        ledger.begin(lesson_id="chase-invoice", version=made["version"],
                     cases=[{"case_id": "x", "role": "bonus"}])


def test_a_runner_that_says_something_that_is_not_a_result_is_an_error(store, lessons,
                                                                       evidence):
    made = propose(lessons, evidence)
    subject = EvaluationLedger(store, runner=Runner(default="magnificent"),
                              code_version="hm-0.4", model_version="remote-9b",
                              lessons=lessons)
    lessons.evaluations = subject
    report = subject.run(lesson_id="chase-invoice", version=made["version"],
                         cases=list(CASES))
    assert report["verdict"] == "failed"
    assert all(case["outcome"] == "error" for case in report["cases"])
    assert "not an outcome" in report["cases"][0]["detail"]


def test_an_unknown_case_outcome_cannot_be_filed(store, lessons, ledger, evidence):
    made = propose(lessons, evidence)
    opened = ledger.begin(lesson_id="chase-invoice", version=made["version"],
                          cases=list(CASES))
    with pytest.raises(EvidenceError, match="unknown case outcome"):
        ledger.record(evaluation_id=opened["id"], token=opened["token"],
                      case_id="overdue-invoice", outcome="magnificent")


# -- promotion by evaluation -------------------------------------------------

def test_a_passing_run_promotes_and_says_who_decided(store, lessons, ledger, evidence):
    made = propose(lessons, evidence)
    report = ledger.run(lesson_id="chase-invoice", version=made["version"],
                        cases=list(CASES))
    assert report["verdict"] == "passed" and report["promoted"] is True
    lesson = lessons.get("chase-invoice")
    assert lesson.status == "active"
    assert lesson.created_by == AGENT and lesson.evaluation_id == report["id"]


def test_a_promotion_names_the_program_that_scored_it():
    """Where the whole path fits in the space an actor has, the attribution keeps it."""
    runner = "/srv/hermes/bin/answer.sh"
    assert promotion_actor(runner) == f"evaluation:{runner}"


def a_runner_whose_attribution_is(space: int) -> str:
    """A script path whose full attribution would be exactly *space* characters."""
    ends = "/srv/hermes-memory/releases/" + "/answer.sh"
    padded = space - len("evaluation:") - len(ends)
    return "/srv/hermes-memory/releases/" + "d" * padded + "/answer.sh"


def test_an_attribution_that_exactly_fills_the_space_keeps_the_whole_path():
    runner = a_runner_whose_attribution_is(ACTOR_LIMIT)
    assert len(f"evaluation:{runner}") == ACTOR_LIMIT, runner
    assert promotion_actor(runner) == f"evaluation:{runner}"


def test_one_character_more_than_the_space_is_a_digest_rather_than_a_refusal():
    """The bound is paid here rather than at the write, which only refuses.

    A release directory is a 40-character commit name under a runtime pointer, so the path a
    worker was started from is routinely longer than the attribution column, and the program
    is still identifiable without the directory that held it.
    """
    runner = a_runner_whose_attribution_is(ACTOR_LIMIT + 1)
    assert promotion_actor(runner) == f"evaluation:answer.sh:{digest([runner])[:12]}"
    assert len(promotion_actor(runner)) <= ACTOR_LIMIT


def test_two_releases_holding_one_script_name_do_not_sign_the_same_way():
    """The digest is the whole of the difference between one release and the next."""
    first = ("/home/operator/data/hermes-memory/runtime/releases/"
             "0123456789abcdef0123456789abcdef01234567/hindsight/bin/answer.sh")
    second = first.replace("0123456789abcdef", "9876543210fedcba")
    assert promotion_actor(first) == promotion_actor(first), "one program, one name, every time"
    assert promotion_actor(first) != promotion_actor(second)


def test_a_deep_run_promotes_within_the_space_an_actor_has(store, lessons, evidence):
    """The bound is met on the way to the write, not discovered by the write refusing it.

    An evaluation row may hold a 120-character runner — that is the same column's bound —
    and the promotion's prefix does not fit inside it, so a run from a deep release directory
    has to be signed by the program's name rather than by the path that happened to hold it.
    """
    runner = a_runner_whose_attribution_is(ACTOR_LIMIT + 11)
    assert len(runner) == ACTOR_LIMIT, runner

    class Deep(Runner):
        name = runner

    subject = EvaluationLedger(store, runner=Deep(), code_version="hm-0.4",
                              model_version="remote-9b", lessons=lessons)
    lessons.evaluations = subject
    made = propose(lessons, evidence)
    report = subject.run(lesson_id="chase-invoice", version=made["version"], cases=list(CASES))
    assert report["promoted"] is True
    row = store.db.execute("SELECT decided_by FROM lessons WHERE id='chase-invoice' "
                           "AND status='active'").fetchone()
    assert row["decided_by"] == f"evaluation:answer.sh:{digest([runner])[:12]}"
    assert len(row["decided_by"]) <= ACTOR_LIMIT


def test_a_promoter_is_not_the_proposer(store, lessons, evidence):
    """Self-promotion by an agent is refused even when it also owns the runner."""
    made = propose(lessons, evidence, proposed_by="fixture-suite")
    subject = EvaluationLedger(store, runner=Runner(), code_version="hm-0.4",
                              model_version="remote-9b", lessons=lessons)
    lessons.evaluations = subject
    report = subject.run(lesson_id="chase-invoice", version=made["version"],
                         cases=list(CASES), promote=False)
    assert report["verdict"] == "passed"
    with pytest.raises(EvidenceError, match="promotes itself"):
        lessons.activate_evaluation(lesson_id="chase-invoice", version=made["version"],
                                    evaluation_id=report["id"])


def test_a_regression_breaks_the_verdict(store, lessons, evidence):
    made = propose(lessons, evidence)
    subject = EvaluationLedger(store, runner=Runner(results={"meeting-moved": "fail"}),
                              code_version="hm-0.4", model_version="remote-9b",
                              lessons=lessons)
    lessons.evaluations = subject
    report = subject.run(lesson_id="chase-invoice", version=made["version"],
                         cases=list(CASES))
    assert report["verdict"] == "failed" and "regression case(s) broke" in report["detail"]
    assert lessons.get("chase-invoice").status == "candidate"


def test_a_targeted_case_that_did_not_pass_is_not_a_pass(store, lessons, evidence):
    made = propose(lessons, evidence)
    subject = EvaluationLedger(store, runner=Runner(default="pass",
                                                   results={"overdue-invoice": "error"}),
                              code_version="hm-0.4", model_version="remote-9b",
                              lessons=lessons)
    lessons.evaluations = subject
    report = subject.run(lesson_id="chase-invoice", version=made["version"],
                         cases=list(CASES))
    assert report["verdict"] == "failed"


def test_a_suite_without_a_regression_case_proves_nothing(store, lessons, ledger,
                                                          evidence):
    made = propose(lessons, evidence)
    report = ledger.run(lesson_id="chase-invoice", version=made["version"],
                        cases=[dict(CASES[0])])
    assert report["verdict"] == "incomplete"
    assert lessons.get("chase-invoice").status == "candidate"


def test_a_result_cannot_be_filed_against_a_closed_run(store, lessons, ledger, evidence):
    made = propose(lessons, evidence)
    opened = ledger.begin(lesson_id="chase-invoice", version=made["version"],
                          cases=list(CASES))
    ledger.record(evaluation_id=opened["id"], token=opened["token"],
                  case_id="overdue-invoice", outcome="pass")
    ledger.finish(evaluation_id=opened["id"], token=opened["token"])
    with pytest.raises(EvidenceError, match="no open run"):
        ledger.record(evaluation_id=opened["id"], token=opened["token"],
                      case_id="meeting-moved", outcome="pass")


def test_a_case_outside_the_declared_suite_is_not_a_result(store, lessons, ledger,
                                                           evidence):
    made = propose(lessons, evidence)
    opened = ledger.begin(lesson_id="chase-invoice", version=made["version"],
                          cases=list(CASES))
    with pytest.raises(EvidenceError, match="was not in the suite"):
        ledger.record(evaluation_id=opened["id"], token=opened["token"],
                      case_id="something-i-also-tried", outcome="pass")


def test_someone_elses_run_cannot_be_scored(store, lessons, ledger, evidence):
    made = propose(lessons, evidence)
    opened = ledger.begin(lesson_id="chase-invoice", version=made["version"],
                          cases=list(CASES))
    with pytest.raises(EvidenceError, match="not the token"):
        ledger.record(evaluation_id=opened["id"], token="stolen",
                      case_id="overdue-invoice", outcome="pass")


def test_the_declared_suite_is_what_gets_run(store, lessons, ledger, evidence):
    made = propose(lessons, evidence)
    ledger.run(lesson_id="chase-invoice", version=made["version"], cases=list(CASES))
    assert sorted(ledger.runner.asked) == sorted(item["case_id"] for item in CASES)


def test_the_same_suite_is_not_recorded_twice(store, lessons, ledger, evidence):
    made = propose(lessons, evidence)
    ledger.run(lesson_id="chase-invoice", version=made["version"], cases=list(CASES),
               promote=False)
    with pytest.raises(EvidenceError, match="already recorded"):
        ledger.begin(lesson_id="chase-invoice", version=made["version"], cases=list(CASES))


def test_an_evaluation_is_not_run_against_a_settled_lesson(store, lessons, ledger,
                                                           evidence):
    made = propose(lessons, evidence)
    lessons.activate(lesson_id="chase-invoice", version=made["version"], actor=OWNER,
                     reason="the owner knows")
    with pytest.raises(EvidenceError, match="already settled"):
        ledger.begin(lesson_id="chase-invoice", version=made["version"], cases=list(CASES))


def test_the_owner_may_promote_without_a_run_and_it_is_attributed(store, lessons,
                                                                  evidence):
    made = propose(lessons, evidence)
    lessons.activate(lesson_id="chase-invoice", version=made["version"], actor=OWNER,
                     reason="my call")
    lesson = lessons.get("chase-invoice")
    assert lesson.status == "active"
    assert lessons.applicable({"source": "gmail"})[0].id == "chase-invoice"


def test_an_agent_cannot_promote_its_own_lesson(store, lessons, evidence):
    made = propose(lessons, evidence)
    with pytest.raises(EvidenceError, match="only the owner"):
        lessons.activate(lesson_id="chase-invoice", version=made["version"], actor=AGENT,
                         reason="it seems right")


# -- staleness ---------------------------------------------------------------

def test_a_code_upgrade_says_so_rather_than_lying_about_the_lesson(store, lessons,
                                                                  ledger, evidence):
    activate(store, lessons, ledger, evidence)
    assert lessons.applicable({"source": "gmail"})
    upgraded = EvaluationLedger(store, runner=Runner(), code_version="hm-0.5",
                               model_version="remote-9b", lessons=lessons)
    lessons.evaluations = upgraded
    assert lessons.applicable({"source": "gmail"}) == [], \
        "the old run was not a run of this code"
    assert lessons.needs_review()[0]["reason"] == "evaluation stale"


def test_a_reworded_lesson_is_not_covered_by_the_old_verdict(store, lessons, ledger,
                                                            evidence):
    activate(store, lessons, ledger, evidence)
    second = propose(lessons, evidence, version=2,
                     text="Chase an overdue invoice three times before phoning.")
    lessons.db.execute("UPDATE lessons SET status='active', evaluation_id=("
                       "SELECT id FROM evaluations LIMIT 1) WHERE id='chase-invoice' AND "
                       "version=2")
    assert lessons.applicable({"source": "gmail"}) == []


def test_an_invalidated_evaluation_stops_licensing(store, lessons, ledger, evidence):
    made = propose(lessons, evidence)
    report = ledger.run(lesson_id="chase-invoice", version=made["version"],
                        cases=list(CASES))
    assert lessons.applicable({"source": "gmail"})
    ledger.invalidate(report["id"], reason="the fixtures were edited afterwards")
    assert lessons.applicable({"source": "gmail"}) == []


def test_a_lesson_is_retrieved_only_once_in_its_newest_version(store, lessons, ledger,
                                                              evidence):
    activate(store, lessons, ledger, evidence)
    propose(lessons, evidence, version=2, text="Chase twice, then phone in the morning.")
    lessons.activate(lesson_id="chase-invoice", version=2, actor=OWNER, reason="mine")
    found = lessons.applicable({"source": "gmail"})
    assert [item.version for item in found] == [2], \
        "two active revisions would both be taught, the older one undoing the newer"
    assert lessons.get("chase-invoice", version=1).status == "retracted"


# -- retraction and withdrawal -----------------------------------------------

def test_retraction_teaches_nothing_from_then_on(store, lessons, ledger, evidence):
    activate(store, lessons, ledger, evidence)
    lessons.retract(lesson_id="chase-invoice", actor=OWNER, reason="wrong for this client")
    assert lessons.applicable({"source": "gmail"}) == []
    assert lessons.get("chase-invoice").status == "retracted"


def test_the_proposing_agent_cannot_retract_on_the_owners_behalf(store, lessons,
                                                                 ledger, evidence):
    activate(store, lessons, ledger, evidence)
    with pytest.raises(EvidenceError, match="never by the agent"):
        lessons.retract(lesson_id="chase-invoice", actor=AGENT, reason="inconvenient")


def test_the_store_itself_may_stop_trusting_a_lesson(store, lessons, ledger, evidence):
    activate(store, lessons, ledger, evidence)
    lessons.retract(lesson_id="chase-invoice", actor="system",
                    reason="its evidence was forgotten")
    assert lessons.applicable({"source": "gmail"}) == []


def test_a_forgotten_lesson_loses_its_support_without_being_deleted(store, lessons,
                                                                    ledger, evidence):
    activate(store, lessons, ledger, evidence)
    store.hide(evidence, reason="owner erased it", actor=OWNER)
    assert lessons.applicable({"source": "gmail"}) == []
    assert lessons.get("chase-invoice").status == "active", \
        "the row stays; only its claim to be believed goes away"
    assert lessons.needs_review()[0]["reason"] == "evidence gone"


def test_checked_failures_outrank_a_lessons_own_confirmation(store, lessons, ledger,
                                                            evidence):
    activate(store, lessons, ledger, evidence)
    for index in range(2):
        lessons.record_contradiction(lesson_id="chase-invoice", version=1,
                                    note=f"it was wrong the {index + 1} time",
                                    actor=OWNER)
    assert lessons.applicable({"source": "gmail"}) == []
    assert "2 checked failure" in lessons.needs_review()[0]["reason"]


def test_an_assistants_own_defence_of_a_lesson_does_not_outweigh_a_failure(store, lessons,
                                                                          ledger,
                                                                          evidence):
    activate(store, lessons, ledger, evidence)
    for index in range(5):
        lessons.record_confirmation(lesson_id="chase-invoice", version=1,
                                   note=f"the assistant said it worked {index}",
                                   actor=AGENT)
    lessons.record_contradiction(lesson_id="chase-invoice", version=1,
                                note="the owner says it was wrong", actor=OWNER)
    assert lessons.applicable({"source": "gmail"}) == []


# -- applicability matching --------------------------------------------------

def test_a_lesson_does_not_apply_to_a_task_it_says_nothing_about(store, lessons, ledger,
                                                                evidence):
    activate(store, lessons, ledger, evidence)
    assert lessons.applicable({"source": "calendar"}) == []
    assert lessons.applicable({"topic": "billing"}) == []


def test_a_combined_rule_needs_every_part(store, lessons, ledger, evidence):
    made = propose(lessons, evidence, lesson_id="billing-only", applicability=INVOICE_RULE)
    ledger.run(lesson_id="billing-only", version=made["version"], cases=list(CASES))
    assert lessons.applicable({"source": "gmail", "topic": "invoice"})
    assert lessons.applicable({"source": "gmail"}) == []


def test_negation_and_sets_are_evaluated_not_guessed():
    assert match({"not": RULE}, {"source": "calendar"})[0] is True
    assert match({"not": RULE}, {"source": "gmail"})[0] is False
    assert match({"field": "tool", "op": "all", "value": ["a", "b"]},
                 {"tool": ["b", "a", "c"]})[0] is True
    assert match({"field": "tool", "op": "all", "value": ["a", "b"]},
                 {"tool": ["a"]})[0] is False
    ok, why = match({"field": "tool", "op": "eq", "value": "x"}, {})
    assert not ok and "was not part of this task" in why


def test_a_missing_field_is_an_absence_not_a_false(store, lessons, ledger, evidence):
    activate(store, lessons, ledger, evidence)
    found = lessons.applicable({"source": "gmail"})
    assert found and "all conditions held" in found[0].why


# -- wiring ------------------------------------------------------------------

@pytest.fixture()
def stack_outbox(store):
    """One confirmed artifact, so a host receipt has something real to point at."""
    policy = AttentionPolicy(store, owner_principal=OWNER)
    policy.configure(actor=OWNER, timezone_name="UTC", shadow=False,
                     max_immediate_per_day=10, cooldown_minutes=0)
    outbox = Outbox(store, policy=policy, owner_principal=OWNER, clock=lambda: EPOCH)
    decision = policy.decide(topic="general", at=MORNING)
    # Ordered by dependency: a goal, then its event, then the intention that event
    # handed over, then the decision about that intention.
    store.db.execute("BEGIN IMMEDIATE")
    try:
        store.db.execute(
            "INSERT INTO goals(id, title, statement, status, revision, timezone, "
            "created_by, created_kind, created_at, updated_at) "
            "VALUES('goal-1','Chase','x','active',1,'UTC',?,'owner',?,?)",
            (OWNER, MORNING, MORNING))
        store.db.execute(
            "INSERT INTO due_events(id, goal_id, revision, fire_at, reason, timezone, "
            "precision, state, created_at) "
            "VALUES('due_smoke','goal-1',1,?,'due','UTC','second','pending',?)",
            (MORNING, MORNING))
        store.db.execute(
            "INSERT INTO decision_intents(id, event_id, goal_id, revision, kind, "
            "policy_version, state, created_at, updated_at) "
            "VALUES('dec_smoke1','due_smoke','goal-1',1,'notify_owner',?,'prepared',?,?)",
            (POLICY_VERSION, MORNING, MORNING))
        store.db.execute(
            "INSERT INTO proactive_decisions(id, intent_id, goal_id, revision, topic, "
            "action, reason, policy_version, shadow, model_used, decided_at) "
            "VALUES('dec_p1','dec_smoke1','goal-1',1,'general',?,'',?,0,0,?)",
            (decision.action, POLICY_VERSION, MORNING))
        store.db.execute("COMMIT")
    except BaseException:
        store.db.execute("ROLLBACK")
        raise
    prepared = outbox.prepare(decision_id="dec_p1", kind="notify_owner", topic="general",
                              payload="The invoice is overdue.", evidence=[])
    outbox.db.execute("UPDATE outbox SET state='confirmed' WHERE id=?", (prepared["id"],))
    return outbox, prepared["id"]


def test_the_packet_renders_an_active_lesson_with_its_reason(store, lessons, ledger,
                                                            evidence):
    from hermes_memory.context.broker import ContextBroker
    activate(store, lessons, ledger, evidence)
    applicable = lessons.applicable({"source": "gmail", "topic": "billing"})
    broker = ContextBroker(store, client=None)
    packet = broker.assemble("invoice", lessons=[item.as_dict() for item in applicable])
    assert packet.lessons and "Chase an overdue invoice" in packet.render()
    assert json.dumps(packet.lessons[0]["text"])


def test_a_lesson_is_taught_only_to_whose_evidence_it_stands_on(store, outcomes, evidence):
    """`identities` is what makes `applicable`'s scoping real rather than decorative.

    The lesson is not re-decided here: the same rule that would show or withhold the
    records it cites decides who may be told what was learned from them, and an owner
    confirmation widens that without anyone editing the lesson.
    """
    from hermes_memory.storage.identity import IdentityStore

    identity = IdentityStore(store, owner_principal=OWNER)
    mine = identity.account("email", "priya@example.com")
    theirs = identity.account("email", "landlord@example.org")
    cited = str(store.commit(envelope(source_id="scoped-1", text="Chase this invoice.",
                                      metadata={"account_ids": [mine]}))["id"])
    habits = LessonStore(store, outcomes=outcomes, identities=identity,
                         owner_principal=OWNER)
    made = propose(habits, cited)
    habits.activate(lesson_id="chase-invoice", version=made["version"], actor=OWNER,
                    reason="the owner's own promotion")

    assert habits.applicable({"source": "gmail"}, account_id=mine)
    assert habits.applicable({"source": "gmail"}, account_id=theirs) == []

    joined = identity.propose(account_a=mine, account_b=theirs,
                              rule="email-thread-participant", basis="one thread",
                              evidence=[cited], proposed_by=AGENT)
    earned = identity.confirm(candidate_id=joined["candidate_id"], actor=OWNER,
                              reason="one person")
    assert habits.applicable({"source": "gmail"}, account_id=theirs), \
        "the owner widened the group, so the lesson travels"

    identity.revoke(edge_id=earned["edge_id"], actor=OWNER, reason="not after all")
    assert habits.applicable({"source": "gmail"}, account_id=theirs) == [], \
        "and taking the join back takes the lesson back with it"


def test_a_lesson_citing_a_revision_is_still_scoped_by_the_record(store, outcomes):
    """A citation may name the revision it was learned at; that is not an escape hatch.

    Reading the scope out of the store means the id has to be resolved the same way
    every other reader resolves it, or a lesson that cites `rec_x@3` would claim
    nobody and be taught anywhere.
    """
    from hermes_memory.storage.identity import IdentityStore

    identity = IdentityStore(store, owner_principal=OWNER)
    mine = identity.account("email", "priya@example.com")
    theirs = identity.account("email", "landlord@example.org")
    cited = str(store.commit(envelope(source_id="scoped-2", text="Chase this invoice.",
                                      metadata={"account_ids": [mine]}))["id"])
    habits = LessonStore(store, outcomes=outcomes, identities=identity,
                         owner_principal=OWNER)
    made = propose(habits, f"{cited}@1")
    habits.activate(lesson_id="chase-invoice", version=made["version"], actor=OWNER,
                    reason="the owner's own promotion")

    assert habits.applicable({"source": "gmail"}, account_id=mine)
    assert habits.applicable({"source": "gmail"}, account_id=theirs) == []


# -- a verdict that stopped meaning anything -----------------------------------

def test_a_promotion_stops_teaching_when_its_evaluation_is_withdrawn(store, lessons,
                                                                     ledger, evidence):
    """`invalidate` is a stored fact, and a read that has no versions to compare still
    has to honour it. Otherwise a lesson licensed by a run that turned out to be about
    the wrong fixture keeps being taught until somebody with the current code versions
    happens to look."""
    made = propose(lessons, evidence)
    report = ledger.run(lesson_id="chase-invoice", version=made["version"],
                        cases=list(CASES))
    assert lessons.applicable({"source": "gmail"}), "the run earned it"
    ledger.invalidate(report["id"], reason="the regression case was the same as the target")
    assert lessons.applicable({"source": "gmail"}) == []
    assert lessons.needs_review()[0]["reason"] == "evaluation withdrawn"


def test_a_reading_without_version_knowledge_still_drops_a_withdrawn_verdict(store, lessons,
                                                                            ledger, evidence):
    made = propose(lessons, evidence)
    report = ledger.run(lesson_id="chase-invoice", version=made["version"],
                        cases=list(CASES))
    lessons.evaluations = None
    ledger.invalidate(report["id"], reason="superseded by a later run")
    assert lessons.applicable({"source": "gmail"}) == []
    assert lessons.needs_review()[0]["reason"] == "evaluation withdrawn"


def test_a_lesson_cannot_cite_an_evaluation_that_never_ran(store, lessons, evidence):
    """The schema, not the code, refuses it: a promotion cannot name a run as its license
    unless that run is a row that happened."""
    made = propose(lessons, evidence)
    lessons.activate(lesson_id="chase-invoice", version=made["version"], actor=OWNER,
                     reason="the owner's own promotion")
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        lessons.db.execute("UPDATE lessons SET evaluation_id='eval_fabricated' WHERE id=?",
                           ("chase-invoice",))
    assert lessons.applicable({"source": "gmail"}), \
        "the owner's promotion stands on its own, without any run behind it"


def test_a_ledger_with_no_authorized_runner_cannot_open_a_run(store, lessons, evidence):
    """The evaluation machinery is separately authorised, and `None` is that absence."""
    made = propose(lessons, evidence)
    reading = EvaluationLedger(store, runner=None, code_version="hm-0.4",
                              model_version="remote-9b", lessons=lessons)
    with pytest.raises(EvidenceError, match="cannot start one"):
        reading.begin(lesson_id="chase-invoice", version=made["version"], cases=list(CASES))
    assert store.db.execute("SELECT count(*) FROM evaluations").fetchone()[0] == 0
