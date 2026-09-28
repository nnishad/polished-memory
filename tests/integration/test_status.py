"""C14 status: every stage says what it observed, and a fault cannot hide in a total.

The rows are seeded rather than built through the pipelines: this suite is about
what a report says about a store in a given state, and going through nine components
to place one row would test those components instead of the reading.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_memory.operations.status import (CONFIGURED, DEGRADED, DISABLED, OPERATIONAL,
                                             PAUSED, REPORTED_STAGES, UNCONFIGURED,
                                             StageReport, StatusReporter)
from hermes_memory.processing.jobs import JobQueue
from hermes_memory.processing.resource_gate import ResourceGate
from hermes_memory.processing.routes import Route
from hermes_memory.storage.evidence import EvidenceError

from conftest import envelope

FUTURE = "2099-01-01T00:00:00+00:00"
PAST = "2000-01-01T00:00:00+00:00"
REMOVED = "1999-01-01T00:00:00+00:00"
RETAIN = Route("retain", "remote-9b", "chat", "http://127.0.0.1:8080/v1", "cred",
               "freshness", 2048)


def settings(**overrides):
    # ``home`` is in the surface because the resource gate is read from the instance
    # admission ledger beside it, not from whichever profile asked first.
    base = {"capture_only": False, "hindsight_url": "http://127.0.0.1:8080/v1",
            "owner_principal": "owner", "home": Path(tempfile.mkdtemp(prefix="hm-"))}
    base.update(overrides)
    return SimpleNamespace(**base)


def insert(store, table, **columns):
    names = ", ".join(columns)
    marks = ", ".join("?" * len(columns))
    store.db.execute(f"INSERT INTO {table}({names}) VALUES({marks})",
                     list(columns.values()))
    return columns


def recorded(store, **overrides) -> str:
    return store.commit(envelope(**overrides))["id"]


def connector(store, sync, source="gmail", *, coverage="current") -> dict:
    state = sync.register(source, policy_version="local-only")
    store.db.execute("UPDATE connectors SET coverage_state=? WHERE source=?",
                     (coverage, source))
    return state


def a_goal(store, *, goal_id="goal-1", status="active", **overrides):
    columns = dict(id=goal_id, title="Call the dentist", statement="Call the dentist",
                   status=status, timezone="UTC", created_by="owner",
                   created_kind="owner", created_at=PAST, updated_at=PAST)
    columns.update(overrides)
    return insert(store, "goals", **columns)


def a_due_event(store, *, event_id="due-1", goal_id="goal-1", fire_at=FUTURE,
                state="pending", **overrides):
    columns = dict(id=event_id, goal_id=goal_id, revision=1, fire_at=fire_at,
                   reason="scheduled", timezone="UTC", precision="minute", state=state,
                   created_at=PAST)
    columns.update(overrides)
    return insert(store, "due_events", **columns)


def an_intent(store, *, intent_id="intent-1", event_id="due-1", goal_id="goal-1",
              state="prepared", **overrides):
    columns = dict(id=intent_id, event_id=event_id, goal_id=goal_id, revision=1,
                   kind="notify_owner", policy_version="policy-v1", state=state,
                   created_at=PAST, updated_at=PAST)
    columns.update(overrides)
    return insert(store, "decision_intents", **columns)


def a_decision(store, *, decision_id="dec-1", intent_id="intent-1", goal_id="goal-1",
               action="notify_owner", **overrides):
    columns = dict(id=decision_id, intent_id=intent_id, goal_id=goal_id, revision=1,
                   topic="health", action=action, reason="policy", policy_version="v1",
                   decided_at=PAST)
    columns.update(overrides)
    return insert(store, "proactive_decisions", **columns)


def an_artifact(store, *, artifact_id="ob-1", decision_id="dec-1", kind="notify_owner",
                state="prepared", **overrides):
    columns = dict(id=artifact_id, decision_id=decision_id, kind=kind, topic="health",
                   recipient="owner", payload="Call the dentist", payload_digest="d" * 16,
                   policy_version="v1", state=state, created_at=PAST, updated_at=PAST)
    columns.update(overrides)
    return insert(store, "outbox", **columns)


def the_whole_handoff(store):
    """goal -> due event -> intent -> decision -> artifact, as the pipeline writes it."""
    a_goal(store)
    a_due_event(store)
    an_intent(store)
    a_decision(store)
    return an_artifact(store)


def a_job(store, *, job_id="job-1", state="queued", created_at=None, **overrides):
    stamp = created_at or datetime.now(timezone.utc).isoformat()
    columns = dict(id=job_id, kind="retain", state=state, priority=3, resource="remote-9b",
                   route="retain", epoch=1, processor_fingerprint="extractor-v3",
                   inputs="[]", input_revision="1", max_attempts=3, token_budget=1000,
                   created_at=stamp, updated_at=stamp)
    columns.update(overrides)
    return insert(store, "processing_jobs", **columns)


# -- an installation that has never run --------------------------------------

def test_nothing_configured_is_reported_as_unconfigured_not_broken(store):
    report = StatusReporter(store).report()
    assert report["overall"] == UNCONFIGURED
    assert set(report["states"]) == set(REPORTED_STAGES)
    assert all(state == UNCONFIGURED for state in report["states"].values())
    assert report["notes"] == []


def test_an_unknown_stage_is_refused_by_naming_the_real_ones(store):
    with pytest.raises(EvidenceError, match="reportable stages are"):
        StatusReporter(store).stage("vibes")


def test_a_state_that_is_not_in_the_vocabulary_is_refused():
    with pytest.raises(EvidenceError, match="unknown status state"):
        StageReport("capture", "fine", "everything is fine")


def test_reading_status_never_writes(store, sync):
    reporter = StatusReporter(store)
    connector(store, sync)
    recorded(store)

    def footprint():
        tables = [row[0] for row in store.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {table: store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in tables}

    reporter.report()
    before = footprint()
    audit_rows = store.db.execute("SELECT count(*) FROM audit").fetchone()[0]
    reporter.report()
    reporter.stage("capture")
    assert footprint() == before
    assert store.db.execute("SELECT count(*) FROM audit").fetchone()[0] == audit_rows


def test_the_report_is_plain_serialisable_data(store, sync):
    connector(store, sync)
    recorded(store)
    json.dumps(StatusReporter(store).report(), default=str)


# -- capture -----------------------------------------------------------------

def test_a_source_that_is_current_makes_capture_operational(store, sync):
    connector(store, sync, coverage="current")
    assert StatusReporter(store).capture().state == OPERATIONAL


@pytest.mark.parametrize("coverage", ["unknown", "partial"])
def test_a_connector_that_has_not_finished_or_not_started_is_not_a_fault(store, sync,
                                                                         coverage):
    report = StatusReporter(store)
    connector(store, sync, coverage=coverage)
    assert report.capture().state == OPERATIONAL


@pytest.mark.parametrize("coverage", ["stale", "unreachable", "revoked"])
def test_coverage_that_went_bad_is_named_rather_than_counted_away(store, sync, coverage):
    reporter = StatusReporter(store)
    connector(store, sync, source="gmail", coverage=coverage)
    connector(store, sync, source="whatsapp", coverage="current")
    report = reporter.capture()
    assert report.state == DEGRADED
    assert report.evidence["unhealthy"] == ["gmail"]


def test_what_the_source_could_not_give_us_keeps_capture_from_looking_clean(store, sync):
    connector(store, sync)
    insert(store, "source_gaps", source="gmail", generation=1, ref="chat.txt#7",
           reason="read failed", first_seen_at=PAST, last_seen_at=PAST)
    report = StatusReporter(store).capture()
    assert report.state == DEGRADED
    assert report.evidence["open_gaps"] == 1
    assert report.evidence["sources"][0]["open_gaps"] == 1


def test_a_gap_the_next_arrival_closed_is_not_still_a_debt(store, sync):
    connector(store, sync)
    insert(store, "source_gaps", source="gmail", generation=1, ref="chat.txt#7",
           reason="read failed", first_seen_at=PAST, last_seen_at=PAST, cleared_at=REMOVED)
    assert StatusReporter(store).capture().state == OPERATIONAL


def test_pausing_every_source_is_reported_as_a_stop_somebody_asked_for(store, sync):
    connector(store, sync, source="gmail")
    connector(store, sync, source="whatsapp")
    for source in ("gmail", "whatsapp"):
        sync.pause_capture(source, actor="owner", reason="privacy review",
                           policy_version="v1")
    report = StatusReporter(store).capture()
    assert report.state == PAUSED
    assert report.evidence["sources"][0]["pause_reason"] == "privacy review"


def test_pausing_one_source_of_two_is_not_reported_as_a_stopped_stage(store, sync):
    connector(store, sync, source="gmail")
    connector(store, sync, source="whatsapp")
    sync.pause_capture("gmail", actor="owner", reason="one at a time", policy_version="v1")
    report = StatusReporter(store).capture()
    assert report.state == OPERATIONAL
    paused = {row["source"]: row["paused"] for row in report.evidence["sources"]}
    assert paused == {"gmail": True, "whatsapp": False}


def test_a_formation_pause_does_not_claim_the_ingestion_stage_is_stopped(store, sync):
    connector(store, sync)
    sync.pause("gmail", actor="owner", reason="thinking", policy_version="v1",
               stages=("formation",))
    assert StatusReporter(store).capture().state == OPERATIONAL


# -- the local index ---------------------------------------------------------

def test_a_record_arriving_is_immediately_searchable(store):
    recorded(store)
    report = StatusReporter(store).raw_indexing()
    assert report.state == OPERATIONAL
    assert report.evidence == {"live": 1, "hidden": 0, "visible": 1, "searchable": 1,
                               "unindexed": 0, "index_leaks": 0}


def test_evidence_the_owner_hidden_is_absent_from_search_without_being_a_fault(store):
    record = recorded(store)
    store.hide(record, reason="not for retrieval", actor="owner")
    report = StatusReporter(store).raw_indexing()
    assert report.state == OPERATIONAL
    assert (report.evidence["live"], report.evidence["visible"],
            report.evidence["searchable"]) == (1, 0, 0)


def test_committed_evidence_with_no_index_entry_is_a_broken_index(store):
    record = recorded(store)
    store.db.execute("DELETE FROM record_fts WHERE id=?", (record,))
    report = StatusReporter(store).raw_indexing()
    assert report.state == DEGRADED
    assert report.evidence["unindexed"] == 1


def test_an_index_entry_for_forgotten_evidence_is_the_dangerous_direction(store):
    record = recorded(store)
    store.db.execute("UPDATE records SET deleted=1 WHERE id=?", (record,))
    report = StatusReporter(store).raw_indexing()
    assert report.state == DEGRADED
    assert report.evidence["index_leaks"] == 1


def test_an_orphaned_index_entry_pointing_at_nothing_is_counted(store):
    recorded(store)
    store.db.execute("INSERT INTO record_fts(id, text) VALUES('gone', 'nothing')")
    report = StatusReporter(store).raw_indexing()
    assert report.state == DEGRADED
    assert report.evidence["index_leaks"] == 1
    assert report.evidence["unindexed"] == 0, "one leak must not be read as one gap"


# -- formation ---------------------------------------------------------------

def test_capture_only_disables_formation_rather_than_calling_it_broken(store):
    report = StatusReporter(store, settings=settings(capture_only=True)).observations()
    assert report.state == DISABLED
    assert report.evidence["reason"] == "capture-only"


def test_a_quarantined_job_is_a_stuck_queue_not_a_busy_one(store):
    a_job(store, state="quarantined")
    report = StatusReporter(store).observations()
    assert report.state == DEGRADED
    assert report.evidence["stuck"] == {"quarantined": 1}


def test_an_uncertain_job_stays_a_debt_until_someone_answers_for_it(store):
    a_job(store, state="uncertain")
    assert StatusReporter(store).observations().state == DEGRADED


def test_work_nobody_has_claimed_yet_is_configured_rather_than_running(store):
    a_job(store, state="queued")
    report = StatusReporter(store).observations()
    assert report.state == CONFIGURED, "a place in a line is not somebody standing in it"
    assert report.evidence["draining"] is False


def test_a_leased_job_is_the_formation_stage_actually_running(store):
    a_job(store, state="leased")
    report = StatusReporter(store).observations()
    assert report.state == OPERATIONAL and report.evidence["draining"] is True
    assert "hermes-memory form" not in report.detail, (
        "work is being done; the note would say the opposite")


def test_a_waiting_job_and_a_leased_one_is_still_a_running_stage(store):
    a_job(store, job_id="job-waiting", state="queued")
    a_job(store, job_id="job-in-flight", state="submitting")
    report = StatusReporter(store).observations()
    assert report.state == OPERATIONAL and report.evidence["draining"] is True


def test_a_finished_queue_with_nothing_in_it_has_never_run(store):
    a_job(store, state="succeeded")
    assert StatusReporter(store).observations().state == UNCONFIGURED


def test_formation_paused_everywhere_is_reported_as_paused(store, sync):
    connector(store, sync)
    sync.pause("gmail", actor="owner", reason="not now", policy_version="v1",
               stages=("formation",))
    assert StatusReporter(store).observations().state == PAUSED


def test_work_waiting_with_no_daemon_says_who_moves_it(store):
    a_job(store, state="queued")
    report = StatusReporter(store).observations()
    assert report.evidence["formation_unattended"] is False
    assert "hermes-memory form" in report.detail, (
        "a busy queue is not a running stage unless something is running it")


def test_a_hold_on_the_shared_models_pauses_formation_for_the_whole_machine(store):
    ResourceGate(store).pause(actor="owner", reason="the models are being moved")
    report = StatusReporter(store).observations()
    assert report.state == PAUSED and report.evidence["instance_hold"] is True
    assert "holding inference" in report.detail
    assert report.evidence["instance_hold_by"] == "owner"
    assert report.evidence["instance_hold_reason"] == "the models are being moved", \
        "a hold that stops every profile says who stopped it and why, in the reading"


def test_an_owners_hold_outranks_a_queue_that_looks_busy(store):
    a_job(store, state="queued")
    ResourceGate(store).pause(actor="owner", reason="held")
    assert StatusReporter(store).observations().state == PAUSED


def test_how_far_behind_a_consumer_is_is_counted_in_changes_not_guessed(store, sync):
    connector(store, sync)
    recorded(store)
    store.commit(envelope(source="gmail", source_id="msg-2", revision="1",
                          text="A second message."))
    store.db.execute("INSERT INTO consumer_checkpoints(consumer, seq, updated_at) "
                     "VALUES('derived-index', 1, ?)", (PAST,))
    journal = StatusReporter(store).observations().evidence["journal"]
    assert journal["head"] == 2
    assert journal["consumers"] == [{"consumer": "derived-index", "seq": 1, "behind": 1,
                                     "updated_at": PAST}]


# -- summaries and their citations -------------------------------------------

def test_a_published_summary_is_the_stage_working(store):
    _summary(store)
    report = StatusReporter(store).summaries()
    assert report.state == OPERATIONAL
    assert report.evidence["by_kind"] == {"day": {"published": 1}}


def test_a_withdrawn_summary_alone_is_not_fault(store):
    _summary(store, status="withdrawn", withdrawn_at=REMOVED)
    assert StatusReporter(store).summaries().state == OPERATIONAL


def test_a_summary_citing_evidence_that_is_gone_stops_being_supported(store):
    record = _summary(store)
    store.db.execute("UPDATE records SET deleted=1 WHERE id=?", (record,))
    report = StatusReporter(store).summaries()
    assert report.state == DEGRADED
    assert report.evidence["unsupported"]["artifacts"] == 1


def test_a_summary_citing_evidence_the_owner_hid_is_not_supported_either(store):
    record = _summary(store)
    store.hide(record, reason="private", actor="owner")
    assert StatusReporter(store).summaries().state == DEGRADED


def test_a_failed_refresh_attempt_is_reported_rather_than_retried_silently(store):
    _summary(store)
    insert(store, "summary_refreshes", scope="day:2026-09-25", kind="day",
           through_at=PAST, requested_at=PAST, state="failed", detail="budget")
    assert StatusReporter(store).summaries().state == DEGRADED


def _summary(store, **overrides):
    record = recorded(store)
    columns = dict(id="sum-1", scope="day:2026-09-25", kind="day", title="A day",
                   body="Things happened", status="published",
                   processor_fingerprint="summariser-v1", epoch=1, budget_tokens=100,
                   created_at=PAST, published_at=PAST)
    columns.update(overrides)
    insert(store, "summaries", **columns)
    insert(store, "derived_citations", artifact_id="sum-1", kind="summary",
           coverage="full", record_id=record, added_at=PAST)
    return record


# -- prospective memory ------------------------------------------------------

def test_a_goal_with_its_clock_kept_is_operational(store):
    a_goal(store)
    a_due_event(store)
    report = StatusReporter(store).goals()
    assert report.state == OPERATIONAL
    assert report.evidence["overdue"] == 0


def test_a_due_event_that_arrived_and_was_never_fired_is_not_quiet(store):
    a_goal(store)
    a_due_event(store, fire_at=PAST)
    report = StatusReporter(store).goals()
    assert report.state == DEGRADED
    assert report.evidence["overdue"] == 1


def test_a_handoff_nobody_can_answer_for_is_a_debt(store):
    a_goal(store)
    a_due_event(store, state="uncertain")
    assert StatusReporter(store).goals().state == DEGRADED


def test_a_claim_whose_holder_stopped_answering_is_found(store):
    from time import time
    a_goal(store)
    a_due_event(store, state="claimed", claim_until=time() - 1)
    report = StatusReporter(store).goals()
    assert report.state == DEGRADED
    assert report.evidence["unresolved"]["expired_claims"] == 1


def test_a_claim_still_held_by_a_live_worker_is_ordinary_work(store):
    from time import time
    a_goal(store)
    a_due_event(store, state="claimed", claim_until=time() + 60)
    assert StatusReporter(store).goals().state == OPERATIONAL


# -- the analyst and the transport -------------------------------------------

def test_analysis_is_disabled_not_broken_without_an_inference_route(store):
    report = StatusReporter(store, settings=settings(capture_only=True)).analysis()
    assert report.state == DISABLED


def test_the_analysts_shadowed_decisions_are_counted_apart(store):
    _handoff(store, decision=dict(shadow=1, model_used=0))
    report = StatusReporter(store).analysis()
    assert report.state == OPERATIONAL
    assert (report.evidence["shadowed"], report.evidence["model_used"]) == (1, 0)


def test_a_decision_with_a_model_in_the_loop_says_so(store):
    _handoff(store, decision=dict(shadow=0, model_used=1))
    assert StatusReporter(store).analysis().evidence["model_used"] == 1


def _handoff(store, decision=None):
    the_whole_handoff(store)
    if decision:
        store.db.execute("UPDATE proactive_decisions SET shadow=?, model_used=?",
                         (decision["shadow"], decision["model_used"]))


def test_an_artifact_waiting_on_a_transport_is_reported_as_work_in_flight(store):
    the_whole_handoff(store)
    report = StatusReporter(store).delivery()
    assert report.state == OPERATIONAL
    assert report.evidence["in_flight"] == 1
    assert any("has not claimed" in note for note in StatusReporter(store).report()
               ["notes"])


def test_an_artifact_sent_without_a_proof_stays_unresolved(store):
    the_whole_handoff(store)
    store.db.execute("UPDATE outbox SET state='accepted_unverified'")
    report = StatusReporter(store).delivery()
    assert report.state == DEGRADED
    assert report.evidence["unproven"] == 1


def test_an_artifact_that_aged_out_unsent_is_a_transport_that_stopped_coming(store):
    the_whole_handoff(store)
    store.db.execute("UPDATE outbox SET expires_at=?", (PAST,))
    report = StatusReporter(store).delivery()
    assert report.state == DEGRADED
    assert report.evidence["past_expiry"] == 1


def test_a_confirmed_artifact_is_the_pipeline_finished(store):
    the_whole_handoff(store)
    store.db.execute("UPDATE outbox SET state='confirmed', delivered_at=?", (PAST,))
    report = StatusReporter(store).delivery()
    assert report.state == OPERATIONAL
    assert report.evidence["in_flight"] == 0


def test_proactivity_paused_everywhere_stops_delivery_without_calling_it_stuck(store,
                                                                              sync):
    connector(store, sync)
    a_goal(store)
    a_due_event(store)
    an_intent(store)
    a_decision(store)
    sync.pause("gmail", actor="owner", reason="quiet week", policy_version="v1",
               stages=("proactivity",))
    assert StatusReporter(store).delivery().state == PAUSED
    assert StatusReporter(store).analysis().state == PAUSED


def test_a_store_that_never_produced_an_artifact_is_unconfigured(store):
    assert StatusReporter(store).delivery().state == UNCONFIGURED


# -- the derived backend -----------------------------------------------------

def test_no_backend_configured_leaves_the_local_channels_answerable(store):
    report = StatusReporter(store, settings=settings(hindsight_url=None)).backend()
    assert report.state == UNCONFIGURED
    assert "local channels" in report.detail


def test_a_configured_backend_with_nothing_to_say_is_configured_not_working(store):
    reporter = StatusReporter(store, settings=settings())
    assert reporter.backend().state == CONFIGURED
    # The one-word summary must not claim a working system on a configured-only one.
    assert reporter.report()["overall"] == CONFIGURED


def test_verified_documents_are_the_backend_working(store):
    _document(store, state="verified")
    report = StatusReporter(store, settings=settings()).backend()
    assert report.state == OPERATIONAL
    assert report.evidence["banks"] == ["hermes"]


def test_a_document_the_backend_refused_is_reported(store):
    _document(store, state="failed")
    report = StatusReporter(store, settings=settings()).backend()
    assert report.state == DEGRADED
    assert "not probed" in report.evidence["reachable"]


def test_a_mapping_left_behind_by_a_reset_can_never_be_acknowledged(store):
    _document(store, state="queued", desired_epoch=1)
    store.bump_epoch(reason="reset", actor="owner")
    report = StatusReporter(store, settings=settings()).backend()
    assert report.state == DEGRADED
    assert report.evidence["superseded_epoch"] == 1


def _document(store, *, state="queued", desired_epoch=1):
    record = recorded(store)
    insert(store, "backend_documents", record_id=record, revision="1",
           backend="hindsight", bank_id="hermes", document_id="doc_1",
           desired_epoch=desired_epoch, state=state)


# -- the resource gate -------------------------------------------------------

def test_a_gate_that_has_never_been_used_is_unconfigured(store):
    assert StatusReporter(store).resource_gate().state == UNCONFIGURED


def test_a_held_slot_names_who_is_standing_in_it(store, gate):
    gate.try_acquire(route="retain", holder="worker-1", resource="remote-9b", priority=1,
                     ttl=60)
    report = StatusReporter(store).resource_gate()
    assert report.state == OPERATIONAL
    assert report.evidence["held"][0]["holder"] == "worker-1"
    assert 0 < report.evidence["held"][0]["lease_seconds_left"] <= 60


def test_a_reservation_that_never_resolved_blocks_the_resource_until_it_does(store, gate):
    reservation = gate.try_acquire(route="retain", holder="worker-1",
                                   resource="remote-9b", priority=1, ttl=60)
    gate.mark_uncertain(reservation, reason="no response")
    report = StatusReporter(store).resource_gate()
    assert report.state == DEGRADED
    assert report.evidence["blocked"] == ["remote-9b"]


def test_a_paused_gate_is_reported_even_when_nothing_is_running(store, gate):
    gate.pause(actor="owner", reason="model update")
    assert StatusReporter(store).resource_gate().state == PAUSED


def test_a_gate_with_only_finished_work_is_idle_rather_than_unused(store, gate):
    reservation = gate.try_acquire(route="retain", holder="worker-1",
                                   resource="remote-9b", priority=1, ttl=60)
    gate.release(reservation, outcome="succeeded")
    report = StatusReporter(store).resource_gate()
    assert report.state == OPERATIONAL
    assert report.evidence["held"] == []


def test_an_operation_that_cannot_be_charged_makes_the_gate_degraded(store):
    """Nobody's allowance paid for this one, and that is a fault, not a curiosity.

    The alternative reading of a null resource is that another profile's budget covered it,
    which is the thing §8.1.2 refuses to allow by construction.
    """
    from hermes_memory.backend.worker_launcher import OperationLedger
    from hermes_memory.processing.instance_gate import instance_gate

    configuration = settings()
    with instance_gate(configuration) as gate:
        OperationLedger(gate.store).record(
            {"_operation_id": "op-9", "operation_type": "consolidate", "bank_id": "hermes"},
            worker_id="worker-1", bank_id="hermes", resource=None)
        report = StatusReporter(store, settings=configuration, gate=gate).resource_gate()
    assert report.state == DEGRADED
    assert report.evidence["operations"]["unattributed"] == 1
    assert "unaccounted" in report.detail


def test_a_cancellation_with_no_answer_makes_the_gate_degraded(store):
    """Somebody asked a running operation to stop and nothing replied. That is still live."""
    from hermes_memory.backend.worker_launcher import OperationLedger
    from hermes_memory.processing.instance_gate import instance_gate

    configuration = settings()
    with instance_gate(configuration) as gate:
        ledger = OperationLedger(gate.store)
        ledger.record({"_operation_id": "op-9", "operation_type": "consolidate",
                       "bank_id": "hermes"}, worker_id="worker-1", bank_id="hermes",
                      resource="remote-9b")
        ledger.mark("op-9", "running")
        ledger.request_cancellation("op-9", actor="jugaadu")
        report = StatusReporter(store, settings=configuration, gate=gate).resource_gate()
    assert report.state == DEGRADED
    assert "cancellation(s) asked for with no answer" in report.detail
    assert report.evidence["operations"]["cancellations_owed"] == 1


def test_a_gate_ledger_from_before_the_operation_table_is_reported_not_crashed(store, gate):
    """A reading cannot upgrade the ledger it was handed, so it says the table is absent."""
    report = StatusReporter(store, settings=settings(), gate=gate).resource_gate()
    assert report.evidence["operations"]["ledger"] == "absent"
    assert report.state != DEGRADED


def test_the_gate_reports_what_it_charged_without_leaking_credentials(store, gate):
    reservation = gate.try_acquire(route="retain", holder="worker-1",
                                   resource="remote-9b", priority=1, ttl=60)
    gate.release(reservation, outcome="succeeded", tokens=120)
    usage = StatusReporter(store).resource_gate().evidence["usage"]
    assert usage["remote-9b"]["tokens"] == 120


# -- erasure, confirmation debts and the queue -------------------------------

def test_a_forgotten_record_with_an_unverified_copy_abroad_is_a_debt(store):
    record = recorded(store)
    insert(store, "erasure_ledger", id="er-1", source="gmail", requested_at=PAST,
           requested_by="owner", requester_kind="owner", reason="private",
           preview="{}", preview_digest="d" * 16, state="erasure_pending", epoch=1)
    insert(store, "erasure_targets", intent_id="er-1", kind="record", reference=record,
           state="verified")
    insert(store, "erasure_targets", intent_id="er-1", kind="tombstone",
           reference=record, state="verified")
    insert(store, "erasure_targets", intent_id="er-1", kind="attachment",
           reference="sha256:abc", state="verified")
    insert(store, "erasure_targets", intent_id="er-1", kind="backend-document",
           reference="doc_1", state="pending")
    backlog = StatusReporter(store).erasure_backlog()
    assert backlog["obligations_open"] == 1, "verified copies are not still owed"
    assert "not verified gone everywhere" in json.dumps(
        StatusReporter(store).report()["notes"])


def test_an_awaiting_confirmation_erasure_is_named_as_the_owners_to_decide(store):
    insert(store, "erasure_ledger", id="er-1", source="gmail", requested_at=PAST,
           requested_by="owner", requester_kind="owner", reason="private",
           preview="{}", preview_digest="d" * 16, state="awaiting_confirmation", epoch=1)
    waiting = StatusReporter(store).pending_confirmations()
    assert waiting["erasure_intents"] == 1
    report = StatusReporter(store).report()
    assert report["erasure_backlog"]["awaiting_owner"] == 1
    # A count with no door behind it is how a decision ends up waited on by nobody.
    assert any("owner --list" in note for note in report["notes"])


def test_an_identity_candidate_waits_for_the_owner_and_for_nobody_else(store):
    for account in ("a", "b"):
        insert(store, "identity_accounts", id=account, namespace="email",
               identifier=f"{account}@example.com", normalized=f"{account}@example.com",
               created_at=PAST)
    insert(store, "identity_candidates", id="cand-1", account_a="a", account_b="b",
           rule="same-display-name", rule_version="v1", basis="name", evidence="[]",
           proposed_by="agent", proposed_kind="agent", proposed_at=PAST, state="pending")
    report = StatusReporter(store).report()
    assert report["awaiting_owner"]["identity_candidates"] == 1
    assert any("no agent may confirm" in note for note in report["notes"])


def test_a_job_that_has_waited_past_the_allowance_says_so(store):
    a_job(store, created_at="2000-01-01T00:00:00+00:00")
    report = StatusReporter(store).report()
    assert report["queue"]["stale"] is True
    assert any("nothing is consuming it" in note for note in report["notes"])


def test_a_job_that_arrived_now_does_not_sound_the_same_alarm(store):
    a_job(store)
    report = StatusReporter(store).report()
    assert report["queue"]["stale"] is False
    assert not any("nothing is consuming it" in note for note in report["notes"])


def test_a_finished_job_is_not_the_thing_that_stuck_the_queue(store):
    a_job(store, state="succeeded", created_at="2000-01-01T00:00:00+00:00")
    assert StatusReporter(store).queue_age()["oldest_seconds"] is None


def test_the_oldest_waiter_is_reported_even_when_a_urgent_job_cut_in_line(store):
    """A queued maintenance job is the proof nothing is consuming, however new the
    interactive job beside it is."""
    a_job(store, job_id="old", priority=3, created_at="2000-01-01T00:00:00+00:00")
    a_job(store, job_id="fresh", priority=0, created_at=datetime.now(timezone.utc)
          .isoformat())
    queue = StatusReporter(store).queue_age()
    assert queue["stale"] is True
    assert queue["oldest_seconds"] > 20 * 365 * 24 * 3600


def test_a_job_row_with_an_impossible_timestamp_is_not_invented_an_age(store):
    a_job(store, created_at="whenever")
    queue = StatusReporter(store).queue_age()
    assert queue["oldest_seconds"] is None
    assert queue["stale"] is False


def test_a_candidate_assertion_is_the_owners_to_confirm(store):
    record = recorded(store)
    insert(store, "assertions", id="asr-1", subject="owner", predicate="location",
           value="Rotterdam", category="profile", evidence_kind="observed_pattern",
           record_id=record, quote_start=0, quote_end=9, quote="The meetin",
           status="candidate", created_by="agent", created_at=PAST)
    waiting = StatusReporter(store).pending_confirmations()
    assert waiting["candidate_assertions"] == 1
    assert waiting["identity_candidates"] == 0


def test_reading_status_leaves_no_transaction_open(store, sync):
    connector(store, sync)
    reporter = StatusReporter(store)
    reporter.report()
    assert store.db.in_transaction is False
    reporter.stage("capture")
    assert store.db.in_transaction is False
    recorded(store)


def test_the_allowance_a_queue_is_judged_against_is_the_operators_number(store):
    a_job(store, created_at=_minutes_ago(90))
    assert StatusReporter(store).queue_age()["stale"] is True
    assert StatusReporter(store, stale_queue_minutes=120).queue_age()["stale"] is False


def test_an_allowance_outside_a_plausible_range_is_refused(store):
    with pytest.raises(EvidenceError, match="between 1 minute and a week"):
        StatusReporter(store, stale_queue_minutes=0)
    with pytest.raises(EvidenceError, match="between 1 minute and a week"):
        StatusReporter(store, stale_queue_minutes=99_999)


def _minutes_ago(minutes: int) -> str:
    from datetime import timedelta

    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


# -- the summary line --------------------------------------------------------

def test_one_degraded_part_makes_the_summary_line_degraded(store):
    a_job(store, state="quarantined")
    report = StatusReporter(store).report()
    assert report["overall"] == DEGRADED
    assert report["states"]["observations"] == DEGRADED


def test_a_paused_store_is_not_reported_as_a_working_one(store, sync):
    connector(store, sync)
    sync.pause_capture("gmail", actor="owner", reason="review", policy_version="v1")
    report = StatusReporter(store).report()
    assert report["overall"] == PAUSED


def test_stage_and_report_agree_because_both_read_the_same_queries(store):
    recorded(store)
    reporter = StatusReporter(store)
    from_report = next(row for row in reporter.report()["stages"]
                       if row["name"] == "raw_indexing")
    assert from_report == reporter.stage("raw_indexing").as_dict()


def test_the_report_is_readable_from_inside_a_write_transaction(store, sync):
    """Status asked by a worker must not be the reason that worker failed."""
    connector(store, sync)
    store.db.execute("BEGIN")
    try:
        assert StatusReporter(store).report()["states"]["capture"] == OPERATIONAL
    finally:
        store.db.execute("ROLLBACK")


def test_a_job_queued_by_the_real_queue_is_counted(store):
    JobQueue(store).enqueue(kind="retain", inputs=["rec-1"], input_revision="1",
                            route=RETAIN, processor_fingerprint="extractor-v3")
    report = StatusReporter(store).observations()
    assert report.evidence["queue"] == {"queued": 1}
    assert report.state == CONFIGURED, "the queue holds work; nothing is performing it"


def test_a_hold_left_over_from_a_superseded_release_says_so(store, tmp_path):
    """A hold's reason is free text about the machine that was running when it was written.

    The release record carries the instant it was staged, so the reading can say "this
    decision predates the release now answering" instead of leaving an operator to work out
    whether the hold still describes anything. A release that records no instant — every tree
    staged before the field existed — says nothing rather than guessing.
    """
    from types import SimpleNamespace

    from hermes_memory.processing.instance_gate import ResourceGate

    store.set_control("global", "delivery", "paused", actor="owner",
                      reason="switching to the staged release deadbeef",
                      policy_version="operator-pause")
    home = tmp_path / "instance"
    current = home / "runtime" / "current"
    current.mkdir(parents=True)
    record = current / "RELEASE.json"
    reporter = lambda: StatusReporter(store, settings=SimpleNamespace(home=home),
                                      gate=ResourceGate(store))

    record.write_text(json.dumps({"staged_at": "2099-01-01T00:00:00+00:00"}), encoding="utf-8")
    detail = reporter().delivery().detail
    assert "set before this release was staged (2099-01-01T00:00:00+00:00)" in detail, \
        "a hold older than the running release is reported as though it were about it"

    record.write_text(json.dumps({"staged_at": "2000-01-01T00:00:00+00:00"}), encoding="utf-8")
    assert "set before this release" not in reporter().delivery().detail, \
        "a decision taken about this release is called a leftover"

    record.write_text(json.dumps({"source_commit": "a" * 40}), encoding="utf-8")
    assert "set before this release" not in reporter().delivery().detail, \
        "a release that records no staging instant was dated anyway"
    assert reporter().release_staged_at() is None

    # A zone-less instant is not a moment; comparing one against a zoned one would raise.
    record.write_text(json.dumps({"staged_at": "2099-01-01T00:00:00"}), encoding="utf-8")
    assert "set before this release" not in reporter().delivery().detail, \
        "a naive timestamp was ordered against a zoned one"

    # The inference hold is the other stage an operator has to know is stale.
    store.set_control("global", "inference", "paused", actor="owner",
                      reason="the gpu is being ventilated", policy_version="operator-pause")
    record.write_text(json.dumps({"staged_at": "2099-01-01T00:00:00+00:00"}), encoding="utf-8")
    assert "set before this release was staged" in reporter().observations().detail, \
        "the inference hold was left as undateable as the delivery one"


def test_an_operators_hold_stops_the_delivery_stage_without_a_single_paused_source(store):
    """The fence is the whole installation's, and the headline has to say so.

    The per-source pause answers "is this one pipeline narrowed"; this answers "is
    anything leaving the machine at all", and it is the one the stop command sets.
    """
    store.set_control("global", "delivery", "paused", actor="owner",
                      reason="the owner is away", policy_version="operator-pause")
    report = StatusReporter(store).delivery()
    assert report.state == PAUSED
    assert "holding delivery" in report.detail
    assert report.evidence["instance_hold"] is True
    assert "(owner: the owner is away)" in report.detail, \
        "the fence is the installation's; the name on it is the operator's"
    assert report.evidence["instance_hold_by"] == "owner"
    store.set_control("global", "delivery", "active", actor="owner",
                      reason="the owner is back", policy_version="operator-pause")
    after = StatusReporter(store).delivery()
    assert after.state == UNCONFIGURED and after.evidence["instance_hold"] is False


# -- the background pass -------------------------------------------------------

def a_heartbeat(store, *, at=None, **metadata):
    columns = dict(at=at or "2026-09-15T09:00:00+00:00", sections=["queue"], prepared=1,
                   deferred=0, suppressed=0, refreshes_promised=0, inference_paused=False)
    columns.update(metadata)
    store._audit("maintenance_pass", "maintenance", columns)


def aged_heartbeat(store, *, at=PAST):
    insert(store, "audit", action="maintenance_pass", object_id="maintenance",
           created_at=at, metadata=json.dumps({"at": at}))


def test_an_installation_with_nothing_scheduled_names_the_command(store):
    report = StatusReporter(store, settings=settings()).background_pass()
    assert report["scheduled"] is False and report["behind"] is False
    assert "hermes-memory maintain" in report["note"]


def test_a_pass_that_has_never_been_recorded_while_reminders_wait_is_behind(store):
    """The failure this line exists for: a dead scheduler looks exactly like a busy machine."""
    a_goal(store)
    a_due_event(store, fire_at=PAST)
    report = StatusReporter(store,
                            settings=settings(maintenance_interval_s=900)).background_pass()
    assert report["scheduled"] is True and report["waiting"] == 1
    assert report["behind"] is True and report["last_pass_at"] is None
    assert "no pass has been recorded" in report["note"]


def test_a_heartbeat_that_records_a_failure_is_not_reported_as_a_healthy_period(store):
    """A loop that raises every period still writes a heartbeat, at the moment it broke.

    Age alone would call that on schedule, so the reading carries the section and the error:
    "the scheduler is running but the pass is not finishing" is a different finding from
    "the scheduler is not running", and the two had the same report.
    """
    a_heartbeat(store, failed_section="queue",
                error="OperationalError: attempt to write a readonly database")
    report = StatusReporter(store,
                            settings=settings(maintenance_interval_s=900)).background_pass()
    assert report["failed_section"] == "queue"
    assert "readonly" in report["error"] and report["behind"] is True
    assert "raised in section" in report["note"] and "not finishing" in report["note"]
    assert report["last_report"]["failed_section"] == "queue"


def test_a_recent_heartbeat_stops_the_alarm_and_says_what_it_did(store):
    a_goal(store)
    a_due_event(store, fire_at=PAST)
    a_heartbeat(store)
    report = StatusReporter(store,
                            settings=settings(maintenance_interval_s=900)).background_pass()
    assert report["behind"] is False, "the pass wrote down that it ran, not the thread"
    assert report["last_report"]["prepared"] == 1
    assert report["seconds_since_last_pass"] is not None


def test_a_heartbeat_older_than_three_periods_is_a_dead_loop(store):
    a_goal(store)
    a_due_event(store, fire_at=PAST)
    aged_heartbeat(store)
    report = StatusReporter(store,
                            settings=settings(maintenance_interval_s=900)).background_pass()
    assert report["behind"] is True
    assert "older than three periods" in report["note"]


def test_a_stale_loop_with_nothing_waiting_is_not_an_alarm(store):
    aged_heartbeat(store)
    report = StatusReporter(store,
                            settings=settings(maintenance_interval_s=900)).background_pass()
    assert report["behind"] is True and report["waiting"] == 0
    assert not [note for note in StatusReporter(store, settings=settings(
        maintenance_interval_s=900)).report()["notes"] if note.startswith("background:")]


def test_the_note_is_only_raised_when_somebody_is_actually_waiting(store):
    a_goal(store)
    a_due_event(store, fire_at=PAST)
    notes = StatusReporter(store,
                           settings=settings(maintenance_interval_s=900)).report()["notes"]
    assert any(note.startswith("background: 1 reminder") for note in notes), notes
