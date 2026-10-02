"""Offline safety regressions for the feature review's R-001 through R-032.

Assertions describe the safe contracts after remediation. Synthetic stores and
injected transports only; no production paths or real external sends.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from conftest import envelope
from hermes_memory.backend.hindsight_client import RecallOutcome
from hermes_memory.context import ContextBroker
from hermes_memory.knowledge.assertions import AssertionStore
from hermes_memory.knowledge.summaries import SummaryStore
from hermes_memory.learning.lessons import LessonStore
from hermes_memory.learning.outcomes import OutcomeLog
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.lifecycle.reset import ResetController
from hermes_memory.proactive.delivery import DeliveryPolicy, deliver_once
from hermes_memory.proactive.outbox import Outbox
from hermes_memory.proactive.policy import AttentionPolicy, POLICY_VERSION
from hermes_memory.prospective.due_events import DueEventLog
from hermes_memory.prospective.goals import GoalStore
from hermes_memory.sources.files import FileSource
from hermes_memory.sources.runtime import ConnectorRuntime
from hermes_memory.sources.structured import StructuredSource
from hermes_memory.sources.sync import SyncController
from hermes_memory.storage.blobs import BlobStore
from hermes_memory.storage.evidence import EvidenceError, prepare_envelope
from hermes_memory.storage.identity import IdentityStore

OWNER = "review-owner"
AT = 1789462800.0
MOMENT = datetime.fromtimestamp(AT, timezone.utc).isoformat()


def assertion(store, record, **extra):
    claims = AssertionStore(store, owner_principal=OWNER)
    made = claims.propose(subject="review-person", predicate="preference",
                          value="synthetic secret", kind="preference",
                          evidence_kind="explicit_statement", record_id=record,
                          quote="synthetic secret", proposed_by="agent", **extra)
    return claims, made["id"]


def summary(store, record, **extra):
    readings = SummaryStore(store, owner_principal=OWNER)
    made = readings.publish(scope="source:gmail", kind="day", title="Review",
                            body="synthetic summary", citations=[{"record_id": record}],
                            processor_fingerprint="review-processor", **extra)
    return readings, made["id"]


def outbox(store, *, evidence=()):
    policy = AttentionPolicy(store, owner_principal=OWNER)
    policy.configure(actor=OWNER, timezone_name="UTC", max_immediate_per_day=10,
                     cooldown_minutes=0, shadow=False,
                     quiet_from="22:00", quiet_until="23:00")
    events = DueEventLog(store, clock=lambda: AT)
    goals = GoalStore(store, events=events, owner_principal=OWNER)
    goal = goals.propose(title="Synthetic goal", statement="Review only", due=MOMENT,
                         proposed_by=OWNER, proposed_kind="owner")["id"]
    event = store.db.execute("SELECT id FROM due_events WHERE goal_id=?", (goal,)).fetchone()[0]
    claim = events.claim(event, holder="review", at=AT)
    intent = events.ack(event_id=event, token=claim.token, decision="awaiting_analysis",
                        policy_version=POLICY_VERSION)["intent"]
    decision = policy.record(policy.decide(topic="general", at=MOMENT),
                             intent_id=intent, goal_id=goal, revision=1)
    box = Outbox(store, policy=policy, owner_principal=OWNER, clock=lambda: AT)
    made = box.prepare(decision_id=decision["id"], kind="notify_owner", topic="general",
                       payload="synthetic notification", evidence=evidence)
    return box, made["id"], goals, goal


def test_r001_cached_identity_access_survives_edge_revocation(store):
    identity = IdentityStore(store, owner_principal=OWNER)
    a = identity.account("email", "a@review.invalid")
    b = identity.account("email", "b@review.invalid")
    record = store.commit(envelope(text="synthetic secret", metadata={"account_ids": [b]}))["id"]
    candidate = identity.propose(account_a=a, account_b=b, rule="explicit-alias-declared",
                                 basis="synthetic alias", evidence=[record], proposed_by="agent")
    edge = identity.confirm(candidate_id=candidate["candidate_id"], actor=OWNER,
                            reason="synthetic join")["edge_id"]
    with_broker = ContextBroker(store, identity=identity, account_id=a)
    try:
        warm = with_broker.assemble("synthetic", include_derived=False)
        stamp = store.watermark()
        identity.revoke(edge_id=edge, actor=OWNER, reason="synthetic revoke")
        assert store.watermark() != stamp
        assert b not in identity.group(a)
        cached = with_broker.assemble("synthetic", include_derived=False)
        assert len(warm.items) == 1 and not cached.items
        with_broker.cache.invalidate(reason="review")
        assert not with_broker.assemble("synthetic", include_derived=False).items
    finally:
        with_broker.close()


@pytest.mark.parametrize("kind", ["assertion", "summary"])
def test_r001_cached_derived_section_survives_owner_withdrawal(store, kind):
    record = store.commit(envelope(text="synthetic secret", metadata={}))["id"]
    collection, identifier = assertion(store, record) if kind == "assertion" else summary(store, record)
    broker = ContextBroker(store, **{("assertions" if kind == "assertion" else "summaries"): collection})
    try:
        first = broker.assemble("synthetic", include_derived=False)
        stamp = store.watermark()
        if kind == "assertion":
            collection.retract(assertion_id=identifier, actor=OWNER, reason="review withdrawal")
        else:
            collection.withdraw(identifier, actor=OWNER, reason="review withdrawal")
        assert store.watermark() != stamp
        second = broker.assemble("synthetic", include_derived=False)
        section = "assertions" if kind == "assertion" else "summaries"
        assert len(getattr(first, section)) == 1
        assert not getattr(second, section)
        broker.cache.invalidate(reason="review")
        assert not getattr(broker.assemble("synthetic", include_derived=False), section)
    finally:
        broker.close()


def test_r002_assertion_exposes_record_withheld_from_caller(store):
    identity = IdentityStore(store)
    a = identity.account("email", "a@review.invalid")
    b = identity.account("email", "b@review.invalid")
    record = store.commit(envelope(text="synthetic secret", metadata={"account_ids": [b]}))["id"]
    claims, _ = assertion(store, record)
    broker = ContextBroker(store, identity=identity, assertions=claims, account_id=a, cache=False)
    try:
        packet = broker.assemble("synthetic", include_derived=False)
        assert not packet.items and packet.withheld == 1
        assert not packet.assertions
        assert "synthetic secret" not in packet.render()
    finally:
        broker.close()


def test_r003_derived_fact_survives_canonical_hide_during_recall(store):
    record = store.commit(envelope(text="synthetic secret", metadata={}))["id"]
    from hermes_memory.storage.evidence import EvidenceStore

    def recall(*args, **kwargs):
        # A separate connection, as a concurrent owner action would use.
        with EvidenceStore(store.path) as writer:
            writer.hide(record, reason="review revoke", actor=OWNER)
        return RecallOutcome(results=({"text": "synthetic secret", "document_id": "review-doc"},))

    broker = ContextBroker(store, client=SimpleNamespace(recall=recall), cache=False)
    try:
        packet = broker.assemble("synthetic")
        assert not packet.items
        assert "revoked_during_recall" in packet.truncated
        assert not packet.facts and "synthetic secret" not in packet.render()
    finally:
        broker.close()


def test_r004_mental_model_can_publish_without_configured_owner(store):
    record = store.commit(envelope(text="synthetic evidence", metadata={}))["id"]
    readings = SummaryStore(store)
    with pytest.raises(EvidenceError):
        readings.publish(scope="project:review", kind="mental_model", title="Review",
                         body="synthetic inference", citations=[{"record_id": record}],
                         processor_fingerprint="review-processor")


def test_r005_canonical_revision_ignores_precision_and_parent_dependencies(store):
    parent = store.commit(envelope(source_id="parent", metadata={}))["id"]
    original = envelope(source_id="child", metadata={})
    store.commit(original)
    changed = {**original, "occurred_precision": "day", "parent_record_ids": [parent]}
    assert prepare_envelope(original).fingerprint != prepare_envelope(changed).fingerprint
    with pytest.raises(EvidenceError):
        store.commit(changed)
    assert store.db.execute("SELECT count(*) FROM record_dependencies").fetchone()[0] == 0


def test_r006_reset_reports_complete_but_keeps_attachment_payloads(store):
    payload = b"synthetic private attachment"
    record = store.commit(envelope(metadata={}, attachments=[
        {"filename": "review.txt", "mime": "text/plain", "data": payload}]))["id"]
    assert BlobStore(store).list_for(record)
    reset = ResetController(store, owner_principal=OWNER)
    preview = reset.preview(actor=OWNER, reason="synthetic reset")
    result = reset.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"], actor=OWNER)
    assert result["state"] == "complete" and not store.live_and_visible(record)
    assert store.db.execute("SELECT count(*) FROM attachments").fetchone()[0] == 0
    assert store.db.execute("SELECT count(*) FROM blob_chunks").fetchone()[0] == 0


def test_r007_replay_of_erased_revision_recreates_destroyed_blob(store):
    item = envelope(metadata={}, attachments=[
        {"filename": "review.txt", "mime": "text/plain", "data": b"synthetic payload"}])
    record = store.commit(item)["id"]
    erase = ErasureManager(store, owner_principal=OWNER)
    preview = erase.preview(record_ids=[record], actor=OWNER, reason="synthetic erase")
    erase.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"], actor=OWNER)
    assert store.db.execute("SELECT count(*) FROM blob_chunks").fetchone()[0] == 0
    assert store.commit(item)["duplicate"]
    assert not store.live_and_visible(record)
    assert store.db.execute("SELECT count(*) FROM blob_chunks").fetchone()[0] == 0


@pytest.mark.parametrize("adapter_kind", ["file", "structured"])
def test_r008_single_export_overflows_declared_page_bound(tmp_path, adapter_kind):
    path = tmp_path / "review.json"
    if adapter_kind == "file":
        rows = [{"id": str(i), "text": "synthetic"} for i in range(12)]
        adapter = FileSource(tmp_path, records_per_page=2)
    else:
        rows = [{"value": i, "unit": "kg", "time": MOMENT} for i in range(12)]
        adapter = StructuredSource(tmp_path, per_sample=True, records_per_page=2)
    path.write_text(json.dumps(rows), encoding="utf-8")
    page = adapter.read_page(None)
    assert len(page.envelopes) == 2 and page.next_cursor is not None
    records, gaps = adapter.read_all()
    assert len(records) == 12 and not gaps


def test_r009_touch_unchanged_file_causes_immutable_revision_conflict(store, tmp_path):
    path = tmp_path / "review.txt"
    path.write_text("synthetic file", encoding="utf-8")
    adapter = FileSource(tmp_path)
    first = adapter.read_page(None).envelopes[0]
    store.commit(first)
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))
    second = adapter.read_page(None).envelopes[0]
    assert first["revision"] == second["revision"]
    assert store.commit(second)["duplicate"]


@pytest.mark.parametrize("per_sample", [True, False])
def test_r010_edited_structured_export_reuses_revision_one(store, tmp_path, per_sample):
    path = tmp_path / "review.csv"
    path.write_text("value,unit,time\n1,kg," + MOMENT + "\n", encoding="utf-8")
    adapter = StructuredSource(tmp_path, per_sample=per_sample)
    first = adapter.read_page(None).envelopes[0]
    store.commit(first)
    path.write_text("value,unit,time\n2,kg," + MOMENT + "\n", encoding="utf-8")
    second = adapter.read_page(None).envelopes[0]
    assert first["source_id"] == second["source_id"]
    assert first["revision"] != second["revision"]
    assert not store.commit(second)["duplicate"]


def test_r011_artifact_can_be_attempted_after_topic_opt_out(store):
    box, identifier, _, _ = outbox(store)
    lease = box.lease

    def revoke_after_claim(**kwargs):
        claim = lease(**kwargs)
        assert claim and claim.artifact.id == identifier
        box.policy.configure(topic="general", actor=OWNER, opted_out=True)
        assert not box.revalidate(identifier, at=AT).ok
        return claim

    box.lease = revoke_after_claim
    sends = []
    report = deliver_once(box, policy=DeliveryPolicy(True, "local:review", OWNER),
                          sink=lambda body: sends.append(body) or {"sent": True}, at=AT)
    assert not report["delivered"] and not sends


def test_r012_generator_evidence_is_lost_by_outbox_prepare(store):
    record = store.commit(envelope(metadata={}))["id"]
    box, identifier, _, _ = outbox(store, evidence=iter([record]))
    artifact = box.get(identifier)
    assert artifact.evidence == (record,)
    assert box.revalidate(identifier, at=AT).ok


def test_r013_cancellation_reason_produces_invalid_json_audit(store):
    goals = GoalStore(store, owner_principal=OWNER)
    goal = goals.propose(title="Review", statement="synthetic", due=MOMENT,
                         proposed_by=OWNER, proposed_kind="owner")["id"]
    goals.cancel(goal_id=goal, actor=OWNER, reason='owner said "stop"')
    raw = store.db.execute("SELECT metadata FROM audit WHERE action='due_event_cancelled'").fetchone()[0]
    assert json.loads(raw)["reason"] == 'owner said "stop"'


def test_r014_lesson_prerequisites_and_exceptions_are_not_enforced(store):
    record = store.commit(envelope(metadata={}))["id"]
    habits = LessonStore(store, outcomes=OutcomeLog(store, owner_principal=OWNER), owner_principal=OWNER)
    made = habits.propose(lesson_id="review-rule", text="synthetic lesson",
                          applicability={"field": "tool", "op": "eq", "value": "shell"},
                          prerequisites=["owner-approved"], exceptions=["production"],
                          evidence=[record], proposed_by="agent", proposed_kind="agent")
    habits.activate(lesson_id=made["id"], version=made["version"], actor=OWNER, reason="review")
    found = habits.applicable({"tool": "shell", "host": "production"})
    assert not found
    assert len(habits.applicable({"tool": "shell", "conditions": {
        "owner-approved": True, "production": False}})) == 1


def test_r015_empty_host_receipt_is_counted_as_checked_success(store):
    outcomes = OutcomeLog(store, owner_principal=OWNER, outbox=SimpleNamespace(get=lambda _: None))
    with pytest.raises(EvidenceError):
        outcomes.record(subject_kind="lesson", subject_id="review@1", kind="host_receipt",
                        valence="success", note="synthetic unproved claim", actor="agent", evidence=[])
    assert outcomes.tally("lesson", "review@1")["support"] == 0


def test_r016_seconds_budget_does_not_refuse_further_dispatch(store):
    from hermes_memory.processing.budgets import Budget, Budgets
    budgets = Budgets(store, daily={"review": Budget(tokens=1000, calls=10, seconds=1)})
    budgets.charge("review", tokens=1, seconds=10)
    assert budgets.used("review")["seconds"] > budgets.daily["review"].seconds
    from hermes_memory.processing.budgets import BudgetExhausted
    with pytest.raises(BudgetExhausted):
        budgets.admit("review", estimated_tokens=1)


def test_r017_backend_fact_bypasses_caller_source_window_and_account_scope(store):
    identity = IdentityStore(store)
    a = identity.account("email", "a@review.invalid")
    b = identity.account("email", "b@review.invalid")
    record = store.commit(envelope(text="synthetic private", source="private",
                                    metadata={"account_ids": [b]}))["id"]
    calls = []

    def recall(query, **kwargs):
        calls.append(kwargs)
        return RecallOutcome(results=({"text": "synthetic private", "record_id": record},))

    broker = ContextBroker(store, identity=identity, account_id=a,
                           client=SimpleNamespace(recall=recall), cache=False)
    try:
        packet = broker.assemble("synthetic", sources=["public"],
                                  window=("2027-01-01T00:00:00+00:00", None))
        assert not packet.items and not packet.facts
        assert "tags" not in calls[0] and "temporal_window" not in calls[0]
    finally:
        broker.close()


def test_r018_broker_limit_one_returns_multiple_raw_items(store):
    for index in range(5):
        store.commit(envelope(source_id=str(index), text="synthetic match", metadata={}))
    broker = ContextBroker(store, cache=False)
    try:
        assert len(broker.assemble("synthetic", limit=1, include_derived=False).items) == 1
    finally:
        broker.close()


def test_r019_summary_identity_omits_citation_manifest(store):
    first = store.commit(envelope(source_id="first", metadata={}))["id"]
    second = store.commit(envelope(source_id="second", metadata={}))["id"]
    readings, identifier = summary(store, first)
    replay = readings.publish(scope="source:gmail", kind="day", title="Review",
                              body="synthetic summary", citations=[{"record_id": second}],
                              processor_fingerprint="review-processor")
    assert replay["id"] != identifier and replay["published"]
    rows = store.db.execute("SELECT record_id FROM derived_citations WHERE artifact_id=?", (identifier,)).fetchall()
    assert [row[0] for row in rows] == [first]
    assert store.db.execute("SELECT record_id FROM derived_citations WHERE artifact_id=?",
                            (replay["id"],)).fetchone()[0] == second


def job_queue(store, clock=lambda: AT):
    from hermes_memory.processing.jobs import JobQueue
    from hermes_memory.processing.routes import Route
    queue = JobQueue(store, clock=clock)
    route = Route("review", "review-resource", "chat", "http://127.0.0.1:9999/v1",
                  "review-credential", "maintenance", 100)
    record = store.commit(envelope(metadata={}))["id"]
    identifier = queue.enqueue(kind="review", inputs=[record], input_revision="1",
                               route=route, processor_fingerprint="review-v1")["job_id"]
    return queue, identifier


def test_r020_released_old_job_claim_can_mutate_new_holder_state(store):
    queue, identifier = job_queue(store)
    old = queue.claim(worker="old")
    queue.release(old)
    current = queue.claim(worker="current")
    assert old.lease != current.lease
    with pytest.raises(EvidenceError):
        queue.begin_submission(old, submission_id="stale-submission")
    assert queue.get(identifier).state == "leased"
    assert queue.get(identifier).lease == current.lease
    assert queue.get(identifier).submission_id is None


def test_r020_pre_reset_worker_can_resurrect_epoch_quarantined_job(store):
    queue, identifier = job_queue(store)
    old = queue.claim(worker="old")
    store.bump_epoch(reason="synthetic reset", actor=OWNER)
    queue.abandon_stale_epoch()
    assert queue.get(identifier).state == "quarantined"
    with pytest.raises(EvidenceError):
        queue.complete(old, covered=list(old.inputs))
    assert queue.get(identifier).state == "quarantined"
    assert queue.get(identifier).epoch < store.epoch()


def test_r021_reenqueued_cancelled_job_keeps_obsolete_epoch(store):
    from hermes_memory.processing.routes import Route
    queue, identifier = job_queue(store)
    queue.cancel(identifier, actor=OWNER, reason="review")
    old = queue.get(identifier)
    store.bump_epoch(reason="review reset", actor=OWNER)
    route = Route("review", "review-resource", "chat", "http://127.0.0.1:9999/v1",
                  "review-credential", "maintenance", 100)
    reopened = queue.enqueue(kind="review", inputs=list(old.inputs), input_revision="1",
                             route=route, processor_fingerprint="review-v1")
    assert reopened["created"] and reopened["state"] == "queued"
    assert queue.get(reopened["job_id"]).epoch == store.epoch()
    assert queue.claim(worker="review").id == reopened["job_id"]


def test_r022_week_summary_window_covers_thirteen_days(store):
    from hermes_memory.processing.summarization import scope_records
    dates = ["2026-09-19", "2026-09-25", "2026-10-01"]
    for day in dates:
        store.commit(envelope(source_id=day, occurred_at=day + "T12:00:00+00:00", metadata={}))
    found = scope_records(store, scope="week:2026-09-25")
    assert len(found) == 2
    instants = sorted(item["occurred_at"] for item in found)
    assert (datetime.fromisoformat(instants[-1]) - datetime.fromisoformat(instants[0])).days == 6


def test_r023_whatsapp_page_drops_rest_of_single_chat_without_gap(store, tmp_path):
    from hermes_memory.sources.whatsapp_export import WhatsAppExport
    (tmp_path / "_chat.txt").write_text(
        "\n".join(f"[25/09/2026, 09:00] Review: synthetic {i}" for i in range(5)), encoding="utf-8")
    adapter = WhatsAppExport(tmp_path, records_per_page=2)
    sync = SyncController(store)
    sync.register(adapter.source, policy_version="local-only")
    report = ConnectorRuntime(store, sync, holder="review").run(adapter)
    assert report.stopped == "complete" and report.coverage_state == "current"
    assert report.records == 5 and report.gaps == 0
    assert store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 5


def test_r024_mcp_page_limit_discards_tail_before_remote_cursor(store):
    from hermes_memory.sources.mcp import McpSource
    calls = []

    def call_tool(name, args):
        calls.append(args)
        if args.get("after") == "after-3":
            return {"items": [], "next_cursor": None}
        return {"items": [{"id": str(i), "text": "synthetic " + str(i)} for i in range(3)],
                "next_cursor": "after-3"}

    adapter = McpSource(SimpleNamespace(call_tool=call_tool), {"tool": "review"}, records_per_page=2)
    sync = SyncController(store)
    sync.register(adapter.source, policy_version="local-only")
    report = ConnectorRuntime(store, sync, holder="review").run(adapter)
    assert report.records == 3 and report.gaps == 0 and report.coverage_state == "current"
    assert "after" not in calls[1] and calls[2]["after"] == "after-3"


@pytest.mark.parametrize("adapter_kind", ["file", "email", "whatsapp"])
def test_r025_import_adapters_bypass_shared_secret_redaction(store, tmp_path, adapter_kind):
    from hermes_memory.sources.base import REDACTED, normalize_text
    from hermes_memory.sources.email import EmailSource
    from hermes_memory.sources.whatsapp_export import WhatsAppExport
    token = "sk-" + "A" * 24  # deliberately synthetic, never a live credential
    assert normalize_text(token) == REDACTED
    if adapter_kind == "file":
        (tmp_path / "review.txt").write_text(token, encoding="utf-8")
        adapter = FileSource(tmp_path)
    elif adapter_kind == "email":
        (tmp_path / "review.eml").write_text("From: a@review.invalid\nTo: b@review.invalid\n"
                                             "Message-ID: <review@review.invalid>\n\n" + token, encoding="utf-8")
        adapter = EmailSource(tmp_path)
    else:
        (tmp_path / "_chat.txt").write_text("[25/09/2026, 09:00] Review: " + token, encoding="utf-8")
        adapter = WhatsAppExport(tmp_path)
    row = adapter.read_page(None).envelopes[0]
    record = store.commit(row)["id"]
    assert token not in store.get(record).text and REDACTED in store.get(record).text


def test_r026_candidate_assertion_supersedes_confirmed_claim_before_approval(store):
    first = store.commit(envelope(source_id="first", text="synthetic secret", metadata={}))["id"]
    second = store.commit(envelope(source_id="second", text="synthetic replacement", metadata={}))["id"]
    claims, identifier = assertion(store, first)
    proposed = claims.propose(subject="review-person", predicate="preference",
                              value="synthetic replacement", kind="preference",
                              evidence_kind="observed_pattern", record_id=second,
                              quote="synthetic replacement", proposed_by="agent", supersedes=identifier)
    assert proposed["status"] == "candidate"
    assert claims.get(identifier).status == "confirmed"
    assert claims.current(subject="review-person")[0].id == identifier
    claims.confirm(assertion_id=proposed["id"], actor=OWNER, reason="review approval")
    assert claims.get(identifier).status == "superseded"
    assert claims.current(subject="review-person")[0].id == proposed["id"]


def test_r027_command_transport_manufactures_positive_receipt_from_invalid_stdout(monkeypatch):
    import sys
    from hermes_memory.proactive.delivery import command_sink
    monkeypatch.setattr("hermes_memory.proactive.delivery.subprocess.run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout=b"not a receipt", stderr=b""))
    sink = command_sink([sys.executable], destination="telegram:123", hermes_home="/tmp/review-home")
    with pytest.raises(RuntimeError, match="uncertain"):
        sink("synthetic question")


def test_r028_restore_reintroduces_destroyed_attachment_chunks(store, tmp_path):
    from hermes_memory.lifecycle.recovery import Recovery
    from hermes_memory.lifecycle.snapshots import Snapshots
    payload = b"synthetic forgotten file"
    record = store.commit(envelope(metadata={}, attachments=[
        {"filename": "review.txt", "mime": "text/plain", "data": payload}]))["id"]
    snapshots = Snapshots(store, directory=tmp_path / "snapshots")
    snapshot = snapshots.create(actor=OWNER, reason="review")['snapshot']
    erase = ErasureManager(store, owner_principal=OWNER)
    preview = erase.preview(record_ids=[record], actor=OWNER, reason="review")
    erase.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"], actor=OWNER)
    assert store.db.execute("SELECT count(*) FROM blob_chunks").fetchone()[0] == 0
    Recovery(store, snapshots=snapshots, owner_principal=OWNER).restore(snapshot.id, actor=OWNER)
    assert not store.live_and_visible(record)
    assert store.db.execute("SELECT count(*) FROM blob_chunks").fetchone()[0] == 0


def test_r029_restore_rewinds_sent_outbox_to_sendable_state(store, tmp_path):
    from hermes_memory.lifecycle.recovery import Recovery
    from hermes_memory.lifecycle.snapshots import Snapshots
    box, identifier, _, _ = outbox(store)
    snapshots = Snapshots(store, directory=tmp_path / "snapshots")
    snapshot = snapshots.create(actor=OWNER, reason="review")['snapshot']
    policy = DeliveryPolicy(True, "local:review", OWNER)
    sends = []
    sink = lambda body: sends.append(body) or {"sent": True}
    assert deliver_once(box, policy=policy, sink=sink, at=AT)["delivered"]
    assert box.get(identifier).state == "accepted_unverified"
    Recovery(store, snapshots=snapshots, owner_principal=OWNER).restore(snapshot.id, actor=OWNER)
    assert box.get(identifier).state == "uncertain"
    assert not deliver_once(box, policy=policy, sink=sink, at=AT)["delivered"]
    assert len(sends) == 1


def test_r030_orphan_tombstone_dropped_on_restore_allows_forgotten_record_reimport(store, tmp_path):
    from hermes_memory.lifecycle.recovery import Recovery
    from hermes_memory.lifecycle.snapshots import Snapshots
    store.commit(envelope(source_id="before", text="before snapshot", metadata={}))
    snapshots = Snapshots(store, directory=tmp_path / "snapshots")
    snapshot = snapshots.create(actor=OWNER, reason="review")['snapshot']
    item = envelope(source_id="after", text="synthetic forgotten", metadata={})
    record = store.commit(item)["id"]
    erase = ErasureManager(store, owner_principal=OWNER)
    preview = erase.preview(record_ids=[record], actor=OWNER, reason="review")
    erase.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"], actor=OWNER)
    assert not store.live_and_visible(record)
    Recovery(store, snapshots=snapshots, owner_principal=OWNER).restore(snapshot.id, actor=OWNER)
    assert store.db.execute("SELECT count(*) FROM tombstones WHERE record_id=?", (record,)).fetchone()[0] == 0
    store.commit(item)
    assert not store.live_and_visible(record)
    assert not store.search("forgotten")


def test_r031_reminder_threshold_accepts_forgotten_expired_measurement(store):
    from hermes_memory.prospective.predicates import SATISFIED, evaluate
    record = store.commit(envelope(text="synthetic measurement 100 kg", metadata={}))["id"]
    claims = AssertionStore(store, owner_principal=OWNER)
    claims.propose(subject="review-device", predicate="weight", value="100", kind="measurement",
                   unit="kg", evidence_kind="explicit_statement", record_id=record,
                   quote="100 kg", proposed_by="agent", valid_to="2026-09-01T00:00:00+00:00")
    store.hide(record, reason="review revoke", actor=OWNER)
    assert not claims.current(subject="review-device", at=MOMENT)
    result = evaluate(store.db, "measured_threshold", {"subject": "review-device", "predicate": "weight",
                      "operator": ">", "value": 50, "unit": "kg"}, now_iso=MOMENT)
    assert result.state != SATISFIED


def test_r032_new_message_from_accepts_account_string_in_arbitrary_metadata(store):
    from hermes_memory.prospective.predicates import SATISFIED, evaluate
    identity = IdentityStore(store)
    account = identity.account("email", "review@review.invalid")
    store.commit(envelope(metadata={"note": "unrelated annotation review@review.invalid"}))
    result = evaluate(store.db, "new_message_from", {"account_id": account,
                      "since": "2020-01-01T00:00:00+00:00"}, now_iso=MOMENT)
    assert result.state != SATISFIED


def test_r002_authorized_assertion_and_fact_remain_available(store):
    identity = IdentityStore(store)
    account = identity.account("email", "a@review.invalid")
    record = store.commit(envelope(text="synthetic secret", metadata={"account_ids": [account]}))["id"]
    claims, _ = assertion(store, record)
    backend = SimpleNamespace(recall=lambda *a, **k: RecallOutcome(results=(
        {"text": "synthetic derived", "record_id": record},)))
    broker = ContextBroker(store, identity=identity, assertions=claims, account_id=account,
                           client=backend, cache=False)
    try:
        packet = broker.assemble("synthetic", sources=["gmail"])
        assert packet.items and packet.assertions and packet.facts
    finally:
        broker.close()


def test_r017_mixed_resolved_and_unresolved_provenance_is_withheld(store):
    record = store.commit(envelope(metadata={}))["id"]
    backend = SimpleNamespace(recall=lambda *a, **k: RecallOutcome(results=(
        {"text": "synthetic derived", "record_id": record, "document_ids": ["missing"]},)))
    broker = ContextBroker(store, client=backend, cache=False)
    try:
        assert not broker.assemble("synthetic").facts
    finally:
        broker.close()


def test_r004_owner_approved_mental_model_still_publishes(store):
    record = store.commit(envelope(metadata={}))["id"]
    summaries = SummaryStore(store, owner_principal=OWNER)
    arguments = dict(scope="project:review", kind="mental_model", title="Review", body="synthetic",
                     citations=[{"record_id": record}], processor_fingerprint="review")
    with pytest.raises(EvidenceError):
        summaries.publish(**arguments, approved_by="agent")
    assert summaries.publish(**arguments, approved_by=OWNER)["published"]


def test_r005_legacy_fingerprint_replay_checks_all_semantics(store):
    from hermes_memory.ids import digest
    item = envelope(metadata={})
    record = store.commit(item)["id"]
    old = digest([prepare_envelope(item).occurred_at, item["kind"], item["text"], {}, []])
    store.db.execute("UPDATE records SET fingerprint=? WHERE id=?", (old, record))
    assert store.commit(item)["duplicate"]
    with pytest.raises(EvidenceError):
        store.commit({**item, "occurred_precision": "day"})


def test_r009_historical_mtime_is_not_a_new_revision(store, tmp_path):
    path = tmp_path / "review.txt"
    path.write_text("synthetic", encoding="utf-8")
    adapter = FileSource(tmp_path)
    original = adapter.read_page(None).envelopes[0]
    original["metadata"]["mtime"] = "historical observation"
    store.commit(original)
    assert store.commit(adapter.read_page(None).envelopes[0])["duplicate"]


def test_r007_direct_blob_attachment_cannot_bypass_erasure(store):
    from hermes_memory.storage.blobs import BlobError
    record = store.commit(envelope(metadata={}))["id"]
    erase = ErasureManager(store, owner_principal=OWNER)
    preview = erase.preview(record_ids=[record], actor=OWNER, reason="review")
    erase.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"], actor=OWNER)
    store.db.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(BlobError):
            BlobStore(store).attach(record, [{"filename": "review.txt", "data": b"synthetic", "mime": "text/plain"}])
    finally:
        store.db.execute("ROLLBACK")
    assert store.db.execute("SELECT count(*) FROM blob_chunks").fetchone()[0] == 0


def test_r008_file_resume_restarts_changed_content_without_losing_rows(tmp_path):
    path = tmp_path / "review.json"
    rows = [{"id": str(i), "text": "synthetic"} for i in range(5)]
    path.write_text(json.dumps(rows), encoding="utf-8")
    adapter = FileSource(tmp_path, records_per_page=2)
    first = adapter.read_page(None)
    rows.insert(0, {"id": "new", "text": "synthetic inserted"})
    path.write_text(json.dumps(rows), encoding="utf-8")
    second = adapter.read_page(first.next_cursor)
    assert second.envelopes[0]["source_id"] == "new"


def test_r024_mcp_resume_rejects_changed_remote_page(store):
    from hermes_memory.sources.base import CursorExpired
    from hermes_memory.sources.mcp import McpSource
    answers = [{"items": [{"id": str(i), "text": "synthetic"} for i in range(3)]},
               {"items": [{"id": str(i), "text": "changed"} for i in range(3)]}]
    adapter = McpSource(SimpleNamespace(call_tool=lambda *a: answers.pop(0)),
                        {"tool": "review"}, records_per_page=2)
    first = adapter.read_page(None)
    with pytest.raises(CursorExpired):
        adapter.read_page(first.next_cursor)


def test_r025_redaction_covers_metadata_and_receipts(store):
    token = "sk-" + "A" * 24
    record = store.commit(envelope(text=token, metadata={"nested": [{"note": token}]}))["id"]
    assert token not in json.dumps(store.get(record).metadata)
    receipt = store.db.execute("SELECT envelope FROM ingestion_receipts WHERE record_id=?", (record,)).fetchone()[0]
    assert token not in receipt


def test_r011_expired_handoff_lease_is_refused(store):
    box, identifier, _, _ = outbox(store)
    claim = box.lease(holder="review", at=AT, lease_s=1)
    with pytest.raises(EvidenceError, match="expired"):
        box.attempt(artifact_id=identifier, token=claim.token, at=AT + 2)
    assert box.get(identifier).state == "leased"


def test_r012_oversized_manifest_is_refused_before_persistence(store):
    record = store.commit(envelope(metadata={}))["id"]
    with pytest.raises(EvidenceError):
        outbox(store, evidence=iter([record] * 101))
    assert store.db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0


def test_r015_receipts_are_subject_bound_and_not_recounted(store):
    box, identifier, _, _ = outbox(store)
    assert deliver_once(box, policy=DeliveryPolicy(True, "local:review", OWNER),
                        sink=lambda _: {"sent": True}, at=AT)["delivered"]
    outcomes = OutcomeLog(store, outbox=box, owner_principal=OWNER)
    with pytest.raises(EvidenceError, match="exact artifact"):
        outcomes.record(subject_kind="lesson", subject_id="review@1", kind="host_receipt",
                        valence="success", note="unrelated lesson", actor="hermes", evidence=[identifier])
    for note in ("first report", "same receipt again"):
        outcomes.record(subject_kind="artifact", subject_id=identifier, kind="host_receipt",
                        valence="success", note=note, actor="hermes", evidence=[identifier])
    assert outcomes.tally("artifact", identifier)["support"] == 1


@pytest.mark.parametrize("method", ["release", "retry", "partial", "uncertain", "mark_running"])
def test_r020_all_worker_mutations_reject_replaced_claims(store, method):
    queue, identifier = job_queue(store)
    stale = queue.claim(worker="old")
    queue.release(stale)
    fresh = queue.claim(worker="new")
    args = {"retry": {"error": "stale"}, "partial": {"covered": 0, "reason": "stale"},
            "uncertain": {"reason": "stale"}}.get(method, {})
    with pytest.raises(EvidenceError):
        getattr(queue, method)(stale, **args)
    assert queue.get(identifier).lease == fresh.lease and queue.get(identifier).state == "leased"


def test_r030_independent_fence_survives_two_different_restores(store, tmp_path):
    from hermes_memory.lifecycle.recovery import Recovery
    from hermes_memory.lifecycle.snapshots import Snapshots
    snapshots = Snapshots(store, directory=tmp_path / "snapshots")
    early = snapshots.create(actor=OWNER, reason="before record")["snapshot"]
    record = store.commit(envelope(metadata={}, attachments=[
        {"filename": "review.txt", "mime": "text/plain", "data": b"synthetic erased"}]))["id"]
    later = snapshots.create(actor=OWNER, reason="with record")["snapshot"]
    erase = ErasureManager(store, owner_principal=OWNER)
    preview = erase.preview(record_ids=[record], actor=OWNER, reason="review")
    erase.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"], actor=OWNER)
    recovery = Recovery(store, snapshots=snapshots, owner_principal=OWNER)
    recovery.restore(early.id, actor=OWNER)
    assert store.get(record) is None
    recovery.restore(later.id, actor=OWNER)
    assert store.get(record) is None
    assert store.db.execute("SELECT count(*) FROM blob_chunks").fetchone()[0] == 0


@pytest.mark.parametrize("metadata_kind", ["mention", "recipient", "author"])
def test_r032_sender_predicate_distinguishes_author_from_participant(store, metadata_kind):
    from hermes_memory.prospective.predicates import SATISFIED, evaluate
    identity = IdentityStore(store)
    account = identity.account("email", "review@review.invalid")
    metadata = {"account_ids": [account]}
    metadata.update({"mention": {"note": "review@review.invalid"},
                     "recipient": {"to": [{"address": "review@review.invalid"}]},
                     "author": {"from": {"address": "review@review.invalid"}}}[metadata_kind])
    store.commit(envelope(metadata=metadata))
    verdict = evaluate(store.db, "new_message_from", {"account_id": account,
                       "since": "2020-01-01T00:00:00+00:00"}, now_iso=MOMENT)
    assert (verdict.state == SATISFIED) == (metadata_kind == "author")


def test_r030_v14_migration_recovers_already_orphaned_owner_intent(tmp_path):
    from hermes_memory.ids import now
    from hermes_memory.storage.evidence import EvidenceStore
    from hermes_memory.storage.migrations import MIGRATIONS, connect
    path = tmp_path / "legacy.db"
    db = connect(path)
    try:
        db.execute("CREATE TABLE schema_migrations(name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        for migration in MIGRATIONS[:14]:
            for statement in migration.statements:
                db.execute(statement)
            db.execute("INSERT INTO schema_migrations VALUES(?,?)", (migration.name, now()))
        item = envelope(metadata={})
        record = prepare_envelope(item).id
        db.execute("INSERT INTO erasure_ledger(id,source,requested_at,requested_by,requester_kind,"
                   "reason,preview,preview_digest,state,confirmed_at,confirmed_by,epoch) "
                   "VALUES('legacy','gmail',?,?,'owner','review',?,'synthetic','complete',?,?,1)",
                   (now(), OWNER, json.dumps({"records": [record]}), now(), OWNER))
        db.execute("COMMIT")
    finally:
        db.close()
    with EvidenceStore(path) as upgraded:
        assert upgraded.db.execute("SELECT record_id FROM erasure_fences").fetchone()[0] == record
        upgraded.commit(item)
        assert not upgraded.live_and_visible(record)


def test_r020_restore_fences_a_pre_restore_worker(store, tmp_path):
    from hermes_memory.lifecycle.recovery import Recovery
    from hermes_memory.lifecycle.snapshots import Snapshots
    queue, identifier = job_queue(store)
    stale = queue.claim(worker="old")
    snapshots = Snapshots(store, directory=tmp_path / "snapshots")
    made = snapshots.create(actor=OWNER, reason="with leased job")["snapshot"]
    Recovery(store, snapshots=snapshots, owner_principal=OWNER).restore(made.id, actor=OWNER)
    with pytest.raises(EvidenceError):
        queue.complete(stale, covered=list(stale.inputs))
    assert queue.get(identifier).state != "succeeded"


def test_r031_current_measurement_still_satisfies_threshold(store):
    from hermes_memory.prospective.predicates import SATISFIED, evaluate
    record = store.commit(envelope(text="synthetic measurement 100 kg", metadata={}))["id"]
    claims = AssertionStore(store, owner_principal=OWNER)
    claims.propose(subject="review-device", predicate="weight", value="100", kind="measurement",
                   unit="kg", evidence_kind="explicit_statement", record_id=record,
                   quote="100 kg", proposed_by="agent", valid_from="2026-09-01T00:00:00+00:00")
    verdict = evaluate(store.db, "measured_threshold", {"subject": "review-device", "predicate": "weight",
                       "operator": ">", "value": 50, "unit": "kg"}, now_iso=MOMENT)
    assert verdict.state == SATISFIED


def test_r019_equivalent_citation_order_is_idempotent(store):
    first = store.commit(envelope(source_id="first", metadata={}))["id"]
    second = store.commit(envelope(source_id="second", metadata={}))["id"]
    summaries = SummaryStore(store, owner_principal=OWNER)
    arguments = dict(scope="project:review", kind="day", title="Review", body="synthetic",
                     processor_fingerprint="review")
    made = summaries.publish(**arguments, citations=[{"record_id": first}, {"record_id": second}])
    replay = summaries.publish(**arguments, citations=[{"record_id": second}, {"record_id": first}])
    assert replay["id"] == made["id"] and not replay["published"]


def test_r001_cached_access_expires_with_identity_interval(store, monkeypatch):
    monkeypatch.setattr("hermes_memory.storage.identity.now", lambda: "2026-09-01T00:00:00+00:00")
    identity = IdentityStore(store, owner_principal=OWNER)
    a, b = (identity.account("email", name + "@review.invalid") for name in ("a", "b"))
    record = store.commit(envelope(text="synthetic secret", metadata={"account_ids": [b]}))["id"]
    proposed = identity.propose(account_a=a, account_b=b, rule="explicit-alias-declared",
                                basis="synthetic", evidence=[record], proposed_by="agent")
    identity.confirm(candidate_id=proposed["candidate_id"], actor=OWNER, reason="review",
                     valid_from="2026-08-01T00:00:00+00:00",
                     valid_until="2026-09-02T00:00:00+00:00")
    broker = ContextBroker(store, identity=identity, account_id=a)
    try:
        assert broker.assemble("synthetic", include_derived=False).items
        monkeypatch.setattr("hermes_memory.storage.identity.now", lambda: "2026-09-03T00:00:00+00:00")
        assert not broker.assemble("synthetic", include_derived=False).items
    finally:
        broker.close()
