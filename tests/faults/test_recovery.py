"""C4 snapshots and recovery: a rollback that cannot undo a decision.

The items these pin are the ones the plan named — restore, restart and the crash
window between them. A snapshot is only worth having if it can be proved whole, and
a restore is only safe if the forgetting that happened after it stays done.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from conftest import envelope
from hermes_memory.lifecycle.erasure import PENDING, ErasureManager
from hermes_memory.backend.document_map import DocumentMap
from hermes_memory.lifecycle.recovery import LEDGER_TABLES, Recovery
from hermes_memory.lifecycle.reset import ResetController
from hermes_memory.lifecycle.snapshots import Snapshots
from hermes_memory.processing.jobs import QUARANTINED, UNCERTAIN, JobQueue
from hermes_memory.processing.routes import Route
from hermes_memory.proactive.outbox import Outbox
from hermes_memory.proactive.policy import POLICY_VERSION, AttentionPolicy
from hermes_memory.prospective.due_events import CLAIMED, DueEventLog
from hermes_memory.prospective.goals import GoalStore
from hermes_memory.storage.evidence import EvidenceError, EvidenceStore, journal
from hermes_memory.storage.lineage import Lineage
from hermes_memory.storage.migrations import MIGRATIONS

OWNER = "owner-principal"
AGENT = "hermes-agent"
UTC = "UTC"
MORNING = "2026-09-15T09:00:00+00:00"
# The epoch form of MORNING: leases lapse on monotonic seconds, and the tests have
# to say the same instant the wall-clock strings do.
EPOCH = 1789462800.0
ROUTE = Route("retain", "remote-9b", "chat", "http://127.0.0.1:8080/v1", "cred",
              "freshness", 2048)


@pytest.fixture()
def snapshots(store, tmp_path):
    return Snapshots(store, directory=tmp_path / "snapshots")


@pytest.fixture()
def recovery(store, snapshots):
    return Recovery(store, snapshots=snapshots, owner_principal=OWNER)


@pytest.fixture()
def stack(store):
    """The live-work objects a startup sweep has to look at."""
    policy = AttentionPolicy(store, owner_principal=OWNER)
    policy.configure(actor=OWNER, timezone_name=UTC, quiet_from="22:00",
                     quiet_until="06:00", max_immediate_per_day=10,
                     cooldown_minutes=0, shadow=False)
    events = DueEventLog(store, clock=lambda: EPOCH)
    goals = GoalStore(store, events=events, owner_principal=OWNER)
    outbox = Outbox(store, policy=policy, owner_principal=OWNER, clock=lambda: EPOCH,
                    goals=goals, events=events)
    return {"policy": policy, "events": events, "goals": goals, "outbox": outbox,
            "jobs": JobQueue(store, clock=lambda: EPOCH)}


@pytest.fixture()
def wired(store, snapshots, stack):
    return Recovery(store, snapshots=snapshots, owner_principal=OWNER,
                    events=stack["events"], outbox=stack["outbox"],
                    jobs=stack["jobs"])


@pytest.fixture()
def forget(store):
    manager = ErasureManager(store, owner_principal=OWNER)

    def _forget(record_id, *, reason="mentioned in error", actor=OWNER):
        preview = manager.preview(record_ids=[record_id], actor=actor, reason=reason)
        return manager.confirm(intent_id=preview["intent_id"],
                               preview_digest=preview["preview_digest"], actor=actor)
    return _forget


def committed(store, **overrides):
    return store.commit(envelope(**overrides))["id"]


def scheduled(stack, store):
    """One owner-approved obligation with a live claim on it."""
    stack["goals"].propose(title="Send the invoice", statement="Client waiting.",
                           timezone_name=UTC, due="2026-09-16T09:00", proposed_by=OWNER,
                           proposed_kind="owner")
    event_id = store.db.execute("SELECT id FROM due_events LIMIT 1").fetchone()[0]
    return event_id


def stranded_outbox(store, decision_id, *, at=EPOCH):
    """An artifact someone took responsibility for and never came back with."""
    artifact_id = "outb_stranded"
    store.db.execute(
        "INSERT INTO outbox(id, decision_id, kind, topic, recipient, payload, "
        "payload_digest, evidence, policy_version, state, attempts, lease_token, "
        "lease_until, held_by, created_at, updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (artifact_id, decision_id, "notify_owner", "general", OWNER, "Do the thing.",
         "0" * 64, "[]", POLICY_VERSION, "leased", 1, "tok-1", at, "worker-a",
         "2026-09-15T09:00:00+00:00", "2026-09-15T09:00:01+00:00"))
    return artifact_id


def recorded_decision(stack, store):
    decision = stack["policy"].decide(topic="general", at=MORNING)
    goal_id = scheduled(stack, store)
    claim = stack["events"].claim(goal_id, holder="worker-a", at=EPOCH)
    intent = stack["events"].ack(event_id=goal_id, token=claim.token,
                                 decision="notify_owner",
                                 policy_version=POLICY_VERSION)["intent"]
    return stack["policy"].record(decision, intent_id=intent, goal_id=goal_id,
                                  revision=1)["id"]


def restate(item):
    """Re-record a snapshot's digest, so a file edit cannot be caught by the checksum."""
    payload = json.loads(item.manifest.read_text(encoding="utf-8"))
    payload["checksum"] = hashlib.sha256(item.database.read_bytes()).hexdigest()
    payload["size_bytes"] = item.database.stat().st_size
    item.manifest.write_text(json.dumps(payload, sort_keys=True, indent=1),
                            encoding="utf-8")


def retamper(item, *statements):
    connection = sqlite3.connect(item.database)
    try:
        for statement in statements:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()
    restate(item)

# -- taking a snapshot -------------------------------------------------------

def test_a_snapshot_records_which_moment_it_copied(store, snapshots):
    for index in (1, 2):
        committed(store, source_id=f"msg-{index}")
    made = snapshots.create(reason="before the tidy", actor=OWNER)["snapshot"]

    assert made.epoch == store.epoch()
    assert made.records == 2
    assert made.tombstones == 0
    assert made.seq == int(store.db.execute(
        "SELECT COALESCE(max(seq), 0) FROM change_journal").fetchone()[0])
    assert made.database.exists()


def test_a_snapshot_is_one_file_that_is_entirely_itself(store, snapshots):
    """A digest of a WAL database describes half a database: the rest is in a sidecar.

    A snapshot has one writer and no concurrent reader, so it is sealed in rollback
    mode — the file alone is the whole store, and the manifest's digest can be a
    statement about it.
    """
    committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]

    assert sorted(path.name for path in made.path.iterdir()) == sorted(
        ["canonical.db", made.manifest.name])
    probe = sqlite3.connect(made.database)
    try:
        assert str(probe.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "delete"
    finally:
        probe.close()


def test_a_snapshot_verifies_itself(store, snapshots):
    committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    checked = snapshots.verify(made.id)

    assert checked["ok"] is True
    assert checked["problems"] == []
    assert checked["records"] == 1
    assert checked["schema"] == len(MIGRATIONS)


def test_a_snapshot_is_a_whole_store_that_another_database_can_adopt(tmp_path, store,
                                                                    snapshots):
    committed(store, source_id="msg-7", text="A fact only this store knows.")
    made = snapshots.create(reason="hand it over", actor=OWNER)["snapshot"]

    with EvidenceStore(tmp_path / "other.db") as other:
        into = Snapshots(other, directory=snapshots.root)
        result = Recovery(other, snapshots=into, owner_principal=OWNER).restore(
            made.id, actor=OWNER)

        assert result["restored"] == made.id
        assert [item.id for item in other.search("only this store knows")]


def test_the_snapshot_files_are_owner_only(store, snapshots):
    committed(store, text="Private matter.")
    made = snapshots.create(reason="private copy", actor=OWNER)["snapshot"]

    assert made.path.stat().st_mode & 0o077 == 0
    assert made.database.stat().st_mode & 0o077 == 0
    assert made.manifest.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("reason", ["", "   ", "x" * 301])
def test_a_snapshot_needs_a_sane_reason(store, snapshots, reason):
    committed(store)
    with pytest.raises(EvidenceError, match="reason"):
        snapshots.create(reason=reason, actor=OWNER)


@pytest.mark.parametrize("actor", ["", "x" * 121])
def test_a_snapshot_records_who_asked(store, snapshots, actor):
    committed(store)
    with pytest.raises(EvidenceError, match="actor"):
        snapshots.create(reason="baseline", actor=actor)


def test_taking_a_snapshot_leaves_the_live_store_alone(store, snapshots):
    record = committed(store)
    before = [dict(row) for row in store.db.execute("SELECT * FROM records")]
    snapshots.create(reason="baseline", actor=OWNER)

    assert [dict(row) for row in store.db.execute("SELECT * FROM records")] == before
    assert store.get(record) is not None


# -- proving a snapshot is the one it claims to be ---------------------------

def test_appended_bytes_are_not_a_snapshot(store, snapshots):
    committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    with made.database.open("ab") as handle:
        handle.write(b"junk")

    checked = snapshots.verify(made.id)
    assert checked["ok"] is False
    assert "altered" in checked["problems"][0]


def test_a_row_rewritten_inside_the_snapshot_is_still_caught(store, snapshots):
    """The digest is one check of four: a matching file can still be a different moment."""
    committed(store, source_id="msg-1")
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    retamper(made, "UPDATE memory_epoch SET value=9 WHERE id=1")

    checked = snapshots.verify(made.id)
    assert checked["ok"] is False
    assert any("epoch is 9" in problem for problem in checked["problems"])


def test_a_missing_database_is_reported_not_raised(store, snapshots):
    committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    made.database.unlink()

    checked = snapshots.verify(made.id)
    assert checked["ok"] is False
    assert checked["problems"] == ["the snapshot database file is missing"]


def test_an_unreadable_file_is_described_as_unreadable(store, snapshots):
    committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    made.database.write_bytes(b"not a database at all, this")
    restate(made)

    checked = snapshots.verify(made.id)
    assert checked["ok"] is False
    assert any("unreadable" in problem for problem in checked["problems"])


def test_an_older_schema_is_flagged_and_still_restorable(store, snapshots, recovery):
    committed(store, source_id="msg-1")
    made = snapshots.create(reason="older build", actor=OWNER)["snapshot"]
    retamper(made,
             "DROP TABLE IF EXISTS evaluation_cases",
             "DROP TABLE IF EXISTS lessons",
             "DROP TABLE IF EXISTS evaluations",
             "DROP TABLE IF EXISTS outcomes",
             "DELETE FROM schema_migrations WHERE name='0011_learning'")

    checked = snapshots.verify(made.id)
    assert checked["ok"] is True and checked["problems"] == []
    assert any("migration" in note for note in checked["notes"])

    recovery.restore(made.id, actor=OWNER)
    assert int(store.db.execute(
        "SELECT count(*) FROM schema_migrations").fetchone()[0]) == len(MIGRATIONS)
    assert store.db.execute(
        "SELECT 1 FROM sqlite_master WHERE name='lessons'").fetchone() is not None


def test_restoring_leaves_the_snapshot_itself_untouched(store, snapshots, recovery):
    """The donor is opened read-only: a store connection would rewrite the header."""
    committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]

    recovery.restore(made.id, actor=OWNER)

    assert snapshots.verify(made.id)["ok"] is True


def test_a_dangling_reference_inside_a_snapshot_blocks_the_restore(store, snapshots,
                                                                  recovery):
    """The digest proves the bytes; only sqlite can say whether they mean anything."""
    committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    retamper(made, "INSERT INTO record_dependencies(child_id, parent_id) "
                   "VALUES('rec_ghost', 'rec_also_ghost')")

    checked = snapshots.verify(made.id)
    assert checked["ok"] is False
    assert any("foreign-key" in problem for problem in checked["problems"])
    with pytest.raises(EvidenceError, match="foreign-key"):
        recovery.restore(made.id, actor=OWNER)


def test_an_uninspectable_store_is_reported_rather_than_raised(store, recovery):
    """Integrity is the call an operator makes *because* something looks broken.

    A half-applied migration leaves a store that opens but cannot answer the
    questions this build asks of it; a traceback would report the doctor rather
    than the patient.
    """
    committed(store)
    store.db.execute("DROP TABLE outbox")

    report = recovery.integrity()

    assert report["ok"] is False
    assert any("cannot be inspected" in problem for problem in report["problems"])
    assert report["epoch"] == store.epoch()


# -- the owned directory -----------------------------------------------------

def test_a_directory_without_a_manifest_is_not_a_snapshot(store, snapshots, tmp_path):
    committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    (snapshots.root / "notes").mkdir()

    assert [item.id for item in snapshots.list()] == [made.id]
    with pytest.raises(EvidenceError, match="no snapshot"):
        snapshots.resolve("snap_missing")


def test_a_manifest_that_cannot_be_read_is_not_a_snapshot(store, snapshots):
    committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    made.manifest.write_text("{not json", encoding="utf-8")

    assert snapshots.list() == []
    with pytest.raises(EvidenceError, match="does not verify|no snapshot"):
        snapshots.verify(made.id)


def test_snapshots_are_listed_newest_first(store, snapshots):
    committed(store)
    ids = [snapshots.create(reason=f"take {index}", actor=OWNER)["snapshot"].id
           for index in range(3)]

    assert [item.id for item in snapshots.list()] == list(reversed(ids))


def test_listing_is_bounded(store, snapshots):
    committed(store)
    for index in range(4):
        snapshots.create(reason=f"take {index}", actor=OWNER)

    assert len(snapshots.list(limit=2)) == 2
    assert len(snapshots.list(limit="nonsense")) == 4


def test_prune_keeps_the_newest_and_reports_the_rest(store, snapshots):
    committed(store)
    ids = [snapshots.create(reason=f"take {index}", actor=OWNER)["snapshot"].id
           for index in range(3)]

    assert snapshots.prune(keep=1, actor=OWNER)["removed"] == [ids[1], ids[0]]
    assert [item.id for item in snapshots.list()] == [ids[-1]]


def test_prune_never_removes_the_only_recent_snapshot(store, snapshots):
    committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]

    assert snapshots.prune(keep=0, actor=OWNER)["removed"] == []
    assert snapshots.prune(keep=5, actor=OWNER)["removed"] == []
    assert snapshots.list()[0].id == made.id


def test_prune_refuses_to_delete_anything_outside_the_owned_directory(store, snapshots,
                                                                     tmp_path):
    """A snapshot directory can be a symlink, and a recursive delete follows one."""
    committed(store)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    kept = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    elsewhere = Snapshots(store, directory=outside)
    victim = elsewhere.create(reason="somewhere else", actor=OWNER)["snapshot"]
    (snapshots.root / "19700101T000000.000000Z-link").symlink_to(victim.path)

    with pytest.raises(EvidenceError, match="outside"):
        snapshots.prune(keep=1, actor=OWNER)

    assert victim.database.exists()
    assert kept.database.exists()


def test_snapshots_are_audited_without_being_read(store, snapshots):
    committed(store)
    snapshots.create(reason="first", actor=OWNER)
    snapshots.create(reason="second", actor=OWNER)
    snapshots.prune(keep=1, actor=OWNER)
    actions = {row["action"] for row in store.db.execute("SELECT action FROM audit")}

    assert {"snapshot_create", "snapshot_prune"} <= actions


# -- the unclean start -------------------------------------------------------

def test_a_lapsed_due_event_claim_becomes_uncertain(store, wired, stack):
    event_id = scheduled(stack, store)
    stack["events"].claim(event_id, holder="worker-a", at=EPOCH)

    report = wired.reconcile(at=EPOCH + 3600)

    assert report["uncertain"]["due_events"] == 1
    assert store.db.execute("SELECT state FROM due_events WHERE id=?",
                            (event_id,)).fetchone()[0] == "uncertain"


def test_a_claim_that_has_not_lapsed_is_left_with_its_holder(store, wired, stack):
    event_id = scheduled(stack, store)
    stack["events"].claim(event_id, holder="worker-a", at=EPOCH, lease_s=600)

    assert wired.reconcile(at=EPOCH + 60) == {"uncertain": {}, "stale_leases": []}
    assert store.db.execute("SELECT state, claimed_by FROM due_events WHERE id=?",
                            (event_id,)).fetchone()["state"] == CLAIMED


def test_a_stranded_outbox_handover_becomes_uncertain(store, wired, stack):
    artifact_id = stranded_outbox(store, recorded_decision(stack, store))

    report = wired.reconcile(at=EPOCH + 3600)

    assert report["uncertain"]["outbox"] == 1
    assert store.db.execute("SELECT state, lease_token, held_by FROM outbox WHERE id=?",
                            (artifact_id,)).fetchone()["state"] == "uncertain"


def test_a_lapsed_job_lease_becomes_uncertain_work_not_a_free_retry(store, wired, stack):
    jobs = stack["jobs"]
    job = jobs.enqueue(kind="retain", inputs=["rec_a"], input_revision="1", route=ROUTE,
                       processor_fingerprint="extractor-v3")
    claimed = jobs.claim(worker="worker-a", ttl=10)
    assert claimed.id == job["job_id"]

    report = wired.reconcile(at=EPOCH + 600)

    assert report["uncertain"]["lease_expired"] == 1
    assert jobs.get(job["job_id"]).state == UNCERTAIN


def test_a_job_queued_under_a_superseded_epoch_is_quarantined(store, wired, stack):
    jobs = stack["jobs"]
    job = jobs.enqueue(kind="retain", inputs=["rec_a"], input_revision="1", route=ROUTE,
                       processor_fingerprint="extractor-v3")
    store.db.execute("UPDATE memory_epoch SET value=value+1 WHERE id=1")

    report = wired.reconcile(at=EPOCH)

    assert report["uncertain"]["stale_epoch"] == 1
    assert jobs.get(job["job_id"]).state == QUARANTINED


def test_an_expired_connector_lease_is_released_without_losing_the_cursor(store, wired):
    store.db.execute(
        "INSERT INTO connectors(source, generation, policy_version, lease, lease_until, "
        "holder, cursor, updated_at) VALUES('gmail',1,'v1','old-lease',?,?, 'cursor-77', ?)",
        (EPOCH - 1, "worker-a", "2026-09-15T09:00:00+00:00"))

    released = wired.reconcile(at=EPOCH)["stale_leases"]

    assert released == ["gmail:worker-a"]
    row = store.db.execute("SELECT lease, cursor, generation FROM connectors").fetchone()
    assert row["lease"] is None and row["cursor"] == "cursor-77" and row["generation"] == 1


def test_a_live_connector_lease_is_not_stolen(store, wired):
    store.db.execute(
        "INSERT INTO connectors(source, generation, policy_version, lease, lease_until, "
        "holder, updated_at) VALUES('gmail',1,'v1','live',?,?, ?)",
        (EPOCH + 300, "worker-a", "2026-09-15T09:00:00+00:00"))

    assert wired.reconcile(at=EPOCH)["stale_leases"] == []
    assert store.db.execute("SELECT lease FROM connectors").fetchone()[0] == "live"


def test_reconciling_a_quiet_store_says_nothing(store, wired):
    committed(store)

    assert wired.reconcile(at=EPOCH) == {"uncertain": {}, "stale_leases": []}


def test_reconciling_twice_reports_the_damage_once(store, wired, stack):
    event_id = scheduled(stack, store)
    stack["events"].claim(event_id, holder="worker-a", at=EPOCH)

    first = wired.reconcile(at=EPOCH + 3600)
    second = wired.reconcile(at=EPOCH + 3600)

    assert first["uncertain"]["due_events"] == 1
    assert second == {"uncertain": {}, "stale_leases": []}  # nothing left to report


def test_reconcile_does_not_resurrect_a_cancelled_or_handed_off_event(store, wired, stack):
    event_id = scheduled(stack, store)
    claim = stack["events"].claim(event_id, holder="worker-a", at=EPOCH)
    stack["events"].ack(event_id=event_id, token=claim.token, decision="next_turn",
                        policy_version=POLICY_VERSION)

    wired.reconcile(at=EPOCH + 3600)

    assert store.db.execute("SELECT state FROM due_events WHERE id=?",
                            (event_id,)).fetchone()[0] == "handed_off"


# -- integrity ---------------------------------------------------------------

def test_integrity_describes_a_clean_store(store, recovery):
    committed(store)

    report = recovery.integrity()

    assert report["ok"] is True and report["problems"] == []
    assert report["records"] == 1 and report["erasure_pending"] == 0


def test_integrity_counts_work_that_is_still_outstanding(store, recovery, forget):
    record = committed(store, source_id="msg-1")
    DocumentMap(store).begin(record, store.get(record).revision)
    forget(record)
    owed = [(row["kind"], row["reference"]) for row in store.db.execute(
        "SELECT kind, reference FROM erasure_targets WHERE state!='verified'")]
    before = {table: int(store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
              for table in LEDGER_TABLES}

    report = recovery.integrity()

    assert report["erasure_pending"] == len(owed) == 2
    assert report["ok"] is True
    assert {table: int(store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in LEDGER_TABLES} == before


# -- who may roll the store back --------------------------------------------

def test_a_restore_needs_an_owner_principal_to_exist_at_all(store, snapshots):
    committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    unguarded = Recovery(store, snapshots=snapshots)

    with pytest.raises(EvidenceError, match="HERMES_MEMORY_OWNER_PRINCIPAL"):
        unguarded.restore(made.id, actor=OWNER)


def test_an_agent_cannot_roll_the_store_back(store, snapshots, recovery):
    committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]

    with pytest.raises(EvidenceError, match="owner principal"):
        recovery.restore(made.id, actor=AGENT)


def test_an_unverifiable_snapshot_is_not_restored(store, snapshots, recovery):
    record = committed(store)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    with made.database.open("ab") as handle:
        handle.write(b"junk")

    with pytest.raises(EvidenceError, match="does not verify"):
        recovery.restore(made.id, actor=OWNER)

    assert store.get(record) is not None


def test_an_unknown_snapshot_cannot_be_named(store, recovery):
    with pytest.raises(EvidenceError, match="no snapshot"):
        recovery.restore("snap_never_taken", actor=OWNER)


# -- what a restore must not undo -------------------------------------------

def test_evidence_written_after_the_snapshot_is_gone(store, snapshots, recovery):
    kept = committed(store, source_id="msg-1", text="Written before.")
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    dropped = committed(store, source_id="msg-2", text="Written afterwards.")

    recovery.restore(made.id, actor=OWNER)

    assert store.get(kept) is not None
    assert store.get(dropped) is None


def test_a_forgotten_record_stays_forgotten_through_a_rollback(store, snapshots,
                                                              recovery, forget):
    secret = committed(store, source_id="msg-1", text="A diagnosis nobody else needs.")
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    forget(secret)

    result = recovery.restore(made.id, actor=OWNER)

    assert result["reapplied"] == 1
    assert store.get(secret) is None
    assert store.search("diagnosis") == []
    assert Lineage(store).explain(secret)["state"] == "erased"
    # Hiding the row is not the same as the words being gone: the restored copy
    # carries the forgotten text in its index until it is taken out again.
    assert int(store.db.execute("SELECT count(*) FROM record_fts WHERE id=?",
                                (secret,)).fetchone()[0]) == 0
    # A consumer that filters on visibility rather than the delete flag would serve
    # this row straight out of the restored copy, so the fence says both.
    assert int(store.db.execute("SELECT hidden FROM record_visibility WHERE record_id=?",
                                (secret,)).fetchone()[0]) == 1


def test_an_erasure_of_evidence_the_snapshot_never_held_completes_anyway(store, snapshots,
                                                                        recovery, forget):
    made = snapshots.create(reason="before the arrival", actor=OWNER)["snapshot"]
    late = committed(store, source_id="msg-late",
                     text="A diagnosis written after the snapshot was taken.")
    intent = forget(late)["intent_id"]

    result = recovery.restore(made.id, actor=OWNER)

    # There is no row in this copy to re-hide: the evidence arrived after the snapshot and
    # left again before anything was copied. Refusing to restore over the dangling marker —
    # which is what the foreign key did first — turns a backup into a restore that cannot
    # happen, so the decision and its obligations travel and the marker does not.
    assert result["tombstones_without_a_record"] == 1
    assert store.get(late) is None
    assert store.db.execute("SELECT 1 FROM erasure_ledger WHERE id=?",
                            (intent,)).fetchone() is not None
    assert "does not contain" in result["note"]


def test_the_erasure_ledger_survives_the_rollback(store, snapshots, recovery, forget):
    secret = committed(store, source_id="msg-1")
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    intent = forget(secret)["intent_id"]

    recovery.restore(made.id, actor=OWNER)

    assert store.db.execute(
        "SELECT state FROM erasure_ledger WHERE id=?", (intent,)).fetchone()[0] == "complete"
    assert int(store.db.execute("SELECT count(*) FROM tombstones").fetchone()[0]) == 1


def test_obligations_outstanding_at_the_time_are_still_owed_afterwards(
        store, snapshots, recovery, forget):
    record = committed(store, source_id="msg-1")
    DocumentMap(store).begin(record, store.get(record).revision)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    intent = forget(record)["intent_id"]
    assert store.db.execute(
        "SELECT state FROM erasure_ledger WHERE id=?", (intent,)).fetchone()[0] == PENDING

    owed = sorted((row["kind"], row["reference"]) for row in store.db.execute(
        "SELECT kind, reference FROM erasure_targets WHERE intent_id=?", (intent,)))

    recovery.restore(made.id, actor=OWNER)

    assert sorted((row["kind"], row["reference"]) for row in store.db.execute(
        "SELECT kind, reference FROM erasure_targets WHERE state!='verified'")) == owed
    assert store.db.execute(
        "SELECT state FROM erasure_ledger WHERE id=?", (intent,)).fetchone()[0] == PENDING


def test_a_snapshot_with_forgetting_of_its_own_carries_both_halves(store, snapshots,
                                                                  recovery, forget):
    """The erasures the snapshot already holds and the ones since must both survive."""
    first = committed(store, source_id="msg-1", text="First thing forgotten.")
    second = committed(store, source_id="msg-2", text="Second thing forgotten.")
    before = forget(first)["intent_id"]
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    after = forget(second)["intent_id"]

    recovery.restore(made.id, actor=OWNER)

    states = {row["id"]: row["state"] for row in
              store.db.execute("SELECT id, state FROM erasure_ledger")}
    assert states == {before: "complete", after: "complete"}
    assert store.get(first) is None and store.get(second) is None
    assert store.search("forgotten") == []


def test_the_epoch_never_goes_backwards(store, snapshots, recovery):
    committed(store, source_id="msg-1")
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    reset = ResetController(store, owner_principal=OWNER)
    preview = reset.preview(actor=OWNER, reason="starting over")
    reset.confirm(intent_id=preview["intent_id"],
                  preview_digest=preview["preview_digest"], actor=OWNER)
    assert store.epoch() == 2

    recovery.restore(made.id, actor=OWNER)

    assert store.epoch() == 2
    assert recovery.integrity()["records"] == 0


def test_a_consumer_checkpoint_is_never_walked_backwards(store, snapshots, recovery):
    """A consumer that has read past the snapshot's end must not re-emit it as new."""
    store.db.execute("INSERT INTO consumer_checkpoints(consumer, seq, updated_at) "
                     "VALUES('broker', 1, ?)", ("2026-09-15T09:00:00+00:00",))
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    committed(store, source_id="msg-9")
    store.db.execute("UPDATE consumer_checkpoints SET seq=(SELECT max(seq) FROM "
                     "change_journal) WHERE consumer='broker'")
    ahead = int(store.db.execute(
        "SELECT seq FROM consumer_checkpoints").fetchone()[0])

    recovery.restore(made.id, actor=OWNER)

    assert int(store.db.execute(
        "SELECT seq FROM consumer_checkpoints WHERE consumer='broker'").fetchone()[0]) == ahead


def test_a_checkpoint_that_advanced_after_the_carry_is_not_walked_back(store, snapshots,
                                                                      recovery):
    """Carry and apply are separate transactions: a consumer may have read on."""
    store.db.execute("INSERT INTO consumer_checkpoints(consumer, seq, updated_at) "
                     "VALUES('broker', 3, ?)", ("2026-09-15T09:00:00+00:00",))
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    carried = recovery.carry()
    store.db.execute("UPDATE consumer_checkpoints SET seq=9 WHERE consumer='broker'")

    recovery.apply(carried)

    assert int(store.db.execute(
        "SELECT seq FROM consumer_checkpoints WHERE consumer='broker'").fetchone()[0]) == 9


def test_restoring_drops_projections_into_a_bank_that_is_gone(store, snapshots, recovery):
    record = committed(store, source_id="msg-1")
    DocumentMap(store, bank_id="old-bank").begin(record, store.get(record).revision)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]

    result = recovery.restore(made.id, actor=OWNER, bank_id="new-bank")

    assert result["stale_projections_dropped"] == 1
    assert store.db.execute(
        "SELECT count(*) FROM backend_documents WHERE bank_id='old-bank'").fetchone()[0] == 0


def test_a_restore_that_names_no_bank_keeps_the_document_map(store, snapshots, recovery):
    record = committed(store, source_id="msg-1")
    DocumentMap(store).begin(record, store.get(record).revision)
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]

    assert recovery.restore(made.id, actor=OWNER)["stale_projections_dropped"] == 0
    assert store.db.execute("SELECT count(*) FROM backend_documents").fetchone()[0] == 1


def test_a_content_clash_is_refused_rather_than_guessed(store, snapshots, recovery,
                                                        forget):
    """An id carrying different bytes than its tombstone records is damage, not a decision."""
    secret = committed(store, source_id="msg-1", text="A diagnosis nobody else needs.")
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    forget(secret)
    after = committed(store, source_id="msg-2", text="Written afterwards.")
    retamper(made, "UPDATE records SET fingerprint='changed' WHERE id='" + secret + "'")

    with pytest.raises(EvidenceError, match="refusing to guess"):
        recovery.restore(made.id, actor=OWNER)

    assert store.get(after) is not None
    assert store.get(secret) is None
    assert store.db.execute(
        "SELECT count(*) FROM erasure_ledger").fetchone()[0] == 1


def test_managers_built_before_a_restore_keep_working(store, snapshots, recovery):
    """The restore writes through the live connection; nothing may hold a dead handle."""
    record = committed(store, source_id="msg-1")
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    lineage, erasure = Lineage(store), ErasureManager(store, owner_principal=OWNER)
    committed(store, source_id="msg-2")

    recovery.restore(made.id, actor=OWNER)

    assert lineage.explain(record)["state"] == "visible"
    assert [item.id for item in store.search("Thursday")] == [record]
    preview = erasure.preview(record_ids=[record], actor=OWNER, reason="after the rollback")
    assert preview["records"] == [record]


def test_a_restore_can_be_done_twice(store, snapshots, recovery, forget):
    secret = committed(store, source_id="msg-1", text="A diagnosis nobody else needs.")
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    forget(secret)
    committed(store, source_id="msg-2")

    first = recovery.restore(made.id, actor=OWNER)
    second = recovery.restore(made.id, actor=OWNER)

    # Restoring twice must not halve the forgetting: the snapshot brings the row
    # back and the carried tombstone takes it away again, every time.
    assert first["reapplied"] == second["reapplied"] == 1
    assert store.get(secret) is None


def test_the_guard_holds_the_state_from_before_the_rollback(store, snapshots, recovery):
    committed(store, source_id="msg-1")
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    after = committed(store, source_id="msg-2", text="Written afterwards.")

    result = recovery.restore(made.id, actor=OWNER)

    guard = sqlite3.connect(result["pre_restore_backup"])
    try:
        assert guard.execute("SELECT count(*) FROM records WHERE id=?",
                             (after,)).fetchone()[0] == 1
    finally:
        guard.close()
    assert store.get(after) is None


def test_a_restore_is_audited_with_the_ledger_it_carried(store, snapshots, recovery,
                                                         forget):
    secret = committed(store, source_id="msg-1")
    made = snapshots.create(reason="baseline", actor=OWNER)["snapshot"]
    forget(secret)

    recovery.restore(made.id, actor=OWNER)

    row = store.db.execute(
        "SELECT object_id, metadata FROM audit WHERE action='restore_complete'").fetchone()
    assert row["object_id"] == made.id
    assert json.loads(row["metadata"])["reapplied"] == 1
