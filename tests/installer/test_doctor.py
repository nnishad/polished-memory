"""C14 doctor: it must find the fault, name the fix, and never become the fault.

Every default-run assertion here carries a tripwire backend that fails the test if it
is touched, because a diagnostic that quietly sends private evidence somewhere is
worse than no diagnostic at all.
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_memory.backend.hindsight_client import HindsightError
from hermes_memory.config import load_settings
from hermes_memory.ids import document_id_is_ambiguous
from hermes_memory.operations.doctor import (FAIL, OK, PROBE_DOCUMENT_ID, PROBE_QUERY,
                                            PROBE_SETTLE_READS, PROBE_SETTLE_WAIT_S,
                                            PROBE_TEXT, WARN, Doctor, Finding)
from hermes_memory.storage.evidence import EvidenceStore

APPROVED_LAN = "192.168.68.65"
SECRET = "sk-supersecretvalue1"          # a key-shaped string, as long as a real one
STAMP = "2026-09-25T12:00:00+00:00"


class Tripwire:
    """A backend that fails the test by being reached at all."""

    def __init__(self):
        self.reached: list[str] = []

    def __getattr__(self, name):
        if name.startswith("__") or name == "reached":
            raise AttributeError(name)

        def touched(*args, **kwargs):
            self.reached.append(name)
            raise AssertionError(f"the doctor reached the backend via {name}")
        return touched


class FakeBackend:
    """The probe surface, with the answers a real backend would give.

    It obeys what it is asked rather than answering one fixed thing: a probe whose fake
    returns results whatever the door did would let the door stop looking entirely and
    still report a healthy backend. Causality is part of that — nothing is findable before
    anything has been derived — because the fault the door has to tell apart is exactly
    between those two stages.
    """

    def __init__(self, *, bank_id="probe-bank", healthy=True, results=(1, 2),
                 fail_on=None, units_after=0, findable=True):
        self.bank_id = bank_id
        self.healthy = healthy
        self.results = tuple(results)
        self.fail_on = set(fail_on or ())
        self.units_after = units_after      # state reads that still report nothing derived
        self.findable = findable
        self.derived = False
        self.calls: list[str] = []

    def health(self):
        self.calls.append("health")
        if not self.healthy or "health" in self.fail_on:
            raise RuntimeError("connection refused")
        return {"version": "0.5.13"}

    def retain(self, **kwargs):
        self.calls.append("retain")
        if "retain" in self.fail_on:
            raise RuntimeError("the backend is read-only")
        # The real client refuses such an id before it sends anything. A fake that took any
        # id at all let the probe keep a name that could never work outside these tests.
        if document_id_is_ambiguous(kwargs["document_id"]):
            raise HindsightError(f"ambiguous document id {kwargs['document_id']!r}")
        self.derived = False
        return {"document_id": kwargs["document_id"]}

    def document_state(self, document_id):
        """The deterministic reading: does the backend have a unit for this document yet."""
        self.calls.append("state")
        if "state" in self.fail_on:
            raise RuntimeError("the list endpoint is not answering")
        self.derived = self.calls.count("state") > self.units_after
        return {"document_id": document_id,
                "state": "present" if self.derived else "absent",
                "count": 1 if self.derived else 0}

    def recall(self, query, **kwargs):
        self.calls.append("recall")
        if "recall" in self.fail_on:
            raise RuntimeError("no index yet")
        if not self.derived or not self.findable:
            return SimpleNamespace(results=())   # nothing derived, or nothing searchable
        return SimpleNamespace(results=self.results)

    def delete_document(self, document_id):
        self.calls.append("delete")
        if "delete" in self.fail_on:
            raise RuntimeError("the bank is read-only")
        self.derived = False
        return {"deleted": True}


def configured(**values):
    """The attribute surface the doctor actually reads off Settings."""
    base = dict(capture_only=False, inference_enabled=True, hindsight_url=None,
                hindsight_api_key_env=None, background_budget_tokens=50_000,
                owner_principal="owner:judge", gate_token="t",
                allowed_inference_hosts=frozenset(), data_dir=None, db_path=None,
                blob_dir=None, home=Path(tempfile.mkdtemp(prefix="hm-doctor-")))
    base.update(values)
    return SimpleNamespace(**base)


def settings_at(tmp_path, monkeypatch, values=None):
    """Write the owned env file the way an operator would, and load it for real."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "hermes-memory.env").write_text(
        "\n".join(f"HERMES_MEMORY_{key}={value}"
                   for key, value in (values or {}).items()) + "\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    return load_settings()


@pytest.fixture()
def installed(tmp_path, monkeypatch):
    """Real Settings, and a real store at the path those Settings name."""
    settings = settings_at(tmp_path, monkeypatch, {
        "OWNER_PRINCIPAL": "owner:judge", "GATE_TOKEN": "gate-token-value",
        "HINDSIGHT_URL": f"http://{APPROVED_LAN}:8080/v1",
        "ALLOWED_INFERENCE_HOSTS": APPROVED_LAN, "INFERENCE_ENABLED": "true",
        "BACKGROUND_BUDGET_TOKENS": "50000",
    })
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(settings.data_dir, 0o700)
    settings.blob_dir.mkdir(parents=True, exist_ok=True)
    with EvidenceStore(settings.db_path) as store:
        yield store, settings


def insert(store, table, **columns):
    names = ", ".join(columns)
    store.db.execute(f"INSERT INTO {table}({names}) "
                     f"VALUES({','.join('?' * len(columns))})", list(columns.values()))


def evidence(store, text, source_id="m-1"):
    return store.commit({"source": "gmail", "source_id": source_id, "revision": "1",
                         "kind": "email", "text": text, "observed_at": STAMP,
                         "occurred_at": None, "occurred_precision": "unknown",
                         "metadata": {}})["id"]


# -- the default run ---------------------------------------------------------

def test_a_bare_store_is_described_rather_than_condemned(store):
    tripwire = Tripwire()
    report = Doctor(store, backend=tripwire).examine()
    assert [item["check"] for item in report["findings"]] == [
        "layout", "database", "schema", "configuration", "coverage", "queue",
        "background", "provenance", "lineage", "erasure", "delivery", "gate", "credentials",
        "leases", "backend", "release"]
    assert report["probes"] == {"connectivity": False, "synthetic": False}
    assert tripwire.reached == []


def test_a_bare_store_warns_about_the_undetermined_parts_without_failing(store):
    report = Doctor(store, backend=Tripwire()).examine()
    assert report["ok"] is True
    assert report["exit_code"] == 0
    assert report["severity"] == WARN, "no settings supplied is not the same as broken"


def test_reading_status_never_writes(store):
    evidence(store, "The meeting moved to Thursday.")
    tables = [row[0] for row in store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]

    def footprint():
        return {table: store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in tables}

    doctor = Doctor(store, backend=Tripwire())
    before = footprint()
    doctor.examine()
    assert footprint() == before
    assert store.db.in_transaction is False


def test_one_fault_moves_the_exit_code_and_names_the_action(store):
    _job(store, state="uncertain")
    report = Doctor(store, backend=Tripwire()).examine()
    assert report["severity"] == FAIL
    assert report["exit_code"] == 1
    assert report["ok"] is False, "a failing report cannot also claim to be fine"
    assert report["actions"], "a failing report must carry the thing to do about it"


# -- layout, database and schema --------------------------------------------

def test_an_owned_layout_is_reported_once_it_exists(installed):
    store, settings = installed
    finding = Doctor(store, settings=settings, backend=Tripwire()).layout()
    assert finding.severity == OK
    assert finding.evidence["data_dir"] == str(settings.data_dir)


def test_a_missing_data_directory_is_a_failure_with_the_setup_remedy(tmp_path, monkeypatch,
                                                                    store):
    settings = settings_at(tmp_path, monkeypatch)
    finding = Doctor(store, settings=settings, backend=Tripwire()).layout()
    assert finding.severity == FAIL
    assert finding.remedy == "run setup to create the owned data directory"


def test_a_data_directory_the_rest_of_the_machine_can_read_is_a_failure(installed):
    store, settings = installed
    os.chmod(settings.data_dir, 0o755)
    finding = Doctor(store, settings=settings, backend=Tripwire()).layout()
    assert finding.severity == FAIL
    assert "chmod 700" in finding.remedy
    assert stat.S_IMODE(os.stat(settings.data_dir).st_mode) == 0o755


def test_a_store_not_in_wal_mode_cannot_honour_the_durability_claims(store):
    store.db.execute("PRAGMA journal_mode=memory")
    finding = Doctor(store, backend=Tripwire()).database()
    assert finding.severity == FAIL
    assert "not 'wal'" in finding.detail


def test_a_write_ahead_log_that_never_drains_is_mentioned(store, tmp_path, monkeypatch):
    from hermes_memory.operations import doctor as module

    store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    (tmp_path / "canonical.db-wal").write_bytes(b"x" * 4096)
    monkeypatch.setattr(module, "WAL_WARN_BYTES", 1024)
    finding = Doctor(store, settings=configured(db_path=tmp_path / "canonical.db")
                     ).database()
    assert finding.severity == WARN
    assert "checkpoint" in finding.detail


def test_a_schema_this_build_does_not_know_is_refused_before_anything_else(store):
    insert(store, "schema_migrations", name="9999_the_future", applied_at=STAMP)
    finding = Doctor(store, backend=Tripwire()).schema()
    assert finding.severity == FAIL
    assert "refused on purpose" in finding.remedy


def test_outstanding_migrations_are_reported_without_being_applied(store):
    store.db.execute("DELETE FROM schema_migrations WHERE name='0012_source_gaps'")
    finding = Doctor(store, backend=Tripwire()).schema()
    assert finding.severity == WARN
    assert finding.remedy == "run setup or upgrade to apply them"
    assert store.db.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 11


# -- configuration -----------------------------------------------------------

def test_a_full_configuration_is_the_healthy_case(installed):
    store, settings = installed
    finding = Doctor(store, settings=settings, backend=Tripwire()).configuration()
    assert finding.severity == OK
    assert finding.evidence["allowed_hosts"] == [APPROVED_LAN]


def test_inference_enabled_with_no_budget_is_a_contradiction(store):
    finding = Doctor(store, settings=configured(background_budget_tokens=0)
                     ).configuration()
    assert finding.severity == WARN
    assert "no background budget" in finding.detail


def test_an_unnamed_owner_stops_every_owner_only_action(store):
    finding = Doctor(store, settings=configured(owner_principal=None)).configuration()
    assert finding.severity == WARN
    assert "no owner principal" in finding.detail


def test_a_missing_gate_token_is_named(store):
    finding = Doctor(store, settings=configured(gate_token=None)).configuration()
    assert "no gate token" in finding.detail


def test_a_projection_ledger_that_failed_is_a_failure(store):
    record = evidence(store, "The kettle boiled.")
    insert(store, "backend_documents", record_id=record, revision="1",
           backend="hindsight", bank_id="hermes", document_id="doc_1",
           desired_epoch=1, state="failed")
    finding = Doctor(store, settings=configured(hindsight_url="http://x/v1")).backend_ledger()
    assert finding.severity == FAIL
    assert "reconcile" in finding.remedy


def test_a_backend_url_with_inference_off_is_still_worth_mentioning(store):
    finding = Doctor(store, settings=configured(hindsight_url="http://x/v1",
                                                inference_enabled=False)).configuration()
    assert "inference is not enabled" in finding.detail


# -- the pipeline ------------------------------------------------------------

def test_no_source_registered_is_the_one_unconfigured_thing_worth_saying(store):
    finding = Doctor(store, backend=Tripwire()).coverage()
    assert finding.severity == WARN
    assert finding.remedy == "add one with setup or import a fixture"


def test_an_unreadable_source_is_a_failure(store, sync):
    sync.register("gmail", policy_version="local-only")
    store.db.execute("UPDATE connectors SET coverage_state='unreachable' "
                     "WHERE source='gmail'")
    finding = Doctor(store).coverage()
    assert finding.severity == FAIL
    assert finding.evidence["unhealthy"] == ["gmail"]


def test_a_source_working_through_its_backfill_is_not_a_fault(store, sync):
    sync.register("gmail", policy_version="local-only")
    store.db.execute("UPDATE connectors SET coverage_state='partial' WHERE source='gmail'")
    assert Doctor(store).coverage().severity == OK


def test_a_quarantined_job_names_the_work_that_will_not_retry_itself(store):
    _job(store, state="quarantined")
    finding = Doctor(store).queue()
    assert finding.severity == FAIL
    assert "quarantined" in json.dumps(finding.evidence["stuck"])


def test_a_queue_nothing_is_consuming_names_the_command_that_empties_it(store):
    _job(store, state="queued", created_at="2000-01-01T00:00:00+00:00")
    finding = Doctor(store).queue()
    assert finding.severity == WARN
    assert "hermes-memory form" in finding.remedy, "there is no worker to start"
    assert finding.evidence["operator_needed"] is True


def test_capture_only_formation_is_a_supported_shape(store):
    finding = Doctor(store, settings=configured(capture_only=True)).queue()
    assert finding.severity == OK


def test_a_summary_that_cites_vanished_evidence_is_a_failure(store):
    record = evidence(store, "The kettle boiled.")
    insert(store, "summaries", id="sum-1", scope="day:2026-09-25", kind="day",
           title="Boiling", body="The kettle boiled", status="published",
           processor_fingerprint="v1", epoch=1, budget_tokens=10,
           created_at=STAMP, published_at=STAMP)
    insert(store, "derived_citations", artifact_id="sum-1", kind="summary",
           coverage="full", record_id=record, added_at=STAMP)
    store.db.execute("UPDATE records SET deleted=1 WHERE id=?", (record,))
    finding = Doctor(store).provenance()
    assert finding.severity == FAIL
    assert "not support" in finding.remedy


def test_an_erasure_that_is_only_half_done_is_a_failure(store):
    insert(store, "erasure_ledger", id="er-1", source="gmail", requested_at=STAMP,
           requested_by="owner", requester_kind="owner", reason="private",
           preview="{}", preview_digest="d" * 16, state="erasure_pending", epoch=1)
    insert(store, "erasure_targets", intent_id="er-1", kind="backend-document",
           reference="doc_1", state="pending")
    finding = Doctor(store).erasure()
    assert finding.severity == FAIL
    assert finding.evidence["obligations_open"] == 1


def test_an_erasure_awaiting_its_owner_is_not_a_software_failure(store):
    insert(store, "erasure_ledger", id="er-1", source="gmail", requested_at=STAMP,
           requested_by="owner", requester_kind="owner", reason="private",
           preview="{}", preview_digest="d" * 16, state="awaiting_confirmation", epoch=1)
    finding = Doctor(store).erasure()
    assert finding.severity == WARN
    assert "an agent cannot confirm an erasure" in finding.remedy
    assert "--confirm-forgetting" in finding.remedy


def test_an_unproven_send_is_a_failure(store):
    _handoff(store)
    store.db.execute("UPDATE outbox SET state='uncertain'")
    finding = Doctor(store).delivery()
    assert finding.severity == FAIL
    assert "correlate the host receipts" in finding.remedy


def test_artifacts_that_aged_out_unsent_indicate_a_transport_that_stopped(store):
    _handoff(store)
    store.db.execute("UPDATE outbox SET expires_at='2000-01-01T00:00:00+00:00'")
    finding = Doctor(store).delivery()
    assert finding.severity == FAIL
    assert "not claiming the outbox" in finding.remedy


def test_a_proactivity_pause_is_reported_as_a_decision_in_force(store, sync):
    sync.register("gmail", policy_version="local-only")
    sync.pause("gmail", actor="owner", reason="quiet week", policy_version="v1",
               stages=("proactivity",))
    assert Doctor(store).delivery().severity == WARN


def test_a_reservation_that_never_resolved_blocks_a_physical_resource(store, gate):
    reservation = gate.try_acquire(route="retain", holder="worker-1",
                                   resource="remote-9b", priority=1, ttl=60)
    gate.mark_uncertain(reservation, reason="no response")
    finding = Doctor(store).gate()
    assert finding.severity == FAIL
    assert finding.evidence["blocked"] == ["remote-9b"]


def test_an_inference_pause_is_reported_as_a_decision_in_force(store, gate):
    gate.pause(actor="owner", reason="model update")
    assert Doctor(store).gate().severity == WARN


def test_an_operation_nobody_could_charge_points_at_the_revision_contract(store):
    """A null resource is the launcher saying it could not read the task.

    That is a different fault from a stuck device, and the remedy has to say so: the
    operator's next command is the launcher's own check in the backend environment, not a
    reservation that nothing is holding.
    """
    from hermes_memory.backend.worker_launcher import OperationLedger
    from hermes_memory.processing.instance_gate import instance_gate

    config = configured()
    with instance_gate(config) as ledger_gate:
        OperationLedger(ledger_gate.store).record(
            {"_operation_id": "op-7", "operation_type": "refresh", "bank_id": "hermes"},
            worker_id="worker-1", bank_id="hermes", resource=None)
        finding = Doctor(store, settings=config).gate()
    assert finding.severity == FAIL
    assert "worker_launcher --check" in finding.remedy
    assert finding.evidence["operations"]["unattributed"] == 1


def test_a_cancellation_awaiting_an_answer_names_the_command_that_finishes_it(store):
    """An unanswered stop is a queue with names on it, so the remedy names the queue."""
    from hermes_memory.backend.worker_launcher import OperationLedger
    from hermes_memory.processing.instance_gate import instance_gate

    config = configured()
    with instance_gate(config) as ledger_gate:
        ledger = OperationLedger(ledger_gate.store)
        ledger.record({"_operation_id": "op-8", "operation_type": "retain",
                       "bank_id": "hermes"}, worker_id="worker-1", bank_id="hermes",
                      resource="remote-9b")
        ledger.mark("op-8", "running")
        ledger.request_cancellation("op-8", actor="owner:judge")
        finding = Doctor(store, settings=config).gate()
    assert finding.severity == FAIL
    assert "cancel --list" in finding.remedy
    assert finding.evidence["operations"]["cancellations_owed"] == 1


def test_an_edge_to_a_missing_record_fails_the_lineage_check(store):
    """A dependency row whose parent never existed is a write outside the owning transaction."""
    record = evidence(store, "One.")
    forged = "rec_never_existed"
    store.db.execute("PRAGMA foreign_keys=OFF")
    store.db.execute("INSERT INTO record_dependencies(child_id, parent_id) VALUES(?,?)",
                     (record, forged))
    store.db.execute("PRAGMA foreign_keys=ON")
    finding = Doctor(store).lineage()
    assert finding.severity == FAIL
    assert forged in str(finding.evidence["edges"])
    assert "explain" in finding.remedy


def test_a_clean_store_reports_every_dependency_resolving(store):
    evidence(store, "A plain note with no links.")
    finding = Doctor(store).lineage()
    assert finding.severity == OK
    assert finding.evidence["orphans"] == 1, \
        "a leaf is normal, and the report says the count rather than judging it"


def test_the_release_check_vouches_for_the_manifest_that_ships(store, tmp_path, monkeypatch):
    """A release names itself by a digest; the doctor reads the file that carries it."""
    import json

    from hermes_memory.install import compatibility

    release = tmp_path / "release"
    (release / "deployment").mkdir(parents=True)
    path = compatibility.write(path=release / "deployment" / "compatibility.json")
    monkeypatch.setenv("HERMES_MEMORY_RELEASE", str(release))
    finding = Doctor(store).release()
    assert finding.severity == OK
    assert finding.evidence["pinned"] == compatibility.PINNED_VERSION
    assert finding.evidence["manifest"] == str(path)


def test_a_manifest_that_claims_another_engine_stops_the_installation(store, tmp_path,
                                                                     monkeypatch):
    """The one artefact that says "this release works with that backend" has to be true.

    Read from a tampered manifest, every other check here would still pass: the code is
    fine, the store is fine, and the release is a lie about what it was composed against.
    """
    import json

    from hermes_memory.install import compatibility

    release = tmp_path / "release"
    (release / "deployment").mkdir(parents=True)
    path = release / "deployment" / "compatibility.json"
    compatibility.write(path=path)
    facts = json.loads(path.read_text(encoding="utf-8"))
    facts["hindsight"]["engine_pinned"] = "0.9.0"
    path.write_text(json.dumps(facts), encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_RELEASE", str(release))

    finding = Doctor(store).release()
    assert finding.severity == FAIL
    assert "0.9.0" in finding.detail and "0.10.1" in finding.detail
    assert "compatibility.json" in finding.remedy


def test_a_release_manifest_that_is_absent_is_a_failure_with_the_command_to_make_it(
        store, tmp_path, monkeypatch):
    """The tree that was supposed to carry the file and did not is a fault, not a shrug."""
    from hermes_memory.install import compatibility

    monkeypatch.delenv("HERMES_MEMORY_RELEASE", raising=False)
    monkeypatch.setattr(compatibility, "source_checkout", lambda: tmp_path / "tree")
    finding = Doctor(store).release()
    assert finding.severity == FAIL and finding.remedy
    assert "does not exist" in finding.detail
    assert "compatibility --write" in finding.remedy


# -- credentials and fences --------------------------------------------------

def test_a_key_in_the_record_set_is_found_and_not_echoed_back(store):
    evidence(store, f"here is my key {SECRET} for the api")
    finding = Doctor(store).credentials_in_records()
    assert finding.severity == FAIL
    assert SECRET not in json.dumps(finding.as_dict())
    assert "rotate" in finding.remedy


def test_ordinary_evidence_about_a_key_is_not_flagged(store):
    evidence(store, "I rotated the signing key on Tuesday.")
    finding = Doctor(store).credentials_in_records()
    assert finding.severity == OK
    assert finding.evidence["scanned"] == 1


def test_a_lease_held_past_its_word_is_a_wedged_worker(store, sync):
    sync.register("gmail", policy_version="local-only")
    store.db.execute("UPDATE connectors SET lease='abc', holder='worker-9', "
                     "lease_until=1 WHERE source='gmail'")
    finding = Doctor(store).leases()
    assert finding.severity == WARN
    assert finding.evidence["stale"] == ["gmail by worker-9"]


def test_a_lease_that_is_current_is_not_reported(store, sync):
    sync.register("gmail", policy_version="local-only")
    fence = sync.acquire("gmail", holder="worker-9", ttl=300)
    finding = Doctor(store).leases()
    assert finding.severity == OK
    sync.release(fence)


# -- probes ------------------------------------------------------------------

def test_a_probe_only_happens_when_it_is_asked_for_by_name(store):
    backend = FakeBackend()
    report = Doctor(store, backend=lambda: backend).examine(connectivity=True)
    assert backend.calls == ["health"]
    assert report["probes"] == {"connectivity": True, "synthetic": False}


def test_a_synthetic_probe_implies_connectivity(store):
    backend = FakeBackend()
    report = Doctor(store, backend=lambda: backend).examine(synthetic=True)
    assert report["probes"] == {"connectivity": True, "synthetic": True}
    assert backend.calls[0] == "health"


def test_an_unreachable_backend_is_a_failure_rather_than_an_empty_success(store):
    finding = Doctor(store, backend=lambda: FakeBackend(healthy=False)
                     ).backend_connectivity()
    assert finding.severity == FAIL
    assert "connection refused" in finding.detail
    assert "--synthetic-probe" in finding.remedy


def test_a_round_trip_probe_cleans_up_after_itself(store):
    backend = FakeBackend(results=(1,))
    finding = Doctor(store, backend=lambda: backend).backend_synthetic_round_trip()
    assert finding.severity == OK
    assert backend.calls == ["retain", "state", "recall", "delete"]
    assert finding.as_dict()["units"] == 1 and finding.as_dict()["returned"] == 1


def test_the_probe_asks_recall_in_the_words_it_just_wrote():
    """A query unrelated to what was retained cannot tell a broken search from a fair question.

    The other half of the sentence's shape is that it asserts something with a date and an
    object in it: the boilerplate this probe used to retain — a sentence about being a probe —
    is the kind of text an extraction model answers "nothing" to, and on this installation it
    did so about a third of the time, which the door then reported as an empty index.
    """
    asked = set(PROBE_QUERY.lower().split())
    written = {word.strip(".") for word in PROBE_TEXT.lower().split()}
    assert {"synthetic", "probe", "reservation", "118"} <= asked & written


def test_the_probe_retains_under_a_name_the_client_will_accept():
    """The probe's own id is checked by the same rule that checks an owner's.

    It used to carry an underscore, which is precisely what the engine escapes when it
    composes chunk ids, and the refusal is in this framework rather than in the backend — so
    the one opt-in model-backed health door the plan promises could not open at all, and every
    test around it passed because the fake took any name it was given.
    """
    assert not document_id_is_ambiguous(PROBE_DOCUMENT_ID)


def test_the_probe_window_is_bounded_enough_to_be_run_by_a_waiting_person(store):
    """The settle window is the door's own, so its worst case is a stated policy.

    `doctor --synthetic-probe` is run by somebody standing at a terminal because something is
    already wrong. A number that drifts into a minute of silence turns the diagnostic into the
    fault, so the ceiling is asserted rather than assumed.
    """
    assert 1 <= PROBE_SETTLE_READS <= 10
    assert (PROBE_SETTLE_READS - 1) * PROBE_SETTLE_WAIT_S <= 45.0


def test_the_probe_document_is_deleted_even_when_the_round_trip_fails(store):
    backend = FakeBackend(fail_on={"recall"})
    finding = Doctor(store, backend=lambda: backend).backend_synthetic_round_trip()
    assert finding.severity == FAIL
    assert "delete" in backend.calls, "a probe that leaves a document behind is a leak"
    assert finding.as_dict()["error"] == "no index yet", (
        "the reason the round trip failed is not lost to the cleanup")


def test_a_probe_that_fails_and_cannot_clean_up_reports_both_facts(store):
    """One finding, two faults: what broke, and what the door could not put back."""
    backend = FakeBackend(fail_on={"recall", "delete"})
    finding = Doctor(store, backend=lambda: backend).backend_synthetic_round_trip(reads=1)
    assert finding.severity == FAIL
    evidence = finding.as_dict()
    assert evidence["error"] == "no index yet"
    assert evidence["not_deleted"] == "the bank is read-only"


def test_a_retain_that_takes_a_moment_to_derive_is_not_called_broken(store):
    """The engine answers a retain before it has decided what the text was worth.

    How long it is given is the door's own bounded window, counted and reported: this is the
    wait that separates a derivation still in flight from a backend that stores without
    ever indexing.
    """
    slept: list[float] = []
    backend = FakeBackend(results=(1,), units_after=2)
    finding = Doctor(store, backend=lambda: backend).backend_synthetic_round_trip(
        reads=4, wait_s=7.5, sleeper=slept.append)
    assert finding.severity == OK
    assert slept == [7.5, 7.5], "the door waits on its own clock, not the backend's"
    assert backend.calls.count("state") == 3
    assert backend.calls.count("recall") == 1, "the search is asked once the units exist"
    assert finding.as_dict()["waited_seconds"] == 15.0
    assert "after 15s of waiting" in finding.detail


def test_units_that_never_arrive_are_a_failure_after_the_wait(store):
    """Running out of the window is still the failure; the wait only makes it honest."""
    slept: list[float] = []
    finding = Doctor(store, backend=lambda: FakeBackend(units_after=9)
                     ).backend_synthetic_round_trip(reads=3, wait_s=5.0,
                                                    sleeper=slept.append)
    assert finding.severity == FAIL
    assert slept == [5.0, 5.0], "the last read is not followed by a wait nobody will act on"
    assert "(10s spent waiting for it)" in finding.detail
    assert finding.as_dict()["units"] == 0


def test_a_backend_that_derives_nothing_names_the_model_rather_than_the_index(store):
    """The live fault, and the live lie about it.

    A retain answered success, cost 2035 tokens, and left no memory unit behind — the
    extraction model had simply found nothing in the sentence. Reported as an empty index, it
    sent the operator to the wrong component while the finding swore retrieval was silently
    broken.
    """
    backend = FakeBackend(units_after=3)
    finding = Doctor(store, backend=lambda: backend).backend_synthetic_round_trip(reads=2)
    assert finding.severity == FAIL
    assert "derived no memory unit" in finding.detail
    assert "extraction" in finding.remedy
    assert "delete" in backend.calls, "a failed derivation still gets its document back"
    assert "recall" not in backend.calls, (
        "nothing was derived, so asking the search path costs a model call for no answer")


def test_a_window_of_no_reads_still_looks_once(store):
    """The bound is a floor as well as a ceiling: zero reads is not a reading."""
    backend = FakeBackend(results=(1,))
    finding = Doctor(store, backend=lambda: backend).backend_synthetic_round_trip(reads=0)
    assert finding.severity == OK
    assert backend.calls.count("state") == 1


def test_units_that_cannot_be_found_are_the_search_paths_fault_not_the_writes(store):
    """Derived, stored, and still not findable: a different part, and a different remedy."""
    finding = Doctor(store, backend=lambda: FakeBackend(results=(), findable=False)
                     ).backend_synthetic_round_trip(reads=1)
    assert finding.severity == FAIL
    assert "cannot find any of them" in finding.detail
    assert "embeddings" in finding.remedy
    assert finding.as_dict()["units"] == 1, "the write is proved even as the read fails"


def test_a_probe_that_cannot_clean_up_after_itself_says_so(store):
    backend = FakeBackend(results=(1,), fail_on={"delete"})
    finding = Doctor(store, backend=lambda: backend).backend_synthetic_round_trip(reads=1)
    assert finding.severity == OK, "the round trip worked; the leak is a separate fact"
    assert "read-only" in finding.as_dict()["not_deleted"], finding.as_dict()


def test_a_probe_without_a_configured_backend_says_so_rather_than_crashing(store):
    doctor = Doctor(store, backend=lambda: None)
    assert doctor.backend_connectivity().severity == WARN
    assert doctor.backend_synthetic_round_trip().severity == WARN


def test_the_synthetic_probe_never_writes_into_the_owners_bank(installed):
    store, settings = installed
    seen: list[str] = []

    def build():
        client = FakeBackend(bank_id="unconfigured")
        seen.append(client.bank_id)
        return client

    doctor = Doctor(store, settings=settings, backend=build)
    assert doctor.backend_synthetic_round_trip().severity == OK
    assert seen == ["unconfigured"], "the injected client stands in for the real one"
    assert store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 0


def test_the_real_probe_client_is_built_on_its_own_bank(tmp_path, monkeypatch, store):
    settings = settings_at(tmp_path, monkeypatch, {
        "HINDSIGHT_URL": f"http://{APPROVED_LAN}:8080/v1",
        "ALLOWED_INFERENCE_HOSTS": APPROVED_LAN})
    client = Doctor(store, settings=settings).client()
    assert client.bank_id == "hermes-doctor-probe"
    assert client.base_url == f"http://{APPROVED_LAN}:8080/v1"


def test_no_client_is_built_when_no_backend_is_configured(tmp_path, monkeypatch, store):
    settings = settings_at(tmp_path, monkeypatch)
    assert Doctor(store, settings=settings).client() is None


# -- the report --------------------------------------------------------------

def test_remedies_are_collected_in_the_order_the_checks_ran(store):
    _job(store, state="uncertain")
    evidence(store, f"here is my key {SECRET} for the api")
    report = Doctor(store, backend=Tripwire()).examine()
    order = [item["check"] for item in report["findings"] if item["remedy"]]
    assert order.index("queue") < order.index("credentials")
    assert report["actions"] == [item["remedy"] for item in report["findings"]
                                 if item["remedy"]]


def test_an_unknown_severity_is_refused():
    with pytest.raises(ValueError, match="unknown severity"):
        Finding("layout", "probably-fine", "seems alright")


def test_a_doctor_with_no_settings_says_which_part_is_undetermined(store):
    assert Doctor(store).layout().severity == WARN
    assert Doctor(store).configuration().severity == WARN
    assert Doctor(store).backend_ledger().severity == OK, \
        "no settings means no backend configured, which is a shape not a fault"


def _job(store, *, state, job_id="job-1", created_at=None):
    stamp = created_at or STAMP
    insert(store, "processing_jobs", id=job_id, kind="retain", state=state, priority=3,
           resource="remote-9b", route="retain", epoch=1,
           processor_fingerprint="extractor-v3", inputs="[]", input_revision="1",
           max_attempts=3, token_budget=1000, created_at=stamp, updated_at=stamp)


def _handoff(store):
    insert(store, "goals", id="goal-1", title="Call", statement="Call the dentist",
           status="active", timezone="UTC", created_by="owner", created_kind="owner",
           created_at=STAMP, updated_at=STAMP)
    insert(store, "due_events", id="due-1", goal_id="goal-1", revision=1,
           fire_at="2099-01-01T00:00:00+00:00", reason="scheduled", timezone="UTC",
           precision="minute", state="pending", created_at=STAMP)
    insert(store, "decision_intents", id="intent-1", event_id="due-1", goal_id="goal-1",
           revision=1, kind="notify_owner", policy_version="v1", state="prepared",
           created_at=STAMP, updated_at=STAMP)
    insert(store, "proactive_decisions", id="dec-1", intent_id="intent-1",
           goal_id="goal-1", revision=1, topic="health", action="notify_owner",
           reason="policy", policy_version="v1", decided_at=STAMP)
    insert(store, "outbox", id="ob-1", decision_id="dec-1", kind="notify_owner",
           topic="health", recipient="owner", payload="Call the dentist",
           payload_digest="d" * 16, policy_version="v1", state="prepared",
           created_at=STAMP, updated_at=STAMP)


# -- the background pass -------------------------------------------------------

def _waiting(store, *, fire_at="2026-09-15T09:00:00+00:00"):
    insert(store, "goals", id="goal-1", title="Call", statement="Call the dentist",
           status="active", timezone="UTC", created_by="owner", created_kind="owner",
           created_at=fire_at, updated_at=fire_at)
    insert(store, "due_events", id="due-1", goal_id="goal-1", revision=1,
           fire_at=fire_at, reason="scheduled", timezone="UTC", precision="minute",
           state="pending", created_at=fire_at)


def _heartbeat(store, *, at="2026-09-15T09:00:00+00:00", created_at=None, **metadata):
    columns = {"at": at}
    columns.update(metadata)
    insert(store, "audit", action="maintenance_pass", object_id="maintenance",
           created_at=created_at or at, metadata=json.dumps(columns))


def looped_settings(interval):
    """Settings with only what this check reads: a period, and a home to look for no
    admission ledger in."""
    return SimpleNamespace(maintenance_interval_s=interval,
                           home=Path(tempfile.mkdtemp(prefix="hm-")))


def test_an_installation_where_nothing_runs_the_pass_says_so_and_names_the_setting(store):
    finding = Doctor(store, settings=looped_settings(0)).background()
    assert finding.severity == WARN
    assert "nothing runs the background pass" in finding.detail
    assert "HERMES_MEMORY_MAINTENANCE_INTERVAL_S" in finding.remedy
    assert "`hermes-memory maintain`" in finding.remedy


def test_a_dead_scheduler_with_a_reminder_waiting_is_a_warning_with_a_remedy(store):
    """This is the failure the whole check exists for: the queue looks empty, and the
    reminder is simply never taken."""
    _waiting(store)
    finding = Doctor(store, settings=looped_settings(900)).background()
    assert finding.severity == WARN
    assert "1 reminder(s) are due" in finding.detail
    assert "hermes-memory maintain" in finding.remedy
    assert finding.evidence["waiting"] == 1 and finding.evidence["behind"] is True


def test_a_scheduler_that_runs_but_cannot_finish_a_pass_is_a_failure_not_an_healthy_loop(
        store):
    """The distinction this machine got wrong for hours: not "not running", but "raising".

    A pass that breaks in one section stops there, so every section after it silently stops
    happening while the loop keeps its period and nothing is recorded. The heartbeat now
    carries where it broke, and the check says that instead of reporting an on-time pass.
    """
    from hermes_memory.ids import now

    _heartbeat(store, created_at=now(), failed_section="queue",
               error="OperationalError: attempt to write a readonly database")
    finding = Doctor(store, settings=looped_settings(900)).background()
    assert finding.severity == FAIL, finding.detail
    assert "raised in section 'queue'" in finding.detail
    assert "hermes-memory maintain" in finding.remedy
    assert finding.evidence["failed_section"] == "queue"


def test_a_pass_that_recorded_its_own_run_stops_the_alarm(store):
    _waiting(store)
    _heartbeat(store, created_at=None)
    from hermes_memory.ids import now
    store.db.execute("UPDATE audit SET created_at=? WHERE action='maintenance_pass'",
                     (now(),))
    finding = Doctor(store, settings=looped_settings(900)).background()
    assert finding.severity == OK, finding.detail
    assert finding.remedy is None, "a check that passes has nothing to tell anybody to do"


def test_a_stale_loop_with_nothing_waiting_is_not_reported_as_an_alarm(store):
    _heartbeat(store)
    finding = Doctor(store, settings=looped_settings(900)).background()
    assert finding.severity == OK
    assert finding.evidence["behind"] is True and finding.evidence["waiting"] == 0, \
        "nothing owed nobody is not a fault"


def test_a_build_with_no_manifest_warns_instead_of_condemning_a_path_it_never_had(
        store, tmp_path, monkeypatch):
    """An installed wheel has no deployment tree beside it; that is not a broken memory.

    The old fallback named `<venv>/lib/python3.12/deployment/compatibility.json` and then
    failed on it, which sent an operator to regenerate a file their machine has no source
    for. The finding now says what is missing and where the file is supposed to come from.
    """
    from hermes_memory.install import compatibility

    monkeypatch.delenv("HERMES_MEMORY_RELEASE", raising=False)
    monkeypatch.setattr(compatibility, "source_checkout", lambda: None)
    lonely = tmp_path / "venv/lib/python3.12/site-packages/hermes_memory"
    lonely.mkdir(parents=True)
    monkeypatch.setattr(compatibility, "PACKAGE_DIR", lonely)
    finding = Doctor(store).release()
    assert finding.severity == WARN, finding.detail
    assert finding.evidence["manifest"] is None
    assert "HERMES_MEMORY_RELEASE" in finding.remedy
