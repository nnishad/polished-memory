"""C12/C14: controlled raw-fact formation, planned and performed without a model.

The claim under test is that an installation with a queue and a worker can be *honest*
about them. Two failures matter more here than any other: a plan that performed something
it did not say it performed, and a report that says observations are being formed when
nothing would ever ask the backend anything. So each test below either counts the requests
that left the process — which must be none — or names the exact condition that makes a
pass refuse.
"""
from __future__ import annotations

import json
import sqlite3
import stat
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from conftest import envelope
from hermes_memory.backend.document_map import FAILED, VERIFIED, DocumentMap
from hermes_memory.backend.hindsight_client import HindsightUnavailable
from hermes_memory.config import load_settings
from hermes_memory.ids import backend_document_id
from hermes_memory.processing.budgets import Budget, Budgets
from hermes_memory.processing.allowance import ANY_RESOURCE, Allowances
from hermes_memory.processing import formation
from hermes_memory.processing.formation import (DEFAULT_BATCH, KIND, MAX_BATCH, MAX_JOBS,
                                                FormationError, backend_client,
                                                count_unprojected, formation_apply,
                                                formation_plan, formation_reconcile,
                                                processor_fingerprint, retain_route,
                                                unprojected)
from hermes_memory.processing.instance_gate import gate_path, instance_gate
from hermes_memory.processing.jobs import (CANCELLED, RETRY_WAIT, SUCCEEDED, UNCERTAIN,
                                           JobQueue)
from hermes_memory.storage.evidence import EvidenceError, EvidenceStore

OWNER = "jugaadu"
BANK = "hermes"
REMOTE = "remote-9b"
RETAIN_UPSTREAM = "http://127.0.0.1:11434/v1"


class Answers:
    """A backend that answers the way the pinned one does, and remembers being asked.

    A submission says the work was accepted and reports no cost — the model has not run
    yet. The operation is the answer that says it did, and usage arrives with it.
    """

    def __init__(self):
        self.retain_calls = []
        self.asked: list[str] = []

    def retain_async(self, items, *, submission_id):
        self.retain_calls.append({"items": items, "submission_id": submission_id})
        return {"ok": True, "operation_id": submission_id}

    def operation(self, operation_id):
        self.asked.append(operation_id)
        return {"status": "completed", "usage": {"total_tokens": 321}}

    def document_state(self, document_id):
        return {"document_id": document_id, "state": "present", "count": 1}


class Unreachable:
    """The transport died mid-request, which is not the same as the work having failed."""

    def retain_async(self, items, *, submission_id):
        raise HindsightUnavailable("connection reset while the request was in flight")

    def operation(self, operation_id):
        raise HindsightUnavailable("connection reset while asking about an operation")

    def document_state(self, document_id):
        raise HindsightUnavailable("the backend could not be asked about the document")


@pytest.fixture()
def installation(tmp_path, monkeypatch):
    """A configured, initialised, inference-enabled installation holding three records."""
    home = tmp_path / "instance"
    (home / "data").mkdir(parents=True)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS=200000\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n"
        "HERMES_MEMORY_HINDSIGHT_URL=http://127.0.0.1:8123\n"
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS=127.0.0.1\n"
        f"HERMES_MEMORY_TEXT_BASE_URL={RETAIN_UPSTREAM}\n"
        "HERMES_MEMORY_EMBEDDINGS_BASE_URL=http://127.0.0.1:11435/v1\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_RETAIN=cred-retain\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_EMBEDDINGS=cred-embed\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_CONSOLIDATE=cred-consolidate\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_REFLECT=cred-reflect\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_FOREGROUND=cred-foreground\n",
        encoding="utf-8")
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        for index in range(3):
            store.commit(envelope(source_id=f"msg-{index}", text=f"note {index}"))
    return settings


def approve(settings, client, **kwargs):
    """The two-step an operator performs: read the list, then approve exactly that list."""
    plan = formation_plan(settings, **kwargs)
    return formation_apply(settings, review=plan["review_digest"], actor=OWNER, client=client,
                           **kwargs)


def without(installation, key):
    """Drop one owned setting, the way an operator editing the env file would."""
    env_file = installation.home / "hermes-memory.env"
    env_file.write_text("".join(
        line for line in env_file.read_text(encoding="utf-8").splitlines(keepends=True)
        if not line.startswith(f"{key}=")), encoding="utf-8")


def records(store):
    return [row["id"] for row in store.db.execute(
        "SELECT id FROM records ORDER BY ingested_at, id")]


def test_invalid_poll_bounds_refuse_before_any_job_is_written(installation):
    client = Answers()
    shown = formation_plan(installation)
    with pytest.raises(ValueError, match="follow_max_polls"):
        formation_apply(installation, actor=OWNER, review=shown["review_digest"],
                        client=client, follow_max_polls=0)
    with EvidenceStore(installation.db_path) as store:
        assert store.db.execute("SELECT count(*) FROM processing_jobs").fetchone()[0] == 0
        assert store.db.execute("SELECT count(*) FROM backend_documents").fetchone()[0] == 0
    assert client.retain_calls == []


# -- what the reading sees ---------------------------------------------------

def test_every_live_record_is_unprojected_until_something_says_otherwise(installation):
    with EvidenceStore(installation.db_path) as store:
        rows = unprojected(store, bank_id=BANK)
        assert [row["record_id"] for row in rows] == records(store)
        assert {row["revision"] for row in rows} == {"1"}
        assert count_unprojected(store, bank_id=BANK) == 3


def test_a_hidden_record_is_not_waiting_to_be_formed(installation):
    with EvidenceStore(installation.db_path) as store:
        first = records(store)[0]
        store.hide(first, reason="superseded by a correction", actor=OWNER)
        assert count_unprojected(store, bank_id=BANK) == 2
        assert first not in [row["record_id"] for row in unprojected(store, bank_id=BANK)], \
            "evidence the owner withdrew must never be re-projected"


def test_unprojection_works_through_the_oldest_gap_first(installation):
    with EvidenceStore(installation.db_path) as store:
        order = records(store)
        assert [row["record_id"] for row in unprojected(store, bank_id=BANK, limit=2)] == \
            order[:2], "arrival order, not whoever asked loudest"


def test_a_verified_projection_under_a_superseded_epoch_is_not_coverage(installation):
    with EvidenceStore(installation.db_path) as store:
        documents = DocumentMap(store, bank_id=BANK)
        for record_id in records(store):
            documents.begin(record_id, "1")
            documents.confirm(record_id, "1")
        assert count_unprojected(store, bank_id=BANK) == 0
        store.bump_epoch(reason="the owner reset the bank", actor=OWNER)
        assert count_unprojected(store, bank_id=BANK) == 3, \
            "a projection confirmed before a reset describes a backend that reset cleared"


def test_a_pre_reset_confirmation_cannot_be_promoted_into_current_coverage(
        installation):
    with EvidenceStore(installation.db_path) as store:
        documents = DocumentMap(store, bank_id=BANK)
        for record_id in records(store):
            documents.begin(record_id, "1")
            documents.confirm(record_id, "1")
        store.bump_epoch(reason="reset", actor=OWNER)
        assert count_unprojected(store, bank_id=BANK) == 3
        for record_id in records(store):
            with pytest.raises(EvidenceError, match="stale"):
                documents.confirm(record_id, "1")
        assert count_unprojected(store, bank_id=BANK) == 3


def test_another_bank_s_own_projections_are_not_this_one_s_coverage(installation):
    with EvidenceStore(installation.db_path) as store:
        documents = DocumentMap(store, bank_id="someone-else")
        documents.begin(records(store)[0], "1")
        documents.confirm(records(store)[0], "1")
        assert count_unprojected(store, bank_id=BANK) == 3


# -- the plan is a reading ---------------------------------------------------

def test_planning_builds_no_backend_and_creates_no_ledger(installation, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a plan that dialled the backend is not a reading")

    monkeypatch.setattr(formation, "backend_client", refuse)
    plan = formation_plan(installation)
    assert plan["ok"] and plan["selected"]
    assert not gate_path(installation).exists(), "a reading created the thing it measured"


def test_an_approved_pass_uses_the_client_it_was_handed_and_not_a_fresh_one(installation,
                                                                            monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the caller's client is the one that must answer")

    monkeypatch.setattr(formation, "backend_client", refuse)
    backend = Answers()
    assert formation_apply(installation, review=(
        formation_plan(installation)["review_digest"]), actor=OWNER,
        client=backend)["drain"]["attempted"] == 3
    assert len(backend.retain_calls) == 3


def test_planning_leaves_the_archive_bytes_as_it_found_them(installation):
    with EvidenceStore(installation.db_path) as store:
        store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw = installation.db_path.read_bytes()
    formation_plan(installation)
    assert installation.db_path.read_bytes() == raw, "a plan that changed the store is a write"


def test_planning_cannot_write_the_ledger_it_is_reading(installation):
    with instance_gate(installation) as gate:
        gate.db.execute("INSERT INTO gate_ledger(reservation_id, resource, route, holder, "
                        "event, at, detail) VALUES('r','x','y','z','logged',1,'')")
    formation_plan(installation)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        with instance_gate(installation, create=False) as reading:
            reading.db.execute("DELETE FROM gate_ledger")


def test_the_plan_names_the_route_it_would_use_and_never_a_credential(installation):
    plan = formation_plan(installation)
    assert plan["route"] == "retain" and plan["resource"] == REMOTE
    assert plan["upstream"] == RETAIN_UPSTREAM and plan["max_output_tokens"] == 2048
    assert plan["priority"] == "freshness"
    assert "cred-retain" not in json.dumps(plan, sort_keys=True)


def test_a_job_is_allowed_its_prompt_estimate_plus_the_route_output_cap(installation):
    plan = formation_plan(installation)
    assert plan["per_job_tokens"] == 2048 + 2000
    assert plan["planned_jobs"] == plan["budget"]["planned"] == 3


def test_the_daily_allowance_is_a_ceiling_on_one_job_as_well(installation, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS", "1500")
    plan = formation_plan(load_settings())
    assert plan["per_job_tokens"] == 1500
    assert any("cannot fit one item" in line for line in plan["blocking"])


@pytest.mark.parametrize(("key", "value", "reason"), [
    ("HERMES_MEMORY_INFERENCE_ENABLED", "false", "inference is switched off"),
    ("HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS", "0", "daily background budget is 0"),
    ("HERMES_MEMORY_HINDSIGHT_URL", "", "nothing to project evidence into"),
])
def test_an_installation_that_cannot_afford_inference_says_so(installation, monkeypatch,
                                                              key, value, reason):
    monkeypatch.setenv(key, value)
    plan = formation_plan(load_settings())
    assert not plan["ok"] and any(reason in line for line in plan["blocking"])


def test_no_archive_is_a_reason_to_refuse_and_not_to_create_one(installation):
    installation.db_path.unlink()
    plan = formation_plan(installation)
    assert not installation.db_path.exists(), "the plan opened a store it was told was absent"
    assert any("no canonical store" in line for line in plan["blocking"])


def test_a_route_with_no_credential_selects_nothing(installation):
    without(installation, "HERMES_MEMORY_ROUTE_CREDENTIAL_RETAIN")
    plan = formation_plan(load_settings())
    # Withheld rather than fatal: the gate still serves the routes that are authenticated,
    # but a pass that needs the unminted one has nothing to spend and says which key is
    # missing.
    assert any("ROUTE_CREDENTIAL_RETAIN is not set" in line for line in plan["blocking"]), \
        plan["blocking"]


def test_an_operator_hold_on_inference_blocks_the_pass_before_it_is_approved(installation):
    with instance_gate(installation) as gate:
        gate.pause(actor=OWNER, reason="the models are being updated")
    plan = formation_plan(installation)
    assert plan["gate"]["paused"] and not plan["ok"]
    assert any("paused by the operator" in line for line in plan["blocking"])


def test_nothing_unprojected_is_a_refusal_rather_than_an_empty_success(installation):
    approve(installation, Answers())
    plan = formation_plan(installation)
    assert plan["unprojected"] == 0
    assert any("nothing is unprojected" in line for line in plan["blocking"])


def test_the_plan_states_what_it_did_not_do(installation):
    plan = formation_plan(installation)
    assert "no job was written to the queue" in plan["not_performed"]
    assert plan["note"].startswith("planning only")


# -- the digest --------------------------------------------------------------

def test_the_digest_is_a_stable_reading_of_the_same_state(installation):
    assert formation_plan(installation)["review_digest"] == \
        formation_plan(installation)["review_digest"]


def test_new_evidence_changes_the_plan_because_it_changes_the_work(installation):
    first = formation_plan(installation)["review_digest"]
    with EvidenceStore(installation.db_path) as store:
        store.commit(envelope(source_id="msg-late", text="arrived after the list was shown"))
    assert formation_plan(installation)["review_digest"] != first


def test_spending_the_budget_changes_the_digest_even_on_the_same_records(installation):
    first = formation_plan(installation)["review_digest"]
    with instance_gate(installation) as gate:
        Budgets(gate.store, daily={REMOTE: Budget(tokens=200000)}).charge(REMOTE, tokens=1000)
    assert formation_plan(installation)["review_digest"] != first


@pytest.mark.parametrize(("key", "value"), [
    ("HERMES_MEMORY_TEXT_BASE_URL", "http://127.0.0.1:19999/v1"),
    ("HERMES_MEMORY_MAX_OUTPUT_TOKENS_RETAIN", "512"),
])
def test_a_different_processor_is_a_different_plan(installation, monkeypatch, key, value):
    before = formation_plan(installation)
    monkeypatch.setenv(key, value)
    after = formation_plan(load_settings())
    assert after["review_digest"] != before["review_digest"]
    assert after["processor_fingerprint"] != before["processor_fingerprint"]


def test_rotating_a_route_credential_leaves_the_coverage_claim_alone(installation, monkeypatch):
    before = processor_fingerprint(installation, retain_route(installation))
    env_file = installation.home / "hermes-memory.env"
    env_file.write_text(env_file.read_text(encoding="utf-8").replace(
        "cred-retain", "cred-rotated"), encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_ROUTE_CREDENTIAL_RETAIN", "cred-rotated")
    settings = load_settings()
    assert processor_fingerprint(settings, retain_route(settings)) == before, \
        "a key rotation does not invalidate facts already extracted"


def test_retention_context_contract_is_part_of_processor_identity(installation, monkeypatch):
    from hermes_memory.processing import formation

    route = retain_route(installation)
    before = processor_fingerprint(installation, route)
    monkeypatch.setattr(formation, "RETAIN_CONTEXT_VERSION", formation.RETAIN_CONTEXT_VERSION + 1)
    assert processor_fingerprint(installation, route) != before


def test_declared_model_changes_affect_processor_identity(installation):
    from dataclasses import replace

    before = processor_fingerprint(installation, retain_route(installation))
    changed = replace(installation, text_route=replace(installation.text_route, model="different-model"))
    assert processor_fingerprint(changed, retain_route(changed)) != before


def test_the_wall_clock_in_a_plan_is_not_part_of_the_approval(installation):
    plan = formation_plan(installation)
    assert plan["observed_at"]
    assert plan["review_digest"] == formation_plan(installation)["review_digest"]


def test_an_approval_granted_before_a_pause_cannot_be_spent_after_it(installation):
    """The owner's stop outranks a decision they made while the models were running.

    The digest is over the blocking reasons as well as the work, so pausing inference
    changes the plan rather than merely the outcome of running the old one.
    """
    plan = formation_plan(installation)
    with instance_gate(installation) as gate:
        gate.pause(actor=OWNER, reason="stop, mid-approval")
    with pytest.raises(FormationError, match="does not match what would happen now"):
        formation_apply(installation, review=plan["review_digest"], actor=OWNER,
                        client=Answers())


# -- refusal before action ---------------------------------------------------

def test_an_unattributed_pass_is_refused(installation):
    plan = formation_plan(installation)
    with pytest.raises(FormationError, match="actor must be named"):
        formation_apply(installation, review=plan["review_digest"], actor="   ")


def test_approving_without_a_digest_is_refused_rather_than_guessed(installation):
    with pytest.raises(FormationError, match="without --review first"):
        formation_apply(installation, review="", actor=OWNER, client=Answers())


def test_a_digest_of_a_plan_that_no_longer_describes_the_pass_is_refused(installation):
    stale = formation_plan(installation)
    with EvidenceStore(installation.db_path) as store:
        store.commit(envelope(source_id="msg-new", text="new evidence"))
    with pytest.raises(FormationError, match="does not match what would happen now"):
        formation_apply(installation, review=stale["review_digest"], actor=OWNER,
                        client=Answers())


def test_a_refused_pass_leaves_the_archive_exactly_as_it_was(installation):
    with instance_gate(installation) as gate:
        gate.pause(actor=OWNER, reason="held")
    plan = formation_plan(installation)
    with EvidenceStore(installation.db_path) as store:
        store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw = installation.db_path.read_bytes()
    with pytest.raises(FormationError, match="refused: "):
        formation_apply(installation, review=plan["review_digest"], actor=OWNER,
                        client=Answers())
    assert installation.db_path.read_bytes() == raw


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": MAX_BATCH + 1}, {"limit": "20"}])
def test_a_pass_must_be_bounded_by_a_number_that_makes_sense(installation, kwargs):
    with pytest.raises(FormationError, match="limit must be an integer"):
        formation_plan(installation, **kwargs)
    with pytest.raises(FormationError, match="limit must be an integer"):
        formation_apply(installation, review="anything", actor=OWNER, client=Answers(),
                        **kwargs)


@pytest.mark.parametrize("max_jobs", [0, MAX_JOBS + 1, None])
def test_the_number_of_jobs_attempted_is_also_bounded(installation, max_jobs):
    plan = formation_plan(installation)
    with pytest.raises(FormationError, match="max_jobs must be between"):
        formation_apply(installation, review=plan["review_digest"], actor=OWNER,
                        client=Answers(), max_jobs=max_jobs)


def test_an_empty_selection_is_refused_rather_than_run_as_a_no_op(installation):
    with EvidenceStore(installation.db_path) as store:
        with pytest.raises(FormationError, match="limit must be an integer"):
            unprojected(store, bank_id=BANK, limit=0)


# -- the pass, performed -----------------------------------------------------

def test_a_bounded_pass_projects_every_record_it_promised(installation):
    backend = Answers()
    plan = formation_plan(installation)
    receipt = formation_apply(installation, review=plan["review_digest"], actor=OWNER,
                              client=backend)
    assert receipt["queued"]["created"] == 3
    assert receipt["drain"]["counts"] == {SUCCEEDED: 3}
    assert receipt["drain"]["attempted"] == 3
    assert len(backend.retain_calls) == 3
    assert formation_plan(installation)["unprojected"] == 0


def test_each_job_carries_the_revision_of_the_record_it_projects(installation):
    approve(installation, Answers())
    with EvidenceStore(installation.db_path) as store:
        rows = store.db.execute(
            "SELECT b.record_id AS record_id, b.revision AS mapped, r.revision AS current, "
            "b.state AS state FROM backend_documents b JOIN records r ON r.id = b.record_id "
            "ORDER BY b.record_id").fetchall()
        assert len(rows) == 3
        for row in rows:
            assert row["mapped"] == row["current"], \
                "a projection keyed on a revision the record never had"
            assert row["state"] == "verified"
            assert backend_document_id(row["record_id"], row["mapped"])


def test_a_pass_carries_each_records_own_revision_not_a_placeholder(installation):
    with EvidenceStore(installation.db_path) as store:
        store.commit(envelope(source_id="msg-v2", revision="42", text="a later version"))
        backend = Answers()
    receipt = approve(installation, backend)
    assert receipt["queued"]["created"] == 4
    with EvidenceStore(installation.db_path) as store:
        mapped = dict(store.db.execute(
            "SELECT record_id, revision FROM backend_documents").fetchall())
        newest = store.db.execute(
            "SELECT id, revision FROM records WHERE source_id='msg-v2'").fetchone()
        assert mapped[newest["id"]] == "42", \
            "one revision for the whole job is a placeholder, not a projection key"


def test_the_pass_says_who_authorised_what_it_spent(installation):
    plan = formation_plan(installation)
    receipt = formation_apply(installation, review=plan["review_digest"], actor=OWNER,
                              client=Answers())
    assert receipt["actor"] == OWNER
    assert receipt["review_digest"] == plan["review_digest"], \
        "the receipt names the list that was approved, not one regenerated afterwards"
    assert "usage charged to the instance gate" in " ".join(receipt["performed"])


def test_what_the_backend_reported_is_charged_to_the_shared_ledger_not_the_archive(
        installation):
    approve(installation, Answers())
    with instance_gate(installation) as gate:
        assert gate.usage()[REMOTE]["tokens"] == 963
    with EvidenceStore(installation.db_path) as store:
        assert store.db.execute("SELECT count(*) FROM budget_usage").fetchone()[0] == 0, \
            "a profile archive is not where a physical device's day is counted"


def test_the_gate_ledger_keeps_its_owner_only_permissions(installation):
    approve(installation, Answers())
    assert stat.S_IMODE(gate_path(installation).stat().st_mode) == 0o600


def test_a_succeeded_job_is_not_re_run_because_a_projection_row_went_missing(installation):
    """Losing a mapping is a reconciliation question, not permission to spend again."""
    approve(installation, Answers())
    with EvidenceStore(installation.db_path) as store:
        store.db.execute("UPDATE backend_documents SET state='queued'")
        assert count_unprojected(store, bank_id=BANK) == 3
    backend = Answers()
    receipt = approve(installation, backend)
    assert receipt["queued"] == {"created": 0, "existing": 3, "records": 3, "route": "retain"}
    assert backend.retain_calls == [], "the archive was re-sent to the model without approval"
    assert receipt["drain"]["attempted"] == 0


def test_the_pass_reconciles_a_lease_that_outlived_its_holder_before_claiming_anything(
        installation):
    with EvidenceStore(installation.db_path) as store:
        jobs = JobQueue(store)
        doomed = jobs.enqueue(kind=KIND, inputs=[records(store)[0]], input_revision="1",
                              route=retain_route(installation),
                              processor_fingerprint="an-older-processor",
                              priority="maintenance")["job_id"]
        store.db.execute("UPDATE processing_jobs SET state='leased', lease='gone', "
                         "lease_until=1 WHERE id=?", (doomed,))
    receipt = approve(installation, Answers())
    assert receipt["reconciled"]["lease_expired"] == 1
    assert receipt["drain"]["counts"] == {SUCCEEDED: 3, UNCERTAIN: 1}, \
        "an expired lease is unresolved work, never a free second attempt"
    assert receipt["drain"]["attempted"] == 3


def test_a_backend_that_might_still_be_running_leaves_the_slot_quarantined(installation):
    receipt = approve(installation, Unreachable())
    assert receipt["drain"]["counts"] == {UNCERTAIN: 1, "queued": 2}, \
        "one unproven request holds the device, and the rest of the pass waits"
    with instance_gate(installation) as gate:
        assert gate.blocked_resources() == [REMOTE], \
            "an unproven request holds the device until its outcome is established"
    plan = formation_plan(installation)
    assert plan["gate"]["blocked"] == [REMOTE], \
        "a quarantined slot is reported; contention is not the pass having failed"


# -- the route and the client ------------------------------------------------

def test_the_retain_route_is_the_one_the_configuration_names(installation):
    route = retain_route(installation)
    assert route.name == "retain" and route.resource == REMOTE
    assert route.priority == "freshness", "raw facts are freshness work, not maintenance"


def test_the_backend_client_is_addressed_by_the_installation_not_the_caller(installation):
    client = backend_client(installation)
    assert client.base_url == "http://127.0.0.1:8123" and client.bank_id == BANK
    assert client.api_key is None, "no ambient credential discovery"


def test_an_installation_with_no_backend_has_nothing_to_address(installation):
    without(installation, "HERMES_MEMORY_HINDSIGHT_URL")
    settings = load_settings()
    assert settings.hindsight_url is None
    with pytest.raises(FormationError, match="no backend endpoint"):
        backend_client(settings)


# -- sizes -------------------------------------------------------------------

def test_a_forgotten_record_is_absent_rather_than_waiting_to_be_formed(installation):
    """A projection of erased evidence is the one thing no pass may ever do.

    The row is not hidden and not superseded: it is deleted, and a reader that counted
    ``records`` without the fence would hand the model back something the owner already
    took away.
    """
    from hermes_memory.lifecycle.erasure import ErasureManager

    with EvidenceStore(installation.db_path) as store:
        doomed = records(store)[0]
        manager = ErasureManager(store, owner_principal=OWNER)
        preview = manager.preview(record_ids=[doomed], actor=OWNER,
                                  reason="the owner asked for this one back")
        manager.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor=OWNER)
        assert count_unprojected(store, bank_id=BANK) == 2
        assert doomed not in [row["record_id"] for row in unprojected(store, bank_id=BANK)]


def test_a_pass_never_projects_evidence_that_vanished_after_it_was_planned(installation):
    """The approval named three records; only two of them still exist to be projected."""
    from hermes_memory.lifecycle.erasure import ErasureManager

    backend = Answers()
    plan = formation_plan(installation)
    with EvidenceStore(installation.db_path) as store:
        doomed = records(store)[-1]
        manager = ErasureManager(store, owner_principal=OWNER)
        preview = manager.preview(record_ids=[doomed], actor=OWNER, reason="mid-approval")
        manager.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor=OWNER)
    with pytest.raises(FormationError, match="does not match what would happen now"):
        formation_apply(installation, review=plan["review_digest"], actor=OWNER,
                        client=backend)
    assert backend.retain_calls == []


def test_the_daily_ceiling_is_the_machine_s_and_not_this_profile_s(installation):
    """Another holder's spend is this pass's ceiling, or the budget meters nothing.

    The gate charges the shared ledger, so dispatch has to be admitted against the same
    place. A per-archive meter would let each enrolled profile spend a full day on one
    physical model.
    """
    with instance_gate(installation) as gate:
        Budgets(gate.store, daily={REMOTE: Budget(tokens=200000)}).charge(
            REMOTE, tokens=197_500)
    backend = Answers()
    receipt = approve(installation, backend)
    assert receipt["budget"]["resources"][REMOTE]["tokens_used"] > 197_500
    assert len(backend.retain_calls) == 2, \
        "the third dispatch cannot fit what is left of today"
    assert "budget_exhausted" in json.dumps(receipt["drain"]["outcomes"])


def test_the_default_pass_size_is_inside_the_bound(installation):
    assert 1 <= DEFAULT_BATCH <= MAX_BATCH
    assert formation_plan(installation, limit=DEFAULT_BATCH)["planned_jobs"] == 3
    assert formation_plan(installation, limit=1)["planned_jobs"] == 1, \
        "one record at a time is a pass an operator can reason about"


def test_a_period_of_the_reading_is_reported_as_the_day_the_budget_counts(installation):
    """A remaining allowance with no day beside it is a number from nowhere.

    The ledger is keyed by UTC date, so the charge and the plan are made to agree on
    which today they mean rather than on a date this file was written.
    """
    with instance_gate(installation) as gate:
        Budgets(gate.store, daily={REMOTE: Budget(tokens=200000)}).charge(REMOTE, tokens=500)
    plan = formation_plan(installation)
    assert plan["budget"]["ledger"] == "instance"
    assert plan["budget"]["remaining"] == 200000 - 500
    assert plan["budget"]["period"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")


# -- asking the backend about work this machine cannot account for -----------

def strand(settings, *, text="an old note"):
    """Commit a record and submit its projection without waiting for an answer."""
    with EvidenceStore(settings.db_path) as store:
        record = store.commit(envelope(source_id=f"msg-{text}", text=text))["id"]
        begun = DocumentMap(store, bank_id=BANK).begin(record, "1", async_submission=True)
    return record, begun["submission_id"]


def test_reconcile_asks_the_backend_about_every_submission_we_cannot_account_for(installation):
    record, submission = strand(installation)
    backend = Answers()
    report = formation_reconcile(installation, client=backend)
    assert report["settled"] == 1 and report["verified"] == 1
    assert backend.asked == [submission], "the identity on the row is the question asked"
    with EvidenceStore(installation.db_path) as store:
        assert DocumentMap(store, bank_id=BANK).state(record, "1") == VERIFIED


def test_reconcile_needs_no_backend_when_there_is_nothing_to_ask(installation):
    """An empty ledger is a complete answer, not a reason to open a socket.

    A machine with no backend configured can still be asked whether it is waiting on one,
    and the answer to that is not "refused: no endpoint".
    """
    without = replace(installation, hindsight_url="")
    report = formation_reconcile(without)
    assert report["ok"] is True and report["asked"] == [] and report["settled"] == 0
    strand(without)
    with pytest.raises(FormationError, match="no backend endpoint"):
        formation_reconcile(without)


def test_reconcile_asks_nothing_when_there_is_nothing_to_ask(installation):
    backend = Answers()
    report = formation_reconcile(installation, client=backend)
    assert report["asked"] == [] and report["settled"] == 0
    assert backend.asked == [], "an empty ledger is not a question to ask a backend"


def test_reconcile_claims_no_coverage_the_backend_did_not_give(installation):
    record, _ = strand(installation)
    backend = Answers()
    backend.operation = lambda operation_id: {"status": "processing"}
    report = formation_reconcile(installation, client=backend)
    assert report["pending"] == 1 and report["verified"] == 0
    with EvidenceStore(installation.db_path) as store:
        docs = DocumentMap(store, bank_id=BANK)
        assert docs.state(record, "1") == "submitted"
        assert count_unprojected(store, bank_id=BANK) == 4, \
            "unresolved work is still unprojected, not coverage"


def a_stranded_submission(settings, *, text="a note nobody saw the end of"):
    """A projection the backend holds an answer to, and the queue row that carried it.

    This is the residue a killed pass leaves: the operation identity is written on both rows,
    the outcome exists only in the engine, and the doctor names reconciliation as the remedy.
    """
    record, submission = strand(settings, text=text)
    with EvidenceStore(settings.db_path) as store:
        jobs = JobQueue(store)
        job_id = jobs.enqueue(kind=KIND, inputs=[record], input_revision="1",
                              route=retain_route(settings),
                              processor_fingerprint="the-processor-that-died",
                              priority="freshness")["job_id"]
        jobs.begin_submission(jobs.get(job_id), submission_id=submission)
        jobs.uncertain(jobs.get(job_id), reason="the bounded wait ran out")
    return record, submission, job_id


def test_reconcile_closes_the_job_the_backend_answered_for(installation):
    """An uncertain row is a question, and this door is the answer to it.

    Reconciling already wrote verified coverage from the same answer, so leaving the queue row
    uncertain made the remedy the doctor names do nothing to the thing it names: a machine with
    four settled submissions went on reporting four jobs that will not be retried by
    themselves, forever, with an operator told to run the door that had already run.
    """
    record, submission, job_id = a_stranded_submission(installation)
    report = formation_reconcile(installation, client=Answers())
    assert report["settled"] == 1 and report["jobs_settled"] == 1
    assert report["answers"]["established"] == [submission]
    with EvidenceStore(installation.db_path) as store:
        job = JobQueue(store).get(job_id)
        assert job.state == SUCCEEDED, "the engine said it finished"
        assert job.tokens_used == 0, "the answer says the work landed, not what it cost"
        assert DocumentMap(store, bank_id=BANK).state(record, "1") == VERIFIED
        assert store.db.execute("SELECT count(*) FROM audit WHERE action="
                                "'job_settled_by_reconciliation'").fetchone()[0] == 1


def test_reconcile_leaves_a_job_the_backend_has_not_answered_for_alone(installation):
    """`processing` is not `completed`: the question stays, and so does the row.

    Closing an uncertain job on a running operation would claim coverage the engine has not
    given, which is the one failure this whole path exists to avoid.
    """
    record, submission, job_id = a_stranded_submission(installation)
    backend = Answers()
    backend.operation = lambda operation_id: {"status": "processing"}
    report = formation_reconcile(installation, client=backend)
    assert report["jobs_settled"] == 0 and report["pending"] == 1
    assert report["answers"]["established"] == []
    with EvidenceStore(installation.db_path) as store:
        assert JobQueue(store).get(job_id).state == UNCERTAIN


def test_reconcile_settles_only_the_row_that_carried_the_answer(installation):
    """One confirmed submission settles one job, not the backlog it sits beside."""
    answered = a_stranded_submission(installation, text="answered")
    still_open = a_stranded_submission(installation, text="still open")
    backend = Answers()
    identity = answered[1]

    def operation(operation_id):
        return ({"status": "completed"} if operation_id == identity
                else {"status": "processing"})

    backend.operation = operation
    report = formation_reconcile(installation, client=backend)
    assert report["jobs_settled"] == 1 and report["pending"] == 1
    with EvidenceStore(installation.db_path) as store:
        jobs = JobQueue(store)
        assert jobs.get(answered[2]).state == SUCCEEDED
        assert jobs.get(still_open[2]).state == UNCERTAIN


def test_reconcile_does_not_resurrect_work_the_operator_stopped(installation):
    """A cancelled job stays cancelled even when the engine says the operation finished.

    The stop was somebody's decision and a late answer is not a second approval. The queue's
    own transition guard refuses to move a cancelled row; this door writes SQL of its own, so
    the same rule has to hold here explicitly.
    """
    record, submission, job_id = a_stranded_submission(installation)
    with EvidenceStore(installation.db_path) as store:
        JobQueue(store).cancel(job_id, actor=OWNER, reason="superseded by hand")

    report = formation_reconcile(installation, client=Answers())
    assert report["settled"] == 1, "the projection is still answered for"
    assert report["jobs_settled"] == 0, "and the stopped job is not moved by that answer"
    with EvidenceStore(installation.db_path) as store:
        assert JobQueue(store).get(job_id).state == CANCELLED


def test_reconcile_settles_a_synchronous_submission_by_the_document_it_landed_in(
        installation):
    """A retain that named no operation is still answerable: the document is the identity.

    A synchronous submission records no operation id, so the job row carries the document id
    and the probe is the only answer it can ever get. The identity that settles the coverage
    has to be the same one that settles the work, or a pass interrupted before an operation id
    existed would stay uncertain whatever the backend then said.
    """
    with EvidenceStore(installation.db_path) as store:
        record = store.commit(envelope(source_id="msg-sync",
                                      text="a synchronous retain"))["id"]
        begun = DocumentMap(store, bank_id=BANK).begin(record, "1")
        assert begun["submission_id"] is None, "a synchronous call mints no operation id"
        jobs = JobQueue(store)
        job_id = jobs.enqueue(kind=KIND, inputs=[record], input_revision="1",
                              route=retain_route(installation),
                              processor_fingerprint="the-processor-that-died",
                              priority="freshness")["job_id"]
        jobs.begin_submission(jobs.get(job_id), submission_id=begun["document_id"])
        jobs.uncertain(jobs.get(job_id), reason="the wait ran out")

    backend = Answers()
    backend.document_state = lambda document: {"state": "present"}
    report = formation_reconcile(installation, client=backend)
    assert report["verified"] == 1 and report["settled"] == 1
    assert report["jobs_settled"] == 1
    assert report["answers"]["established"] == [begun["document_id"]]
    with EvidenceStore(installation.db_path) as store:
        assert JobQueue(store).get(job_id).state == SUCCEEDED


def a_forgotten_projection(installation, *, text="a retain whose record was erased"):
    """A job whose submission left no projection, plus the tombstone that says why.

    Both rows are written rather than one: a tombstone without the intent that made it is a
    foreign key away from existing, and the intent is the fact that makes this an owner's
    decision instead of a lost submission.
    """
    record, submission, job_id = a_stranded_submission(installation, text=text)
    with EvidenceStore(installation.db_path) as store:
        DocumentMap(store, bank_id=BANK).mark_absent(record, "1")
        store.db.execute(
            "INSERT INTO erasure_ledger(id, source, requested_at, requested_by, "
            "requester_kind, reason, preview, preview_digest, state, epoch) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("erase_reconciled", record, "2026-09-27T00:00:00+00:00", OWNER, "owner",
             "the owner asked", "{}", "0" * 64, "erased", store.epoch()))
        store.db.execute("INSERT INTO tombstones(record_id, intent_id, fingerprint, "
                         "deleted_at) VALUES(?,?,?,?)",
                         (record, "erase_reconciled", "0" * 64, "2026-09-27T00:00:00+00:00"))
    return record, submission, job_id


def test_reconcile_closes_a_row_whose_answer_arrived_in_an_earlier_pass(installation):
    """The live shape: the projection was answered before the queue could be told.

    Four jobs sat uncertain beside four settled submissions on a real installation. The door
    asked only about rows still `outstanding`, and a projection stops being outstanding the
    moment it is verified — so the backlog it named as its own remedy was unreachable by it. A
    queue row has no view on when its answer arrived.
    """
    record, submission, job_id = a_stranded_submission(installation)
    with EvidenceStore(installation.db_path) as store:
        DocumentMap(store, bank_id=BANK).confirm(record, "1")

    report = formation_reconcile(installation, client=Answers())
    assert report["asked"] == [], "nothing to ask: the projection already has its answer"
    assert report["settled"] == 0 and report["verified"] == 0
    assert report["jobs_settled"] == 1, "and the row is closed anyway"
    assert report["answers"]["established"] == [submission]
    with EvidenceStore(installation.db_path) as store:
        assert JobQueue(store).get(job_id).state == SUCCEEDED


def test_reconcile_sends_a_denied_submission_back_through_the_attempt_budget(installation):
    """§8.2: an answer that says "that never happened" is not a reason to ask forever.

    The row is uncertain because the outcome was unknown, and the rule that uncertain work is
    never retried by itself exists to stop a machine resending private text on a guess. Once the
    engine names the outcome, the ordinary failed-attempt edge is the honest one: the attempt is
    counted, the backoff holds it, and the budget can still quarantine it. Nothing is dispatched
    here and nothing is charged — the next claim is a worker's, under the gate.
    """
    record, submission, job_id = a_stranded_submission(installation)
    backend = Answers()
    backend.operation = lambda operation_id: {"status": "failed", "error": "no such model"}
    report = formation_reconcile(installation, client=backend)
    assert report["jobs_retried"] == 1 and report["jobs_settled"] == 0
    assert report["answers"]["refused"] == [submission]
    with EvidenceStore(installation.db_path) as store:
        job = JobQueue(store).get(job_id)
        assert job.state == RETRY_WAIT, "the work is owed again, not forgiven"
        assert job.attempts == 1, "and the attempt the uncertainty swallowed is counted"
        assert "did not reach it" in job.last_error
        assert DocumentMap(store, bank_id=BANK).state(record, "1") == FAILED
        assert store.db.execute("SELECT count(*) FROM audit WHERE action="
                                "'job_refused_by_reconciliation'").fetchone()[0] == 1


def test_a_denied_submission_is_not_counted_twice_by_two_passes(installation):
    """A row waiting for its backoff is not in flight, so the next pass leaves it alone."""
    record, submission, job_id = a_stranded_submission(installation)
    backend = Answers()
    backend.operation = lambda operation_id: {"status": "failed"}
    assert formation_reconcile(installation, client=backend)["jobs_retried"] == 1
    second = formation_reconcile(installation, client=backend)
    assert second["jobs_retried"] == 0, "the projection is asked again; the job is not"
    with EvidenceStore(installation.db_path) as store:
        assert JobQueue(store).get(job_id).attempts == 1


def test_reconcile_ends_work_whose_record_was_forgotten_rather_than_re_forming_it(
        installation):
    """C4's whole point, applied to the queue: a missing projection has two causes.

    `absent` is written both when a submission never arrived and when the owner erased the
    record and the erasure path settled the mapping beside its tombstone. Only the second is a
    decision a machine may not undo: re-forming it would put text the owner deleted back into
    the engine. So the tombstone is asked, and a forgotten record's work is ended instead of
    tried again.
    """
    record, submission, job_id = a_forgotten_projection(installation)
    report = formation_reconcile(installation, client=Answers())
    assert report["answers"]["ended"] == [submission]
    assert report["answers"]["refused"] == []
    assert report["jobs_ended"] == 1 and report["jobs_retried"] == 0
    with EvidenceStore(installation.db_path) as store:
        job = JobQueue(store).get(job_id)
        assert job.state == CANCELLED
        assert "forgotten" in job.last_error


def test_an_absent_projection_the_owner_never_erased_is_tried_again(installation):
    """The other cause of the same stored state, and the opposite answer.

    Without the tombstone the absence is a lost submission: the record is still here, its
    coverage is not, and the work is owed. Refusing to move it because the column cannot tell
    the two apart would leave a plain failure looking like an owner's decision.
    """
    record, submission, job_id = a_stranded_submission(installation)
    with EvidenceStore(installation.db_path) as store:
        DocumentMap(store, bank_id=BANK).mark_absent(record, "1")
    report = formation_reconcile(installation, client=Answers())
    assert report["answers"]["refused"] == [submission]
    assert report["jobs_retried"] == 1 and report["jobs_ended"] == 0
    with EvidenceStore(installation.db_path) as store:
        assert JobQueue(store).get(job_id).state == RETRY_WAIT


def test_an_operation_somebody_stopped_is_ended_not_resubmitted(installation):
    """`cancelled` at the backend is a person's decision, and a retry would overrule it.

    The projection ledger records a stop and a failure in the same column, so the distinction
    only exists in the answer this pass heard. That is why the fresh verdict outranks the
    ledger's reading of the same identity: the row is ended the first time the stop is seen, and
    a later pass has no in-flight row left to misclassify.
    """
    record, submission, job_id = a_stranded_submission(installation)
    backend = Answers()
    backend.operation = lambda operation_id: {"status": "cancelled"}
    report = formation_reconcile(installation, client=backend)
    assert report["stopped_operations"] == [submission]
    assert report["answers"]["ended"] == [submission]
    assert report["answers"]["refused"] == [], "the same identity is not also owed again"
    assert report["jobs_ended"] == 1 and report["jobs_retried"] == 0
    with EvidenceStore(installation.db_path) as store:
        assert JobQueue(store).get(job_id).state == CANCELLED


def test_a_stopped_job_is_not_restarted_by_a_denied_answer(installation):
    """A cancellation a person made is not undone by the backend's answer about the attempt."""
    record, submission, job_id = a_stranded_submission(installation)
    with EvidenceStore(installation.db_path) as store:
        JobQueue(store).cancel(job_id, actor=OWNER, reason="superseded by hand")
    backend = Answers()
    backend.operation = lambda operation_id: {"status": "failed"}
    report = formation_reconcile(installation, client=backend)
    assert report["jobs_retried"] == 0 and report["jobs_ended"] == 0
    with EvidenceStore(installation.db_path) as store:
        job = JobQueue(store).get(job_id)
        assert job.state == CANCELLED and job.attempts == 0, "no attempt was counted against a stop"


def test_the_reconcile_report_says_what_is_still_unanswered(installation):
    """A note that always promised more work was a report that could not be finished with.

    The door used to close every run with "a row still in `pending` was asked and is not
    finished; run this again", including the runs where nothing was pending. An operator who
    obeyed got the same sentence back, so the sentence carried no information about whether the
    backlog was any smaller.
    """
    record, submission, job_id = a_stranded_submission(installation)
    backend = Answers()
    backend.operation = lambda operation_id: {"status": "processing"}
    running = formation_reconcile(installation, client=backend)
    assert "1 submission(s) still have no answer" in running["note"]
    assert running["jobs_settled"] == 0 and running["jobs_retried"] == 0
    with EvidenceStore(installation.db_path) as store:
        DocumentMap(store, bank_id=BANK).mark_absent(record, "1")
    settled = formation_reconcile(installation, client=backend)
    assert "run this again" not in settled["note"], "nothing is left unanswered"
    assert settled["jobs_retried"] == 1


def test_reconcile_spends_nothing_on_the_shared_device(installation):
    """It asks about work already done; it does not do any, so it takes no slot."""
    strand(installation)
    with instance_gate(installation) as gate:
        before = gate.usage()
    formation_reconcile(installation, client=Answers())
    with instance_gate(installation) as gate:
        assert gate.usage() == before, "no token and no call was charged to the device"
        assert gate.blocked_resources() == [], "and no slot was left claimed"


def test_reconcile_refuses_an_unbounded_limit_and_an_absent_archive(installation):
    with pytest.raises(FormationError, match="limit must be"):
        formation_reconcile(installation, limit=0, client=Answers())
    missing = replace(installation, db_path=installation.db_path.parent / "nope.db")
    with pytest.raises(FormationError, match="no canonical store"):
        formation_reconcile(missing, client=Answers())


def test_the_pass_queues_for_its_device_with_the_owner_s_number(installation, monkeypatch):
    """HERMES_MEMORY_GATE_QUEUE_S is the owner's word for how long a busy device is worth
    standing in line for, and the pass is the caller that has to honour it. A drain that
    refused on contact would leave the configured wait with nothing to do."""
    given: dict[str, object] = {}
    real = formation.FormationWorker

    def compose(**kwargs):
        given.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(formation, "FormationWorker", compose)
    report = approve(replace(installation, gate_queue_s=77.0), Answers(), limit=1)
    assert report["ok"] is True
    assert given["slot_queue_s"] == 77.0


# -- a standing grant instead of a fresh read --------------------------------

def grant(settings, **kwargs):
    """The owner's bounded permission, issued through the library the CLI door calls.

    Defaults are deliberately generous: a test that matters here is about a *bound*, and says
    so by tightening one number rather than by failing for an unrelated cap.
    """
    arguments = {"records": 20, "tokens": 200_000, "duration_s": 6.0,
                 "resource": ANY_RESOURCE, "reason": "form while I sleep"}
    arguments.update(kwargs)
    with instance_gate(settings) as gate:
        return Allowances(gate.store, owner_principal=settings.owner_principal).grant(
            actor=OWNER, **arguments)


def unattended(settings, client=None, **kwargs):
    """The pass an operator schedules: the grant is the approval, so no digest is read."""
    backend = client or Answers()
    receipt = formation_apply(settings, under_allowance=kwargs.pop("allowance", ANY_RESOURCE),
                              actor=OWNER, client=backend, **kwargs)
    return receipt, backend


def test_a_standing_grant_performs_the_pass_that_needed_a_digest(installation):
    created = grant(installation)
    receipt, backend = unattended(installation, allowance=created["id"])
    assert receipt["drain"]["attempted"] == 3 and len(backend.retain_calls) == 3
    assert formation_plan(installation)["unprojected"] == 0


def test_the_grant_found_by_the_star_is_the_one_the_owner_issued(installation):
    created = grant(installation)
    receipt, _ = unattended(installation)
    assert receipt["allowance"]["id"] == created["id"]


def test_what_a_grant_loses_is_what_the_backend_reported_not_what_was_estimated(installation):
    """The estimate is the ceiling a pass may ask for; the charge is the number it used.

    Three records cost 963 tokens on the answering backend, while the plan priced them at the
    per-job ceiling. A grant debited the estimate would read as nearly spent after a healthy
    pass and refuse the next one for a spend that never happened.
    """
    created = grant(installation, tokens=20_000)
    receipt, _ = unattended(installation, allowance=created["id"])
    assert receipt["allowance"]["tokens"] == {"used": 963, "cap": 20_000, "left": 19_037}
    assert receipt["allowance"]["records"]["used"] == 3
    assert receipt["allowance"]["state"] == "active"
    assert receipt["allowance"]["reason"] == "form while I sleep"


def test_a_grant_bounds_the_drain_and_not_only_the_selection(installation):
    """One record left of a permission must not become every job the queue happens to hold.

    The queue can keep work from an earlier pass that ran out of budget or was superseded. A
    drain under a grant for one more record is a pass over one, whatever else is waiting, and
    the claim `--limit` makes about the size of a pass holds under either door.
    """
    with EvidenceStore(installation.db_path) as store:
        jobs = JobQueue(store)
        route = retain_route(installation)
        for row in store.db.execute("SELECT id, revision FROM records ORDER BY ingested_at "
                                    "LIMIT 3").fetchall():
            jobs.enqueue(kind=KIND, inputs=[row["id"]], input_revision=row["revision"],
                         route=route,
                         processor_fingerprint=processor_fingerprint(installation, route),
                         priority="maintenance", token_budget=4048)
    created = grant(installation, records=1, tokens=200_000)
    receipt, backend = unattended(installation, allowance=created["id"], limit=1)
    assert receipt["drain"]["attempted"] == 1
    assert len(backend.retain_calls) == 1
    assert receipt["allowance"]["records"] == {"used": 1, "cap": 1, "left": 0}
    assert receipt["allowance"]["state"] == "spent"


def test_the_pass_that_ran_under_a_grant_is_attributed_in_the_audit(installation):
    created = grant(installation)
    unattended(installation, allowance=created["id"])
    with EvidenceStore(installation.db_path) as store:
        row = store.db.execute("SELECT action, object_id, metadata FROM audit WHERE "
                              "action='formation_under_allowance'").fetchone()
    assert row is not None and row["object_id"] == created["id"]
    written = json.loads(row["metadata"])
    assert written["actor"] == OWNER and written["records"] == 3
    assert written["tokens"] == 963 and written["resource"] == REMOTE


def test_a_grant_does_not_unlock_a_pass_the_plan_already_refused(installation):
    created = grant(installation)
    with instance_gate(installation) as gate:
        gate.pause(actor=OWNER, reason="the models are being updated")
    with pytest.raises(FormationError, match="refused: all inference is paused"):
        unattended(installation, allowance=created["id"])


def test_a_grant_does_not_outrank_the_daily_ceiling(installation):
    """Two controls, and the day is the one that wins: a grant cannot buy tokens that do not
    exist in it."""
    created = grant(installation)
    with instance_gate(installation) as gate:
        Budgets(gate.store, daily={REMOTE: Budget(tokens=200000)},
                scope="global").charge(REMOTE, tokens=199_500)
    with pytest.raises(FormationError, match="cannot fit one item"):
        unattended(installation, allowance=created["id"])


def test_a_grant_with_records_left_but_a_pass_too_big_for_them_is_refused(installation):
    created = grant(installation, records=2)
    with pytest.raises(FormationError, match="of 2 record\\(s\\) left"):
        unattended(installation, allowance=created["id"])


def test_a_grant_for_another_device_is_not_this_pass_s_permission(installation):
    created = grant(installation, resource="a-quiet-gpu")
    with pytest.raises(FormationError, match="a different device is a different decision"):
        unattended(installation, allowance=created["id"])


def test_an_expired_grant_is_refused_and_moved_out_of_active(installation):
    created = grant(installation)
    with instance_gate(installation) as gate:
        gate.store.db.execute("UPDATE allowances SET expires_at=? WHERE id=?",
                              ("2000-01-01T00:00:00+00:00", created["id"]))
    with pytest.raises(FormationError, match="is expired"):
        unattended(installation, allowance=created["id"])
    with instance_gate(installation) as gate:
        assert gate.store.db.execute("SELECT state FROM allowances WHERE id=?",
                                     (created["id"],)).fetchone()[0] == "expired"


def test_a_refused_grant_leaves_the_archive_exactly_as_it_was(installation):
    created = grant(installation, records=1)
    with EvidenceStore(installation.db_path) as store:
        store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw = installation.db_path.read_bytes()
    backend = Answers()
    with pytest.raises(FormationError, match="refused:"):
        unattended(installation, client=backend, allowance=created["id"])
    assert installation.db_path.read_bytes() == raw
    assert backend.retain_calls == []
    with instance_gate(installation) as gate:
        row = gate.store.db.execute("SELECT used_records, passes FROM allowances WHERE id=?",
                                    (created["id"],)).fetchone()
    assert (row["used_records"], row["passes"]) == (0, 0)


def test_the_two_doors_are_offered_one_at_a_time(installation):
    plan = formation_plan(installation)
    for kwargs in ({}, {"review": plan["review_digest"],
                        "under_allowance": ANY_RESOURCE}):
        with pytest.raises(FormationError, match="one of two ways"):
            formation_apply(installation, actor=OWNER, client=Answers(), **kwargs)


def test_a_live_grant_is_shown_but_does_not_expire_the_approval(installation):
    """The digest is the work; whether a fresh read happens to be needed is not part of it.

    A grant issued while an operator was reading the list would otherwise invalidate the
    approval they were about to type, for a pass that is exactly the one they saw.
    """
    before = formation_plan(installation)
    created = grant(installation)
    after = formation_plan(installation)
    assert after["review_digest"] == before["review_digest"]
    assert after["allowance"]["id"] == created["id"]
    assert created["id"] in after["note"] and "--under-allowance" not in after["note"]
    assert before["allowance"] is None
    assert "no standing allowance was charged" in " ".join(after["not_performed"])


def test_a_granted_pass_still_names_the_plan_it_performed(installation):
    """The receipt carries the digest nobody signed, so the work is reconstructible."""
    plan = formation_plan(installation)
    grant(installation)
    receipt, _ = unattended(installation)
    assert receipt["review_digest"] == plan["review_digest"]
