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
from hermes_memory.backend.document_map import VERIFIED, DocumentMap
from hermes_memory.backend.hindsight_client import HindsightUnavailable
from hermes_memory.config import load_settings
from hermes_memory.ids import backend_document_id
from hermes_memory.processing.budgets import Budget, Budgets
from hermes_memory.processing import formation
from hermes_memory.processing.formation import (DEFAULT_BATCH, KIND, MAX_BATCH, MAX_JOBS,
                                                FormationError, backend_client,
                                                count_unprojected, formation_apply,
                                                formation_plan, formation_reconcile,
                                                processor_fingerprint, retain_route,
                                                unprojected)
from hermes_memory.processing.instance_gate import gate_path, instance_gate
from hermes_memory.processing.jobs import SUCCEEDED, UNCERTAIN, JobQueue
from hermes_memory.storage.evidence import EvidenceStore

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


def test_confirming_a_projection_is_coverage_for_the_epoch_it_was_confirmed_under(
        installation):
    with EvidenceStore(installation.db_path) as store:
        documents = DocumentMap(store, bank_id=BANK)
        for record_id in records(store):
            documents.begin(record_id, "1")
            documents.confirm(record_id, "1")
        store.bump_epoch(reason="reset", actor=OWNER)
        assert count_unprojected(store, bank_id=BANK) == 3
        for record_id in records(store):
            documents.confirm(record_id, "1")
        assert count_unprojected(store, bank_id=BANK) == 0, \
            "a confirmation today is coverage today, not a receipt from an older epoch"


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
