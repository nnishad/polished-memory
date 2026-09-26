"""Operator CLI: readiness per capability, and every reading opened read-only."""
from __future__ import annotations

import json
import os

from pathlib import Path

import pytest

from hermes_memory.cli import main
from hermes_memory.config import load_settings
from hermes_memory.install.profiles import InstallationError, ProfileRegistry
from hermes_memory.ids import now
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.processing.instance_gate import (GATE_FILENAME, GateStore,
                                                   gate_path, instance_gate)
from hermes_memory.storage.evidence import EvidenceStore

SOURCE = "gmail"
OWNER = "jugaadu"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    root = tmp_path / "hm"
    root.mkdir()
    (root / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={tmp_path / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(root))
    monkeypatch.delenv("HERMES_MEMORY_DATA_DIR", raising=False)
    return root


def run(*argv):
    captured = pytest.MonkeyPatch()
    out = []
    captured.setattr("builtins.print", lambda *a, **k: out.append(a[0] if a else ""))
    try:
        code = main(list(argv))
    finally:
        captured.undo()
    return code, json.loads(out[-1])


def errors(*argv):
    captured = pytest.MonkeyPatch()
    written = []
    captured.setattr("sys.stderr.write", lambda text: written.append(text))
    try:
        try:
            code = main(list(argv))
        except SystemExit as exit_code:      # argparse refuses before main can
            code = int(exit_code.code or 2)
    finally:
        captured.undo()
    return code, "".join(written)


# -- status ------------------------------------------------------------------

def test_init_creates_a_store_and_status_reports_each_capability_separately(home):
    assert run("init")[0] == 0
    code, report = run("status")
    assert code == 0
    assert report["capture_only"] is True
    assert report["hindsight_route"] == "unset"
    assert report["owner_principal"] == OWNER
    assert report["capabilities"]["capture"] is True
    assert report["capabilities"]["formation"] is False, "no route and no budget"
    assert report["capabilities"]["forgetting"] is True


def test_status_before_init_says_which_stages_cannot_be_read(home):
    code, report = run("status")
    assert code == 0
    assert report["stages"] is None
    assert report["capabilities"]["capture"] is False
    assert "init" in report["note"]


def test_status_reports_every_stage_without_inventing_one(home):
    run("init")
    report = run("status")[1]["stages"]
    assert [item["name"] for item in report["stages"]] == [
        "capture", "raw_indexing", "observations", "summaries", "goals", "analysis",
        "delivery", "backend", "resource_gate"]
    assert report["overall"] == "unconfigured"
    assert report["erasure_backlog"]["obligations_open"] == 0


def test_an_outstanding_erasure_is_visible_in_status(home, tmp_path):
    run("init")
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        record = _a_message(store)
        store.project(record, "1", backend="hindsight", bank_id="hermes",
                      document_id="hdoc" + record.removeprefix("rec_"))
        manager = ErasureManager(store, owner_principal=OWNER)
        preview = manager.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
        manager.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor=OWNER)
    report = run("status")[1]["stages"]
    assert report["erasure_backlog"]["obligations_open"] == 2
    assert any("erasure" in note for note in report["notes"])


# -- init --------------------------------------------------------------------

def test_init_dry_run_writes_nothing(home):
    code, report = run("init", "--dry-run")
    assert code == 0 and report["would_create"]
    assert not load_settings().db_path.exists()


# -- doctor ------------------------------------------------------------------

def test_doctor_is_read_only_and_names_the_missing_store(home):
    code, report = run("doctor")
    assert code == 1
    assert report["severity"] == "fail"
    assert [item["check"] for item in report["findings"]] == ["layout"]
    assert "hermes-memory init" in report["findings"][0]["remedy"]
    assert not load_settings().db_path.exists()


def test_doctor_distinguishes_capture_from_formation(home):
    run("init")
    code, report = run("doctor")
    assert code == 0
    assert report["probes"] == {"connectivity": False, "synthetic": False}
    checks = {item["check"]: item for item in report["findings"]}
    assert checks["coverage"]["severity"] == "warn", "nothing has been connected yet"
    assert checks["configuration"]["severity"] == "warn"
    assert "no backend URL" in checks["configuration"]["detail"]


def test_doctor_reports_a_healthy_local_installation_without_warnings(home):
    run("init")
    checks = {item["check"]: item for item in run("doctor")[1]["findings"]}
    for name in ("database", "schema", "erasure", "credentials", "leases"):
        assert checks[name]["severity"] == "ok"


def test_an_outstanding_erasure_obligation_makes_doctor_fail(home):
    """Forgetting is not finished while a derived copy still exists elsewhere."""
    run("init")
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        record = _a_message(store)
        store.project(record, "1", backend="hindsight", bank_id="hermes",
                      document_id="hdoc" + record.removeprefix("rec_"))
        manager = ErasureManager(store, owner_principal=OWNER)
        preview = manager.preview(record_ids=[record], actor=OWNER, reason="withdrawn")
        manager.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor=OWNER)
    code, report = run("doctor")
    assert code == 1, "an unfinished erasure must not report a clean bill"
    erasure = next(item for item in report["findings"] if item["check"] == "erasure")
    assert erasure["severity"] == "fail"
    assert erasure["obligations_open"] == 2


def test_an_unnamed_owner_makes_forgetting_unconfirmable(home, tmp_path):
    env = home / "hermes-memory.env"
    env.write_text(env.read_text(encoding="utf-8").replace(f"={OWNER}", "="),
                   encoding="utf-8")
    run("init")
    code, report = run("doctor")
    assert code == 0
    configuration = next(item for item in report["findings"]
                         if item["check"] == "configuration")
    assert "no owner principal" in configuration["detail"]
    capabilities = run("status")[1]["capabilities"]
    assert capabilities["capture"] is True, "capture does not need an owner"
    assert capabilities["forgetting"] is False
    assert capabilities["delivery"] is False


def test_doctor_never_writes_to_an_existing_store(home):
    run("init")
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        _a_message(store)
        tables = [row[0] for row in store.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        before = {table: store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in tables}
    run("doctor")
    with EvidenceStore(settings.db_path) as store:
        after = {table: store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                 for table in tables}
    assert after == before


def test_a_probe_is_only_run_when_it_is_named(home, monkeypatch):
    run("init")
    reached = []
    monkeypatch.setattr("hermes_memory.backend.hindsight_client.HindsightClient.health",
                        lambda self: reached.append("health") or {"version": "0.5.13"})
    code, report = run("doctor")
    assert reached == [], "the default doctor must not open a socket"
    assert "backend-connectivity" not in [item["check"] for item in report["findings"]]


def test_every_reading_works_on_a_store_that_cannot_be_written(home):
    """A reading that needed write access would fail here, and that is the point.

    Opening the store the way a component does applies pending migrations and sets
    ``user_version``, both of which are writes. The readings must not do either.
    """
    run("init")
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        record = _a_message(store)
    os.chmod(settings.db_path, 0o444)
    try:
        assert run("status")[0] == 0
        assert run("audit")[0] == 0
        assert run("explain", "--record", record)[0] == 0
        assert run("doctor")[0] == 0
    finally:
        os.chmod(settings.db_path, 0o644)


# -- audit and explain -------------------------------------------------------

def test_audit_reads_the_ledger_the_store_already_wrote(home):
    run("init")
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        record = _a_message(store)
        store.hide(record, reason="mine only", actor=OWNER)
    code, entries = run("audit", "--object", record)
    assert code == 0
    assert [item["action"] for item in entries] == ["evidence_hide", "evidence_commit"]
    assert run("audit", "--action", "evidence_hide")[1][0]["actor"] == OWNER


def test_audit_filters_that_match_nothing_answer_with_nothing(home):
    run("init")
    assert run("audit", "--action", "never_happened")[1] == []


def test_an_unregistered_source_is_refused_rather_than_reported_as_idle(home):
    run("init")
    code, message = errors("audit", "--source", "gmail")
    assert code == 2
    assert "no connector named" in message


def test_explain_answers_why_a_record_came_back(home):
    run("init")
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        record = _a_message(store)
    code, report = run("explain", "--record", record)
    assert code == 0
    assert report["retrievable"] is True
    assert [item["gate"] for item in report["gates"]] == ["not forgotten",
                                                          "visible to retrieval",
                                                          "in the search index"]


def test_explain_never_prints_evidence_text_unasked(home):
    run("init")
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        record = _a_message(store, text="my invoice number is 4471")
    assert "4471" not in json.dumps(run("explain", "--record", record)[1])


def test_an_unknown_id_is_a_refusal_with_an_exit_code(home):
    run("init")
    code, message = errors("explain", "--record", "rec_nope")
    assert code == 2
    assert "no record" in message


def test_explain_lists_what_was_decided_against(home):
    run("init")
    code, report = run("explain", "--not-told", "general")
    assert code == 0
    assert report["decided_against"] == []
    assert report["policy"]["state"] == "allowed"


# -- the shape of the tool ---------------------------------------------------

def test_a_refused_route_exits_two(home):
    (home / "hermes-memory.env").write_text(
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_HINDSIGHT_URL=https://api.openai.com/v1\n", encoding="utf-8")
    code, message = errors("status")
    assert code == 2
    assert "configuration refused" in message


def test_every_command_needs_to_be_one_home(home):
    code, message = errors()
    assert code == 2
    assert "the following arguments are required: command" in message


def _a_message(store, text="invoice 42 is paid", **overrides):
    payload = {"source": SOURCE, "source_id": overrides.get("source_id", "msg-1"),
               "revision": "1", "kind": "email", "text": text, "observed_at": now(),
               "occurred_at": None, "occurred_precision": "unknown", "metadata": {}}
    payload.update({key: value for key, value in overrides.items() if key != "source_id"})
    return store.commit(payload)["id"]


# -- the profile map -----------------------------------------------------------

def test_listing_profiles_creates_no_ledger(home):
    code, report = run("profiles")
    assert code == 0 and report["profiles"] == []
    assert not (home / "installation.db").exists(), \
        "a listing that wrote the ledger it was reporting would change the installation"


def test_a_plan_creates_no_ledger_either(home):
    run("enroll", "--hermes-home", str(home / "profiles" / "work"))
    assert not (home / "installation.db").exists()


def test_an_approval_against_no_ledger_writes_to_none_of_it(home):
    """A detached view is empty, and writing to an empty view must not look like a
    success that evaporated with the process."""
    plan = run("enroll", "--hermes-home", str(home / "profiles" / "work"))[1]
    settings = load_settings()
    registry = ProfileRegistry.detached(root=settings.home,
                                       owner_principal=settings.owner_principal,
                                       default_home=settings.data_dir)
    with pytest.raises(InstallationError, match="no ledger to write to"):
        registry.enroll("work", home / "profiles" / "work", actor="jugaadu",
                        review_digest=plan["review_digest"])
    registry.db.close()


def test_enrollment_is_two_steps_and_the_first_writes_nothing(home):
    code, plan = run("enroll", "--hermes-home", str(home / "profiles" / "work"))
    assert code == 0
    assert plan["profile"] == "work" and plan["data_dir"] == str(home / "profiles" / "work")
    assert plan["bank_id"] == "hermes-work" and plan["credential_scope"] == "profile-work"
    assert "--review" in plan["next"]
    assert run("profiles")[1]["profiles"] == []

    approved = run("enroll", "--hermes-home", str(home / "profiles" / "work"),
                   "--review", plan["review_digest"])
    assert approved[0] == 0 and approved[1]["changed"] is True
    listed = run("profiles")[1]
    assert [item["profile"] for item in listed["profiles"]] == ["work"]
    assert listed["profiles"][0]["hermes_home"] == str(home / "profiles" / "work")
    assert listed["store_present"] == {"work": False}, \
        "enrollment maps a profile to memory; it does not create it"


def test_an_approval_that_was_never_shown_is_refused(home):
    run("enroll", "--hermes-home", str(home / "profiles" / "work"))
    errors = _capture_stderr()
    code = _capture_exit(["enroll", "--hermes-home", str(home / "profiles" / "work"),
                          "--review", "0" * 64], errors)
    assert code == 2
    assert any("does not match" in line for line in errors)
    assert run("profiles")[1]["profiles"] == []


def test_an_agent_cannot_enroll_a_home_it_named(home):
    errors = _capture_stderr()
    code = _capture_exit(["enroll", "--hermes-home", str(home / "profiles" / "sneaky"),
                          "--review", "0" * 64, "--actor", "agent"], errors)
    assert code == 2 and any("only the owner" in line for line in errors)


def test_no_owner_configured_means_no_one_can_approve(home):
    env = home / "hermes-memory.env"
    env.write_text(env.read_text(encoding="utf-8").replace("HERMES_MEMORY_OWNER_PRINCIPAL=jugaadu",
                                                           "HERMES_MEMORY_OWNER_PRINCIPAL="),
                   encoding="utf-8")
    plan = run("enroll", "--hermes-home", str(home / "profiles" / "work"))[1]
    errors = _capture_stderr()
    code = _capture_exit(["enroll", "--hermes-home", str(home / "profiles" / "work"),
                          "--review", plan["review_digest"]], errors)
    assert code == 2 and any("no owner principal is configured" in line for line in errors)


def test_retiring_a_profile_keeps_its_evidence_and_stops_serving_it(home, tmp_path):
    plan = run("enroll", "--hermes-home", str(home / "profiles" / "work"))[1]
    run("enroll", "--hermes-home", str(home / "profiles" / "work"),
        "--review", plan["review_digest"])
    store = Path(plan["data_dir"]) / "canonical.db"
    store.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    store.write_text("private evidence", encoding="utf-8")

    code, report = run("retire", "--profile", "work", "--reason", "machine retired")
    assert code == 0 and report["changed"] is True
    assert store.read_text(encoding="utf-8") == "private evidence"
    listed = run("profiles")[1]
    assert listed["profiles"] == []
    assert [item["profile"] for item in listed["retired"]] == ["work"]

    refused = _capture_stderr()
    assert _capture_exit(["retire", "--profile", "ghost", "--reason", "nothing"],
                         refused) == 2


def test_a_default_home_enrolls_as_the_default_profile(home):
    code, plan = run("enroll", "--hermes-home", str(home / "profiles" / "home"))
    assert code == 0 and plan["profile"] == "default"
    assert plan["data_dir"] == str(home.parent / "data"), \
        "the default profile keeps the configured data directory"


def test_status_reports_delivery_only_when_an_owner_chose_a_destination(home, monkeypatch):
    assert run("status")[1]["capabilities"]["delivery"] is False
    env = home / "hermes-memory.env"
    env.write_text(env.read_text(encoding="utf-8")
                   + "HERMES_MEMORY_DELIVERY_ENABLED=true\n"
                   "HERMES_MEMORY_DELIVERY_TARGET=signal:owner-1234\n", encoding="utf-8")
    run("init")
    assert run("status")[1]["capabilities"]["delivery"] is True


def _capture_stderr():
    return []


def _capture_exit(argv, errors):
    capture = pytest.MonkeyPatch()
    capture.setattr("sys.stderr.write", lambda text: errors.append(text))
    try:
        return main(list(argv))
    finally:
        capture.undo()


# -- inventory -----------------------------------------------------------------

def test_inventory_reports_the_installation_without_changing_it(home):
    code, report = run("inventory", "--hermes-home", str(home / ".." / "hermes"))
    assert code == 0
    assert report["installation"]["home"] == str(home)
    assert report["installation"]["capture_only"] is True
    assert report["installation"]["store_present"] is False
    assert report["endpoints"]["probed"] is False
    assert report["host"]["guarded_delivery_supported"] is False
    assert report["unknowns"], "an inventory that admits nothing it could not read is guessing"
    assert not (home / "installation.db").exists()


def test_the_conflicts_view_is_the_one_a_script_gates_on(home):
    code, report = run("inventory")
    assert code == 0 and report["conflicts"]
    errors = _capture_stderr()
    assert _capture_exit(["inventory", "--conflicts"], errors) == 1, \
        "a blocking fact must be visible in the exit status"


def test_a_clean_installation_reports_no_conflicts(home):
    run("init")
    (home / "hermes-memory.env").chmod(0o600)
    _, report = run("inventory")
    assert report["conflicts"] == [], report["conflicts"]


# -- the services and the fence ------------------------------------------------

@pytest.fixture()
def service_home(home, tmp_path, monkeypatch):
    """The same installation, with a config home these tests own.

    Without this the unit directory is the real account's, and a test that writes a
    unit file would leave it behind for the next boot to start.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return home


@pytest.fixture()
def runner(monkeypatch):
    """A systemctl stand-in, because a test must not start or stop anything real."""
    calls = []

    def execute(argv):
        calls.append(list(argv))
        return 0, ""

    monkeypatch.setattr("hermes_memory.cli._HOST_RUNNER", execute)
    return calls


def test_services_prints_the_plan_and_writes_nothing(service_home):
    code, report = run("services")
    assert code == 0
    assert [entry["state"] for entry in report["units"]] == \
        ["written", "not-wanted", "not-wanted"]
    assert report["would_change"] == ["hermes-memory.service"]
    assert report["next"] == (
        f"hermes-memory services --install {report['review_digest']}")
    assert not Path(report["unit_dir"]).exists()


def test_installing_a_service_needs_the_digest_that_was_shown(service_home):
    code, message = errors("services", "--install", "0" * 64)
    assert code == 2 and "does not match" in message
    assert not (service_home.parent / "config" / "systemd" / "user"
                / "hermes-memory.service").exists()


def test_an_approved_plan_writes_the_owned_unit_and_then_has_nothing_to_change(service_home):
    _, proposal = run("services")
    code, receipt = run("services", "--install", proposal["review_digest"],
                        "--actor", OWNER)
    assert code == 0
    assert receipt["units_written"] == ["hermes-memory.service"]
    assert receipt["reload_needed"] is True
    unit = Path(receipt["unit_dir"]) / "hermes-memory.service"
    assert unit.is_file() and f"Environment=HERMES_MEMORY_HOME={service_home}" in \
        unit.read_text(encoding="utf-8")
    _, again = run("services")
    assert again["would_change"] == []
    assert "nothing to write" in again["next"]


def test_a_unit_that_is_not_ours_stops_both_the_install_and_the_start(service_home, runner):
    from hermes_memory.install.services import unit_directory

    directory = unit_directory()
    directory.mkdir(parents=True, mode=0o700)
    (directory / "hermes-memory.service").write_text("[Service]\nExecStart=/opt/theirs\n",
                                                     encoding="utf-8")
    _, report = run("services")
    assert report["blocked"] == ["hermes-memory.service"]
    code, message = errors("start")
    assert code == 2 and "not ours to start" in message
    assert runner == []


def test_start_goes_gate_then_backend_then_worker(service_home, runner):
    (service_home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={service_home.parent / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_HINDSIGHT_URL=http://127.0.0.1:8888\n"
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS=127.0.0.1\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n", encoding="utf-8")
    code, report = run("start")
    assert code == 0
    assert report["started"] == ["hermes-memory.service", "hermes-memory-hindsight.service",
                                 "hermes-memory-worker.service"]
    assert [call[2:] for call in runner] == [["start", name] for name in report["started"]]
    # Nothing was installed by this test, so the units on disk are not what this
    # installation would write - and a start that stayed quiet about that would let a
    # half-configured machine look healthy.
    assert report["units_not_as_described"] == report["started"]


def test_stopping_holds_the_fence_first_and_the_hold_outlives_the_process(
        service_home, runner):
    run("init")
    code, report = run("stop", "--actor", OWNER, "--reason", "overnight maintenance")
    assert code == 0
    assert report["paused"] == ["inference", "delivery"]
    assert report["stopped"] == ["hermes-memory.service"]
    assert runner == [["systemctl", "--user", "stop", "hermes-memory.service"]]
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        assert store.stage_is_paused("global", "delivery") is True
    with instance_gate(settings) as gate:
        assert gate.paused is True, "the hold on the shared models is in one ledger"


def test_stopping_holds_delivery_for_every_profile_not_just_the_first(service_home,
                                                                       runner,
                                                                       tmp_path):
    """An owner's stop is one decision about one machine.

    The delivery outbox is per-profile, so the hold has to be written on each enrolled
    archive; holding only the first one would leave the second person's drafts queued for
    a morning that was supposed to be quiet.
    """
    run("init")
    settings = load_settings()
    stores = {}
    registry = ProfileRegistry.open(settings)
    for name in ("work", "personal"):
        home = tmp_path / "homes" / name
        home.mkdir(parents=True)
        proposal = registry.plan(name, home)
        registry.enroll(name, home, actor=OWNER, review_digest=proposal["review_digest"])
        store_at = Path(proposal["data_dir"]) / "canonical.db"
        with EvidenceStore(store_at):
            pass
        stores[name] = store_at
    registry.db.close()

    code, report = run("stop", "--actor", OWNER, "--reason", "overnight maintenance")
    assert code == 0 and report["paused"] == ["inference", "delivery"]
    assert sorted(report["held_for"]) == ["personal", "work"]
    for name, path in stores.items():
        with EvidenceStore(path) as scoped:
            assert scoped.stage_is_paused("global", "delivery") is True, name


def test_stopping_without_an_owner_touches_no_process(service_home, tmp_path, monkeypatch,
                                                      runner):
    other = tmp_path / "no-owner"
    other.mkdir()
    (other / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={other / 'data'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(other))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(other / "config"))
    code, message = errors("stop")
    assert code == 2 and "owner" in message
    assert runner == []


def test_a_pause_is_recorded_with_the_actor_who_set_it_and_can_be_lifted(service_home):
    run("init")
    code, report = run("pause", "--scope", "delivery", "--actor", OWNER,
                       "--reason", "the owner is away")
    assert code == 0
    assert report["held"] == {"inference": False, "delivery": True}
    assert report["survives_a_restart"] is True
    _, lifted = run("pause", "--scope", "delivery", "--resume", "--actor", OWNER)
    assert lifted["held"] == {"inference": False, "delivery": False}

    run("pause", "--scope", "inference", "--actor", OWNER, "--reason", "upgraded today")
    _, both = run("pause", "--scope", "delivery", "--actor", OWNER)
    assert both["held"] == {"inference": True, "delivery": True}, \
        "the report says what is held now, not what this one command just did"
    _, cleared = run("pause", "--scope", "inference", "--resume", "--actor", OWNER)
    assert cleared["held"]["inference"] is False


def test_a_pause_needs_somebody_to_be_answerable_for(service_home, tmp_path, monkeypatch):
    other = tmp_path / "unowned"
    other.mkdir()
    (other / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={other / 'data'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(other))
    code, message = errors("pause", "--scope", "inference")
    assert code == 2 and "owner" in message


def test_serve_refuses_before_binding_when_it_has_no_address(service_home):
    code, message = errors("serve")
    assert code == 2 and "ADMISSION_URL" in message


# -- the setup transaction ------------------------------------------------------

@pytest.fixture()
def staged(home, tmp_path, monkeypatch):
    """The same installation with a release actually staged where the units point."""
    release = home / "runtime" / "current"
    (release / "bin").mkdir(parents=True)
    (release / "bin" / "hermes-memory").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    activity = tmp_path / "homes" / "work"
    activity.mkdir(parents=True)
    (activity / "config.yaml").write_text(
        "model:\n  provider: openai\n  model: deepseek-chat\n\n"
        "memory:\n  memory_enabled: true\n", encoding="utf-8")
    calls: list[list[str]] = []

    def execute(argv):
        calls.append(list(argv))
        if argv[1:3] == ["config", "set"]:
            (activity / "config.yaml").write_text(
                (activity / "config.yaml").read_text(encoding="utf-8")
                + "memory:\n  provider: hermes-memory\n", encoding="utf-8")
        return 0, ""

    monkeypatch.setattr("hermes_memory.cli._HOST_RUNNER", execute)
    return activity, calls


def test_setup_prints_the_eleven_steps_and_writes_nothing(home, staged):
    activity, calls = staged
    code, report = run("setup", "--hermes-home", str(activity))
    assert code == 0
    assert [item["step"] for item in report["steps"]][:3] == ["inventory", "plan", "stage"]
    assert len(report["steps"]) == 11
    assert report["next"].endswith(report["review_digest"])
    assert calls == []
    assert not (home / "installation.db").exists()


def test_setup_needs_a_home_to_be_named(home, tmp_path):
    """argparse refuses before the installation is read at all."""
    code, message = errors("setup")
    assert code == 2 and "hermes-home" in message


def test_setup_refuses_an_approval_that_is_not_the_plan_now(home, staged):
    activity, _ = staged
    code, message = errors("setup", "--hermes-home", str(activity), "--review", "0" * 64)
    assert code == 2 and "does not match" in message
    assert not (home / "installation.db").exists()


def test_the_operator_cli_can_act_on_the_machine_it_runs_on():
    """The default executor is the real one, and saying so is a testable claim.

    Every other test hands in a recorder; if the production default were "nobody", a
    setup run from a shell would quietly refuse to finish and nothing here would notice.
    """
    from hermes_memory.cli import _HOST_RUNNER
    from hermes_memory.install.services import subprocess_runner

    assert _HOST_RUNNER is subprocess_runner


def test_a_home_that_cannot_be_resolved_is_refused_before_anything_is_read(home, staged,
                                                                          monkeypatch):
    monkeypatch.setattr("hermes_memory.cli._HOST_RUNNER", lambda argv: (0, ""))
    code, message = errors("setup", "--hermes-home", "homes/work")
    assert code == 2 and "absolute" in message


def test_an_approved_setup_enrolls_the_profile_and_selects_the_provider(home, staged):
    activity, calls = staged
    _, proposal = run("setup", "--hermes-home", str(activity),
                      "--ref", "a" * 40)
    code, receipt = run("setup", "--hermes-home", str(activity), "--ref", "a" * 40,
                        "--review", proposal["review_digest"])
    assert code == 0, receipt
    assert receipt["done"][0] == "inventory" and receipt["done"][-1] == "finish"
    assert [call[1:3] for call in calls] == [["plugins", "install"], ["plugins", "enable"],
                                             ["config", "set"], ["--user", "daemon-reload"]]
    assert "provider: hermes-memory" in (activity / "config.yaml").read_text(encoding="utf-8")
    assert "model: deepseek-chat" in (activity / "config.yaml").read_text(encoding="utf-8")
    _, profiles = run("profiles")
    assert [item["profile"] for item in profiles["profiles"]] == ["work"]


# -- the archive, the release and the sources ---------------------------------

@pytest.fixture()
def exports(tmp_path):
    """A directory of exports the owner pointed at, in their own words."""
    root = tmp_path / "exports"
    root.mkdir()
    (root / "lease.md").write_text("The lease on the flat ends in March.\n",
                                   encoding="utf-8")
    (root / "move.md").write_text("The meeting moved to Thursday.\n", encoding="utf-8")
    return root


def test_backup_copies_the_store_and_says_whether_the_copy_verifies(home):
    run("init")
    code, receipt = run("backup", "--reason", "before the switch")
    assert code == 0
    made = receipt["backups"][0]
    assert made["verified"] is True and made["snapshot"]["reason"] == "before the switch"
    assert Path(made["snapshot"]["directory"]).is_dir()


def test_backup_names_every_profile_it_copied(home):
    run("init")
    registry = ProfileRegistry.open(load_settings())
    registry.db.close()
    _, receipt = run("backup", "--reason", "all of them")
    assert [item["profile"] for item in receipt["backups"]] == ["default"]
    assert receipt["skipped"] == []


def test_a_backup_reports_what_the_verifier_actually_found(home, monkeypatch):
    """The claim is the verifier's answer, not a constant in the report."""
    from hermes_memory.lifecycle.snapshots import Snapshots

    run("init")
    monkeypatch.setattr(Snapshots, "verify", lambda self, snapshot_id: {"ok": False})
    code, receipt = run("backup", "--reason", "unverified")
    assert code == 0
    assert receipt["backups"][0]["verified"] is False


def test_a_store_that_is_not_there_yet_is_skipped_rather_than_created(home, tmp_path):
    code, receipt = run("backup", "--reason", "nothing to copy")
    assert code == 0
    assert receipt["backups"] == []
    assert "no store at" in receipt["skipped"][0]["reason"]
    assert not (tmp_path / "data" / "canonical.db").exists()


def test_a_backup_needs_somebody_to_be_answerable_for(tmp_path, monkeypatch):
    root = tmp_path / "unowned"
    (root / "data").mkdir(parents=True)
    (root / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={root / 'data'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(root))
    run("init")
    code, message = errors("backup")
    assert code == 2 and "owner" in message


def test_a_backup_can_be_read_back_without_taking_another_one(home):
    run("init")
    _, made = run("backup", "--reason", "first")
    code, listing = run("backup", "--list")
    assert code == 0
    assert [item["id"] for item in listing["profiles"][0]["snapshots"]] == \
        [made["backups"][0]["snapshot"]["id"]]
    assert len(listing["profiles"][0]["snapshots"]) == 1, "listing took nothing"


def test_an_upgrade_plan_is_read_only_even_though_it_is_called_upgrade(home, staged):
    activity, calls = staged
    code, report = run("upgrade", "--version", str(home / "runtime" / "current"),
                       "--hermes-home", str(activity))
    assert code == 0 and calls == []
    assert report["target"]["executable"].endswith("bin/hermes-memory")
    assert not (home / "installation.db").exists()
    assert report["not_performed"] and "performs them" in report["note"]


def test_an_upgrade_with_no_target_named_is_a_hop_not_a_switch(home, service_home):
    code, report = run("upgrade")
    assert code == 0
    assert report["blocking"] and "no target release" in report["blocking"][0]
    assert report["target_release"] is None


def test_a_release_tree_missing_its_plugin_is_only_half_an_upgrade(home, staged):
    activity, _ = staged
    _code, report = run("upgrade", "--version", str(home / "runtime" / "current"),
                        "--hermes-home", str(activity))
    assert any("plugin" in line for line in report["blocking"])
    assert report["target"]["runnable"] is False


def test_the_uninstall_list_is_printed_before_anything_is_removed(home, service_home,
                                                                  runner):
    run("init")
    _, written = run("services")
    run("services", "--install", written["review_digest"], "--actor", OWNER)
    code, proposal = run("uninstall", "--keep-data")
    assert code == 0
    assert [item["unit"] for item in proposal["removals"]] == ["hermes-memory.service"]
    assert proposal["purging_data"] is False
    assert proposal["review_digest"] in proposal["next"], \
        "the printed next step approves this exact list"
    assert proposal["next"].endswith(f"--actor {OWNER}")
    assert Path(proposal["removals"][0]["path"]).exists(), "a plan removes nothing"
    assert runner == []


def test_an_uninstall_without_keep_data_is_refused_by_name(home, service_home, runner):
    run("init")
    _, written = run("services")
    run("services", "--install", written["review_digest"], "--actor", OWNER)
    _code, proposal = run("uninstall")
    code, message = errors("uninstall", "--review", proposal["review_digest"],
                           "--actor", OWNER)
    assert code == 2
    assert "no flag here that removes them" in message
    assert Path(proposal["removals"][0]["path"]).exists()


def test_an_approved_uninstall_removes_the_units_and_keeps_the_memory(home, service_home,
                                                                      runner):
    run("init")
    _, written = run("services")
    run("services", "--install", written["review_digest"], "--actor", OWNER)
    _code, proposal = run("uninstall", "--keep-data")
    code, receipt = run("uninstall", "--keep-data",
                        "--review", proposal["review_digest"], "--actor", OWNER)
    assert code == 0, receipt
    assert [call[1:3] for call in runner] == [["plugins", "disable"], ["plugins", "remove"]]
    assert all(not Path(path).exists() for path in receipt["removed"])
    assert (home / "hermes-memory.env").is_file()
    assert str(Path(home / "hermes-memory.env")) in " ".join(receipt["kept"])
    assert not any(path.endswith(".service") for path in receipt["kept"])


def test_the_uninstall_gives_back_a_selection_the_owner_has_not_changed(home, staged):
    """The whole transaction, then its undo, on one machine nobody else can see."""
    activity, calls = staged
    _code, proposal = run("setup", "--hermes-home", str(activity), "--ref", "a" * 40)
    run("setup", "--hermes-home", str(activity), "--ref", "a" * 40,
        "--review", proposal["review_digest"])
    assert "provider: hermes-memory" in (activity / "config.yaml").read_text(encoding="utf-8")

    def undoing(argv):
        calls.append(list(argv))
        if argv[1:3] == ["config", "unset"]:
            (activity / "config.yaml").write_text(
                (activity / "config.yaml").read_text(encoding="utf-8")
                .replace("  provider: hermes-memory\n", ""), encoding="utf-8")
        return 0, ""

    calls.clear()
    monkey = pytest.MonkeyPatch()
    monkey.setattr("hermes_memory.cli._HOST_RUNNER", undoing)
    try:
        _code, plan = run("uninstall", "--keep-data", "--hermes-home", str(activity))
        assert plan["provider"]["command"] == ["hermes", "config", "unset",
                                               "memory.provider"]
        code, receipt = run("uninstall", "--keep-data", "--hermes-home", str(activity),
                            "--review", plan["review_digest"], "--actor", OWNER)
    finally:
        monkey.undo()
    assert code == 0, receipt
    assert [call[1:3] for call in calls] == [["config", "unset"], ["plugins", "disable"],
                                             ["plugins", "remove"]]
    assert "provider: hermes-memory" not in (activity / "config.yaml").read_text(
        encoding="utf-8")
    assert (home / "installation.db").is_file(), "the map to the memory outlives its owner"


def test_an_uninstall_approval_cannot_be_spent_on_a_different_list(home, service_home,
                                                                   runner):
    run("init")
    _code, proposal = run("uninstall", "--keep-data")
    code, message = errors("uninstall", "--keep-data", "--review", "0" * 64,
                           "--actor", OWNER)
    assert code == 2 and "does not match" in message


def test_sources_list_reads_the_connectors_own_account(home, exports):
    run("init")
    _code, empty = run("sources", "list")
    assert empty["sources"] == []
    run("import", "--source", "files", "--path", str(exports))
    code, report = run("sources", "list", "--gaps")
    assert code == 0
    assert report["registered"] == 1
    entry = report["sources"][0]
    assert entry["source"] == "files" and entry["policy"] == "local-only"
    assert entry["coverage"] == "current" and entry["open_gaps"] == []


def test_sources_is_a_verb_and_not_a_free_text_argument(home):
    code, message = errors("sources", "show")
    assert code == 2 and "invalid choice" in message


def test_an_import_reads_a_directory_the_owner_named(home, exports):
    run("init")
    code, report = run("import", "--source", "files", "--path", str(exports))
    assert code == 0
    assert report["stopped"] == "complete" and report["records"] == 2
    assert report["coverage_state"] == "current"


def test_an_import_twice_is_one_import(home, exports):
    run("init")
    run("import", "--source", "files", "--path", str(exports))
    _code, again = run("import", "--source", "files", "--path", str(exports))
    assert again["records"] == 0 and again["repeats"] >= 1


def test_a_dry_run_import_reads_nothing_at_all(home, exports):
    run("init")
    _code, before = run("sources", "list")
    code, report = run("import", "--source", "files", "--path", str(exports), "--dry-run")
    assert code == 0
    assert report["reachable"]["ok"] is True
    assert report["reachable"]["content_read"] is False
    assert run("sources", "list")[1]["registered"] == before["registered"]


def test_an_import_refuses_to_invent_a_connector(home, exports):
    run("init")
    code, message = errors("import", "--source", "gmail", "--path", str(exports))
    assert code == 2 and "no export reader" in message
    assert "files" in message


def test_an_import_will_not_create_the_store_it_writes_into(home, exports):
    code, message = errors("import", "--source", "files", "--path", str(exports))
    assert code == 2 and "init" in message
