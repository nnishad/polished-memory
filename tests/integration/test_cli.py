"""Operator CLI: readiness per capability, and every reading opened read-only."""
from __future__ import annotations

import json
import os
import shutil
from types import SimpleNamespace

from pathlib import Path

import pytest

from hermes_memory.cli import RESTORE_PLAN_VERSION, main
from hermes_memory.config import load_settings
from hermes_memory.install.release import carry
from hermes_memory.install.profiles import InstallationError, ProfileRegistry
from hermes_memory.knowledge.assertions import AssertionStore
from hermes_memory.backend.document_map import VERIFIED, DocumentMap
from hermes_memory.backend.hindsight_client import HindsightUnavailable
from hermes_memory.ids import digest, now
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.processing.instance_gate import (GATE_FILENAME, GateStore,
                                                   gate_path, instance_gate)
from hermes_memory.processing.jobs import JobQueue
from hermes_memory.processing.routes import Route
from hermes_memory.sources.sync import SyncController
from hermes_memory.storage.evidence import EvidenceStore, ReadOnlyStore
from hermes_memory.storage.identity import IdentityStore

from conftest import envelope

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
        DocumentMap(store).begin(record, "1")
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
        DocumentMap(store).begin(record, "1")
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
        # A marker that cannot occur by accident in the paths this report prints: the
        # store path carries pytest's run number, and a plain digit sequence would
        # eventually collide with it.
        record = _a_message(store, text="my invoice number is qz4471km")
    assert "qz4471km" not in json.dumps(run("explain", "--record", record)[1])


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


def test_the_profiles_reading_names_the_decisions_behind_the_map(home):
    """The map says what is enrolled; it does not say who said so or under which plan.

    Both are on the instance ledger already, so a listing that stops at the current
    state leaves the owner with no way to ask what an earlier approval did.
    """
    _, plan = run("enroll", "--hermes-home", str(home / "profiles" / "work"))
    run("enroll", "--hermes-home", str(home / "profiles" / "work"),
        "--review", plan["review_digest"], "--actor", OWNER)

    listed = run("profiles")[1]
    assert [(row["action"], row["actor"], row["profile"]) for row in listed["changes"]] == [
        ("enroll", OWNER, "work")]
    assert listed["changes"][0]["review_digest"] == plan["review_digest"]
    assert run("profiles", "--profile", "elsewhere")[1]["changes"] == [], \
        "one profile's history is not the whole ledger"

    traced = run("profiles", "--review", plan["review_digest"])[1]
    assert traced["applied"] == {
        "id": listed["changes"][0]["id"], "profile": "work", "action": "enroll",
        "actor": OWNER, "at": listed["changes"][0]["at"]}
    assert run("profiles", "--review", "0" * 64)[1]["applied"] is None, \
        "a digest nobody approved is answered as such, not as an error"


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


def test_an_enrolled_profile_gets_its_own_store_when_init_is_asked_for_it(home):
    """Enrollment maps a person to a memory; `init` is what opens it.

    Without this door the only way to create a second profile's store was the setup
    transaction, which also registers the plugin with the host — a bigger decision than
    "make the database this profile was mapped to".
    """
    plan = run("enroll", "--hermes-home", str(home / "profiles" / "work"))[1]
    run("enroll", "--hermes-home", str(home / "profiles" / "work"),
        "--review", plan["review_digest"])
    assert run("profiles")[1]["store_present"] == {"work": False}

    code, made = run("init", "--hermes-home", str(home / "profiles" / "work"))
    assert code == 0 and made["profile"] == "work"
    store = Path(made["initialized"])
    assert store.is_file() and store.name == "canonical.db"
    assert run("profiles")[1]["store_present"] == {"work": True}
    assert store.parent.joinpath("blobs").is_dir() or store.parent.parent.joinpath(
        "blobs").is_dir(), "a store with nowhere for an attachment is a doctor failure later"


def test_init_refuses_to_open_a_store_for_a_home_nobody_enrolled(home):
    run("init")
    code, message = errors("init", "--hermes-home", str(home / "profiles" / "never"))
    assert code == 2 and "no profile is enrolled" in message


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


def test_advice_about_an_installation_does_not_gate_a_script(home, capsys):
    """`--conflicts` is the exit status a script gates on, so it may only carry blockers.

    This fixture has something worth saying about itself and nothing that stops a setup run,
    and the flag used to fail on exactly that — a healthy installation reporting itself
    blocked to the one command written to answer the question.
    """
    code, report = run("inventory")
    assert code == 0 and report["conflicts"], "there is advice to be had here"
    assert _capture_exit(["inventory", "--conflicts"], _capture_stderr()) == 0, \
        "advice is not a reason not to run setup"


def test_a_blocking_fact_is_the_only_kind_that_moves_that_exit_status(home, capsys,
                                                                     monkeypatch):
    """The mapping from the two kinds of sentence to the flag's output and status."""
    from hermes_memory.install import inventory

    monkeypatch.setattr(inventory, "conflicts", lambda report: [
        "the hindsight endpoint wants port 8080, which something is already listening on",
        f"{home / 'hermes-memory.env'} is readable beyond its owner and holds the route "
        "configuration"])
    errors = _capture_stderr()
    assert _capture_exit(["inventory", "--conflicts"], errors) == 1, \
        "a blocking fact must be visible in the exit status"
    printed = json.loads(capsys.readouterr().out)
    assert printed == ["the hindsight endpoint wants port 8080, which something is already "
                       "listening on"], "the flag prints only the reasons a run is a bad idea"


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


def test_a_door_with_no_service_templates_refuses_rather_than_tracebacks(service_home,
                                                                         monkeypatch):
    """An installation without ``deployment/systemd`` cannot plan units; say that, in 2.

    A bare wheel is the ordinary case for that: the traceback named a path inside the
    venv and left the operator to guess which door was broken.
    """
    from hermes_memory.install.profiles import InstallationError

    def missing(*args, **kwargs):
        raise InstallationError("no service templates found. Point HERMES_MEMORY_RELEASE "
                                "at an unpacked release, which ships them.")

    monkeypatch.setattr("hermes_memory.install.services.template_root", missing)
    for argv in (["services"], ["start"], ["upgrade"], ["uninstall", "--keep-data"]):
        code, message = errors(*argv)
        assert code == 2 and "refused:" in message and "Traceback" not in message, (
            f"`{' '.join(argv)}` crashed instead of refusing: {message[:400]}")
        assert "HERMES_MEMORY_RELEASE" in message, "the refusal names the door to bring them to"


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


def test_autostart_is_asked_for_by_name_and_a_start_never_decides_it(service_home, runner):
    """Enabling at login and starting now are two verbs, and only one is repeated forever.

    A start that also enabled would turn one operator's one-off reboot into a permanent
    autostart, and §10.4 keeps the two decisions apart for that reason. The receipt names
    every unit the manager was told about, in the order it was told them, because "I
    enabled your services" that turned out to mean one of three is the worst kind of yes.
    """
    (service_home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={service_home.parent / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_HINDSIGHT_URL=http://127.0.0.1:8888\n"
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS=127.0.0.1\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n", encoding="utf-8")
    expected = ["hermes-memory.service", "hermes-memory-hindsight.service",
                "hermes-memory-worker.service"]
    _, enabled = run("services", "--autostart", "enable")
    assert enabled["units"] == expected
    assert [call[2:] for call in runner] == [["enable", name] for name in expected]
    runner.clear()
    run("start")
    assert [call[2] for call in runner] == ["start"] * 3
    runner.clear()
    _, off = run("services", "--autostart", "disable")
    assert off["units"] == list(reversed(expected))
    assert [call[2:] for call in runner] == [["disable", name] for name in off["units"]]
    # Asking about autostart writes no unit file: the plan was never approved here.
    assert not (service_home.parent / "config" / "systemd" / "user"
                / "hermes-memory.service").exists()


def test_an_autostart_the_manager_refuses_says_which_command_failed(service_home,
                                                                    monkeypatch):
    """`enable` for a unit that was never installed is a real and common mistake."""
    monkeypatch.setattr("hermes_memory.cli._HOST_RUNNER",
                        lambda argv: (1, "Unit hermes-memory.service not loaded."))
    code, message = errors("services", "--autostart", "enable")
    assert code == 2
    assert "not loaded" in message and "systemctl --user enable" in message


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


def test_start_under_an_inference_hold_raises_only_the_admission_unit(service_home, runner):
    """§10.5: the hold outlives a restart, and the backend's own startup dispatch is denied.

    This is the live failure the door used to cause: `stop` held inference, `start` asked for
    the backend, its embedding probe got a 503 from the admission gate, the unit restarted
    into its start limit and came up `failed` with the worker taken down beside it. A machine
    that looks broken because its owner asked it to wait is not a machine that is waiting.
    """
    from hermes_memory.config import load_settings
    from hermes_memory.processing.instance_gate import instance_gate

    (service_home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={service_home.parent / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_HINDSIGHT_URL=http://127.0.0.1:8888\n"
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS=127.0.0.1\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n", encoding="utf-8")
    with instance_gate(load_settings()) as gate:
        gate.pause(actor=OWNER, reason="the owner is holding the models")

    code, report = run("start")
    assert code == 0
    assert report["started"] == ["hermes-memory.service"]
    assert report["not_started"] == ["hermes-memory-hindsight.service",
                                     "hermes-memory-worker.service"]
    assert [call[2:] for call in runner] == [["start", "hermes-memory.service"]], \
        "a held backend is not even asked for, let alone restarted into its limit"
    assert report["inference_hold"]["actor"] == OWNER
    assert report["resumes_with"] == "hermes-memory pause --scope inference --resume"
    assert "embedding probe" in report["note"], "the reason has to be in the answer"

    # The round trip: a hold that has been lifted is not a hold, and an installation that
    # only ever saw the paused row must not keep refusing its backend.
    runner.clear()
    with instance_gate(load_settings()) as gate:
        gate.resume(actor=OWNER, reason="the owner is back")
    code, report = run("start")
    assert code == 0 and len(report["started"]) == 3
    assert report["not_started"] == [] and report["inference_hold"] is None
    assert report["resumes_with"] is None
    assert [call[2:] for call in runner] == [["start", name] for name in report["started"]]


def test_a_start_with_no_ledger_at_all_invents_none(service_home, runner):
    """`start` reads the hold, and a reading that writes a ledger is not a reading.

    Without this rule every fresh installation would gain an admission database from the
    first `start`, and the honest answer "nothing has been queued from here yet" would stop
    being available to anything that asked afterwards.
    """
    from hermes_memory.config import load_settings
    from hermes_memory.processing.instance_gate import gate_path

    (service_home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={service_home.parent / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_HINDSIGHT_URL=http://127.0.0.1:8888\n"
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS=127.0.0.1\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n", encoding="utf-8")
    code, report = run("start")
    assert code == 0 and len(report["started"]) == 3
    assert not gate_path(load_settings()).exists()


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
    assert report["held"] == {"inference": False, "inference_hold": None,
                              "delivery": True, "capture": []}
    assert report["survives_a_restart"] is True
    _, lifted = run("pause", "--scope", "delivery", "--resume", "--actor", OWNER)
    assert lifted["held"] == {"inference": False, "inference_hold": None,
                              "delivery": False, "capture": []}

    run("pause", "--scope", "inference", "--actor", OWNER, "--reason", "the fan is loud")
    _, beside = run("pause", "--scope", "delivery", "--actor", OWNER,
                    "--reason", "the owner is away")
    hold = beside["held"]["inference_hold"]
    assert beside["held"]["inference"] is True
    assert (hold["actor"], hold["reason"], hold["state"]) == (OWNER, "the fan is loud",
                                                              "paused"), \
        "a hold that stops every profile names whoever decided it"
    assert hold["changed_at"]

    _, paused = run("pause", "--scope", "delivery", "--actor", OWNER)
    assert (paused["held"]["inference"], paused["held"]["delivery"]) == (True, True), \
        "the report says what is held now, not what this one command just did"
    assert paused["held"]["inference_hold"]["reason"] == "the fan is loud", \
        "the reason on the ledger is the one the standing decision gave"
    _, cleared = run("pause", "--scope", "inference", "--resume", "--actor", OWNER)
    assert cleared["held"]["inference"] is False
    assert cleared["held"]["inference_hold"] is None, "a lifted hold stops being reported"


def test_a_pause_needs_somebody_to_be_answerable_for(service_home, tmp_path, monkeypatch):
    other = tmp_path / "unowned"
    other.mkdir()
    (other / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={other / 'data'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(other))
    code, message = errors("pause", "--scope", "inference")
    assert code == 2 and "owner" in message


# -- the admission ledger, and the request nobody can answer for ---------------

def stranded_at_the_gate(*, resource="remote-9b"):
    """One instance reservation whose upstream answer never came back."""
    with instance_gate(load_settings()) as gate:
        reservation = gate.try_acquire(route="retain", holder="worker-1", resource=resource,
                                       priority=2, ttl=60)
        gate.mark_uncertain(reservation, reason="connection lost mid-request")
        return reservation.id


def test_the_gate_door_says_who_holds_each_device_and_which_answer_is_missing(home):
    """`doctor` says a device is blocked; this is the reading that names the row."""
    reservation = stranded_at_the_gate()
    code, reading = run("gate")
    assert code == 0
    assert [row["id"] for row in reading["unresolved"]] == [reservation]
    assert reading["held"] == [], "an uncertain reservation is not a live holder"
    assert reading["blocked"] == ["remote-9b"]
    assert reading["paused"] is False and reading["hold"] is None


def test_settling_a_lost_request_through_the_door_gives_the_device_back(home):
    """The reconciler that found the record projected is an answer about the slot too.

    Without this the honest report and the working installation were incompatible: the
    gate kept the device blocked exactly as designed, and nothing in the framework could
    say what had become of the request, so one dropped connection retired a GPU.
    """
    reservation = stranded_at_the_gate()
    code, report = run("gate", "--resolve", reservation, "--outcome", "cancelled",
                       "--reason", "the backend acknowledged the cancellation")
    assert code == 0
    assert report["settled"]["settled_by"] == OWNER
    assert report["unresolved"] == [] and report["blocked"] == []
    assert report["usage"] == {}, "nobody saw the token count, so nothing is charged"


def test_the_gate_door_refuses_to_free_a_device_on_an_unstated_outcome(home):
    reservation = stranded_at_the_gate()
    code, message = errors("gate", "--resolve", reservation, "--reason", "looked at it")
    assert code == 2 and "freed by an answer" in message
    with instance_gate(load_settings()) as gate:
        assert [row["id"] for row in gate.unresolved()] == [reservation], \
            "a refusal must not free the device it refuses to speak for"


def test_the_gate_door_refuses_an_unattributed_settlement(home, tmp_path, monkeypatch):
    unowned = tmp_path / "no-owner"
    unowned.mkdir()
    (unowned / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={unowned / 'data'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(unowned))
    reservation = stranded_at_the_gate()
    code, message = errors("gate", "--resolve", reservation, "--outcome", "failed")
    assert code == 2 and "reason" in message
    code, message = errors("gate", "--resolve", reservation, "--outcome", "failed",
                           "--reason", "checked the model server by hand")
    assert code == 2 and "who settled" in message


def test_the_gate_door_refuses_flags_that_belong_to_a_settlement(home):
    code, message = errors("gate", "--outcome", "cancelled")
    assert code == 2 and "--resolve" in message


def register_source(name=SOURCE, *, policy="local-only", pages=0):
    """A connector this installation actually reads, optionally with a cursor."""
    from hermes_memory.sources.runtime import ConnectorRuntime

    from connector_script import Scripted, notes

    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        sync = SyncController(store)
        sync.register(name, policy_version=policy)
        if pages:
            ConnectorRuntime(store, sync, holder="test worker").run(
                Scripted([notes(2, next_cursor=f"page-{index + 1}")
                          for index in range(pages)]))
    return settings


def held_capture():
    from hermes_memory.cli import _holds

    return _holds(load_settings())["capture"]


def read_adapter(count=1, *, next_cursor=None):
    from connector_script import Scripted, notes

    return Scripted([notes(count, next_cursor=next_cursor)])


def connector_for(settings):
    from hermes_memory.sources.runtime import ConnectorRuntime

    store = EvidenceStore(settings.db_path)
    return ConnectorRuntime(store, SyncController(store), holder="cli test"), store


def test_a_pause_that_names_no_source_names_nothing_to_hold(service_home):
    """--scope capture without a source is not a hold anybody could check afterwards."""
    run("init")
    register_source()
    code, message = errors("pause", "--scope", "capture", "--actor", OWNER)
    assert code == 2 and "which connector" in message
    assert held_capture() == [], "a refusal writes nothing"


def test_a_source_flag_on_a_hold_that_has_no_sources_is_refused(service_home):
    """Ignoring --source quietly would report a per-connector hold that does not exist."""
    run("init")
    settings = register_source()
    code, message = errors("pause", "--scope", "delivery", "--source", SOURCE,
                           "--actor", OWNER)
    assert code == 2 and "--scope delivery" in message
    with EvidenceStore(settings.db_path) as store:
        assert store.stage_is_paused("global", "delivery") is False


def test_a_capture_hold_stops_the_reading_and_nothing_else(service_home):
    from hermes_memory.sources.runtime import PAUSED

    run("init")
    settings = register_source(pages=1)
    code, report = run("pause", "--scope", "capture", "--source", SOURCE, "--actor", OWNER,
                       "--reason", "the mailbox is under review")
    assert code == 0
    assert [item["source"] for item in report["held"]["capture"]] == [SOURCE]
    assert report["written_to"] == [settings.profile]

    runtime, store = connector_for(settings)
    try:
        result = runtime.run(read_adapter(3))
    finally:
        store.close()
    assert result.stopped == PAUSED and result.records == 0, \
        "the hold the CLI wrote is the hold the runtime honours"
    with EvidenceStore(settings.db_path) as store:
        assert SyncController(store).paused_stages(SOURCE) == ["capture"], \
            "formation and proactivity were never asked to stop"
        assert store.db.execute("SELECT cursor FROM connectors").fetchone()[0] == "page-1", \
            "a hold is not a reset; the connector still knows where it stopped"

    _, lifted = run("pause", "--scope", "capture", "--source", SOURCE, "--resume",
                    "--actor", OWNER)
    assert lifted["held"]["capture"] == []
    runtime, store = connector_for(settings)
    try:
        assert runtime.run(read_adapter(1)).stopped != PAUSED
    finally:
        store.close()


def test_lifting_a_capture_hold_leaves_the_stages_nobody_asked_about_held(service_home):
    """``--scope capture`` is a narrow instruction, in both directions."""
    from hermes_memory.sources.sync import DOWNSTREAM_STAGES

    run("init")
    settings = register_source()
    with EvidenceStore(settings.db_path) as store:
        SyncController(store).pause(SOURCE, actor=OWNER, reason="no compute today",
                                    policy_version="local-only", stages=DOWNSTREAM_STAGES)
    run("pause", "--scope", "capture", "--source", SOURCE, "--actor", OWNER,
        "--reason", "and stop reading it too")
    code, lifted = run("pause", "--scope", "capture", "--source", SOURCE, "--resume",
                       "--actor", OWNER)
    assert code == 0
    with EvidenceStore(settings.db_path) as store:
        assert sorted(SyncController(store).paused_stages(SOURCE)) == \
            ["formation", "proactivity"], \
            "resuming the reading is not a decision to start speaking about it"


def test_a_capture_hold_is_written_on_every_memory_that_reads_the_source(service_home,
                                                                        tmp_path):
    """Two profiles reading one mailbox stop together, or the hold is a lie."""
    run("init")
    settings = load_settings()
    registry = ProfileRegistry.open(settings)
    paths = {}
    for name in ("work", "personal"):
        house = tmp_path / "profile-homes" / name
        house.mkdir(parents=True)
        proposal = registry.plan(name, house)
        registry.enroll(name, house, actor=OWNER, review_digest=proposal["review_digest"])
        path = Path(proposal["data_dir"]) / "canonical.db"
        with EvidenceStore(path) as store:
            SyncController(store).register(SOURCE, policy_version="local-only")
        paths[name] = path
    registry.db.close()

    code, report = run("pause", "--scope", "capture", "--source", SOURCE, "--actor", OWNER,
                       "--reason", "the owner revoked the app password")
    assert code == 0
    assert sorted(report["written_to"]) == ["personal", "work"]
    for name, path in paths.items():
        with EvidenceStore(path) as store:
            assert store.stage_is_paused(SOURCE, "capture") is True, name
    assert sorted(item["profile"] for item in report["held"]["capture"]) == \
        ["personal", "work"], "the report is read back from both stores"


def test_a_capture_hold_on_an_unregistered_source_refuses(service_home):
    run("init")
    register_source("imail")
    code, message = errors("pause", "--scope", "capture", "--source", SOURCE,
                           "--actor", OWNER)
    assert code == 2 and "no memory" in message, "a typo must not look like a hold"


# -- reconfiguring a connector ---------------------------------------------------

def preview(source=SOURCE, policy="private-api", reason="the account moved", **flags):
    argv = ["sources", "reconfigure", "--source", source, "--policy", policy,
            "--actor", OWNER, "--reason", reason]
    for name, value in flags.items():
        argv += [f"--{name}", value]
    return run(*argv)


def test_a_reconfigure_shows_what_it_will_strand_before_it_strands_it(service_home):
    run("init")
    settings = register_source(pages=1)
    code, report = preview()
    assert code == 0
    entry = report["connectors"][0]
    assert entry["generation_now"] == 1 and entry["cursor_held"] is True
    assert entry["policy_now"] == "local-only" and entry["policy_next"] == "private-api"
    assert report["review_digest"], "the approval is of this exact reading"
    with EvidenceStore(settings.db_path) as store:
        state = SyncController(store).state(SOURCE)
    assert state["generation"] == 1 and state["policy_version"] == "local-only", \
        "a preview performs nothing"


def test_approving_a_reconfigure_starts_a_new_generation_and_drops_the_cursor(
        service_home):
    run("init")
    settings = register_source(pages=1)
    _, report = preview()
    code, done = preview(review=report["review_digest"])
    assert code == 0 and done["reconfigured"][0]["generation"] == 2
    with EvidenceStore(settings.db_path) as store:
        state = SyncController(store).state(SOURCE)
        audit = store.db.execute("SELECT metadata FROM audit WHERE "
                                 "action='connector_reconfigure'").fetchall()
    assert state["generation"] == 2 and state["cursor"] is None
    assert state["coverage_state"] == "unknown"
    assert state["policy_version"] == "private-api"
    assert len(audit) == 1 and "the account moved" in audit[0]["metadata"], \
        "the reason outlives the shell that typed it"


def test_a_reconfigure_strands_an_in_flight_writer_and_says_so(service_home):
    """The generation change is visible before it happens, and the old fence dies by it."""
    from hermes_memory.sources.sync import StaleFence

    run("init")
    settings = register_source(pages=1)
    with EvidenceStore(settings.db_path) as store:
        stranded = SyncController(store).acquire(SOURCE, holder="worker-1", ttl=3600)
    _, report = preview()
    assert report["connectors"][0]["lease"] == "held", \
        "a writer is mid-page, and the operator approving this can see it"
    code, done = preview(review=report["review_digest"])
    assert code == 0 and done["reconfigured"][0]["generation"] == 2
    with pytest.raises(StaleFence, match="reconfigured"):
        with ReadOnlyStore(settings.db_path) as store:
            stranded.validate(store.db)
    with EvidenceStore(settings.db_path) as store:
        assert SyncController(store).state(SOURCE)["lease"] is None, \
            "the connector does not sit waiting on a lease that stopped meaning anything"


def test_a_reconfigure_approved_against_another_plan_is_refused(service_home):
    """The digest is of one command's worth of decision, and only of that one.

    A different scope is a different thing being approved; a different attribution is a
    different person approving it, and the audit line records the reason, so an operator
    who wants either changed previews again rather than pastes the old digest.
    """
    run("init")
    settings = register_source()
    _, report = preview()
    for name, value in (("policy", "disabled"), ("reason", "something else entirely"),
                        ("actor", "somebody-else")):
        argv = ["sources", "reconfigure", "--source", SOURCE, "--policy", "private-api",
                "--actor", OWNER, "--reason", "the account moved",
                "--review", report["review_digest"]]
        argv[argv.index(f"--{name}") + 1] = value
        code, message = errors(*argv)
        assert code == 2 and "not the one" in message, name
    with EvidenceStore(settings.db_path) as store:
        assert store.db.execute("SELECT policy_version FROM connectors").fetchone()[0] \
            == "local-only"


def test_a_reconfigure_needs_a_source_a_reason_and_a_known_connector(service_home):
    run("init")
    register_source()
    code, message = errors("sources", "reconfigure", "--source", SOURCE, "--actor", OWNER)
    assert code == 2 and "why" in message
    _, message = errors("sources", "reconfigure", "--source", "jira", "--policy",
                        "local-only", "--actor", OWNER, "--reason", "a new tool")
    assert code == 2 and "no memory" in message
    _, message = errors("sources", "reconfigure", "--actor", OWNER, "--reason", "whatever")
    assert code == 2 and "which connector" in message
    _, message = errors("sources", "list", "--source", SOURCE)
    assert code == 2 and "reconfigure" in message, \
        "a listing that names one source is not the reading it claims to be"
    _, message = errors("sources", "reconfigure", "--source", SOURCE, "--actor", OWNER,
                        "--reason", "remapped", "--gaps")
    assert code == 2 and "reads" in message, "--gaps asks for evidence a write never reads"
    assert run("sources", "list")[1]["registered"] == 1


def test_a_reconfigure_writes_nothing_without_somebody_to_answer_for_it(service_home,
                                                                       tmp_path,
                                                                       monkeypatch):
    other = tmp_path / "unowned"
    other.mkdir()
    (other / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={other / 'data'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(other))
    run("init")
    code, message = errors("sources", "reconfigure", "--source", SOURCE, "--policy",
                           "private-api", "--reason", "remapped")
    assert code == 2 and "owner principal" in message


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
    carry(source=Path(__file__).resolve().parents[2], into=release,
          names=("integrations",))
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


# -- the owner's own decisions -------------------------------------------------

def _preview_an_erasure():
    """An agent opens a forgetting request, which is as far as an agent gets."""
    run("init")
    with EvidenceStore(load_settings().db_path) as store:
        record = _a_message(store, "the invoice I am withdrawing")
        preview = ErasureManager(store, owner_principal=OWNER).preview(
            record_ids=[record], actor="agent:session-1", actor_kind="agent",
            reason="asked for in chat")
    return preview, record


def test_the_owner_list_shows_the_digest_an_approval_has_to_carry(home):
    """A decision the owner cannot see the shape of is a decision they cannot make.

    The listing names who asked, what it would cost and which digest signs it — and not
    the record itself, because a listing lands in a terminal's scrollback.
    """
    preview, record = _preview_an_erasure()
    code, report = run("owner", "--list")
    assert code == 0
    awaiting = report["awaiting"][0]["awaiting_forgetting"]
    assert [item["intent_id"] for item in awaiting] == [preview["intent_id"]]
    assert awaiting[0]["preview_digest"] == preview["preview_digest"]
    assert awaiting[0]["requested_by"] == "agent:session-1"
    assert awaiting[0]["requester_kind"] == "agent"
    assert awaiting[0]["confirmable_by"] == OWNER
    assert awaiting[0]["records"] == 1
    assert record not in json.dumps(report)


def test_an_agent_s_opened_forgetting_is_closed_by_the_owner_alone(home):
    preview, record = _preview_an_erasure()
    intent, digest_value = preview["intent_id"], preview["preview_digest"]
    code, message = errors("owner", "--confirm-forgetting", intent)
    assert code == 2 and "digest of the preview" in message
    code, message = errors("owner", "--confirm-forgetting", intent, "--digest", "0" * 64,
                           "--actor", OWNER)
    assert code == 2 and "does not match" in message
    code, message = errors("owner", "--confirm-forgetting", intent, "--digest", digest_value,
                           "--actor", "agent:session-1")
    assert code == 2 and "owner principal" in message
    with ReadOnlyStore(load_settings().db_path) as store:
        assert store.live_and_visible(record), "a refused confirmation changed nothing"
    code, outcome = run("owner", "--confirm-forgetting", intent,
                        "--digest", digest_value, "--actor", OWNER)
    assert code == 0
    assert outcome["state"] in ("erasure_pending", "complete")
    with ReadOnlyStore(load_settings().db_path) as store:
        assert not store.live_and_visible(record)
    _, after = run("owner", "--list")
    assert after["awaiting"][0]["awaiting_forgetting"] == [], \
        "a forgetting that has already happened is still asking for a decision"


def test_a_candidate_claim_is_confirmed_and_retracted_by_the_owner_only(home):
    """A pattern somebody noticed is not something the archive asserts on its own.

    `status` counts these as awaiting the owner, so the door has to reach them too —
    otherwise the report says somebody should decide and nobody can.
    """
    run("init")
    with EvidenceStore(load_settings().db_path) as store:
        record = _a_message(store, "Sam always pays the invoice in full")
        proposed = AssertionStore(store, owner_principal=OWNER).propose(
            subject="person:sam", predicate="reliability", value="pays in full",
            kind="belief", evidence_kind="observed_pattern", record_id=record,
            quote="pays the invoice in full", proposed_by="agent:session-1")
    assert proposed["status"] == "candidate"
    code, report = run("owner", "--list")
    listed = report["awaiting"][0]["candidate_assertions"]
    assert [item["assertion_id"] for item in listed] == [proposed["id"]]
    assert listed[0]["value"] == "pays in full"
    code, message = errors("owner", "--confirm-assertion", proposed["id"], "--actor", OWNER)
    assert code == 2 and "say why" in message
    code, message = errors("owner", "--confirm-assertion", proposed["id"], "--actor",
                           "agent:session-1", "--reason", "my own pattern")
    assert code == 2 and "belongs to the owner" in message
    code, decision = run("owner", "--confirm-assertion", proposed["id"], "--actor", OWNER,
                         "--reason", "the owner recognises the pattern")
    assert code == 0 and decision["status"] == "confirmed" and decision["changed"] is True
    _, asked = run("owner", "--list")
    assert asked["awaiting"][0]["candidate_assertions"] == [], \
        "a claim the archive already stands behind is not still asking for a decision"
    code, retracted = run("owner", "--retract-assertion", proposed["id"], "--actor", OWNER,
                          "--reason", "one late invoice is enough to stop asserting it")
    assert code == 0 and retracted["status"] == "retracted"
    _, after = run("owner", "--list")
    assert after["awaiting"][0]["candidate_assertions"] == []


def test_an_identity_is_confirmed_rejected_and_revoked_by_the_owner_only(home):
    run("init")
    with EvidenceStore(load_settings().db_path) as store:
        record = _a_message(store, "written by Sam")
        identities = IdentityStore(store, owner_principal=OWNER)
        first = identities.account("email", "sam@example.com")
        second = identities.account("email", "samuel@example.com")
        candidate = identities.propose(
            account_a=first, account_b=second, rule="email-normalized-equal",
            basis="one mailbox, two spellings", evidence=[record],
            proposed_by="agent:session-1")["candidate_id"]
    code, message = errors("owner", "--confirm-identity", candidate, "--actor", OWNER)
    assert code == 2 and "say why" in message
    code, message = errors("owner", "--confirm-identity", candidate, "--actor", "agent:one",
                           "--reason", "not my call to make")
    assert code == 2 and "owner principal" in message
    code, decision = run("owner", "--confirm-identity", candidate, "--actor", OWNER,
                         "--reason", "the owner recognised both addresses")
    assert code == 0 and decision["state"] == "confirmed"
    code, revoked = run("owner", "--revoke-edge", decision["edge_id"], "--actor", OWNER,
                        "--reason", "the second address belongs to a colleague")
    assert code == 0 and revoked["state"] == "revoked"


def test_a_rejection_is_durable_and_says_so(home):
    run("init")
    with EvidenceStore(load_settings().db_path) as store:
        record = _a_message(store, "written by Sam")
        identities = IdentityStore(store, owner_principal=OWNER)
        first = identities.account("email", "sam@example.com")
        second = identities.account("email", "samuel@example.com")
        candidate = identities.propose(
            account_a=first, account_b=second, rule="email-normalized-equal",
            basis="one mailbox, two spellings", evidence=[record],
            proposed_by="agent:session-1")["candidate_id"]
    code, rejected = run("owner", "--reject-identity", candidate, "--actor", OWNER,
                         "--reason", "they are two people")
    assert code == 0 and rejected["state"] == "rejected"
    code, message = errors("owner", "--confirm-identity", candidate, "--actor", OWNER,
                           "--reason", "changed my mind")
    assert code == 2 and "re-proposing" in message


def test_a_decision_belongs_to_one_memory(home):
    run("init")
    registry = ProfileRegistry.open(load_settings())
    try:
        for name in ("work", "personal"):
            profile_home = home / "profiles" / name
            profile_home.mkdir(parents=True)
            proposal = registry.plan(name, profile_home)
            registry.enroll(name, profile_home, actor=OWNER,
                            review_digest=proposal["review_digest"])
    finally:
        registry.db.close()
    code, message = errors("owner", "--confirm-identity", "idc_1", "--actor", OWNER,
                           "--reason", "which memory?")
    assert code == 2 and "name one with --profile" in message
    code, report = run("owner", "--list")
    assert code == 0 and len(report["awaiting"]) == 2


def test_an_unowned_installation_can_list_but_not_decide(tmp_path, monkeypatch):
    root = tmp_path / "unowned"
    (root / "data").mkdir(parents=True)
    (root / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={root / 'data'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(root))
    monkeypatch.delenv("HERMES_MEMORY_DATA_DIR", raising=False)
    run("init")
    code, report = run("owner", "--list")
    assert code == 0 and report["awaiting"][0]["awaiting_forgetting"] == []
    code, message = errors("owner", "--confirm-forgetting", "erase_1", "--digest", "0" * 64)
    assert code == 2 and "HERMES_MEMORY_OWNER_PRINCIPAL" in message


def test_a_listing_never_opens_a_store_it_could_change(home, monkeypatch):
    """`--list` makes no decision, so it must not hold the handle that could make one.

    The writable store applies migrations on open; a reading that quietly upgraded a
    database would be a write the owner never approved, in a command they run to look.
    """
    run("init")

    def refusing(*args, **kwargs):
        raise AssertionError("a listing opened the store for writing")

    monkeypatch.setattr("hermes_memory.cli.EvidenceStore", refusing)
    code, report = run("owner", "--list")
    assert code == 0 and report["awaiting"]


def test_a_listing_and_a_decision_are_not_asked_for_in_one_breath(home):
    run("init")
    code, message = errors("owner", "--list", "--confirm-identity", "idc_1",
                           "--actor", OWNER, "--reason", "both at once")
    assert code == 2 and "one or the other" in message
    code, message = errors("owner")
    assert code == 2 and "nothing was asked" in message


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


def _snapshot_of(home, text):
    """One record, one snapshot, then whatever the caller does next."""
    run("init")
    with EvidenceStore(load_settings().db_path) as store:
        _a_message(store, text, source_id="snap-1")
        store.bump_epoch(reason="test", actor=OWNER)
    _, made = run("backup", "--reason", "before the mistake")
    return made["backups"][0]["snapshot"]["id"]


def test_a_restore_shows_what_it_would_destroy_and_destroys_nothing(home, tmp_path):
    """The first invocation is a reading, and the reading names the live epoch.

    A rollback's whole cost is the writes after the snapshot, so those have to be visible
    before anybody approves — and the store must still be exactly as it was afterwards.
    """
    snapshot_id = _snapshot_of(home, "the lease ends in March")
    with EvidenceStore(load_settings().db_path) as store:
        later = _a_message(store, "written after the snapshot", source_id="later-1")
        store.bump_epoch(reason="later write", actor=OWNER)
    code, proposal = run("restore", "--snapshot", snapshot_id)
    assert code == 0
    assert proposal["snapshot"]["id"] == snapshot_id
    assert proposal["live_epoch"] > proposal["snapshot"]["epoch"]
    assert proposal["review_digest"] in proposal["next"]
    assert "--review" in proposal["next"]
    assert proposal["decisions_kept"], "the ledger the restore re-applies is not counted"
    # A plan goes to a terminal, and often into a log. It counts the rows it will
    # re-apply; it does not print the record IDs, source IDs or texts inside them.
    assert all(isinstance(kept, int) for kept in proposal["decisions_kept"].values())
    printed = json.dumps(proposal)
    assert "snap-1" not in printed and "lease" not in printed
    # The test's own promise: the writes after the snapshot are the whole price, and a
    # reading that does not say them approves a destruction nobody was shown.
    cost = proposal["cost"]
    assert cost["readable_now"] == 2 and cost["readable_in_the_snapshot"] == 1
    assert cost["destroyed"] == 1, "the plan does not say what the rollback destroys"
    assert cost["brought_back"] == 0 and \
        cost["brought_back_and_already_forgotten"] == 0
    assert all(isinstance(price, int) for price in cost.values())
    with EvidenceStore(load_settings().db_path) as store:
        assert [item.id for item in store.search("written after the snapshot")] == [later]


def test_a_restore_names_the_evidence_it_will_erase_again_before_it_is_approved(home):
    """Evidence forgotten *after* the snapshot comes back inside the copy, and the plan says so.

    §C4's rule is that a restore cannot resurrect the forgotten. Said afterwards, that is a
    receipt; said beforehand, it is something the owner approves. The two numbers have to
    agree, or the plan quoted a price the act did not pay.
    """
    snapshot_id = _snapshot_of(home, "the figure I withdrew")
    with EvidenceStore(load_settings().db_path) as store:
        record = str(store.db.execute("SELECT id FROM records").fetchone()[0])
        preview = ErasureManager(store, owner_principal=OWNER).preview(
            record_ids=[record], actor="agent:session-1", actor_kind="agent",
            reason="withdrawn after the snapshot was taken")
    code, outcome = run("owner", "--confirm-forgetting", preview["intent_id"],
                        "--digest", preview["preview_digest"], "--actor", OWNER)
    assert code == 0
    _, proposal = run("restore", "--snapshot", snapshot_id)
    cost = proposal["cost"]
    assert cost["destroyed"] == 0, "nothing was written after this snapshot"
    assert cost["brought_back"] == 1 and cost["brought_back_and_already_forgotten"] == 1, \
        "the forgotten record comes back into the copy unannounced"
    _, report = run("restore", "--snapshot", snapshot_id,
                    "--review", proposal["review_digest"], "--actor", OWNER)
    assert report["restored"] == snapshot_id
    assert report["reapplied"] == cost["brought_back_and_already_forgotten"], \
        "the restore did not keep the price the plan quoted"
    with ReadOnlyStore(load_settings().db_path) as store:
        assert not store.live_and_visible(record), "a restore resurrected the forgotten"


def test_a_record_buried_before_the_snapshot_was_taken_is_not_counted_as_coming_back(home):
    """`readable` has to mean readable on both sides of the comparison.

    A snapshot taken after a forgetting still holds the row with the tombstone on it. Counting
    that as evidence the rollback would lose talks the owner out of a restore that costs
    nothing; counting it as evidence the rollback would bring back is the same mistake turned
    the other way.
    """
    snapshot_id = _snapshot_of(home, "the figure I withdrew")
    with EvidenceStore(load_settings().db_path) as store:
        buried = str(store.db.execute("SELECT id FROM records").fetchone()[0])
        preview = ErasureManager(store, owner_principal=OWNER).preview(
            record_ids=[buried], actor="agent:session-1", actor_kind="agent",
            reason="withdrawn before the second snapshot")
    code, _ = run("owner", "--confirm-forgetting", preview["intent_id"],
                  "--digest", preview["preview_digest"], "--actor", OWNER)
    assert code == 0
    _, second = run("backup", "--reason", "after the forgetting")

    code, proposal = run("restore", "--snapshot", snapshot_id)
    assert code == 0
    cost = proposal["cost"]
    assert cost["readable_now"] == 0 and cost["readable_in_the_snapshot"] == 1
    assert cost["destroyed"] == 0, "a record that was already buried is not a loss"
    assert cost["brought_back"] == 1 and cost["brought_back_and_already_forgotten"] == 1, \
        "the copy that predates the forgetting is not said to resurrect it"
    _, later = run("restore", "--snapshot", second["backups"][0]["snapshot"]["id"])
    assert later["cost"]["destroyed"] == 0 and later["cost"]["brought_back"] == 0, \
        "a tombstone the snapshot already holds is counted as evidence returning"


def test_the_digest_approves_everything_the_restore_plan_showed(home):
    """An approval covers the whole reading, or the plan can print a price it never meant.

    The cost is computed rather than typed, so nothing else in this file would notice it being
    left out of what the digest seals. This notices.
    """
    snapshot_id = _snapshot_of(home, "the lease ends in March")
    with EvidenceStore(load_settings().db_path) as store:
        _a_message(store, "written after the snapshot", source_id="later-1")
        store.bump_epoch(reason="later write", actor=OWNER)
    code, proposal = run("restore", "--snapshot", snapshot_id)
    assert code == 0
    shown = {key: value for key, value in proposal.items()
             if key not in ("review_digest", "next")}
    assert proposal["cost"] in shown.values()
    assert proposal["review_digest"] == digest([RESTORE_PLAN_VERSION, shown]), \
        "the plan showed something its own approval does not cover"


def test_an_approved_restore_takes_the_store_back(home):
    snapshot_id = _snapshot_of(home, "the lease ends in March")
    with EvidenceStore(load_settings().db_path) as store:
        _a_message(store, "written after the snapshot", source_id="later-1")
        store.bump_epoch(reason="later write", actor=OWNER)
    _, proposal = run("restore", "--snapshot", snapshot_id)
    code, report = run("restore", "--snapshot", snapshot_id,
                       "--review", proposal["review_digest"], "--actor", OWNER)
    assert code == 0
    assert report["restored"] == snapshot_id
    assert report["pre_restore_backup"].endswith(".db")
    with ReadOnlyStore(load_settings().db_path) as store:
        assert [item.text for item in store.search("lease")]
        assert store.search("written after the snapshot") == []


def test_an_approval_of_a_store_that_has_moved_since_is_refused(home):
    """The digest covers the live epoch, so a concurrent capture invalidates it.

    Approving a plan and then having the ground move under it is how a rollback destroys
    records nobody ever looked at; the answer here is a refusal and a new reading.
    """
    snapshot_id = _snapshot_of(home, "the lease ends in March")
    _, proposal = run("restore", "--snapshot", snapshot_id)
    with EvidenceStore(load_settings().db_path) as store:
        _a_message(store, "a write that arrived after the plan was shown",
                   source_id="moved-1")
        store.bump_epoch(reason="the ground moved", actor=OWNER)
    code, message = errors("restore", "--snapshot", snapshot_id,
                           "--review", proposal["review_digest"], "--actor", OWNER)
    assert code == 2 and "has moved" in message
    with ReadOnlyStore(load_settings().db_path) as store:
        assert store.search("arrived after the plan")


def test_a_restore_names_the_profile_when_more_than_one_store_is_enrolled(home):
    run("init")
    registry = ProfileRegistry.open(load_settings())
    try:
        for name in ("work", "personal"):
            profile_home = home / "profiles" / name
            profile_home.mkdir(parents=True)
            proposal = registry.plan(name, profile_home)
            registry.enroll(name, profile_home, actor=OWNER,
                            review_digest=proposal["review_digest"])
    finally:
        registry.db.close()
    code, message = errors("restore", "--snapshot", "anything")
    assert code == 2 and "name one with --profile" in message


def test_a_snapshot_that_does_not_verify_is_not_a_thing_to_restore(home, monkeypatch):
    from hermes_memory.lifecycle.snapshots import Snapshots

    snapshot_id = _snapshot_of(home, "the lease ends in March")
    monkeypatch.setattr(Snapshots, "verify", lambda self, sid: {
        "id": sid, "ok": False, "problems": ["the file no longer matches the digest"],
        "notes": []})
    code, message = errors("restore", "--snapshot", snapshot_id)
    assert code == 2 and "no longer matches the digest" in message


def test_an_unowned_installation_cannot_confirm_its_own_restore(tmp_path, monkeypatch):
    root = tmp_path / "unowned"
    (root / "data").mkdir(parents=True)
    (root / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={root / 'data'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(root))
    monkeypatch.delenv("HERMES_MEMORY_DATA_DIR", raising=False)
    run("init")
    code, message = errors("restore", "--snapshot", "anything")
    assert code == 2 and "owner principal" in message


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
    # The fixture stages a whole release, so take the half away to state the case: a tree
    # whose runtime is there and whose plugin is not must not be offered as an upgrade.
    shutil.rmtree(home / "runtime" / "current" / "integrations")
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


# -- typed measurements ------------------------------------------------------

@pytest.fixture()
def samples(tmp_path):
    """A weight export and a heart-rate export, in the shapes devices actually write."""
    root = tmp_path / "health"
    root.mkdir()
    (root / "weight.csv").write_text(
        "date,kg\n"
        "2026-03-01T08:00:00+00:00,70.0\n"
        "2026-03-02T08:00:00+00:00,71.0\n"
        "2026-03-03T08:00:00+00:00,72.0\n"
        "2026-04-01T08:00:00+00:00,99.0\n", encoding="utf-8")
    (root / "heart_rate.jsonl").write_text("\n".join(json.dumps(row) for row in (
        {"time": "2026-03-01T08:00:00+00:00", "value": 60, "unit": "bpm",
         "device": "watch-a"},
        {"time": "2026-03-01T08:15:00+00:00", "value": 150, "unit": "bpm",
         "device": "watch-a"},
        {"time": "2026-03-01T08:30:00+00:00", "value": 70, "unit": "bpm",
         "device": "chest strap"})) + "\n", encoding="utf-8")
    return root


def test_measure_lists_the_series_the_import_created(home, samples):
    run("init")
    run("import", "--source", "structured", "--granularity", "sample", "--path", str(samples))
    code, report = run("measure", "--list")
    assert code == 0
    assert [(item["measure"], item["device"], item["unit"], item["samples"])
            for item in report["series"]] == [
                ("heart_rate", "chest strap", "bpm", 1),
                ("heart_rate", "watch-a", "bpm", 2),
                ("weight", "unattributed", "kg", 4)]
    assert "--what" in report["ask"]


def test_measure_answers_a_window_the_operator_chose(home, samples):
    run("init")
    run("import", "--source", "structured", "--granularity", "sample", "--path", str(samples))
    code, report = run("measure", "--what", "weight",
                       "--since", "2026-03-01T00:00:00+00:00",
                       "--until", "2026-03-31T23:59:59+00:00")
    assert code == 0
    assert report["samples"] == 3 and report["statistics"]["mean"] == 71.0
    assert report["units_seen"] == {"kg": 3}
    assert report["first_at"] == "2026-03-01T08:00:00+00:00"
    assert report["last_at"] == "2026-03-03T08:00:00+00:00"
    assert len(report["records"]) == 3, "the reading cites the samples it averaged"


def test_measure_narrows_to_one_device_rather_than_mixing_two(home, samples):
    run("init")
    run("import", "--source", "structured", "--granularity", "sample", "--path", str(samples))
    _code, wrist = run("measure", "--what", "heart_rate", "--device", "watch-a")
    assert wrist["samples"] == 2
    _code, both = run("measure", "--what", "heart_rate")
    assert both["samples"] == 3, "no --device means every device that reported"
    assert both["statistics"]["mean"] == pytest.approx(93.333333, abs=1e-4)


def test_an_import_of_rows_is_not_an_import_of_a_summary(home, samples):
    """The same directory kept two ways answers two different questions.

    A series record describes a group once, at the moment it was read; the rows keep
    every sample, which is what makes a later window askable at all.
    """
    run("init")
    code, report = run("import", "--source", "structured", "--granularity", "series",
                       "--path", str(samples))
    assert code == 0 and report["records"] == 3, "one per device+measure+unit group"
    assert run("measure", "--what", "weight")[1]["samples"] == 0, \
        "a description of samples is not a sample"

    code, report = run("import", "--source", "structured", "--granularity", "sample",
                       "--path", str(samples))
    assert code == 0 and report["records"] == 7, "one record per row the source wrote"
    assert run("measure", "--what", "weight")[1]["samples"] == 4


def test_keeping_a_sample_export_as_rows_demands_the_choice(home, samples):
    run("init")
    code, message = errors("import", "--source", "structured", "--path", str(samples))
    assert code == 2 and "--granularity sample" in message
    assert "measure" in message, "the refusal says what the choice decides"


def test_a_grouping_choice_a_reader_does_not_have_is_refused(home, exports):
    run("init")
    code, message = errors("import", "--source", "files", "--granularity", "sample",
                           "--path", str(exports))
    assert code == 2 and "one record per item" in message
    assert run("import", "--source", "files", "--path", str(exports))[1]["records"] == 2


def test_one_enrolled_memory_needs_no_naming_and_two_do(home, samples, tmp_path):
    """A reading is about a person, so ambiguity is refused rather than resolved by luck.

    One profile is unambiguous — it is the store the imports have been writing into — and
    the doors answer from it. Add a second and every one of them has to be told whose.
    """
    run("init")
    plan = run("enroll", "--hermes-home", str(home / "profiles" / "work"))[1]
    run("enroll", "--hermes-home", str(home / "profiles" / "work"),
        "--review", plan["review_digest"])
    run("init", "--hermes-home", str(home / "profiles" / "work"))
    run("import", "--source", "structured", "--granularity", "sample", "--path", str(samples),
        "--profile", "work")

    assert run("measure", "--what", "weight")[1]["samples"] == 4, "one memory, no question"
    assert run("sources", "list")[1]["registered"] == 1
    assert run("audit", "--actions")[1]["actions"]

    second = run("enroll", "--hermes-home", str(tmp_path / "other-home"))[1]
    run("enroll", "--hermes-home", str(tmp_path / "other-home"), "--review",
        second["review_digest"])
    for argv in (["measure", "--what", "weight"], ["sources", "list"], ["audit", "--actions"]):
        code, message = errors(*argv)
        assert code == 2 and "2 memories are enrolled" in message, (
            f"`{' '.join(argv)}` answered for somebody: {message[:200]}")
        assert "--profile or --hermes-home" in message

    named = run("measure", "--what", "weight", "--hermes-home",
                str(home / "profiles" / "work"))[1]
    assert named["samples"] == 4, "naming the memory is all the door was missing"


def test_measure_reads_the_memory_named_rather_than_a_fallback(home, samples):
    """A number is attributed to a person, so which memory was read is part of the answer.

    An unenrolled home is refused rather than answered out of the default profile: that
    substitution is the exact failure the profile map exists to prevent.
    """
    run("init")
    run("import", "--source", "structured", "--granularity", "sample", "--path", str(samples))
    assert run("measure", "--what", "weight")[1]["samples"] == 4
    code, message = errors("measure", "--what", "weight", "--hermes-home", str(home / "other"))
    assert code == 2 and "no profile is enrolled" in message


def test_measure_refuses_to_be_a_listing_and_an_answer_at_once(home, samples):
    run("init")
    code, message = errors("measure", "--list", "--what", "weight")
    assert code == 2 and "name the measure alone" in message
    code, message = errors("measure", "--list", "--device", "scale-a")
    assert code == 2, "a narrowed listing is still an answer being asked for"


def test_measure_without_a_question_is_refused_rather_than_guessing(home):
    run("init")
    code, message = errors("measure")
    assert code == 2 and "has to say what to read" in message
    assert "--list" in message


def test_a_bound_without_a_timezone_is_refused_at_the_door(home, samples):
    """A date-only bound would silently drop the March samples it meant to keep."""
    run("init")
    run("import", "--source", "structured", "--granularity", "sample", "--path", str(samples))
    code, message = errors("measure", "--what", "weight", "--since", "2026-03-01")
    assert code == 2 and "timezone-aware" in message


def test_a_series_that_changed_units_refuses_to_be_averaged_at_the_door(home, tmp_path):
    run("init")
    root = tmp_path / "mixed"
    root.mkdir()
    (root / "weight.jsonl").write_text("\n".join(json.dumps(row) for row in (
        {"time": "2026-03-01T08:00:00+00:00", "value": 70, "unit": "kg"},
        {"time": "2026-03-02T08:00:00+00:00", "value": 154, "unit": "lb"})) + "\n",
        encoding="utf-8")
    run("import", "--source", "structured", "--granularity", "sample", "--path", str(root))

    code, report = run("measure", "--what", "weight")
    assert code == 0
    assert report["statistics"] is None
    assert report["unit_conflict"] == ["kg", "lb"]
    assert "kg" in report["refused"] and "lb" in report["refused"]

    _code, named = run("measure", "--what", "weight", "--unit", "kg")
    assert named["statistics"]["mean"] == 70.0
    assert named["units_seen"] == {"kg": 1, "lb": 1}


# -- formation ---------------------------------------------------------------

# A machine that can afford inference: a route, an allowlist that admits it, a budget
# that is not zero, and one credential per route that has to exist.
INFERENCE_ENV = {
    "INFERENCE_ENABLED": "true",
    "BACKGROUND_BUDGET_TOKENS": "200000",
    "HINDSIGHT_URL": "http://127.0.0.1:8123",
    "ALLOWED_INFERENCE_HOSTS": "127.0.0.1",
    "TEXT_BASE_URL": "http://127.0.0.1:11434/v1",
    "EMBEDDINGS_BASE_URL": "http://127.0.0.1:11435/v1",
    "ROUTE_CREDENTIAL_RETAIN": "cred-retain",
    "ROUTE_CREDENTIAL_EMBEDDINGS": "cred-embed",
    "ROUTE_CREDENTIAL_CONSOLIDATE": "cred-consolidate",
    "ROUTE_CREDENTIAL_REFLECT": "cred-reflect",
    "ROUTE_CREDENTIAL_FOREGROUND": "cred-foreground",
}


@pytest.fixture()
def forming(tmp_path, monkeypatch):
    """An inference-enabled installation holding three records nothing has projected."""
    root = tmp_path / "forming"
    (root / "data").mkdir(parents=True)
    lines = [f"HERMES_MEMORY_DATA_DIR={root / 'data'}",
             f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}"]
    lines += [f"HERMES_MEMORY_{key}={value}" for key, value in INFERENCE_ENV.items()]
    (root / "hermes-memory.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(root))
    for key, value in INFERENCE_ENV.items():
        monkeypatch.delenv(f"HERMES_MEMORY_{key}", raising=False)
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        for index in range(3):
            store.commit(envelope(source_id=f"msg-{index}", text=f"note {index}"))
    return settings


class Answers:
    """The backend, stubbed: this suite must never reach for a model.

    A submission is answered with the operation identity it was made under, and the
    operation is what reports the cost — nothing has been read at submission time.
    """

    def __init__(self):
        self.retain_calls = []
        self.cancelled: list[str] = []
        self.asked: list[str] = []

    def retain_async(self, items, *, submission_id):
        self.retain_calls.append({"items": items, "submission_id": submission_id})
        return {"ok": True, "operation_id": submission_id}

    def cancel_operation(self, operation_id):
        self.cancelled.append(operation_id)
        return {"ok": True, "operation_id": operation_id, "state": "cancelled"}

    def operation(self, operation_id):
        self.asked.append(operation_id)
        return {"status": "completed", "usage": {"total_tokens": 321}}

    def document_state(self, document_id):
        return {"document_id": document_id, "state": "present", "count": 1}


@pytest.fixture()
def stubbed(monkeypatch):
    """Point the one socket-opening seam in formation at a stub."""
    from hermes_memory.processing import formation

    backend = Answers()
    monkeypatch.setattr(formation, "backend_client", lambda settings: backend)
    return backend


def test_form_prints_the_list_and_performs_nothing(forming):
    code, plan = run("form")
    assert code == 0
    assert plan["ok"] is True and plan["planned_jobs"] == 3
    assert plan["note"].startswith("planning only")
    assert not gate_path(forming).exists(), "the list was read, not spent"


def test_a_capture_only_installation_refuses_before_naming_any_work(home):
    run("init")
    code, plan = run("form")
    assert code == 0 and plan["ok"] is False
    assert any("inference is switched off" in line for line in plan["blocking"])
    assert plan["selected"] == []


def test_form_approves_only_the_list_that_was_shown(forming):
    plan = run("form")[1]
    code, message = errors("form", "--review", "not-the-digest", "--actor", OWNER)
    assert code == 2 and "does not match what would happen now" in message
    assert plan["review_digest"]


def test_an_approved_pass_names_its_actor_and_charges_the_shared_ledger(forming, stubbed):
    plan = run("form")[1]
    code, receipt = run("form", "--review", plan["review_digest"], "--actor", OWNER)
    assert code == 0
    assert receipt["actor"] == OWNER and receipt["queued"]["created"] == 3
    assert receipt["drain"]["counts"] == {"succeeded": 3}
    assert len(stubbed.retain_calls) == 3
    with GateStore(gate_path(forming)) as ledger:
        assert ledger.db.execute("SELECT sum(tokens) FROM budget_usage").fetchone()[0] == 963


def test_a_pass_asks_for_an_actor_because_it_spends_a_shared_device(forming, tmp_path,
                                                                   monkeypatch, stubbed):
    env_file = forming.home / "hermes-memory.env"
    env_file.write_text("".join(
        line + "\n" for line in env_file.read_text(encoding="utf-8").splitlines()
        if not line.startswith("HERMES_MEMORY_OWNER_PRINCIPAL")), encoding="utf-8")
    plan = run("form")[1]
    code, message = errors("form", "--review", plan["review_digest"])
    assert code == 2 and "actor must be named" in message
    assert stubbed.retain_calls == []


def test_the_command_takes_a_bound_because_it_is_a_pass_not_a_drain(forming):
    code, plan = run("form", "--limit", "1")
    assert code == 0 and plan["planned_jobs"] == 1 and len(plan["selected"]) == 1
    assert errors("form", "--limit", "0")[0] == 2


def test_an_operator_hold_stops_the_pass_that_was_approved_before_it(forming, stubbed):
    plan = run("form")[1]
    assert run("pause", "--scope", "inference", "--reason", "holding", "--actor", OWNER)[0] == 0
    code, message = errors("form", "--review", plan["review_digest"], "--actor", OWNER)
    assert code == 2 and "does not match" in message
    assert stubbed.retain_calls == []


def stranded_projection(settings, *, source_id="msg-stranded"):
    """A projection that was submitted and whose answer never came back."""
    with EvidenceStore(settings.db_path) as store:
        record = store.commit(envelope(source_id=source_id, text="an old note"))["id"]
        begun = DocumentMap(store, bank_id=settings.bank_id).begin(
            record, "1", async_submission=True)
    return record, begun["submission_id"]


def test_form_reconcile_asks_the_backend_about_work_we_cannot_account_for(forming, stubbed):
    record, submission = stranded_projection(forming)
    code, report = run("form", "--reconcile")
    assert code == 0
    assert report["settled"] == 1 and report["verified"] == 1
    assert report["asked"][0]["operation_id"] == submission
    assert stubbed.asked == [submission] and stubbed.retain_calls == [], \
        "reconciling asks about work already submitted; it does no work"
    with EvidenceStore(forming.db_path) as store:
        assert DocumentMap(store, bank_id=forming.bank_id).state(record, "1") == VERIFIED


def test_form_reconcile_reports_a_question_the_backend_could_not_answer(forming, stubbed,
                                                                       monkeypatch):
    """An unreachable backend is not an outcome, and the door says so rather than 0."""
    record, _ = stranded_projection(forming)

    class Unreachable:
        def operation(self, operation_id):
            raise HindsightUnavailable("connection refused while asking about the operation")

    from hermes_memory.processing import formation

    monkeypatch.setattr(formation, "backend_client", lambda settings: Unreachable())
    code, report = run("form", "--reconcile")
    assert code == 0 and report["unreachable"] == 1 and report["verified"] == 0
    with EvidenceStore(forming.db_path) as store:
        assert DocumentMap(store, bank_id=forming.bank_id).state(record, "1") == "submitted"


def test_form_reconcile_queues_nothing_so_it_asks_for_no_approval(forming, stubbed):
    code, message = errors("form", "--reconcile", "--review", "any-digest", "--actor", OWNER)
    assert code == 2 and "queues no work" in message
    assert stubbed.asked == [] and stubbed.retain_calls == []


def test_status_says_that_nothing_drains_the_queue_on_its_own(forming, stubbed):
    plan = run("form")[1]
    assert run("form", "--review", plan["review_digest"], "--actor", OWNER)[0] == 0
    code, report = run("status")
    assert code == 0
    assert report["capabilities"]["formation"] is True
    assert report["capabilities"]["formation_unattended"] is False, \
        "an installation that can form is not one that forms by itself"
    stage = next(item for item in report["stages"]["stages"] if item["name"] == "observations")
    assert stage["formation_unattended"] is False and stage["instance_hold"] is False


def test_status_reports_the_inference_hold_on_the_observations_too(forming):
    assert run("pause", "--scope", "inference", "--reason", "the models are moving",
               "--actor", OWNER)[0] == 0
    stage = next(item for item in run("status")[1]["stages"]["stages"]
                 if item["name"] == "observations")
    assert stage["state"] == "paused" and stage["instance_hold"] is True
    assert "holding inference" in stage["detail"]


# -- whose memory is being asked about ---------------------------------------

@pytest.fixture()
def enrolled(forming, tmp_path):
    """A second enrolled person on the same machine, whose archive is still empty."""
    registry = ProfileRegistry.open(forming)
    try:
        home = tmp_path / "homes" / "work"
        home.mkdir(parents=True)
        proposal = registry.plan("work", home)
        registry.enroll("work", home, actor=OWNER,
                        review_digest=proposal["review_digest"])
    finally:
        registry.db.close()
    data_dir = Path(proposal["data_dir"])
    with EvidenceStore(data_dir / "canonical.db"):
        pass
    return SimpleNamespace(home=home, data_dir=data_dir, store=data_dir / "canonical.db")


def test_status_answers_about_the_profile_the_owner_named(forming, enrolled):
    _, unqualified = run("status")
    assert unqualified["profile"] == "work", (
        "once a profile is enrolled, that memory is the one this installation maintains; "
        "answering from the empty default store reports real records as zero")
    code, work = run("status", "--hermes-home", str(enrolled.home))
    assert code == 0
    assert work["profile"] == "work"
    assert work["stages"]["profile"] == "work", (
        "the headline named one person and the stages answered for another")
    assert work["database"] == str(enrolled.store), "the answer named the wrong archive"


def test_two_memories_and_no_naming_is_a_question_not_a_guess(forming, enrolled, tmp_path):
    """Whose status? With two people on one machine the doors ask instead of choosing.

    The single-profile case resolves because there is only one archive to read; a second
    one makes any unqualified reading a coin toss about somebody's evidence.
    """
    registry = ProfileRegistry.open(forming)
    try:
        other = tmp_path / "homes" / "flat"
        other.mkdir(parents=True)
        proposal = registry.plan("flat", other)
        registry.enroll("flat", other, actor=OWNER, review_digest=proposal["review_digest"])
    finally:
        registry.db.close()

    for argv in (["status"], ["doctor"], ["measure", "--what", "weight"],
                 ["sources", "list"], ["audit", "--actions"]):
        code, message = errors(*argv)
        assert code == 2 and "2 memories are enrolled" in message, (
            f"`{' '.join(argv)}` answered for somebody: {message[:200]}")
        assert "--hermes-home" in message
    assert run("status", "--hermes-home", str(enrolled.home))[0] == 0
    assert run("doctor", "--hermes-home", str(enrolled.home))[0] in (0, 1)


def test_asking_about_a_profile_never_brings_the_instance_ledger_into_being(forming,
                                                                           tmp_path):
    ledger = forming.home / "installation.db"
    assert not ledger.exists()
    stranger = tmp_path / "homes" / "nobody"
    stranger.mkdir(parents=True)
    assert errors("status", "--hermes-home", str(stranger))[0] == 2
    assert not ledger.exists(), "a reading wrote the map it was consulting"


def test_the_named_profile_is_the_one_a_pass_would_project(forming, enrolled):
    """The default archive holds three records; the profile asked about holds none."""
    assert run("form")[1]["selected"] != []
    plan = run("form", "--hermes-home", str(enrolled.home))[1]
    assert plan["profile"] == "work" and plan["selected"] == []
    assert any("nothing is unprojected" in line for line in plan["blocking"])


def test_doctor_examines_the_memory_that_belongs_to_that_home(forming, enrolled):
    _, report = run("doctor", "--hermes-home", str(enrolled.home))
    assert report["profile"] == "work"
    dumped = json.dumps(report, sort_keys=True, default=str)
    assert str(enrolled.data_dir) in dumped and str(forming.data_dir) not in dumped


def test_an_unenrolled_home_is_refused_rather_than_answered_from_the_default(tmp_path,
                                                                            forming):
    stranger = tmp_path / "homes" / "stranger"
    stranger.mkdir(parents=True)
    code, message = errors("status", "--hermes-home", str(stranger))
    assert code == 2 and "no profile is enrolled" in message
    assert "hermes-memory enroll" in message, "the refusal says what to do instead"


def test_a_retired_home_lost_the_answer_with_the_mapping(forming, enrolled):
    registry = ProfileRegistry.open(forming)
    try:
        registry.retire("work", actor=OWNER, reason="the person left")
    finally:
        registry.db.close()
    for command in ("status", "doctor", "form"):
        code, message = errors(command, "--hermes-home", str(enrolled.home))
        assert code == 2 and "retired" in message, command


# -- the background pass -------------------------------------------------------

def reminded(*, at="2026-09-15T09:00:00+00:00", title="Call the dentist"):
    """One goal the owner asked for, written into the installation's own store."""
    from hermes_memory.prospective.due_events import DueEventLog
    from hermes_memory.prospective.goals import GoalStore

    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        return GoalStore(store, events=DueEventLog(store),
                         owner_principal=OWNER).propose(
                             title=title, statement="The appointment is booked.",
                             timezone_name="UTC", due=at, proposed_by=OWNER,
                             proposed_kind="owner")["id"]


def test_maintain_refuses_an_installation_that_does_not_exist_yet(home):
    """The answer is JSON with an exit code, because a timer reads one and a person the other."""
    code, report = run("maintain")
    assert code == 2 and report["ok"] is False
    assert "no canonical store" in report["refused"]
    assert "hermes-memory init" in report["refused"], \
        "the refusal says what to do instead"


def test_the_pass_creates_nothing_it_only_reads(home):
    assert run("init")[0] == 0
    ledger = home / "installation.db"
    assert not ledger.exists()
    assert run("maintain")[0] == 0
    assert not ledger.exists(), "a pass that looks at the admission ledger did not invent one"


def test_maintain_reports_every_section_and_spends_nothing(home):
    from hermes_memory.processing.maintenance import SECTIONS

    assert run("init")[0] == 0
    reminded()
    code, report = run("maintain", "--at", "2026-09-15T09:00:00+00:00")
    assert code == 0 and report["ok"] is True
    assert report["sections"] == list(SECTIONS)
    assert report["inference_paused"] is False
    assert report["proactive"]["deferred"] == 1, "shadow mode is this installation's default"
    with ReadOnlyStore(load_settings().db_path) as store:
        assert [row["state"] for row in store.db.execute("SELECT state FROM due_events")] \
            == ["pending"], "a door that promised to look left the promise standing"
        assert store.db.execute("SELECT count(*) AS n FROM outbox").fetchone()["n"] == 0


def test_a_single_section_can_be_asked_for(home):
    assert run("init")[0] == 0
    reminded()
    code, report = run("maintain", "--section", "queue")
    assert code == 0 and report["sections"] == ["queue"] and "proactive" not in report


def test_a_nonsense_section_never_reaches_the_pass(home):
    assert run("init")[0] == 0
    assert errors("maintain", "--section", "everything")[0] == 2


def test_a_bound_that_cannot_be_meant_is_refused(home):
    assert run("init")[0] == 0
    for limit in ("0", "-3"):
        code, message = errors("maintain", "--limit", limit)
        assert code == 2 and "--limit must be between" in message
    assert run("maintain", "--limit", "1")[0] == 0


def test_an_instant_the_door_cannot_place_in_time_is_refused(home):
    assert run("init")[0] == 0
    code, message = errors("maintain", "--at", "2026-09-15T09:00:00")
    assert code == 2 and "timezone" in message


def test_maintain_answers_for_the_profile_the_owner_named(forming, enrolled):
    from hermes_memory.processing.maintenance import SECTIONS

    reminded()
    code, report = run("maintain", "--hermes-home", str(enrolled.home))
    assert code == 0 and report["profile"] == "work"
    assert report["sections"] == list(SECTIONS)
    assert report["proactive"]["deferred"] == 0, \
        "the default archive's reminder was not this profile's to take"
    assert errors("maintain", "--hermes-home", str(enrolled.home / "nope"))[0] == 2


# -- stopping work ---------------------------------------------------------------

ROUTE = Route("retain", "remote-9b", "chat", "http://127.0.0.1:8080/v1", "cred", "freshness",
              2048)


def dispatch(*, operation="op-1", record="rec_a", state="running"):
    """Queue work and hand it to the backend, so a cancellation has something to stop.

    Two files are written because the design keeps them apart: the queue lives in the
    profile store, the operation lives in the instance's admission ledger.
    """
    from hermes_memory.backend.worker_launcher import OPERATION_KEY, OperationLedger
    from hermes_memory.processing.instance_gate import GateStore, gate_path

    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        jobs = JobQueue(store)
        job_id = jobs.enqueue(kind="retain", inputs=[record], input_revision="1", route=ROUTE,
                              processor_fingerprint="extractor-v3")["job_id"]
        jobs.mark_running(jobs.claim(worker="w1"), operation_id=operation)
    gate = GateStore(gate_path(settings))
    try:
        ledger = OperationLedger(gate)
        ledger.record({OPERATION_KEY: operation, "bank_id": settings.bank_id,
                       "operation_type": "retain"}, worker_id="w1",
                      bank_id=settings.bank_id, resource="remote-9b")
        ledger.mark(operation, state)
    finally:
        gate.db.close()
    return job_id


def test_cancel_refuses_an_installation_that_does_not_exist_yet(home):
    code, report = run("cancel", "--job", "job-1", "--reason", "mistake")
    assert code == 2 and report["ok"] is False and "no canonical store" in report["refused"]


def test_cancel_names_something_or_argparse_refuses(home):
    assert run("init")[0] == 0
    assert errors("cancel")[0] == 2
    assert errors("cancel", "--job", "job-1", "--operation", "op-1")[0] == 2


def test_a_cancellation_has_to_say_why(home):
    assert run("init")[0] == 0
    job_id = dispatch()
    code, message = errors("cancel", "--job", job_id)
    assert code == 2 and "reason must be named" in message


def test_cancel_stops_the_job_and_files_the_intent_it_could_not_confirm(home):
    """No backend is configured here, so the honest answer is: stopped ours, asked nobody."""
    assert run("init")[0] == 0
    job_id = dispatch()
    code, report = run("cancel", "--job", job_id, "--reason", "wrong bank")
    assert code == 0 and report["ok"] is True
    assert report["job_state"] == "cancelled" and report["actor"] == OWNER
    operation = report["operations"][0]
    assert operation["backend"] == "not_attempted"
    assert operation["cancellation"] == "requested"
    assert operation["state_before"] == "running"
    assert run("cancel", "--list")[1]["owed"][0]["operation_id"] == "op-1"


def test_cancel_an_operation_by_name_when_the_job_is_somebody_elses(home):
    """A caller may stop one operation without touching the queue row that produced it."""
    assert run("init")[0] == 0
    job_id = dispatch(operation="op-7")
    code, report = run("cancel", "--operation", "op-7", "--reason", "the owner asked")
    assert code == 0 and report["operation"] == "op-7"
    assert report["backend"] == "not_attempted"
    assert report["cancellation"] == "requested"
    with EvidenceStore(load_settings().db_path) as store:
        assert JobQueue(store).get(job_id).state == "running", \
            "the job kept running: only the operation was asked to stop"
    assert [row["operation_id"] for row in run("cancel", "--list")[1]["owed"]] == ["op-7"]


def test_the_listing_is_a_reading_that_creates_nothing(home):
    assert run("init")[0] == 0
    code, report = run("cancel", "--list")
    assert code == 0 and report["ledger"] == "absent" and report["owed"] == []
    assert not gate_path(load_settings()).exists()


def test_an_operation_no_ledger_recorded_is_refused_with_a_way_forward(home):
    assert run("init")[0] == 0
    code, report = run("cancel", "--operation", "op-none", "--reason", "mistake")
    assert code == 2 and "nothing has been dispatched" in report["refused"]


def test_a_job_cancel_needs_no_admission_ledger(home):
    """The queue is ours whether or not anything has ever been dispatched from here."""
    assert run("init")[0] == 0
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        job_id = JobQueue(store).enqueue(
            kind="retain", inputs=["rec_q"], input_revision="1", route=ROUTE,
            processor_fingerprint="extractor-v3")["job_id"]
    code, report = run("cancel", "--job", job_id, "--reason", "superseded")
    assert code == 0 and report["operations"] == []
    assert not gate_path(settings).exists()


def test_cancel_answers_only_for_the_profile_the_owner_named(forming, enrolled, stubbed):
    """A job id from one person's store is not a job in another person's.

    This installation has a backend route, so the cancellation does pick up the phone — over
    the seam the fixture wires. Left to derive its client from the owned env file, the door
    would be writing to whatever lives at the configured address, which makes the suite a
    fact about the machine rather than about this build.
    """
    assert not (enrolled.data_dir / "gate.db").exists()
    code, message = errors("cancel", "--job", "job-nope", "--reason", "mistake",
                           "--hermes-home", str(enrolled.home))
    assert code == 2 and "no queued job" in message
    job_id = dispatch()
    code, report = run("cancel", "--job", job_id, "--reason", "mistake")
    assert code == 0 and report["operations"][0]["backend"] == "confirmed"
    assert stubbed.cancelled == ["op-1"], "the stop was asked through the seam, not a socket"
    assert errors("cancel", "--job", job_id, "--reason", "mistake",
                  "--hermes-home", str(enrolled.home))[1].count("no queued job") == 1


def test_cancel_reports_a_settled_operation_without_taking_credit(home):
    assert run("init")[0] == 0
    dispatch(operation="op-3", state="finished")
    code, report = run("cancel", "--operation", "op-3", "--reason", "forgot earlier")
    assert code == 0 and report["backend"] == "not_attempted"
    assert "cannot reach into the past" in report["note"]
    assert run("cancel", "--list")[1]["owed"] == []


# -- the owner's prospective memory ---------------------------------------------

def proposed(*, title="Renew the passport", at="2026-09-15T09:00:00+00:00",
             kind="agent", proposer="agent:sess-1"):
    """An agent's proposal, written into the installation's own store."""
    from hermes_memory.prospective.due_events import DueEventLog
    from hermes_memory.prospective.goals import GoalStore

    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        return GoalStore(store, events=DueEventLog(store),
                         owner_principal=OWNER).propose(
                             title=title, statement="It expires in six weeks.",
                             timezone_name="UTC", due=at, proposed_by=proposer,
                             proposed_kind=kind)["id"]


def listed():
    return run("goal", "--list")[1]["goals"]


def test_goal_lists_what_is_owed(home):
    assert run("init")[0] == 0
    goal_id = reminded()
    code, report = run("goal", "--list")
    assert code == 0
    assert [item["id"] for item in report["goals"]] == [goal_id]
    assert report["goals"][0]["status"] == "active"


def test_an_agents_proposal_waits_for_the_owner_to_adopt_it(home):
    assert run("init")[0] == 0
    goal_id = proposed()
    assert [item["status"] for item in listed()] == ["candidate"]
    assert run("maintain", "--at", "2026-09-15T12:00:00+00:00")[1]["proactive"][
        "decided"] == 0, "nothing reminds for a promise nobody adopted"
    code, answer = run("goal", "--id", goal_id, "--activate", "--reason", "yes, that is mine")
    assert code == 0 and answer["status"] == "active"
    assert listed()[0]["status"] == "active"


def test_a_snooze_holds_the_reminder_and_says_until(home):
    assert run("init")[0] == 0
    goal_id = reminded()
    code, answer = run("goal", "--id", goal_id, "--snooze", "2026-09-16T09:00:00+00:00",
                       "--reason", "after the holiday")
    assert code == 0
    assert listed()[0]["snoozed_until"] == "2026-09-16T09:00:00+00:00"
    held = run("maintain", "--at", "2026-09-15T12:00:00+00:00")[1]["proactive"]
    assert held["decided"] == 0 and held["suppressed"] == 0, \
        "a held promise is neither said nor destroyed"


def test_a_snooze_that_would_suppress_nothing_points_at_the_other_door(home):
    """Snoozing past nothing is a mistake about which lever to pull, so it says so."""
    assert run("init")[0] == 0
    goal_id = reminded(at="2026-09-20T09:00:00+00:00")
    code, message = errors("goal", "--id", goal_id, "--snooze", "2026-09-19T09:30:00+00:00",
                           "--reason", "five minutes before it is due anyway")
    assert code == 2 and "revise the due time" in message
    assert listed()[0]["snoozed_until"] is None


def test_revising_moves_the_due_time_and_leaves_the_old_revision_on_record(home):
    from hermes_memory.ids import timestamp

    assert run("init")[0] == 0
    goal_id = reminded()
    code, answer = run("goal", "--id", goal_id, "--revise",
                       "--due", "2026-09-16T09:00:00+00:00", "--reason", "the deadline moved")
    assert code == 0 and answer["revision"] == 2
    assert listed()[0]["due_at"] == timestamp("2026-09-16T09:00:00+00:00")
    history = run("goal", "--id", goal_id, "--history")[1]["history"]
    assert [row["revision"] for row in history] == [1, 2]
    assert "the deadline moved" in history[1]["reason"]


def test_completing_a_goal_silences_the_reminder_that_was_queued(home):
    assert run("init")[0] == 0
    goal_id = reminded()
    code, answer = run("goal", "--id", goal_id, "--complete", "--reason", "called them")
    assert code == 0 and answer["status"] == "completed"
    assert listed() == []
    assert run("maintain", "--at", "2026-09-15T12:00:00+00:00")[1]["proactive"][
        "decided"] == 0


def test_a_goal_transition_has_to_say_who_and_why(home):
    assert run("init")[0] == 0
    goal_id = reminded()
    code, message = errors("goal", "--id", goal_id, "--complete")
    assert code == 2 and "reason" in message
    code, message = errors("goal", "--id", goal_id, "--complete", "--reason", "done",
                           "--actor", "agent:sess-1")
    assert code == 2 and "belongs to the owner" in message


def test_a_revision_that_would_change_nothing_is_refused(home):
    assert run("init")[0] == 0
    goal_id = reminded()
    code, message = errors("goal", "--id", goal_id, "--revise", "--reason", "tidying")
    assert code == 2 and "--due or --statement" in message


def test_the_goal_door_needs_a_verb_or_the_list(home):
    assert run("init")[0] == 0
    assert errors("goal")[0] == 2
    assert errors("goal", "--list", "--complete", "--id", "gol_x")[0] == 2
    assert errors("goal", "--complete", "--reason", "vague")[0] == 2
    assert errors("goal", "--list", "--due-now")[0] == 2


def test_what_is_due_says_what_it_is_waiting_on(home):
    """The owner's question is not "is it late" but "has the thing I asked for happened"."""
    assert run("init")[0] == 0
    reminded(title="Call the dentist")
    reminded(title="Chase the invoice", at="2026-09-15T09:00:00+00:00")
    code, report = run("goal", "--due-now")
    assert code == 0
    assert {item["title"] for item in report["due"]} == {"Call the dentist",
                                                         "Chase the invoice"}
    assert all(item["ready"] for item in report["due"]), "a plain reminder needs no check"
    assert all(item["conditions"] == [] for item in report["due"])


def test_a_condition_holds_a_due_promise_in_front_of_the_owner(home):
    assert run("init")[0] == 0
    from hermes_memory.prospective.due_events import DueEventLog
    from hermes_memory.prospective.goals import GoalStore
    from hermes_memory.ids import timestamp

    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        ledger = GoalStore(store, events=DueEventLog(store), owner_principal=OWNER)
        ledger.propose(title="The thing waited for", statement="Nothing yet.",
                       timezone_name="UTC", proposed_by=OWNER, proposed_kind="owner")
        waited = ledger.propose(
            title="Chase the reply", statement="Ask again once it lands.",
            timezone_name="UTC", due=timestamp("2026-09-15T09:00:00+00:00"),
            proposed_by=OWNER, proposed_kind="owner",
            conditions=[{"kind": "measured_threshold",
                         "params": {"subject": "invoice-9", "predicate": "days_overdue",
                                    "operator": ">", "value": 14, "unit": "days"}}])["id"]
    code, report = run("goal", "--due-now")
    item = [row for row in report["due"] if row["goal"] == waited][0]
    assert item["ready"] is False
    assert item["unknown"] == ["measured_threshold"], "nothing has been measured about it"
    assert item["conditions"][0]["state"] == "unknown"
    assert item["title"] == "Chase the reply"


def test_the_goal_door_refuses_an_installation_with_no_store(home):
    code, message = errors("goal", "--list")
    assert code == 2 and "no canonical store" in message


# -- the habits the archive might learn ----------------------------------------

RULE = {"all": [{"field": "source", "op": "eq", "value": "gmail"}]}


def proposed_lesson(*, lesson_id="chase-invoice", text="Chase an overdue invoice twice by "
                                                      "mail before phoning.", kind="agent",
                    proposer=None):
    """One candidate habit, written through the real store into this installation."""
    from hermes_memory.learning.lessons import LessonStore
    from hermes_memory.learning.outcomes import OutcomeLog

    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        record = _a_message(store, "Chase the invoice twice before phoning.")
        made = LessonStore(store, outcomes=OutcomeLog(store),
                           owner_principal=OWNER).propose(
                               lesson_id=lesson_id, text=text, applicability=RULE,
                               evidence=[record], proposed_by=proposer or "agent:session-1",
                               proposed_kind=kind)
    return made["id"], made["version"]


def test_a_proposed_habit_is_listed_for_the_owner_to_read(home):
    run("init")
    lesson_id, version = proposed_lesson()
    code, report = run("owner", "--list")
    listed = report["awaiting"][0]["lesson_candidates"]
    assert code == 0 and [item["lesson"] for item in listed] == [f"{lesson_id}@{version}"]
    assert listed[0]["proposed_kind"] == "agent", "the archive says who offered the rule"
    assert "Chase an overdue invoice" in listed[0]["text"], \
        "the sentence is the decision, so it has to be readable before it is approved"
    assert listed[0]["applicability"] == RULE


def test_only_the_owner_makes_a_habit_apply_and_says_why(home):
    run("init")
    lesson_id, version = proposed_lesson()
    code, message = errors("owner", "--activate-lesson", f"{lesson_id}@{version}",
                           "--actor", OWNER)
    assert code == 2 and "say why" in message
    code, message = errors("owner", "--activate-lesson", f"{lesson_id}@{version}",
                           "--actor", "agent:session-1", "--reason", "my own good idea")
    assert code == 2 and "only the owner" in message
    code, decision = run("owner", "--activate-lesson", f"{lesson_id}@{version}",
                         "--actor", OWNER, "--reason", "recognised as how we work")
    assert code == 0 and decision["status"] == "active"
    _, after = run("owner", "--list")
    assert after["awaiting"][0]["lesson_candidates"] == [], \
        "a habit the archive applies is not still asking to be"


def test_a_promotion_names_the_version_it_approves(home):
    """The versions of one lesson disagree with each other by construction."""
    run("init")
    lesson_id, first = proposed_lesson()
    second = proposed_lesson()[1]
    assert first == 1 and second == 2
    code, message = errors("owner", "--activate-lesson", lesson_id, "--actor", OWNER,
                           "--reason", "which one?")
    assert code == 2 and "names the version" in message
    code, message = errors("owner", "--activate-lesson", f"{lesson_id}@1", "--version", "2",
                           "--actor", OWNER, "--reason", "contradicting myself")
    assert code == 2 and "cannot be about two revisions" in message
    code, message = errors("owner", "--activate-lesson", f"{lesson_id}@one", "--actor",
                           OWNER, "--reason", "guessing")
    assert code == 2 and "not a lesson reference" in message
    code, decision = run("owner", "--activate-lesson", lesson_id, "--version", "2",
                         "--actor", OWNER, "--reason", "the newer wording is ours")
    assert code == 0 and decision["status"] == "active"


def test_a_habit_is_withdrawn_immediately_and_without_a_review_cycle(home):
    run("init")
    lesson_id, version = proposed_lesson()
    assert run("owner", "--activate-lesson", f"{lesson_id}@{version}", "--actor", OWNER,
               "--reason", "try it")[1]["status"] == "active"
    code, message = errors("owner", "--retract-lesson", lesson_id, "--actor", "agent:one",
                           "--reason", "inconvenient")
    assert code == 2 and "never by the agent" in message
    code, decision = run("owner", "--retract-lesson", lesson_id, "--actor", OWNER,
                         "--reason", "wrong for this client")
    assert code == 0 and decision["status"] == "retracted"


def test_the_owner_can_report_that_a_habit_worked_or_failed(home):
    """Distinct from an assistant saying so: a checked report is weighted, a claim is not."""
    run("init")
    lesson_id, version = proposed_lesson()
    with EvidenceStore(load_settings().db_path) as store:
        record = _a_message(store, "The client phoned after the second reminder.",
                            source_id="msg-2")
    code, report = run("owner", "--contradict-lesson", f"{lesson_id}@{version}",
                       "--actor", OWNER, "--reason", "it cost us the client's patience",
                       "--evidence", record)
    assert code == 0 and report["recorded"] is True
    assert report["counts_as_support"] is True, \
        "what the owner checked is evidence about the world, not about a message"
    code, report = run("owner", "--confirm-lesson", f"{lesson_id}@{version}",
                       "--actor", OWNER, "--reason", "it worked on the next two accounts")
    assert code == 0 and report["recorded"] is True
    _, listing = run("owner", "--list")
    counted = listing["awaiting"][0]["lesson_candidates"][0]
    assert counted["against"] >= 1 and counted["support"] >= 1, \
        "what the owner reported is visible beside what is being asked for"


def test_a_habit_the_archive_no_longer_stands_behind_is_named_for_review(home):
    run("init")
    lesson_id, version = proposed_lesson()
    assert run("owner", "--activate-lesson", f"{lesson_id}@{version}", "--actor", OWNER,
               "--reason", "ours now")[1]["status"] == "active"
    with EvidenceStore(load_settings().db_path) as store:
        record = store.db.execute("SELECT id FROM records LIMIT 1").fetchone()[0]
        store.hide(record, reason="the owner forgot it", actor=OWNER)
    code, report = run("owner", "--list")
    review = report["awaiting"][0]["lessons_for_review"]
    assert code == 0 and [item["id"] for item in review] == [f"{lesson_id}@{version}"]
    assert review[0]["reason"] == "evidence gone"
    _, status = run("status")
    assert status["stages"]["awaiting_owner"]["lessons_for_review"] == 1
    assert status["stages"]["awaiting_owner"]["lesson_candidates"] == 0
    assert any(note.startswith("learning:") for note in status["stages"]["notes"])


def test_a_proposed_habit_is_counted_among_the_owner_only_decisions(home):
    """The note that names the door has to include this one, or the count lies."""
    run("init")
    proposed_lesson()
    _, status = run("status")
    waiting = status["stages"]["awaiting_owner"]
    assert waiting["lesson_candidates"] == 1
    assert waiting["lessons_awaiting_promotion"][0]["id"] == "chase-invoice@1"
    assert any("1 decision(s) are yours alone" in note for note in status["stages"]["notes"]), \
        "the lesson is inside the count that names the door, not beside it"


def test_a_proposed_reminder_waits_for_the_owner_in_every_reading(home):
    """A goal an agent invented schedules nothing, so the waiting list is its only voice."""
    assert run("init")[0] == 0
    goal_id = proposed()
    _, status = run("status")
    waiting = status["stages"]["awaiting_owner"]
    assert waiting["goal_candidates"] == 1
    assert waiting["goals_awaiting_adoption"][0]["goal"] == goal_id
    assert waiting["goals_awaiting_adoption"][0]["wants_a_due_time"] is True
    assert any(note.startswith("prospective:") for note in status["stages"]["notes"])
    assert any("1 decision(s) are yours alone" in note
               for note in status["stages"]["notes"]), \
        "the reminder is inside the count that names the door"
    _, listing = run("owner", "--list")
    entry = listing["awaiting"][0]["goal_candidates"]
    assert [item["goal"] for item in entry] == [goal_id]
    assert entry[0]["title"] == "Renew the passport"
    assert entry[0]["proposed_kind"] == "agent"
    assert run("maintain", "--at", "2026-09-15T12:00:00+00:00")[1]["proactive"][
        "decided"] == 0, "waiting for the owner is not the same as being quiet"

def refused(*args):
    """The exit code of a command that printed nothing because it refused."""
    import contextlib
    import io

    from hermes_memory.cli import main

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            return main(list(args))
        except SystemExit as exit_code:
            return exit_code.code


def test_the_audit_ledger_can_be_read_four_ways(home):
    """A count of what happened, who did it, what only the owner decided, and one record.

    Each of these is a question an operator asks in a different situation, and until now
    the door opened onto only the raw ledger, which answers none of them by itself.
    """
    run("init")
    with EvidenceStore(load_settings().db_path) as store:
        store.commit(envelope(text="the invoice for the lamprey survey"))
        record = store.db.execute("SELECT id FROM records").fetchone()[0]
    code, actions = run("audit", "--actions")
    assert code == 0 and actions["actions"], "something has happened in this store"
    code, actors = run("audit", "--actors")
    assert code == 0 and isinstance(actors["actors"], dict)
    code, decisions = run("audit", "--decisions", "erasure")
    assert code == 0 and decisions["category"] == "erasure"
    assert decisions["decided_by"] == [], "nobody has confirmed an erasure in this store yet"
    code, timeline = run("audit", "--timeline", record)
    assert code == 0 and "text" not in timeline, \
        "a timeline is something an operator pastes into a message"
    _, quoted = run("audit", "--timeline", record, "--include-private")
    assert "text" in quoted


def test_an_unaskable_decision_category_is_refused_at_the_door(home):
    run("init")
    assert refused("audit", "--decisions", "whatever") == 2


def test_a_backup_can_be_told_how_many_snapshots_to_retain(home):
    run("init")
    with EvidenceStore(load_settings().db_path) as store:
        store.commit(envelope())
    assert refused("backup", "--keep", "0") == 2, "retention that keeps nothing is a delete"
    for _ in range(2):
        assert run("backup")[0] == 0
    _, listing = run("backup", "--list")
    assert len(listing["profiles"][0]["snapshots"]) == 2
    code, report = run("backup", "--keep", "1")
    assert code == 0 and report["pruned"][0]["profile"]
    assert report["backups"][0]["retained"] == 1
    _, after = run("backup", "--list")
    kept = after["profiles"][0]["snapshots"]
    assert len(kept) == 1, "the newest stays; retention removes the older copies"
