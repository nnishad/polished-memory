"""Operator CLI: readiness per capability, and every reading opened read-only."""
from __future__ import annotations

import json
import os

import pytest

from hermes_memory.cli import main
from hermes_memory.config import load_settings
from hermes_memory.ids import now
from hermes_memory.lifecycle.erasure import ErasureManager
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
