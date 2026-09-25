"""Operator CLI: readiness is per stage, and outstanding debt cannot be hidden."""
from __future__ import annotations

import json

import pytest

from hermes_memory.cli import main
from hermes_memory.config import load_settings
from hermes_memory.ids import now
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.storage.evidence import EvidenceStore

SOURCE = "gmail"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    root = tmp_path / "hm"
    root.mkdir()
    (root / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={tmp_path / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n"
        "HERMES_MEMORY_OWNER_PRINCIPAL=jugaadu\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(root))
    monkeypatch.delenv("HERMES_MEMORY_DATA_DIR", raising=False)
    return root


def run(*argv):
    capture = pytest.MonkeyPatch()
    out = []
    capture.setattr("builtins.print", lambda *a, **k: out.append(a[0] if a else ""))
    try:
        code = main(list(argv))
    finally:
        capture.undo()
    return code, json.loads(out[-1])


def test_init_creates_a_store_and_status_reports_capture_only(home):
    assert run("init")[0] == 0
    code, report = run("status")
    assert code == 0
    assert report["capture_only"] is True
    assert report["hindsight_route"] == "unset"
    assert report["erasure"]["confirmable_by"] == "jugaadu"


def test_init_dry_run_writes_nothing(home):
    code, report = run("init", "--dry-run")
    assert code == 0 and report["would_create"]
    assert not (load_settings().db_path).exists()


def test_doctor_is_read_only_and_names_the_missing_store(home):
    code, report = run("doctor")
    assert code == 1
    assert report["ready_for_capture"] is False
    assert report["problems"] == ["data directory does not exist; run `hermes-memory init`"]
    assert not load_settings().db_path.exists()


def test_doctor_distinguishes_capture_from_formation_readiness(home):
    run("init")
    code, report = run("doctor")
    assert code == 0
    assert report["ready_for_capture"] is True
    assert report["ready_for_formation"] is False, "no route or budget means no formation"
    assert report["ready_for_forgetting"] is True


def test_an_outstanding_erasure_obligation_makes_doctor_fail(home, tmp_path):
    """Forgetting is not finished while a derived copy still exists elsewhere."""
    run("init")
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        record = store.commit({
            "source": SOURCE, "source_id": "msg-1", "revision": "1", "kind": "email",
            "text": "invoice 42 is paid", "observed_at": now(),
            "occurred_at": None, "occurred_precision": "unknown", "metadata": {},
        })["id"]
        store.project(record, "1", backend="hindsight", bank_id="hermes",
                      document_id="hdoc" + record.removeprefix("rec_"))
        manager = ErasureManager(store, owner_principal="jugaadu")
        preview = manager.preview(record_ids=[record], actor="jugaadu", reason="withdrawn")
        manager.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor="jugaadu")

    code, report = run("doctor")
    assert code == 1, "an unfinished erasure must not report a clean bill"
    assert report["erasure"]["obligations_outstanding"] == 2
    assert report["erasure"]["awaiting_confirmation"] == 0
    # Capture still works; the debt is a problem, not a broken stage.
    assert report["ready_for_capture"] is True
    assert any("outstanding" in problem for problem in report["problems"])


def test_an_unnamed_owner_shows_forgetting_as_unconfirmable(home, monkeypatch):
    env = home / "hermes-memory.env"
    env.write_text(env.read_text(encoding="utf-8").replace("=jugaadu", "="), encoding="utf-8")
    run("init")
    code, report = run("doctor")
    assert code == 0
    assert report["forgetting"] == "requests can be previewed but not confirmed"
    assert report["ready_for_forgetting"] is False


def test_a_refused_route_exits_two(home, monkeypatch):
    (home / "hermes-memory.env").write_text(
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_HINDSIGHT_URL=https://api.openai.com/v1\n",
        encoding="utf-8",
    )
    capture = pytest.MonkeyPatch()
    errors = []
    capture.setattr("sys.stderr.write", lambda text: errors.append(text))
    try:
        code = main(["status"])
    finally:
        capture.undo()
    assert code == 2
    assert any("configuration refused" in line for line in errors)
