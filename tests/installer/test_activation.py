"""Activation ordering and fail-closed publication, with private synthetic stores."""
import json
import subprocess
from pathlib import Path

import pytest

from hermes_memory.config import load_settings
from hermes_memory.ids import content_digest
from hermes_memory.install import activation
from hermes_memory.install.release import _registerable_snapshot, _source_fingerprints
from hermes_memory.ids import digest
from hermes_memory.storage.evidence import EvidenceStore
from hermes_memory.install.upgrade import UpgradeError


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    home = tmp_path / "memory"
    home.mkdir()
    (home / "hermes-memory.env").write_text("HERMES_MEMORY_OWNER_PRINCIPAL=owner\n")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    settings = load_settings()
    with EvidenceStore(settings.db_path):
        pass
    old = home / "runtime/old"
    old.mkdir(parents=True)
    (home / "runtime/current").symlink_to(old)
    target = home / "runtime/target"
    registration = target / "source"
    registration.mkdir(parents=True)
    (registration / "pyproject.toml").write_text("[project]\nname='synthetic'\n")
    fingerprints = _source_fingerprints(registration)
    commit = _registerable_snapshot(registration, digest(fingerprints))
    (target / "deployment").mkdir()
    (target / "deployment/compatibility.json").write_text(json.dumps({"framework_digest": "paired"}))
    wheel = target / "synthetic.whl"
    wheel.write_bytes(b"synthetic wheel")
    (target / "RELEASE.json").write_text(json.dumps({"registration_source": str(registration),
        "registration_commit": commit, "source_fingerprints": fingerprints,
        "source_digest": digest(fingerprints), "wheel": str(wheel),
        "wheel_digest": content_digest(wheel.read_bytes())}))
    host = tmp_path / "host"
    host.mkdir()
    (host / "config.yaml").write_text("memory:\n  provider: hermes-memory\n")
    (host / "plugins").mkdir()
    (host / "plugins/.install-metadata.json").write_text("{}")
    host_source = tmp_path / "hermes-source"
    (host_source / "venv/bin").mkdir(parents=True)
    (host_source / "venv/bin/hermes").write_text("synthetic")
    monkeypatch.setattr(activation, "verify", lambda **_: {"complete": True, "missing": []})
    monkeypatch.setattr(activation, "services_plan", lambda *_, **__: {
        "blocked": [], "would_change": [], "missing_paths": []})
    monkeypatch.setattr(activation, "_in_flight", lambda _: {"held": [], "unreadable": 0})
    calls = []

    def runner(argv, **kwargs):
        argv = list(map(str, argv))
        calls.append(argv)
        output = ""
        if "show" in argv:
            output = f"Environment=HERMES_HOME={host}\nExecStart={host_source}/venv/bin/python\n"
        elif "is-active" in argv:
            output = "inactive\n"
        elif activation.PROBE in argv:
            output = json.dumps({"digest": "paired", "schema": 17})
        elif "install" in argv and "plugins" in argv:
            (host / "plugins/.install-metadata.json").write_text(json.dumps({
                "hermes-memory": {"revision": commit}}))
        return subprocess.CompletedProcess(argv, 0, output, "")

    options = {"release": target, "hermes_home": host, "hermes_source": host_source, "runner": runner}
    return settings, options, calls, home, old


def test_plan_does_not_stop_or_publish(candidate):
    settings, options, calls, home, old = candidate
    proposal = activation.activation_plan(settings, **options)
    assert proposal["blocking"] == []
    assert (home / "runtime/current").resolve() == old
    assert not (home / "activation-backups").exists()
    assert not any("stop" in call or "install" in call for call in calls)


def test_owner_and_changed_review_refuse_before_quiescence(candidate):
    settings, options, calls, home, _ = candidate
    proposal = activation.activation_plan(settings, **options)
    with pytest.raises(UpgradeError, match="owner"):
        activation.activate_release(settings, **options, actor="agent", review=proposal["review_digest"])
    with pytest.raises(UpgradeError, match="changed review"):
        activation.activate_release(settings, **options, actor="owner", review="wrong")
    assert not any("stop" in call for call in calls)


def test_rehearsal_precedes_host_install_and_atomic_publication(candidate):
    settings, options, calls, home, _ = candidate
    proposal = activation.activation_plan(settings, **options)
    receipt = activation.activate_release(settings, **options, actor="owner", review=proposal["review_digest"])
    assert receipt["published"] and receipt["state"] == "started-unverified"
    assert (home / "runtime/current").resolve() == options["release"]
    migration_calls = [index for index, call in enumerate(calls) if activation.MIGRATE in call]
    install_index = next(index for index, call in enumerate(calls) if "pip" in call)
    start_index = next(index for index, call in enumerate(calls) if "start" in call)
    assert migration_calls[0] < install_index < migration_calls[1] < start_index
    assert Path(receipt["backup"], "canonical-0.db").is_file()


def test_rehearsal_failure_never_installs_publishes_or_restarts_old_code(candidate):
    settings, options, calls, home, old = candidate
    original = options["runner"]

    def failed(argv, **kwargs):
        if activation.MIGRATE in argv:
            return subprocess.CompletedProcess(argv, 1, "", "synthetic failure")
        return original(argv, **kwargs)

    options["runner"] = failed
    proposal = activation.activation_plan(settings, **options)
    with pytest.raises(UpgradeError, match="command failed"):
        activation.activate_release(settings, **options, actor="owner", review=proposal["review_digest"])
    assert (home / "runtime/current").resolve() == old
    assert not any("pip" in call or "start" in call for call in calls)
    saved = next((home / "activation-backups").glob("*/receipt.json"))
    assert json.loads(saved.read_text())["state"] == "failed-stopped-forward-repair-required"
