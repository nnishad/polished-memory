"""F1.a: the provider exposes the pre-send boundary to the host.

The host calls pre_send(draft, turn_context) as its last door; the provider checks
the draft against the packet this turn injected. These pin the wire shape the
host relies on and that a memory-shaped draft with nothing to cite is flagged,
not passed.
"""
from __future__ import annotations

import json

import pytest

from hermes_memory.install.profiles import ProfileRegistry, open_installation
from plugin_loader import load_plugin

OWNER = "jugaadu"


@pytest.fixture()
def plugin(tmp_path, monkeypatch):
    home = tmp_path / "memory-home"
    home.mkdir()
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={tmp_path / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    return load_plugin()


@pytest.fixture()
def provider(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path / "profile"))
    db = open_installation(tmp_path / "profile" / "installation.db")
    registry = ProfileRegistry(db, root=tmp_path / "profile", owner_principal=OWNER,
                              default_home=tmp_path / "profile" / "data")
    proposal = registry.plan("default", tmp_path / "profile")
    registry.enroll("default", tmp_path / "profile", actor=OWNER,
                    review_digest=proposal["review_digest"])
    instance = plugin.HermesMemoryProvider()
    assert instance.is_available(), instance.unavailable_reason()
    instance.initialize("sess-1", hermes_home=str(tmp_path / "profile"), platform="cli")
    return instance


def test_a_non_memory_draft_passes_without_touching_the_model(provider):
    verdict = json.loads(json.dumps(provider.pre_send("The weather looks nice today.")))
    assert verdict["ok"] is True
    assert verdict["disposition"] == "pass"
    assert verdict["detector"]["skipped_reason"] == "no_memory_claims"
    assert verdict["receipt"] == {}


def test_a_memory_claim_with_nothing_injected_is_flagged_in_warn(provider):
    provider.prefetch("peanut allergy", session_id="sess-1")
    verdict = provider.pre_send("You are allergic to peanuts and must avoid them.",
                                {"session_id": "sess-1"})
    assert verdict["ok"] is True
    assert verdict["disposition"] == "revise"
    assert verdict["mode"] == "warn"
    assert verdict["action"]["mode"] == "none"
    assert verdict["receipt"]["id"].startswith("bnd_")


def test_the_host_sees_the_claim_labels_and_receipt_for_audit(provider):
    provider.prefetch("peanut allergy", session_id="sess-1")
    verdict = provider.pre_send("You are allergic to peanuts and must avoid them.",
                                {"session_id": "sess-1"})
    assert verdict["claims"][0]["label"] == "insufficient_evidence"
    assert verdict["receipt"]["boundary_version"] == "send-boundary-v1"
