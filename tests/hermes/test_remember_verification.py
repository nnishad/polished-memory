"""F2: agent-written memories carry their trust state, and evidence is checked.

An unevidenced memory_remember is the agent's own prose, not a sighting: it is
stored labeled `agent_authored`, rendered as an unverified note, and is
inadmissible as evidence for a checked claim — closing the loop where the
agent's writing could certify itself a turn later. With evidence, only the
supported subset is stored, with a verification receipt.
"""
from __future__ import annotations

import json

import pytest

from conftest import envelope
from hermes_memory.backend.hindsight_client import HindsightError
from hermes_memory.context import ContextBroker
from hermes_memory.config import load_settings
from hermes_memory.install.profiles import ProfileRegistry, open_installation
from hermes_memory.processing.verification import verify_memory_claims
from hermes_memory.storage.evidence import EvidenceStore
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


def enroll(instance_home, activity_home, *, profile="default", owner=OWNER):
    db = open_installation(instance_home / "installation.db")
    registry = ProfileRegistry(db, root=instance_home, owner_principal=owner,
                              default_home=instance_home / "data")
    proposal = registry.plan(profile, activity_home)
    registry.enroll(profile, activity_home, actor=owner,
                    review_digest=proposal["review_digest"])
    return registry


@pytest.fixture()
def provider(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path / "profile"))
    enroll(tmp_path / "profile", tmp_path / "profile")
    instance = plugin.HermesMemoryProvider()
    assert instance.is_available(), instance.unavailable_reason()
    instance.initialize("sess-1", hermes_home=str(tmp_path / "profile"), platform="cli")
    return instance


def remember(provider, content, **extra):
    return json.loads(provider.handle_tool_call("memory_remember",
                                                {"content": content, **extra}))


def test_an_unevidenced_memory_is_stored_as_a_labeled_agent_note(provider):
    answer = remember(provider, "user asked me to remind them about the visa on friday")
    assert answer["ok"] is True and answer["verified"] is False
    assert answer["agent_authored"] is True
    with provider._open_store() as store:
        record = store.get(answer["id"])
    assert record.metadata["agent_authored"] is True
    assert "verification" not in record.metadata


def test_the_agent_note_is_rendered_as_unverified_on_recall(provider):
    answer = remember(provider, "user asked me to remind them about the visa on friday")
    with provider._open_store() as store:
        broker = ContextBroker(store, cache=None)
        packet = broker.assemble("visa friday remind", limit=5)
    item = next(item for item in packet.items if item.id == answer["id"])
    assert item.agent_authored is True
    assert item.as_dict()["agent_authored"] is True
    assert "agent note, unverified" in packet.render()


def test_an_agent_note_is_not_admissible_evidence(store, tmp_path, monkeypatch):
    home = tmp_path / "verify-home"
    home.mkdir()
    (home / "hermes-memory.env").write_text(
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_HINDSIGHT_URL=http://127.0.0.1:8888\n"
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS=127.0.0.1\n"
        "HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS=1000000\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    note = store.commit(envelope(text="the owner prefers oat milk",
                                 source_id="note-1",
                                 metadata={"agent_authored": True}))["id"]
    # The refusal happens at evidence resolution, before any model hop, so no
    # transport is needed: citing an unverified note is refused outright.
    with pytest.raises(HindsightError, match="not admissible evidence"):
        verify_memory_claims(load_settings(), store, [
            {"text": "the owner prefers oat milk", "record_ids": [note],
             "evidence": [{"record_id": note, "quote": "the owner prefers oat milk"}]}])


def test_strict_mode_refuses_an_unevidenced_personal_claim_but_not_an_operational_note(
        provider, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_REMEMBER_STRICT", "true")
    refused = remember(provider, "you are allergic to peanuts and must avoid them")
    assert refused["ok"] is False
    assert "needs evidence" in refused.get("error", "")
    allowed = remember(provider, "queued a reminder to ask about the visa on friday")
    assert allowed["ok"] is True and allowed["agent_authored"] is True


def test_an_evidenced_memory_stores_only_the_supported_subset_with_a_receipt(
        provider, monkeypatch):
    def fake_verify(settings, store, claims, account_id=None):
        return {"ok": True, "all_supported": False,
                "text": "the owner prefers coffee in the morning",
                "verification": {"contract": "scoped-evidence-entailment-v4",
                                 "model": "judge", "verifier": "independent",
                                 "checked": 1, "accepted": 1, "rejected": [],
                                 "claims_digest": "abc123"}}
    monkeypatch.setattr("hermes_memory.processing.verification.verify_memory_claims",
                        fake_verify)
    answer = remember(provider, "the owner prefers coffee in the morning and weighs 90kg",
                      evidence=[{"record_id": "r1", "quote": "coffee every morning"}])
    assert answer["ok"] is True and answer["verified"] is True
    assert answer["agent_authored"] is False
    with provider._open_store() as store:
        record = store.get(answer["id"])
    assert record.text == "the owner prefers coffee in the morning"
    assert record.metadata["agent_authored"] is False
    assert record.metadata["verification"]["claims_digest"] == "abc123"
    assert record.metadata["verification"]["verifier"] == "independent"


def test_a_statement_the_evidence_does_not_support_is_not_stored(provider, monkeypatch):
    def fake_verify(settings, store, claims, account_id=None):
        return {"ok": True, "all_supported": False, "text": "",
                "verification": {"contract": "c", "model": "m", "checked": 1,
                                 "accepted": 0, "rejected": [], "claims_digest": "x"}}
    monkeypatch.setattr("hermes_memory.processing.verification.verify_memory_claims",
                        fake_verify)
    answer = remember(provider, "the owner has a severe peanut allergy",
                      evidence=[{"record_id": "r1", "quote": "peanuts maybe"}])
    assert answer["ok"] is False
    assert "does not support" in answer.get("error", "")
