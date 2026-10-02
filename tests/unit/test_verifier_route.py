"""F3: the entailment judge may be a different admitted route than the writer.

The gate selects an upstream by credential, so independence is a property of the
credential+model pair presented on the verification hop, not of a second base
URL. These tests pin that the hop really presents the verifier's credential and
model, that a "verifier" sharing the generation credential is refused, and that
a summary's processor fingerprint names its judge.
"""
import json

import pytest

from hermes_memory.backend.hindsight_client import HindsightError, TransportResult
from hermes_memory.config import ModelRoute, load_settings
from hermes_memory.processing.routes import build_routes
from hermes_memory.processing.summarization import summary_fingerprint
from hermes_memory.processing.synthesis import ScopedSynthesizer

BASE = {
    "DATA_DIR": None,
    "INFERENCE_ENABLED": "true",
    "OWNER_PRINCIPAL": "jugaadu",
    "HINDSIGHT_URL": "http://127.0.0.1:8888",
    "ALLOWED_INFERENCE_HOSTS": "127.0.0.1,192.168.68.65",
    "TEXT_BASE_URL": "http://192.168.68.65:8080/v1",
    "TEXT_RESOURCE": "remote-9b",
    "EMBEDDINGS_BASE_URL": "http://127.0.0.1:11434/v1",
    "EMBEDDINGS_RESOURCE": "local-gpu",
}


def configured(tmp_path, monkeypatch, *, credentials, over=None):
    home = tmp_path / "instance"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    values = dict(BASE, DATA_DIR=str(home / "data"))
    for key, value in (over or {}).items():
        if value is None:
            values.pop(key)
        else:
            values[key] = value
    lines = [f"HERMES_MEMORY_{key}={value}" for key, value in values.items()]
    lines += [f"HERMES_MEMORY_ROUTE_CREDENTIAL_{name}={value}"
              for name, value in credentials.items()]
    (home / "hermes-memory.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_settings()


def synthesizer(calls, *, verifier_credential=None, verifier_model=None,
                credential="reflect-credential", model="writer-model"):
    def transport(method, url, payload, headers):
        calls.append((method, url, payload, headers))
        value = ({"verdicts": [{"claim_index": 0, "label": "supported"}]}
                 if "entailment" in payload["messages"][0]["content"]
                 else {"claims": [{"text": "the speaker does not want tea now",
                                   "record_ids": ["r1"],
                                   "evidence": [{"record_id": "r1",
                                                 "quote": "mujhe chai nahi chahiye"}]}]})
        return TransportResult(200, {"choices": [{"message": {"content": json.dumps(value)}}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    return ScopedSynthesizer(base_url="http://127.0.0.1:8123/v1", credential=credential,
                             model=model, transport=transport,
                             verifier_credential=verifier_credential,
                             verifier_model=verifier_model)


CLAIMS = [{"text": "the speaker does not want tea now", "record_ids": ["r1"],
           "evidence": [{"record_id": "r1", "quote": "mujhe chai nahi chahiye"}]}]
EVIDENCE = [{"record_id": "r1", "text": "mujhe chai nahi chahiye"}]


def test_a_configured_verifier_is_admitted_only_with_its_own_credential(tmp_path, monkeypatch):
    settings = configured(tmp_path, monkeypatch, credentials={
        "RETAIN": "cred-retain", "EMBEDDINGS": "cred-embed", "VERIFIER": "cred-verifier"},
        over={"VERIFIER_BASE_URL": "http://192.168.68.65:8081/v1",
              "VERIFIER_MODEL": "judge-model", "VERIFIER_RESOURCE": "remote-9b"})
    table = build_routes(settings, credentials=settings.route_credentials)
    assert "verifier" in table.names()
    assert table.by_name("verifier").credential == "cred-verifier"
    assert table.by_name("verifier").upstream == "http://192.168.68.65:8081/v1"


def test_a_verifier_without_a_credential_is_withheld_like_any_other_route(tmp_path, monkeypatch):
    settings = configured(tmp_path, monkeypatch, credentials={
        "RETAIN": "cred-retain", "EMBEDDINGS": "cred-embed"},
        over={"VERIFIER_BASE_URL": "http://192.168.68.65:8081/v1",
              "VERIFIER_MODEL": "judge-model"})
    table = build_routes(settings, credentials=settings.route_credentials)
    assert "verifier" not in table.names()
    assert "verifier" in table.withheld


def test_no_verifier_configured_leaves_the_table_without_one(tmp_path, monkeypatch):
    settings = configured(tmp_path, monkeypatch, credentials={
        "RETAIN": "cred-retain", "EMBEDDINGS": "cred-embed"})
    table = build_routes(settings, credentials=settings.route_credentials)
    assert "verifier" not in table.names() and "verifier" not in table.withheld


def test_a_verifier_sharing_the_generation_credential_is_refused():
    with pytest.raises(HindsightError, match="distinct admitted route"):
        ScopedSynthesizer(base_url="http://127.0.0.1:8123/v1", credential="same",
                          model="writer-model", verifier_credential="same",
                          verifier_model="judge-model")


def test_a_verifier_credential_without_a_model_is_refused():
    with pytest.raises(HindsightError, match="its own model name"):
        ScopedSynthesizer(base_url="http://127.0.0.1:8123/v1", credential="writer-cred",
                          model="writer-model", verifier_credential="judge-cred",
                          verifier_model=None)


def test_the_verification_hop_presents_the_verifier_credential_and_model():
    calls = []
    client = synthesizer(calls, verifier_credential="judge-cred", verifier_model="judge-model")
    outcome = client.verify_claims(CLAIMS, evidence=EVIDENCE)
    assert len(calls) == 1
    assert calls[0][3]["Authorization"] == "Bearer judge-cred"
    assert calls[0][2]["model"] == "judge-model"
    assert outcome["verification"]["verifier"] == "independent"
    assert outcome["verification"]["model"] == "judge-model"


def test_without_a_verifier_the_hop_stays_on_the_generation_route_and_says_so():
    calls = []
    client = synthesizer(calls)
    outcome = client.verify_claims(CLAIMS, evidence=EVIDENCE)
    assert calls[0][3]["Authorization"] == "Bearer reflect-credential"
    assert calls[0][2]["model"] == "writer-model"
    assert outcome["verification"]["verifier"] == "generation-route"


def test_synthesis_writes_on_one_route_and_verifies_on_the_other():
    calls = []
    client = synthesizer(calls, verifier_credential="judge-cred", verifier_model="judge-model")
    client.synthesize("tea preference", evidence=EVIDENCE, max_tokens=64)
    assert len(calls) == 2
    assert calls[0][3]["Authorization"] == "Bearer reflect-credential"
    assert calls[0][2]["model"] == "writer-model"
    assert calls[1][3]["Authorization"] == "Bearer judge-cred"
    assert calls[1][2]["model"] == "judge-model"


def test_the_summary_fingerprint_names_its_judge(tmp_path, monkeypatch):
    plain = configured(tmp_path / "plain", monkeypatch, credentials={
        "RETAIN": "cred-retain", "EMBEDDINGS": "cred-embed", "REFLECT": "cred-reflect"})
    judged = configured(tmp_path / "judged", monkeypatch, credentials={
        "RETAIN": "cred-retain", "EMBEDDINGS": "cred-embed", "REFLECT": "cred-reflect",
        "VERIFIER": "cred-verifier"},
        over={"VERIFIER_BASE_URL": "http://192.168.68.65:8081/v1",
              "VERIFIER_MODEL": "judge-model"})
    route = build_routes(plain, credentials=plain.route_credentials).by_name("reflect")
    assert summary_fingerprint(plain, route) != summary_fingerprint(judged, route)
