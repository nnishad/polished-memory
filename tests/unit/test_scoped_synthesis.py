import json

import pytest

from hermes_memory.backend.hindsight_client import HindsightError, TransportResult
from hermes_memory.processing.synthesis import ScopedSynthesizer
from hermes_memory.processing.synthesis import claims_json


def test_exact_whole_message_json_envelope_is_supported():
    assert claims_json('```json\n{"claims": []}\n```') == {"claims": []}


@pytest.mark.parametrize("value", ['Here is JSON: {"claims": []}',
    '```json\n{"claims": []}\n```\nextra prose',
    '```json\n{"claims": []}\n```\n```json\n{}\n```', '{"claims": ['])
def test_prose_fragments_and_truncated_json_are_not_repaired(value):
    with pytest.raises(ValueError):
        claims_json(value)


def synthesizer(claims, calls, labels=None):
    def transport(method, url, payload, headers):
        calls.append((method, url, payload, headers))
        value = ({"verdicts": [{"claim_index": index, "label": label}
                  for index, label in enumerate(labels or ["supported"] * len(claims))]}
                 if "entailment" in payload["messages"][0]["content"] else {"claims": claims})
        return TransportResult(200, {"choices": [{"message": {"content": json.dumps(value)}}],
                                     "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
    return ScopedSynthesizer(base_url="http://127.0.0.1:8123/v1", credential="reflect-credential",
                             model="explicit-model", transport=transport)


def test_summary_input_and_actual_support_are_explicit_and_gated():
    calls = []
    client = synthesizer([{"text": "The speaker does not want tea now", "record_ids": ["r1"],
                           "evidence": [{"record_id": "r1", "quote": "mujhe chai nahi chahiye"}]}], calls)
    result = client.synthesize("tea preference", evidence=[
        {"record_id": "r1", "text": "mujhe chai nahi chahiye"},
        {"record_id": "r2", "text": "unrelated note"}], max_tokens=100)
    assert result["record_ids"] == ["r1"]
    assert result["admission_accounted"] is True
    assert len(calls) == 2 and result["verification"]["accepted"] == 1
    assert result["input_tokens"] == 200 and result["output_tokens"] == 40
    assert calls[0][1] == "http://127.0.0.1:8123/v1/chat/completions"
    assert calls[0][3]["Authorization"] == "Bearer reflect-credential"
    output = calls[0][2]["response_format"]
    assert output["type"] == "json_object" and output["schema"]["additionalProperties"] is False
    assert output["schema"]["properties"]["claims"]["items"]["properties"][
        "record_ids"]["items"]["enum"] == ["r1", "r2"]
    assert "mujhe chai nahi chahiye" in calls[0][2]["messages"][1]["content"]


@pytest.mark.parametrize("claims", [[], [{"text": "guess", "record_ids": []}],
                                    [{"text": "foreign", "record_ids": ["r2"]}]])
def test_unsupported_summary_output_is_not_publishable(claims):
    client = synthesizer(claims, [])
    with pytest.raises(HindsightError):
        client.synthesize("q", evidence=[{"record_id": "r1", "text": "evidence"}], max_tokens=100)


@pytest.mark.parametrize("url", ["http://192.168.68.69:8080/v1", "https://127.0.0.1/v1",
                                 "http://127.0.0.1/v1?upstream=remote"])
def test_synthesis_cannot_bypass_the_owned_admission_endpoint(url):
    with pytest.raises(HindsightError):
        ScopedSynthesizer(base_url=url, credential="cred", model="model")


@pytest.mark.parametrize("evidence", [[{}], [False], [{"record_id": [], "text": "x"}],
                                      [{"record_id": "r1", "text": None}]])
def test_malformed_evidence_is_refused_before_network(evidence):
    calls = []
    with pytest.raises(HindsightError):
        synthesizer([], calls).synthesize("q", evidence=evidence, max_tokens=100)
    assert calls == []


@pytest.mark.parametrize("usage", [False, [], {"prompt_tokens": -1},
                                   {"prompt_tokens": "100"}, {"completion_tokens": True}])
def test_malformed_usage_is_a_controlled_failure(usage):
    client = ScopedSynthesizer(base_url="http://127.0.0.1:8123/v1", credential="cred", model="model",
        transport=lambda *args: TransportResult(200, {
            "choices": [{"message": {"content": json.dumps({"claims": [
                {"text": "Supported", "record_ids": ["r1"]}]})}}], "usage": usage}))
    with pytest.raises(HindsightError, match="token usage"):
        client.synthesize("q", evidence=[{"record_id": "r1", "text": "evidence"}], max_tokens=100)
