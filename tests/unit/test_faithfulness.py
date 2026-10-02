import json

import pytest

from hermes_memory.backend.hindsight_client import HindsightError, TransportResult
from hermes_memory.processing.faithfulness import (canonical_evidence, normalize_claims,
    validate_publication, verdicts_json, verified_result, deterministic_rejections)
from hermes_memory.processing.synthesis import ScopedSynthesizer


def claim(text, quote="Nisha is allergic to peanuts.", ref="r1"):
    return {"text": text, "record_ids": [ref], "evidence": [{"record_id": ref, "quote": quote}]}


def adapter(outputs):
    calls = []
    def transport(method, url, payload, headers):
        calls.append(payload)
        output = outputs[len(calls) - 1]
        if isinstance(output, TransportResult):
            return output
        return TransportResult(200, {"choices": [{"finish_reason": "stop",
            "message": {"content": json.dumps(output)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
    return ScopedSynthesizer(base_url="http://127.0.0.1:8123/v1", credential="synthetic",
                              model="synthetic-verifier", transport=transport), calls


EVIDENCE = [{"record_id": "r1", "revision": "1", "text": "Nisha is allergic to peanuts."}]


@pytest.mark.parametrize("text,label", [
    ("Nisha weighs 90 kilograms.", "insufficient_evidence"),
    ("Nisha has a severe peanut allergy.", "insufficient_evidence"),
    ("Nisha is allergic to almonds.", "insufficient_evidence"),
    ("Ravi is allergic to peanuts.", "insufficient_evidence"),
    ("Nisha is not allergic to peanuts.", "contradicted"),
    ("Nisha is allergic to peanuts and weighs 90 kg.", "insufficient_evidence"),
])
def test_valid_id_and_real_quote_do_not_bypass_negative_verdict(text, label):
    client, calls = adapter([{"claims": [claim(text)]},
        {"verdicts": [{"claim_index": 0, "label": label}]}])
    with pytest.raises(HindsightError, match="no claims were supported"):
        client.synthesize("personal details", evidence=EVIDENCE, max_tokens=512)
    assert len(calls) == 2


def test_mixed_result_publishes_only_checked_claims_and_charges_both_calls():
    client, calls = adapter([{"claims": [claim("Nisha is allergic to peanuts."),
                                        claim("Nisha weighs 90 kg.")]},
        {"verdicts": [{"claim_index": 1, "label": "insufficient_evidence"},
                      {"claim_index": 0, "label": "supported"}]}])
    output = client.synthesize("details", evidence=EVIDENCE, max_tokens=512)
    assert output["text"] == "Nisha is allergic to peanuts."
    assert output["verification"]["checked"] == 2
    assert output["verification"]["rejected"] == [{"claim_index": 1, "label": "insufficient_evidence"}]
    assert output["input_tokens"] == 200 and output["output_tokens"] == 40
    assert len(calls) == 2
    assert validate_publication(output, EVIDENCE) == output


@pytest.mark.parametrize("quote", ["Nisha weighs 90 kg.", "", "Nisha IS allergic to peanuts."])
def test_invented_or_changed_quote_is_refused_before_verifier(quote):
    client, calls = adapter([{"claims": [claim("Nisha weighs 90 kg.", quote=quote)]}])
    with pytest.raises(HindsightError):
        client.synthesize("details", evidence=EVIDENCE, max_tokens=512)
    assert len(calls) == 1


@pytest.mark.parametrize("value", [None, {}, {"verdicts": []},
    {"verdicts": [{"claim_index": 0, "label": "relevant"}]},
    {"verdicts": [{"claim_index": True, "label": "supported"}]},
    {"verdicts": [{"claim_index": 1, "label": "supported"}]},
    {"verdicts": [{"claim_index": 0, "label": "supported", "extra": "ignore"}]}])
def test_malformed_or_missing_verdicts_never_mean_supported(value):
    with pytest.raises(HindsightError):
        verdicts_json(value, 1)


def test_duplicate_verdicts_are_refused():
    with pytest.raises(HindsightError):
        verdicts_json({"verdicts": [{"claim_index": 0, "label": "supported"}] * 2}, 2)


@pytest.mark.parametrize("response", [TransportResult(503, {}), TransportResult(0, None),
    TransportResult(200, {"choices": [{"finish_reason": "length",
        "message": {"content": '{"verdicts": []}'}}]})])
def test_failed_or_truncated_verifier_withholds_generated_answer(response):
    client, calls = adapter([{"claims": [claim("Nisha is allergic to peanuts.")]}, response])
    with pytest.raises(HindsightError):
        client.synthesize("details", evidence=EVIDENCE, max_tokens=512)
    assert len(calls) == 2


@pytest.mark.parametrize("text", ["Mujhe chai nahi chahiye, coffee pasand hai.",
    "मुझे चाय नहीं चाहिए, कॉफी पसंद है।", "Shayad kal Pune jaunga.",
    "Agar baarish hui toh nahi jaunga."])
def test_original_language_is_preserved_in_exact_spans_and_verification_input(text):
    evidence = [{"record_id": "r1", "revision": "2", "text": text}]
    client, calls = adapter([{"verdicts": [{"claim_index": 0, "label": "supported"}]}])
    output = client.verify_claims([claim(text, quote=text)], evidence=evidence)
    span = output["claims"][0]["evidence"][0]
    assert span == {"record_id": "r1", "quote": text, "start": 0,
                    "end": len(text), "revision": "2"}
    assert text in calls[0]["messages"][1]["content"]


@pytest.mark.parametrize("changed", ["revision", "text", "role", "occurred_at"])
def test_receipt_cannot_be_reused_for_changed_evidence(changed):
    records = canonical_evidence(EVIDENCE)
    output = verified_result(normalize_claims([claim("Nisha is allergic to peanuts.")], records),
                             ["supported"], records, model="test")
    altered = [{**EVIDENCE[0], changed: "changed"}]
    with pytest.raises(HindsightError):
        validate_publication(output, altered)


def test_receipt_cannot_cover_added_prose_or_changed_claim():
    records = canonical_evidence(EVIDENCE)
    output = verified_result(normalize_claims([claim("Nisha is allergic to peanuts.")], records),
                             ["supported"], records, model="test")
    with pytest.raises(HindsightError):
        validate_publication({**output, "text": output["text"] + " Severe."}, EVIDENCE)


def test_inference_output_cannot_smuggle_a_verification_attestation():
    records = canonical_evidence(EVIDENCE)
    candidate = {**claim("Nisha weighs 90 kg."), "verification": "supported"}
    with pytest.raises(HindsightError):
        normalize_claims([candidate], records)


@pytest.mark.parametrize("source,text,reason", [
    ("Nisha is allergic to peanuts.", "Nisha has a severe peanut allergy.", "unsupported_severity"),
    ("निशा को मूँगफली से एलर्जी है।", "निशा को गंभीर एलर्जी है।", "unsupported_severity"),
    ("मीरा शायद शुक्रवार को जयपुर जाएगी।", "Meera pakka Friday ko Jaipur jayegi.", "certainty_strengthening"),
    ("Meera might travel on Friday.", "मीरा जरूर शुक्रवार को जाएगी।", "certainty_strengthening"),
    ("Nisha is allergic to peanuts.", "Nisha weighs 90 kg.", "new_numeric_detail"),
])
def test_lexical_safety_veto_overrides_an_incorrect_supported_model_verdict(source, text, reason):
    client, _ = adapter([{"verdicts": [{"claim_index": 0, "label": "supported"}]}])
    output = client.verify_claims([claim(text, quote=source)], evidence=[
        {"record_id": "r1", "text": source}])
    assert output["claims"] == [] and output["text"] == ""
    assert output["verification"]["deterministic_rejections"] == [{"claim_index": 0, "reason": reason}]


def test_cross_language_numeric_and_severity_support_is_not_vetoed():
    records = canonical_evidence([{"record_id": "r1", "text": "९ बजे; गंभीर एलर्जी"}])
    claims = normalize_claims([claim("9 बजे; severe allergy", quote="९ बजे; गंभीर एलर्जी")], records)
    assert deterministic_rejections(claims, records) == []
