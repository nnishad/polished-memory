"""Bounded evidence/claim contracts shared by generation and publication.

Exact quote membership is mechanical validation, not semantic entailment.
Receipts describe a verifier decision, not proof of source truth.
"""
from __future__ import annotations

import json
import re
import unicodedata

from ..backend.hindsight_client import HindsightError
from ..ids import digest

MAX_INPUT_BYTES = 24_000
MAX_CLAIMS = 16
MAX_CLAIM_BYTES = 16_000
VERIFICATION_TOKENS = 1024
SYNTHESIS_CONTRACT = "scoped-evidence-entailment-v4"
VERIFIER_POLICY = "canonical-entailment-v1"
LABELS = ("supported", "contradicted", "insufficient_evidence")


def _contains(text, phrases):
    # Unicode boundaries retain Hindi combining marks (rather than \b).
    text = unicodedata.normalize("NFC", text).casefold()
    for phrase in phrases:
        start = text.find(phrase)
        while start >= 0:
            end = start + len(phrase)
            if (not start or not text[start - 1].isalnum()) and (
                    end == len(text) or not text[end].isalnum()):
                return True
            start = text.find(phrase, start + 1)
    return False


def deterministic_rejections(claims, records):
    """Conservative lexical vetoes, not a replacement for semantic verification.

    Cross-language synonym groups avoid requiring identical English/Hindi words.
    These deliberately incomplete checks can reject ambiguous otherwise-valid
    claims; acceptance still requires the semantic judge.
    """
    uncertain = ("might", "may", "perhaps", "possibly", "maybe", "shayad", "शायद", "संभव")
    certain = ("definitely", "certainly", "guaranteed", "pakka", "पक्का", "निश्चित", "जरूर", "ज़रूर")
    severe = ("severe", "life-threatening", "fatal", "anaphylaxis", "गंभीर", "जानलेवा", "bahut serious")
    def numbers(text):
        text = "".join(str(unicodedata.decimal(char)) if char.isdecimal() else char for char in text)
        return set(re.findall(r"\d+(?:[.,]\d+)*", text))
    rejected = []
    for index, claim in enumerate(claims):
        source = "\n".join(records[ref]["text"] for ref in claim["record_ids"])
        text = claim["text"]
        reason = None
        if numbers(text) - numbers(source):
            reason = "new_numeric_detail"
        elif _contains(text, severe) and not _contains(source, severe):
            reason = "unsupported_severity"
        elif _contains(text, certain) and _contains(source, uncertain) and not _contains(source, certain):
            reason = "certainty_strengthening"
        if reason:
            rejected.append({"claim_index": index, "reason": reason})
    return rejected


def canonical_evidence(evidence):
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 500:
        raise HindsightError("synthesis requires bounded explicit evidence")
    records = {}
    for item in evidence:
        if (not isinstance(item, dict) or not isinstance(item.get("record_id"), str)
                or not 1 <= len(item["record_id"]) <= 200
                or not isinstance(item.get("text"), str)
                or (item.get("revision") is not None
                    and not isinstance(item["revision"], str))):
            raise HindsightError("synthesis input requires record identities and evidence text")
        if item["record_id"] in records:
            raise HindsightError("synthesis input contains duplicate identities")
        # Closed metadata contract: no model-chosen authority or verifier status.
        records[item["record_id"]] = {key: item.get(key) for key in
            ("record_id", "text", "revision", "source", "role", "occurred_at")}
    try:
        encoded = json.dumps(list(records.values()), ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise HindsightError("synthesis evidence metadata is malformed") from error
    if len(encoded.encode()) > MAX_INPUT_BYTES:
        raise HindsightError("synthesis input exceeds its byte bound; narrow the scope")
    return records


def evidence_digest(records):
    return digest([VERIFIER_POLICY, [records[key] for key in sorted(records)]])


def normalize_claims(claims, records):
    """Resolve exact evidence quotes locally; offsets/revisions are never trusted."""
    if not isinstance(claims, list) or not 1 <= len(claims) <= MAX_CLAIMS:
        raise HindsightError("synthesis requires bounded claims")
    normalized = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"text", "record_ids", "evidence"}:
            raise HindsightError("claim requires text, record_ids and exact evidence quotes")
        text, refs, spans = claim["text"], claim["record_ids"], claim["evidence"]
        if (not isinstance(text, str) or not text.strip() or len(text) > 2000
                or not isinstance(refs, list) or not 1 <= len(refs) <= 32
                or any(not isinstance(ref, str) or ref not in records for ref in refs)
                or len(set(refs)) != len(refs)
                or not isinstance(spans, list) or not 1 <= len(spans) <= 32):
            raise HindsightError("claim has malformed or unsupported references")
        resolved = []
        for span in spans:
            if (not isinstance(span, dict) or set(span) != {"record_id", "quote"}
                    or not isinstance(span["record_id"], str) or span["record_id"] not in refs
                    or not isinstance(span["quote"], str) or not span["quote"].strip()):
                raise HindsightError("claim evidence quote is malformed")
            record = records[span["record_id"]]
            start = record["text"].find(span["quote"])
            if start < 0:
                raise HindsightError("claim evidence quote is not present in its canonical record")
            resolved.append({**span, "start": start, "end": start + len(span["quote"]),
                             "revision": record["revision"]})
        if {span["record_id"] for span in resolved} != set(refs):
            raise HindsightError("every claim reference needs an exact evidence quote")
        normalized.append({"text": text.strip(), "record_ids": sorted(refs), "evidence": resolved})
    if (len(json.dumps(normalized, ensure_ascii=False).encode()) > MAX_CLAIM_BYTES
            or len("\n".join(claim["text"] for claim in normalized)) > 8000):
        raise HindsightError("claims exceed their body ceiling")
    return normalized


def verdicts_json(value, count):
    if not isinstance(value, dict) or set(value) != {"verdicts"}:
        raise HindsightError("verifier returned malformed verdicts")
    rows = value["verdicts"]
    if not isinstance(rows, list) or len(rows) != count:
        raise HindsightError("verifier returned incomplete verdicts")
    verdicts = {}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"claim_index", "label"}
                or type(row["claim_index"]) is not int or not 0 <= row["claim_index"] < count
                or row["claim_index"] in verdicts or row["label"] not in LABELS):
            raise HindsightError("verifier returned malformed verdicts")
        verdicts[row["claim_index"]] = row["label"]
    return [verdicts[index] for index in range(count)]


def verified_result(claims, verdicts, records, *, model):
    accepted = [claim for claim, label in zip(claims, verdicts) if label == "supported"]
    text = "\n".join(claim["text"] for claim in accepted)
    return {"text": text, "claims": accepted,
            "record_ids": sorted({ref for claim in accepted for ref in claim["record_ids"]}),
            "verification": {"contract": SYNTHESIS_CONTRACT, "policy": VERIFIER_POLICY,
                "model": model, "evidence_digest": evidence_digest(records),
                "claims_digest": digest(accepted), "text_digest": digest(text),
                "checked": len(claims), "accepted": len(accepted),
                "rejected": [{"claim_index": index, "label": label}
                             for index, label in enumerate(verdicts) if label != "supported"]}}


def validate_publication(outcome, evidence, *, allow_empty=False):
    """No unchecked text, altered rendering or stale-source receipt may publish.

    This validates a trusted adapter's receipt, not a model-provided attestation.
    The caller must separately enforce current visibility and watermark checks.
    """
    records = canonical_evidence(evidence)
    try:
        receipt = outcome["verification"]
        claims = outcome["claims"]
        raw = [{"text": claim["text"], "record_ids": claim["record_ids"],
                "evidence": [{"record_id": span["record_id"], "quote": span["quote"]}
                             for span in claim["evidence"]]} for claim in claims]
        normalized = [] if allow_empty and raw == [] else normalize_claims(raw, records)
        text = "\n".join(claim["text"] for claim in normalized)
        refs = sorted({ref for claim in normalized for ref in claim["record_ids"]})
        if (receipt["contract"] != SYNTHESIS_CONTRACT or receipt["policy"] != VERIFIER_POLICY
                or receipt["evidence_digest"] != evidence_digest(records)
                or receipt["claims_digest"] != digest(normalized)
                or receipt["text_digest"] != digest(text) or outcome["text"] != text
                or claims != normalized or outcome["record_ids"] != refs
                or type(receipt["accepted"]) is not int or receipt["accepted"] != len(claims)
                or type(receipt["checked"]) is not int
                or not max(1, len(claims)) <= receipt["checked"] <= MAX_CLAIMS
                or not isinstance(receipt["model"], str) or not receipt["model"].strip()
                or not isinstance(receipt["rejected"], list)
                or len(receipt["rejected"]) != receipt["checked"] - len(claims)):
            raise ValueError("invalid receipt")
        indices = set()
        for row in receipt["rejected"]:
            if (not isinstance(row, dict) or set(row) != {"claim_index", "label"}
                    or type(row["claim_index"]) is not int
                    or not 0 <= row["claim_index"] < receipt["checked"]
                    or row["claim_index"] in indices
                    or row["label"] not in ("contradicted", "insufficient_evidence")):
                raise ValueError("invalid rejected verdict")
            indices.add(row["claim_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise HindsightError("publication requires unchanged semantically checked claims") from error
    return outcome
