"""F1: the pre-send verification boundary.

The last hop was the unchecked one: summaries were verified at publication and
`memory_verify` existed for an agent willing to call it, but nothing stopped a
final conversational answer from asserting personal facts straight from prose.
This module is the door a draft must pass through before it leaves.

Three properties are deliberate.

* **Cheap first.** A deterministic detector decides whether a draft contains
  personal-memory assertions at all; a message with none costs no model call and
  passes with a receipt saying why it was skipped.
* **Evidence is what the agent was given.** Claims are checked against the
  packet injected into this turn, never against records the draft merely
  resembles. The agent may assert what it was handed; reaching beyond that is
  `insufficient_evidence`, not a guess the boundary blesses.
* **Degradation is reported, never silent.** A verifier that cannot be reached
  in `enforce` mode holds the send with a reason; in `warn` mode it records the
  degradation and lets the message through, because warn exists to measure the
  boundary's false-positive rate before anyone trusts it to block.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..backend.hindsight_client import HindsightError
from ..ids import digest, now
from .faithfulness import MAX_CLAIMS
from .verification import verify_memory_claims

__all__ = ["SendBoundary", "BoundaryVerdict", "BOUNDARY_VERSION", "MODES",
           "memory_sentences"]

BOUNDARY_VERSION = "send-boundary-v1"
MODES = ("off", "warn", "enforce")

# How many packet records one draft may cite as its evidence: the whole injected
# packet, capped, because "what you were told this turn" is the honest bound.
_MAX_EVIDENCE_RECORDS = 8
# Sentences checked per draft; the verifier's own claim ceiling is the hard stop.
_MAX_SENTENCES = MAX_CLAIMS
# Significant-token overlap that ties a sentence to a packet record without a
# personal pronoun in it ("peanut allergy is severe" names no one).
_MIN_SHARED_TOKENS = 2
_STOPWORDS = frozenset({"the", "and", "for", "with", "that", "this", "have", "has",
                        "was", "were", "will", "would", "could", "should", "about",
                        "from", "into", "your", "you", "they", "them", "their"})

# Second-person or remembered-fact assertions: the shape a personal-memory claim
# takes. Conservative on purpose — over-matching costs one verifier call,
# under-matching ships an unchecked claim about a person.
_PERSONAL = re.compile(
    r"\b(you|your|yours|you're|you are|you have|you've|you had|you will|you'll|"
    r"you like|you prefer|you said|you told|you mentioned|we discussed|"
    r"I remember|I recall|you remember|last time|previously|as you)\b", re.IGNORECASE)
_SUBJECT = re.compile(
    r"\b(allerg|allergic|allergy|medication|medicine|doctor|appointment|pregnan|"
    r"blood|pressure|weight|weigh|diabet|asthma|diet|vegan|vegetarian|prefer|"
    r"dislike|hate|love|birthday|anniversary|married|wife|husband|partner|son|"
    r"daughter|mother|father|boss|job|salary|rent|mortgage|flight|train|trip|"
    r"travel|visa|passport|password|pin|account|address|phone)\b", re.IGNORECASE)
# Hindi/Hinglish seeds so the boundary does not go quiet in the languages the
# archive actually speaks.
_PERSONAL_HI = re.compile(
    r"(आप|तुम|तेरा|तेरी|आपका|आपकी|याद|पसंद|नापसंद|एलर्जी|दवा|शादी|बेटा|बेटी|माँ|पिता)",
    re.IGNORECASE)
# Remembered-fact phrasing qualifies on its own: "I recall you said …" is a memory
# claim whatever its subject, and must not slip past for lacking a keyword.
_REMEMBERED = re.compile(
    r"\b(I remember|I recall|you said|you told me|you mentioned|we agreed|last time|"
    r"previously|as before|the usual)\b", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zऀ-ॿ]+", (text or "").casefold())
            if len(token) > 3 and token not in _STOPWORDS}


def _sentences(draft: str) -> list[str]:
    return [part.strip(" \t-•*") for part in re.split(r"(?<=[.!?।])\s+|\n+", draft or "")
            if part.strip()]


def memory_sentences(draft: str, packet_texts: Iterable[str] = ()) -> list[str]:
    """The draft's personal-memory assertions, deterministically, no model.

    A sentence qualifies by second-person/remembered-fact phrasing, by naming a
    personal subject (health, preference, relationship, schedule, credential),
    or by sharing enough significant tokens with something the packet injected.
    """
    pool = [_tokens(text) for text in packet_texts]
    found: list[str] = []
    for sentence in _sentences(draft):
        if not sentence or len(sentence) < 12:
            continue
        tokens = _tokens(sentence)
        personal = bool(_PERSONAL.search(sentence) or _PERSONAL_HI.search(sentence))
        subject = bool(_SUBJECT.search(sentence))
        echoed = any(len(tokens & shared) >= _MIN_SHARED_TOKENS for shared in pool)
        # A personal-subject assertion is a memory claim even in the third person and
        # even with nothing injected to support it: asserting beyond the packet is
        # exactly what the boundary exists to catch, so it proceeds as unverifiable
        # rather than being skipped.
        if _REMEMBERED.search(sentence) or subject or (personal and echoed):
            found.append(sentence)
        if len(found) >= _MAX_SENTENCES:
            break
    return found


@dataclass(frozen=True)
class BoundaryVerdict:
    """What the boundary decided, and the receipt proving it decided."""

    disposition: str            # pass | revise | block
    mode: str                   # off | warn | enforce
    claims: tuple[dict[str, Any], ...] = ()
    detector: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    receipt: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"disposition": self.disposition, "mode": self.mode,
                "claims": [dict(claim) for claim in self.claims],
                "detector": dict(self.detector), "action": dict(self.action),
                "receipt": dict(self.receipt)}


class SendBoundary:
    """Check a draft's personal-memory claims before the draft is sent."""

    def __init__(self, *, settings, store, mode: str | None = None,
                 account_id: str | None = None):
        self.settings = settings
        self.store = store
        self.mode = mode or getattr(settings, "boundary_mode", "warn")
        if self.mode not in MODES:
            raise HindsightError(f"unknown boundary mode {self.mode!r}")
        self.account_id = account_id

    def check(self, draft: str, *, packet=None, session_id: str | None = None) -> BoundaryVerdict:
        if self.mode == "off":
            return BoundaryVerdict(disposition="pass", mode=self.mode,
                                   detector={"memory_shaped": False,
                                             "skipped_reason": "boundary off"})
        if not (draft or "").strip():
            return BoundaryVerdict(disposition="pass", mode=self.mode,
                                   detector={"memory_shaped": False,
                                             "skipped_reason": "empty draft"})

        packet_texts = [item.text for item in (packet.items if packet else ())]
        sentences = memory_sentences(draft, packet_texts)
        if not sentences:
            # The common case: nothing personal in the draft. No model call, no
            # receipt row — a log of every non-memory message is noise, not audit.
            return BoundaryVerdict(disposition="pass", mode=self.mode,
                                   detector={"memory_shaped": False,
                                             "skipped_reason": "no_memory_claims"})

        # Evidence is what this turn injected: the lexical spans plus the canonical
        # records each derived fact was locally resolved to (the broker stamps
        # fact["record_ids"] only for provenance that crossed its own boundary).
        record_ids = [item.id for item in (packet.items if packet else ())]
        for fact in (getattr(packet, "facts", ()) if packet else ()) or ():
            record_ids.extend(str(rid) for rid in (fact.get("record_ids") or ()))
        record_ids = list(dict.fromkeys(record_ids))[:_MAX_EVIDENCE_RECORDS]
        texts: dict[str, str] = {}
        for rid in record_ids:
            record = self.store.get(rid)
            # An agent note is not admissible backing: drop it from the evidence set
            # rather than letting it poison a claim the canonical records do support.
            if record is None or record.metadata.get("agent_authored"):
                continue
            texts[rid] = record.text
        # Cite only what is admissible: verification resolves the claim's record_ids
        # itself, so an agent note left in the list would refuse the whole call.
        record_ids = list(texts)
        evidence = [{"record_id": rid, "quote": text} for rid, text in texts.items()]
        claims = [{"text": sentence, "record_ids": list(record_ids),
                   "evidence": list(evidence)} for sentence in sentences]
        if not evidence:
            # A memory-shaped draft with nothing injected to support it: the agent
            # is asserting beyond what it was given. That is insufficient evidence,
            # decided without a model, because there is literally nothing to cite.
            labels = ["insufficient_evidence"] * len(claims)
            outcome = None
        else:
            try:
                outcome = verify_memory_claims(self.settings, self.store, claims,
                                               account_id=self.account_id)
            except HindsightError as error:
                return self._degraded(str(error), session_id, packet, sentences)
            rejected = {item["claim_index"]: item["label"]
                        for item in outcome["verification"]["rejected"]}
            vetoed = {item["claim_index"] for item in
                      outcome["verification"].get("deterministic_rejections", [])}
            labels = [rejected.get(index, "supported") for index in range(len(claims))]
            for index in vetoed:
                if index < len(labels):
                    labels[index] = "insufficient_evidence"

        detail = tuple({"text": claim["text"], "label": label}
                       for claim, label in zip(claims, labels))
        if any(label == "contradicted" for label in labels):
            disposition = "block"
        elif any(label != "supported" for label in labels):
            disposition = "revise"
        else:
            disposition = "pass"
        return self._verdict(disposition, {"memory_shaped": True,
                                           "sentences": len(sentences),
                                           "evidence_records": len(evidence)},
                             detail, session_id, packet, outcome=outcome)

    # -- dispositions --------------------------------------------------------

    def _degraded(self, reason: str, session_id, packet, sentences) -> BoundaryVerdict:
        """The verifier could not be reached. Enforce holds; warn says so and sends."""
        if self.mode == "enforce":
            action = {"mode": "hold",
                      "reason": f"verification unavailable; send held: {reason[:200]}"}
            disposition = "block"
        else:
            action = {"mode": "none",
                      "notes": [f"boundary degraded to warn: {reason[:200]}"]}
            disposition = "revise"
        return self._verdict(disposition, {"memory_shaped": True,
                                           "sentences": len(sentences),
                                           "degraded": True},
                             (), session_id, packet, action=action)

    def _verdict(self, disposition, detector, claims, session_id, packet,
                 outcome=None, action=None) -> BoundaryVerdict:
        if action is None:
            action = self._action_for(disposition, claims)
        receipt_id = digest([BOUNDARY_VERSION, self.mode, disposition,
                             session_id or "", packet.packet_id if packet else "",
                             json.dumps(detector, sort_keys=True),
                             json.dumps(list(claims), sort_keys=True), now()])[:24]
        watermark = self.store.watermark()
        self._record(receipt_id, session_id=session_id,
                     packet_id=packet.packet_id if packet else "",
                     watermark=f"{watermark[0]}:{watermark[1]}",
                     disposition=disposition, claims=claims,
                     digests={"boundary": BOUNDARY_VERSION,
                              "verification": (outcome or {}).get("verification", {})
                              .get("claims_digest")})
        return BoundaryVerdict(disposition=disposition, mode=self.mode, claims=claims,
                               detector=detector, action=action,
                               receipt={"id": "bnd_" + receipt_id,
                                        "boundary_version": BOUNDARY_VERSION,
                                        "watermark": f"{watermark[0]}:{watermark[1]}",
                                        "checked_at": now()})

    def _action_for(self, disposition: str, claims) -> dict[str, Any]:
        if self.mode != "enforce" or disposition == "pass":
            return {"mode": "none"}
        if disposition == "block":
            return {"mode": "hold",
                    "reason": "the draft contradicts canonical evidence or trips a "
                              "deterministic veto; correct it before sending"}
        unverified = [claim["text"] for claim in claims if claim.get("label") != "supported"]
        return {"mode": "downgrade",
                "suggested_prefix": "I recall, but can't confirm from what's stored: ",
                "unverified_sentences": unverified,
                "note": f"unsupported: {'; '.join(unverified)[:200]}"}

    def _record(self, receipt_id, *, session_id, packet_id, watermark, disposition,
                claims, digests) -> None:
        self.store.db.execute(
            "INSERT INTO boundary_receipts(id, session_id, packet_id, watermark, mode, "
            "disposition, claims_json, digests, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (receipt_id, session_id, packet_id, watermark, self.mode, disposition,
             json.dumps(list(claims), ensure_ascii=False),
             json.dumps(digests, sort_keys=True), now()))
