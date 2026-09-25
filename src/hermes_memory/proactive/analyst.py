"""C10 packet analyst: what a model may say about evidence it was handed.

The analyst reads one frozen packet and answers one question: is there anything
here the owner should be told, drafted, or reminded of, and which spans support
it. It is not a second extraction engine and it does not build a fact graph — the
Hindsight backend already does the retrieval, and duplicating it here would create
a second version of the truth with weaker provenance.

Three rules hold it in place. Its citations may only name spans that were in the
packet, so it cannot launder an unsourced claim into a memory. Its outcome may only
move *down* the ladder the attention gate chose, so a confident model cannot promote
its own notification. And what it proposes is never written as a relation or a
record: a hypothesis with no validated span behind it stays a hypothesis.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

from ..storage.evidence import EvidenceError

__all__ = ["PacketAnalyst", "Analysis", "REQUEST_OUTCOMES", "UNAVAILABLE"]

# What the model may ask for. `silent` is an outcome, not an absence: "I looked and
# nothing here needs the owner" has to be distinguishable from "I could not look".
REQUEST_OUTCOMES = ("silent", "next_turn", "digest", "notify_owner", "draft")
LADDER = {"silent": 0, "next_turn": 1, "draft": 1, "digest": 2, "notify_owner": 3}
UNAVAILABLE = "unavailable"
MAX_MESSAGE_CHARS = 900
MAX_CITATIONS = 24


@dataclass(frozen=True)
class Analysis:
    outcome: str
    message: str
    reason: str
    citations: tuple[str, ...] = ()
    source: str = UNAVAILABLE
    dropped: tuple[str, ...] = ()
    gate: str = "silent"

    @property
    def worth_sending(self) -> bool:
        return self.outcome != "silent" and bool(self.message)

    def as_dict(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "message": self.message, "reason": self.reason,
                "citations": list(self.citations), "source": self.source,
                "dropped": list(self.dropped), "gate": self.gate}


class PacketAnalyst:
    """The only place in C10 that reads evidence bodies, and it reads a frozen set."""

    def __init__(self, store, *, model: Any = None, timeout_s: float = 6.0,
                 max_message_chars: int = MAX_MESSAGE_CHARS):
        self.store = store
        self.model = model
        if not isinstance(timeout_s, (int, float)) or not 0.1 <= timeout_s <= 30:
            raise EvidenceError("the analyst's timeout must be between 0.1 and 30 seconds")
        if not isinstance(max_message_chars, int) or not 80 <= max_message_chars <= 4000:
            raise EvidenceError("max_message_chars must be between 80 and 4000")
        self.timeout_s = float(timeout_s)
        self.max_message_chars = max_message_chars
        self._pool: ThreadPoolExecutor | None = None

    # -- the call ------------------------------------------------------------

    def analyse(self, *, packet: Any, topic: str, gate: str,
                question: str | None = None, spans: Sequence[str] | None = None) -> Analysis:
        """Ask for one outcome, and demote whatever comes back above the gate."""
        if gate not in REQUEST_OUTCOMES:
            raise EvidenceError(f"the gate offered {gate!r}, which is not an outcome")
        allowed = _span_ids(packet, spans)
        if self.model is None:
            # No inference is a supported operating state, not an error: report it
            # as a distinct outcome so status can say "not analysed" out loud.
            return Analysis("silent", "", "no packet analyst is configured, so nothing "
                                          "was interpreted", source=UNAVAILABLE,
                            gate=gate)
        if not allowed:
            return Analysis("silent", "", "the packet carried no spans to cite, so any "
                                          "claim would be unsourced", source="refused",
                            gate=gate)
        request = {"topic": topic, "gate": gate, "question": question or
                   "Is there anything here the owner should be told, drafted, or reminded of?",
                   "spans": allowed, "packet": _freeze(packet)}
        try:
            raw = self._call(request)
        except TimeoutError:
            return Analysis("silent", "", f"the analyst did not answer within "
                                          f"{self.timeout_s:g}s", source="timeout", gate=gate)
        except Exception as error:
            return Analysis("silent", "", f"the analyst failed: {str(error)[:180]}",
                            source="failed", gate=gate)
        return self._decode(raw, allowed=allowed, gate=gate)

    def _call(self, request: dict[str, Any]) -> Any:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analyst")
        # A hung upstream must not hold the proactive worker: the same reason the
        # broker bounds its derived channel.
        future = self._pool.submit(self.model.analyse, request)
        try:
            return future.result(timeout=self.timeout_s)
        except TimeoutError:
            future.cancel()
            raise

    # -- the guardrail -------------------------------------------------------

    def _decode(self, raw: Any, *, allowed: Sequence[str], gate: str) -> Analysis:
        payload = raw
        if isinstance(raw, (str, bytes)):
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return Analysis("silent", "", "the analyst did not return decodable JSON, "
                                              "so nothing it meant can be relied on",
                                source="malformed", gate=gate)
        if not isinstance(payload, dict):
            return Analysis("silent", "", "the analyst returned a shape that is not an "
                                          "outcome", source="malformed", gate=gate)
        outcome = str(payload.get("outcome") or "silent")
        if outcome not in REQUEST_OUTCOMES:
            return Analysis("silent", "", f"the analyst asked for {outcome!r}, which is not "
                                          "an outcome this system can honour",
                            source="refused", gate=gate)
        dropped: list[str] = []
        if LADDER[outcome] > LADDER[gate]:
            dropped.append(f"{outcome} is above the gate's {gate}; a model may not promote "
                           "its own interruption")
            outcome = gate
        message = _clean(payload.get("message"), self.max_message_chars)
        citations = []
        for citation in list(payload.get("citations") or [])[:MAX_CITATIONS]:
            text = str(citation)
            if text in allowed:
                citations.append(text)
            else:
                dropped.append(f"{text[:60]} was not in the supplied packet")
        if outcome != "silent" and not citations:
            dropped.append("every citation was outside the packet, so the message has no "
                           "support to stand on")
            outcome = "silent"
        if outcome != "silent" and not message:
            dropped.append("the message was empty or only control characters")
            outcome = "silent"
        relations = payload.get("relations") or []
        if relations:
            # Never stored, in any table. A model-proposed edge is not a validated
            # assertion, and a knowledge graph full of confident guesses is the
            # failure mode this whole layer exists to avoid.
            dropped.append(f"{len(relations)} proposed relation(s) discarded: the analyst "
                           "does not write knowledge")
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            # An answer with no reason is not an audit trail. Say what was dropped
            # instead, so the silence is still explained by something.
            reason = dropped[0] if dropped else "the analyst gave no reason for its answer"
        return Analysis(outcome, message, reason[:400], citations=tuple(citations),
                        source="analysed", dropped=tuple(dropped), gate=gate)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None


def _span_ids(packet: Any, spans: Sequence[str] | None) -> list[str]:
    """The citation universe: the packet's own spans, and nothing else.

    A derived item cites the backend documents it was read out of, so those count as
    spans too; anything else the model names was not in what it was shown.
    """
    if spans is not None:
        return [str(span) for span in spans]
    found: list[str] = []
    for item in getattr(packet, "items", ()) or ():
        for candidate in (getattr(item, "id", None), *(getattr(item, "provenance", ()) or ())):
            text = str(candidate or "")
            if text and text not in found:
                found.append(text)
    return found


def _freeze(packet: Any) -> list[dict[str, Any]]:
    """A copy, not the live packet: nothing the analyst reads may change under it."""
    frozen = []
    for item in getattr(packet, "items", ()) or ():
        as_dict = getattr(item, "as_dict", None)
        if callable(as_dict):
            frozen.append(json.loads(json.dumps(as_dict(), ensure_ascii=False, default=str)))
        else:
            frozen.append({"span": str(getattr(item, "record_id", item))})
    return frozen


def _clean(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    # Control characters are dropped rather than escaped: this string is rendered
    # straight into a host message, and a terminal escape from a model is a prompt
    # injection with a nicer name.
    text = "".join(char for char in value
                   if char in "\n\t" or (0x20 <= ord(char) and ord(char) != 0x7F))
    return " ".join(text.split())[:limit].rstrip()
