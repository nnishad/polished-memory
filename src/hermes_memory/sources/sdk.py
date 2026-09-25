"""C2 Hermes event adapter: the host's own capture stream, normalised.

The events arrive through a *drain* handed to the constructor rather than an import.
The spool belongs to the plugin, which lives outside this package and needs the host
process to import at all; a connector that could only run inside Hermes could not be
tested outside it, and reaching into the plugin's database file from here would make
one component's durability boundary another's private schema.

Identity is the host's event id, never the text. Two events carrying the same
sentence are two events — the host retried a turn, or the owner pasted one paragraph
into two conversations. Each record carries a ``text_digest`` of its own body and an
``independent`` flag saying whether it can count as a second witness to what it
says: a model's answer and a mirrored native note both describe the exchange, and
neither is a fresh observation of the world, however original it reads.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping

from ..ids import digest, now
from ..storage.evidence import EvidenceError
from .base import (Capabilities, CursorExpired, Page, Skipped, SourceAdapter,
                   normalize_text, normalize_time, redact_secrets)

__all__ = ["HermesEvents", "MAX_EVENT_CHARS"]

MAX_EVENT_CHARS = 50_000
_MAX_ID = 200
_MAX_METADATA_FIELDS = 10
_MAX_METADATA_CHARS = 200
_NATIVE_ACTIONS = frozenset({"add", "replace", "remove"})
_ROLES = frozenset({"user", "assistant", "system", "tool"})
# Only what a participant said can corroborate anything. Everything else here is a
# report about the conversation, and reporting it twice must not look like agreement.
_INDEPENDENT_ROLES = frozenset({"user"})
_ORIGIN = {"user": "owner-statement", "assistant": "model-output", "system": "host-instruction",
           "tool": "tool-output", "note": "mirrored-native-note"}
_SECRET_FIELD = re.compile(
    r"(api[_-]?key|token|secret|password|passwd|credential|authorization|cookie|bearer|access[_-]?key)",
    re.IGNORECASE)


class HermesEvents(SourceAdapter):
    """``drain(after, limit) -> events`` as a connector-readable source.

    Each event is ``{"event_id", "session_id", "created_at", "payload": {...}}`` —
    the shape the plugin's durable spool accepts. A cursor is an opaque position the
    stream itself orders by; this adapter never compares two of them.
    """

    source = "hermes"
    capabilities = Capabilities(history=True, live=True, deletion_events=False,
                                revision_history=False, max_records_per_page=200,
                                max_bytes_per_page=2_000_000)

    def __init__(self, drain: Callable[[str | None, int], Iterable[Mapping[str, Any]]],
                 *, source: str = "hermes", records_per_page: int | None = None):
        if not callable(drain):
            raise EvidenceError(
                "the event adapter takes a drain callable of (after, limit); the spool "
                "is handed over, not opened here")
        if records_per_page is not None and not 1 <= records_per_page <= 1000:
            raise ValueError("records_per_page must be between 1 and 1000")
        self._drain = drain
        self.source = source
        if records_per_page:
            self.capabilities = replace(self.capabilities,
                                        max_records_per_page=records_per_page)

    # -- contract ------------------------------------------------------------

    def check(self) -> dict[str, Any]:
        """Probe the pipe with one event and report its age.

        An identifier and an arrival time describe the connection; the text inside
        an event is content, and content is what ``read_page`` is for.
        """
        try:
            batch = self._take(None, 1)
        except CursorExpired:
            raise
        except EvidenceError as error:
            return {"ok": False, "reason": str(error)[:500], "content_read": False}
        oldest = batch[0] if batch else None
        return {"ok": True, "live": True, "pending": bool(batch),
                "oldest_at": _arrival_of(oldest) if oldest is not None else None,
                "content_read": False, "observed_at": now()}

    def read_page(self, cursor: str | None) -> Page:
        limit = self.capabilities.max_records_per_page
        budget = self.capabilities.max_bytes_per_page
        batch = self._take(cursor, limit)
        envelopes: list[dict[str, Any]] = []
        skipped: list[Skipped] = []
        examined: list[str] = []
        seen: set[str] = set()
        used = 0
        for event in batch:
            if len(examined) >= limit * 4:
                break
            ref, produced, reason = self._one(event, seen)
            if produced is None:
                # Every examined event advances the position, including one that
                # yielded nothing, so a gap is passed over once rather than retried
                # for as long as the connector lives.
                examined.append(ref)
                skipped.append(Skipped(ref, reason))
                continue
            size = sum(len(item["text"].encode("utf-8")) for item in produced)
            if envelopes and (len(envelopes) + len(produced) > limit
                              or used + size > budget):
                # Stop before taking it, so the cursor stays behind this event and
                # the next page reads it again rather than losing it. One event is
                # never split across pages, even so: a turn divided between two
                # commits can leave a question durable and its answer not, and the
                # first event on a page is taken whatever it holds.
                break
            used += size
            examined.append(ref)
            envelopes.extend(produced)
        # A live stream reports where it stopped whenever it examined anything,
        # including the page that ran to the tail: the empty read that proves the
        # tail was reached is what ends the pass, and it is the one that costs a
        # poll rather than a re-read of everything before it.
        return Page(envelopes=tuple(envelopes), skipped=tuple(skipped),
                    next_cursor=examined[-1] if examined else None)

    # -- one event -----------------------------------------------------------

    def _one(self, event: Any, seen: set[str]) -> tuple[str, list[dict[str, Any]] | None, str]:
        if not isinstance(event, Mapping):
            return _unaddressed(event), None, "the stream yielded an event that is not an object"
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            # An event we cannot name cannot be replayed without duplicating it, and
            # a position in the stream is not an identity: the host may compact the
            # spool and shift every position under us.
            return _unaddressed(event), None, "the event carried no event_id"
        if len(event_id) > _MAX_ID:
            return event_id[:_MAX_ID], None, f"the event id is longer than {_MAX_ID} characters"
        if event_id in seen:
            return event_id, None, "the stream reported this event twice in one page"
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            seen.add(event_id)
            return event_id, None, "the event carried no payload object"
        kind = str(payload.get("kind") or "").strip()
        handler = _HANDLERS.get(kind)
        if handler is None:
            seen.add(event_id)
            # Reported under its own event id, so that a later release which learns
            # this kind closes the gap by delivering the event rather than by
            # somebody remembering to look.
            return event_id, None, f"unsupported event kind {kind or '(blank)'}"
        seen.add(event_id)
        produced = handler(self, event, payload, event_id)
        if isinstance(produced, str):
            return event_id, None, produced
        return event_id, list(produced), ""

    def _frame(self, event: Mapping[str, Any], payload: Mapping[str, Any],
               event_id: str) -> _Frame:
        """The metadata every record from this event shares, assembled once.

        Every value here is a fact about the *event*, never about this read: a source
        that re-offers a page under a new name must land on the same fingerprint, or
        the store correctly refuses it as the same revision with different bytes. How
        stale an arrival is gets decided downstream from ``spooled_at`` against
        ``observed_at``, both of which are kept.
        """
        occurred, precision, note = normalize_time(
            payload.get("occurred_at") or event.get("timestamp"))
        author = _author(payload.get("author") if "author" in payload else event.get("author"))
        return _Frame(
            occurred=occurred, precision=precision,
            metadata={
                "event_id": event_id,
                "session_id": _short(event.get("session_id")) or None,
                "event_kind": str(payload.get("kind") or ""),
                "spooled_at": _arrival_of(event),
                "author": author or None,
                "time_basis": "event field" if occurred else "none",
                **({"time_note": note} if note else {}),
            },
        )

    def _stamp(self, frame: _Frame, *, role: str, text: str, kind: str,
               event_id: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        body = text[:MAX_EVENT_CHARS]
        return self.envelope(
            source_id=event_id, revision="1", kind=kind, text=body, observed_at=now(),
            occurred_at=frame.occurred, occurred_precision=frame.precision,
            metadata={
                **frame.metadata,
                "role": role, "origin": _ORIGIN.get(role, role),
                "independent": role in _INDEPENDENT_ROLES,
                "text_digest": digest(["v1", body])[:32],
                "chars": len(body),
                **({"truncated": True} if len(text) > MAX_EVENT_CHARS else {}),
                **(dict(extra) if extra else {}),
            },
        )

    def _take(self, after: str | None, limit: int) -> list[Any]:
        """Ask the stream for events, and refuse an answer that is not events."""
        try:
            batch = self._drain(after, limit)
        except (CursorExpired, EvidenceError):
            raise
        except Exception as error:
            raise EvidenceError(
                f"the event stream did not answer: {_reason(error)}") from None
        if batch is None or isinstance(batch, (str, bytes, Mapping)):
            raise EvidenceError("the drain must yield a sequence of event objects")
        try:
            return list(batch)
        except TypeError:
            raise EvidenceError("the drain must yield a sequence of event objects") from None


@dataclass(frozen=True)
class _Frame:
    """What one event contributes to every record taken from it."""

    metadata: dict[str, Any]
    occurred: str | None
    precision: str


def _turn(adapter: HermesEvents, event: Mapping, payload: Mapping,
          event_id: str) -> list[dict[str, Any]] | str:
    """One turn, two records.

    The spool stores a turn as one event and the evidence is two claims: that a
    participant said something, and that the model answered this way. In one row,
    'the user said X' and 'the assistant said X' collapse into 'X', and that is
    precisely the distinction a memory has to keep.
    """
    frame = adapter._frame(event, payload, event_id)
    produced = []
    for role, field in (("user", "user"), ("assistant", "assistant")):
        body = normalize_text(payload.get(field))
        if body is None:
            continue
        produced.append(adapter._stamp(frame, role=role, text=body, kind="conversation_turn",
                                       event_id=f"{event_id}#{role}"))
    if not produced:
        return "the turn carried no text on either side"
    return produced


def _transcript(adapter: HermesEvents, event: Mapping, payload: Mapping,
                event_id: str) -> list[dict[str, Any]] | str:
    """A message checkpointed before the host compacted the conversation away."""
    role = normalize_text(payload.get("role"))
    if role not in _ROLES:
        return f"the pre-compress message claimed role {role!r}"
    body = normalize_text(payload.get("text"))
    if body is None:
        return "the pre-compress message had no text that could be decoded"
    return [adapter._stamp(adapter._frame(event, payload, event_id), role=role, text=body,
                           kind="transcript", event_id=event_id)]


def _native_note(adapter: HermesEvents, event: Mapping, payload: Mapping,
                 event_id: str) -> list[dict[str, Any]] | str:
    """A mirror of a native Hermes note.

    Mirroring is not replacing: the note's own store stays authoritative for it, and
    removing a note here does not forget the evidence behind it — so a removal is
    filed as a report of a removal, which is all this adapter can honestly see.
    """
    action = normalize_text(payload.get("action"))
    if action not in _NATIVE_ACTIONS:
        return f"unsupported native memory action {action!r}"
    content = normalize_text(payload.get("content"))
    if content is None:
        return "the native note carried no text"
    target = normalize_text(payload.get("target")) or "unknown"
    return [adapter._stamp(adapter._frame(event, payload, event_id), role="note", text=content,
                           kind="note", event_id=event_id,
                           extra={"native_action": action, "native_target": target})]


_HANDLERS = {"conversation_turn": _turn, "pre_compress": _transcript,
             "native_memory_write": _native_note}


# -- normalization helpers ---------------------------------------------------

def _author(value: Any) -> dict[str, Any]:
    """Attribution, filtered.

    The host hands over a dictionary of its own choosing. A session context that
    happens to carry a route's bearer token must not turn into searchable text in
    the archive, so secret-shaped keys and nested structures do not pass, and what
    was dropped is counted rather than quietly omitted.
    """
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    dropped = 0
    for key, item in value.items():
        name = str(key)
        if _SECRET_FIELD.search(name) or not isinstance(item, (str, int, float, bool)):
            dropped += 1
            continue
        if len(out) >= _MAX_METADATA_FIELDS:
            dropped += 1
            continue
        text = item if isinstance(item, str) else str(item)
        out[name[:60]] = redact_secrets(text)[:_MAX_METADATA_CHARS]
    if dropped:
        out["redacted_fields"] = dropped
    return out


def _short(value: Any) -> str | None:
    if not isinstance(value, (str, int)):
        return None
    text = str(value).strip()
    return text[:_MAX_ID] or None


def _arrival_of(event: Any) -> str | None:
    """When the host accepted the event, if it said so as an absolute instant."""
    if not isinstance(event, Mapping):
        return None
    value = event.get("created_at") or event.get("spooled_at")
    arrival, precision, _ = normalize_time(value if isinstance(value, str) else None)
    return arrival if precision in {"second", "minute"} else None


def _unaddressed(event: Any) -> str:
    return "<unaddressed>:" + digest([repr(event)])[:16]


def _reason(error: BaseException) -> str:
    return " ".join(str(error).split())[:400] or type(error).__name__
