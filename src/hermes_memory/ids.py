"""ID and timestamp helpers.

``digest`` is the one derivation every identifier in the system comes from, so a
record id, a citation, a review digest and an operation id are all reproducible
from their inputs and none of them depends on a process-local hash.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone

__all__ = ["digest", "content_digest", "now", "record_id", "record_pk",
           "backend_document_id", "timestamp", "intervals_overlap", "interval_contains"]

_INVALID_BACKEND_CHARS = re.compile(r"[~_]")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def content_digest(raw: bytes) -> str:
    """Hash of *bytes*. Kept separate from ``digest``: JSON cannot carry raw bytes."""
    if not isinstance(raw, bytes):
        raise TypeError("content_digest takes bytes")
    return hashlib.sha256(raw).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 100:
        raise ValueError("timestamp must be nonempty text, at most 100 characters")
    parsed = datetime.fromisoformat(value.replace("z", "Z").replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def intervals_overlap(a_from: str | None, a_to: str | None, b_from: str | None,
                      b_to: str | None) -> bool:
    """Do two validity intervals share a moment? Open ends are unbounded."""
    if a_from and b_to and a_from > b_to:
        return False
    if b_from and a_to and b_from > a_to:
        return False
    return True


def interval_contains(start: str | None, end: str | None, at: str) -> bool:
    return not ((start and at < start) or (end and at > end))


def record_id(source: str, source_id: str, revision: str) -> str:
    """Stable canonical record ID: rec_<32 hex>.

    The underscore stays because it is what every stored id, citation and tombstone
    in this schema carries. Only backend document IDs avoid the character, because
    Hindsight's own chunk parser cannot reversible-split it (see backend_document_id).
    """
    return "rec_" + digest([source, source_id, revision])[:32]


def record_pk(citation: str) -> str:
    """The record a citation names, ignoring the revision or span it was quoted at.

    A citation is written `rec_…@revision#char` and read as often as it is written,
    so the one place that decides what the record is has to be here rather than in
    each reader's string handling.
    """
    return str(citation).split("@", 1)[0].split("#", 1)[0]


def backend_document_id(record: str, revision: str) -> str:
    """Opaque Hindsight document ID for one canonical revision.

    Hindsight escapes ``_`` and ``~`` when composing chunk IDs, and
    ``parse_chunk_id`` refuses to split an ambiguous ID. Keeping these
    characters out of the document ID makes the mapping reversible, so we
    never have to string-parse a chunk ID to recover a record (§1.3).
    """
    candidate = "hdoc" + digest([record, revision])[:32]
    assert not _INVALID_BACKEND_CHARS.search(candidate)
    return candidate


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"
