"""ID and timestamp helpers.

``digest`` is byte-for-byte compatible with the retired framework so that
canonical record IDs computed from the same (source, source_id, revision)
triple continue to resolve across a migration.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone

__all__ = ["digest", "content_digest", "now", "record_id", "backend_document_id",
           "timestamp", "intervals_overlap", "interval_contains"]

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

    Canonical IDs intentionally keep the legacy underscore prefix; only
    backend document IDs are underscore-free (see backend_document_id).
    """
    return "rec_" + digest([source, source_id, revision])[:32]


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
