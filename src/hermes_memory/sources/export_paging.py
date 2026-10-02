"""Resumable, bounded pages for local exports that expand into many records."""
from __future__ import annotations

import json

from ..ids import digest
from .base import CursorExpired, Page, Skipped

PREFIX = "export-v1:"


def read_export_page(adapter, cursor, keys, read):
    """Read returns (envelopes, gaps). Cursors retain a content-bound row offset.

    A changed file is restarted rather than skipping its new prefix; immutable
    revision ingestion makes that replay safe. Legacy path-only cursors resume
    after their last fully consumed file.
    """
    key, offset, fingerprint = None, 0, None
    if cursor and cursor.startswith(PREFIX):
        try:
            key, offset, fingerprint = json.loads(cursor[len(PREFIX):])
            if not isinstance(key, str) or not isinstance(offset, int) or offset < 0:
                raise ValueError()
        except (ValueError, TypeError):
            raise CursorExpired("invalid export cursor") from None
        remaining = [name for name in keys if name >= key]
    else:
        remaining = [name for name in keys if cursor is None or name > cursor]
    envelopes, skipped, used = [], [], 0
    limit, byte_limit = adapter.capabilities.max_records_per_page, adapter.capabilities.max_bytes_per_page
    for index, name in enumerate(remaining[:limit * 4]):
        produced, gaps = read(name)
        skipped.extend(gaps)
        # Observation time is not revision identity and changes between reads.
        stamp = digest([{k: v for k, v in item.items() if k != "observed_at"} for item in produced])
        start = offset if name == key and stamp == fingerprint else 0
        for position in range(start, len(produced)):
            item = produced[position]
            size = len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
            if size > byte_limit:
                skipped.append(Skipped(str(item["source_id"]), "record exceeds page byte bound"))
                continue
            if len(envelopes) >= limit or used + size > byte_limit:
                resume = PREFIX + json.dumps([name, position, stamp], separators=(",", ":"))
                return Page(envelopes=tuple(envelopes), skipped=tuple(skipped), next_cursor=resume)
            envelopes.append(item)
            used += size
        if len(envelopes) >= limit and index + 1 < len(remaining):
            return Page(envelopes=tuple(envelopes), skipped=tuple(skipped), next_cursor=name)
    examined = min(len(remaining), limit * 4)
    return Page(envelopes=tuple(envelopes), skipped=tuple(skipped),
                next_cursor=remaining[examined - 1] if examined < len(remaining) else None)
