"""C2 file adapter: text, Markdown and JSON exports as bounded evidence."""
from __future__ import annotations

import json
import posixpath
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..ids import content_digest, digest, now
from .base import Capabilities, Page, Skipped, SourceAdapter, normalize_time

__all__ = ["FileSource"]

_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
_JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}
_MAX_FILE_BYTES = 8_000_000


class FileSource(SourceAdapter):
    """Reads a directory tree of exports.

    The revision is the content digest, so an edited file becomes a new
    revision of the same source_id rather than a brand-new fact, and an
    unchanged file replays as a duplicate. Ordering and cursor comparison both
    use the same normalized path key, so replaying one tree always produces the
    same page boundaries.
    """

    capabilities = Capabilities(history=True, live=False, deletion_events=False,
                                revision_history=True, max_records_per_page=50)

    def __init__(self, root: str | Path, *, source: str = "files",
                 records_per_page: int | None = None):
        self.root = Path(root).expanduser().resolve()
        self.source = source
        if records_per_page:
            if not 1 <= records_per_page <= 1000:
                raise ValueError("records_per_page must be between 1 and 1000")
            self.capabilities = replace(self.capabilities,
                                        max_records_per_page=records_per_page)

    def check(self) -> dict[str, Any]:
        if not self.root.is_dir():
            return {"ok": False, "reason": f"{self.root} is not a directory",
                    "content_read": False}
        return {"ok": True, "root": str(self.root), "candidates": len(list(self._keys())),
                "content_read": False, "observed_at": now()}

    def read_page(self, cursor: str | None) -> Page:
        from .export_paging import read_export_page
        def read(key):
            path = self.root / key
            size = self._size_of(path)
            if size is None:
                return [], [Skipped(key, "stat failed or the file disappeared")]
            if size > _MAX_FILE_BYTES:
                return [], [Skipped(key, f"{size} bytes exceeds the {_MAX_FILE_BYTES} byte per-file bound")]
            produced, reason = self._read(path, key)
            return (produced, []) if produced is not None else ([], [Skipped(key, reason)])
        return read_export_page(self, cursor, self._keys(), read)

    def _size_of(self, path: Path) -> int | None:
        try:
            return path.stat().st_size
        except OSError:
            return None


    def _keys(self) -> list[str]:
        if not self.root.is_dir():
            return []
        found = []
        for path in self.root.rglob("*"):
            if path.is_symlink():
                # A link out of the export directory would ingest a file the
                # operator never pointed the connector at.
                continue
            if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES | _JSON_SUFFIXES:
                continue
            if path.name.startswith("."):
                continue
            found.append(posixpath.normpath(path.relative_to(self.root).as_posix()))
        return sorted(found)

    # -- normalization -------------------------------------------------------

    def _read(self, path: Path, key: str):
        try:
            raw = path.read_bytes()
        except OSError as error:
            return None, f"read failed: {error.strerror or error}"
        if not raw.strip():
            return None, "file is empty"
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, "not valid UTF-8; refusing to substitute replacement characters"
        try:
            if path.suffix.lower() in _JSON_SUFFIXES:
                return self._structured(key, text, path, raw), None
            return [self._whole_file(key, text, path, raw)], None
        except (json.JSONDecodeError, ValueError) as error:
            return None, f"malformed export: {error}"

    def _whole_file(self, key: str, text: str, path: Path, raw: bytes) -> dict[str, Any]:
        occurred, precision, note = normalize_time(_front_matter_date(text))
        return self.envelope(
            source_id=key, revision=_revision(key, raw), kind="file", text=text,
            observed_at=now(), occurred_at=occurred, occurred_precision=precision,
            metadata={"path": key, "bytes": len(raw),
                      "media_type": _media_type(path.suffix),
                      **({"time_note": note} if note else {})},
        )

    def _structured(self, key: str, text: str, path: Path, raw: bytes) -> list[dict[str, Any]]:
        payload = (_jsonl(text) if path.suffix.lower() in {".jsonl", ".ndjson"}
                   else json.loads(text))
        revision = _revision(key, raw)
        envelopes = []
        for index, item in enumerate(_flatten(payload)):
            if not isinstance(item, dict):
                continue
            body = next((item[name] for name in ("text", "body", "content")
                         if isinstance(item.get(name), str) and item[name].strip()), None)
            if body is None:
                continue
            occurred, precision, note = normalize_time(
                item.get("time") or item.get("timestamp") or item.get("date"))
            envelopes.append(self.envelope(
                source_id=str(item.get("id") or f"{key}#{index}"), revision=revision,
                kind=str(item.get("kind") or "file_record"), text=body, observed_at=now(),
                occurred_at=occurred, occurred_precision=precision,
                metadata={"path": key, "index": index, "bytes": len(raw),
                          "media_type": "application/json",
                          "time_basis": str(item.get("time_basis") or "source field"),
                          **({"time_note": note} if note else {})},
            ))
        return envelopes


def _revision(key: str, raw: bytes) -> str:
    return digest(["file", key, content_digest(raw)])[:32]


def _jsonl(text: str) -> list:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _flatten(payload: Any) -> list:
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        for name in ("records", "messages", "items", "entries"):
            if isinstance(payload.get(name), list):
                return list(payload[name])
        return [payload]
    return []


def _front_matter_date(text: str) -> str | None:
    """A Markdown date line is a claim about the note, so it is read literally."""
    for line in text.splitlines()[:20]:
        stripped = line.strip()
        if stripped.lower().startswith("date:"):
            return stripped.split(":", 1)[1].strip().strip("\"'") or None
    return None


def _mtime_of(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def _media_type(suffix: str) -> str:
    return {".md": "text/markdown", ".markdown": "text/markdown",
            ".json": "application/json", ".jsonl": "application/x-ndjson",
            ".ndjson": "application/x-ndjson"}.get(suffix.lower(), "text/plain")
