"""C2 structured fixtures: measurements that keep their units, gaps and quality.

A sensor log read as prose loses the only things that make it useful: what the
number was *in*, which device said it, how far apart the samples were, and which
samples are missing. This adapter keeps all of that in the record and computes the
statistics in code — no model is asked to average a heart rate — because a
measurement claim has to be reproducible from the same rows later.

A series is grouped by what is actually being measured (device, kind and unit
together), not by file: two devices in one export are two different statements
about the world, and a unit change mid-series is a gap in meaning rather than a
convenient continuation.
"""
from __future__ import annotations

import csv
import json
import posixpath
import re
from pathlib import Path
from typing import Any, Iterable

from ..ids import now, digest
from ..storage.measurements import summarise
from .base import Capabilities, Page, Skipped, SourceAdapter, normalize_time

__all__ = ["StructuredSource", "MEASUREMENT_KEYS"]

MEASUREMENT_KEYS = ("value", "reading", "amount", "measurement")
TIME_KEYS = ("time", "timestamp", "date", "at", "observed_at")
DEVICE_KEYS = ("device", "source", "sensor", "mac", "id")
QUALITY_KEYS = ("quality", "status", "confidence", "flag")
_UNIT = re.compile(r"^([A-Za-zµΩ°%/·²³]{1,16})(\s|$)")
_MAX_SAMPLES_PER_SERIES = 5000


class StructuredSource(SourceAdapter):
    """``.csv`` / ``.jsonl`` / ``.json`` sample fixtures, one record per series."""

    source = "structured"
    granularities = ("sample", "series")
    capabilities = Capabilities(history=True, live=False, deletion_events=False,
                                revision_history=False, max_records_per_page=40,
                                max_bytes_per_page=4_000_000)

    def __init__(self, root: str | Path, *, source: str = "structured",
                 per_sample: bool = False, records_per_page: int | None = None):
        self.root = Path(root).expanduser().resolve()
        self.source = source
        self.per_sample = per_sample
        if records_per_page is not None and not 1 <= records_per_page <= 1000:
            raise ValueError("records_per_page must be between 1 and 1000")
        if records_per_page:
            from dataclasses import replace
            self.capabilities = replace(self.capabilities,
                                        max_records_per_page=records_per_page)

    # -- contract ------------------------------------------------------------

    def check(self) -> dict[str, Any]:
        if not self.root.is_dir():
            return {"ok": False, "reason": f"{self.root} is not a directory",
                    "content_read": False}
        keys = self._keys()
        return {"ok": True, "root": str(self.root), "candidates": len(keys),
                "grouping": "sample" if self.per_sample else "series",
                "content_read": False, "observed_at": now()}

    def read_page(self, cursor: str | None) -> Page:
        from .export_paging import read_export_page
        def read(key):
            size = self._size_of(key)
            if size is None:
                return [], [Skipped(key, "stat failed or the file disappeared")]
            if size > self.capabilities.max_bytes_per_page:
                return [], [Skipped(key, f"{size} bytes exceeds the byte per-page bound")]
            rows, gaps = self._rows(key)
            produced = [row[1] for row in rows if row[1] is not None] if self.per_sample else self._series(key, rows)
            return produced, gaps
        return read_export_page(self, cursor, self._keys(), read)

    # -- reading -------------------------------------------------------------

    def _keys(self) -> list[str]:
        if not self.root.is_dir():
            return []
        found = []
        for path in self.root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() not in {".csv", ".tsv", ".json", ".jsonl", ".ndjson"}:
                continue
            if path.name.startswith("."):
                # A fixture too large for one page stays in the list: read_page reports
                # it as the gap it is, rather than the listing making it vanish.
                continue
            found.append(posixpath.normpath(path.relative_to(self.root).as_posix()))
        return sorted(found)

    def _size_of(self, key: str) -> int | None:
        try:
            return (self.root / key).stat().st_size
        except OSError:
            return None

    def _rows(self, key: str) -> tuple[list[tuple[int, dict[str, Any] | None, str]],
                                        list[Skipped]]:
        """One entry per row: the normalised envelope, or why that row is not one."""
        path = self.root / key
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return [], [Skipped(key, "read failed or the file is not UTF-8")]
        try:
            records = list(_records(raw, path))
        except (ValueError, json.JSONDecodeError, csv.Error) as error:
            return [], [Skipped(key, f"malformed fixture: {str(error)[:200]}")]
        gaps: list[Skipped] = []
        out = []
        for position, record in enumerate(records):
            envelope, reason = self._sample(key, position, record)
            if envelope is None:
                gaps.append(Skipped(f"{key}#{position}", reason))
            else:
                out.append((position, envelope, reason))
        return out, gaps

    def _sample(self, key: str, position: int, record: Any) -> tuple[dict[str, Any] | None, str]:
        if not isinstance(record, dict):
            return None, "the row is not an object"
        value = _first(record, MEASUREMENT_KEYS)
        column = None
        if value is None or str(value).strip() == "":
            # A fixture whose value column is named for its unit ("kg", "bpm") is the
            # common case in exported health data, and guessing between two numeric
            # columns is not something to do silently.
            value, column = _only_numeric(record)
            if value is None:
                return None, "the row carries no measurement value"
        try:
            number = float(str(value).strip())
        except ValueError:
            return None, f"{value!r} is not a number"
        unit = _unit_of(record, str(value), column)
        # The file is usually the measure ("weight.csv") and the column the unit
        # ("kg"); a column named for a measure only wins when the file says nothing.
        measure = str(_first(record, ("measure", "type", "kind", "metric"))
                      or path_measure(key) or column or "measurement")
        device = str(_first(record, DEVICE_KEYS) or "unattributed")[:200]
        quality = str(_first(record, QUALITY_KEYS) or "unknown")[:60]
        moment = normalize_time(_first(record, TIME_KEYS))
        occurred, precision, note = moment
        return self.envelope(
            source_id=f"{key}#{position}", revision=digest(record)[:32], kind="measurement",
            text=f"{measure} {number}{(' ' + unit) if unit else ''} at "
                 f"{occurred or 'an unplaced time'}",
            observed_at=now(), occurred_at=occurred, occurred_precision=precision,
            metadata={"fixture": key, "position": position, "measure": measure,
                      "value": number, "unit": unit or None, "device": device,
                      "quality": quality,
                      "sequence": position,
                      **({"time_note": note} if note else {})},
        ), ""

    def _series(self, key: str, rows: list[tuple[int, dict, str]]) -> list[dict[str, Any]]:
        """Group accepted samples by what they measure, and describe each group."""
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for _position, envelope, _reason in rows:
            meta = envelope["metadata"]
            groups.setdefault((meta["device"], meta["measure"], meta["unit"] or ""),
                              []).append(envelope)
        out = []
        for (device, measure, unit), samples in sorted(groups.items()):
            summary = summarise([item["metadata"]["value"] for item in samples],
                                moments=[item["occurred_at"] for item in samples])
            placed = [item["occurred_at"] for item in samples if item["occurred_at"]]
            unplaced = len(samples) - len(placed)
            named = f" in {unit}" if unit else ""
            middle = f", median {summary['median']}" if summary["median"] is not None else ""
            span = (f", range {summary['min']} to {summary['max']}"
                    if summary["min"] is not None else "")
            out.append(self.envelope(
                source_id=f"{key}#series:{device}:{measure}:{unit}",
                revision=digest([item["revision"] for item in samples])[:32],
                kind="measurement_series",
                text=f"{measure} from {device}: {summary['count']} samples{named}{middle}{span}",
                observed_at=now(),
                occurred_at=placed[0] if placed else None,
                occurred_precision="second" if placed else "unknown",
                metadata={"fixture": key, "device": device, "measure": measure,
                          "unit": unit or None, "samples": summary["count"],
                          "statistics": summary,
                          "first_at": placed[0] if placed else None,
                          "last_at": placed[-1] if placed else None,
                          "unplaced_times": unplaced or None,
                          "quality": _qualities(samples),
                          "interval_seconds": summary["interval_seconds"],
                          "gaps": summary["gaps"],
                          "positions": [item["metadata"]["position"] for item in
                                        samples[:_MAX_SAMPLES_PER_SERIES]]},
            ))
        return out


def path_measure(key: str) -> str | None:
    stem = Path(key).stem.lower()
    return stem if re.fullmatch(r"[a-z0-9_+-]{1,60}", stem) else None


def _unit_of(record: dict[str, Any], written: str, column: str | None = None) -> str | None:
    explicit = _first(record, ("unit", "units"))
    if explicit:
        return str(explicit).strip()[:32] or None
    if column:
        return column[:32]
    match = _UNIT.match(str(written).strip())
    return match.group(1) if match else None


_NON_NUMERIC_HINTS = set(TIME_KEYS) | set(DEVICE_KEYS) | set(QUALITY_KEYS) | \
    set(MEASUREMENT_KEYS) | {"measure", "type", "kind", "metric", "unit", "units", "id",
                             "name", "label"}


def _only_numeric(record: dict[str, Any]) -> tuple[Any, str | None]:
    """The one numeric column, or nothing at all.

    Two of them is a question for the operator rather than a coin toss: which one is
    the weight and which one is the body fat changes what the series means.
    """
    candidates = []
    for key, value in record.items():
        name = str(key or "").strip().lower()
        if not name or name in _NON_NUMERIC_HINTS or isinstance(value, bool):
            continue
        try:
            float(str(value).strip())
        except (TypeError, ValueError):
            continue
        if str(value).strip():
            candidates.append((value, re.sub(r"[_\d]+$", "", name) or name))
    if len(candidates) == 1:
        return candidates[0]
    return None, None


def _first(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in record.items()
               if key is not None}
    for name in names:
        if name in lowered and lowered[name] not in (None, ""):
            return lowered[name]
    return None


def _qualities(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in samples:
        quality = str(item["metadata"]["quality"])
        counts[quality] = counts.get(quality, 0) + 1
    return counts


def _records(raw: str, path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if suffix == ".json":
        payload = json.loads(raw)
        for name in ("samples", "records", "measurements", "readings", "items", "data"):
            if isinstance(payload, dict) and isinstance(payload.get(name), list):
                return payload[name]
        return payload if isinstance(payload, list) else [payload]
    delimiter = "\t" if suffix == ".tsv" else ","
    return [dict(row) for row in csv.DictReader(raw.splitlines(), delimiter=delimiter)]
