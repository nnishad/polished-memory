"""C13 typed measurements: a window over stored samples, computed in code.

A measurement is a number that says what it measures, what unit it is in, where it came
from and when. The plan keeps those four attached because a reading without them is not
reusable: ``72`` is not an answer unless it is known whether it is bpm or degrees, and a
mean over a series that changed units mid-file is arithmetic performed on a category
error.

So this module reads the canonical records the adapters wrote, refuses to average across
a unit conflict rather than reporting a plausible wrong number, and treats time it cannot
place as a gap it says so about. Every statistic here is reproducible from the same rows,
which is the whole reason it is computed in code: a model's summary of a heart rate cannot
be checked against the heart rate.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from typing import Any, Iterable

from ..ids import timestamp
from .evidence import EvidenceError

__all__ = ["Measurements", "summarise", "MEASUREMENT_KIND", "MAX_SAMPLES_PER_READ"]

MEASUREMENT_KIND = "measurement"
# A read that would have to pull more than this says so instead of answering: an
# unbounded aggregate over somebody's whole life is neither fast nor a number anyone
# asked for.
MAX_SAMPLES_PER_READ = 20_000
# The ids are reported so a reading can be checked against its evidence, bounded because
# a list of twenty thousand ids is not a citation.
MAX_RECORDS_REPORTED = 25


def summarise(values: Iterable[float], *,
              moments: Iterable[str | None] = ()) -> dict[str, Any]:
    """Deterministic description of a sample series.

    ``gaps`` is a statement about the missing data: a stretch where the device went quiet
    for more than twice its own cadence. It is computed from placed times only, so a
    series with unplaced samples reports an interval it can stand behind and counts the
    rest separately.
    """
    numbers = [float(value) for value in values]
    out: dict[str, Any] = {"count": len(numbers), "min": None, "max": None, "mean": None,
                           "median": None, "stdev": None, "interval_seconds": None,
                           "gaps": []}
    if not numbers:
        return out
    out["min"], out["max"] = min(numbers), max(numbers)
    out["mean"] = round(statistics.fmean(numbers), 6)
    out["median"] = round(statistics.median(numbers), 6)
    if len(numbers) > 1:
        out["stdev"] = round(statistics.stdev(numbers), 6)
    times = [value for value in moments if value]
    if len(times) > 1:
        steps = _steps(times)
        out["interval_seconds"] = round(statistics.median(steps), 3)
        expected = out["interval_seconds"] or 0
        out["gaps"] = [{"after": times[index], "seconds": round(step, 3)}
                       for index, step in enumerate(steps)
                       if expected and step > expected * 2][:50]
    return out


class Measurements:
    """The read side of the measurement plane: what is held, and a window over it."""

    def __init__(self, store: Any, *, max_samples: int = MAX_SAMPLES_PER_READ):
        if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples < 1:
            raise EvidenceError("max_samples must be a positive integer")
        self.store = store
        self.db = store.db
        self.max_samples = max_samples

    def available(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Every series this store can answer for, grouped as the samples group themselves.

        Reported before any question is asked, because "no readings" and "nobody has
        written readings down in that shape" are different answers and only one of them
        is the operator's to act on.
        """
        rows = self.db.execute(
            "SELECT json_extract(metadata,'$.measure') AS measure, "
            "       json_extract(metadata,'$.device')   AS device, "
            "       json_extract(metadata,'$.unit')     AS unit, "
            "       count(*) AS samples, min(occurred_at) AS first_at, "
            "       max(occurred_at) AS last_at, group_concat(DISTINCT source) AS sources "
            "FROM records r "
            "WHERE r.kind=? AND r.deleted=0 AND "
            "      COALESCE((SELECT v.hidden FROM record_visibility v "
            "                WHERE v.record_id=r.id), 0) = 0 "
            "GROUP BY measure, device, unit ORDER BY measure, device, unit LIMIT ?",
            (MEASUREMENT_KIND, _bounded(limit))).fetchall()
        return [{"measure": row["measure"], "device": row["device"], "unit": row["unit"],
                 "samples": int(row["samples"]), "first_at": row["first_at"],
                 "last_at": row["last_at"],
                 "sources": sorted(str(row["sources"] or "").split(","))}
                for row in rows]

    def series(self, measure: str, *, device: str | None = None, source: str | None = None,
               unit: str | None = None, since: str | None = None,
               until: str | None = None) -> dict[str, Any]:
        """One measure's samples in one window, described by numbers computed here.

        The answer is the reading *and* its accounting: how many samples were placed in
        time, how many carried no time at all, how many named a different unit and were
        left out. A mean is only worth as much as that list.
        """
        measure = _term(measure, "measure")
        # The window is kept apart from the rest of the filter because it is the one
        # predicate that can discard a row for a reason the reader cannot see in the
        # result: `occurred_at >= ?` drops every sample with no time at all, silently.
        base = ["r.kind=?", "r.deleted=0",
                "COALESCE((SELECT v.hidden FROM record_visibility v "
                "WHERE v.record_id=r.id), 0) = 0",
                "json_extract(r.metadata,'$.measure')=?"]
        held = [MEASUREMENT_KIND, measure]
        if device:
            base.append("json_extract(r.metadata,'$.device')=?")
            held.append(_term(device, "device"))
        if source:
            base.append("r.source=?")
            held.append(_term(source, "source"))
        bounds = {"since": _moment(since, "since"), "until": _moment(until, "until")}
        since_at, until_at = bounds["since"], bounds["until"]
        window: list[str] = []
        limits: list[str] = []
        if since_at:
            window.append("r.occurred_at>=?")
            limits.append(since_at)
        if until_at:
            window.append("r.occurred_at<=?")
            limits.append(until_at)
        rows = self.db.execute(
            f"SELECT r.id, r.source, r.occurred_at, r.occurred_precision, r.metadata "
            f"FROM records r WHERE {' AND '.join([*base, *window])} "
            f"ORDER BY r.occurred_at, r.id LIMIT ?",
            [*held, *limits, self.max_samples + 1]).fetchall()
        overflow = len(rows) > self.max_samples
        # Counted from the rows the window dropped, in the same terms the reader asked in,
        # because "3 samples have no time" means something else when one of them is in lb
        # and the question was about kg.
        counted = [*base]
        asked_for = list(held)
        if unit:
            counted.append("json_extract(r.metadata,'$.unit')=?")
            asked_for.append(unit)
        unplaced_excluded = int(self.db.execute(
            f"SELECT count(*) FROM records r WHERE {' AND '.join(counted)} "
            "AND r.occurred_at IS NULL", asked_for).fetchone()[0]) if window else 0

        values: list[float] = []
        moments: list[str | None] = []
        cited: list[str] = []
        units: dict[str, int] = {}
        qualities: dict[str, int] = {}
        unreadable = 0
        unplaced = 0
        for row in rows[:self.max_samples]:
            try:
                metadata = json.loads(row["metadata"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            named = str(metadata.get("unit") or "")
            units[named] = units.get(named, 0) + 1
            if unit and named != unit:
                continue
            quality = str(metadata.get("quality") or "unknown")
            qualities[quality] = qualities.get(quality, 0) + 1
            try:
                value = float(str(metadata.get("value")))
            except (TypeError, ValueError):
                # A sample whose value is not a number is a source bug, and the honest
                # report is that it could not be read rather than a skipped row.
                unreadable += 1
                continue
            values.append(value)
            cited.append(str(row["id"]))
            moment = row["occurred_at"]
            if moment:
                moments.append(str(moment))
            else:
                unplaced += 1
        conflicting = sorted(name for name, count in units.items() if count) if not unit else []
        report: dict[str, Any] = {
            "measure": measure, "device": device, "source": source, "unit": unit,
            "window": bounds, "samples": len(values), "quality": qualities,
            "units_seen": dict(sorted(units.items())), "unreadable": unreadable or None,
            "unplaced_times": unplaced or None,
            # The citation list is of the rows that are *in* the reading: a sample excluded
            # for naming another unit is reported in units_seen, not quietly cited as proof.
            "records": cited[:MAX_RECORDS_REPORTED],
            "records_more": max(0, len(cited) - MAX_RECORDS_REPORTED) or None,
        }
        if window:
            report["unplaced_excluded"] = unplaced_excluded or None
        if overflow:
            report["refused"] = (f"more than {self.max_samples} samples match; narrow "
                                 "the window or name a device rather than averaging the "
                                 "whole archive")
            report["statistics"] = None
            return report
        if len(conflicting) > 1:
            # The arithmetic is what is dangerous here, not the answer: a mean across two
            # units is a number that looks like a measurement.
            report["unit_conflict"] = conflicting
            report["statistics"] = None
            report["refused"] = ("the stored samples name more than one unit "
                                 f"({', '.join(repr(name) for name in conflicting)}); "
                                 "name one with --unit and the excluded count is reported")
            return report
        seen = {name for name, count in units.items() if count}
        if unit is None and len(seen) == 1 and seen != {""}:
            # `--list` already names the unit of every series it holds, so a reading that
            # answered null knew less than the census about the same rows — and the mean
            # below is in that unit whether anybody named it or not.
            report["unit"] = next(iter(seen))
        report["statistics"] = summarise(values, moments=moments)
        report["first_at"] = min((item for item in moments), default=None)
        report["last_at"] = max((item for item in moments), default=None)
        if not values:
            report["note"] = ("nothing was held for that question; this is an empty "
                              "reading, not a zero")
        return report


def _steps(times: list[str]) -> list[float]:
    parsed = [datetime.fromisoformat(value) for value in times]
    return [abs((second - first).total_seconds()) for first, second in zip(parsed, parsed[1:])
            if (second - first).total_seconds() >= 0]


def _bounded(value: int, *, maximum: int = 500) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise EvidenceError(f"limit must be an integer between 1 and {maximum}")
    return value


def _term(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 200:
        raise EvidenceError(f"{label} must be nonempty text of at most 200 characters")
    return text


def _moment(value: Any, label: str) -> str | None:
    """A window bound in the form the stored times are written in, or nothing.

    A naive stamp is refused rather than compared as text: ``"2026-03-01"`` sorts below
    every timestamp of that day, so a loose bound would quietly drop the samples it was
    meant to keep.
    """
    if value in (None, ""):
        return None
    try:
        return timestamp(str(value))
    except ValueError as error:
        raise EvidenceError(
            f"{label} must be a timezone-aware timestamp: {error}") from error
