"""C2 source contract: capability declaration, bounded paging, honest normalization."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from ..ids import timestamp
from ..storage.evidence import EvidenceError

__all__ = ["Capabilities", "Page", "Skipped", "SourceAdapter", "normalize_time"]


@dataclass(frozen=True)
class Capabilities:
    """What a source can actually do, declared rather than assumed.

    Discovery drives the plan: a source that cannot report deletions can never
    promise that forgetting reached it, and that limitation has to be visible
    before the first read rather than discovered during an audit.
    """

    history: bool = False
    live: bool = False
    deletion_events: bool = False
    revision_history: bool = False
    max_records_per_page: int = 200
    max_bytes_per_page: int = 4_000_000


@dataclass(frozen=True)
class Skipped:
    """A recoverable gap. Reported, counted and never silently dropped."""

    ref: str
    reason: str


@dataclass(frozen=True)
class Page:
    """One bounded read result.

    ``next_cursor`` None means the end. Gaps are reported through *skipped*
    rather than a 'complete' flag, because a page can be finished and still
    have incomplete coverage, and the two must not be conflated.
    """

    envelopes: tuple[dict[str, Any], ...] = ()
    next_cursor: str | None = None
    skipped: tuple[Skipped, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "envelopes", tuple(self.envelopes))
        object.__setattr__(self, "skipped", tuple(self.skipped))


class SourceAdapter:
    """Base contract every adapter implements.

    Fetching and committing stay separate: an adapter normalises bytes into
    envelopes and never opens a store, a lease or a network transaction itself.
    """

    source: str = ""
    capabilities: Capabilities = Capabilities()

    def check(self) -> dict[str, Any]:
        """Cheap reachability and permission probe. Never reads content."""
        raise NotImplementedError

    def read_page(self, cursor: str | None) -> Page:
        """Return one bounded page. ``next_cursor`` None means the end."""
        raise NotImplementedError

    def normalize(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def envelope(self, **values: Any) -> dict[str, Any]:
        """Stamp an adapter's own source name onto a normalised record."""
        if not self.source:
            raise EvidenceError("adapter must declare a source name")
        values["source"] = self.source
        return values

    def read_all(self, *, max_pages: int = 10_000) -> tuple[list[dict[str, Any]], list[Skipped]]:
        """Drain the source. An adapter that never reaches its end is stopped."""
        envelopes: list[dict[str, Any]] = []
        skipped: list[Skipped] = []
        cursor: str | None = None
        for _ in range(max_pages):
            page = self.read_page(cursor)
            envelopes.extend(page.envelopes)
            skipped.extend(page.skipped)
            if page.next_cursor is None:
                return envelopes, skipped
            cursor = page.next_cursor
        raise EvidenceError(f"source {self.source!r} did not finish within {max_pages} pages")


def normalize_time(value: Any) -> tuple[str | None, str, str | None]:
    """Resolve a source timestamp to (occurred_at, precision, note).

    An unambiguous instant is required before we claim when something happened.
    A bare '2026-03-04 17:00' in an export with no zone is not one: guessing
    local time would file the event under the wrong day for anyone who has ever
    crossed a timezone, and guessing UTC would be equally arbitrary.
    """
    if value in (None, ""):
        return None, "unknown", "no time reported by source"
    if not isinstance(value, str):
        return None, "unknown", "source time was not text"
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None, "unknown", "unparsable source time"
    if parsed.tzinfo is None:
        # Date-only strings are ambiguous too, but a bare date is precise as a
        # day, so it is reported as a day rather than thrown away.
        if _DATE_ONLY.match(text):
            return f"{text}T00:00:00+00:00", "day", "date-only source time, day precision"
        return None, "unknown", "source time carried no timezone offset"
    return timestamp(text), _precision_of(text), None


_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _precision_of(text: str) -> str:
    """Read precision off the written form, not the parsed value.

    Deciding from the parsed datetime would call 17:00:00 minute-precise, since
    its seconds happen to be zero, and quietly widen a timestamp the source
    actually stated to the second.
    """
    time_part = re.split(r"[T ]", text, maxsplit=1)
    if len(time_part) == 1:
        return "day"
    clock = re.match(r"^(\d{2}):(\d{2})(?::(\d{2}))?", time_part[1])
    if not clock:
        return "minute"
    return "second" if clock.group(3) else "minute"
