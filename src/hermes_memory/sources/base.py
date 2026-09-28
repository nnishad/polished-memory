"""C2 source contract: capability declaration, bounded paging, honest normalization."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..ids import timestamp
from ..storage.evidence import EvidenceError

__all__ = ["Capabilities", "CursorExpired", "Page", "PART_SEPARATOR", "REDACTED", "Skipped",
           "SourceAdapter", "normalize_text", "normalize_time", "redact_secrets", "revocation",
           "part_container"]

# How an adapter names something *inside* one thing it reported as a whole: the two sides of a
# turn, one row of a file, one message of a session. The source's own id stays the container,
# so a gap on the container and a part that arrives later are the same event's two reports.
PART_SEPARATOR = "#"


def part_container(source_id: str) -> str | None:
    """The id of the thing ``source_id`` is a part of, or None when it names a whole.

    Only the suffix is stripped, once, and only at the separator: an id that contains one
    inside its own name still has exactly one container, and guessing further would invent a
    parent the source never reported.
    """
    head, _, tail = str(source_id).rpartition(PART_SEPARATOR)
    return head if tail and head else None


class CursorExpired(EvidenceError):
    """The source no longer accepts the position we stored for it.

    Raised by an adapter rather than caught: the difference between "unreachable" and
    "start over" matters, because a wedged cursor fails the same way forever while a
    restarted one fails once and then re-reads what it can.
    """


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

    ``page_token`` is how the source names this read. It decides what a replay
    is: a source whose position is stable across reads (an export, a file list)
    names the same page twice and the commit is a no-op, while a live source
    whose position is re-read as things arrive leaves it blank and the runtime
    names the read itself, so a poll that finds new mail is new work rather than
    a contradiction.
    """

    envelopes: tuple[dict[str, Any], ...] = ()
    next_cursor: str | None = None
    skipped: tuple[Skipped, ...] = ()
    page_token: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "envelopes", tuple(self.envelopes))
        object.__setattr__(self, "skipped", tuple(self.skipped))


class SourceAdapter:
    """Base contract every adapter implements.

    Fetching and committing stay separate: an adapter turns bytes into envelopes and
    never opens a store, a lease or a network transaction itself.

    There is deliberately no ``normalize(raw)`` callback on this contract. Normalization
    is shared code — ``normalize_text``, ``normalize_time``, ``redact_secrets`` — that an
    adapter calls from wherever its own format actually arrives, rather than a hook the
    framework could claim to have run while every adapter quietly did its shaping inside
    ``read_page`` anyway.
    """

    source: str = ""
    capabilities: Capabilities = Capabilities()
    # What this reader can be asked to keep. Most sources state one thing per record and
    # have no choice to make; a reader of sample fixtures can either keep every row or
    # collapse them into one described series, and only the operator gets to decide which
    # question they will still be able to ask next year.
    granularities: tuple[str, ...] = ()

    def check(self) -> dict[str, Any]:
        """Cheap reachability and permission probe. Never reads content."""
        raise NotImplementedError

    def read_page(self, cursor: str | None) -> Page:
        """Return one bounded page. ``next_cursor`` None means the end."""
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
_REPLACEMENT = "\ufffd"
REDACTED = "[redacted]"
_SECRET_VALUE = re.compile(
    r"(?:Bearer\s+[A-Za-z0-9._~+/=-]{12,}|sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{20,})")
_REVOKED = re.compile(
    r"(revok|invalid_grant|expired token|unauthorized|forbidden|insufficient_scope|"
    r"no longer authorized|access denied)", re.IGNORECASE)


def normalize_text(value: Any) -> str | None:
    """Text an adapter may put in a record, or None when the source gave none.

    Bytes are decoded strictly and text that already carries a replacement
    character is refused rather than stored: damaged bytes kept in a record are
    shown to a model later as if they had been understood. A value that is not
    text at all is missing, not empty — the difference is what a reported gap is
    for. Secret-shaped substrings are replaced here, in the one place every
    adapter's text passes through, because every adapter reads from something
    somebody else wrote.
    """
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str) or _REPLACEMENT in value:
        return None
    return redact_secrets(value).strip() or None


def redact_secrets(text: str) -> str:
    """Take bearer tokens and API keys out of text that happens to contain one.

    Someone pastes a credential into a conversation, a channel posts its own webhook
    URL, and the archive becomes a searchable copy of it. What is left says a
    credential was there, which is more use to an owner than either the secret or
    silence.
    """
    return _SECRET_VALUE.sub(REDACTED, text)


def revocation(reason: Any) -> dict[str, str]:
    """``{'coverage_state': 'revoked'}`` when the source's own words say that.

    A dead endpoint and a withdrawn grant look the same to a retry loop and are not
    the same to an operator: one is fixed by waiting, the other by asking again. An
    adapter that can tell them apart should say so in its check, and this is how.
    """
    return {"coverage_state": "revoked"} if _REVOKED.search(str(reason or "")) else {}


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
