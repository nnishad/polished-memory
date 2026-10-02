"""C2 WhatsApp export adapter: a chat log with no message ids and no timezone.

An export is append-only text, so a message is identified by where it sits in the
file rather than by anything the app wrote — stable across a re-read, and honest
about being a position. The timestamps are worse: a chat export records a clock
reading in whatever zone the phone was in, with no mark of which. Those stay
unknown until an operator says what the zone was; guessing would file a whole
conversation under the wrong day and quietly move every deadline inside it.

System notices (the encryption banner, "You deleted this message") are kept as
records of type ``system`` rather than dropped: they are not evidence about the
world, and anything weighing this chat later needs to be able to tell that apart.
"""
from __future__ import annotations

import posixpath
import re
import zoneinfo
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..ids import now
from .base import Capabilities, Page, Skipped, SourceAdapter

__all__ = ["WhatsAppExport", "MEDIA_PLACEHOLDER"]

_STAMPED = re.compile(
    r"^\[(?P<date>[^\]]+),\s+(?P<clock>[^\]]+)\]\s+(?P<author>[^:]{1,80}):\s(?P<text>.*)$")
_BARE = re.compile(
    r"^(?P<date>\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}),\s+(?P<clock>\d{1,2}:\d{2}(?::\d{2})?"
    r"\s?(?:AM|PM|am|pm)?)\s+-\s+(?P<author>[^:]{1,80}):\s(?P<text>.*)$")
MEDIA_PLACEHOLDER = re.compile(r"^<(?:attached|Media omitted|image omitted)[^>]*>$|"
                               r"^(?:IMAGE|VIDEO|STICKER|AUDIO|DOC|GIF|PTT) omitted$")
# The notices a WhatsApp export writes with no timestamp and no author. Prose that
# merely mentions someone leaving a group is not in here: a line of chat that starts
# with "left" is a continuation, and guessing wrong would split one message in two.
_SYSTEM = re.compile(r"^(?:Messages and calls are end-to-end encrypted"
                     r"(?:[^.]*\.)?|Created|You deleted this message|"
                     r"You missed a voice call|This message was deleted)", re.IGNORECASE)
_MAX_TEXT_CHARS = 60_000


class WhatsAppExport(SourceAdapter):
    """``_chat.txt`` exports, one record per message."""

    source = "whatsapp"
    capabilities = Capabilities(history=True, live=False, deletion_events=False,
                                revision_history=False, max_records_per_page=200,
                                max_bytes_per_page=4_000_000)

    def __init__(self, root: str | Path, *, source: str = "whatsapp",
                 timezone_name: str | None = None, pattern: str = "*chat*.txt",
                 records_per_page: int | None = None):
        self.root = Path(root).expanduser().resolve()
        self.source = source
        self.pattern = pattern
        self.zone = self._zone(timezone_name)
        if records_per_page is not None and not 1 <= records_per_page <= 1000:
            raise ValueError("records_per_page must be between 1 and 1000")
        if records_per_page:
            from dataclasses import replace
            self.capabilities = replace(self.capabilities,
                                        max_records_per_page=records_per_page)

    @staticmethod
    def _zone(name: str | None) -> Any:
        if not name:
            return None
        try:
            return zoneinfo.ZoneInfo(name)
        except (KeyError, ValueError):
            raise ValueError(f"unknown timezone {name!r}; the export zone has to be named "
                             "by the operator, not guessed") from None

    # -- contract ------------------------------------------------------------

    def check(self) -> dict[str, Any]:
        if not self.root.is_dir():
            return {"ok": False, "reason": f"{self.root} is not a directory",
                    "content_read": False}
        keys = self._keys()
        return {"ok": True, "root": str(self.root), "chats": len(keys),
                "export_timezone": str(self.zone) if self.zone else None,
                "time_basis": "declared" if self.zone else "unknown",
                "content_read": False, "observed_at": now()}

    def read_page(self, cursor: str | None) -> Page:
        from .export_paging import read_export_page
        def read(key):
            size = self._size_of(key)
            if size is None:
                return [], [Skipped(key, "stat failed or the file disappeared")]
            if size > self.capabilities.max_bytes_per_page:
                return [], [Skipped(key, f"{size} bytes exceeds the byte per-page bound")]
            raw = self._raw(key)
            if raw is None:
                return [], [Skipped(key, "read failed or the file is not UTF-8")]
            envelopes, skipped = [], []
            for position, item, reason in self._messages(key, raw):
                if item is None:
                    skipped.append(Skipped(f"{key}#{position}", reason))
                else:
                    envelopes.append(item)
            return envelopes, skipped
        return read_export_page(self, cursor, self._keys(), read)

    # -- reading -------------------------------------------------------------

    def _keys(self) -> list[str]:
        if not self.root.is_dir():
            return []
        found = []
        for path in self.root.rglob(self.pattern):
            if path.is_symlink() or not path.is_file():
                continue
            # A chat log too large for one page stays in the list; read_page says so.
            found.append(posixpath.normpath(path.relative_to(self.root).as_posix()))
        return sorted(found)

    def _size_of(self, key: str) -> int | None:
        try:
            return (self.root / key).stat().st_size
        except OSError:
            return None

    def _raw(self, key: str) -> str | None:
        try:
            return (self.root / key).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def _messages(self, key: str, text: str):
        """Yield (position, envelope, reason) per message.

        A line that does not open with a stamp is a continuation: WhatsApp wraps
        long messages over lines, and treating each of those as its own message
        would turn one thought into several pieces of evidence.
        """
        position = -1
        current: dict[str, Any] | None = None
        for line in text.splitlines():
            match = _STAMPED.match(line) or _BARE.match(line)
            if match:
                if current is not None:
                    yield self._record(key, current)
                position += 1
                current = {"position": position, "author": match["author"].strip(),
                           "stamp": f"{match['date']} {match['clock']}",
                           "lines": [match["text"]]}
                continue
            stripped = line.strip()
            if current is None:
                if not stripped:
                    continue
                if _SYSTEM.match(stripped):
                    position += 1
                    current = {"position": position, "author": "system", "stamp": "",
                               "lines": [stripped]}
                    continue
                # Nothing to attach it to: a stray line before the first message is
                # reported rather than guessed at.
                yield position + 1, None, "a headerless line before the first message"
                continue
            if not stripped:
                continue
            if _SYSTEM.match(stripped):
                # Its own record, not the tail of the message above it.
                yield self._record(key, current)
                position += 1
                current = {"position": position, "author": "system", "stamp": "",
                           "lines": [stripped]}
                continue
            current["lines"].append(line)
        if current is not None:
            yield self._record(key, current)

    def _record(self, key: str, message: dict[str, Any]) -> tuple[int, dict | None, str]:
        body = "\n".join(message["lines"]).strip()
        position = message["position"]
        if not body:
            return position, None, "the message is empty"
        if MEDIA_PLACEHOLDER.match(body):
            return position, None, f"{body!r} is a media placeholder; the bytes are not here"
        occurred, precision, note = self._moment(message["stamp"])
        system = message["author"] == "system" or bool(_SYSTEM.match(body))
        if system and not message["stamp"]:
            note = "a system notice carries no timestamp"
        return position, self.envelope(
            source_id=f"{key}#{position}", revision="1",
            kind="system_notice" if system else "message",
            text=body[:_MAX_TEXT_CHARS], observed_at=now(), occurred_at=occurred,
            occurred_precision=precision,
            metadata={"chat": key, "position": position, "author": message["author"],
                      "stamp": message["stamp"], "system": True if system else None,
                      "lines": len(message["lines"]),
                      "time_basis": (str(self.zone) if self.zone else "unknown")
                      if occurred else "none",
                      **({"time_note": note} if note else {}),
                      **({"truncated": True} if len(body) > _MAX_TEXT_CHARS else {})},
        ), ""

    def _moment(self, stamp: str) -> tuple[str | None, str, str | None]:
        """Resolve a local clock reading, only if the operator said which one."""
        parsed = _parse_stamp(stamp)
        if parsed is None:
            return None, "unknown", "unparsable message timestamp"
        if self.zone is None:
            return None, "unknown", ("the export states no timezone; pass timezone_name to "
                                     "read these messages as instants")
        try:
            local = self.zone.localize(parsed) if hasattr(self.zone, "localize") else \
                parsed.replace(tzinfo=self.zone)
        except (ArithmeticError, OverflowError, ValueError):
            return None, "unknown", "the message timestamp could not be placed in the " \
                                    "declared timezone"
        moment = local.astimezone(timezone.utc)
        # A wall clock reading that repeats at the autumn fold is two instants, and
        # the earlier one is the only reading that cannot already have happened.
        return moment.isoformat(), "minute" if second_free(stamp) else "second", None


def second_free(stamp: str) -> bool:
    return not re.search(r":\d{2}:\d{2}", stamp)


def _parse_stamp(stamp: str) -> datetime | None:
    stamp = stamp.strip().replace(".", "/").replace("-", "/")
    for order in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y %I:%M:%S %p",
                  "%d/%m/%Y %I:%M %p", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
                  "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M %p"):
        try:
            return datetime.strptime(stamp, order)
        except ValueError:
            continue
    return None
