"""C2 email adapter: an export of messages, read without inventing anything.

Two properties carry most of the design. A message's identity is its Message-ID, so
a revision is not a content hash here — the header is immutable, and re-reading the
same export has to land on the same record. And a reply's provenance is kept as what
the headers actually said (``In-Reply-To``, ``References``, the forwarded marker,
how much of the body was quoted) rather than as a resolved link, because a link to a
message this export does not contain would reject the whole page and lose the reply
with it.

Attachment bytes are deliberately not ingested here: an export directory is not an
authorisation to copy every file inside it into the store. What arrives is the
descriptor — name, media type, size, whether it was inline — plus a gap for anything
the bound refused.

``GmailSource`` at the end of this file reads a live mailbox through a transport the
operator supplies, and shares the parse above line for line: two readings of one
message have to agree about what it said.
"""
from __future__ import annotations

import base64
import email
import email.policy
import mailbox
import posixpath
import re
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterator

from ..ids import now
from ..storage.evidence import EvidenceError
from .base import (Capabilities, CursorExpired, Page, Skipped, SourceAdapter,
                   revocation)

__all__ = ["EmailSource", "MAX_MESSAGE_BYTES", "attachment_descriptors"]

MAX_MESSAGE_BYTES = 8_000_000
_MAX_TEXT_CHARS = 120_000
_MAX_LIST = 20
_FORWARD_HEADER = re.compile(r"^(Begin forwarded message|Forwarded message|\[forwarded\])",
                             re.IGNORECASE | re.MULTILINE)
# The delimiter is "-- " on its own line, and an export writes its line endings as
# CRLF as often as LF — matching only one of the two leaves every real signature in
# the body and every test passing.
_SIGNATURE = re.compile(r"\r?\n-- ?\r?\n")
_QUOTED = re.compile(r"^\s*>", re.MULTILINE)
_REPLACEMENT = "\ufffd"


class EmailSource(SourceAdapter):
    """``.eml`` and ``.mbox`` files under a directory the operator pointed at."""

    source = "email"
    capabilities = Capabilities(history=True, live=False, deletion_events=False,
                                revision_history=False, max_records_per_page=100,
                                max_bytes_per_page=4_000_000)

    def __init__(self, root: str | Path, *, source: str = "email",
                 records_per_page: int | None = None):
        self.root = Path(root).expanduser().resolve()
        self.source = source
        if records_per_page is not None and not 1 <= records_per_page <= 1000:
            raise ValueError("records_per_page must be between 1 and 1000")
        if records_per_page:
            self.capabilities = replace(self.capabilities,
                                        max_records_per_page=records_per_page)

    # -- contract ------------------------------------------------------------

    def check(self) -> dict[str, Any]:
        """Count the files. Reading one is what ``read_page`` is for."""
        if not self.root.is_dir():
            return {"ok": False, "reason": f"{self.root} is not a directory",
                    "content_read": False}
        keys = self._keys()
        return {"ok": True, "root": str(self.root), "candidates": len(keys),
                "formats": sorted({Path(key).suffix.lstrip(".") for key in keys}),
                "content_read": False, "observed_at": now()}

    def read_page(self, cursor: str | None) -> Page:
        from .export_paging import read_export_page
        def read(key):
            size = self._size_of(key)
            if size is None:
                return [], [Skipped(key, "stat failed or the file disappeared")]
            if size > MAX_MESSAGE_BYTES:
                return [], [Skipped(key, f"{size} bytes exceeds the {MAX_MESSAGE_BYTES} byte per-message bound")]
            envelopes, skipped = [], []
            for ref, envelope, reason in self._expand(key):
                if envelope is None:
                    # A multi-message file reports which message it could not take,
                    # or two failures in one export would be one row.
                    skipped.append(Skipped(ref, reason))
                else:
                    envelopes.append(envelope)
            return envelopes, skipped
        return read_export_page(self, cursor, self._keys(), read)

    # -- reading -------------------------------------------------------------

    def _keys(self) -> list[str]:
        if not self.root.is_dir():
            return []
        found = []
        for path in self.root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                # A link out of the export directory would ingest a file nobody
                # pointed the connector at.
                continue
            if path.suffix.lower() not in {".eml", ".mbox"} or path.name.startswith("."):
                continue
            found.append(posixpath.normpath(path.relative_to(self.root).as_posix()))
        return sorted(found)

    def _size_of(self, key: str) -> int | None:
        try:
            return (self.root / key).stat().st_size
        except OSError:
            return None

    def _expand(self, key: str) -> Iterator[tuple[str, dict[str, Any] | None, str]]:
        """One (reference, record) per message, or the reference and the reason.

        A ``.mbox`` holds many messages in one file, so the file's name is not enough
        to say which one failed: the identity of both the record and its gap has to
        come from inside the file.
        """
        path = self.root / key
        if path.suffix.lower() == ".mbox":
            try:
                box = mailbox.mbox(str(path))
            except OSError as error:
                yield key, None, f"mbox could not be opened: {error.strerror or error}"
                return
            try:
                for position, message in enumerate(box):
                    ref = f"{key}#{position}"
                    envelope, reason = self._one(ref, message.__bytes__())
                    yield ref, envelope, reason
            finally:
                box.close()
            return
        try:
            raw = path.read_bytes()
        except OSError as error:
            yield key, None, f"read failed: {error.strerror or error}"
            return
        envelope, reason = self._one(key, raw)
        yield key, envelope, reason

    def _one(self, ref: str, raw: bytes) -> tuple[dict[str, Any] | None, str]:
        return parse_message(self.source, ref, raw)


class GmailSource(SourceAdapter):
    """A live mailbox, read through a connector the operator switched on.

    Nothing here opens a socket: the account transport is handed in, already
    authorised for one address and one query, and `enabled` defaults to False because
    reading somebody's mail is not a thing a library decides. Two properties shape the
    rest. A message's identity is its RFC 2822 Message-ID rather than the mailbox's
    own id, so one message seen both in a backup directory and in the mailbox is one
    record and not two; and the parse is the export's parse, so the two readings of a
    message cannot disagree about what it said.

    Labels are deliberately absent. Which labels a message carries changes when it is
    read, filed or trashed, and mutable mailbox state inside a fingerprint would make
    the second sight of a message a different revision of the same id — which the
    store is right to refuse. The mailbox reports removals as gaps for the same
    reason: this connector can say something went away, and cannot say the evidence
    did.
    """

    source = "gmail"
    capabilities = Capabilities(history=True, live=True, deletion_events=False,
                                revision_history=False, max_records_per_page=100,
                                max_bytes_per_page=4_000_000)

    def __init__(self, api: Any, *, query: str, enabled: bool = False,
                 source: str = "gmail", records_per_page: int | None = None):
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise EvidenceError(
                "the Gmail connector reads an operator-written query; a default of "
                "'everything' would be a scope nobody authorised")
        if records_per_page is not None and not 1 <= records_per_page <= 1000:
            raise ValueError("records_per_page must be between 1 and 1000")
        for method in ("profile", "messages", "message", "history"):
            if not callable(getattr(api, method, None)):
                raise EvidenceError(f"the Gmail transport offers no {method}() to read through")
        self.api = api
        self.query = query.strip()
        self.enabled = bool(enabled)
        self.source = source
        if records_per_page:
            self.capabilities = replace(self.capabilities,
                                        max_records_per_page=records_per_page)

    # -- contract ------------------------------------------------------------

    def check(self) -> dict[str, Any]:
        """Who we are allowed to read, and what was asked for. No message content."""
        if not self.enabled:
            return {"ok": False, "content_read": False,
                    "reason": "the Gmail connector is not enabled for this source"}
        try:
            profile = self.api.profile()
        except Exception as error:
            reason = _reason_of(error)
            return {"ok": False, "content_read": False,
                    "reason": f"the account did not answer: {reason}", **revocation(reason)}
        stated = profile if isinstance(profile, dict) else {}
        scopes = [str(scope)[:120] for scope in (stated.get("scopes") or [])][:_MAX_LIST]
        return {"ok": True, "content_read": False, "observed_at": now(),
                "address": _short(stated.get("address"), 200) or None,
                "query": self.query, "scopes": scopes or None,
                "readonly": "https://mail.google.com/" in scopes or None}

    def read_page(self, cursor: str | None) -> Page:
        """One page: a query page while catching up, then everything the history says."""
        self._guard()
        limit = self.capabilities.max_records_per_page
        answer = self._read(cursor, limit)
        stated = answer if isinstance(answer, dict) else {}
        entries = _list_of(stated.get("items"))
        envelopes: list[dict[str, Any]] = []
        skipped: list[Skipped] = []
        seen: set[str] = set()
        used = 0
        for index, entry in enumerate(entries[:limit * 4]):
            reference, envelope, reason = self._message(entry, index)
            if envelope is None:
                skipped.append(Skipped(reference, reason))
                continue
            size = len(envelope["text"].encode("utf-8"))
            if envelopes and (len(envelopes) >= limit
                              or used + size > self.capabilities.max_bytes_per_page):
                skipped.append(Skipped(
                    reference, "the page was full before this message; it is read again"))
                break
            if envelope["source_id"] in seen:
                skipped.append(Skipped(reference, "the mailbox offered this message twice"))
                continue
            seen.add(envelope["source_id"])
            used += size
            envelopes.append(envelope)
        for removed in _list_of(stated.get("removed"))[:limit]:
            skipped.append(Skipped(_short(removed) or "unknown",
                                   "the message left the mailbox; this connector reports "
                                   "the loss and does not propagate a deletion"))
        head = _short(stated.get("history_id"), 200)
        token = _short(stated.get("next_page_token"), 2000)
        if token:
            next_cursor = _PAGE + token
        elif head and (envelopes or cursor is None):
            next_cursor = _HISTORY + head
        else:
            # Caught up: nothing arrived since the stored head, and there is no point
            # claiming a position that has not moved.
            next_cursor = None
        return Page(envelopes=tuple(envelopes), skipped=tuple(skipped),
                    next_cursor=next_cursor)

    # -- reading -------------------------------------------------------------

    def _guard(self) -> None:
        if not self.enabled:
            raise EvidenceError(
                "the Gmail connector is not enabled for this source; enabling one is an "
                "owner action and nothing in a connector does it on its own")

    def _read(self, cursor: str | None, limit: int) -> Any:
        """Ask for the next page *of the position the connector holds*.

        A stored position this connector never wrote is a different stream — an older
        configuration, or a cursor from a source that was replaced — and saying so is
        what lets the runtime restart the reading rather than keep asking a question
        that will always be answered the same way.
        """
        try:
            if cursor is None:
                return self.api.messages(query=self.query, page_token=None, max_results=limit)
            if cursor.startswith(_PAGE):
                return self.api.messages(query=self.query, page_token=cursor[len(_PAGE):],
                                         max_results=limit)
            if cursor.startswith(_HISTORY):
                return self.api.history(since=cursor[len(_HISTORY):], query=self.query)
        except Exception as error:
            raise EvidenceError(f"the account did not answer: {_reason_of(error)}") from None
        raise CursorExpired(f"the stored position {cursor[:60]!r} is not one this connector writes")

    def _message(self, entry: Any, index: int) -> tuple[str, dict[str, Any] | None, str]:
        stated = entry if isinstance(entry, dict) else {}
        identifier = _short(stated.get("id"), 200)
        if identifier is None:
            return f"gmail#{index}", None, "the mailbox listed an entry with no id"
        reference = f"gmail:{identifier}"
        raw, reason = _bytes_of(stated.get("raw"))
        if raw is None and stated.get("raw") is not None:
            return reference, None, f"the raw message could not be decoded: {reason}"
        if raw is None:
            # The listing gave an id and no body, so the body has to be asked for:
            # one more call per message, and the quota consequence is the point of
            # naming it here rather than discovering it in a bill.
            try:
                fetched = self.api.message(identifier)
            except Exception as error:
                return reference, None, f"the message could not be fetched: {_reason_of(error)}"
            raw, reason = _bytes_of((fetched or {}).get("raw") if isinstance(fetched, dict)
                                    else fetched)
            if raw is None:
                return reference, None, f"the message carried no readable body: {reason}"
            stated = fetched if isinstance(fetched, dict) else stated
        if len(raw) > MAX_MESSAGE_BYTES:
            return reference, None, (
                f"{len(raw)} bytes exceeds the {MAX_MESSAGE_BYTES} byte per-message bound")
        envelope, reason = parse_message(self.source, reference, raw)
        if envelope is None:
            return reference, None, reason
        envelope["metadata"].update({
            "path": reference, "gmail_id": identifier,
            "thread_id": _short(stated.get("threadId"), 200) or None,
            "received_at": _epoch_of(stated.get("internalDate")),
        })
        return reference, envelope, ""


_PAGE = "p:"
_HISTORY = "h:"


def _bytes_of(value: Any) -> tuple[bytes | None, str]:
    """A base64url message body, as sent.

    Gmail pads its base64url irregularly, so the padding is completed rather than
    the value refused; anything that still will not decode is a gap, not an empty
    message.
    """
    if value is None:
        return None, "absent"
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), ""
    if not isinstance(value, str) or not value.strip():
        return None, "it was not text"
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)), ""
    except (ValueError, TypeError) as error:
        return None, str(error)[:120] or "undecodable"


def _epoch_of(value: Any) -> str | None:
    """Milliseconds since the epoch, as the absolute instant it claims to be."""
    try:
        seconds = int(str(value)) / 1000
    except (TypeError, ValueError):
        return None
    if not 0 <= seconds < 4_102_444_800:
        return None
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def _list_of(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _short(value: Any, maximum: int = 200) -> str | None:
    if not isinstance(value, (str, int)):
        return None
    text = str(value).strip()
    return text[:maximum] or None


def _reason_of(error: BaseException) -> str:
    return " ".join(str(error).split())[:400] or type(error).__name__


# -- normalization helpers ---------------------------------------------------

def parse_message(source: str, ref: str, raw: bytes) -> tuple[dict[str, Any] | None, str]:
    """One RFC 2822 message into an envelope, or the reason it could not become one.

    This is the whole of email normalisation, shared by the export adapter and the
    live connector: a message that arrives both in a directory somebody backed up and
    in a mailbox still has to produce the same text, the same participants and the
    same provenance, or the two readings of it cannot be recognised as one thing.
    """
    try:
        # Bytes rather than a decoded string: the character set belongs to each part,
        # and text already forced through one character set has lost what it said.
        message = email.message_from_bytes(raw, policy=email.policy.default)
    except (ValueError, email.errors.MessageError) as error:
        return None, f"malformed export: {str(error)[:200]}"
    message_id = str(message.get("Message-ID") or "").strip("<> \t")
    if not message_id:
        # A message with no identity cannot be re-read without duplicating it, and an
        # id invented from its content would make an edited copy of the same
        # conversation look like a second event.
        return None, "no Message-ID header, so the message cannot be identified"
    body, part_type = _body_of(message)
    if body is None:
        return None, "no text part could be decoded"
    if not body.strip():
        return None, "the text part is empty"
    if _REPLACEMENT in body:
        # A part that did not survive its own declared character set is a gap. Keeping
        # the damaged text would let it be read back as if it had been understood, and
        # shown to a model as evidence.
        return None, f"the {part_type} part could not be decoded as its " \
            "declared character set"
    occurred, precision, note = _date_of(message)
    quoted = len(_QUOTED.findall(body))
    return {
        "source": source,
        "source_id": message_id,
        "revision": "1",
        "kind": "email",
        "text": body[:_MAX_TEXT_CHARS],
        "observed_at": now(),
        "occurred_at": occurred,
        "occurred_precision": precision,
        "metadata": {
            "path": ref, "message_id": message_id,
            "subject": str(message.get("Subject") or "")[:500] or None,
            "from": _person(message.get("From")),
            "participants": _people(message.get("To"), message.get("Cc")),
            "reply_to": _bare(str(message.get("In-Reply-To") or "")) or None,
            "references": [_bare(value) for value in
                           str(message.get("References") or "").split()
                           if _bare(value)][-_MAX_LIST:] or None,
            "forwarded": True if _FORWARD_HEADER.search(body) else None,
            "quoted_lines": quoted or None,
            "media_type": part_type,
            "attachments": attachment_descriptors(message) or None,
            "relays": _relays(message.get_all("Received")),
            "bytes": len(raw),
            "truncated": True if len(body) > _MAX_TEXT_CHARS else None,
            **({"time_note": note} if note else {}),
        },
    }, ""



def _body_of(message: Any) -> tuple[str | None, str]:
    """The plain text of a message, or the HTML of it with the markup taken off.

    A part that cannot be decoded is missing, not raw: undecodable bytes kept in a
    record would be shown to a model later as if they had been understood.
    """
    try:
        part = message.get_body(preferencelist=("plain", "html"))
    except (AttributeError, TypeError):
        return None, ""
    if part is None:
        return None, ""
    try:
        content = part.get_content()
    except Exception:
        return None, ""
    if not isinstance(content, str):
        return None, ""
    kind = f"{part.get_content_type()}/{part.get_content_charset() or 'unknown'}"
    if part.get_content_type() == "text/html":
        content = _strip_html(content)
    return _without_signature(content), kind


def _strip_html(text: str) -> str:
    stripped = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text,
                      flags=re.IGNORECASE | re.DOTALL)
    stripped = re.sub(r"<br\s*/?>|</p>", "\n", stripped, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"<[^>]+>", "", stripped))


def _without_signature(text: str) -> str:
    cut = _SIGNATURE.search(text)
    return text if cut is None else text[:cut.start()]


def _date_of(message: Any) -> tuple[str | None, str, str | None]:
    header = str(message.get("Date") or "").strip()
    if not header:
        return None, "unknown", "no Date header"
    try:
        parsed = parsedate_to_datetime(header)
    except (TypeError, ValueError):
        return None, "unknown", "unparsable Date header"
    if parsed.tzinfo is None:
        # The export gave a clock reading and not which clock. Guessing the local
        # zone would file the message under the wrong day for anyone who has ever
        # crossed one, and guessing UTC would be the same guess wearing a label.
        return None, "unknown", "Date header carried no timezone offset"
    return parsed.astimezone(timezone.utc).isoformat(), "second", None


def _person(value: Any) -> dict[str, str | None] | None:
    if not value:
        return None
    name, address = parseaddr(str(value))
    if not address:
        return None
    return {"name": name or None, "address": address.lower()}


def _people(*values: Any) -> list[dict[str, str | None]]:
    out: list[dict[str, str | None]] = []
    for name, address in getaddresses([str(value) for value in values if value]):
        if address:
            out.append({"name": name or None, "address": address.lower()})
    return out[:_MAX_LIST]


def _bare(value: str) -> str | None:
    match = re.search(r"<([^>]+)>", value)
    return (match.group(1) if match else value).strip() or None


def _relays(received: Any) -> int:
    """How many relays the message claims to have passed through.

    Kept as a count because it is the only cheap statement of how far the message
    travelled before it reached this mailbox.
    """
    return len(received) if isinstance(received, list) else (1 if received else 0)


def attachment_descriptors(message: Any) -> list[dict[str, Any]]:
    """Name every attachment without ingesting it.

    The bound is recorded rather than applied silently: an operator asking why a
    file is not in the store should get an answer out of the record.
    """
    found = []
    for part in message.walk():
        if part.is_multipart():
            continue
        disposition = str(part.get("Content-Disposition") or "")
        name = part.get_filename()
        if not name and "attachment" not in disposition.lower():
            continue
        payload = part.get_payload(decode=True) or b""
        found.append({"name": str(name or "")[:200], "media_type": str(part.get_content_type()),
                      "bytes": len(payload), "inline": "inline" in disposition.lower(),
                      "ingested": False})
    return found[:_MAX_LIST]
