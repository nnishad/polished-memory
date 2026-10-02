"""C2 MCP adapter: one operator-named thing on a server, and nothing else.

An MCP server arrives with its own opinion about what is worth reading. So this
adapter is built around exactly one *mapping*, written by the operator, and it never
asks the server what else it can do: no tool listing, no resource browsing, no prompt
execution, no sampling on somebody else's behalf. Discovery is what turns a mapped
integration into a channel a third party can steer, and the job of a connector is to
read a known thing, not to find out what exists.

Credentials are deliberately not this module's business either: the client the
operator wires up is already authorised for the one target in the mapping, so there is
no broad credential here to inherit, to hand on, or to leak into a record. A record
gets the declared fields plus the *names* of the ones that were dropped — so "the
server sent a token and we did not store it" is answerable from the evidence itself.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Mapping

from ..ids import digest, now
from ..storage.evidence import EvidenceError
from .base import (Capabilities, CursorExpired, Page, Skipped, SourceAdapter,
                   normalize_text, normalize_time, revocation)

__all__ = ["McpSource", "SourceMap", "MAX_MCP_TEXT_CHARS"]

MAX_MCP_TEXT_CHARS = 50_000
_MAX_ITEMS = 1000
_MAX_DROPPED = 20
_CHANNELS = ("tool", "resource")


@dataclass(frozen=True)
class SourceMap:
    """One operator decision: this named thing on that server is this kind of evidence.

    Every field is a declaration rather than a guess. The map says which key holds an
    identity and which holds the text, because assuming either would put a server's
    choice of column names into the identity space of canonical evidence.
    """

    name: str
    target: str
    channel: str = "tool"
    record_id: str = "id"
    text: str = "text"
    time: str | None = None
    revision: str | None = None
    kind: str = "message"
    scope: str = "operator-declared"

    def __post_init__(self) -> None:
        for name in ("name", "target", "channel", "record_id", "text", "kind", "scope"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EvidenceError(f"source map field {name!r} must be nonempty text")
            object.__setattr__(self, name, value.strip()[:200])
        if self.channel not in _CHANNELS:
            raise EvidenceError(
                f"a source map reads a {' or '.join(_CHANNELS)} channel, not {self.channel!r}")
        for name in ("time", "revision"):
            value = getattr(self, name)
            if value in (None, ""):
                object.__setattr__(self, name, None)
                continue
            if not isinstance(value, str) or not value.strip():
                raise EvidenceError(f"source map field {name!r} is a field name or nothing")
            object.__setattr__(self, name, value.strip()[:200])

    @classmethod
    def from_mapping(cls, declared: Mapping[str, Any]) -> "SourceMap":
        """Read the operator's declaration, refusing to guess at anything missing.

        An unknown key is refused rather than ignored: a map that spells ``time`` as
        ``timestamp`` would otherwise load cleanly and drop every date in silence.
        """
        if not isinstance(declared, Mapping):
            raise EvidenceError("a source map is an object of declared fields")
        given = {str(key): value for key, value in declared.items()}
        named = [name for name in _CHANNELS + ("prompt",) if given.get(name)]
        if len(named) > 1:
            raise EvidenceError("a source map names one channel: a tool or a resource")
        if not named:
            raise EvidenceError("a source map must name the tool or resource it reads")
        if named == ["prompt"]:
            # A prompt is an instruction to a model. Filed as evidence it would be
            # retrieved later as though somebody had said it, and re-offered to a
            # model as the past rather than as a request.
            raise EvidenceError("a prompt is an instruction to a model, not evidence")
        channel = named[0]
        target = str(given.pop(channel)).strip()
        if not target:
            raise EvidenceError(f"the {channel} in this source map has no name")
        if "id" in given:
            # Operators write ``id``; the field is ``record_id`` because a map has an
            # id of its own. One spelling per thing, so a map cannot mean two.
            given["record_id"] = given.pop("id")
        given["name"] = str(given.get("name") or target)
        given["target"] = target
        given["channel"] = channel
        unknown = set(given) - _ADMISSIBLE
        if unknown:
            raise EvidenceError(
                f"unknown source map fields {sorted(unknown)}; a map declares "
                f"{sorted(_ADMISSIBLE)}")
        try:
            return cls(**given)
        except TypeError as error:
            raise EvidenceError(f"the source map cannot be read: {error}") from None


class McpSource(SourceAdapter):
    """Read one mapped tool or resource through a client the operator authorised."""

    capabilities = Capabilities(history=True, live=False, deletion_events=False,
                                revision_history=False, max_records_per_page=100,
                                max_bytes_per_page=4_000_000)

    def __init__(self, client: Any, mapping: Mapping[str, Any] | SourceMap, *,
                 source: str | None = None, live: bool = False,
                 records_per_page: int | None = None):
        subject = (mapping if isinstance(mapping, SourceMap)
                   else SourceMap.from_mapping(mapping))
        if client is None:
            raise EvidenceError("a source map needs the client its server is reached through")
        method = "call_tool" if subject.channel == "tool" else "read_resource"
        if not callable(getattr(client, method, None)):
            raise EvidenceError(
                f"this client cannot serve a {subject.channel} map: it has no {method}()")
        self.map = subject
        self._client = client
        self.source = source or f"mcp-{subject.name}"
        if records_per_page is not None and not 1 <= records_per_page <= 1000:
            raise ValueError("records_per_page must be between 1 and 1000")
        self.capabilities = replace(
            self.capabilities, live=live,
            **({"max_records_per_page": records_per_page} if records_per_page else {}))

    # -- contract ------------------------------------------------------------

    def check(self) -> dict[str, Any]:
        """What this map authorises, without spending a read to find out.

        Asking a server what it has is the discovery this adapter exists to refuse, so
        an outage is reported by the first read rather than predicted here. A client
        that can report its own standing is asked for that, and for nothing else.
        """
        report: dict[str, Any] = {"ok": True, "channel": self.map.channel,
                                  "target": self.map.target, "scope": self.map.scope,
                                  "content_read": False, "observed_at": now()}
        status = getattr(self._client, "status", None)
        if not callable(status):
            report["note"] = "the client reports no standing; the first read will tell"
            return report
        try:
            said = status()
        except Exception as error:
            return {"ok": False, "content_read": False,
                    "reason": f"the server would not say: {_reason(error)}",
                    **revocation(_reason(error))}
        if not isinstance(said, Mapping):
            report["note"] = "the client's status was not a report; the first read will tell"
            return report
        authorized = said.get("authorized")
        reason = str(said.get("reason") or "")[:500]
        if authorized is False:
            report.update({"ok": False,
                           "reason": reason or "the server says this grant is not active"})
            report.update(revocation(report["reason"]))
        elif authorized is not None:
            report["authorized"] = True
        return report

    def read_page(self, cursor: str | None) -> Page:
        import json
        remote, offset, expected = cursor, 0, None
        prefix = "mcp-page-v1:"
        if cursor and cursor.startswith(prefix):
            try:
                remote, offset, expected = json.loads(cursor[len(prefix):])
                if not isinstance(offset, int) or offset < 0:
                    raise ValueError()
            except (ValueError, TypeError):
                raise CursorExpired("invalid MCP page cursor") from None
        answer = self._ask(remote)
        items = _items_of(answer)
        stamp = digest(json.dumps(items, sort_keys=True, default=lambda value: {
            "bytes_hex": value.hex()} if isinstance(value, bytes) else repr(value)))
        if expected is not None and expected != stamp:
            raise CursorExpired("MCP page changed while resuming; restart the source scan")
        position = _position_of(answer)
        envelopes: list[dict[str, Any]] = []
        skipped: list[Skipped] = []
        seen: set[str] = set()
        used = 0
        for index, item in enumerate(items[:_MAX_ITEMS]):
            reference, produced, reason = self._one(item, index, seen)
            if produced is None:
                skipped.append(Skipped(reference, reason))
                continue
            if index < offset:
                continue
            size = len(json.dumps(produced, ensure_ascii=False).encode("utf-8"))
            if size > self.capabilities.max_bytes_per_page:
                skipped.append(Skipped(reference, "record exceeds the page byte bound"))
                continue
            if envelopes and (len(envelopes) >= self.capabilities.max_records_per_page
                              or used + size > self.capabilities.max_bytes_per_page):
                # Server cursors advance past whole responses. Replay the same
                # response with a content-bound local offset until its tail is consumed.
                return Page(envelopes=tuple(envelopes), skipped=tuple(skipped),
                            next_cursor=prefix + json.dumps([remote, index, stamp], separators=(",", ":")))
            used += size
            envelopes.append(produced)
        if len(items) > _MAX_ITEMS:
            skipped.append(Skipped(f"{self.map.name}#beyond-{_MAX_ITEMS}",
                                   f"the server offered {len(items)} items and one page "
                                   f"takes at most {_MAX_ITEMS}"))
        return Page(envelopes=tuple(envelopes), skipped=tuple(skipped), next_cursor=position)

    # -- one item ------------------------------------------------------------

    def _one(self, item: Any, index: int, seen: set[str]):
        if not isinstance(item, Mapping):
            return f"{self.map.name}#{index}", None, "the server yielded a non-object"
        identifier = normalize_text(item.get(self.map.record_id))
        if identifier is None:
            # Without the server's own key nothing can notice this item again: a
            # position in a list is not an identity, and one insertion upstream would
            # turn every later item into a different fact.
            return (f"{self.map.name}#{index}", None,
                    f"the item carries no {self.map.record_id!r} to identify it by")
        reference = f"{self.map.name}:{identifier[:180]}"
        revision = normalize_text(item.get(self.map.revision)) if self.map.revision else None
        # The pair is the thing that was offered twice, not the id alone: a versioned
        # resource legitimately repeats its id, and refusing the second copy would
        # leave a document frozen at whatever revision arrived first.
        if (identifier, revision) in seen:
            return reference, None, "the server offered this revision twice in one answer"
        body = normalize_text(item.get(self.map.text))
        if body is None:
            seen.add((identifier, revision))
            return reference, None, f"the item's {self.map.text!r} field holds no readable text"
        seen.add((identifier, revision))
        occurred, precision, note = normalize_time(
            item.get(self.map.time) if self.map.time else None)
        dropped = sorted(str(key) for key in item if key not in self._declared)
        return reference, self.envelope(
            source_id=identifier[:500], revision=(revision or "1")[:500], kind=self.map.kind,
            text=body[:MAX_MCP_TEXT_CHARS], observed_at=now(), occurred_at=occurred,
            occurred_precision=precision,
            metadata={
                "map": self.map.name, "mcp_channel": self.map.channel,
                "mcp_target": self.map.target, "scope": self.map.scope,
                "time_basis": self.map.time if (self.map.time and occurred) else "none",
                **({"time_note": note} if note else {}),
                "text_digest": digest(["v1", body])[:32],
                "chars": len(body[:MAX_MCP_TEXT_CHARS]),
                **({"truncated": True} if len(body) > MAX_MCP_TEXT_CHARS else {}),
                **({"undeclared_fields_dropped": dropped[:_MAX_DROPPED]} if dropped else {}),
            },
        ), ""

    @property
    def _declared(self) -> frozenset:
        names = {self.map.record_id, self.map.text}
        names.update(name for name in (self.map.time, self.map.revision) if name)
        return frozenset(names)

    def _ask(self, cursor: str | None) -> Any:
        """One call, at the mapped target, with nothing but the paging arguments."""
        try:
            if self.map.channel == "tool":
                return self._client.call_tool(
                    self.map.target,
                    {"limit": self.capabilities.max_records_per_page,
                     **({"after": cursor} if cursor is not None else {})})
            if cursor is not None:
                # A resource is one document. Resuming it is not a thing this adapter
                # can pretend to do, and silently re-reading it would look like progress.
                raise CursorExpired("a resource has no position to resume from")
            return self._client.read_resource(self.map.target)
        except (CursorExpired, EvidenceError):
            raise
        except Exception as error:
            raise EvidenceError(f"the MCP server did not answer: {_reason(error)}") from None


def _items_of(answer: Any) -> list[Any]:
    if not isinstance(answer, Mapping):
        raise EvidenceError("the server's answer is not an object items can be read from")
    items = answer.get("items")
    if items is None and "contents" in answer:
        # ``resources/read`` answers with contents. Same shape under the protocol's
        # own other name, and refusing it would leave a mapped resource unreadable.
        items = answer.get("contents")
    if items is None:
        raise EvidenceError("the server's answer carries no items")
    if isinstance(items, (str, bytes)) or not isinstance(items, list):
        raise EvidenceError("the server's items must be a list of objects")
    return items


def _position_of(answer: Mapping[str, Any]) -> str | None:
    raw = answer.get("next_cursor", answer.get("nextCursor"))
    if raw in (None, ""):
        return None
    if not isinstance(raw, (str, int)):
        raise EvidenceError("a cursor has to be something the server will accept back")
    return str(raw)[:2000]


def _reason(error: BaseException) -> str:
    return " ".join(str(error).split())[:400] or type(error).__name__


_ADMISSIBLE = frozenset(field.name for field in fields(SourceMap))
