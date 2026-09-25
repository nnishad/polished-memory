"""C3 connector runtime: read a source, commit it, and know when to stop.

Fetching upstream and forming downstream are kept apart on purpose. This module
does the first half only: it takes a lease, reads bounded pages, and hands each
one to the sync controller, which commits the page and moves the cursor in a single
transaction. Everything downstream — projections, summaries, the attention gate —
replays the change journal instead, so a backend that is offline or a stage the
operator paused never holds the cursor back and never loses what arrives.

The pause is honoured here rather than deep inside an adapter, because the answer
has to be the same one the record gives: a source the operator stopped is not
read, and the run says so in the log rather than quietly fetching anyway.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from ..ids import digest
from ..storage.evidence import EvidenceError
from .base import CursorExpired, Page, SourceAdapter
from .sync import StaleFence, SyncController

__all__ = ["ConnectorRuntime", "Run", "COMPLETE", "PAUSED", "STALE", "UNREACHABLE",
           "EXHAUSTED", "CONTENDED", "STALLED", "CURSOR_EXPIRED"]

CURSOR_EXPIRED = "cursor_expired"


class _Unreadable(EvidenceError):
    """The source itself could not be talked to."""

DEFAULT_LEASE_S = 120.0
DEFAULT_MAX_PAGES = 200

# Why a pass ended. Each is a different answer to "should this be tried again":
# a paused source waits for the operator, a stale one waits for a new lease, and
# a stalled one is a source that is not making progress on its own.
COMPLETE = "complete"
PAUSED = "paused"
STALE = "stale"
UNREACHABLE = "unreachable"
EXHAUSTED = "exhausted"
CONTENDED = "contended"
STALLED = "stalled"


@dataclass(frozen=True)
class Run:
    """One connector pass, described by what it committed and how it ended."""

    source: str
    generation: int
    pages: int
    records: int
    repeats: int
    gaps: int
    cursor: str | None
    coverage_state: str
    stopped: str
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "generation": self.generation, "pages": self.pages,
                "records": self.records, "repeats": self.repeats, "gaps": self.gaps,
                "cursor": self.cursor, "coverage_state": self.coverage_state,
                "stopped": self.stopped, "note": self.note}


class ConnectorRuntime:
    """Drive one adapter under one lease."""

    def __init__(self, store, sync: SyncController | None = None, *, holder: str,
                 lease_s: float = DEFAULT_LEASE_S, clock: Callable[[], float] = None):
        if not isinstance(holder, str) or not holder.strip() or len(holder) > 200:
            raise EvidenceError("holder must be nonempty text naming the worker")
        if not isinstance(lease_s, (int, float)) or not 1 <= lease_s <= 3600:
            raise EvidenceError("lease_s must be between 1 and 3600 seconds")
        self.store = store
        self.clock = clock or time.time
        self.sync = sync or SyncController(store, clock=self.clock)
        self.holder = holder
        self.lease_s = float(lease_s)

    def run(self, adapter: SourceAdapter, *, max_pages: int = DEFAULT_MAX_PAGES) -> Run:
        """Read *adapter* from where the connector stopped and commit each page.

        Returns rather than raises when the source cannot be read: a poller that
        crashes on a revoked token and a poller that reports one want different
        fixes, and only the second is diagnosable from the record.
        """
        source = getattr(adapter, "source", None)
        if not isinstance(source, str) or not source.strip():
            raise EvidenceError("an adapter must declare the source it reads")
        limit = max_pages if isinstance(max_pages, int) and 1 <= max_pages <= 10_000 \
            else DEFAULT_MAX_PAGES
        if self.store.stage_is_paused(source, "capture"):
            return self._report(source, self.sync.state(source), stopped=PAUSED,
                                note="capture is paused for this source; nothing was read")
        state = self.sync.state(source)
        if state["lease_active"] and state["holder"] not in (None, self.holder):
            return self._report(source, state, stopped=CONTENDED,
                                note=f"the lease is held by {state['holder']!r}")

        fence = self.sync.acquire(source, holder=self.holder, ttl=self.lease_s)
        cursor = self.sync.state(source)["cursor"]
        pages = records = repeats = gaps = 0
        stopped, note = COMPLETE, ""
        try:
            reach = self._ask(adapter)
            if not (isinstance(reach, dict) and reach.get("ok")):
                reason = _reason_of(reach)
                self._mark(fence, "unreachable", reason)
                return self._report(source, self.sync.state(source), pages=pages,
                                    stopped=UNREACHABLE, note=reason)
            while pages < limit:
                self.sync.renew(fence, ttl=self.lease_s)
                page = self._read(adapter, cursor)
                outcome = self.sync.publish(
                    fence, self._token(fence, page), page.envelopes,
                    next_cursor=page.next_cursor, skipped=page.skipped)
                pages += 1
                records += int(outcome["new"])
                repeats += int(outcome["duplicate"])
                gaps += len(page.skipped)
                if page.next_cursor is None:
                    break
                if page.next_cursor == cursor and not outcome["new"]:
                    # Nothing arrived and the position did not move: repeating the
                    # read would only spend the source's quota. A live source whose
                    # head is a stable label is expected to stop here, and the next
                    # poll is a new pass.
                    self._mark(fence, "partial",
                               "the source re-served the cursor it was given and had "
                               "nothing new")
                    stopped, note = STALLED, "the page cursor did not move"
                    break
                cursor = page.next_cursor
            else:
                stopped = EXHAUSTED
                note = (f"the page budget ({limit}) ran out; the cursor is where the "
                        "next pass continues")
        except CursorExpired as error:
            self._restart(fence, str(error))
            stopped, note = CURSOR_EXPIRED, str(error)[:500]
        except StaleFence as error:
            stopped, note = STALE, str(error)[:500]
        except _Unreadable as error:
            self._mark(fence, "unreachable", str(error))
            stopped, note = UNREACHABLE, str(error)[:500]
        finally:
            self.sync.release(fence)
        return self._report(source, self.sync.state(source), pages=pages, records=records,
                            repeats=repeats, gaps=gaps, stopped=stopped, note=note)

    # -- reading -------------------------------------------------------------

    def _ask(self, adapter: SourceAdapter) -> Any:
        """The reachability probe, as an answer or as an outage."""
        try:
            return adapter.check()
        except CursorExpired:
            raise
        except EvidenceError as error:
            raise _Unreadable(str(error)) from None

    def _read(self, adapter: SourceAdapter, cursor: str | None) -> Page:
        """One page from the source.

        A read that fails is the source not answering, which is a coverage state. A
        commit that refuses is a contradiction between what the adapter declared and
        what it handed over, and that goes on to the caller rather than being filed
        as an outage: no amount of retrying a broken adapter makes it honest.
        """
        try:
            return adapter.read_page(cursor)
        except CursorExpired:
            raise
        except EvidenceError as error:
            raise _Unreadable(str(error)) from None

    # -- page identity -------------------------------------------------------

    def _token(self, fence, page: Page) -> str:
        """Name the page by what it holds, so a replay is recognised and re-read cheap.

        Keying on the *position* looks reasonable and wedges: a source that is asked
        for the same cursor after something new arrived — the tail of a mailbox, the
        file that was skipped last time — would be reported as having contradicted
        itself, and the connector could never get past it. An adapter that can name
        its own page (a history range, a page token from the API) is believed over
        this, because then the source really does hold the identity, and a different
        set of bytes under that name is a contradiction worth refusing.
        """
        if page.page_token:
            return str(page.page_token)[:500]
        rows = [[item.get("source_id"), item.get("revision")] for item in page.envelopes]
        return "content-" + digest([fence.source, fence.generation, rows])[:32]

    # -- reporting -----------------------------------------------------------

    def _restart(self, fence, reason: str) -> None:
        """Forget the position the source refused, and say which one that was."""
        try:
            self.sync.restart(fence, reason=reason)
        except StaleFence:
            pass

    def _mark(self, fence, coverage_state: str, reason: str) -> None:
        """Record coverage unless the lease has already passed to someone else."""
        try:
            self.sync.mark_gap(fence, coverage_state=coverage_state, reason=reason)
        except StaleFence:
            pass

    def _report(self, source: str, state: dict[str, Any], *, pages: int = 0,
                records: int = 0, repeats: int = 0, gaps: int = 0, stopped: str = COMPLETE,
                note: str = "") -> Run:
        return Run(source=source, generation=int(state["generation"]), pages=pages,
                   records=records, repeats=repeats, gaps=gaps,
                   cursor=state["cursor"], coverage_state=state["coverage_state"],
                   stopped=stopped, note=" ".join(str(note).split())[:500])


def _reason_of(reach: Any) -> str:
    """What the source said about itself, or the fact that it said nothing."""
    if isinstance(reach, dict):
        return " ".join(str(reach.get("reason") or "the source reported no reason").split())[:500]
    return f"the source check returned {reach!r}"[:500]
