"""A source adapter that answers from a script, for whoever needs to prove a fence holds.

The connector runtime, the sync controller and the operator's ``pause`` door all have to
watch a source that lies, runs short, or refuses at a chosen moment. That is the same
fake in three suites, so it lives here: a copy per suite would mean each one testing a
slightly different promise about paging, and the real adapters obey only one.
"""
from __future__ import annotations

from conftest import envelope
from hermes_memory.ids import timestamp
from hermes_memory.sources.base import Capabilities, Page, SourceAdapter
from hermes_memory.storage.evidence import EvidenceError

SOURCE = "gmail"
OTHER = "imail"
AT = timestamp("2026-09-15T09:00:00+00:00")


class Scripted(SourceAdapter):
    """A source that answers from a script and counts the times it was asked."""

    def __init__(self, pages, *, source=SOURCE, live=False, reachable=True, on_read=None,
                 fails=None):
        self.pages = list(pages)
        self.source = source
        self.capabilities = Capabilities(history=True, live=live, max_records_per_page=5)
        self.reachable = reachable
        self.on_read = on_read
        self.fails = set(fails or ())
        self.reads = 0

    def check(self):
        if isinstance(self.reachable, Exception):
            raise self.reachable
        return ({"ok": True, "content_read": False} if self.reachable
                else {"ok": False, "reason": "token revoked"})

    def read_page(self, cursor):
        if self.on_read is not None:
            self.on_read(self.reads)
        if self.reads in self.fails:
            raise EvidenceError("the source refused this read")
        index = self.reads
        page = self.pages[index] if index < len(self.pages) else Page()
        self.reads += 1
        return page


def notes(count, *, start=0, prefix="msg", source=SOURCE, next_cursor=None,
          skipped=(), token=None):
    return Page(
        envelopes=tuple(envelope(source=source, source_id=f"{prefix}-{start + index}",
                                 text=f"Note {start + index}.", observed_at=AT)
                        for index in range(count)),
        next_cursor=next_cursor, skipped=tuple(skipped), page_token=token)
