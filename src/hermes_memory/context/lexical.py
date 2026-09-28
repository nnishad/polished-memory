"""C9 lexical channel: local full-text retrieval, no model and no network.

This is the channel that still answers while Hindsight is stopped, inference is
paused and the budget is spent. It is deliberately boring: a query becomes a
conjunction of quoted terms — which is the store's business, in
``storage.evidence`` — and the channel only says whether the index answered and
what it answered with.

A syntax error is caught here rather than allowed to escape, because "the archive
has nothing" and "the archive could not be read" are different claims to make
about somebody's memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..storage.evidence import MAX_TERMS, fts_terms

__all__ = ["LexicalOutcome", "LexicalChannel", "AVAILABLE", "UNAVAILABLE"]

AVAILABLE = "available"
UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class LexicalOutcome:
    state: str
    items: list = field(default_factory=list)
    detail: str = ""
    dropped_terms: int = 0


class LexicalChannel:
    """Search committed, visible evidence by term."""

    def __init__(self, store):
        self.store = store

    def gather(self, query: str, *, limit: int) -> LexicalOutcome:
        """Empty hands and an outage are different answers."""
        terms = fts_terms(query)
        try:
            found = self.store.search(query, limit=limit)
        except Exception as error:  # an unreadable index is an outage, not zero hits
            return LexicalOutcome(UNAVAILABLE,
                                  detail=f"{type(error).__name__}: {error}"[:200])
        return LexicalOutcome(AVAILABLE, items=found,
                              dropped_terms=max(0, len(terms) - MAX_TERMS))
