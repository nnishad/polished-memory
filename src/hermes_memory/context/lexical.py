"""C9 lexical channel: local full-text retrieval, no model and no network.

This is the channel that still answers while Hindsight is stopped, inference is
paused and the budget is spent. It is deliberately boring: a query becomes a
conjunction of quoted terms, the store does the rest.

The quoting matters. FTS5 parses its argument, so a question containing a stray
quote or the word AND is a syntax error, and a syntax error surfaces as "no
matches" unless it is caught here — which would turn a punctuation character
into a false claim that the archive is empty.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["fts_expression", "LexicalOutcome", "LexicalChannel", "AVAILABLE", "UNAVAILABLE"]

AVAILABLE = "available"
UNAVAILABLE = "unavailable"

_TOKEN = re.compile(r"[^\W]+", re.UNICODE)
# An over-long query is a pasted document, not a question. Matching on the first
# terms of it still answers; matching on all of them would match nothing.
_MAX_TERMS = 24


@dataclass(frozen=True)
class LexicalOutcome:
    state: str
    items: list = field(default_factory=list)
    detail: str = ""
    dropped_terms: int = 0

    @property
    def available(self) -> bool:
        return self.state == AVAILABLE


def fts_expression(query: str) -> str:
    """Turn free text into a safe FTS5 conjunction of quoted terms, or '' if unmatchable."""
    terms: list[str] = []
    for token in _TOKEN.findall(query or ""):
        quoted = '"' + token.replace('"', '""') + '"'
        if quoted not in terms:
            terms.append(quoted)
        if len(terms) >= _MAX_TERMS:
            break
    return " ".join(terms)


class LexicalChannel:
    """Search committed, visible evidence by term."""

    def __init__(self, store):
        self.store = store

    def gather(self, query: str, *, limit: int) -> LexicalOutcome:
        """Empty hands and an outage are different answers."""
        terms = _TOKEN.findall(query or "")
        match = fts_expression(query)
        if not match:
            return LexicalOutcome(AVAILABLE)
        try:
            found = self.store.search(match, limit=limit)
        except Exception as error:  # an unreadable index is an outage, not zero hits
            return LexicalOutcome(UNAVAILABLE,
                                  detail=f"{type(error).__name__}: {error}"[:200])
        return LexicalOutcome(AVAILABLE, items=found,
                              dropped_terms=max(0, len(dict.fromkeys(terms)) - _MAX_TERMS))
