"""C9 packet cache: bounded, short-lived, and invalidated by the store itself.

Re-running a retrieval for every turn of one conversation is how a memory
system becomes the slowest thing in the loop. Caching it is how a forgotten
secret gets re-served twenty turns later. Both halves are handled here: an
entry carries the epoch and journal watermark it was assembled under, and a
lookup that cannot reproduce that stamp misses. A reset, a erasure, a hide or a
correction therefore costs one cache miss, not a policy decision about whether
to flush.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable

__all__ = ["PacketCache"]


class PacketCache:
    """In-process cache keyed by question *and* caller scope."""

    def __init__(self, store, *, max_entries: int = 64, ttl_s: float = 20.0,
                 clock: Callable[[], float] = time.monotonic):
        if not isinstance(max_entries, int) or not 1 <= max_entries <= 10_000:
            raise ValueError("max_entries must be an integer between 1 and 10000")
        if not isinstance(ttl_s, (int, float)) or not 0 < ttl_s <= 3600:
            raise ValueError("ttl_s must be a positive number of seconds, at most one hour")
        self.store = store
        self.max_entries = max_entries
        self.ttl_s = float(ttl_s)
        self.clock = clock
        self._entries: OrderedDict[tuple, tuple[Any, int, int, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.stale = 0
        self.last_invalidation: dict[str, Any] = {}

    def _key(self, query: str, *, account_id: str | None, scope: tuple[str, ...]) -> tuple:
        return (" ".join((query or "").casefold().split()), account_id or "", scope)

    def get(self, query: str, *, account_id: str | None = None,
            scope: tuple[str, ...] = ()) -> Any | None:
        key = self._key(query, account_id=account_id, scope=scope)
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        packet, epoch, revision, stored_at = entry
        if self.clock() - stored_at > self.ttl_s:
            del self._entries[key]
            self.stale += 1
            self.misses += 1
            return None
        current_epoch, current_revision = self.store.watermark()
        if (epoch, revision) != (current_epoch, current_revision):
            # The archive moved under this answer: forgotten, hidden, corrected,
            # or reset. Serving it anyway would be a retrieval bug with the
            # appearance of a working cache.
            del self._entries[key]
            self.stale += 1
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return packet

    def put(self, query: str, packet: Any, *, account_id: str | None = None,
            scope: tuple[str, ...] = ()) -> None:
        # Stamp with what the packet was *built from*, not what the archive says
        # now: the two differ whenever something changed while the request was in
        # flight, and recording the later stamp would bless a stale answer.
        epoch = getattr(packet, "epoch", None)
        revision = getattr(packet, "revision", None)
        if epoch is None or revision is None:
            epoch, revision = self.store.watermark()
        key = self._key(query, account_id=account_id, scope=scope)
        self._entries[key] = (packet, epoch, revision, self.clock())
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def invalidate(self, *, reason: str = "") -> int:
        """Drop everything. Returns the number of entries discarded."""
        count = len(self._entries)
        self._entries.clear()
        self.last_invalidation = {"reason": reason[:200], "entries": count}
        return count

    def as_dict(self) -> dict[str, Any]:
        return {"entries": len(self._entries), "max_entries": self.max_entries,
                "ttl_s": self.ttl_s, "hits": self.hits, "misses": self.misses,
                "stale": self.stale}

    def __len__(self) -> int:
        return len(self._entries)
