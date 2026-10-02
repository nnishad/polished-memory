"""F5: near-duplicate spans are folded, and the folding is visible.

The same sighting re-ingested under another id (a forwarded message, a re-export)
would otherwise spend the packet's budget saying one thing twice and crowd out a
distinct record. Folding keeps rank order, names what it dropped on the packet,
and is bounded so a wide candidate list cannot make recall quadratic.
"""
from __future__ import annotations

import pytest

from conftest import envelope
from hermes_memory.context import ContextBroker
from hermes_memory.context import broker as broker_module

LONG = ("The weekly planning meeting moved to Thursday at three in the afternoon "
        "in room four by the east stairwell")
DISTINCT = ("The planning meeting agenda for Thursday includes the budget review "
            "items and the hiring loop notes")


@pytest.fixture()
def broker(store):
    built = ContextBroker(store, cache=None)
    yield built
    built.close()


def commit(store, text, source_id):
    return store.commit(envelope(text=text, source_id=source_id))["id"]


def test_a_repeated_sighting_is_folded_into_its_higher_ranked_twin(store, broker):
    first = commit(store, LONG, "dup-a")
    second = commit(store, LONG, "dup-b")
    packet = broker.assemble("planning meeting Thursday room", limit=5)
    kept = [item.id for item in packet.items]
    assert first in kept and second not in kept
    assert packet.deduped == (second,)
    assert packet.as_dict()["deduped"] == [second]
    assert "near-duplicate" in packet.render()


def test_distinct_records_are_not_folded(store, broker):
    one = commit(store, LONG, "a")
    two = commit(store, DISTINCT, "b")
    packet = broker.assemble("planning meeting Thursday", limit=5)
    assert {one, two} <= {item.id for item in packet.items}
    assert packet.deduped == ()
    assert "near-duplicate" not in packet.render()


def test_a_folded_duplicate_does_not_crowd_out_a_distinct_record(store, broker):
    commit(store, LONG, "dup-a")
    dup = commit(store, LONG, "dup-b")
    distinct = commit(store, DISTINCT, "distinct")
    packet = broker.assemble("planning meeting Thursday", limit=2)
    kept = [item.id for item in packet.items]
    assert distinct in kept and dup not in kept
    assert packet.deduped == (dup,)


def test_the_comparison_cap_keeps_what_it_cannot_afford_to_compare(store, broker,
                                                                   monkeypatch):
    monkeypatch.setattr(broker_module, "_DEDUP_MAX_COMPARISONS", 1)
    first = commit(store, LONG, "dup-a")
    commit(store, DISTINCT, "mid")
    second = commit(store, LONG, "dup-b")
    packet = broker.assemble("planning meeting Thursday", limit=5)
    kept = [item.id for item in packet.items]
    # Past the cap the remaining candidates are kept as-is: an incomplete dedup is
    # a smaller lie than a late packet, and the cap is reported by omission.
    assert first in kept and second in kept
    assert packet.deduped == ()
