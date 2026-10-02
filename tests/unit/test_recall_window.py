"""F4: temporal windows are a first-class, validated recall parameter.

The broker could always filter on occurred_at, but nothing surfaced the window,
so "nothing in March" and "nothing at all" read identically. The window is now
validated at the provider edge (absolute ISO only, inversion refused), threaded
into assemble, and reported on the packet.
"""
from __future__ import annotations

import importlib

import pytest

from conftest import envelope
from hermes_memory.context import ContextBroker
from plugin_loader import load_plugin

load_plugin()
provider = importlib.import_module("hm_plugin.provider")
_window = provider._window


def test_no_bounds_means_no_window():
    assert _window(None, None) is None


def test_bounds_are_normalized_to_iso():
    assert _window("2026-09-01", "2026-09-30") == (
        "2026-09-01T00:00:00", "2026-09-30T00:00:00")
    assert _window("2026-09-05T06:00:00Z", None) == ("2026-09-05T06:00:00+00:00", None)


@pytest.mark.parametrize("since, until", [
    ("2026-09-30", "2026-09-01"),
    ("not-a-date", None),
    (None, "yesterday"),
    (12345, None),
])
def test_inverted_and_unparseable_windows_are_refused(since, until):
    with pytest.raises(ValueError):
        _window(since, until)


@pytest.fixture()
def broker(store):
    built = ContextBroker(store, cache=None)
    yield built
    built.close()


def _commit(store, text, source_id, occurred_at):
    return store.commit(envelope(text=text, source_id=source_id, occurred_at=occurred_at,
                                 occurred_precision="second" if occurred_at else "unknown"))["id"]


def test_a_window_excludes_dated_evidence_outside_it_but_keeps_undated(store, broker):
    inside = _commit(store, "we booked the Jaipur train for friday", "in",
                     "2026-09-24T09:15:00+02:00")
    outside = _commit(store, "the jaipur trip was first planned in august", "out",
                      "2026-08-02T09:15:00+02:00")
    undated = _commit(store, "jaipur plans are still tentative overall", "undated", None)
    packet = broker.assemble("jaipur", limit=8, window=("2026-09-01T00:00:00",
                                                        "2026-09-30T00:00:00"))
    kept = {item.id for item in packet.items}
    assert inside in kept and undated in kept and outside not in kept
    assert packet.window == ("2026-09-01T00:00:00", "2026-09-30T00:00:00")
    assert packet.as_dict()["window"] == ["2026-09-01T00:00:00", "2026-09-30T00:00:00"]
    assert "requested window" in packet.render()


def test_without_a_window_the_packet_says_so(store, broker):
    _commit(store, "we booked the Jaipur train for friday", "in",
            "2026-09-24T09:15:00+02:00")
    packet = broker.assemble("jaipur", limit=8)
    assert packet.window is None
    assert packet.as_dict()["window"] is None
    assert "requested window" not in packet.render()
