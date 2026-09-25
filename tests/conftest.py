"""Shared fixtures and the canonical envelope factory.

The envelope deliberately carries real multilingual and structured metadata:
every suite that touches evidence should be exercising the same shape, not a
flattened test-only variant that the adapters never produce.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_memory.storage.evidence import EvidenceStore

SOURCE = "gmail"
OBSERVED = "2026-09-25T12:00:00+00:00"


def envelope(**overrides) -> dict:
    base = {
        "source": SOURCE,
        "source_id": "msg-1",
        "revision": "1",
        "kind": "email",
        "text": "The meeting moved to Thursday at 3pm.",
        "observed_at": OBSERVED,
        "occurred_at": "2026-09-24T09:15:00+02:00",
        "occurred_precision": "second",
        "metadata": {"participants": [{"namespace": "email", "address": "a@example.com"}]},
    }
    base.update(overrides)
    return base


@pytest.fixture()
def store(tmp_path: Path):
    with EvidenceStore(tmp_path / "canonical.db") as opened:
        yield opened


@pytest.fixture()
def sync(store):
    """The connector ledger over the same store, for tests that need a real row."""
    from hermes_memory.sources.sync import SyncController

    return SyncController(store)


@pytest.fixture()
def gate(store):
    from hermes_memory.processing.resource_gate import ResourceGate

    return ResourceGate(store)
