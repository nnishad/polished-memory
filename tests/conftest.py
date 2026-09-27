"""Shared fixtures and the canonical envelope factory.

The envelope deliberately carries real multilingual and structured metadata:
every suite that touches evidence should be exercising the same shape, not a
flattened test-only variant that the adapters never produce.
"""
from __future__ import annotations

import socket
from pathlib import Path

import pytest

from hermes_memory.storage.evidence import EvidenceStore

SOURCE = "gmail"
OBSERVED = "2026-09-25T12:00:00+00:00"
__all__ = ["SOURCE", "OBSERVED", "envelope"]


def _port(address) -> int | None:
    if isinstance(address, tuple) and len(address) > 1:
        try:
            return int(address[1])
        except (TypeError, ValueError):
            return None
    return None


@pytest.fixture(autouse=True)
def no_unstarted_socket(monkeypatch, request):
    """Refuse any connection to a listener this test did not start itself.

    Every backend, gate and mapper client takes an injected transport, so a real ``connect``
    means a door was wired with the installation's own configuration rather than a double.
    Naming the live ports instead would have missed the point: this suite's stand-ins share
    them — a route at ``127.0.0.1:8080`` is a llama.cpp server on this machine and an
    assertion about a URL string on that one — so a stray dial is either a spend of the
    owner's model or a quiet refusal depending on what happens to be running.
    """
    bound: set[int] = set()
    real_bind = socket.socket.bind
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def bind(self, address, *arguments):
        result = real_bind(self, address, *arguments)
        try:
            port = _port(self.getsockname())
        except OSError:  # a socket that lost its binding has no listener to dial anyway
            port = None
        if port:
            bound.add(port)
        return result

    def guard(name: str, original):
        def call(self, address, *arguments):
            if _port(address) not in bound:
                pytest.fail(f"{request.node.name} called socket.{name} on {address!r}, which "
                            "is not a listener this test bound; pass a transport or a stub "
                            "client, or start the socket here")
            return original(self, address, *arguments)
        return call

    monkeypatch.setattr(socket.socket, "bind", bind)
    monkeypatch.setattr(socket.socket, "connect", guard("connect", real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard("connect_ex", real_connect_ex))


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
