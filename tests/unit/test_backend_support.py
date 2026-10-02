"""Observation support must be complete, bounded, current, and bank-scoped."""
import pytest

from conftest import envelope
from hermes_memory.backend.document_map import DocumentMap
from hermes_memory.backend.support import SupportError, SupportResolver
from hermes_memory.context import ContextBroker
from test_broker import FakeBackend


def mapped(store, *, bank="hermes", **overrides):
    identifier = store.commit(envelope(**overrides))["id"]
    docs = DocumentMap(store, bank_id=bank)
    entry = docs.begin(identifier, "1")
    docs.confirm(identifier, "1")
    return identifier, entry["document_id"]


def test_complete_observation_support_reaches_the_packet(store):
    identifier, document = mapped(store, text="tea preference", metadata={})
    backend = FakeBackend(results=[{"id": "obs", "text": "The speaker avoids tea",
                                    "source_fact_ids": ["source"]}],
                          source_facts=[{"id": "source", "document_id": document}])
    broker = ContextBroker(store, client=backend, cache=False)
    try:
        packet = broker.assemble("tea")
        assert packet.facts[0]["record_ids"] == [identifier]
        assert packet.channels.derived == "available"
    finally:
        broker.close()


@pytest.mark.parametrize("sources", [(), ({"id": "source", "source_fact_ids": ["source"]},),
                                    ({"id": "source", "source_fact_ids": ["missing"]},)])
def test_missing_or_cyclic_support_is_not_laundered_by_a_direct_reference(store, sources):
    identifier, _ = mapped(store, metadata={})
    resolver = SupportResolver(store, source_facts=sources)
    with pytest.raises(SupportError):
        resolver.resolve({"record_id": identifier, "source_fact_ids": ["source"]})


def test_cross_bank_or_unverified_document_is_not_canonical_support(store):
    identifier, document = mapped(store, bank="foreign", metadata={})
    with pytest.raises(SupportError):
        SupportResolver(store, bank_id="hermes").resolve({"document_id": document})
    store.db.execute("UPDATE backend_documents SET state='submitted' WHERE record_id=?", (identifier,))
    with pytest.raises(SupportError):
        SupportResolver(store, bank_id="foreign").resolve({"document_id": document})


def test_stale_epoch_and_hidden_source_withhold_support(store):
    identifier, document = mapped(store, metadata={})
    store.db.execute("UPDATE backend_documents SET desired_epoch=0 WHERE record_id=?", (identifier,))
    with pytest.raises(SupportError):
        SupportResolver(store).resolve({"document_id": document})
    store.db.execute("UPDATE backend_documents SET desired_epoch=? WHERE record_id=?",
                     (store.epoch(), identifier))
    store.hide(identifier, reason="withdrawn", actor="test-owner")
    with pytest.raises(SupportError):
        SupportResolver(store).resolve({"document_id": document})


def test_all_support_accounts_must_be_authorized(store):
    identifier, document = mapped(store, metadata={"account_ids": ["private-account"]})
    backend = FakeBackend(results=[{"text": "private observation", "source_fact_ids": ["source"]}],
                          source_facts=[{"id": "source", "document_id": document}])
    broker = ContextBroker(store, client=backend, cache=False)
    try:
        assert not broker.assemble("tea").facts
    finally:
        broker.close()


def test_manifest_and_traversal_bounds_fail_closed(store):
    identifier, _ = mapped(store, metadata={})
    with pytest.raises(SupportError):
        SupportResolver(store).resolve({"record_ids": [identifier] * 501})
    with pytest.raises(SupportError):
        SupportResolver(store, source_facts=[{"id": "x", "text": "a" * 262145}])


@pytest.mark.parametrize("value", [False, "", 0, {}])
def test_malformed_declared_support_is_not_treated_as_absent(store, value):
    identifier, _ = mapped(store, metadata={})
    with pytest.raises(SupportError):
        SupportResolver(store).resolve({"record_id": identifier, "source_fact_ids": value})
