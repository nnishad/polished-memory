"""Shadow rebuild and cutover contracts, without model or production calls."""
import json

import pytest

from conftest import envelope
from hermes_memory.backend.document_map import DocumentMap
from hermes_memory.backend.generations import GenerationRegistry, recall_bank
from hermes_memory.backend.support import SupportError, SupportResolver
from hermes_memory.processing.formation import count_unprojected
from hermes_memory.storage.evidence import EvidenceError


def manifest(model="synthetic-model"):
    return {"version": "processor-manifest-v1", "text_model": model,
            "embedding_dimensions": 1024, "bank_prompt_config": "synthetic-verified-contract"}


def prepare(store, spec=None):
    registry = GenerationRegistry(store, family_bank="hermes", owner_principal="owner")
    spec = spec or manifest()
    plan = registry.plan(spec)
    return registry, registry.prepare(spec, actor="owner", review=plan["review_digest"])


def project(store, generation, record_id, revision="1"):
    docs = DocumentMap(store, bank_id=generation["bank_id"], generation_id=generation["id"])
    item = docs.begin(record_id, revision, async_submission=True)
    docs.pin_payload(record_id, revision, {"document_id": item["document_id"], "content": "synthetic"})
    docs.confirm(record_id, revision)
    return docs, item


def test_new_generation_reselects_verified_evidence_without_canonical_edit(store):
    record = store.commit(envelope())["id"]
    old = DocumentMap(store)
    old.begin(record, "1")
    old.confirm(record, "1")
    registry, generation = prepare(store)
    assert count_unprojected(store, bank_id="hermes") == 0
    assert count_unprojected(store, bank_id=generation["bank_id"]) == 1
    assert recall_bank(store, "hermes") == "hermes"
    docs, item = project(store, generation, record)
    plan = registry.cutover_plan(generation["id"])
    assert plan["blocking"] == [] and plan["verified"] == 1
    registry.activate(generation["id"], actor="owner", review=plan["review_digest"])
    assert recall_bank(store, "hermes") == generation["bank_id"]
    assert store.get(record).revision == "1"
    assert old.state(record, "1") == "verified"
    assert len(store.db.execute("SELECT * FROM backend_documents").fetchall()) == 2
    assert SupportResolver(store, bank_id=generation["bank_id"]).resolve(
        {"document_id": item["document_id"]}) == (record,)
    with pytest.raises(SupportError, match="active"):
        SupportResolver(store, bank_id="hermes")


def test_same_generation_is_idempotent_and_model_change_gets_another_bank(store):
    registry, first = prepare(store)
    plan = registry.plan(manifest())
    second = registry.prepare(manifest(), actor="owner", review=plan["review_digest"])
    assert second["id"] == first["id"]
    assert registry.plan(manifest("new-model"))["bank_id"] != first["bank_id"]


def test_owner_and_review_fences_prevent_generation_writes(store):
    registry = GenerationRegistry(store, owner_principal="owner")
    plan = registry.plan(manifest())
    with pytest.raises(EvidenceError, match="owner"):
        registry.prepare(manifest(), actor="agent", review=plan["review_digest"])
    store.commit(envelope())
    with pytest.raises(EvidenceError, match="review changed"):
        registry.prepare(manifest(), actor="owner", review=plan["review_digest"])
    assert store.db.execute("SELECT count(*) FROM projection_generations").fetchone()[0] == 0


def test_incomplete_and_forged_input_manifest_cannot_cut_over(store):
    record = store.commit(envelope())["id"]
    registry, generation = prepare(store)
    assert registry.cutover_plan(generation["id"])["blocking"]
    project(store, generation, record)
    store.db.execute("UPDATE backend_documents SET input_manifest=? WHERE generation_id=?",
                     (json.dumps({"version": "projection-input-v1", "inputs": []}), generation["id"]))
    plan = registry.cutover_plan(generation["id"])
    with pytest.raises(EvidenceError, match="cutover refused"):
        registry.activate(generation["id"], actor="owner", review=plan["review_digest"])
    assert registry.get(generation["id"])["state"] == "building"


def test_concurrent_source_change_invalidates_reviewed_cutover(store):
    record = store.commit(envelope())["id"]
    registry, generation = prepare(store)
    project(store, generation, record)
    plan = registry.cutover_plan(generation["id"])
    store.commit(envelope(source_id="next", text="more evidence"))
    with pytest.raises(EvidenceError, match="cutover refused"):
        registry.activate(generation["id"], actor="owner", review=plan["review_digest"])


def test_stale_acknowledgement_cannot_promote_epoch_or_become_a_retry(store):
    record = store.commit(envelope())["id"]
    registry, generation = prepare(store)
    docs, item = project(store, generation, record)
    store.bump_epoch(reason="reset", actor="owner")
    assert recall_bank(store, "hermes") is None
    for action in (lambda: docs.begin(record, "1"), lambda: docs.confirm(record, "1"),
                   lambda: docs.pin_payload(record, "1", {"document_id": item["document_id"]})):
        with pytest.raises(EvidenceError, match="stale"):
            action()


def test_hidden_evidence_has_no_projection_intent_or_support(store):
    record = store.commit(envelope())["id"]
    registry, generation = prepare(store)
    store.hide(record, reason="withdrawn", actor="owner")
    with pytest.raises(EvidenceError):
        project(store, generation, record)
    assert store.db.execute("SELECT count(*) FROM backend_documents").fetchone()[0] == 0


def test_unknown_embedding_and_prompt_contracts_block_activation(store):
    registry, generation = prepare(store, {"version": "processor-manifest-v1"})
    assert any("unverified" in item for item in registry.cutover_plan(generation["id"])["blocking"])
