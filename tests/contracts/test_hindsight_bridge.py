"""C5 bridge contract, exercised with no backend and no network."""
from __future__ import annotations

import json
import pathlib
import uuid

import pytest

from hermes_memory.backend.capabilities import (CAPABILITIES, PINNED_VERSION,
                                                UnsupportedCapability, capabilities_for)
from hermes_memory.backend.document_map import (ABSENT, FAILED, QUEUED, VERIFIED,
                                   DocumentMap)
from hermes_memory.backend.hindsight_client import (HindsightClient, HindsightError,
                                                   HindsightUnavailable, SubmissionConflict,
                                                   TransportResult)

from conftest import envelope

BANK = "hermes"
ROUTES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "hindsight_0_10_1_routes.json"


class Recorder:
    """A stand-in backend that records requests and replays scripted replies."""

    def __init__(self, replies=None):
        self.calls = []
        self.replies = dict(replies or {})
        self.default = TransportResult(200, {"ok": True})

    def when(self, key, result):
        self.replies[key] = result
        return self

    def __call__(self, method, url, payload, headers):
        self.calls.append({"method": method, "url": url, "payload": payload,
                           "headers": dict(headers)})
        for key, result in self.replies.items():
            # A reply is registered as "VERB /path"; the client supplies the host and the
            # reading carries its filters as a query string, so compare the verb exactly and
            # match the path on whatever comes before the "?".
            verb, _, path = key.partition(" ")
            if verb == method and url.split("?", 1)[0].endswith(path):
                return result
        return self.default


@pytest.fixture()
def transport():
    return Recorder()


@pytest.fixture()
def client(transport):
    return HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK,
                           api_key="secret-in-vault", transport=transport)


@pytest.fixture()
def mapped(store):
    record = store.commit(envelope(source_id="msg-1"))
    docs = DocumentMap(store)
    return docs, record["id"]


# -- handshake ---------------------------------------------------------------

def test_reflection_support_readback_is_exact_and_recursive(client, transport):
    transport.when(f"POST /v1/default/banks/{BANK}/reflect", TransportResult(200,
        {"text": "Coffee", "based_on": {"memories": [{"id": "observation-1", "text": "Coffee"}]}}))
    transport.when(f"GET /v1/default/banks/{BANK}/memories/observation-1", TransportResult(200,
        {"id": "observation-1", "text": "Coffee", "source_memory_ids": ["fact-1"], "fact_type": "observation"}))
    transport.when(f"GET /v1/default/banks/{BANK}/memories/fact-1", TransportResult(200,
        {"id": "fact-1", "document_id": "canonical-document", "text": "Coffee"}))
    result = client.reflect("Drink?", include_support=True)
    assert result["facts"][0]["source_fact_ids"] == ["fact-1"]
    assert result["source_facts"][0]["document_id"] == "canonical-document"
    assert len(transport.calls) == 3


def test_reflection_support_cycle_and_identity_drift_are_refused(client, transport):
    transport.when(f"GET /v1/default/banks/{BANK}/memories/fact-1", TransportResult(200,
        {"id": "fact-1", "source_memory_ids": ["fact-1"]}))
    with pytest.raises(HindsightError, match="cyclic"):
        client.hydrate_support([{"id": "fact-1"}])
    transport.when(f"GET /v1/default/banks/{BANK}/memories/fact-1", TransportResult(200,
        {"id": "other-fact", "document_id": "other"}))
    with pytest.raises(HindsightError, match="different"):
        client.hydrate_support([{"id": "fact-1"}])


@pytest.mark.parametrize("identifier", ["../other", "x?bank=other", "", "x" * 129])
def test_memory_readback_rejects_path_injection_before_socket(client, transport, identifier):
    with pytest.raises(HindsightError, match="path-safe"):
        client.get_memory(identifier)
    assert not transport.calls


def test_reflection_support_identity_budget_precedes_socket(client, transport):
    with pytest.raises(HindsightError, match="bound"):
        client.hydrate_support([{"id": str(i)} for i in range(65)])
    assert not transport.calls

def test_an_unreachable_backend_raises_instead_of_starting_something(transport):
    transport.default = TransportResult(0, transport_error="connection refused")
    client = HindsightClient(base_url="http://127.0.0.1:1", bank_id=BANK, transport=transport)
    with pytest.raises(HindsightUnavailable, match="unreachable"):
        client.health()
    assert len(transport.calls) == 1, "no retry storm, no process spawn"


def test_constructing_a_client_never_touches_the_network():
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("constructor reached the network")

    client = HindsightClient(base_url="http://127.0.0.1:8123", bank_id=BANK, transport=explode)
    assert client.capabilities.version == PINNED_VERSION


def test_scoped_consolidation_uses_the_pinned_native_shape(client, transport):
    transport.default = TransportResult(200, {"operation_id": "operation-1", "deduplicated": False})
    assert client.consolidate(observation_scopes=[["project:budget"]])["operation_id"] == "operation-1"
    assert transport.calls[0]["url"].endswith(f"/banks/{BANK}/consolidate")
    assert transport.calls[0]["payload"] == {"observation_scopes": [["project:budget"]]}


@pytest.mark.parametrize("scopes", [None, [], [[]], ["tag"], [[""]], [["x" * 201]]])
def test_consolidation_cannot_implicitly_request_every_scope(client, transport, scopes):
    with pytest.raises(HindsightError):
        client.consolidate(observation_scopes=scopes)
    assert transport.calls == []


def test_source_fact_map_keys_survive_missing_redundant_ids():
    from hermes_memory.backend.hindsight_client import RecallOutcome

    assert RecallOutcome.from_body({"source_facts": {"f1": {"text": "fact"}}}).source_facts == (
        {"id": "f1", "text": "fact"},)
    with pytest.raises(HindsightError, match="identities"):
        RecallOutcome.from_body({"source_facts": {"f1": {"id": "another"}}})


def test_an_api_key_is_sent_but_never_written_into_a_request_body(client, transport):
    client.retain(document_id="hdocabc", content="hello")
    call = transport.calls[0]
    assert call["headers"]["Authorization"] == "Bearer secret-in-vault"
    assert "secret-in-vault" not in json.dumps(call["payload"])


# -- capability enforcement --------------------------------------------------

def test_a_request_is_refused_when_the_backend_does_not_route_it(client):
    client._capabilities = capabilities_for(PINNED_VERSION, route_names={"retain"})
    with pytest.raises(UnsupportedCapability, match="not available"):
        client.stats()
    with pytest.raises(UnsupportedCapability):
        client.recall("anything")
    assert client.retain(document_id="hdoc1", content="x")["ok"] is True, (
        "the one routed operation still works")


def test_the_pinned_version_supports_the_operations_we_reconcile_with(client):
    for name in ("retain", "retain_async_identity", "recall", "delete_memories",
                 "list_operations", "get_operation", "cancel_operation", "bank_stats"):
        assert name in client.capabilities.supported, name


def test_read_only_probes_are_not_reported_as_the_whole_route_table(client, transport):
    """A 404 on three read-only routes says nothing at all about `retain`.

    Read as a census, that partial look told the doctor that a backend which had been
    answering retains for hours routed neither retain nor recall — and `require` takes the
    report at its word, so the next caller holding that client would refuse the work outright
    on the strength of a lookup the engine had never been asked to provide.
    """
    transport.when("GET /v1/default/banks", TransportResult(200, {"banks": []}))
    transport.when(f"GET /v1/default/banks/{BANK}/stats", TransportResult(404, {}))
    transport.when(f"GET /v1/default/banks/{BANK}/operations", TransportResult(404, {}))

    observed = client.negotiate(probe_routes=True)
    assert {"retain", "recall", "delete_memories"} <= observed.supported, observed.unsupported
    assert sorted(observed.unsupported) == ["bank_stats", "list_operations"]
    assert all("answered 404" in line for line in observed.mismatches), observed.mismatches


def test_an_older_backend_reports_the_newer_capabilities_it_lacks():
    caps = capabilities_for("0.9.2")
    assert "dry_run_extract" not in caps.supported, "advertised at 0.10.1, absent at 0.9.2"


# -- retain ------------------------------------------------------------------

def test_one_canonical_revision_maps_to_one_backend_document(client, transport):
    client.retain(document_id="hdoc1a2b", content="The invoice was paid.",
                  timestamp="2026-09-24T09:15:00+00:00")
    payload = transport.calls[0]["payload"]
    assert len(payload["items"]) == 1
    assert payload["items"][0]["document_id"] == "hdoc1a2b"
    assert payload["async"] is False


@pytest.mark.parametrize("bad", ["hdoc_with_underscore", "hdoc~tilde", ""])
def test_an_ambiguous_document_id_is_refused_before_any_request(client, bad):
    """Hindsight escapes '_' and '~' in chunk IDs, making them unresolvable."""
    with pytest.raises(HindsightError, match="document_id must be"):
        client.retain(document_id=bad, content="x")


def test_an_async_retain_requires_an_identity_persisted_first(client):
    with pytest.raises(HindsightError, match="requires a submission_id"):
        client.retain_async([{"content": "x", "document_id": "hdoc1"}], submission_id="")
    with pytest.raises(HindsightError, match="must be a UUID"):
        client.retain_async([{"content": "x"}], submission_id="job-17")


def test_the_submission_identity_is_sent_as_the_operation_id(client, transport):
    submission = str(uuid.uuid4())
    client.retain_async([{"content": "x", "document_id": "hdoc1"}], submission_id=submission)
    payload = transport.calls[0]["payload"]
    assert payload["operation_id"] == submission and payload["async"] is True


def test_a_reused_identity_for_different_content_is_a_conflict_not_a_retry(client, transport):
    transport.when("POST http://127.0.0.1:8813/v1/default/banks/hermes/memories",
                   TransportResult(409, {"detail": "operation_id already used"}))
    with pytest.raises(SubmissionConflict, match="reconciling"):
        client.retain_async([{"content": "x", "document_id": "hdoc1"}],
                            submission_id=str(uuid.uuid4()))


def test_an_oversized_async_batch_is_refused(client):
    with pytest.raises(HindsightError, match="1 and 200"):
        client.retain_async([{"content": "x"}] * 201, submission_id=str(uuid.uuid4()))


# -- lost acknowledgements ---------------------------------------------------

def test_a_timeout_is_reported_as_unconfirmed_rather_than_failed(client, transport):
    """The backend may be running it now; 'failed' would let us resubmit blindly."""
    transport.default = TransportResult(0, transport_error="timed out")
    with pytest.raises(HindsightUnavailable, match="may still be running"):
        client.retain(document_id="hdoc1", content="x")


def test_a_503_carries_its_status_for_the_callers_backoff(client, transport):
    transport.default = TransportResult(503, {"detail": "admission control saturated"})
    with pytest.raises(HindsightError) as caught:
        client.stats()
    assert caught.value.status == 503


def test_reconciliation_settles_a_submission_we_cannot_account_for(store, mapped, transport):
    docs, record = mapped
    begun = docs.begin(record, "1", async_submission=True)
    assert docs.state(record, "1") == "submitted"
    assert begun["submission_id"]

    operation_id = begun["submission_id"]
    client = HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK, transport=transport)
    transport.when(f"GET /v1/default/banks/{BANK}/operations/{operation_id}",
                   TransportResult(200, {"state": "completed"}))
    outcome = docs.reconcile(client=client)
    assert outcome["verified"] == 1
    assert docs.state(record, "1") == VERIFIED


def test_reconciliation_leaves_a_still_running_operation_pending(mapped, transport):
    docs, record = mapped
    begun = docs.begin(record, "1", async_submission=True)
    client = HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK, transport=transport)
    transport.when(f"GET /v1/default/banks/{BANK}/operations/{begun['submission_id']}",
                   TransportResult(200, {"status": "processing"}))
    outcome = docs.reconcile(client=client)
    assert outcome["pending"] == 1 and outcome["verified"] == 0
    assert docs.state(record, "1") == "submitted"
    assert store_row(docs.store, record)["error"] is None, \
        "an ordinary 'not yet' must not look like an anomaly to whoever reads it next"


def test_an_undocumented_operation_answer_is_written_down_not_guessed(mapped, transport):
    """A word the pinned release does not use is neither finished nor running.

    Reading it as pending without a reason would leave the row looking like ordinary
    waiting, which is the one state an operator has no reason to look at.
    """
    docs, record = mapped
    begun = docs.begin(record, "1", async_submission=True)
    client = HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK, transport=transport)
    transport.when(f"GET /v1/default/banks/{BANK}/operations/{begun['submission_id']}",
                   TransportResult(200, {"status": "quiescing"}))
    outcome = docs.reconcile(client=client)
    assert outcome["pending"] == 1 and outcome["verified"] == 0
    assert docs.state(record, "1") == "submitted"
    assert "quiescing" in str(store_row(docs.store, record)["error"])


def test_an_operation_the_backend_has_no_record_of_is_closed_rather_than_asked_forever(
        mapped, transport):
    """'Not yet' and 'never' are different answers, and only one is worth waiting on.

    A 404 is the engine's own ``not_found``, so it is settled as a failed projection — which
    makes the record unprojected again and lets the next pass form it deliberately.
    """
    docs, record = mapped
    begun = docs.begin(record, "1", async_submission=True)
    client = HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK, transport=transport)
    transport.default = TransportResult(404, {"detail": "no such operation"})
    outcome = docs.reconcile(client=client)
    assert outcome["settled"] == 1 and outcome["pending"] == 0
    assert docs.state(record, "1") == FAILED
    assert "not_found" in str(store_row(docs.store, record)["error"])


def test_reconciliation_does_not_invent_an_outcome_when_the_backend_is_down(mapped, transport):
    docs, record = mapped
    docs.begin(record, "1", async_submission=True)
    transport.default = TransportResult(0, transport_error="connection refused")
    client = HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK, transport=transport)
    outcome = docs.reconcile(client=client)
    assert outcome["unreachable"] == 1
    assert docs.state(record, "1") == "submitted", "an unreachable backend proves nothing"


def store_row(store, record):
    return store.db.execute("SELECT * FROM backend_documents WHERE record_id=?",
                            (record,)).fetchone()


def test_a_synchronous_submission_with_no_answer_is_settled_by_the_document_itself(
        mapped, transport):
    """The lost acknowledgement of a *sync* retain has no operation to ask about.

    The document can still be asked about, and only the backend can say whether the
    bytes it was supposed to hold are there.
    """
    docs, record = mapped
    docs.begin(record, "1")
    assert docs.state(record, "1") == QUEUED
    client = HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK, transport=transport)
    transport.when(f"GET /v1/default/banks/{BANK}/memories/list",
                   TransportResult(200, {"memories": [{"id": "one"}]}))

    outcome = docs.reconcile(client=client)

    assert outcome["verified"] == 1 and outcome["absent"] == 0
    assert docs.state(record, "1") == VERIFIED


def test_present_sync_document_with_withdrawn_input_is_failed_not_coverage(mapped, transport):
    docs, record = mapped
    docs.begin(record, "1")
    docs.store.hide(record, reason="withdrawn", actor="owner")
    client = HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK, transport=transport)
    transport.when(f"GET /v1/default/banks/{BANK}/memories/list",
                   TransportResult(200, {"memories": [{"id": "one"}]}))
    outcome = docs.reconcile(client=client)
    assert outcome["verified"] == 0
    assert docs.state(record, "1") == FAILED
    assert "withdrawn" in store_row(docs.store, record)["error"]


def test_a_document_the_backend_does_not_have_is_recorded_as_gone(mapped, transport):
    docs, record = mapped
    docs.begin(record, "1")
    document = docs.document_id(record, "1")
    client = HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK, transport=transport)
    transport.when(f"GET /v1/default/banks/{BANK}/memories/list",
                   TransportResult(200, {"memories": []}))

    outcome = docs.reconcile(client=client)

    assert outcome["absent"] == 1 and outcome["verified"] == 0
    assert docs.state(record, "1") == ABSENT, \
        f"the backend says {document} is not there, and that is an answer"


def test_an_unreachable_backend_leaves_a_queued_mapping_queued(mapped, transport):
    docs, record = mapped
    docs.begin(record, "1")
    transport.default = TransportResult(0, transport_error="connection refused")
    client = HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK, transport=transport)

    outcome = docs.reconcile(client=client)

    assert outcome["unreachable"] == 1 and outcome["settled"] == 0
    assert docs.state(record, "1") == QUEUED, "a failed question is not an answer"


def test_a_backend_that_refuses_the_question_is_not_treated_as_absent(store, mapped,
        transport):
    docs, record = mapped
    docs.begin(record, "1")
    transport.default = TransportResult(403, {"detail": "forbidden"})
    client = HindsightClient(base_url="http://127.0.0.1:8813", bank_id=BANK, transport=transport)

    outcome = docs.reconcile(client=client)

    assert outcome["unreachable"] == 1 and outcome["settled"] == 0
    assert docs.state(record, "1") == QUEUED, \
        "a 403 says we may not ask, not that the document is gone"
    assert "403" in str(store_row(store, record)["error"]), \
        "and the reason is left on the row for whoever looks next"


# -- document map ------------------------------------------------------------

def test_intent_is_recorded_before_the_request_exists(store, mapped):
    docs, record = mapped
    begun = docs.begin(record, "1")
    assert begun["state"] == "queued"
    assert begun["submission_id"] is None, "a sync retain has no operation to reconcile"
    assert begun["document_id"] == docs.document_id(record, "1")


def test_a_verified_projection_is_not_resubmitted(store, mapped):
    docs, record = mapped
    docs.begin(record, "1")
    docs.confirm(record, "1")
    again = docs.begin(record, "1", async_submission=True)
    assert again["already_projected"] is True and again["state"] == VERIFIED
    assert again["submission_id"] is None, "no new identity, so no duplicate work"


def test_a_retry_reuses_the_identity_already_on_record(store, mapped):
    docs, record = mapped
    first = docs.begin(record, "1", async_submission=True)
    second = docs.begin(record, "1", async_submission=True)
    assert first["submission_id"] == second["submission_id"], (
        "minting a new id would abandon the operation the backend may be running")


def test_the_document_id_is_derived_not_parsed(store, mapped):
    docs, record = mapped
    document_id = docs.document_id(record, "1")
    assert document_id.startswith("hdoc") and "_" not in document_id


def test_begin_refuses_a_record_that_does_not_exist(store):
    docs = DocumentMap(store)
    with pytest.raises(Exception, match="unknown record"):
        docs.begin("rec_" + "0" * 32, "1")


# -- recall ------------------------------------------------------------------

def test_provenance_is_requested_as_a_nested_object_with_its_own_budget(client, transport):
    transport.default = TransportResult(200, {"results": []})
    client.recall("garage code", include_source_facts=True, source_facts_tokens=700)
    include = transport.calls[0]["payload"]["include"]
    assert include["source_facts"] == {"max_tokens": 700}
    assert "include_source_facts" not in json.dumps(transport.calls[0]["payload"]), (
        "the internal flag name is not the HTTP payload shape")


def test_a_truncated_source_fact_set_is_reported_not_hidden(client, transport):
    transport.default = TransportResult(200, {
        "results": [{"id": "m1", "text": "the code is 8841"}],
        "include": {"source_facts": {"f1": {"id": "f1"}}, "source_facts_truncated": True},
    })
    outcome = client.recall("garage")
    assert outcome.truncated == ("source_facts",)
    assert outcome.provenance_complete is False
    assert outcome.as_dict()["results"] == 1


def test_a_full_source_fact_set_claims_provenance(client, transport):
    transport.default = TransportResult(200, {"results": [{"id": "m1"}],
                                              "include": {"source_facts": {"f1": {}}}})
    assert client.recall("garage").provenance_complete is True


def test_an_empty_query_is_refused_locally(client, transport):
    with pytest.raises(HindsightError, match="must not be empty"):
        client.recall("   ")
    assert transport.calls == []


# -- erasure -----------------------------------------------------------------

def test_a_missing_backend_document_counts_as_verified_gone(client, transport):
    transport.default = TransportResult(404, {"detail": "no such document"})
    outcome = client.delete_document("hdoc1")
    assert outcome["absent"] is True and outcome["deleted"] is False


def test_document_state_distinguishes_absent_from_unknown(client, transport):
    transport.default = TransportResult(200, {"memories": []})
    assert client.document_state("hdoc1")["state"] == "absent"
    transport.default = TransportResult(500, {"detail": "backend error"})
    assert client.document_state("hdoc1")["state"] == "unknown"


def test_the_submission_identity_survives_the_row_it_was_written_with(store, mapped):
    """Reconciliation is worthless if the id only lived in a return value."""
    docs, record = mapped
    begun = docs.begin(record, "1", async_submission=True)
    row = store.db.execute(
        "SELECT operation_id, state FROM backend_documents WHERE record_id=? AND revision=?",
        (record, "1")).fetchone()
    assert row["operation_id"] == begun["submission_id"]
    assert row["state"] == "submitted"


def test_a_sync_projection_stays_queued_until_something_happens(store, mapped):
    docs, record = mapped
    docs.begin(record, "1")
    assert docs.outstanding()[0]["operation_id"] is None


# -- what the pinned revision actually serves ----------------------------------

def test_every_declared_capability_is_a_route_the_pinned_backend_serves():
    """A verb remembered wrongly is a HTTP 405 on the first call that matters.

    Every other test in this file answers with a stand-in that serves whatever the table
    asks for, so a row the pinned release does not have is invisible until a real
    installation tries to reconcile a lost request — which is precisely the moment it has
    to work. The table beside this file is a reading of that release's own OpenAPI
    document, so the check is the machine's answer rather than somebody's note about it.
    """
    declared = {f"{item.method} {item.path}" for item in CAPABILITIES}
    served = set(json.loads(ROUTES.read_text(encoding="utf-8"))["routes"])
    assert declared <= served, f"this build asks for routes {sorted(declared - served)}"


def test_the_route_table_read_belongs_to_the_revision_this_build_pins():
    """A pin bump without a re-read of the release's routes is an unverified claim."""
    recorded = json.loads(ROUTES.read_text(encoding="utf-8"))
    assert recorded["pinned_version"] == PINNED_VERSION, \
        "re-read /openapi.json from the new pinned backend and update the fixture"


def test_the_reconciliation_reading_asks_as_a_filtered_get_with_the_document_named(
        client, transport):
    """A table row can name the right verb while the client still posts to it.

    The filter lives in the query string on this revision, so the document being asked
    about has to be visible in the URL: a GET with no filter answers "what is in this
    bank", which would read as proof that a missing document is present.
    """
    transport.default = TransportResult(200, {"memories": [{"id": "hdoc1"}]})
    assert client.document_state("hdoc1")["state"] == "present"
    asked = transport.calls[-1]
    assert asked["method"] == "GET" and asked["payload"] is None
    assert "document_id=hdoc1" in asked["url"] and "limit=5" in asked["url"]
