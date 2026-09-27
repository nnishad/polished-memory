"""C12 gate endpoint: what it serves, what it refuses, and what it never does."""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from hermes_memory.processing.gate_server import (GateApp, _apply_cap,
                                                    _usage_tokens, upstream_url)
from hermes_memory.processing.resource_gate import ResourceGate
from hermes_memory.processing.routes import RouteTable, Route
from hermes_memory.storage.evidence import EvidenceStore

REMOTE = "remote-9b"
GPU = "local-gpu"

TABLE = RouteTable({
    "foreground": Route("foreground", REMOTE, "chat", "http://127.0.0.1:8080/v1", "cred-chat",
                        "interactive", 1024),
    "embeddings": Route("embeddings", GPU, "embeddings", "http://127.0.0.1:11434/v1",
                        "cred-emb", "freshness", 0),
})
CREDENTIALS = {REMOTE: "real-upstream-secret-a", GPU: "real-upstream-secret-b"}


class Upstream:
    def __init__(self, status=200, body=None, error=None):
        self.calls = []
        self.status = status
        self.body = json.dumps(body or {"choices": [], "usage": {"total_tokens": 42}}).encode()
        self.error = error

    def __call__(self, url, body, headers):
        self.calls.append({"url": url, "payload": json.loads(body), "headers": dict(headers)})
        if self.error:
            return _Result(0, transport_error=self.error)
        return _Result(self.status, self.body)


class _Result:
    def __init__(self, status, body=b"", transport_error=None):
        self.status, self.body, self.transport_error = status, body, transport_error

    @property
    def reached(self):
        return self.transport_error is None


@pytest.fixture()
def harness(store):
    gate = ResourceGate(store)
    upstream = Upstream()
    app = GateApp(routes=TABLE, gate=gate, upstream_credentials=CREDENTIALS, upstream=upstream)
    return app, gate, upstream


def call(app, path="/v1/chat/completions", *, token="cred-chat", body=None, method="POST"):
    payload = json.dumps(body if body is not None else
                         {"messages": [{"role": "user", "content": "hi"}]}).encode()
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": method, "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": [(b"authorization", f"Bearer {token}".encode()),
                         (b"content-type", b"application/json")],
             "client": ("127.0.0.1", 5555), "server": ("127.0.0.1", 8813), "scheme": "http"}
    sent = []
    given = [{"type": "http.request", "body": payload, "more_body": False}]

    async def receive():
        return given.pop(0) if given else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = sent[0]
    headers = {k.decode(): v.decode() for k, v in start["headers"]}
    return {"status": start["status"], "headers": headers,
            "body": json.loads(sent[-1]["body"] or b"{}")}


# -- surface -----------------------------------------------------------------

def test_only_the_named_paths_are_served(harness):
    app, gate, upstream = harness
    for path in ("/v1/completions", "/v1/chat/completions/extra", "/", "/proxy",
                 "/v1/models"):
        response = call(app, path)
        assert response["status"] == 404, path
        assert "serves only" in response["body"]["error"]["message"]
    assert upstream.calls == [], "a refused path must never reach the model server"


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "PATCH", "OPTIONS"])
def test_a_verb_the_gate_does_not_serve_reserves_nothing(harness, method):
    """PUT and DELETE are refused, not half-supported.

    A refusal that reached the gate would let a caller occupy a device slot without a
    credential being read, and one that reached the upstream would turn a verb nobody
    authorised into a write somewhere else.
    """
    app, gate, upstream = harness
    response = call(app, "/v1/chat/completions", method=method)
    assert response["status"] == 404
    assert "by POST" in response["body"]["error"]["message"]
    assert upstream.calls == []
    assert gate.held() == []


def test_an_unknown_credential_selects_no_upstream(harness):
    app, _, upstream = harness
    response = call(app, token="cred-that-does-not-exist")
    assert response["status"] == 401
    assert "does not select any route" in response["body"]["error"]["message"]
    assert upstream.calls == []


def test_a_credential_cannot_name_its_own_upstream(harness):
    """The caller supplies no URL by construction; only the route table does."""
    app, _, upstream = harness
    call(app, body={"model": "x", "messages": [], "upstream": "http://198.51.100.7:9/v1"})
    assert upstream.calls[0]["url"] == "http://127.0.0.1:8080/v1/chat/completions"


# -- forwarding --------------------------------------------------------------

def test_a_chat_request_is_forwarded_with_the_upstream_credential(harness):
    app, gate, upstream = harness
    response = call(app)
    assert response["status"] == 200
    sent = upstream.calls[0]
    assert sent["headers"]["Authorization"] == "Bearer real-upstream-secret-a"
    assert "cred-chat" not in json.dumps(sent["headers"])
    assert gate.usage()[REMOTE]["tokens"] == 42


def test_embeddings_use_the_gpu_slot_and_a_different_upstream(harness):
    app, gate, upstream = harness
    call(app, "/v1/embeddings", token="cred-emb", body={"input": "hello", "model": "emb"})
    # The configured root already carries /v1; asking the server for it twice is a 404
    assert upstream.calls[0]["url"] == "http://127.0.0.1:11434/v1/embeddings"
    assert gate.usage()[GPU]["tokens"] == 42
    assert gate.usage().get(REMOTE, {}).get("calls", 0) == 0, "different device, different slot"


# -- caps --------------------------------------------------------------------

def test_a_missing_output_cap_is_forced(harness):
    app, _, upstream = harness
    call(app)
    assert upstream.calls[0]["payload"]["max_tokens"] == 1024


def test_a_huge_requested_cap_is_lowered_and_said_so(harness):
    app, _, upstream = harness
    response = call(app, body={"messages": [], "max_tokens": 999_999})
    assert upstream.calls[0]["payload"]["max_tokens"] == 1024
    assert response["headers"]["x-gate-max-tokens-capped"] == "1024"


def test_a_smaller_requested_cap_is_respected(harness):
    app, _, upstream = harness
    response = call(app, body={"messages": [], "max_tokens": 64})
    assert upstream.calls[0]["payload"]["max_tokens"] == 64
    assert "x-gate-max-tokens-capped" not in response["headers"]


def test_embedding_routes_are_not_given_a_completion_cap(harness):
    app, _, upstream = harness
    call(app, "/v1/embeddings", token="cred-emb", body={"input": "x"})
    assert "max_tokens" not in upstream.calls[0]["payload"]


def test_thinking_stays_off_through_the_gate(harness):
    """Measured: on the local 4B it consumed 256 tokens and produced no output."""
    app, _, upstream = harness
    call(app)
    assert upstream.calls[0]["payload"]["extra_body"]["enable_thinking"] is False


@pytest.mark.parametrize("payload, limit, expected", [
    ({"max_tokens": 10}, 1024, 10), ({"max_tokens": 5000}, 1024, 1024),
    ({}, 1024, 1024), ({"max_tokens": 10}, 0, 10),
])
def test_the_cap_rule_in_isolation(payload, limit, expected):
    adjusted, capped = _apply_cap(dict(payload), limit)
    assert adjusted.get("max_tokens") == expected


# -- admission ---------------------------------------------------------------

def test_a_busy_resource_gets_429_not_a_silent_queue(harness):
    app, gate, upstream = harness
    held = gate.try_acquire(route="reflect", holder="worker", resource=REMOTE, priority=3)
    response = call(app)
    assert response["status"] == 429
    assert response["headers"]["retry-after"] == "2"
    assert upstream.calls == []
    gate.release(held, outcome="succeeded")


def test_a_pause_returns_503_immediately(harness):
    app, gate, upstream = harness
    gate.pause(actor="owner", reason="gpu needed elsewhere")
    assert call(app)["status"] == 503
    assert upstream.calls == []


def test_an_unreachable_upstream_keeps_the_slot_blocked(harness):
    """A dead socket is not proof the model stopped generating."""
    app, gate, upstream = harness
    upstream.error = "connection reset by peer"
    response = call(app)
    assert response["status"] == 504
    assert response["body"]["error"]["type"] == "uncertain"
    assert gate.blocked_resources() == [REMOTE]


def test_a_500_from_upstream_frees_the_slot(harness):
    app, gate, upstream = harness
    upstream.status = 500
    assert call(app)["status"] == 500
    assert gate.blocked_resources() == []


def test_a_malformed_body_is_refused_before_the_slot_is_taken(harness):
    app, gate, upstream = harness

    async def run():
        sent = []
        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
                 "headers": [(b"authorization", b"Bearer cred-chat")]}

        async def receive():
            return {"type": "http.request", "body": b"{not json", "more_body": False}

        async def send(message):
            sent.append(message)
        await app(scope, receive, send)
        return sent

    sent = asyncio.run(run())
    assert sent[0]["status"] == 400
    assert gate.blocked_resources() == []
    assert upstream.calls == []


def test_an_oversized_body_is_refused(harness):
    app, gate, upstream = harness

    async def run():
        sent = []
        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
                 "headers": [(b"authorization", b"Bearer cred-chat")]}
        chunks = iter([{"type": "http.request", "body": b"x" * 500_000, "more_body": True}]
                      * 6 + [{"type": "http.disconnect"}])

        async def receive():
            return next(chunks)

        async def send(message):
            sent.append(message)
        await app(scope, receive, send)
        return sent

    assert asyncio.run(run())[0]["status"] == 413
    assert gate.blocked_resources() == []


def test_repeated_calls_serialise_onto_one_slot(harness):
    """The gate's whole purpose: no two memory requests share the device."""
    app, gate, upstream = harness
    statuses = []
    for _ in range(4):
        response = call(app)
        statuses.append(response["status"])
        if response["status"] == 200:
            # Hold it, as a worker would: the next caller must be told to back off.
            held = gate.db.execute(
                "SELECT id FROM gate_reservations WHERE state='released' ORDER BY id DESC LIMIT 1")
            assert held.fetchone() is not None
    assert statuses[0] == 200


def test_usage_is_counted_from_what_the_upstream_reported(harness):
    app, gate, upstream = harness
    upstream.body = json.dumps({"usage": {"total_tokens": 1234}}).encode()
    call(app)
    assert gate.usage()[REMOTE]["tokens"] == 1234


@pytest.mark.parametrize("body, expected", [
    (b'{"usage": {"total_tokens": 7}}', 7), (b"{}", 0), (b"not json", 0),
    (b'{"usage": "nonsense"}', 0), (b'{"usage": {"total_tokens": null}}', 0),
    (b"", 0),
])
def test_token_accounting_survives_junk(body, expected):
    assert _usage_tokens(body) == expected


# -- where a forwarded path goes ---------------------------------------------

def _route(upstream, credential):
    return Route("retain", REMOTE, "chat", upstream, credential, "freshness", 2048)


def test_an_upstream_named_by_its_versioned_root_is_not_asked_for_that_root_twice():
    """The engine configures `http://host/v1`; the forwarded path says `/v1/embeddings`.

    Joined plainly that is `/v1/v1/embeddings`, which a model server answers with a 404 the
    gate passes straight on — an admitted request failing for a reason nobody can see.
    """
    versioned = _route("http://127.0.0.1:8080/v1", "cred-a")
    trailing = _route("http://127.0.0.1:8080/v1/", "cred-b")
    plain = _route("http://127.0.0.1:8080", "cred-c")
    served_under_a_path = _route("http://127.0.0.1:8080/engine/v1", "cred-d")
    assert upstream_url(versioned, "/v1/chat/completions") == \
        "http://127.0.0.1:8080/v1/chat/completions"
    assert upstream_url(trailing, "/v1/embeddings") == "http://127.0.0.1:8080/v1/embeddings"
    assert upstream_url(plain, "/v1/chat/completions") == \
        "http://127.0.0.1:8080/v1/chat/completions"
    assert upstream_url(served_under_a_path, "/v1/embeddings") == \
        "http://127.0.0.1:8080/engine/v1/embeddings"


# -- waiting for a busy device ------------------------------------------------

def queued_harness(store, *, queue_s=5.0):
    """The same gate, with the wait an installation is configured to give it."""
    gate = ResourceGate(store)
    upstream = Upstream()
    app = GateApp(routes=TABLE, gate=gate, upstream_credentials=CREDENTIALS,
                  upstream=upstream, queue_s=queue_s)
    return app, gate, upstream


def hold(gate, *, resource, route="someone-else's-job", ttl=300.0):
    return gate.acquire(route=route, holder="test:holder", resource=resource,
                        priority=1, ttl=ttl, timeout=0)


def hand_back(store, reservation, *, after=0.3):
    """Give the device back from another thread, the way another process would.

    Leaving the lease to expire is not the same situation: an expired lease becomes
    uncertain and keeps the resource blocked, and the point here is a slot that frees.
    """
    def release():
        time.sleep(after)
        with EvidenceStore(store.path) as other:
            ResourceGate(other).release(reservation, outcome="succeeded", tokens=1)

    threading.Thread(target=release, daemon=True).start()


def test_a_background_caller_waits_for_a_busy_device_and_is_then_served(store):
    """One slot per device serialises that device; it does not turn its callers away.

    The engine presents several sub-calls of one operation at a time, so a gate that
    refused on contact answered the second of them with a 429 the client does not
    re-drive — and the installation could never form anything at all.
    """
    app, gate, upstream = queued_harness(store)
    hand_back(store, hold(gate, resource=GPU))
    started = time.monotonic()
    response = call(app, "/v1/embeddings", token="cred-emb",
                    body={"input": "hello", "model": "emb"})
    assert response["status"] == 200, response["body"]
    assert len(upstream.calls) == 1
    assert time.monotonic() - started >= 0.2, "the caller was served without waiting"


def test_a_wait_that_runs_out_says_what_it_waited_for(store):
    app, gate, upstream = queued_harness(store, queue_s=0.2)
    hold(gate, resource=GPU)
    response = call(app, "/v1/embeddings", token="cred-emb",
                    body={"input": "hello", "model": "emb"})
    assert response["status"] == 429
    assert "waited 0.2s" in response["body"]["error"]["message"]
    assert upstream.calls == []
    # The caller's own queue entry goes with its refusal, or every abandoned request
    # would hold the line behind it.
    assert gate.occupancy().get(GPU, {}).get("waiting", 0) == 0


def test_the_call_a_human_is_waiting_for_is_never_queued_behind_background_work(store):
    """Refusing the interactive route at once is what makes the host's degradation real."""
    app, gate, upstream = queued_harness(store, queue_s=30.0)
    hold(gate, resource=REMOTE)
    started = time.monotonic()
    response = call(app, token="cred-chat")
    assert response["status"] == 429
    assert "never waits" in response["body"]["error"]["message"]
    assert time.monotonic() - started < 1.0, "an interactive caller was made to queue"
    assert upstream.calls == []


def test_a_busy_device_is_reaped_before_the_wait_it_starts(store):
    """A lease that expired while this request was arriving is still not a free device.

    Waiting must not become a way around the rule that an unestablished completion keeps
    the slot: the waiter gets the same refusal a caller that never waited would get. And it
    gets it at once — a device nobody can answer for is not going to free by standing here,
    so the configured wait would only be a slower way of saying the same thing.
    """
    app, gate, upstream = queued_harness(store, queue_s=0.2)
    held = hold(gate, resource=GPU, ttl=60.0)
    # Another connection's clock is the only honest way to age a lease here: this gate
    # marks it uncertain itself, on its next attempt, from its own row.
    with EvidenceStore(store.path) as other:
        ResourceGate(other).mark_uncertain(held, reason="test: lease expired")
    started = time.monotonic()
    response = call(app, "/v1/embeddings", token="cred-emb",
                    body={"input": "hello", "model": "emb"})
    assert response["status"] == 429
    assert "nobody has established" in response["body"]["error"]["message"], \
        "the refusal says which kind of busy this is, and what would actually free it"
    assert time.monotonic() - started < 0.15, "no wait was spent on an unanswerable device"
    assert gate.occupancy()[GPU]["uncertain"] == 1
    assert upstream.calls == []
