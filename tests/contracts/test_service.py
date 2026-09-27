"""The runtime process: an ASGI gate on loopback, framed by the standard library.

Two families of claim are checked here. The first is the bind: what address is
admissible, and that the installation's own file answers rather than the shell. The
second is the adapter, which is new code standing between a socket and an app that is
already tested - so framing, status, headers, oversized bodies and a fault inside the
app all have to come out the way the app intended, with nothing that looks like an idle
slot when it is not.
"""
from __future__ import annotations

import http.client
import json
import socket
import sqlite3
import threading
import time

import pytest

from hermes_memory.config import SettingError, load_settings
from hermes_memory.processing.gate_server import UpstreamResult
from hermes_memory.processing.instance_gate import instance_gate
from hermes_memory.processing.resource_gate import ResourceGate
from hermes_memory.service import (HEALTH_PATH, GateServer, admission_url, bind_endpoint,
                                   gate_app_factory, serve)
from hermes_memory.storage.evidence import EvidenceStore

OWNER = "jugaadu"
ROUTES = ("retain", "consolidate", "reflect", "foreground", "embeddings")
TOKENS = {name: f"credential-{name}" for name in ROUTES}


def config(home, **extra):
    values = {"DATA_DIR": home / "data", "INFERENCE_ENABLED": "true",
              "HINDSIGHT_URL": "http://127.0.0.1:8888/v1",
              "ALLOWED_INFERENCE_HOSTS": "127.0.0.1",
              "TEXT_BASE_URL": "http://127.0.0.1:9001/v1",
              "EMBEDDINGS_BASE_URL": "http://127.0.0.1:9002/v1",
              "ADMISSION_URL": "http://127.0.0.1:8124",
              "OWNER_PRINCIPAL": OWNER}
    for name, credential in TOKENS.items():
        values[f"ROUTE_CREDENTIAL_{name.upper()}"] = credential
    values.update(extra)
    return "".join(f"HERMES_MEMORY_{key}={value}\n" for key, value in values.items())


def installed(home, monkeypatch, **extra):
    home.mkdir(parents=True, exist_ok=True)
    (home / "hermes-memory.env").write_text(config(home, **extra), encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    return load_settings()


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    for name in ROUTES:
        monkeypatch.delenv(f"HERMES_MEMORY_ROUTE_CREDENTIAL_{name.upper()}", raising=False)
    return installed(tmp_path / "instance", monkeypatch)


def boom(*args, **kwargs):
    raise AssertionError("no request in these tests reaches a model")


class Started:
    """A real socket on loopback, shut down when the test is done."""

    def __init__(self, factory):
        self.server = GateServer(("127.0.0.1", 0), factory)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return self.server.bound[1]

    def call(self, method, path, *, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.request(method, path,
                               body=json.dumps(body).encode()
                               if isinstance(body, dict) else body,
                               headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def raw(self, payload: bytes) -> tuple[str, bool]:
        """Send bytes a client library would not have framed for us.

        Returns the whole exchange and whether the server ended it. A refusal that
        leaves the connection open mid-body is not a refusal, and only an end-of-file
        says the difference apart from waiting for a timeout.
        """
        with socket.create_connection(("127.0.0.1", self.port), 10) as sock:
            sock.sendall(payload)
            sock.settimeout(5)
            collected = bytearray()
            closed = False
            try:
                while chunk := sock.recv(65_536):
                    collected.extend(chunk)
                closed = True
            except (TimeoutError, socket.timeout):
                pass
            return collected.decode("latin-1"), closed

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)


@pytest.fixture()
def gate(settings):
    started = Started(gate_app_factory(settings, upstream=boom))
    yield started
    started.close()


def chat(credential=TOKENS["foreground"], body=None, extra=None):
    payload = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(body or {})
    headers = {"authorization": f"Bearer {credential}"}
    headers.update(extra or {})
    return payload, headers


# -- where it may listen ------------------------------------------------------

def test_an_installation_with_no_admission_address_refuses_to_guess_one(tmp_path, monkeypatch):
    home = tmp_path / "bare"
    for name in ROUTES:
        monkeypatch.delenv(f"HERMES_MEMORY_ROUTE_CREDENTIAL_{name.upper()}", raising=False)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    home.mkdir()
    (home / "hermes-memory.env").write_text(
        config(home).replace("HERMES_MEMORY_ADMISSION_URL=http://127.0.0.1:8124\n", ""),
        encoding="utf-8")
    with pytest.raises(SettingError, match="ADMISSION_URL"):
        bind_endpoint(load_settings())


def test_the_bound_address_comes_from_the_file_and_not_from_the_shell(settings, monkeypatch):
    """A stray export must not move a service a unit file has already been written for."""
    monkeypatch.setenv("HERMES_MEMORY_ADMISSION_URL", "http://127.0.0.1:9999")
    assert admission_url(settings) == "http://127.0.0.1:8124"
    assert bind_endpoint(settings) == ("127.0.0.1", 8124)


@pytest.mark.parametrize("url,expected", [
    ("http://127.0.0.1:8124", ("127.0.0.1", 8124)),
    ("http://localhost:8124", ("localhost", 8124)),
    ("http://[::1]:8124", ("::1", 8124)),
    ("http://127.0.0.1", ("127.0.0.1", 80)),
])
def test_a_loopback_endpoint_is_bindable(tmp_path, monkeypatch, url, expected):
    for name in ROUTES:
        monkeypatch.delenv(f"HERMES_MEMORY_ROUTE_CREDENTIAL_{name.upper()}", raising=False)
    settings = installed(tmp_path / "loopback", monkeypatch, ADMISSION_URL=url)
    assert bind_endpoint(settings) == expected


@pytest.mark.parametrize("url,said", [
    ("http://0.0.0.0:8124", "not loopback"),
    ("http://192.168.4.20:8124", "not loopback"),
    ("http://169.254.169.254:80", "not loopback"),
    ("http://gate.example:8124", "not loopback"),
    ("https://127.0.0.1:8124", "no certificate"),
    ("gate.sock", "not an http"),
    ("http://", "not an http"),
])
def test_anything_that_is_not_a_loopback_socket_is_refused_by_name(tmp_path, monkeypatch,
                                                                   url, said):
    """Each shape gets its own reason, because each has a different fix.

    "Not loopback" is answered by editing the address; "not an http endpoint" is
    answered by noticing a socket path was pasted into the wrong key.
    """
    for name in ROUTES:
        monkeypatch.delenv(f"HERMES_MEMORY_ROUTE_CREDENTIAL_{name.upper()}", raising=False)
    settings = installed(tmp_path / "refused", monkeypatch, ADMISSION_URL=url)
    with pytest.raises(SettingError, match=said):
        bind_endpoint(settings)


# -- health ------------------------------------------------------------------

def test_health_answers_without_a_credential_and_asks_nothing_of_a_model(gate):
    status, headers, body = gate.call("GET", HEALTH_PATH)
    assert status == 200
    report = json.loads(body)
    assert report["ok"] is True
    assert report["inference_performed"] is False
    assert report["listening_on"][1] == gate.port
    assert headers["content-type"] == "application/json"


def test_health_reports_the_fence_as_the_gate_knows_it(settings):
    """The pause is instance state, so it is read from the ledger and not remembered."""
    with instance_gate(settings) as gate:
        gate.pause(actor=OWNER, reason="owner stopped formation")
    started = Started(gate_app_factory(settings, upstream=boom))
    try:
        report = json.loads(started.call("GET", HEALTH_PATH)[2])
        assert report["paused"] is True
        assert report["busy_resources"] == []
    finally:
        started.close()


def test_an_unauthenticated_forward_is_refused_before_the_gate_is_touched(gate):
    status, _, body = gate.call("POST", "/v1/chat/completions",
                                body={"model": "m", "messages": []})
    assert status == 401
    assert "credential" in json.loads(body)["error"]["message"]


# -- the adapter in front of the tested app ---------------------------------

def test_an_upstream_that_faults_is_answered_as_a_gate_fault_with_no_traceback(gate):
    """The adapter must not turn an exception into a half-written response.

    The app marks the reservation uncertain before answering, so what arrives here has
    to be a readable 500: a socket that dies mid-response is what a client retries into
    a second occupied slot.
    """
    payload, headers = chat()
    status, _, body = gate.call("POST", "/v1/chat/completions", body=payload, headers=headers)
    assert status == 500
    error = json.loads(body)["error"]
    assert error["type"] == "server_error"
    assert b"Traceback" not in body


def test_an_upstream_answer_travels_through_the_adapter_unchanged(settings):
    seen = []

    def upstream(url, raw, headers):
        seen.append((url, json.loads(raw), dict(headers)))
        return UpstreamResult(201, json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}).encode())

    started = Started(gate_app_factory(settings, upstream=upstream))
    try:
        payload, headers = chat()
        status, response_headers, body = started.call("POST", "/v1/chat/completions",
                                                      body=payload, headers=headers)
        assert status == 201
        assert json.loads(body)["choices"][0]["message"]["content"] == "ok"
        assert response_headers["content-type"] == "application/json"
        assert seen[0][0].endswith("/v1/chat/completions")
        assert seen[0][1]["messages"][0]["content"] == "hi"
    finally:
        started.close()


def test_a_path_the_gate_does_not_serve_is_named_back_as_a_refusal(gate):
    status, _, body = gate.call("POST", "/v1/proxy/anything", body={})
    assert status == 404
    message = json.loads(body)["error"]["message"]
    assert "/v1/chat/completions" in message and "/v1/embeddings" in message


def test_a_get_on_a_forwarded_path_is_still_a_method_error(gate):
    status, _, body = gate.call("GET", "/v1/embeddings")
    assert status == 404
    assert "by POST" in json.loads(body)["error"]["message"]


@pytest.mark.parametrize("method", ["PUT", "DELETE"])
def test_a_write_verb_reaches_the_refusal_rather_than_a_default_reply(gate, method):
    """`do_PUT`/`do_DELETE` exist so that this answer is ours.

    Without them the handler would reply 501 Unimplemented, which describes a stub
    rather than a gate with a named surface; the refusal names the verbs and paths
    this installation does serve, including the method and path that were asked for.
    """
    status, _, body = gate.call(method, "/v1/chat/completions", body={})
    assert status == 404
    message = json.loads(body)["error"]["message"]
    assert f"{method} '/v1/chat/completions'" in message
    assert "by POST" in message


def test_a_body_larger_than_the_gate_holds_is_answered_without_being_read():
    reads = []

    class Recorder:
        async def __call__(self, scope, receive, send):
            reads.append(scope["method"])

    started = Started(lambda: Recorder())
    # The length is what is checked, so a request that claims four megabytes is refused
    # before a single byte of it is read - which is also why the connection ends there.
    text, closed = started.raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                               b"Content-Length: 4000000\r\n\r\n{}")
    assert "413" in text.splitlines()[0]
    assert "exceeds" in text
    assert closed, "a refusal has to end the connection it will not resynchronise on"
    assert reads == [], "an oversized body must not reach the app at all"
    started.close()


def test_a_answer_with_a_length_arrives_is_framed_so_keep_alive_still_works(gate):
    """HTTP/1.1 without honest framing is a client that hangs, not one that retries."""
    connection = http.client.HTTPConnection("127.0.0.1", gate.port, timeout=10)
    try:
        for _ in range(3):
            connection.request("GET", HEALTH_PATH)
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["ok"] is True
        assert response.version == 11
    finally:
        connection.close()


def test_a_content_length_that_is_not_a_number_is_answered_as_a_bad_request(gate):
    text, closed = gate.raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                            b"Content-Length: twelve\r\n\r\n{}")
    assert "400" in text.splitlines()[0]
    assert "content-length" in text.lower()
    assert closed


def test_a_fault_outside_the_apps_control_is_still_an_answer(settings):
    """The app catches an upstream fault itself; this is the fault it cannot see.

    Something wrong in the adapter's own framing has to leave the caller with a status
    and a JSON error rather than a socket that stops in the middle of a response.
    """
    class Exploding:
        async def __call__(self, scope, receive, send):
            raise RuntimeError("framing broke")

    started = Started(lambda: Exploding())
    status, _, body = started.call("POST", "/v1/chat/completions", body={}, headers={})
    assert status == 500
    assert json.loads(body)["error"]["type"] == "server_error"
    started.close()


def test_an_app_that_answers_nothing_is_reported_as_a_bad_gateway():
    class Silent:
        async def __call__(self, scope, receive, send):
            return None

    started = Started(lambda: Silent())
    status, _, body = started.call("POST", "/v1/chat/completions", body={}, headers={})
    assert status == 502
    assert "answered nothing" in json.loads(body)["error"]["message"]
    started.close()


def test_the_cap_notice_the_gate_adds_arrives_with_the_answer(settings):
    def upstream(url, raw, headers):
        return UpstreamResult(200, b'{"choices": []}')

    started = Started(gate_app_factory(settings, upstream=upstream))
    payload, headers = chat(body={"max_tokens": 99_000})
    status, response_headers, _ = started.call("POST", "/v1/chat/completions", body=payload,
                                               headers=headers)
    assert status == 200
    assert int(response_headers["x-gate-max-tokens-capped"]) == \
        settings.max_output_tokens.foreground
    started.close()


def test_the_gate_is_served_per_thread_so_a_busy_device_is_still_answerable(settings):
    """One blocked request must not block the socket.

    Serialising models is the design; serialising the listening socket turns a busy GPU
    into an installation that looks dead to its own health check - and two sqlite
    connections sharing one thread's handle would raise before that mattered.
    """
    release = threading.Event()

    def slow(url, raw, headers):
        release.wait(10)
        return UpstreamResult(200, b'{"choices": []}')

    started = Started(gate_app_factory(settings, upstream=slow))
    payload, headers = chat()
    holder = threading.Thread(target=lambda: started.call("POST", "/v1/chat/completions",
                                                          body=payload, headers=headers))
    holder.start()
    time.sleep(0.3)
    try:
        status, _, body = started.call("GET", HEALTH_PATH)
        report = json.loads(body)
        assert status == 200
        assert report["busy_resources"], "the held slot is visible from outside"
    finally:
        release.set()
        holder.join(timeout=15)
        started.close()


# -- the process entry point --------------------------------------------------

def test_readiness_is_announced_only_after_the_socket_is_listening(settings):
    announced = []
    thread = threading.Thread(target=lambda: serve(
        settings, upstream=boom, host="127.0.0.1", port=0, report=announced.append),
        daemon=True)
    thread.start()
    for _ in range(400):
        if announced:
            break
        time.sleep(0.01)
    assert announced, "the process said nothing"
    port = announced[0]["listening_on"][1]
    assert port > 0, "readiness was announced before a real port was known"
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", HEALTH_PATH)
    assert connection.getresponse().status == 200
    assert set(announced[0]["routes"]) == set(ROUTES)
    assert announced[0]["routes_withheld"] == {}, "a healthy installation says so"
    thread.join(timeout=2)
    assert thread.is_alive(), "serving returned early"


def test_a_capture_only_installation_maps_nothing_and_refuses_everything(tmp_path, monkeypatch):
    """No upstream is configured, so the gate has no route to invent a default for.

    The process still has to come up - the units and the doctor ask it whether it is
    alive - and it must answer a forward with a refusal rather than a guess.
    """
    home = tmp_path / "capture-only"
    home.mkdir()
    (home / "hermes-memory.env").write_text(
        "".join(f"HERMES_MEMORY_{key}={value}\n" for key, value in {
            "DATA_DIR": home / "data", "INFERENCE_ENABLED": "false",
            "ADMISSION_URL": "http://127.0.0.1:8124"}.items()), encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    bare = load_settings()
    started = Started(gate_app_factory(bare))
    try:
        assert started.call("GET", HEALTH_PATH)[0] == 200
        status, _, body = started.call("POST", "/v1/chat/completions", body={},
                                       headers={"authorization": "Bearer anything"})
        assert status == 401
        assert "route" in json.loads(body)["error"]["message"]
    finally:
        started.close()


def test_the_store_a_request_opened_is_closed_when_that_connection_ends(settings):
    """Threads are created per connection, and each one borrowed a database handle.

    A server that leaks a connection per request would run out after a night of
    retries; the sqlite handle is released with the thread that made it.
    """
    opened = []

    def store_factory():
        store = EvidenceStore(settings.db_path)
        opened.append(store)
        return store

    started = Started(gate_app_factory(settings, store_factory=store_factory, upstream=boom))
    started.call("GET", HEALTH_PATH)
    started.close()
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].db.execute("SELECT 1")


# -- the background pass in the runtime process --------------------------------

def announced_by(thread_target, *, timeout=5.0):
    """Start serving in a thread and hand back what readiness said."""
    printed: list[dict] = []
    thread = threading.Thread(target=lambda: thread_target(printed.append), daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while not printed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert printed, "the process announced nothing"
    return printed


def test_the_runtime_process_runs_the_pass_the_configuration_asked_for(settings):
    """§4 puts proactive state and the schedule in this process, and readiness says so.

    The line matters because it is the only place an operator can see, from the running
    service, whether anything will ever act by itself.
    """
    printed = announced_by(
        lambda report: serve(settings, upstream=boom, host="127.0.0.1", port=0,
                            report=report))
    assert printed[0]["maintenance"]["interval_s"] == settings.maintenance_interval_s
    assert printed[0]["maintenance"]["running"] is True


def test_a_period_of_zero_hands_the_timer_back_without_starting_a_thread(tmp_path,
                                                                        monkeypatch):
    from hermes_memory.config import load_settings

    for name in ROUTES:
        monkeypatch.delenv(f"HERMES_MEMORY_ROUTE_CREDENTIAL_{name.upper()}", raising=False)
    quiet = installed(tmp_path / "quiet", monkeypatch, MAINTENANCE_INTERVAL_S="0")
    printed = announced_by(
        lambda report: serve(quiet, upstream=boom, host="127.0.0.1", port=0, report=report))
    assert load_settings().maintenance_interval_s == 0
    assert printed[0]["maintenance"] == {"interval_s": 0.0, "running": False, "passes": 0,
                                         "failures": 0, "last": {}, "never": True}


def test_the_loop_actually_runs_while_the_process_is_up(settings):
    from hermes_memory.processing.maintenance import Ticker

    calls: list[int] = []
    ticker = Ticker(runner=lambda: (calls.append(1), {"ok": True, "at": "then"})[1],
                    interval_s=0.02, first_delay_s=0.0)
    printed = announced_by(
        lambda report: serve(settings, upstream=boom, host="127.0.0.1", port=0,
                            report=report, maintenance=ticker))
    assert printed[0]["maintenance"]["running"] is True
    deadline = time.monotonic() + 5
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(calls) >= 2, "the process was up and the pass never ran"
    ticker.stop()


def test_a_process_that_is_leaving_takes_its_scheduler_with_it(settings):
    """The stop path, not the sleep path: a departing unit must not keep writing.

    ``serve`` cannot be told to stop from outside, and a test that left it running would
    prove nothing about shutdown, so the announcement itself is the interruption.
    """
    from hermes_memory.processing.maintenance import Ticker

    ticker = Ticker(runner=lambda: {"ok": True}, interval_s=60, first_delay_s=0.0)

    def stopping(payload):
        raise KeyboardInterrupt

    assert serve(settings, upstream=boom, host="127.0.0.1", port=0, report=stopping,
                 maintenance=ticker) == 0
    assert ticker.state()["running"] is False
