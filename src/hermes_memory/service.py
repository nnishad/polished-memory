"""The process the runtime unit starts: the admission endpoint, and the background pass.

The gate exists so that one place counts what the machine can actually run. For that to
be true it has to be a process rather than a library — several Hermes activities, the
backend and its worker all ask the same device — and a process needs something to speak
to. So this is the smallest possible server: an ASGI adapter over the standard library,
bound to loopback, forwarding only the two paths the gate knows, plus one unauthenticated
health line that proves the socket is answering without spending a token.

Alongside it runs the maintenance pass, on a period the owner sets. It is here rather than
in a unit of its own because proactive state belongs to this process, and because the pass
takes no slot, spends no budget and asks no model — the two facts that make it safe to
start without waiting for an approval this process has no way to carry. A loop that dies is
reported: the pass writes down each time it runs, and status and doctor read that line.

There is no dependency on an ASGI server here on purpose. ``dependencies`` in this
package is empty, and an installer that quietly pulled a web framework into a memory
runtime would own the CVEs that came with it.
"""
from __future__ import annotations

import asyncio
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from .config import DEFAULT_ENV_FILENAME, SettingError, env_file_values
from .processing.gate_server import GateApp
from .processing.maintenance import Ticker, run as maintenance_run
from .processing.resource_gate import ResourceGate
from .processing.routes import RouteTable, build_routes
from .storage.evidence import EvidenceStore

__all__ = ["HEALTH_PATH", "LOOPBACK", "bind_endpoint", "build_app", "gate_app_factory",
           "GateServer", "serve"]

# Loopback only. A gate on a LAN address is a credentialed proxy that anybody on the
# network can occupy a GPU with; the LAN allowance in config is for the upstreams behind
# it, never for this front door.
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
HEALTH_PATH = "/health"
MAX_REQUEST_BYTES = 2_000_000


def admission_url(settings) -> str | None:
    """The endpoint this installation named for admission, read from its own file.

    The process environment is deliberately not consulted: which socket a running
    service binds is a fact about the installation, and a stray export in the shell that
    happened to start it is a different system's opinion.
    """
    values = env_file_values(Path(settings.home) / DEFAULT_ENV_FILENAME)
    return values.get("HERMES_MEMORY_ADMISSION_URL")


def bind_endpoint(settings) -> tuple[str, int]:
    """Where the gate will listen, refusing anything that is not loopback.

    An unset endpoint is not a job for a default port: the unit, the backend
    configuration and the doctor all have to name the same address, so the owner writes
    it down once and this refuses to guess.
    """
    url = admission_url(settings)
    if not url:
        raise SettingError(
            "no HERMES_MEMORY_ADMISSION_URL is configured for this installation, so the "
            "gate has no address to bind; refusing to start on a guessed port")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise SettingError(f"{url!r} is not an http(s) endpoint the gate can bind")
    if parsed.hostname not in LOOPBACK:
        raise SettingError(
            f"the admission endpoint names {parsed.hostname}, which is not loopback; the "
            "gate is a credentialed endpoint and does not listen on a network")
    if parsed.scheme == "https":
        raise SettingError("the gate serves plaintext on loopback; it holds no certificate, "
                           "and must not be reached over TLS through somebody else's proxy")
    return parsed.hostname, parsed.port or 80


def build_app(settings, *, store, upstream: Callable | None = None,
              gate: ResourceGate | None = None) -> GateApp:
    """The gate as this installation's configuration describes it.

    ``store`` is required rather than created so that the caller decides which database
    is opened and when it is closed; ``upstream`` is injected so no test reaches a model.
    The admission ledger is *not* the evidence store: one installation, one queue for its
    physical models, however many profiles are standing in it.
    """
    from .processing.instance_gate import instance_gate

    routes = (RouteTable({}) if not settings.hindsight_url else
              build_routes(settings, credentials=settings.route_credentials))
    return GateApp(routes=routes, gate=gate or instance_gate(settings),
                   upstream_credentials=settings.route_credentials,
                   queue_s=settings.gate_queue_s,
                   upstream=upstream, store=store)


class GateServer(ThreadingHTTPServer):
    """A server that answers health without touching the gate, and everything else via ASGI."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], factory: Callable[[], GateApp]) -> None:
        # A factory rather than an app: the gate holds a database connection, and a
        # connection belongs to the thread that opened it.
        self._factory = factory
        self._local = threading.local()
        super().__init__(address, _Handler)

    @property
    def bound(self) -> tuple[str, int]:
        return str(self.server_address[0]), int(self.server_address[1])

    def app(self) -> GateApp:
        """This thread's gate, opened once per connection.

        Serialising the whole socket behind one request would turn a busy GPU into an
        installation that looks dead to its own health check, so the server is threaded;
        sharing one sqlite connection across those threads is the other way to fail.
        """
        holder = getattr(self._local, "app", None)
        if holder is None:
            holder = self._local.app = self._factory()
        return holder

    def shutdown_request(self, request) -> None:
        try:
            super().shutdown_request(request)
        finally:
            holder = getattr(self._local, "app", None)
            close = getattr(holder, "close", None)
            if close is not None:
                close()
            self._local.app = None


def gate_app_factory(settings, *, store_factory: Callable[[], Any] | None = None,
                     upstream: Callable | None = None) -> Callable[[], GateApp]:
    """How this installation builds a gate, for whoever is about to serve a request."""

    def build() -> GateApp:
        store = (store_factory or (lambda: EvidenceStore(settings.db_path)))()
        return build_app(settings, store=store, upstream=upstream)

    return build


class _Handler(BaseHTTPRequestHandler):
    """One request per thread, framed as HTTP/1.1 with an explicit length on every answer.

    The request line is the only thing logged: a credential arrives as a bearer header,
    so anything wider would put a key in a journal that outlives it.
    """

    server: GateServer
    protocol_version = "HTTP/1.1"

    # -- the surface ---------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - the name is the http verb
        if urllib.parse.urlsplit(self.path).path != HEALTH_PATH:
            self._run_asgi("GET")
            return
        # Answered here rather than in the app so that a paused or fully occupied gate
        # still says "the socket is alive" without asking anything of a model.
        gate = self.server.app().gate
        self._send(200, json.dumps({"ok": True, "service": "hermes-memory-gate",
                                    "listening_on": list(self.server.bound),
                                    "paused": gate.paused,
                                    "busy_resources": gate.blocked_resources(),
                                    "inference_performed": False}).encode())

    def do_POST(self) -> None:  # noqa: N802
        self._run_asgi("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._run_asgi("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._run_asgi("DELETE")

    # -- the plumbing --------------------------------------------------------

    def _run_asgi(self, method: str) -> None:
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            self._refuse(400, "unparseable content-length")
            return
        if length > MAX_REQUEST_BYTES:
            # The gate bounds a body before it can hold a slot open. The body is not
            # read, so the connection has to end here rather than resync on rubbish.
            self._refuse(413, "request body exceeds the gate")
            return
        raw = self.rfile.read(length) if length > 0 else b""
        parsed = urllib.parse.urlsplit(self.path)
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": self.protocol_version, "method": method,
            "path": urllib.parse.unquote(parsed.path), "query_string": parsed.query.encode(),
            "root_path": "", "scheme": "http",
            "server": list(self.server.bound), "client": [self.client_address[0], None],
            "headers": [(str(key).lower().encode("latin-1"), str(value).encode("latin-1"))
                        for key, value in self.headers.items()],
        }
        state: dict[str, Any] = {"status": None, "headers": [], "body": bytearray()}

        async def receive():
            if state.get("delivered"):
                return {"type": "http.disconnect"}
            state["delivered"] = True
            return {"type": "http.request", "body": raw, "more_body": False}

        async def send(message):
            kind = message.get("type")
            if kind == "http.response.start":
                state["status"] = int(message["status"])
                state["headers"] = [(key.decode("latin-1"), value.decode("latin-1"))
                                    for key, value in message.get("headers", [])]
            elif kind == "http.response.body":
                state["body"].extend(message.get("body") or b"")

        try:
            asyncio.run(self.server.app()(scope, receive, send))
        except Exception as error:
            # A fault inside the gate must not look like an idle slot; the app itself
            # marks the reservation uncertain before it answers 500, so all that is left
            # here is to say so without a stack trace into a log.
            self._send(500, json.dumps({"error": {"message": f"gate fault: {error}"[:300],
                                                  "type": "server_error"}}).encode())
            return
        if state["status"] is None:
            self._send(502, b'{"error": {"message": "the gate answered nothing"}}')
            return
        self._send(state["status"], bytes(state["body"]), state["headers"])

    def _refuse(self, status: int, message: str) -> None:
        """Answer and end the connection: nothing was read that we did not consume."""
        self.close_connection = True
        self._send(status, json.dumps({"error": {"message": message, "type": "client_error"}})
                   .encode())

    def _send(self, status: int, body: bytes, headers: list | None = None) -> None:
        self.send_response(status)
        for key, value in (headers or []):
            if key.lower() in ("content-length", "transfer-encoding"):
                continue
            self.send_header(key, value)
        self.send_header("content-length", str(len(body)))
        self.send_header("content-type", "application/json")
        self.end_headers()
        if body:
            self.wfile.write(body)


def serve(settings, *, store_factory: Callable[[], Any] | None = None,
          upstream: Callable | None = None, host: str | None = None,
          port: int | None = None, report: Callable[[dict], Any] | None = None,
          maintenance: Ticker | None = None) -> int:
    """Bind, announce readiness, then block until the manager stops us.

    Binding before printing anything is the point: a unit that announced itself and then
    failed to listen would look like a running service to everything that ordered after
    it. The readiness line carries no credential and no path to private data.
    """
    address = (host, port) if host is not None and port is not None \
        else bind_endpoint(settings)
    factory = gate_app_factory(settings, store_factory=store_factory, upstream=upstream)
    server = GateServer(address, factory)
    announcing = factory()
    try:
        routes = announcing.routes.names()
        withheld = announcing.routes.withheld
    finally:
        store = getattr(getattr(announcing, "gate", None), "store", None)
        if store is not None and hasattr(store, "close"):
            store.close()
    # The background pass runs in the process that owns proactive state, per §4, and not
    # in a unit of its own. It spends no model, no device slot and no budget, which is what
    # makes it safe to start here rather than to wait for somebody to approve a run. The one
    # backend traffic it may generate is an erasure obligation the owner already confirmed:
    # deleting a derived copy is not inference, and leaving it to wait for a model pass would
    # mean a forgetting that quietly never happened.
    ticker = maintenance or Ticker(runner=lambda: maintenance_run(settings),
                                   interval_s=getattr(settings, "maintenance_interval_s", 0))
    ticker.start()
    try:
        bound_host, bound_port = server.bound
        ready = {"ok": True, "service": "hermes-memory-gate",
                 "listening_on": [bound_host, bound_port], "health_path": HEALTH_PATH,
                 "routes": routes, "routes_withheld": withheld,
                 "maintenance": ticker.state()}
        (report or (lambda payload: print(json.dumps(payload, sort_keys=True))))(ready)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Whatever ended the loop — a signal, a stopped server or a failed announcement —
        # the scheduler stops with it, so a process that is leaving cannot keep writing.
        ticker.stop()
        server.server_close()
    return 0
