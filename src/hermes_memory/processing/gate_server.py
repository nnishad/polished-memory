"""C12 loopback admission endpoint: a route mapper, not a general proxy.

The whole point of putting something here rather than pointing Hindsight
straight at the model servers is that this is the one place a memory-originated
request can be counted, capped and serialized onto a physical device. So the
surface is deliberately tiny: two paths, credentials that select a route, and
no parameter by which a caller can name an upstream. Anything else is refused
without being forwarded, because a proxy that "just passes through" an unknown
path is a way around the single-slot rule.

ASGI and stdlib only. Tests drive this app directly, so no port is bound and no
server process is needed to prove the contract.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .resource_gate import GateBusy, GatePaused, ResourceGate
from .routes import RouteTable

__all__ = ["GateApp", "FORWARDED_PATHS", "UpstreamResult", "urllib_upstream",
           "upstream_url"]

# Only the subset memory actually needs. A general proxy would let a caller
# reach an arbitrary endpoint through the slot we reserved.
FORWARDED_PATHS = {
    "/v1/chat/completions": "chat",
    "/v1/embeddings": "embeddings",
}
MAX_BODY_BYTES = 2_000_000


@dataclass(frozen=True)
class UpstreamResult:
    status: int
    body: bytes = b""
    transport_error: str | None = None

    @property
    def reached(self) -> bool:
        return self.transport_error is None


def urllib_upstream(timeout: float = 600.0) -> Callable:
    """The only socket in this module. Upstream credentials are attached here."""

    def call(url: str, body: bytes, headers: dict[str, str]) -> UpstreamResult:
        request = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return UpstreamResult(response.status, response.read())
        except urllib.error.HTTPError as error:
            return UpstreamResult(error.code, error.read() or b"{}")
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            return UpstreamResult(0, transport_error=str(error)[:400])

    return call


class GateApp:
    """An ASGI application presenting the approved routes behind the resource gate."""

    def __init__(self, *, routes: RouteTable, gate: ResourceGate, upstream_credentials,
                 upstream: Callable | None = None,
                 token_estimate: Callable[[dict], int] | None = None,
                 queue_s: float = 0.0, store: Any = None):
        self.routes = routes
        self.gate = gate
        self.store = store
        self.queue_s = queue_s
        self.upstream_credentials = dict(upstream_credentials)
        self.upstream = upstream or urllib_upstream()
        self.token_estimate = token_estimate or _estimate_tokens

    def wait_for(self, route) -> float:
        """How long this caller may stand in the queue for its device.

        An interactive request is the one the host is already late for: waiting behind a
        consolidation run would trade a visible refusal for an invisible overrun, so it keeps
        the immediate answer and the caller degrades. Everything else is work with no human on
        the other end of it, and a device held by one caller is a reason to be next, not a
        reason to fail — the engine presents several sub-calls of one operation at once, each
        inside its own concurrency limit, so refusing on contact stops the installation
        forming anything at all.
        """
        return 0.0 if route.priority == "interactive" else self.queue_s

    def close(self) -> None:
        """Release what this application was built from.

        The gate and the evidence store are usually two different databases now — one
        admission ledger for the installation, one archive per profile — and closing one
        while leaving the other open is a leak with a plausible name.
        """
        self.gate.close()
        close = getattr(self.store, "close", None)
        if close is not None:
            close()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await _respond(send, 404, {"error": {"message": "not an http scope"}})
            return
        path = scope.get("path", "")
        if scope["method"] != "POST" or path not in FORWARDED_PATHS:
            # Named rather than open: this tells a misconfigured caller exactly
            # what is allowed instead of failing opaquely upstream.
            allowed = ", ".join(sorted(FORWARDED_PATHS))
            refused = f"{scope['method']} {path!r}"
            await _respond(send, 404, {"error": {
                "message": f"the gate serves only [{allowed}] by POST; {refused} is refused"}})
            return

        credential = _bearer(scope.get("headers", []))
        try:
            route = self.routes.by_credential(credential)
        except Exception as error:
            await _respond(send, 401, {"error": {"message": str(error)[:300]}})
            return

        raw = await _body(receive)
        if raw is None:
            await _respond(send, 413, {"error": {"message": "request body exceeds the gate"}})
            return
        try:
            payload = json.loads(raw or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
        except (json.JSONDecodeError, ValueError) as error:
            await _respond(send, 400, {"error": {"message": f"unparseable body: {error}"[:300]}})
            return

        payload, capped = _apply_cap(payload, route.max_output_tokens)
        reservation = None
        waited = self.wait_for(route)
        try:
            reservation = self.gate.acquire(
                route=route.name, holder=f"gate:{route.name}", resource=route.resource,
                priority=route.priority_rank(), ttl=max(60.0, self.gate.default_ttl),
                timeout=waited)
        except GatePaused as error:
            await _respond(send, 503, {"error": {"message": str(error)[:300], "type": "paused"}})
            return
        if reservation is None:
            # 429 with Retry-After: the caller must back off, not assume failure
            # and resend into an occupied device. For a queued route this is said
            # only after its wait ran out, so the number is in the message — and a
            # device nobody can answer for is named as such, because no retry by this
            # caller is what frees it.
            blocked = self.gate.unresolved_for(route.resource)
            message = (
                f"{route.resource} is blocked by a request whose outcome nobody has "
                "established; waiting on it would not free it" if blocked else
                f"{route.resource} is busy" + (
                    f": waited {waited:g}s for the slot" if waited else
                    ", and this route never waits for one"))
            await _respond(send, 429, {"error": {"message": message, "type": "rate_limit"}},
                           headers=[(b"retry-after", b"2")])
            return

        try:
            result = self.upstream(upstream_url(route, path),
                                   json.dumps(payload).encode("utf-8"),
                                   _upstream_headers(route, self.upstream_credentials))
        except Exception as error:  # a defect here must not free a possibly-busy slot
            self.gate.mark_uncertain(reservation, reason=f"gate fault: {error}"[:400])
            await _respond(send, 500, {"error": {"message": "gate fault", "type": "server_error"}})
            return

        if not result.reached:
            # Nothing proves the upstream is idle. Keep the slot blocked and say
            # so plainly rather than reporting a clean failure.
            self.gate.mark_uncertain(reservation, reason=result.transport_error[:400])
            await _respond(send, 504, {"error": {"message": "upstream unreachable; execution "
                                                            "state unknown",
                                                 "type": "uncertain"}})
            return

        tokens = _usage_tokens(result.body)
        self.gate.release(reservation, outcome="succeeded" if result.status < 400 else "failed",
                          tokens=tokens)
        headers = [(b"content-type", b"application/json")]
        if capped:
            # Telling the caller we shrank its request is the difference between
            # a surprising answer and a silently wrong one.
            headers.append((b"x-gate-max-tokens-capped", str(capped).encode("ascii")))
        await _respond(send, result.status, result.body, headers=headers, raw=True)


def upstream_url(route, path: str) -> str:
    """Where a forwarded path actually goes, without asking for ``/v1/v1/embeddings``.

    An OpenAI-compatible upstream is configured by its versioned root — ``http://host/v1`` —
    which is what every client library and this installation's own template names. The paths
    this gate forwards carry that prefix, so joining them plainly doubled it and the model
    server answered 404 to a request the gate had already admitted.
    """
    base = route.upstream.rstrip("/")
    if base.endswith("/v1") and path.startswith("/v1/"):
        base = base[:-3]
    return f"{base}{path}"


def _apply_cap(payload: dict[str, Any], limit: int) -> tuple[dict[str, Any], int | None]:
    """Force an output ceiling on generation.

    The client's own value is honoured only when it is smaller. An absent or
    enormous max_tokens is what let one local 4B request ask for an unbounded
    completion, and a cap that some paths set and others omit is no cap.
    """
    if limit <= 0:
        return payload, None
    requested = payload.get("max_tokens")
    if isinstance(requested, int) and 0 < requested <= limit:
        return payload, None
    adjusted = dict(payload)
    adjusted["max_tokens"] = limit
    adjusted.setdefault("extra_body", {})
    if isinstance(adjusted["extra_body"], dict):
        adjusted["extra_body"] = {**adjusted["extra_body"], "enable_thinking": False}
    return adjusted, limit


def _estimate_tokens(payload: dict[str, Any]) -> int:
    content = json.dumps(payload.get("messages") or payload.get("input") or "", sort_keys=True)
    # A conservative characters-per-token heuristic. Under-counting is the
    # dangerous direction, so this rounds up.
    return max(1, len(content) // 3)


def _bearer(headers) -> str:
    for name, value in headers:
        if name == b"authorization":
            text = value.decode("latin-1").strip()
            return text[7:].strip() if text.lower().startswith("bearer ") else text
    return ""


def _upstream_headers(route, credentials: dict[str, str]) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    secret = credentials.get(route.resource)
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    return headers


def _usage_tokens(body: bytes) -> int:
    try:
        parsed = json.loads(body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    usage = parsed.get("usage") if isinstance(parsed, dict) else None
    if not isinstance(usage, dict):
        return 0
    try:
        return max(0, int(usage.get("total_tokens") or 0))
    except (TypeError, ValueError):
        return 0


async def _body(receive) -> bytes | None:
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return b""
        raw = message.get("body", b"")
        chunks.append(raw)
        total += len(raw)
        if total > MAX_BODY_BYTES:
            return None
        if not message.get("more_body"):
            return b"".join(chunks)


async def _respond(send, status: int, payload, *, headers=None, raw: bool = False) -> None:
    body = payload if raw else json.dumps(payload).encode("utf-8")
    base = [(b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii"))]
    await send({"type": "http.response.start", "status": status,
                "headers": base + list(headers or [])})
    await send({"type": "http.response.body", "body": body})
