"""C5 Hindsight bridge: explicit, capability-checked, reconcilable.

Three rules shape this client. It never starts, installs or upgrades a backend
— an unavailable Hindsight produces degraded context and durable queued work,
not a background package operation from inside a plugin. It will not send a
request the pinned revision cannot answer. And it assumes every acknowledged
submission may have been lost on the way back, so an async retain requires a
caller-persisted submission identity before the request is made, and a timeout
is reconciled through the operations API rather than treated as a failure.

The transport is injected, so the whole contract is exercised offline.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..ids import document_id_is_ambiguous
from .capabilities import (CAPABILITIES, PINNED_VERSION, Capabilities,
                           UnsupportedCapability, capabilities_for)

__all__ = ["HindsightClient", "HindsightError", "HindsightUnavailable", "SubmissionConflict",
           "RecallOutcome", "TransportResult", "http_transport", "operation_state",
           "operation_reason"]

DEFAULT_TIMEOUT_S = 20.0
# Kept well under Hindsight's own limits: a query the backend rejects for
# length is a wasted round trip, and a truncated result must be visible to us.
_MAX_QUERY_TOKENS = 512


def operation_state(answer: dict[str, Any]) -> str:
    """What the backend said about an operation, in the pinned release's own words.

    An answer with no state is ``""`` rather than a guess: both callers have to decide what
    to do with a thing they cannot name, and "unknown" would read like a state the engine
    reports when it has none of its own.
    """
    if not isinstance(answer, dict):
        return ""
    return str(answer.get("status") or answer.get("state") or "")


def operation_reason(answer: dict[str, Any]) -> str:
    """Why an operation ended the way it did, when the backend gave a why at all."""
    reason = str(answer.get("error_message") or "").strip() if isinstance(answer, dict) else ""
    return reason


class HindsightError(Exception):
    def __init__(self, message: str, *, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class HindsightUnavailable(HindsightError):
    """No usable backend. The caller degrades; nobody starts one from here."""


class SubmissionConflict(HindsightError):
    """A submission identity was reused for different content: HTTP 409."""


@dataclass(frozen=True)
class TransportResult:
    status: int
    body: Any = None
    transport_error: str | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and self.transport_error is None


def http_transport(timeout: float = DEFAULT_TIMEOUT_S):
    """The only place a socket is opened. Loopback or approved LAN, never a fallback."""

    def call(method: str, url: str, payload: dict | None, headers: dict) -> TransportResult:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                return TransportResult(response.status, _decode(raw))
        except urllib.error.HTTPError as error:
            return TransportResult(error.code, _decode(error.read() or b"{}"))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            # Deliberately not converted into "failed": the request may have
            # reached the backend and be running right now.
            return TransportResult(0, transport_error=str(error)[:400])

    return call


class HindsightClient:
    def __init__(self, *, base_url: str, bank_id: str, api_key: str | None = None,
                 transport: Callable | None = None, capabilities: Capabilities | None = None,
                 timeout: float = DEFAULT_TIMEOUT_S):
        if not base_url or not bank_id:
            raise HindsightError("base_url and bank_id must be configured explicitly")
        self.base_url = base_url.rstrip("/")
        self.bank_id = bank_id
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport or http_transport(timeout)
        self._capabilities = capabilities

    # -- handshake -----------------------------------------------------------

    def health(self) -> dict[str, Any]:
        result = self._call("GET", "/health", None, capability=None)
        if result.transport_error or not result.ok:
            raise HindsightUnavailable(
                f"Hindsight at {self.base_url} is unreachable: "
                f"{result.transport_error or f'HTTP {result.status}'}")
        return result.body if isinstance(result.body, dict) else {}

    def negotiate(self, *, probe_routes: bool = True) -> Capabilities:
        """Record what the backend actually serves before depending on it."""
        if self._capabilities is not None:
            return self._capabilities
        version = PINNED_VERSION
        observed = "declared"
        try:
            report = self.health()
            found = str(report.get("version") or report.get("api_version") or "").strip()
            if found:
                version, observed = found, "health"
        except HindsightUnavailable:
            raise
        routes = None
        if probe_routes:
            # A version string alone does not prove an endpoint exists; a 404 does prove the
            # opposite. So the read-only route families are asked about, and what comes back
            # is subtracted from the table — it is a partial look, not a census, and three
            # probes must not be allowed to say that a backend which has been answering
            # `retain` for hours does not route it.
            routes = set()
            for name in ("list_banks", "bank_stats", "list_operations"):
                probe = self._call("GET", self._path_for(name), None, capability=None)
                if probe.status == 404:
                    routes.add(name)
        self._capabilities = capabilities_for(version, ruled_out=routes)
        self._capabilities = Capabilities(**{**self._capabilities.__dict__,
                                             "observed_via": observed})
        return self._capabilities

    # -- writes --------------------------------------------------------------

    def retain(self, *, document_id: str, content: str, timestamp: str | None = None,
               metadata: dict[str, str] | None = None, tags: list[str] | None = None,
               context: str | None = None) -> dict[str, Any]:
        """One bounded synchronous retain for exactly one canonical document.

        A single document per call is deliberate: batching unrelated messages
        into one fake document is how derived facts become impossible to trace
        back to the revision that produced them.
        """
        self.capabilities.require("retain")
        if document_id_is_ambiguous(document_id):
            raise HindsightError(
                "document_id must be nonempty and free of '_' and '~': Hindsight escapes "
                "them when composing chunk IDs, and an ambiguous chunk ID cannot be mapped "
                "back to a canonical revision")
        item: dict[str, Any] = {"content": content, "document_id": document_id}
        if timestamp:
            item["timestamp"] = timestamp
        if context:
            item["context"] = context
        if metadata:
            item["metadata"] = {key: str(value) for key, value in metadata.items()}
        if tags:
            item["tags"] = list(tags)
        result = self._call("POST", self._path_for("retain"), {"items": [item], "async": False},
                            capability="retain")
        return self._unwrap(result, "retain")

    def retain_async(self, items: list[dict[str, Any]], *, submission_id: str) -> dict[str, Any]:
        """Queue work that outlives this call. Requires an identity you already persisted.

        Refusing to proceed without one is the point: without a durable
        submission id, a lost acknowledgement is indistinguishable from a lost
        request, and the only safe recovery is to not retry at all.
        """
        self.capabilities.require("retain_async_identity")
        if not submission_id:
            raise HindsightError(
                "retain_async requires a submission_id persisted before the request; a lost "
                "acknowledgement would otherwise be unreconcilable")
        try:
            uuid.UUID(submission_id)
        except (ValueError, AttributeError, TypeError) as error:
            raise HindsightError("submission_id must be a UUID string") from error
        if not items or len(items) > 200:
            raise HindsightError("an async retain carries between 1 and 200 items")
        payload = {"items": items, "async": True, "operation_id": submission_id}
        result = self._call("POST", self._path_for("retain"), payload,
                            capability="retain_async_identity")
        if result.status == 409:
            raise SubmissionConflict(
                f"submission {submission_id} already exists for different content; reconciling "
                "through the operations API rather than resubmitting")
        return self._unwrap(result, "retain_async")

    def delete_document(self, document_id: str) -> dict[str, Any]:
        """Satisfies one backend_document erasure obligation."""
        self.capabilities.require("delete_memories")
        result = self._call("DELETE", self._path_for("delete_memories"),
                            {"document_ids": [document_id]}, capability="delete_memories")
        if result.status == 404:
            # Absent is already gone: verifying by absence is legitimate here,
            # so long as the caller treats it as verified rather than skipped.
            return {"deleted": False, "absent": True, "document_id": document_id}
        return self._unwrap(result, "delete_memories")

    # -- reads ---------------------------------------------------------------

    def recall(self, query: str, *, types: list[str] | None = None, max_tokens: int = 2048,
               budget: str = "mid", tags: list[str] | None = None,
               query_timestamp: str | None = None, temporal_window: dict | None = None,
               include_source_facts: bool = True, include_chunks: bool = False,
               source_facts_tokens: int = 1024, chunks_tokens: int = 1024) -> "RecallOutcome":
        """Ask for provenance explicitly, and report what came back truncated.

        ``include`` is a nested object with its own token budgets — the internal
        Python flag names are not the HTTP payload shape, so a flat
        ``include_source_facts`` key would simply be ignored and we would believe
        we had provenance when we had none.
        """
        self.capabilities.require("recall")
        if not query.strip():
            raise HindsightError("recall query must not be empty")
        if len(query) > 4000:
            raise HindsightError("recall query is too long; narrow it instead of over-fetching")
        include: dict[str, Any] = {"entities": {"max_tokens": 512}}
        if include_source_facts:
            include["source_facts"] = {"max_tokens": source_facts_tokens}
        if include_chunks:
            include["chunks"] = {"max_tokens": chunks_tokens}
        payload: dict[str, Any] = {"query": query, "budget": budget, "max_tokens": max_tokens,
                                   "include": include}
        if types:
            payload["types"] = list(types)
        if tags:
            payload["tags"] = list(tags)
        if query_timestamp:
            payload["query_timestamp"] = query_timestamp
        if temporal_window:
            payload["temporal_window"] = temporal_window
        body = self._unwrap(self._call("POST", self._path_for("recall"), payload,
                                       capability="recall"), "recall")
        return RecallOutcome.from_body(body)

    def consolidate(self, *, observation_scopes: list[list[str]]) -> dict[str, Any]:
        """Explicit scoped async submission; not a scheduler or authority grant."""
        self.capabilities.require("consolidate")
        if not isinstance(observation_scopes, list) or not 1 <= len(observation_scopes) <= 32:
            raise HindsightError("consolidation requires 1–32 explicit tag scopes")
        for scope in observation_scopes:
            if (not isinstance(scope, list) or not 1 <= len(scope) <= 16
                    or any(not isinstance(tag, str) or not tag.strip() or len(tag) > 200
                           for tag in scope)):
                raise HindsightError("each consolidation scope requires 1–16 bounded tags")
        outcome = self._unwrap(self._call("POST", self._path_for("consolidate"),
                                         {"observation_scopes": observation_scopes},
                                         capability="consolidate"), "consolidate")
        if not isinstance(outcome.get("operation_id"), str) or not outcome["operation_id"]:
            raise HindsightError("consolidation returned no durable operation identity")
        return outcome

    def reflect(self, query: str, *, max_tokens: int = 2048, budget: str = "low",
                tags: list[str] | None = None,
                fact_types: list[str] | None = None, include_support: bool = False) -> dict[str, Any]:
        """Ask the backend to synthesize an answer, and take its sourcing report with it.

        ``include.facts`` is asked for always. A synthesized paragraph whose supporting
        memories are not named is a claim with no way back to evidence, which is the one
        thing a summary is not allowed to be.
        """
        self.capabilities.require("reflect")
        if not isinstance(query, str) or not query.strip():
            raise HindsightError("a reflection needs a question; reflecting over nothing "
                                 "produces prose about nothing")
        if len(query) > 4000:
            raise HindsightError("reflect query is too long; narrow the scope instead of "
                                 "over-asking")
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 8192:
            raise HindsightError("max_tokens must be between 1 and 8192")
        payload: dict[str, Any] = {"query": query.strip(), "budget": budget,
                                   "max_tokens": int(max_tokens),
                                   "include": {"facts": {}}}
        if tags:
            payload["tags"] = list(tags)
        if fact_types:
            payload["fact_types"] = list(fact_types)
        body = self._unwrap(self._call("POST", self._path_for("reflect"), payload,
                                       capability="reflect"), "reflect")
        answer = str(body.get("text") or body.get("answer") or "").strip()
        based_on = body.get("based_on") or {}
        memories = list(based_on.get("memories") or [])
        support = self.hydrate_support(memories) if include_support else ()
        if include_support:
            by_id = {fact["id"]: fact for fact in support}
            memories = [by_id[fact["id"]] for fact in memories]
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return {"text": answer, "facts": memories,
                "mental_models": list(based_on.get("mental_models") or []),
                "directives": list(based_on.get("directives") or []),
                "cited_memories": len(memories),
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
                **({"source_facts": list(support)} if include_support else {}),
                "truncated": bool(body.get("truncated"))}

    def get_memory(self, memory_id: str) -> dict[str, Any]:
        """Read one exact identity in this bank; no text matching or cross-bank join."""
        if (not isinstance(memory_id, str) or not 1 <= len(memory_id) <= 128
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in memory_id)):
            raise HindsightError("memory identity must be a bounded path-safe identifier")
        body = self._unwrap(self._call("GET", self._path_for("get_memory", memory_id=memory_id),
                                       None, capability="get_memory"), "get_memory")
        if body.get("id") != memory_id or body.get("bank_id", self.bank_id) != self.bank_id:
            raise HindsightError("memory lookup returned a different identity or bank")
        if body.get("state", "valid") != "valid":
            raise HindsightError("memory lookup returned invalidated evidence")
        return body

    def hydrate_support(self, facts) -> tuple[dict[str, Any], ...]:
        """Explicit bounded read-back for the slim ReflectFact wire schema.

        A returned ID is a citation, not canonical authority. These hydrated
        document references still require SupportResolver and caller checks.
        Keep this opt-in: reflection is background work, not foreground recall.
        """
        if not isinstance(facts, (list, tuple)) or len(facts) > 64:
            raise HindsightError("reflection support exceeds its 64-identity bound")
        resolved, visiting = {}, set()

        def walk(identifier, depth=0):
            if depth > 8 or identifier in visiting:
                raise HindsightError("reflection support is cyclic or too deep")
            if identifier in resolved:
                return
            if len(resolved) + len(visiting) >= 64:
                raise HindsightError("reflection support exceeds its 64-identity bound")
            visiting.add(identifier)
            body = self.get_memory(identifier)
            children = body.get("source_memory_ids") or []
            if not isinstance(children, list) or len(children) > 64:
                raise HindsightError("reflection source identities are malformed")
            node = {**body, "source_fact_ids": list(children)}
            for child in children:
                if not isinstance(child, str):
                    raise HindsightError("reflection source identity is malformed")
                walk(child, depth + 1)
            visiting.remove(identifier)
            resolved[identifier] = node
            if len(json.dumps(list(resolved.values()), ensure_ascii=False).encode()) > 262144:
                raise HindsightError("reflection support exceeds its byte bound")

        for fact in facts:
            if not isinstance(fact, dict) or not isinstance(fact.get("id"), str):
                raise HindsightError("reflection citation has no memory identity")
            walk(fact["id"])
        return tuple(resolved.values())

    def operation(self, operation_id: str) -> dict[str, Any]:
        self.capabilities.require("get_operation")
        result = self._call("GET", self._path_for("get_operation", operation_id=operation_id),
                            None, capability="get_operation")
        if result.status == 404:
            # `not_found` is the engine's own word for this, and the two readers of an
            # operation answer already have to handle it: inventing a third word here would
            # be a state nothing else documents.
            return {"status": "not_found", "operation_id": operation_id}
        return self._unwrap(result, "get_operation")

    def cancel_operation(self, operation_id: str) -> dict[str, Any]:
        self.capabilities.require("cancel_operation")
        return self._unwrap(self._call("DELETE",
                                       self._path_for("cancel_operation",
                                                      operation_id=operation_id),
                                       None, capability="cancel_operation"), "cancel_operation")

    def stats(self) -> dict[str, Any]:
        self.capabilities.require("bank_stats")
        return self._unwrap(self._call("GET", self._path_for("bank_stats"), None,
                                       capability="bank_stats"), "bank_stats")

    def document_state(self, document_id: str) -> dict[str, Any]:
        """Is this document present, and can we prove it? Absent-vs-unknown differs.

        The pinned backend filters this reading by one ``document_id`` query parameter, so
        one call answers one mapping — which is also the only shape that keeps "absent"
        meaningful: a batch answer could not distinguish the document that was not there
        from the one it never looked up.
        """
        self.capabilities.require("list_memories")
        asked = (f"{self._path_for('list_memories')}"
                 f"?{urllib.parse.urlencode({'document_id': document_id, 'limit': 5})}")
        result = self._call("GET", asked, None, capability="list_memories")
        if not result.ok:
            return {"document_id": document_id, "state": "unknown",
                    "reason": f"HTTP {result.status}"}
        body = result.body or {}
        found = body.get("memories") or body.get("items") or []
        return {"document_id": document_id, "state": "present" if found else "absent",
                "count": len(found)}

    # -- plumbing ------------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        if self._capabilities is None:
            # Negotiating hits /health, which the caller must have already
            # accepted; building a client must never reach the network.
            self._capabilities = capabilities_for(PINNED_VERSION)
        return self._capabilities

    def _path_for(self, capability_name: str, **extra: str) -> str:
        # One formatter only. Deriving the path here as well as in the capability table
        # would let a request be built for a route the negotiated backend does not serve.
        return self.capabilities.endpoint(capability_name, bank_id=self.bank_id, **extra)

    def _call(self, method: str, path: str, payload: dict | None, *,
              capability: str | None) -> TransportResult:
        if capability is not None:
            self.capabilities.require(capability)
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        outcome = self.transport(method, f"{self.base_url}{path}", payload, headers)
        if not isinstance(outcome, TransportResult):
            raise HindsightError("transport must return a TransportResult")
        return outcome

    def _unwrap(self, result: TransportResult, what: str) -> dict[str, Any]:
        if result.transport_error:
            raise HindsightUnavailable(
                f"{what} could not be confirmed: {result.transport_error}. The backend may "
                "still be running it; reconcile rather than resubmitting.")
        if not result.ok:
            detail = (result.body or {}).get("detail") if isinstance(result.body, dict) else None
            raise HindsightError(f"{what} failed: HTTP {result.status}"
                                 + (f" — {str(detail)[:300]}" if detail else ""),
                                 status=result.status, body=result.body)
        if result.body is None:
            return {}
        if not isinstance(result.body, dict):
            raise HindsightError(f"{what} returned a non-object body", status=result.status)
        return result.body


def _source_fact_manifest(value) -> tuple[dict[str, Any], ...]:
    """Preserve authoritative map keys when a backend omits redundant item IDs."""
    if not isinstance(value, dict) or len(value) > 500:
        raise HindsightError("source-fact manifest is not a bounded mapping")
    facts = []
    for identifier, fact in value.items():
        if (not isinstance(identifier, str) or not identifier or len(identifier) > 200
                or not isinstance(fact, dict)
                or (fact.get("id") is not None and fact["id"] != identifier)):
            raise HindsightError("source-fact manifest has malformed or conflicting identities")
        facts.append({**fact, "id": identifier})
    return tuple(facts)


@dataclass(frozen=True)
class RecallOutcome:
    """Results plus an explicit account of what was truncated.

    A recall that silently dropped provenance reads exactly like a recall that
    found nothing, and the difference decides whether the agent is entitled to
    say anything at all.
    """

    results: tuple[dict[str, Any], ...] = ()
    source_facts: tuple[dict[str, Any], ...] = ()
    chunks: tuple[dict[str, Any], ...] = ()
    entities: tuple[dict[str, Any], ...] = ()
    truncated: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> "RecallOutcome":
        include = body.get("include") or {}
        truncated = []
        if body.get("source_facts_truncated") or include.get("source_facts_truncated"):
            truncated.append("source_facts")
        if body.get("chunks_truncated") or include.get("chunks_truncated"):
            truncated.append("chunks")
        if body.get("truncated") is True:
            truncated.append("results")
        results = body.get("results") or body.get("memories") or []
        return cls(
            results=tuple(results),
            source_facts=_source_fact_manifest(include.get("source_facts")
                                               or body.get("source_facts") or {}),
            chunks=tuple((include.get("chunks") or body.get("chunks") or {}).values()),
            entities=tuple((include.get("entities") or body.get("entities") or {}).values()),
            truncated=tuple(truncated),
            raw=body,
        )

    @property
    def provenance_complete(self) -> bool:
        return "source_facts" not in self.truncated

    def as_dict(self) -> dict[str, Any]:
        return {"results": len(self.results), "source_facts": len(self.source_facts),
                "chunks": len(self.chunks), "truncated": list(self.truncated),
                "provenance_complete": self.provenance_complete}


def _decode(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"raw": raw[:2000].decode("utf-8", "replace")}
