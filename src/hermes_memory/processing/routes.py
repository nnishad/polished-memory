"""C12 route table: credentials select an operation, never an arbitrary upstream.

The gate is a mapper for the exact OpenAI-compatible subset memory needs, not a
general proxy. A caller presents a synthetic credential and gets the upstream
that credential was minted for; there is no parameter through which a URL can
be supplied, so a misconfigured or malicious client cannot redirect inference
to an unapproved endpoint or straight around the gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import SettingError

__all__ = ["Route", "RouteTable", "RouteError", "PRIORITY", "build_routes"]


class RouteError(SettingError):
    """A route or credential that selects nothing. There is no default to fall back to."""

# Lower is more urgent. Gate priority decides who waits, not who preempts: an
# already-started request is never reordered or killed.
PRIORITY = {
    "interactive": 0,
    "freshness": 1,
    "proactive": 2,
    "maintenance": 3,
}

# Physical resources, not endpoints. The local 4B and the embedding model share
# one GPU, so they share one slot even though they are different services.
RESOURCES = ("remote-9b", "local-gpu")


@dataclass(frozen=True)
class Route:
    name: str
    resource: str
    operation: str
    upstream: str
    credential: str
    priority: str
    max_output_tokens: int

    def priority_rank(self) -> int:
        return PRIORITY[self.priority]


class RouteTable:
    def __init__(self, routes: dict[str, Route]):
        self._by_name = dict(routes)
        self._by_credential = {route.credential: route for route in routes.values()}
        collisions = len(routes) - len(self._by_credential)
        if collisions:
            raise SettingError(f"{collisions} route credential(s) are not unique")

    def by_name(self, name: str) -> Route:
        try:
            return self._by_name[name]
        except KeyError:
            raise RouteError(
                f"unknown route {name!r}; admissible routes are {sorted(self._by_name)}"
                " — there is no default and no fallback"
            ) from None

    def by_credential(self, credential: str) -> Route:
        try:
            return self._by_credential[credential]
        except KeyError:
            raise RouteError(
                "presented credential does not select any route; refusing rather than "
                "guessing an upstream") from None

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def as_dict(self) -> dict[str, Any]:
        """Reportable form. Upstreams are shown; credentials never are."""
        return {name: {"resource": route.resource, "operation": route.operation,
                       "upstream": route.upstream, "priority": route.priority,
                       "max_output_tokens": route.max_output_tokens}
                for name, route in sorted(self._by_name.items())}


def build_routes(settings, *, credentials: dict[str, str] | None = None) -> RouteTable:
    """Derive the route table from owned configuration.

    Every generation route carries an explicit output cap: the engine's own
    default allowed unbounded completions, and a cap that only some paths set is
    the same as no cap on the paths that omit it.
    """
    if not settings.hindsight_url:
        raise SettingError("no inference route is configured; the gate has nothing to map")
    text = settings.text_route
    vision = settings.vision_route
    embeddings = settings.embeddings_route
    if not (text and embeddings):
        raise SettingError("a text route and an embeddings route are both required")
    caps = settings.max_output_tokens
    given = credentials or {}

    def route(name: str, *, resource: str, operation: str, upstream: str | None,
              priority: str, cap: int) -> Route | None:
        if not upstream:
            return None
        credential = given.get(name)
        if not credential:
            raise SettingError(f"route {name!r} has no credential; refusing to serve it "
                               "unauthenticated on loopback")
        return Route(name=name, resource=resource, operation=operation, upstream=upstream,
                     credential=credential, priority=priority, max_output_tokens=cap)

    routes: dict[str, Route] = {}
    for candidate in (
        route("retain", resource=text.resource, operation="chat", upstream=text.base_url,
              priority="freshness", cap=caps.retain),
        route("consolidate", resource=text.resource, operation="chat", upstream=text.base_url,
              priority="maintenance", cap=caps.consolidate),
        route("reflect", resource=text.resource, operation="chat", upstream=text.base_url,
              priority="maintenance", cap=caps.reflect),
        route("foreground", resource=text.resource, operation="chat", upstream=text.base_url,
              priority="interactive", cap=caps.foreground),
        route("embeddings", resource=embeddings.resource, operation="embeddings",
              upstream=embeddings.base_url, priority="freshness", cap=0),
        route("vision", resource=vision.resource if vision else "local-gpu", operation="chat",
              upstream=vision.base_url if vision else None, priority="maintenance",
              cap=caps.reflect),
    ):
        if candidate is not None:
            routes[candidate.name] = candidate
    return RouteTable(routes)
