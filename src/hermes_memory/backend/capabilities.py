"""C5 capability map: what the *pinned* Hindsight revision actually supports.

Every backend call this framework makes is checked against the table below, and
the table is derived from a specific tag rather than from whatever HEAD happens
to contain. A feature advertised at a newer revision but absent at the pinned
one would otherwise fail at runtime in a way that looks like our bug, and the
reverse mistake — believing a capability is absent after an upgrade — silently
disables work that would have run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CAPABILITIES", "Capability", "Capabilities", "PINNED_VERSION",
           "UnsupportedCapability", "OPERATION_STATES", "OPERATION_DONE",
           "OPERATION_RUNNING", "OPERATION_ABANDONED", "OPERATION_STOPPED"]

PINNED_VERSION = "0.10.1"


class UnsupportedCapability(Exception):
    """Raised rather than sending a request the pinned revision cannot answer."""


@dataclass(frozen=True)
class Capability:
    name: str
    method: str
    path: str
    operation_id: str
    since: str
    note: str = ""


# Verified against git tag v0.10.1 in
# hindsight-api-slim/hindsight_api/api/http.py. Paths are the FastAPI route
# templates; the client formats {bank_id} etc. itself.
CAPABILITIES: tuple[Capability, ...] = (
    Capability("health", "GET", "/health", "get_readiness", "0.10.1",
               "Liveness and readiness only; never starts or upgrades a backend."),
    Capability("retain", "POST", "/v1/default/banks/{bank_id}/memories", "retain_memories",
               "0.10.1", "Supports document_id upsert and per-item tags."),
    Capability("retain_async_identity", "POST", "/v1/default/banks/{bank_id}/memories",
               "retain_memories", "0.10.1",
               "Client-supplied operation_id UUID dedupes a retry after a lost ack; "
               "reusing an id for different content returns HTTP 409."),
    Capability("recall", "POST", "/v1/default/banks/{bank_id}/memories/recall",
               "recall_memories", "0.10.1"),
    Capability("list_memories", "GET", "/v1/default/banks/{bank_id}/memories/list",
               "list_memories", "0.10.1",
               "Filtered by query parameter, one document_id per request; the reconciliation "
               "reading, so it is the call that has to answer after a lost connection."),
    Capability("get_memory", "GET", "/v1/default/banks/{bank_id}/memories/{memory_id}",
               "get_memory", "0.10.1"),
    Capability("update_memory", "PATCH", "/v1/default/banks/{bank_id}/memories/{memory_id}",
               "update_memory", "0.10.1"),
    Capability("delete_memories", "DELETE", "/v1/default/banks/{bank_id}/memories",
               "delete_memories", "0.10.1",
               "The backend-side half of an erasure obligation."),
    Capability("list_operations", "GET", "/v1/default/banks/{bank_id}/operations",
               "list_operations", "0.10.1"),
    Capability("get_operation", "GET", "/v1/default/banks/{bank_id}/operations/{operation_id}",
               "get_operation", "0.10.1", "Reconciles a lost acknowledgement."),
    Capability("cancel_operation", "DELETE",
               "/v1/default/banks/{bank_id}/operations/{operation_id}", "cancel_operation",
               "0.10.1"),
    Capability("retry_operation", "POST",
               "/v1/default/banks/{bank_id}/operations/{operation_id}/retry",
               "retry_operation", "0.10.1"),
    Capability("delete_operation", "DELETE",
               "/v1/default/banks/{bank_id}/operations/{operation_id}/delete",
               "delete_operation", "0.10.1"),
    Capability("reflect", "POST", "/v1/default/banks/{bank_id}/reflect", "reflect", "0.10.1"),
    Capability("consolidate", "POST", "/v1/default/banks/{bank_id}/consolidate",
               "trigger_consolidation", "0.10.1",
               "Async explicit observation_scopes; availability is not stage permission."),
    Capability("bank_stats", "GET", "/v1/default/banks/{bank_id}/stats", "get_bank_stats",
               "0.10.1", "Basis for a coverage claim; not proof of completeness."),
    Capability("dry_run_extract", "POST",
               "/v1/default/banks/{bank_id}/memories/dry-run-extract", "dry_run_extract_memories",
               "0.10.1", "Shows what extraction would produce without committing it."),
    Capability("preview_prompts", "POST", "/v1/default/banks/{bank_id}/prompts/preview",
               "preview_prompt", "0.10.1"),
    Capability("list_banks", "GET", "/v1/default/banks", "list_banks", "0.10.1"),
)

# `OperationStatusResponse.status`, word for word. A job's whole recovery story depends on
# being able to say "that answer means it is finished", "means it is still running" or
# "means it will never answer", so the words live beside the pinned routes that produce
# them rather than beside one caller. An answer from outside this list is not read as
# either of the things it might be mistaken for.
OPERATION_STATES: tuple[str, ...] = ("pending", "processing", "completed", "failed",
                                     "cancelled", "not_found")
OPERATION_DONE = frozenset({"completed"})
OPERATION_RUNNING = frozenset({"pending", "processing"})
# Nothing is running for these, so the device is free and the projection did not happen.
OPERATION_ABANDONED = frozenset({"failed", "not_found"})
# Somebody asked for this to stop. Resubmitting it would overrule them, so it is not in
# the abandoned set even though nothing is running: the ending is the point of it.
OPERATION_STOPPED = frozenset({"cancelled"})


@dataclass(frozen=True)
class Capabilities:
    """The admissible subset of CAPABILITIES for one observed version."""

    version: str
    supported: frozenset[str] = frozenset()
    unsupported: tuple[str, ...] = ()
    observed_via: str = "declared"
    mismatches: tuple[str, ...] = field(default_factory=tuple, )

    def require(self, name: str) -> Capability:
        if name not in self.supported:
            raise UnsupportedCapability(
                f"{name!r} is not available on Hindsight {self.version}; this build is "
                f"pinned to {PINNED_VERSION} and will not send a request it cannot answer")
        return next(cap for cap in CAPABILITIES if cap.name == name)

    def endpoint(self, name: str, **path_values: str) -> str:
        cap = self.require(name)
        try:
            return cap.path.format(**path_values)
        except KeyError as error:
            raise UnsupportedCapability(f"missing path value for {name}: {error}") from None

    def as_dict(self) -> dict[str, Any]:
        return {"version": self.version, "pinned_to": PINNED_VERSION,
                "observed_via": self.observed_via, "supported": sorted(self.supported),
                "unsupported": list(self.unsupported), "mismatches": list(self.mismatches)}


def capabilities_for(version: str, *, route_names: set[str] | None = None,
                     ruled_out: set[str] | None = None) -> Capabilities:
    """Build the capability set for an observed version.

    When the backend tells us which routes it actually serves, that overrides the table: a
    build that removed an endpoint would otherwise keep us sending requests to it, and a
    version string alone does not prove a route exists.

    `route_names` is a *census* — pass it only when the whole served set is known. A partial
    observation is `ruled_out`: names that were asked about and answered 404, subtracted from
    the table with everything else left as declared. Confusing the two reported a backend as
    routing neither `retain` nor `recall` on the strength of three read-only probes, both of
    which it had been answering for hours.
    """
    declared = {cap.name for cap in CAPABILITIES if _at_least(version, cap.since)}
    if route_names is not None:
        present = declared & route_names
        absent = sorted(declared - present)
        return Capabilities(version=version, supported=frozenset(present),
                            unsupported=tuple(absent), observed_via="routes",
                            mismatches=tuple(f"advertised at {version} but not routed: {name}"
                                             for name in absent))
    absent = sorted(declared & {str(name) for name in ruled_out or ()})
    if absent:
        return Capabilities(version=version, supported=frozenset(declared - set(absent)),
                            unsupported=tuple(absent), observed_via="routes",
                            mismatches=tuple(f"asked for at {version} and it answered 404: "
                                             f"{name}" for name in absent))
    return Capabilities(version=version, supported=frozenset(declared),
                        observed_via="version")


def _at_least(version: str, floor: str) -> bool:
    try:
        return _parts(version) >= _parts(floor)
    except ValueError:
        return False


def _parts(version: str) -> tuple[int, ...]:
    core = version.split("+", 1)[0].split("-", 1)[0]
    return tuple(int(part) for part in core.split("."))
