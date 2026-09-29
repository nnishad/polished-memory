"""The host's side of the handover: resolve a profile, then run the framework's drain.

The consent rules, the claim/attempt/confirm state machine and the receipt reading live
in ``hermes_memory.proactive.delivery`` now, because they are the framework's decision
and reachable by anything that can open a store — including ``hermes-memory deliver`` on
a machine with no plugin loaded at all. What stays here is the two things only the host
knows: which profile a Hermes home belongs to, and whether an operator has held delivery
for the whole installation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TextIO

from hermes_memory.proactive.delivery import (DeliveryPolicy, bounded_drain,
                                              correlation, deliver_once,  # noqa: F401
                                              deliver_ready)

__all__ = ["DeliveryPolicy", "deliver_once", "deliver_for_home", "correlation"]


def instance_hold(activity) -> str | None:
    """Whether the owner has held delivery for the whole installation.

    A hold and a switch are different decisions: the switch is configuration and per
    profile, the hold is an operator's temporary instruction that covers every outbox on
    the machine. It is read from the store the operator's command wrote it to, and that
    file is not created by asking — a refusal to deliver must not leave state behind.
    """
    from hermes_memory.storage.evidence import EvidenceStore

    path = Path(activity.instance_db_path)
    if not path.is_file():
        return None
    with EvidenceStore(path) as store:
        return ("delivery is paused for this installation by the owner"
                if store.stage_is_paused("global", "delivery") else None)


def deliver_for_home(hermes_home: str | Path, *, settings: Any = None,
                     sink: Callable[[str], Any] | None = None, limit: int = 1,
                     holder: str = "hermes-memory-delivery", out: TextIO | None = None,
                     at: float | None = None) -> dict[str, Any]:
    """Deliver this profile's own ready artifacts, through this profile's own outbox.

    A separate process — the owner's scheduler, or one command — reaches the same guard
    the plugin applies in conversation, and it resolves the profile from the home it was
    pointed at rather than from whichever profile installed the default.
    """
    from hermes_memory.storage.evidence import EvidenceStore

    from .client import bind

    bounded_drain(limit)
    activity = bind(hermes_home, settings=settings)
    try:
        policy = DeliveryPolicy.from_settings(activity.settings)
        # Read the operator's hold only when the configuration already allows a send:
        # asking is a file open, and a refusal that leaves state behind has changed the
        # installation it was only supposed to consult.
        held = None if policy.refusal() else instance_hold(activity)
        with EvidenceStore(activity.db_path) as store:
            outcome = deliver_ready(store, policy=policy, sink=sink, out=out,
                                    limit=limit, holder=holder, held=held, at=at)
        return {"profile": activity.name, "bank_id": activity.bank_id, **outcome}
    finally:
        activity.close()
