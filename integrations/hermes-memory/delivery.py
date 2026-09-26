"""The handover: one artifact, to the one destination the owner named.

The framework decides what is worth saying and the outbox records who was told. This
module is the only place in the plugin that puts memory in front of a transport, and it
is deliberately unimpressive about itself: it is disabled until an owner configures a
concrete private destination, it takes one artifact at a time, and it never chooses a
recipient — an artifact addressed to anyone other than the approved destination is put
back, not sent.

The interesting part is what happens when something goes wrong. A failure before the
bytes leave is a release: nothing was delivered. A failure after the handover began is
``uncertain``, which is not put back in the queue, because the third copy of a
notification is a worse outcome than the missing first one. And a transport that says
"sent" without a digest to check it against gets ``accepted_unverified``, not
confirmation: that is a report of the transport's own intent, not of the owner's inbox.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

__all__ = ["DeliveryPolicy", "deliver_once", "deliver_for_home", "correlation"]

# Destinations that are not a private channel to one person. ``bot-chat`` is refused on
# the plan's own instruction: it creates an inbound agent turn, so "delivering" to it
# would spend model calls the feature is supposed to avoid.
_PUBLIC = frozenset({"bot-chat", "all", "everyone", "broadcast", "public", "any"})
_KINDS = ("next_turn", "digest", "notify_owner")


@dataclass(frozen=True)
class DeliveryPolicy:
    """Whether this installation may hand anything to a transport, and to which one."""

    enabled: bool
    destination: str | None
    owner_principal: str | None
    kinds: tuple[str, ...] = _KINDS

    @classmethod
    def from_settings(cls, settings: Any) -> "DeliveryPolicy":
        return cls(enabled=bool(getattr(settings, "delivery_enabled", False)),
                   destination=getattr(settings, "delivery_target", None),
                   owner_principal=getattr(settings, "owner_principal", None))

    def refusal(self) -> str | None:
        """Why nothing may be delivered, or None when it may.

        Checked on every call rather than once at import: the answer is a property of
        the owner's decision, and an installation can lose it (by retiring the owner
        principal) without restarting anything.
        """
        if not self.enabled:
            return "provider-initiated delivery is disabled by default"
        target = (self.destination or "").strip()
        if not target:
            return ("no approved destination is configured "
                    "(set HERMES_MEMORY_DELIVERY_TARGET)")
        scheme, separator, address = target.partition(":")
        if not separator or not address.strip():
            return (f"{target!r} is not a concrete destination; a delivery target names "
                    "one private channel, like 'signal:1234' or 'email:me@example'")
        if scheme.strip().lower() in _PUBLIC:
            return f"{scheme!r} is not a private destination for one owner"
        if not self.owner_principal:
            return ("an owner principal is required before anything is delivered on "
                    "someone's behalf")
        return None

    def accepts(self, artifact: Any) -> str | None:
        """Why this artifact may not go out, or None when it may.

        The recipient check is the load-bearing one. Proactivity in this framework is
        notify-and-draft: everything in the outbox is addressed to the owner, and a
        transport that would happily message a third party is a different feature with
        different consent, not this one with a different string in it.
        """
        if artifact.kind not in self.kinds:
            return (f"a {artifact.kind!r} artifact is drafted for the owner to act on, "
                    "not for a transport to send")
        if str(artifact.recipient or "").strip() != (self.owner_principal or "").strip():
            return ("the artifact is addressed to someone other than the owner; this "
                    "installation notifies its owner and drafts for their approval, it "
                    "does not message anybody else")
        return None


def correlation(artifact: Any) -> str:
    """The opaque marker a transport can echo back and be believed.

    It carries the artifact id and the prefix of the payload digest the outbox already
    holds, so a receipt can be matched to the thing that was sent rather than to the
    time it happened to arrive.
    """
    return f"hermes-memory {artifact.id} {artifact.payload_digest[:12]}"


def deliver_once(outbox: Any, *, policy: DeliveryPolicy, sink: Callable[[str], Any] | None = None,
                 holder: str = "hermes-memory-delivery", out: TextIO | None = None,
                 at: float | None = None) -> dict[str, Any]:
    """Try to deliver one ready artifact. Returns a report about the attempt.

    ``sink`` is the transport: it receives the rendered text and may return a receipt
    (a dict with ``digest``, or ``sent``), or None for "the script finished and that is
    all we know". With no sink the artifact is written to ``out`` instead, which is the
    contract-tests path and still records the same states.

    A caller that names no transport at all gets a ``ValueError``: that is a bug in the
    caller, not a fact about the queue.
    """
    if sink is None and out is None:
        # Delivering into nowhere and then recording a receipt would be the one
        # report worse than no delivery.
        raise ValueError("name a transport: pass a sink or a stream to write to")
    blocked = policy.refusal()
    if blocked:
        return {"ok": True, "delivered": False, "attempted": False, "reason": blocked}
    claim = outbox.lease(holder=holder, at=at)
    if claim is None:
        return {"ok": True, "delivered": False, "attempted": False,
                "reason": "nothing is ready to send right now"}
    artifact = claim.artifact
    mismatch = policy.accepts(artifact)
    if mismatch:
        # The lease already revalidated the artifact against the world; this is the
        # transport's own consent check, and it releases rather than suppresses
        # because the artifact itself is fine — it is this destination that is not.
        outbox.release(artifact_id=artifact.id, token=claim.token, reason=mismatch)
        return {"ok": False, "delivered": False, "attempted": False,
                "artifact": artifact.id, "reason": mismatch}

    body = f"{correlation(artifact)}\n{artifact.payload}"
    try:
        outbox.attempt(artifact_id=artifact.id, token=claim.token)
    except Exception as error:
        # The handover never began, so nothing can have left. Say so and stop: this
        # is the only path that may report a refusal without an uncertain state.
        return {"ok": False, "delivered": False, "attempted": False,
                "artifact": artifact.id,
                "reason": f"the attempt could not be recorded: {str(error)[:200]}"}

    try:
        if sink is None:
            out.write(body + "\n")
            out.flush()
            receipt: Any = None
        else:
            receipt = sink(body)
    except Exception as error:
        # Past the attempt we cannot prove the bytes stayed here, and a sink that
        # dies mid-write is exactly the crash this state machine was built for.
        settled = outbox.uncertain(artifact_id=artifact.id, token=claim.token,
                                   reason=f"the handover began and the transport then "
                                          f"failed: {str(error)[:200]}")
        return {"ok": False, "delivered": False, "attempted": True,
                "artifact": artifact.id, "state": settled.state,
                "reason": "the outcome of that send is not established; it will not be "
                          "replayed blind"}

    settled = outbox.confirm(artifact_id=artifact.id, token=claim.token, proof=_proof(receipt))
    return {"ok": settled.state == "confirmed", "delivered": settled.state in
            ("confirmed", "accepted_unverified"), "attempted": True,
            "artifact": settled.id, "state": settled.state, "reason": settled.reason,
            "correlation": correlation(artifact)}


def deliver_for_home(hermes_home: str | Path, *, settings: Any = None,
                     sink: Callable[[str], Any] | None = None, limit: int = 1,
                     holder: str = "hermes-memory-delivery", out: TextIO | None = None,
                     at: float | None = None) -> dict[str, Any]:
    """Deliver this profile's own ready artifacts, through this profile's own outbox.

    A separate process — the owner's scheduler, or one command — reaches the same
    guard the plugin applies in conversation, and it resolves the profile from the
    home it was pointed at rather than from whichever profile installed the default.
    ``limit`` is bounded because a run that drains an unbounded queue is a run that
    can be interrupted with half of it half-sent.
    """
    from hermes_memory.proactive.outbox import Outbox
    from hermes_memory.proactive.policy import AttentionPolicy
    from hermes_memory.storage.evidence import EvidenceStore

    from .client import bind

    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 25:
        raise ValueError("limit must be between 1 and 25 artifacts")
    activity = bind(hermes_home, settings=settings)
    try:
        policy = DeliveryPolicy.from_settings(activity.settings)
        blocked = policy.refusal()
        if blocked:
            return {"ok": True, "delivered": 0, "profile": activity.name, "reason": blocked,
                    "reports": []}
        reports = []
        with EvidenceStore(activity.db_path) as store:
            outbox = Outbox(store,
                            policy=AttentionPolicy(store,
                                                   owner_principal=activity.settings.owner_principal),
                            owner_principal=activity.settings.owner_principal)
            for _ in range(limit):
                report = deliver_once(outbox, policy=policy, sink=sink, holder=holder,
                                      out=out, at=at)
                reports.append(report)
                # Anything that did not reach the handover will not change by trying
                # again in the same run: the queue is empty, or this one is refused.
                if not report.get("attempted"):
                    break
        return {"ok": all(item.get("ok") for item in reports),
                "profile": activity.name, "bank_id": activity.bank_id,
                "delivered": sum(1 for item in reports if item.get("delivered")),
                "reason": reports[-1].get("reason") if reports else "nothing was tried",
                "reports": reports}
    finally:
        activity.close()


def _proof(receipt: Any) -> Any:
    if receipt is None:
        return None
    if isinstance(receipt, dict):
        return receipt
    if isinstance(receipt, str) and receipt.strip().startswith("{"):
        try:
            return json.loads(receipt)
        except ValueError:
            return {"sent": True, "echoed": receipt[:200]}
    return {"sent": True, "echoed": str(receipt)[:200]}
