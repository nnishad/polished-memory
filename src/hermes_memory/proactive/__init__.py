"""C10 proactivity: the gate between "something changed" and "the owner hears".

Three separate authorities, deliberately not merged into one class. The eligibility
filter is a cheap deterministic no; the attention policy is the owner's budget; the
analyst is the only part allowed to read evidence bodies, and the outbox is the only
part allowed to say anything about delivery.
"""
from .analyst import Analysis, PacketAnalyst
from .eligibility import Eligibility, Verdict
from .engine import ProactiveEngine, Sweep
from .outbox import Artifact, ArtifactClaim, Outbox, Revalidation
from .policy import POLICY_VERSION, AttentionPolicy, Decision

__all__ = ["AttentionPolicy", "Decision", "POLICY_VERSION", "Eligibility", "Verdict",
           "PacketAnalyst", "Analysis", "Outbox", "Artifact", "ArtifactClaim",
           "Revalidation", "ProactiveEngine", "Sweep"]
