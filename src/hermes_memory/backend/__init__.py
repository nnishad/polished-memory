"""C5 — the derived-backend boundary."""
from .capabilities import (CAPABILITIES, PINNED_VERSION, Capabilities,
                           UnsupportedCapability, capabilities_for)
from .document_map import ABSENT, FAILED, QUEUED, SUBMITTED, VERIFIED, DocumentMap
from .hindsight_client import (HindsightClient, HindsightError, HindsightUnavailable,
                               RecallOutcome, SubmissionConflict, TransportResult)

__all__ = ["HindsightClient", "HindsightError", "HindsightUnavailable", "SubmissionConflict",
           "RecallOutcome", "TransportResult", "DocumentMap", "Capabilities",
           "capability_names", "capabilities_for", "UnsupportedCapability", "CAPABILITIES",
           "PINNED_VERSION", "QUEUED", "SUBMITTED", "VERIFIED", "FAILED", "ABSENT"]


def capability_names() -> list[str]:
    return sorted(cap.name for cap in CAPABILITIES)
