"""C4 — forgetting, reset and restore-safe ledgers."""
from .erasure import COMPLETE, PENDING, AWAITING, ErasureManager

__all__ = ["ErasureManager", "AWAITING", "PENDING", "COMPLETE"]
