"""C6 contradictions: two claims that are both supported and cannot both be true.

This is a comparison over rows the archive already vouches for, and nothing else: no
model is asked whether two sentences conflict, because a model that is confident about a
contradiction is the failure this module exists to catch. Two rules do all the work. A
pair clashes only when its validity intervals actually overlap — two claims about
consecutive periods are a history — and beliefs are never compared, because two people
can both believe something and be right that they believe it.

Candidates are excluded upstream, in the reading of the archive that gets here: a
hypothesis contradicts nothing, and counting them would let an unverified guess drown
every real disagreement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from ..ids import intervals_overlap

if TYPE_CHECKING:                                  # pragma: no cover
    from .assertions import Assertion

__all__ = ["Contradiction", "find"]


@dataclass(frozen=True)
class Contradiction:
    """One subject, one predicate, and the set of claims that cannot all stand."""

    subject: str
    predicate: str
    at: str | None
    assertions: tuple[Assertion, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"subject": self.subject, "predicate": self.predicate, "at": self.at,
                "values": sorted({item.value for item in self.assertions}),
                "assertions": [item.id for item in self.assertions]}

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(sorted({item.value for item in self.assertions}))


def find(assertions: Iterable[Assertion], *, at: str | None = None) -> list[Contradiction]:
    """Group claims by (subject, predicate) and report the groups that clash.

    Ordering is by subject then predicate rather than by anything about the reader, so
    two runs over the same archive say the same thing in the same order.
    """
    groups: dict[tuple[str, str], list[Assertion]] = {}
    for assertion in assertions:
        if assertion.kind == "belief":
            # Two beliefs can both be held at once. Calling that a contradiction
            # would be a category error, not a finding.
            continue
        groups.setdefault((assertion.subject, assertion.predicate), []).append(assertion)
    found = []
    for (subject, predicate), items in sorted(groups.items()):
        clashing = _clashing(items)
        if clashing:
            found.append(Contradiction(subject=subject, predicate=predicate, at=at,
                                       assertions=tuple(item for item in items
                                                        if item.id in clashing)))
    return found


def _clashing(items: Sequence[Assertion]) -> set[str]:
    """The ids of every pair that disagrees about a value held at the same time."""
    clashing: set[str] = set()
    for index, item in enumerate(items):
        for other in items[index + 1:]:
            if (item.value.casefold() != other.value.casefold()
                    and intervals_overlap(item.valid_from, item.valid_to,
                                          other.valid_from, other.valid_to)):
                clashing.update({item.id, other.id})
    return clashing
