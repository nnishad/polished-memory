"""C6 contradictions: the comparison, tested on its own.

The store path is covered in ``test_assertions.py``. What is pinned here is the rule set
itself, without an archive in the way: what counts as a clash, what is deliberately not
compared, and the fact that two runs say the same thing in the same order.
"""
from __future__ import annotations

import dataclasses

import pytest

from hermes_memory.knowledge.assertions import Assertion
from hermes_memory.knowledge.contradictions import Contradiction, find

MORNING = "2026-09-15T09:00:00+00:00"
EVENING = "2026-09-15T18:00:00+00:00"
LAST_YEAR = "2025-09-15T09:00:00+00:00"


def claim(**overrides) -> Assertion:
    """One supported claim, with the fields a clash never looks at already filled in."""
    base = {"id": "a1", "subject": "julia", "predicate": "works_from_home_on",
            "value": "Tuesday", "kind": "preference", "unit": None,
            "evidence_kind": "owner_declared",
            "record_id": "rec-1", "quote": "I work from home on Tuesdays",
            "quote_start": 0, "quote_end": 27, "valid_from": None, "valid_to": None,
            "status": "confirmed", "created_by": "owner", "confirmed_by": "owner",
            "created_at": MORNING, "confirmed_at": MORNING, "supersedes": None,
            "revision": 1}
    return Assertion(**base | overrides)


FIRST = claim()
SECOND = claim(id="a2", value="Friday")


def test_two_claims_held_at_once_clash():
    found = find([FIRST, SECOND])
    assert [(item.subject, item.predicate) for item in found] == \
        [("julia", "works_from_home_on")]
    assert [item.id for item in found[0].assertions] == ["a1", "a2"]


def test_the_same_value_in_another_case_is_not_a_disagreement():
    assert find([FIRST, claim(id="a3", value="TUESDAY")]) == []


def test_an_unbounded_claim_clashes_with_anything_that_disagrees():
    """Both ends open means "always", so there is no moment at which they fit."""
    assert len(find([FIRST, SECOND])) == 1


def test_an_assertion_never_clashes_with_itself():
    assert find([FIRST]) == []
    assert find([FIRST, FIRST]) == []


def test_two_closed_periods_that_never_meet_are_a_history_not_a_clash():
    assert find([claim(valid_to=LAST_YEAR), claim(id="a2", value="Friday",
                                                 valid_from=MORNING)]) == []
    found = find([claim(valid_to=LAST_YEAR), claim(id="a2", value="Friday",
                                                  valid_from=LAST_YEAR)])
    assert len(found) == 1, "sharing one instant is sharing a moment"


@pytest.mark.parametrize("kind", ["belief"])
def test_two_beliefs_are_not_a_contradiction(kind):
    assert find([claim(kind=kind, value="Tuesday is luckier"),
                 claim(id="a2", kind=kind, value="Friday is luckier")]) == []


def test_different_subjects_and_predicates_are_never_compared():
    assert find([FIRST, claim(id="a2", subject="marco", value="Friday")]) == []
    assert find([FIRST, claim(id="a2", predicate="coffee_with", value="Friday")]) == []


def test_the_groups_come_back_in_the_same_order_every_time():
    items = [claim(id="z1", subject="zoe"), claim(id="z2", subject="zoe", value="Friday"),
             claim(id="a9", subject="adam"), claim(id="a2", subject="adam",
                                                   value="Friday"), FIRST, SECOND]
    ordered = [(item.subject, item.predicate) for item in find(items)]
    assert ordered == [("adam", "works_from_home_on"), ("julia", "works_from_home_on"),
                       ("zoe", "works_from_home_on")]
    assert [(item.subject, item.predicate) for item in find(list(reversed(items)))] == \
        ordered, "a scan order must not decide what an operator is told"


def test_a_group_with_three_claims_reports_the_clashing_pair_once():
    """The third claim is from last year; it is history, and it stays out of the finding.

    All three have to be bounded for that to be a real test: an open end is unbounded,
    so an unbounded claim overlaps a period it never mentions.
    """
    found = find([claim(valid_from=MORNING, valid_to=EVENING),
                  claim(id="a2", value="Friday", valid_from=MORNING, valid_to=EVENING),
                  claim(id="a3", value="Saturday", valid_from=LAST_YEAR,
                       valid_to="2025-06-01T00:00:00+00:00")])
    assert len(found) == 1
    assert [item.id for item in found[0].assertions] == ["a1", "a2"]
    assert found[0].values == ("Friday", "Tuesday")


def test_the_report_names_the_moment_it_was_asked_about():
    found = find([FIRST, SECOND], at=EVENING)
    assert found[0].at == EVENING
    assert found[0].as_dict()["at"] == EVENING


def test_a_contradiction_selects_no_winner():
    """The finding is the pair. Choosing between them is the owner's act, not this one."""
    found = find([FIRST, SECOND])[0]
    assert {item.status for item in found.assertions} == {"confirmed"}
    assert not hasattr(found, "resolution") or found.__dict__.get("resolution") is None
    assert set(found.as_dict()) == {"subject", "predicate", "at", "values", "assertions"}


def test_nothing_at_all_in_comes_back_nothing_at_all():
    assert find([]) == []
    assert find(iter([FIRST])) == []


def test_the_dataclass_is_frozen_like_the_claim_it_compares():
    found = find([FIRST, SECOND])[0]
    assert isinstance(found, Contradiction)
    with pytest.raises(dataclasses.FrozenInstanceError):
        found.subject = "somebody else"
