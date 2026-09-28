"""C12: a standing permission to spend the shared models, and the bounds that make it one.

An allowance exists so that formation can run while nobody is awake. The failure it could
cause is worse than the one it cures: a machine still spending a GPU on a decision nobody
remembers making. So every test below is about a bound — the cap, the clock, the device, the
name on the grant — and about what a *refusal* says, because an operator who is told only
"no" goes editing settings rather than granting what they meant.
"""
from __future__ import annotations

import time

import pytest

from hermes_memory.processing.allowance import (ANY_RESOURCE, AllowanceError, Allowances,
                                                MAX_DURATION_S, STAGE)
from hermes_memory.processing.instance_gate import GateStore

OWNER = "jugaadu"
AGENT = "agent:one"
REMOTE = "remote-9b"
MOMENT = 1_800_000_000.0


class Clock:
    """A clock the test moves, so an expiry is a fact being checked rather than a sleep."""

    def __init__(self, moment: float = MOMENT):
        self.now = moment

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def ledger(tmp_path):
    store = GateStore(tmp_path / "gate.db")
    try:
        yield store
    finally:
        store.close()


def grants(store, *, clock=None, owner=OWNER):
    return Allowances(store, owner_principal=owner, clock=clock or Clock())


def issue(store, *, records=10, tokens=50_000, hours=6.0, resource=ANY_RESOURCE,
          clock=None, actor=OWNER, reason="form while I sleep"):
    return grants(store, clock=clock).grant(actor=actor, reason=reason, records=records,
                                            tokens=tokens, duration_s=hours,
                                            resource=resource)


# -- who may grant -----------------------------------------------------------

def test_only_the_owner_grants_and_the_owner_revokes(ledger):
    with pytest.raises(AllowanceError, match="only the owner"):
        issue(ledger, actor=AGENT)
    created = issue(ledger)
    with pytest.raises(AllowanceError, match="only the owner"):
        grants(ledger).revoke(created["id"], actor=AGENT, reason="not theirs to withdraw")


def test_an_installation_with_no_owner_principal_grants_nothing(ledger):
    # Fail closed: with nobody named as the owner, no actor can match, so the machine may not
    # invent a permission to spend its own models.
    with pytest.raises(AllowanceError, match="no owner principal is configured"):
        grants(ledger, owner=None).grant(actor=OWNER, reason="nobody in charge", records=3,
                                         tokens=1000, duration_s=1.0)
    # A reading still works without one: `status` reports who holds the models without naming
    # anybody as the owner.
    assert grants(ledger, owner=None).current() is None


def test_a_grant_that_does_not_say_why_is_refused(ledger):
    for reason in ("", "   ", "x" * 501):
        with pytest.raises(AllowanceError, match="has to say why"):
            issue(ledger, reason=reason)


# -- the bounds --------------------------------------------------------------

def test_a_grant_must_end_and_only_one_way_of_saying_when_is_allowed(ledger):
    clock = Clock()
    with pytest.raises(AllowanceError, match="exactly one of them"):
        grants(ledger, clock=clock).grant(actor=OWNER, reason="no clock", records=3,
                                          tokens=1000)
    with pytest.raises(AllowanceError, match="exactly one of them"):
        grants(ledger, clock=clock).grant(actor=OWNER, reason="two clocks", records=3,
                                          tokens=1000, duration_s=1.0,
                                          expires_at="2027-01-01T00:00:00+00:00")


def test_an_allowance_in_the_past_authorizes_nothing(ledger):
    with pytest.raises(AllowanceError, match="already expired"):
        grants(ledger, clock=Clock()).grant(actor=OWNER, reason="yesterday", records=3,
                                            tokens=1000,
                                            expires_at=time.strftime(
                                                "%Y-%m-%dT%H:%M:%S+00:00",
                                                time.gmtime(MOMENT - 60)))


def test_a_naive_instant_is_refused_because_it_is_a_different_instant_per_machine(ledger):
    with pytest.raises(AllowanceError, match="must be an instant with a timezone"):
        grants(ledger).grant(actor=OWNER, reason="local time", records=3, tokens=1000,
                             expires_at="2027-01-01T00:00:00")


def test_a_permission_longer_than_a_week_is_a_configuration_change(ledger):
    with pytest.raises(AllowanceError, match="at most 7 days"):
        issue(ledger, hours=MAX_DURATION_S / 3600 + 1)


def test_a_cap_of_nothing_is_not_a_permission(ledger):
    for kwargs in ({"records": 0}, {"records": -1}, {"tokens": 0}, {"tokens": -5}):
        with pytest.raises(AllowanceError):
            issue(ledger, **kwargs)


def test_a_second_live_grant_is_refused_and_names_the_first(ledger):
    first = issue(ledger, records=4, tokens=1000)
    with pytest.raises(AllowanceError, match=first["id"]):
        issue(ledger)


def test_revoking_twice_reports_the_state_rather_than_the_act(ledger):
    created = issue(ledger)
    grants(ledger).revoke(created["id"], actor=OWNER, reason="done for tonight")
    with pytest.raises(AllowanceError, match="already revoked"):
        grants(ledger).revoke(created["id"], actor=OWNER, reason="again")


# -- the clock ---------------------------------------------------------------

def test_an_expired_grant_answers_as_no_permission_without_being_written(ledger):
    clock = Clock()
    created = issue(ledger, hours=1.0, clock=clock)
    clock.now += 3601
    # `current()` is reached from `status`, and a status report must not create or migrate a
    # file it promised only to read — so it answers rather than updates.
    reading = grants(ledger, clock=clock)
    assert reading.current() is None
    assert reading.state(created["id"])["state"] == "active"


def test_the_pass_that_may_write_moves_the_stale_row(ledger):
    clock = Clock()
    created = issue(ledger, hours=1.0, clock=clock)
    clock.now += 3601
    assert grants(ledger, clock=clock).retire_expired() == [created["id"]]
    assert grants(ledger, clock=clock).state(created["id"])["state"] == "expired"


def test_a_grant_expiring_now_is_not_live(ledger):
    # The bound is exclusive at the expiry instant: `expires_at` is when permission stops, not
    # the last moment it holds, and a pass at that second would be one the ledger calls ended.
    clock = Clock()
    created = issue(ledger, hours=1.0, clock=clock)
    clock.now += 3600
    assert grants(ledger, clock=clock).current() is None
    row, refusal = grants(ledger, clock=clock).authorize(records=1, tokens=1,
                                                         resource=REMOTE,
                                                         allowance=created["id"])
    assert row is None and "expired at" in refusal


# -- the device --------------------------------------------------------------

def test_a_grant_for_one_device_does_not_cover_another(ledger):
    created = issue(ledger, resource="gpu-a")
    row, refusal = grants(ledger).authorize(records=1, tokens=1, resource="gpu-b",
                                            allowance=created["id"])
    assert row is None
    assert "a different device is a different decision" in refusal


def test_a_wildcard_grant_is_refused_while_anything_is_live(ledger):
    # Two live grants for one device would have to be summed to be read, and a `*` grant sits
    # over the named one it was refused beside.
    named = issue(ledger, resource=REMOTE)
    with pytest.raises(AllowanceError, match=named["id"]):
        issue(ledger, resource=ANY_RESOURCE)


def test_a_grant_for_one_device_is_not_the_live_answer_for_another(ledger):
    # `status` and `capabilities` ask this question with no grant id in hand, so a lookup that
    # matched any live row would report the machine as open for a device nobody permitted.
    created = issue(ledger, resource=REMOTE)
    reading = grants(ledger)
    assert reading.current(resource=REMOTE)["id"] == created["id"]
    assert reading.current(resource="gpu-b") is None
    row, refusal = reading.authorize(records=1, tokens=1, resource="gpu-b")
    assert row is None and "no active allowance covers" in refusal


def test_two_live_grants_can_only_differ_by_the_device_they_cover(ledger):
    """One device, one answer — which is why no precedence rule is needed to read the ledger.

    Two grants may stand together when they name different hardware; a wildcard over either, or
    a second grant for the same device, is refused at the door. A ledger that let both exist
    would have to be added up, and a permission that needs arithmetic is not a permission.
    """
    first = issue(ledger, resource="gpu-a")
    second = issue(ledger, resource="gpu-b")
    assert {item["id"] for item in grants(ledger).ledger()} == {first["id"], second["id"]}
    assert grants(ledger).current(resource="gpu-a")["id"] == first["id"]
    assert grants(ledger).current(resource="gpu-b")["id"] == second["id"]
    for resource in (ANY_RESOURCE, "gpu-a"):
        with pytest.raises(AllowanceError, match="already covers"):
            issue(ledger, resource=resource)


def test_a_grant_charged_past_its_cap_reports_no_credit_rather_than_a_negative_one(ledger):
    """A pass that overran on measured usage leaves the grant spent, not in credit.

    The last pass under a permission is allowed to be a little bigger than what was left,
    because the bound is checked on the plan and the charge is taken on the measurement. What
    must not follow is a ledger that reads `-3 records left` and invites another.
    """
    created = issue(ledger, records=2, tokens=2000)
    after = grants(ledger).consume(created["id"], records=5, tokens=9000)
    assert after["records"] == {"used": 5, "cap": 2, "left": 0}
    assert after["tokens"] == {"used": 9000, "cap": 2000, "left": 0}
    assert after["state"] == "spent"
    assert grants(ledger).current() is None


def test_a_wildcard_grant_covers_whatever_the_route_points_at(ledger):
    created = issue(ledger, resource=ANY_RESOURCE)
    for resource in (REMOTE, "gpu-b"):
        row, refusal = grants(ledger).authorize(records=1, tokens=1, resource=resource,
                                                allowance=created["id"])
        assert refusal == "" and row["id"] == created["id"]
    row, refusal = grants(ledger).authorize(records=1, tokens=1, resource=REMOTE,
                                            allowance=None)
    assert refusal == "" and row["id"] == created["id"]


# -- what a refusal says -----------------------------------------------------

def test_a_pass_too_big_for_the_grant_is_refused_with_the_number_to_fix(ledger):
    created = issue(ledger, records=3, tokens=4000)
    row, refusal = grants(ledger).authorize(records=4, tokens=1000, resource=REMOTE,
                                            allowance=created["id"])
    assert row is None and "3 of 3 record(s) left and this pass would take 4" in refusal
    row, refusal = grants(ledger).authorize(records=1, tokens=5000, resource=REMOTE,
                                            allowance=created["id"])
    assert row is None and "4,000 of 4,000 token(s) left" in refusal


def test_an_unknown_or_closed_grant_is_named_in_the_refusal(ledger):
    created = issue(ledger)
    grants(ledger).revoke(created["id"], actor=OWNER, reason="changed my mind")
    for identifier, needle in ((created["id"], "is revoked"), ("alw_missing",
                                                               "is recorded in this ledger")):
        row, refusal = grants(ledger).authorize(records=1, tokens=1, resource=REMOTE,
                                                allowance=identifier)
        assert row is None and needle in refusal


def test_an_empty_pass_is_refused_before_the_grant_is_read(ledger):
    created = issue(ledger)
    row, refusal = grants(ledger).authorize(records=0, tokens=0, resource=REMOTE,
                                            allowance=created["id"])
    assert row is None and "no pass to authorize" in refusal


def test_the_star_means_whichever_grant_is_live(ledger):
    created = issue(ledger, resource=REMOTE)
    row, refusal = grants(ledger).authorize(records=1, tokens=1, resource=REMOTE,
                                            allowance=ANY_RESOURCE)
    assert refusal == "" and row["id"] == created["id"]


# -- the accounting ----------------------------------------------------------

def test_spend_is_charged_in_the_amount_actually_measured(ledger):
    created = issue(ledger, records=3, tokens=4000)
    after = grants(ledger).consume(created["id"], records=2, tokens=642)
    assert after["records"] == {"used": 2, "cap": 3, "left": 1}
    assert after["tokens"]["used"] == 642 and after["passes"] == 1
    assert after["state"] == "active"


def test_a_grant_runs_out_rather_than_being_overruled(ledger):
    created = issue(ledger, records=2, tokens=10_000)
    grants(ledger).consume(created["id"], records=2, tokens=100)
    assert grants(ledger).state(created["id"])["state"] == "spent"
    assert grants(ledger).current() is None
    row, refusal = grants(ledger).authorize(records=1, tokens=1, resource=REMOTE,
                                            allowance=created["id"])
    assert row is None and "is spent" in refusal


def test_the_tokens_that_run_a_grant_out_are_the_measured_ones(ledger):
    created = issue(ledger, records=100, tokens=1000)
    grants(ledger).consume(created["id"], records=1, tokens=1000)
    assert grants(ledger).state(created["id"])["state"] == "spent"


def test_nothing_is_charged_twice_or_negatively(ledger):
    created = issue(ledger)
    with pytest.raises(AllowanceError, match="cannot be negative"):
        grants(ledger).consume(created["id"], records=-1, tokens=0)
    with pytest.raises(AllowanceError, match="to charge"):
        grants(ledger).consume("alw_nothere", records=1, tokens=1)


def test_what_a_reading_hides_is_what_a_ledger_must_show(ledger):
    created = issue(ledger, records=5, tokens=5000)
    grants(ledger).consume(created["id"], records=5, tokens=10)
    assert [item["id"] for item in grants(ledger).ledger()] == []
    closed = grants(ledger).ledger(include_closed=True)
    assert [item["state"] for item in closed] == ["spent"]
    assert closed[0]["reason"] == "form while I sleep"
    assert closed[0]["actor"] == OWNER


def test_a_grant_is_recorded_under_the_stage_it_governs(ledger):
    created = issue(ledger)
    assert created["stage"] == STAGE and created["scope"] == "global"
    assert created["expires_at"] > created["granted_at"]
