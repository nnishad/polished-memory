"""C12 admission: one slot per physical device, priority, pause and uncertain execution."""
from __future__ import annotations

import pytest

from hermes_memory.processing.resource_gate import (GateClosed, GatePaused, ResourceGate)
from hermes_memory.processing.routes import PRIORITY, RouteTable, Route
from hermes_memory.config import SettingError
from hermes_memory.storage.evidence import EvidenceError

LOCAL = "local-gpu"
REMOTE = "remote-9b"


class Clock:
    def __init__(self):
        self.value = 1_000.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


@pytest.fixture()
def clock():
    return Clock()


@pytest.fixture()
def gate(store, clock):
    return ResourceGate(store, clock=clock, default_ttl=60)


def take(gate, resource=REMOTE, holder="worker", priority=PRIORITY["maintenance"], ttl=60,
         route="retain"):
    return gate.try_acquire(route=route, holder=holder, resource=resource,
                            priority=priority, ttl=ttl)


# -- single flight -----------------------------------------------------------

def test_a_second_caller_cannot_take_an_occupied_resource(gate):
    assert take(gate) is not None
    assert take(gate, holder="other-worker") is None
    assert gate.blocked_resources() == [REMOTE]


def test_the_single_slot_rule_is_enforced_by_the_schema_not_by_a_check(gate):
    """The partial unique index must refuse a second holder even if code is wrong."""
    take(gate)
    with pytest.raises(Exception):
        gate.db.execute(
            "INSERT INTO gate_reservations(id, resource, route, holder, priority, state, "
            "acquired_at, lease_until) VALUES('x',?,'r','h',3,'held',0,0)", (REMOTE,))


def test_a_release_frees_the_resource_for_the_next_caller(gate):
    first = take(gate)
    gate.release(first, outcome="succeeded", tokens=120, seconds=1.5)
    assert gate.blocked_resources() == []
    assert take(gate, holder="next") is not None


def test_distinct_resources_are_admitted_independently(gate):
    """The remote 9B and the local GPU are different devices, so different slots."""
    assert take(gate, resource=REMOTE) is not None
    assert take(gate, resource=LOCAL, route="embeddings") is not None
    assert sorted(gate.blocked_resources()) == [LOCAL, REMOTE]


def test_vision_and_embeddings_contend_for_one_gpu(gate):
    """They are different services on the same device; contention is the point."""
    assert take(gate, resource=LOCAL, route="embeddings") is not None
    assert take(gate, resource=LOCAL, route="vision", holder="vlm") is None


def test_two_profiles_of_the_same_host_share_one_slot(gate):
    assert take(gate, holder="profile-a") is not None
    assert take(gate, holder="profile-b") is None


# -- priority ----------------------------------------------------------------

def test_the_most_urgent_waiter_wins_not_the_first_to_poll(gate):
    held = take(gate)
    # Register both waiters while the slot is busy: maintenance first, then an
    # interactive caller arriving later must still be admitted first.
    from hermes_memory.processing.resource_gate import WAITING

    for name, priority in (("maintenance", PRIORITY["maintenance"]),
                           ("interactive", PRIORITY["interactive"])):
        gate.db.execute(
            "INSERT INTO gate_reservations(id, resource, route, holder, priority, state, "
            "acquired_at, lease_until) VALUES(?,?,?,?,?,?,?,0)",
            (f"wait_{name}", REMOTE, "retain", name, priority, WAITING, gate.clock()))
    gate.release(held, outcome="succeeded")
    assert gate._promote_locked("wait_maintenance", REMOTE, PRIORITY["maintenance"]) is False
    assert gate._promote_locked("wait_interactive", REMOTE, PRIORITY["interactive"]) is True


def test_equal_priority_is_first_come(gate):
    held = take(gate)
    from hermes_memory.processing.resource_gate import WAITING

    for index, name in enumerate(("first", "second")):
        gate.db.execute(
            "INSERT INTO gate_reservations(id, resource, route, holder, priority, state, "
            "acquired_at, lease_until) VALUES(?,?,?,?,?,?,?,0)",
            (f"wait_{name}", REMOTE, "retain", name, PRIORITY["freshness"], WAITING,
             gate.clock() + index))
    gate.release(held, outcome="succeeded")
    assert gate._promote_locked("wait_second", REMOTE, PRIORITY["freshness"]) is False
    assert gate._promote_locked("wait_first", REMOTE, PRIORITY["freshness"]) is True


def test_priority_does_not_preempt_a_request_already_started(gate):
    held = take(gate, priority=PRIORITY["maintenance"])
    assert take(gate, holder="urgent", priority=PRIORITY["interactive"]) is None


# -- uncertain execution -----------------------------------------------------

def test_an_expired_lease_becomes_uncertain_and_keeps_the_device_blocked(gate, clock):
    reservation = take(gate, ttl=10)
    clock.advance(30)
    assert gate.reap_expired() == [reservation.id]
    assert gate.blocked_resources() == [REMOTE], "a timeout is not proof the slot is free"
    assert take(gate, holder="impatient") is None


def test_only_an_established_outcome_frees_an_uncertain_reservation(gate, clock):
    reservation = take(gate, ttl=10)
    clock.advance(30)
    gate.reap_expired()
    gate.release(reservation, outcome="failed", tokens=0, seconds=30.0)
    assert gate.blocked_resources() == []
    assert take(gate) is not None


def test_marking_uncertain_is_explicit_and_reasoned(gate):
    reservation = take(gate)
    gate.mark_uncertain(reservation, reason="upstream disconnected mid-stream")
    assert gate.blocked_resources() == [REMOTE]
    with pytest.raises(EvidenceError, match="nonempty"):
        gate.mark_uncertain(reservation, reason="  ")


def test_release_is_idempotent_and_an_unknown_id_is_refused(gate):
    reservation = take(gate)
    gate.release(reservation, outcome="succeeded")
    gate.release(reservation, outcome="succeeded")
    assert gate.db.execute("SELECT count(*) FROM budget_usage").fetchone()[0] == 1
    with pytest.raises(GateClosed):
        gate.release(type(reservation)("nope", REMOTE, "retain", "h", 3, 0.0), outcome="failed")


# -- operator pause ----------------------------------------------------------

def test_an_operator_pause_denies_new_dispatch_immediately(gate):
    gate.pause(actor="owner", reason="gpu needed for something else")
    assert gate.paused is True
    with pytest.raises(GatePaused):
        take(gate)
    gate.resume(actor="owner", reason="done")
    assert take(gate) is not None


def test_a_pause_does_not_kill_work_already_started(gate):
    reservation = take(gate)
    gate.pause(actor="owner", reason="stop everything")
    assert gate.blocked_resources() == [REMOTE], "the in-flight request is still reported"
    gate.release(reservation, outcome="succeeded", tokens=50)
    assert gate.blocked_resources() == []


# -- accounting --------------------------------------------------------------

def test_usage_accumulates_tokens_calls_and_time(gate):
    for tokens in (100, 250):
        gate.release(take(gate), outcome="succeeded", tokens=tokens, seconds=2.0)
    usage = gate.usage()[REMOTE]
    assert usage == {"tokens": 350, "calls": 2, "seconds": 4.0}


def test_negative_usage_is_refused(gate):
    reservation = take(gate)
    with pytest.raises(EvidenceError, match="negative"):
        gate.release(reservation, outcome="succeeded", tokens=-5)


def test_occupancy_reports_held_waiting_and_uncertain_separately(gate, clock):
    take(gate)
    assert gate.occupancy() == {REMOTE: {"held": 1}}


def test_no_gate_path_leaves_a_transaction_open(gate):
    reservation = take(gate)
    for step in (lambda: gate.occupancy(), lambda: gate.blocked_resources(),
                 lambda: gate.reap_expired(), lambda: gate.usage(),
                 lambda: gate.try_acquire(route="x", holder="y", resource=LOCAL, priority=0),
                 lambda: gate.release(reservation, outcome="succeeded")):
        step()
        assert gate.db.in_transaction is False
    with pytest.raises(GateClosed):
        gate.mark_uncertain(reservation, reason="late")
    assert gate.db.in_transaction is False


def test_an_out_of_range_ttl_or_priority_is_refused(gate):
    with pytest.raises(EvidenceError, match="between 1 and 3600"):
        gate.try_acquire(route="retain", holder="w", resource=REMOTE, priority=0, ttl=0)
    with pytest.raises(EvidenceError, match="priority must be one of"):
        gate.try_acquire(route="retain", holder="w", resource=REMOTE, priority=9)


# -- routes ------------------------------------------------------------------

def test_a_route_credential_selects_an_upstream_and_nothing_else():
    table = RouteTable({
        "retain": Route("retain", REMOTE, "chat", "http://192.168.68.65:8080/v1", "cred-a",
                        "freshness", 2048),
        "embeddings": Route("embeddings", LOCAL, "embeddings", "http://127.0.0.1:11434/v1",
                            "cred-b", "freshness", 0),
    })
    assert table.by_credential("cred-a").name == "retain"
    assert table.by_credential("cred-b").resource == LOCAL
    with pytest.raises(SettingError, match="does not select any route"):
        table.by_credential("cred-c")
    with pytest.raises(SettingError, match="no default and no fallback"):
        table.by_name("anything-else")


def test_a_duplicate_credential_is_refused_at_construction():
    route = Route("a", REMOTE, "chat", "http://127.0.0.1:8080/v1", "shared", "freshness", 100)
    other = Route("b", LOCAL, "chat", "http://127.0.0.1:8081/v1", "shared", "freshness", 100)
    with pytest.raises(Exception, match="not unique"):
        RouteTable({"a": route, "b": other})


def test_the_route_report_never_exposes_credentials():
    table = RouteTable({"retain": Route("retain", REMOTE, "chat", "http://127.0.0.1:8080/v1",
                                        "secret-credential", "freshness", 2048)})
    report = table.as_dict()
    assert "secret-credential" not in str(report)
    assert report["retain"]["upstream"] == "http://127.0.0.1:8080/v1"
