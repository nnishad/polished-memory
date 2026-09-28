"""C12 admission: one slot per physical device, priority, pause and uncertain execution."""
from __future__ import annotations

import pytest

from hermes_memory.processing.resource_gate import (GateClosed, GatePaused, ResourceGate)
from hermes_memory.processing.routes import PRIORITY, RouteTable, RouteError, Route, build_routes
from hermes_memory.config import SettingError, load_settings
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


# -- what a route actually answered ------------------------------------------

def test_the_newest_settled_answer_is_kept_per_route(gate, clock):
    """The ledger is the record of what came back, and only the newest one answers now."""
    first = take(gate, resource=REMOTE, route="retain")
    gate.release(first, outcome="succeeded")
    clock.advance(10)
    earlier = take(gate, resource=LOCAL, route="reflect")
    gate.release(earlier, outcome="succeeded")
    clock.advance(10)
    latest = take(gate, resource=REMOTE, route="reflect")
    gate.release(latest, outcome="failed")

    outcomes = gate.last_outcomes()
    assert sorted(outcomes) == ["reflect", "retain"]
    assert outcomes["reflect"]["outcome"] == "failed", "the newest dispatch is the answer"
    assert outcomes["reflect"]["state"] == "released"
    assert outcomes["reflect"]["resource"] == REMOTE
    assert outcomes["reflect"]["settled_at"] == 1_020.0
    assert outcomes["retain"]["outcome"] == "succeeded"


def test_a_dispatch_that_has_not_settled_is_not_an_answer(gate):
    """A held slot has produced no outcome, so the route is not claimed to be working."""
    take(gate, resource=REMOTE, route="retain")
    assert gate.last_outcomes() == {}


def test_an_answer_nobody_established_is_reported_as_the_open_question_it_is(gate):
    """An unresolved reservation keeps the device blocked; its reason is the route's news."""
    held = take(gate, resource=REMOTE, route="reflect")
    gate.mark_uncertain(held, reason="the connection went away mid-request")
    outcome = gate.last_outcomes()["reflect"]
    assert outcome["state"] == "uncertain"
    assert outcome["outcome"] == "the connection went away mid-request"


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


# -- an operator settling an unresolved request --------------------------------

def stranded(gate, resource=REMOTE):
    """A request whose answer never came back, which is the state nothing else frees."""
    reservation = take(gate, resource=resource)
    gate.mark_uncertain(reservation, reason="connection lost mid-request")
    return reservation


def test_an_operator_who_learnt_the_answer_can_free_the_device(gate, clock):
    """The plan's rule is that a lost request blocks its device until proven otherwise.

    "Until proven otherwise" has to be sayable, or one dropped connection retires a GPU for
    the rest of the installation's life: the reconciler that finds the record projected, and
    the cancellation the backend acknowledged, are both an answer about somebody else's
    reservation, and until now there was nowhere to give it.
    """
    reservation = stranded(gate)
    assert gate.unresolved()[0]["id"] == reservation.id
    settled = gate.resolve(reservation.id, outcome="cancelled", actor="jugaadu",
                           reason="the backend acknowledged the stop")
    assert settled["state"] == "released" and settled["settled_by"] == "jugaadu"
    assert gate.unresolved() == []


def test_a_settled_device_is_admissible_again(gate):
    reservation = take(gate)
    gate.mark_uncertain(reservation, reason="connection lost mid-request")
    assert gate.try_acquire(route="retain", holder="worker-2", resource=REMOTE,
                            priority=2, ttl=60) is None
    gate.resolve(reservation.id, outcome="cancelled", actor="owner",
                 reason="the backend acknowledged the cancellation")
    next_up = take(gate, holder="worker-2")
    assert next_up is not None
    gate.release(next_up, outcome="succeeded", tokens=1)
    assert gate.blocked_resources() == []


def test_settling_needs_an_outcome_a_name_and_a_reason(gate):
    reservation = stranded(gate)
    for kwargs in ({"outcome": "cancelled", "actor": "", "reason": "asked"},
                   {"outcome": "cancelled", "actor": "owner", "reason": "  "},
                   {"outcome": "unknown", "actor": "owner", "reason": "asked"}):
        with pytest.raises(EvidenceError):
            gate.resolve(reservation.id, **kwargs)
    assert gate.blocked_resources() == [REMOTE], "a refused settlement changes nothing"


def test_an_uncertain_answer_is_not_an_answer(gate):
    """The whole point is that somebody established what happened.

    "Still don't know" is the row's current state, and recording it as settled would let the
    next caller onto a device nobody has proved idle — the guess this gate exists to prevent.
    """
    reservation = stranded(gate)
    with pytest.raises(EvidenceError, match="stays unresolved"):
        gate.resolve(reservation.id, outcome="uncertain", actor="owner", reason="gave up")


def test_a_live_lease_is_not_the_operators_to_take(gate):
    """A worker that is still running its request must not be handed out from under."""
    reservation = take(gate)
    with pytest.raises(GateClosed, match="live lease"):
        gate.resolve(reservation.id, outcome="cancelled", actor="owner", reason="impatience")
    assert gate.blocked_resources() == [REMOTE]


def test_a_reservation_its_holder_already_released_is_not_resettled(gate):
    reservation = take(gate)
    gate.release(reservation, outcome="succeeded", tokens=10)
    with pytest.raises(GateClosed, match="already released by its holder"):
        gate.resolve(reservation.id, outcome="cancelled", actor="owner", reason="too late")
    assert gate.usage()[REMOTE]["tokens"] == 10, "the holder's own charge stands"


def test_settling_charges_nothing_because_nobody_saw_the_usage(gate):
    """An invented token count would make the day's spend a guess wearing a total."""
    reservation = stranded(gate)
    gate.resolve(reservation.id, outcome="failed", actor="owner", reason="checked the server")
    assert gate.usage() == {}


def test_the_ledger_keeps_who_answered_for_a_lost_request(gate):
    reservation = stranded(gate)
    gate.resolve(reservation.id, outcome="succeeded", actor="jugaadu",
                 reason="the reconciler found the record projected")
    events = [dict(row) for row in gate.db.execute(
        "SELECT event, detail FROM gate_ledger WHERE reservation_id=? ORDER BY rowid",
        (reservation.id,)).fetchall()]
    settled = events[-1]
    assert settled["event"] == "resolved"
    assert "jugaadu" in settled["detail"] and "projected" in settled["detail"]
    assert "upstream disconnected" not in settled["detail"]


def test_unresolved_lists_the_rows_that_block_a_device_with_nobody_left_to_answer(gate):
    reservation = stranded(gate)
    take(gate, resource=LOCAL)
    rows = gate.unresolved()
    assert [row["id"] for row in rows] == [reservation.id]
    assert rows[0]["resource"] == REMOTE and rows[0]["route"] == "retain"
    assert rows[0]["holder"] == "worker"


# -- vouching for a lease that is still being used -----------------------------

def test_renewing_a_live_lease_moves_the_deadline_and_says_so(gate, clock):
    reservation = take(gate, ttl=60)
    clock.advance(20)
    until = gate.renew(reservation, ttl=60)
    assert until == clock.value + 60 and gate.held()[0]["lease_until"] == until
    rows = gate.db.execute("SELECT event, detail FROM gate_ledger WHERE reservation_id=? "
                           "ORDER BY rowid", (reservation.id,)).fetchall()
    assert [row["event"] for row in rows] == ["acquired", "renewed"], \
        "who vouched, and when, is the answer an expiry later needs"


def test_a_promise_that_ran_out_is_not_renewed_into_existence(gate, clock):
    """The state can still say `held` for as long as nothing reaps it. That is not ours.

    Another admission may have been promoted over the top of the lapsed row, so vouching
    for it again would be a second claim on one device made by the process that lost it.
    """
    reservation = take(gate, ttl=60)
    clock.advance(61)
    assert gate.renew(reservation) is None
    assert gate.blocked_resources() == [REMOTE], "lapsed is not the same as free"


def test_a_released_reservation_is_not_renewable(gate):
    reservation = take(gate)
    gate.release(reservation, outcome="succeeded")
    assert gate.renew(reservation) is None


def test_a_renewal_outside_the_lease_window_is_refused(gate):
    reservation = take(gate)
    for ttl in (0, 3601, -5):
        with pytest.raises(EvidenceError, match="between 1 and 3600"):
            gate.renew(reservation, ttl=ttl)


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


BASE = {
    "DATA_DIR": None,  # filled per home by `configured`
    "INFERENCE_ENABLED": "true",
    "OWNER_PRINCIPAL": "jugaadu",
    "HINDSIGHT_URL": "http://127.0.0.1:8888",
    "ALLOWED_INFERENCE_HOSTS": "127.0.0.1,192.168.68.65",
    "TEXT_BASE_URL": "http://192.168.68.65:8080/v1",
    "TEXT_RESOURCE": REMOTE,
    "EMBEDDINGS_BASE_URL": "http://127.0.0.1:11434/v1",
    "EMBEDDINGS_RESOURCE": LOCAL,
    "VISION_BASE_URL": "http://127.0.0.1:8080/v1",
    "VISION_RESOURCE": LOCAL,
}


def configured(tmp_path, monkeypatch, *, credentials, over=None):
    """A real settings object for a machine with two models and minted route credentials.

    `over` changes or (with None) removes a line, because the refusals below are about what
    an installation left out.
    """
    home = tmp_path / "instance"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    values = dict(BASE, DATA_DIR=str(home / "data"))
    for key, value in (over or {}).items():
        if value is None:
            values.pop(key)
        else:
            values[key] = value
    lines = [f"HERMES_MEMORY_{key}={value}" for key, value in values.items()]
    lines += [f"HERMES_MEMORY_ROUTE_CREDENTIAL_{name}={value}"
              for name, value in credentials.items()]
    (home / "hermes-memory.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_settings()


def test_a_route_the_owner_configured_but_never_minted_is_withheld_not_fatal(tmp_path,
                                                                             monkeypatch):
    """An upstream with no credential is unusable; it is not a reason to stop serving.

    The gate used to refuse to start over it, which took the credentialed routes down with
    it and left a `Restart=always` unit crash-looping because a vision model was waiting for
    its canary. Withholding is per route, and the reason is said out loud.
    """
    settings = configured(tmp_path, monkeypatch,
                          credentials={"RETAIN": "cred-retain", "EMBEDDINGS": "cred-embed"})
    table = build_routes(settings, credentials=settings.route_credentials)
    # Each operation presents its own credential, so minting two of six leaves four dark —
    # and the two that are authenticated are the ones the installation can still use.
    assert table.names() == ["embeddings", "retain"]
    assert sorted(table.withheld) == ["consolidate", "foreground", "reflect", "vision"]
    assert "HERMES_MEMORY_ROUTE_CREDENTIAL_VISION" in table.withheld["vision"]
    with pytest.raises(RouteError, match="withheld"):
        table.by_name("vision")


def test_a_gate_that_would_serve_nothing_refuses_to_start_at_all(tmp_path, monkeypatch):
    """Withholding every route is not a partial gate; it is an installation with no models."""
    settings = configured(tmp_path, monkeypatch, credentials={})
    with pytest.raises(SettingError, match="no inference route is admitted"):
        build_routes(settings, credentials=settings.route_credentials)


def test_no_route_is_built_before_a_backend_is_named(tmp_path, monkeypatch):
    settings = configured(tmp_path, monkeypatch, over={"HINDSIGHT_URL": None},
                          credentials={"RETAIN": "cred-retain"})
    with pytest.raises(SettingError, match="nothing to map"):
        build_routes(settings, credentials=settings.route_credentials)


def test_a_generation_route_needs_both_a_text_and_an_embeddings_upstream(tmp_path, monkeypatch):
    """Recall without embeddings finds nothing, and embeddings without text forms nothing."""
    for missing in ("TEXT_BASE_URL", "EMBEDDINGS_BASE_URL"):
        settings = configured(tmp_path / missing, monkeypatch,
                              over={missing: None}, credentials={"RETAIN": "cred-retain"})
        with pytest.raises(SettingError, match="both required"):
            build_routes(settings, credentials=settings.route_credentials)


def test_the_embeddings_route_carries_no_generation_cap(tmp_path, monkeypatch):
    """An output cap is a promise about tokens a route generates; embeddings generates none."""
    settings = configured(tmp_path, monkeypatch,
                          credentials={"RETAIN": "cred-retain", "EMBEDDINGS": "cred-embed"})
    table = build_routes(settings, credentials=settings.route_credentials)
    assert table.by_name("embeddings").max_output_tokens == 0
    assert table.by_name("retain").max_output_tokens > 0


def test_only_the_route_a_person_is_waiting_on_holds_the_interactive_slot(tmp_path,
                                                                         monkeypatch):
    """The foreground request is the one the host truncates at eight seconds.

    If a background operation shared its priority, a consolidation run already holding the
    model would decide when a reply to the owner is answered, and the degradation would be
    invisible: the host gives up and the memory looks slow rather than misprioritised.
    """
    settings = configured(tmp_path, monkeypatch, credentials={
        "RETAIN": "cred-retain", "CONSOLIDATE": "cred-consolidate", "REFLECT": "cred-reflect",
        "FOREGROUND": "cred-foreground", "EMBEDDINGS": "cred-embed"})
    table = build_routes(settings, credentials=settings.route_credentials)
    ranks = {name: table.by_name(name).priority_rank() for name in table.names()}
    assert ranks["foreground"] == min(ranks.values())
    assert all(ranks["foreground"] < ranks[name] for name in ("retain", "consolidate",
                                                             "reflect"))
