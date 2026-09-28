"""C3 runtime: one lease, bounded pages, and a stop that says why.

The gates these pin are the ones the plan named: crash at a transaction boundary,
cursor expiry, duplicate and out-of-order pages, lease theft, source
reconfiguration, reset mid-fetch, replay, and the difference between pausing a
source and pausing only its ingestion.
"""
from __future__ import annotations

import json

import pytest

from connector_script import AT, OTHER, SOURCE, Scripted, notes
from conftest import envelope
from hermes_memory.sources.base import Page, Skipped
from hermes_memory.sources.runtime import (COMPLETE, CONTENDED, CURSOR_EXPIRED, EXHAUSTED,
                                           PAUSED, STALE, STALLED, UNREACHABLE,
                                           ConnectorRuntime, Run)
from hermes_memory.sources.sync import DOWNSTREAM_STAGES, SyncController
from hermes_memory.storage.evidence import EvidenceError

NOW = 1000.0


class Clock:
    """Lease expiry has to be testable without sleeping."""

    def __init__(self):
        self.value = NOW

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@pytest.fixture()
def sync(store):
    controller = SyncController(store, clock=Clock())
    controller.register(SOURCE, policy_version="local-only")
    return controller


@pytest.fixture()
def runtime(store, sync):
    return ConnectorRuntime(store, sync, holder="worker-a")


def live(store, source=SOURCE):
    return int(store.db.execute(
        "SELECT count(*) FROM records WHERE deleted=0 AND source=?", (source,)).fetchone()[0])


def journaled(store, source=SOURCE):
    return int(store.db.execute(
        "SELECT count(*) FROM change_journal WHERE source=?", (source,)).fetchone()[0])


# -- starting a pass ---------------------------------------------------------

def test_an_unconfigured_source_is_refused_before_anything_is_read(runtime):
    with pytest.raises(EvidenceError, match="not registered"):
        runtime.run(Scripted([notes(1, next_cursor=None)], source="nope"))


@pytest.mark.parametrize("name", ["", "   ", None])
def test_an_adapter_that_does_not_name_its_source_is_refused(runtime, name):
    with pytest.raises(EvidenceError, match="declare the source"):
        runtime.run(Scripted([notes(1)], source=name))


def test_a_pass_takes_the_lease_and_gives_it_back(store, sync, runtime):
    result = runtime.run(Scripted([notes(2, next_cursor=None)]))

    assert result.stopped == COMPLETE and result.records == 2
    state = sync.state(SOURCE)
    assert state["holder"] is None and state["lease"] is None
    assert live(store) == 2


def test_a_lease_held_by_another_worker_is_not_taken(store, sync, runtime):
    other = sync.acquire(SOURCE, holder="worker-b")
    before = sync.state(SOURCE)

    result = runtime.run(Scripted([notes(2, next_cursor=None)]))

    assert result.stopped == CONTENDED and result.records == 0
    assert sync.state(SOURCE)["lease"] == before["lease"]
    assert live(store) == 0
    sync.release(other)


def test_a_worker_may_carry_on_with_its_own_lease(store, sync, runtime):
    sync.acquire(SOURCE, holder="worker-a")

    assert runtime.run(Scripted([notes(1, next_cursor=None)])).stopped == COMPLETE


def test_a_holder_that_lapsed_is_taken_over_by_the_next_pass(store, sync, runtime):
    clock = [NOW]
    holder = SyncController(store, clock=lambda: clock[0])
    holder.register(SOURCE, policy_version="local-only")
    runner = ConnectorRuntime(store, holder, holder="worker-a", lease_s=10)
    runner.run(Scripted([notes(1, next_cursor="0")]))
    clock[0] = NOW + 600
    rival = ConnectorRuntime(store, holder, holder="worker-b", lease_s=10)

    result = rival.run(Scripted([notes(1, start=1, next_cursor=None)]))

    assert result.stopped == COMPLETE and result.records == 1


# -- pages, cursors and replay ----------------------------------------------

def test_each_page_advances_the_cursor_in_one_transaction(store, sync, runtime):
    first = notes(2, next_cursor="0")
    second = notes(2, start=2, next_cursor="1")
    third = notes(1, start=4, next_cursor=None)

    partial = runtime.run(Scripted([first, second, third]), max_pages=2)

    assert partial.stopped == EXHAUSTED and partial.records == 4
    assert sync.state(SOURCE)["cursor"] == "1"
    assert live(store) == 4, "the page that was read is the page that was committed"

    rest = runtime.run(Scripted([third]))
    assert rest.stopped == COMPLETE and rest.records == 1
    assert sync.state(SOURCE)["coverage_state"] == "current"


def test_the_page_budget_is_reported_rather_than_silently_truncated(store, sync, runtime):
    many = [notes(1, start=index, next_cursor=str(index)) for index in range(10)]

    result = runtime.run(Scripted(many), max_pages=3)

    assert result.pages == 3 and "budget" in result.note
    assert result.stopped == EXHAUSTED


def test_an_absurd_page_budget_falls_back_to_the_default(store, sync, runtime):
    result = runtime.run(Scripted([notes(1, next_cursor=None)]), max_pages="many")
    assert result.pages == 1


def test_re_reading_unchanged_pages_writes_nothing_and_moves_nothing(store, sync, runtime):
    script = [notes(2, next_cursor="0"), notes(2, start=2, next_cursor=None)]
    runtime.run(Scripted(script))
    cursor, before = sync.state(SOURCE)["cursor"], live(store)

    again = runtime.run(Scripted(script))

    assert again.records == 0 and again.repeats >= 1
    assert live(store) == before
    assert sync.state(SOURCE)["cursor"] == cursor


def test_new_content_at_a_stored_cursor_is_work_not_a_contradiction(store, sync, runtime):
    """The bug that wedges a connector: keying a page on its position, not its bytes."""
    runtime.run(Scripted([notes(2, next_cursor="head")]))

    later = runtime.run(Scripted([notes(2, start=2, next_cursor="head"),
                                  notes(1, start=4, next_cursor=None)]))

    assert later.stopped == COMPLETE and later.records == 3
    assert live(store) == 5


def test_a_source_that_names_its_pages_and_changes_them_is_refused(store, sync, runtime):
    runtime.run(Scripted([notes(2, next_cursor="0", token="history-9")]))

    with pytest.raises(EvidenceError, match="different content"):
        runtime.run(Scripted([notes(3, start=2, next_cursor="0", token="history-9")]))


def test_a_page_that_repeats_the_cursor_and_adds_nothing_is_a_stop_not_a_spin(
        store, sync, runtime):
    runtime.run(Scripted([notes(1, next_cursor="head")]))
    spinner = Scripted([notes(1, next_cursor="head")] * 20)

    result = runtime.run(spinner)

    assert result.stopped == STALLED and result.records == 0
    assert spinner.reads <= 2, "a stalled source is noticed, not drained"


def test_a_source_that_mislabels_its_records_is_not_filed_as_an_outage(store, sync, runtime):
    wrong = [notes(1, next_cursor=None, source=OTHER)]

    with pytest.raises(EvidenceError, match="claims source"):
        runtime.run(Scripted(wrong))

    assert sync.state(SOURCE)["coverage_state"] == "unknown"
    assert sync.state(SOURCE)["lease"] is None, "the lease is given back either way"


# -- the fence ---------------------------------------------------------------

def test_a_reset_while_a_page_is_in_flight_fences_the_writer(store, sync, runtime):
    def bump(index):
        if index == 1:
            store.db.execute("BEGIN IMMEDIATE")
            store.db.execute("UPDATE memory_epoch SET value=value+1 WHERE id=1")
            store.db.execute("COMMIT")

    script = [notes(2, next_cursor="0"), notes(2, start=2, next_cursor=None)]
    result = runtime.run(Scripted(script, on_read=bump))

    assert result.stopped == STALE
    assert live(store) == 2, "only the page committed before the reset landed"
    assert sync.state(SOURCE)["cursor"] == "0"


def test_a_reconfiguration_while_a_page_is_in_flight_fences_it_too(store, sync, runtime):
    def reconfigure(index):
        if index == 1:
            sync.reconfigure(SOURCE, policy_version="private-api", actor="owner",
                             reason="the account was re-mapped")

    result = runtime.run(Scripted([notes(2, next_cursor="0"),
                                   notes(2, start=2, next_cursor=None)],
                                  on_read=reconfigure))

    assert result.stopped == STALE and "reconfigured" in result.note
    assert live(store) == 2


def test_a_lease_that_lapses_mid_pass_is_not_the_writers_to_keep(store, sync):
    """A stalled connector is stopped by the fence rather than allowed to catch up."""
    class Slow(Scripted):
        def read_page(self, cursor):
            page = super().read_page(cursor)
            if self.reads == 2:                      # the second page is the slow one
                sync.clock.advance(600)
                sync.acquire(SOURCE, holder="worker-b", ttl=3600)
            return page

    runner = ConnectorRuntime(store, sync, holder="worker-a", lease_s=30)
    result = runner.run(Slow([notes(2, next_cursor="0"),
                              notes(2, start=2, next_cursor=None)]))

    assert result.stopped == STALE and result.records == 2
    assert sync.state(SOURCE)["holder"] == "worker-b"
    assert live(store) == 2, "the page committed before the lapse is the page that stays"


def test_a_pass_after_a_reset_starts_under_the_new_generation(store, sync, runtime):
    runtime.run(Scripted([notes(1, next_cursor="0")]))
    sync.reconfigure(SOURCE, policy_version="private-api", actor="owner",
                     reason="new credentials")

    result = runtime.run(Scripted([notes(1, start=1, next_cursor=None)]))

    assert result.generation == 2 and result.records == 1
    assert sync.state(SOURCE)["cursor"] is None, "the terminal page reports no position"


def test_the_lease_is_renewed_across_a_long_backfill(store, sync):
    clock = [NOW]
    renewing = SyncController(store, clock=lambda: clock[0])
    renewing.register(SOURCE, policy_version="local-only")
    runner = ConnectorRuntime(store, renewing, holder="worker-a", lease_s=30)
    # Two pages of work take longer than one lease; without the renewal the second
    # commit would be refused and the backfill would stop half done.
    script = [notes(1, start=index, next_cursor=str(index)) for index in range(2)]
    script += [notes(1, start=2, next_cursor=None)]

    def advance(index):
        clock[0] += 20

    result = runner.run(Scripted(script, on_read=advance))

    assert clock[0] == NOW + 60, "three pages, each slower than the whole lease"
    assert result.stopped == COMPLETE and result.records == 3


def test_an_expired_cursor_is_forgotten_rather_than_retried(store, sync, runtime):
    """A position the source will not accept is restarted, not called an outage."""
    from hermes_memory.sources.base import CursorExpired

    class Stale(Scripted):
        def read_page(self, cursor):
            if cursor == "gone":
                raise CursorExpired("the history id is older than the retention window")
            return super().read_page(cursor)

    fence = sync.acquire(SOURCE, holder="worker-a")
    sync.publish(fence, "page-1", [notes(1).envelopes[0]], next_cursor="gone")
    sync.release(fence)
    assert sync.state(SOURCE)["cursor"] == "gone"

    result = runtime.run(Stale([notes(1, start=5, next_cursor=None)]))

    assert result.stopped == CURSOR_EXPIRED and result.records == 0
    assert sync.state(SOURCE)["cursor"] is None, "the refused position is dropped"
    assert sync.state(SOURCE)["coverage_state"] == "partial"
    assert "retention" in result.note
    audit = store.db.execute(
        "SELECT metadata FROM audit WHERE action='connector_restart'").fetchone()
    assert audit and "retention" in json.loads(audit["metadata"])["reason"]

    # The next pass starts over and the evidence it re-reads is not written twice.
    again = runtime.run(Scripted([notes(1, start=5, next_cursor=None)]))
    assert again.stopped == COMPLETE and again.records == 1
    assert live(store) == 2


def test_a_check_can_report_the_cursor_as_expired(store, sync, runtime):
    """The probe is allowed to be the one that knows the position is dead."""
    from hermes_memory.sources.base import CursorExpired

    class Probed(Scripted):
        def check(self):
            raise CursorExpired("this saved position predates the retention window")

    result = runtime.run(Probed([notes(1, next_cursor=None)]))

    assert result.stopped == CURSOR_EXPIRED
    assert result.records == 0 and sync.state(SOURCE)["cursor"] is None


def test_a_restart_that_races_a_new_lease_does_not_clear_the_position(
        store, sync, runtime):
    """Restarting is a fenced write like any other."""
    from hermes_memory.sources.base import CursorExpired

    class StealThenExpire(Scripted):
        def read_page(self, cursor):
            if self.reads == 1:
                sync.clock.advance(600)
                sync.acquire(SOURCE, holder="worker-b", ttl=3600)
                raise CursorExpired("the cursor went stale with the lease")
            return super().read_page(cursor)

    runner = ConnectorRuntime(store, sync, holder="worker-a", lease_s=30)
    result = runner.run(StealThenExpire([notes(1, next_cursor="0"),
                                         notes(1, start=1, next_cursor=None)]))

    assert result.stopped == CURSOR_EXPIRED
    assert sync.state(SOURCE)["cursor"] == "0", "the new holder's position is not cleared"


# -- gaps --------------------------------------------------------------------

def test_what_the_source_could_not_give_us_is_kept_rather_than_logged(store, sync, runtime):
    script = [notes(2, next_cursor=None,
                    skipped=[Skipped("att-1", "unsupported type"),
                             Skipped("att-2", "over the size bound")])]

    result = runtime.run(Scripted(script))

    assert result.gaps == 2
    gaps = sync.gaps(SOURCE)
    assert [gap["ref"] for gap in gaps] == ["att-1", "att-2"]
    assert gaps[0]["reason"] == "unsupported type"


def test_the_same_gap_reported_twice_is_one_row_that_ages(store, sync, runtime):
    runtime.run(Scripted([notes(1, next_cursor="0",
                                skipped=[Skipped("att-1", "unsupported type")])]))
    before = sync.gaps(SOURCE)[0]

    again = runtime.run(Scripted([notes(1, start=1, next_cursor=None,
                                        skipped=[Skipped("att-1", "still unsupported")])]))

    gaps = sync.gaps(SOURCE)
    assert len(gaps) == 1
    assert gaps[0]["reason"] == "still unsupported"
    assert gaps[0]["first_seen_at"] == before["first_seen_at"]
    assert again.gaps == 1


def test_a_gap_closes_when_the_thing_arrives_and_reopens_if_it_goes_missing(
        store, sync, runtime):
    runtime.run(Scripted([notes(1, next_cursor="0", skipped=[Skipped("att-1", "too big")])]))
    assert [gap["ref"] for gap in sync.gaps(SOURCE)] == ["att-1"]

    runtime.run(Scripted([notes(1, prefix="att", start=1, next_cursor=None)]))

    assert sync.gaps(SOURCE) == []
    history = sync.gaps(SOURCE, include_cleared=True)
    assert len(history) == 1 and history[0]["cleared_at"]

    runtime.run(Scripted([notes(1, start=9, next_cursor=None,
                                skipped=[Skipped("att-1", "still too big")])]))
    assert [gap["ref"] for gap in sync.gaps(SOURCE)] == ["att-1"]


def test_a_gap_on_the_container_closes_when_a_part_of_it_arrives(store, sync, runtime):
    """One event reported as a whole, later delivered as its two sides.

    Matching only the bare id would leave this debt open forever: the container is never handed
    over as one envelope, because it was never one thing to the reader of the source. Every
    adapter that names a part `whole#part` — a turn's two speakers, a row of a file, a message
    of a session — would otherwise report a gap it had already filled.
    """
    runtime.run(Scripted([notes(1, next_cursor="0",
                                skipped=[Skipped("att-1", "unsupported shape")])]))
    assert [gap["ref"] for gap in sync.gaps(SOURCE)] == ["att-1"]

    runtime.run(Scripted([Page(envelopes=(
        envelope(source=SOURCE, source_id="att-1#user", text="Asked.", observed_at=AT),
        envelope(source=SOURCE, source_id="att-1#assistant", text="Answered.",
                 observed_at=AT)), next_cursor=None)]))

    assert sync.gaps(SOURCE) == []
    history = sync.gaps(SOURCE, include_cleared=True)
    assert len(history) == 1 and history[0]["cleared_at"]


def test_a_reconfigured_connector_starts_a_clean_gap_ledger(store, sync, runtime):
    runtime.run(Scripted([notes(1, next_cursor=None, skipped=[Skipped("att-1", "too big")])]))
    sync.reconfigure(SOURCE, policy_version="private-api", actor="owner", reason="remapped")

    assert sync.gaps(SOURCE) == [], "the new credentials are not owed the old failures"
    history = sync.gaps(SOURCE, generation=1, include_cleared=True)
    assert len(history) == 1 and history[0]["reason"] == "too big"


def test_a_gap_without_a_reference_or_a_reason_is_not_a_gap(store, sync, runtime):
    with pytest.raises(EvidenceError, match="reference"):
        sync.publish(sync.acquire(SOURCE, holder="worker-a"), "page-1", [],
                     skipped=[Skipped("", "no reference")])
    with pytest.raises(EvidenceError, match="reason"):
        sync.publish(sync.acquire(SOURCE, holder="worker-a"), "page-2", [],
                     skipped=[Skipped("att-1", "   ")])


def test_the_gap_list_is_bounded_and_open_first(store, sync, runtime):
    script = [notes(1, next_cursor=None,
                    skipped=[Skipped(f"att-{index}", "unsupported") for index in range(5)])]
    runtime.run(Scripted(script))

    assert len(sync.gaps(SOURCE, limit=2)) == 2
    with pytest.raises(EvidenceError, match="gap list"):
        sync.gaps(SOURCE, limit=0)


# -- coverage and freshness --------------------------------------------------

def test_reaching_the_end_of_a_source_is_what_current_means(store, sync, runtime):
    runtime.run(Scripted([notes(1, next_cursor="0")]), max_pages=1)
    assert sync.state(SOURCE)["coverage_state"] == "partial"

    runtime.run(Scripted([notes(1, start=1, next_cursor=None)]))
    assert sync.state(SOURCE)["coverage_state"] == "current"


def test_a_source_that_cannot_be_reached_says_so_and_stops(store, sync, runtime):
    adapter = Scripted([notes(1, next_cursor=None)], reachable=False)

    result = runtime.run(adapter)

    assert result.stopped == UNREACHABLE and result.coverage_state == "unreachable"
    assert result.note == "token revoked"
    assert adapter.reads == 0, "an unreachable source is not read anyway"


def test_a_read_that_fails_is_an_outage_rather_than_a_crash(store, sync, runtime):
    adapter = Scripted([notes(1, next_cursor="0")], fails={0})

    result = runtime.run(adapter)

    assert result.stopped == UNREACHABLE and "refused this read" in result.note
    assert sync.state(SOURCE)["coverage_state"] == "unreachable"
    assert sync.state(SOURCE)["lease"] is None


def test_a_check_that_raises_is_an_outage(store, sync, runtime):
    adapter = Scripted([notes(1)], reachable=EvidenceError("the account was revoked"))

    result = runtime.run(adapter)

    assert result.stopped == UNREACHABLE and "revoked" in result.note


def test_the_last_success_is_the_last_page_that_landed(store, sync, runtime):
    assert sync.state(SOURCE)["last_success_at"] is None
    runtime.run(Scripted([notes(1, next_cursor="0")], fails={0}))
    assert sync.state(SOURCE)["last_success_at"] is None, "a read that failed landed nothing"

    runtime.run(Scripted([notes(1, start=1, next_cursor="0")]), max_pages=1)
    landed = sync.state(SOURCE)["last_success_at"]
    assert landed
    runtime.run(Scripted([notes(1, start=2, next_cursor=None)]))
    assert sync.state(SOURCE)["last_success_at"] > landed, "it is the last *success*, not a date"


# -- pause semantics ---------------------------------------------------------

def test_a_paused_source_is_not_read_at_all(store, sync, runtime):
    sync.pause(SOURCE, actor="owner", reason="account review", policy_version="local-only")
    adapter = Scripted([notes(3, next_cursor=None)])

    result = runtime.run(adapter)

    assert result.stopped == PAUSED and result.records == 0
    assert adapter.reads == 0 and sync.state(SOURCE)["lease"] is None


def test_a_capture_only_pause_stops_the_fetch_and_nothing_else(store, sync, runtime):
    sync.pause_capture(SOURCE, actor="owner", reason="quota", policy_version="local-only")
    assert sync.paused_stages(SOURCE) == ["capture"]

    result = runtime.run(Scripted([notes(1, next_cursor=None)]))

    assert result.stopped == PAUSED
    sync.resume(SOURCE, actor="owner", reason="quota restored", policy_version="local-only")
    assert runtime.run(Scripted([notes(1, start=1, next_cursor=None)])).stopped == COMPLETE


def test_pausing_the_downstream_stages_still_captures(store, sync, runtime):
    sync.pause(SOURCE, actor="owner", reason="too much compute", policy_version="local-only",
               stages=DOWNSTREAM_STAGES)

    result = runtime.run(Scripted([notes(2, next_cursor=None)]))

    assert result.stopped == COMPLETE and result.records == 2


# -- the journal downstream --------------------------------------------------

def test_every_committed_record_is_journaled_once_under_its_generation(store, sync, runtime):
    runtime.run(Scripted([notes(2, next_cursor="0"), notes(1, start=2, next_cursor=None)]))

    rows = store.db.execute("SELECT record_id, generation, change FROM change_journal").fetchall()
    assert len(rows) == 3 == journaled(store)
    assert {row["generation"] for row in rows} == {1}
    assert {row["change"] for row in rows} == {"add"}


def test_a_replayed_pass_adds_nothing_to_the_journal(store, sync, runtime):
    script = [notes(2, next_cursor=None)]
    runtime.run(Scripted(script))
    before = journaled(store)

    runtime.run(Scripted(script))

    assert journaled(store) == before


def test_a_downstream_consumer_reading_on_never_holds_the_cursor_back(store, sync, runtime):
    runtime.run(Scripted([notes(1, next_cursor="0")]))
    sync.advance(consumer="projection", seq=journaled(store))

    result = runtime.run(Scripted([notes(1, start=1, next_cursor=None)]))

    assert result.stopped == COMPLETE and sync.state(SOURCE)["coverage_state"] == "current"
    # The consumer stayed exactly where it left itself, and the new change is
    # waiting for it rather than having been skipped.
    assert sync.checkpoint(consumer="projection") == 1
    assert [row["seq"] for row in sync.changes(consumer="projection")] == [2]


def test_a_run_describes_itself_without_the_source_being_reread(store, sync, runtime):
    result = runtime.run(Scripted([notes(1, next_cursor=None)]))

    assert isinstance(result, Run)
    assert result.as_dict()["source"] == SOURCE
    assert set(result.as_dict()) == {"source", "generation", "pages", "records", "repeats",
                                     "gaps", "cursor", "coverage_state", "stopped", "note"}
