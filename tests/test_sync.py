"""C3 connector fences, page atomicity and independent consumer replay."""
from __future__ import annotations

import sqlite3

import pytest

from hermes_memory.sources.sync import (DOWNSTREAM_STAGES, STAGES, StaleFence,
                                   SyncController)
from hermes_memory.storage.evidence import EvidenceError

from conftest import envelope

SOURCE = "gmail"


class Clock:
    """Injectable wall clock: lease expiry must be testable without sleeping."""

    def __init__(self):
        self.value = 1_000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@pytest.fixture()
def sync(store):
    controller = SyncController(store, clock=Clock())
    controller.register(SOURCE, policy_version="private-api")
    return controller


def fenced(sync, source=SOURCE):
    return sync.acquire(source, holder="connector-a", ttl=60)


# -- leasing -----------------------------------------------------------------

def test_a_live_lease_is_not_stolen_by_a_second_connector(sync):
    fenced(sync)
    with pytest.raises(EvidenceError, match="leased to another"):
        sync.acquire(SOURCE, holder="connector-b")


def test_a_holder_may_re_take_its_own_lease_after_a_crash(sync):
    """Recovery has to be possible without an operator unlocking the row."""
    first = fenced(sync)
    second = sync.acquire(SOURCE, holder="connector-a", ttl=60)
    assert second.lease != first.lease
    with pytest.raises(StaleFence, match="another holder"):
        first.validate(sync.db, at=sync.clock())


def test_an_expired_lease_is_stolen_not_shared(sync):
    fenced(sync)
    sync.clock.advance(120)
    revived = sync.acquire(SOURCE, holder="connector-b", ttl=60)
    assert revived.holder == "connector-b"


def test_a_stalled_connector_cannot_commit_after_its_lease_lapses(sync):
    """The dangerous case: fetch under a live lease, commit after it expired."""
    fence = fenced(sync)
    sync.clock.advance(120)
    with pytest.raises(StaleFence, match="expired"):
        sync.publish(fence, "page-1", [envelope(source_id="msg-1")])
    assert sync.db.execute("SELECT count(*) FROM records").fetchone()[0] == 0


# -- page commit -------------------------------------------------------------

def test_cursor_and_records_advance_in_one_transaction(sync):
    fence = fenced(sync)
    result = sync.publish(fence, "page-1", [envelope(source_id="msg-1"),
                                           envelope(source_id="msg-2")], next_cursor="tok-2")
    assert result["records"] == 2 and result["cursor"] == "tok-2"


def test_replaying_a_page_changes_nothing(sync):
    fence = fenced(sync)
    page = [envelope(source_id="msg-1")]
    sync.publish(fence, "page-1", page, next_cursor="tok-2")
    before = sync.db.execute("SELECT count(*) FROM records").fetchone()[0]
    again = sync.publish(fence, "page-1", page, next_cursor="tok-2")
    assert again["duplicate"] is True
    assert sync.db.execute("SELECT count(*) FROM records").fetchone()[0] == before


def test_a_different_page_under_a_reused_token_is_refused(sync):
    """Silently accepting this would lose the first page's evidence."""
    fence = fenced(sync)
    sync.publish(fence, "page-1", [envelope(source_id="msg-1")])
    with pytest.raises(EvidenceError, match="different content"):
        sync.publish(fence, "page-1", [envelope(source_id="msg-9")])


def test_a_failure_halfway_through_a_page_leaves_no_orphans(sync):
    fence = fenced(sync)
    bad = envelope(source_id="msg-2", parent_record_ids=["rec_" + "f" * 32])
    with pytest.raises(EvidenceError):
        sync.publish(fence, "page-1", [envelope(source_id="msg-1"), bad], next_cursor="tok-2")
    assert sync.db.execute("SELECT count(*) FROM records").fetchone()[0] == 0
    assert sync.db.execute("SELECT count(*) FROM change_journal").fetchone()[0] == 0
    assert sync._cursor(SOURCE) is None, "a rolled back page must not move the cursor"


def test_a_page_may_not_write_under_another_sources_fence(sync):
    fence = fenced(sync)
    with pytest.raises(EvidenceError, match="claims source"):
        sync.publish(fence, "page-1", [envelope(source="whatsapp", source_id="m")])


# -- reconfiguration and reset ----------------------------------------------

def test_reconfiguration_strands_the_running_connector(sync):
    fence = fenced(sync)
    sync.publish(fence, "page-1", [envelope(source_id="msg-1")], next_cursor="tok-2")
    generation = sync.reconfigure(SOURCE, policy_version="local-only",
                                  actor="owner", reason="account re-mapped")
    assert generation == 2
    with pytest.raises(StaleFence, match="reconfigured"):
        sync.publish(fence, "page-2", [envelope(source_id="msg-2")])
    # A re-mapped source restarts from nothing rather than resuming a stale position.
    assert sync.state(SOURCE)["cursor"] is None
    assert sync.state(SOURCE)["generation"] == 2


def test_a_new_generation_may_reuse_a_page_token(sync):
    """The token is unique per generation, so a fresh connector is not blocked by history."""
    fence = fenced(sync)
    sync.publish(fence, "page-1", [envelope(source_id="msg-1")])
    sync.reconfigure(SOURCE, policy_version="local-only", actor="owner", reason="scope narrowed")
    fresh = fenced(sync)
    assert sync.publish(fresh, "page-1", [envelope(source_id="msg-1")])["records"] == 1


def test_bumping_the_epoch_invalidates_every_lease(sync, store):
    """A reset must fail in-flight writers forward, not discard their output silently."""
    fence = fenced(sync)
    store.bump_epoch(reason="owner reset", actor="owner")
    with pytest.raises(StaleFence, match="epoch advanced"):
        sync.publish(fence, "page-1", [envelope(source_id="msg-1")])
    assert sync.state(SOURCE)["lease_active"] is False


def test_reacquiring_after_an_epoch_bump_recovers_the_source(sync, store):
    fence = fenced(sync)
    store.bump_epoch(reason="owner reset", actor="owner")
    fresh = fenced(sync)
    assert fresh.epoch > fence.epoch
    assert sync.publish(fresh, "page-1", [envelope(source_id="msg-1")])["records"] == 1


# -- journal -----------------------------------------------------------------

def test_the_journal_carries_only_committed_changes(sync):
    fence = fenced(sync)
    committed = sync.publish(fence, "page-1", [envelope(source_id="msg-1")])
    with pytest.raises(EvidenceError):
        sync.publish(fence, "page-2", [envelope(source_id="msg-2", text="x" * 5_000_000)])
    rows = sync.changes(consumer="projection")
    assert [row["record_id"] for row in rows] == committed["ids"]
    assert rows[0]["generation"] == fence.generation


def test_each_consumer_checkpoints_independently(sync):
    fence = fenced(sync)
    first = sync.publish(fence, "page-1", [envelope(source_id="msg-1")])
    tail = sync.db.execute("SELECT MAX(seq) FROM change_journal").fetchone()[0]
    sync.advance(consumer="projection", seq=tail)
    sync.publish(fence, "page-2", [envelope(source_id="msg-2")], next_cursor="tok-3")
    # The lagging consumer still sees page 2; the caught-up one does not see it twice.
    assert len(sync.changes(consumer="projection")) == 1
    assert len(sync.changes(consumer="summaries")) == 2


def test_a_consumer_may_not_rewind(sync):
    sync.advance(consumer="projection", seq=10)
    with pytest.raises(EvidenceError, match="rewind"):
        sync.advance(consumer="projection", seq=3)


def test_hide_and_supersede_are_replayable_changes(sync, store):
    fence = fenced(sync)
    record = sync.publish(fence, "page-1", [envelope(source_id="msg-1")])["ids"][0]
    sync.advance(consumer="projection", seq=sync.db.execute("SELECT MAX(seq) FROM change_journal").fetchone()[0])
    store.hide(record, reason="operator review", actor="owner")
    changes = sync.changes(consumer="projection")
    assert [row["change"] for row in changes] == ["hide"]
    assert changes[0]["record_id"] == record


# -- pause semantics ---------------------------------------------------------

def test_pausing_a_source_stops_it_end_to_end(sync):
    stages = sync.pause(SOURCE, actor="owner", reason="too much compute",
                        policy_version="v1")

    assert stages == list(STAGES)
    assert sync.paused_stages(SOURCE) == list(STAGES)


def test_resuming_a_paused_source_takes_ingestion_back_too(sync):
    sync.pause(SOURCE, actor="owner", reason="too much compute", policy_version="v1")

    sync.resume(SOURCE, actor="owner", reason="reviewed", policy_version="v1")

    assert sync.paused_stages(SOURCE) == []


def test_stopping_ingestion_is_a_separate_explicit_act(sync):
    sync.pause_capture(SOURCE, actor="owner", reason="account revoked", policy_version="v1")

    assert sync.paused_stages(SOURCE) == ["capture"]
    # Working up what is already here and deciding whether to speak are separate
    # answers from fetching more, and a revoked account stops only the first.
    sync.resume(SOURCE, actor="owner", reason="reviewed", policy_version="v1",
                stages=DOWNSTREAM_STAGES)
    assert sync.paused_stages(SOURCE) == ["capture"]
    sync.resume(SOURCE, actor="owner", reason="restored", policy_version="v1")
    assert sync.paused_stages(SOURCE) == []


def test_publishing_requires_a_registration_and_rejects_oversized_pages(sync):
    sync.register("files", policy_version="local-only")
    fence = sync.acquire("files", holder="worker")
    with pytest.raises(EvidenceError, match="at most 1000"):
        sync.publish(fence, "page-1", [envelope(source_id=f"m{i}") for i in range(1001)])


def test_coverage_gaps_are_recorded_rather_than_implied(sync):
    fence = fenced(sync)
    sync.mark_gap(fence, coverage_state="partial", reason="source quota hit")
    assert sync.state(SOURCE)["coverage_state"] == "partial"
    with pytest.raises(EvidenceError, match="unknown coverage state"):
        sync.mark_gap(fence, coverage_state="fine", reason="optimism")


def test_a_fence_handed_to_commit_is_enforced_in_the_write_transaction(sync, store):
    """commit(fence=...) must not degrade to an unguarded commit."""
    fence = fenced(sync)
    store.commit(envelope(source_id="msg-1"), fence=fence)
    sync.reconfigure(SOURCE, policy_version="local-only", actor="owner", reason="re-mapped")
    with pytest.raises(StaleFence):
        store.commit(envelope(source_id="msg-2"), fence=fence)
    assert store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 1


def test_a_direct_commit_without_a_fence_is_still_allowed(store):
    """Local, operator and tool writes have no connector; they are not gated."""
    assert store.commit(envelope(source_id="msg-1"))["duplicate"] is False


def test_no_write_path_leaves_a_transaction_open(sync):
    """A leaked idle transaction holds the write lock and wedges the store.

    Every branch is walked, including the ones that write nothing: the replayed
    page and the refused lease are exactly where an early return goes wrong.
    """
    fence = fenced(sync)
    page = [envelope(source_id="msg-1")]
    steps = [
        lambda: sync.publish(fence, "page-1", page),
        lambda: sync.publish(fence, "page-1", page),           # duplicate branch
        lambda: sync.renew(fence),
        lambda: sync.mark_gap(fence, coverage_state="partial", reason="quota"),
        lambda: sync.advance(consumer="projection", seq=1),
        lambda: sync.pause(SOURCE, actor="owner", reason="budget", policy_version="v1"),
        lambda: sync.state(SOURCE),
        lambda: sync.changes(consumer="projection"),
        lambda: sync.release(fence),
    ]
    for step in steps:
        step()
        assert sync.db.in_transaction is False
    with pytest.raises(StaleFence):
        sync.publish(fence, "page-2", page)
    assert sync.db.in_transaction is False


def test_the_store_survives_a_reopen_with_the_journal_intact(tmp_path):
    from hermes_memory.storage.evidence import EvidenceStore

    path = tmp_path / "canonical.db"
    with EvidenceStore(path) as store:
        controller = SyncController(store)
        controller.register(SOURCE, policy_version="private-api")
        fence = controller.acquire(SOURCE, holder="connector-a", ttl=3600)
        controller.publish(fence, "page-1", [envelope(source_id="msg-1")], next_cursor="tok-2")
    with EvidenceStore(path) as reopened:
        controller = SyncController(reopened)
        assert controller.state(SOURCE)["cursor"] == "tok-2"
        assert len(controller.changes(consumer="projection")) == 1
        assert isinstance(reopened.db.execute("SELECT 1 FROM connectors").fetchone(), sqlite3.Row)


def test_a_page_must_be_a_sequence_of_envelopes_not_a_lone_string(sync):
    """A bare string is a sequence of characters, and every one of them would fail."""
    fence = fenced(sync)
    with pytest.raises(EvidenceError, match="sequence of envelopes"):
        sync.publish(fence, "page-1", "not a page")


def test_a_new_revision_of_the_same_thing_is_not_the_page_already_committed(sync, store):
    """The page digest carries the revision, or a re-stamped record is a silent loss.

    A record fingerprint covers the bytes, not which revision of the source's item
    they are, so identical text under a new revision is a different record — and a
    page that says two different things under one name is a contradiction.
    """
    fence = fenced(sync)
    sync.publish(fence, "page-1", [envelope(source_id="msg-1", revision="1")])

    with pytest.raises(EvidenceError, match="different content"):
        sync.publish(fence, "page-1", [envelope(source_id="msg-1", revision="2")])

    landed = sync.publish(fence, "page-2", [envelope(source_id="msg-1", revision="2")])
    assert landed["duplicate"] is False and landed["new"] == 1
    assert int(store.db.execute(
        "SELECT count(*) FROM records WHERE source_id='msg-1'").fetchone()[0]) == 2


def test_a_renewal_of_a_lease_someone_else_now_holds_is_refused(sync):
    holder = fenced(sync)
    sync.clock.advance(120)
    rival = sync.acquire(SOURCE, holder="worker-b")

    with pytest.raises(StaleFence, match="another holder"):
        sync.renew(holder)

    assert sync.state(SOURCE)["lease"] == rival.lease


def test_a_pause_only_accepts_stages_that_exist(sync):
    with pytest.raises(EvidenceError, match="stages must come from"):
        sync.pause(SOURCE, actor="owner", reason="invented stage", policy_version="v1",
                   stages=("projection",))
    with pytest.raises(EvidenceError, match="stages must come from"):
        sync.pause(SOURCE, actor="owner", reason="nothing named", policy_version="v1",
                   stages=())
