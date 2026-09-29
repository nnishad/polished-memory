"""C2's consumer half: a spool the plugin wrote must become evidence on a timer.

The plugin's durability and the connector's cursor are two different contracts, and the bug
this file exists for was the gap between them: every turn was appended correctly, in a file
that nothing in the installation ever opened. So these tests hold both ends — the drain's
position rules, and one maintenance section that turns pending rows into records exactly once.

The spools here are written by the plugin's own `CaptureSpool`, not by a copy of its schema: a
reader that invents a state word, or assumes a column that is not there, has to be caught by the
component that owns the file rather than by two halves of this file agreeing with each other.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

from hermes_memory.processing.maintenance import Maintenance
from hermes_memory.sources.base import CursorExpired
from hermes_memory.sources.capture_spool import (
    SPOOL_SOURCE, SpoolError, spool_backlog, spool_beside, spool_drain)
from hermes_memory.sources.sdk import HermesEvents
from hermes_memory.sources.sync import SyncController
from plugin_loader import PLUGIN

# Spelled out here rather than imported: this is the plugin's layout, and the reader has to
# derive the same path without being told it.
SPOOL_RELATIVE = Path("hermes-memory") / "capture-spool.db"
AT = "2026-09-28T09:00:00+00:00"


def plugin_spool(path: Path):
    """The plugin's own durable spool, loaded from the integration directory."""
    if "hm_capture_spool" not in sys.modules:
        spec = importlib.util.spec_from_file_location("hm_capture_spool",
                                                     PLUGIN / "spool.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules["hm_capture_spool"] = module
        spec.loader.exec_module(module)
    return sys.modules["hm_capture_spool"].CaptureSpool(path)


def write_spool(path: Path, *events: dict) -> Path:
    """Append through the plugin, because that is who writes this file."""
    spool = plugin_spool(path)
    for event in events:
        spool.append(event_id=event["event_id"],
                    session_id=event.get("session_id", "sess-1"),
                    payload=event["payload"], created_at=event["created_at"])
    spool.close()
    return path


def turn(event_id: str, *, user: str, assistant: str, at: str = AT, **kwargs) -> dict:
    return {"event_id": event_id, "created_at": at,
            "payload": {"kind": "conversation_turn", "user": user, "assistant": assistant,
                        **kwargs}}


def spool_for(store) -> Path:
    return spool_beside(store.path)


# -- the drain ---------------------------------------------------------------

def test_the_reader_derives_the_same_path_the_plugin_writes_to(tmp_path):
    assert spool_beside(tmp_path / "canonical.db") == tmp_path / SPOOL_RELATIVE


def test_a_turn_written_on_another_thread_reaches_the_same_spool(tmp_path):
    """Hermes calls `sync_turn` on whichever worker it likes, and the spool is the durability.

    Measured on the live installation: one connection cached on the spool object made every
    turn after the first one fail with "SQLite objects created in a thread can only be used in
    that same thread", so the boundary that is supposed to be the proof a chat answer was
    kept was not keeping anything, once per message.
    """
    import threading

    path = tmp_path / SPOOL_RELATIVE
    spool = plugin_spool(path)
    assert spool.append(event_id="e1", session_id="s", payload={"kind": "conversation_turn"},
                        created_at=AT) is True

    def later():
        spool.append(event_id="e2", session_id="s", payload={"kind": "conversation_turn"},
                     created_at=AT)

    worker = threading.Thread(target=later)
    worker.start()
    worker.join(30)
    assert not worker.is_alive(), "the second thread raised instead of writing"
    assert [row["event_id"] for row in spool.pending()] == ["e1", "e2"]


def test_a_spool_is_read_in_its_own_insertion_order(tmp_path):
    path = write_spool(tmp_path / SPOOL_RELATIVE,
                        turn("e1", user="first", assistant="a", at="2026-09-26T09:00:00+00:00"),
                        turn("e2", user="second", assistant="b", at="2026-09-20T09:00:00+00:00"))
    events = spool_drain(path)()
    assert [item["event_id"] for item in events] == ["e1", "e2"], \
        "the spool's rowid order is the stream's order; ids are the host's strings"
    assert events[0]["payload"]["user"] == "first"


def test_a_position_advances_the_stream_and_the_next_call_picks_up_after_it(tmp_path):
    path = write_spool(tmp_path / SPOOL_RELATIVE,
                       turn("e1", user="a", assistant="b"), turn("e2", user="c", assistant="d"),
                       turn("e3", user="e", assistant="f"))
    drain = spool_drain(path, limit_default=2)
    first = [item["event_id"] for item in drain(None, 2)]
    second = [item["event_id"] for item in drain(first[-1], 2)]
    assert (first, second) == (["e1", "e2"], ["e3"])


def test_rows_behind_a_moved_cursor_are_settled_in_the_spool(tmp_path):
    path = write_spool(tmp_path / SPOOL_RELATIVE, turn("e1", user="a", assistant="b"),
                       turn("e2", user="c", assistant="d"))
    drain = spool_drain(path)
    drain(None, 1)
    drain("e1", 1)
    states = dict(sqlite3.connect(path).execute("SELECT event_id, state FROM spool"))
    assert states == {"e1": "settled", "e2": "pending"}, \
        "a row is retired only once a later call has passed it"


def test_a_retired_row_is_reclaimed_by_the_plugin_that_owns_the_file(tmp_path):
    """The state word has to be the plugin's, or nothing ever deletes these rows.

    `CaptureSpool.forget_settled_before` is the spool's only compaction, and it names one
    state. A reader that retired rows under a word of its own would leave them in the file
    forever and report a state no component of the installation can name.
    """
    path = write_spool(tmp_path / SPOOL_RELATIVE, turn("e1", user="a", assistant="b"),
                       turn("e2", user="c", assistant="d"))
    drain = spool_drain(path)
    drain(None, 1)
    drain("e1", 1)
    spool = plugin_spool(path)
    assert spool.counts() == {"settled": 1, "pending": 1}
    assert spool.forget_settled_before("2099-01-01T00:00:00+00:00") == 1, \
        "the plugin reclaims exactly what this reader retired"
    spool.close()
    assert spool_backlog(path)["pending"] == 1


def test_a_position_the_spool_no_longer_holds_is_said_out_loud(tmp_path):
    path = write_spool(tmp_path / SPOOL_RELATIVE, turn("e1", user="a", assistant="b"))
    with pytest.raises(CursorExpired):
        spool_drain(path)("event-from-a-compacted-spool", 10)


def test_a_read_with_no_position_starts_at_the_beginning_of_the_file(tmp_path):
    """A cleared cursor means "do not skip anything", which is what it is for.

    `reconfigure` clears a cursor precisely because the old position no longer describes the
    same stream, and a reader that answered that read with only the un-retired tail would
    silently skip every row the previous generation had already retired. Re-offering them is
    cheap: the store keys on the event id, so the replay repeats rather than duplicates.
    """
    path = write_spool(tmp_path / SPOOL_RELATIVE,
                       turn("e1", user="a", assistant="b"),
                       turn("e2", user="c", assistant="d"),
                       turn("e3", user="e", assistant="f"))
    drain = spool_drain(path)
    assert [item["event_id"] for item in drain(None, 2)] == ["e1", "e2"]
    assert [item["event_id"] for item in drain("e2", 2)] == ["e3"]
    assert [item["event_id"] for item in drain(None, 5)] == ["e1", "e2", "e3"], \
        "an unpositioned read is the whole spool, retired rows and all"


def test_a_missing_spool_is_an_absence_and_never_a_file_we_created(tmp_path):
    absent = tmp_path / SPOOL_RELATIVE
    with pytest.raises(SpoolError, match="no capture spool"):
        spool_drain(absent)()
    assert not absent.exists(), "creating it would report a drained stream that never existed"
    assert spool_backlog(absent) == {"spool": str(absent), "present": False, "pending": 0,
                                     "settled": 0, "states": {}, "oldest_at": None}


def test_a_file_that_is_not_a_spool_is_refused_rather_than_read_as_one(tmp_path):
    other = tmp_path / "not-a-spool.db"
    sqlite3.connect(other).execute("CREATE TABLE something_else(id TEXT)").close()
    with pytest.raises(SpoolError, match="no 'spool' table"):
        spool_drain(other)()


def test_a_spool_missing_a_column_the_reader_needs_is_named_rather_than_queried(tmp_path):
    """A spool from a plugin this core has never met is not an empty stream.

    Half the layout missing would otherwise read as "nothing captured", which is the exact
    answer the reader is supposed to be able to distinguish from a healthy one.
    """
    path = tmp_path / SPOOL_RELATIVE
    path.parent.mkdir(parents=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE spool(event_id TEXT PRIMARY KEY, payload TEXT, state TEXT)")
    db.execute("INSERT INTO spool VALUES('e1','{}','pending')")
    db.commit()
    db.close()
    with pytest.raises(SpoolError, match="session_id, created_at"):
        spool_drain(path)()
    with pytest.raises(SpoolError, match="session_id, created_at"):
        spool_backlog(path)


def test_undecodable_payload_survives_as_an_event_rather_than_a_crash(tmp_path):
    path = tmp_path / SPOOL_RELATIVE
    path.parent.mkdir(parents=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE spool(event_id TEXT PRIMARY KEY, session_id TEXT, payload TEXT,"
               " created_at TEXT, state TEXT, attempts INTEGER, last_error TEXT)")
    db.execute("INSERT INTO spool VALUES('e1','s','{not json','2026-09-28T09:00:00+00:00',"
               "'pending',0,NULL)")
    db.commit()
    db.close()
    events = spool_drain(path)()
    assert events[0]["payload"]["kind"] == "undecodable"


# -- the section -------------------------------------------------------------

def test_a_turn_in_the_spool_becomes_evidence_on_the_next_pass(store):
    write_spool(spool_for(store), turn("e1", user="Weight is 77.1kg", assistant="noted"))
    report = Maintenance(store, owner_principal="judge").pass_now(sections=("capture",))

    assert report["capture"]["records"] == 2, "one turn is two things that were said"
    assert report["capture"]["pending"] == 0
    rows = [dict(row) for row in store.db.execute(
        "SELECT source, kind, text FROM records ORDER BY rowid")]
    assert {row["source"] for row in rows} == {"hermes"}
    assert [row["text"] for row in rows] == ["Weight is 77.1kg", "noted"]


def test_a_second_pass_adds_nothing_because_the_cursor_is_the_whole_point(store):
    write_spool(spool_for(store), turn("e1", user="a", assistant="b"))
    keeper = Maintenance(store, owner_principal="judge")
    first = keeper.pass_now(sections=("capture",))["capture"]
    second = keeper.pass_now(sections=("capture",))["capture"]
    assert first["records"] == 2
    assert (second["records"], second["pending"]) == (0, 0)
    assert second["stopped"] == "complete", \
        "an exhausted stream is a clean stop; a drained spool re-offering its own tail is not"
    assert store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 2, \
        "the pass that found nothing must not have written it twice"


def test_the_capture_source_is_registered_under_a_policy_it_cannot_guess(store):
    write_spool(spool_for(store), turn("e1", user="a", assistant="b"))
    Maintenance(store, owner_principal="judge").pass_now(sections=("capture",))
    state = SyncController(store).state(SPOOL_SOURCE)
    assert state["policy_version"] == "local-only"
    assert state["coverage_state"] == "current"


def test_the_name_the_events_are_filed_under_is_the_name_the_adapter_declares():
    """The section registers one connector; the adapter decides what it is called."""
    assert HermesEvents(lambda after, limit: []).source == SPOOL_SOURCE


def test_the_pass_that_drains_the_spool_is_the_pass_that_makes_capture_honest(store):
    """Consumer and reading have to agree, or the second is only a second opinion."""
    from hermes_memory.operations.status import DEGRADED, OPERATIONAL, StatusReporter

    write_spool(spool_for(store), turn("e1", user="a", assistant="b"))
    reporter = StatusReporter(store)
    assert reporter.capture().state == DEGRADED, \
        "a spool nobody has read is not an unconfigured stage"
    Maintenance(store, owner_principal="judge").pass_now(sections=("capture",))
    assert reporter.capture().state == OPERATIONAL


def test_no_spool_is_reported_as_no_spool_rather_than_a_healthy_empty_drain(store):
    report = Maintenance(store, owner_principal="judge").pass_now(
        sections=("capture",))["capture"]
    assert report["present"] is False and report["records"] == 0
    assert "nothing to drain" in report["note"]


def test_the_backlog_a_pass_leaves_behind_is_the_number_the_doctor_needs(store):
    """A spool with rows nobody consumed is the failure this section exists to prevent."""
    path = spool_for(store)
    write_spool(path, *(turn(f"e{i}", user="u", assistant="a") for i in range(3)))
    before = spool_backlog(path)
    assert (before["present"], before["pending"]) == (True, 3)
    assert before["settled"] == 0
    assert before["oldest_at"] == AT, \
        "the age of the wait is the number an operator acts on"
    report = Maintenance(store, owner_principal="judge").pass_now(sections=("capture",))
    after = spool_backlog(path)
    assert report["capture"]["pending"] == after["pending"] == 0
    assert after["settled"] == 3, "a drained row says so in the spool too"
    assert report["capture"]["pending_before"] == 3, \
        "the pass says what it found as well as what it left"
