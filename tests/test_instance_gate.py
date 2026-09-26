"""C12/C14: one admission ledger for the whole installation (§10.5).

The claim under test is narrow and load-bearing: two enrolled profiles sharing a GPU are
*one queue*, not two queues that each believe they are alone. Before the ledger moved out
of the profile archives, the single-slot index guaranteed one holder per profile, and an
installation serving two people could hold the same physical model twice while every
status report said otherwise. These tests are written to fail if that ever comes back.
"""
from __future__ import annotations

import sqlite3
import stat
from pathlib import Path

import pytest

from hermes_memory.config import load_settings, scoped_settings
from hermes_memory.operations.status import UNCONFIGURED, StatusReporter
from hermes_memory.processing.instance_gate import (GATE_FILENAME, GateError, GateStore,
                                                   gate_path, instance_gate, status_gate)
from hermes_memory.processing.resource_gate import ResourceGate
from hermes_memory.storage.evidence import EvidenceStore

OWNER = "jugaadu"


@pytest.fixture()
def installation(tmp_path, monkeypatch):
    home = tmp_path / "instance"
    (home / "data").mkdir(parents=True)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS=200000\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n", encoding="utf-8")
    settings = load_settings()
    with EvidenceStore(settings.db_path):
        pass
    return settings


def second_profile(settings, tmp_path):
    """Another enrolled person's configuration: a different archive, the same machine."""
    directory = tmp_path / "profiles" / "work" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    with EvidenceStore(directory / "canonical.db"):
        pass
    return scoped_settings(settings, profile="work", data_dir=directory,
                           bank_id="bank-work", credential_scope="profile-work")


# -- the file ------------------------------------------------------------------

def test_the_ledger_lives_beside_the_instance_and_nowhere_else(installation):
    assert gate_path(installation) == installation.home / GATE_FILENAME
    with instance_gate(installation):
        pass
    assert gate_path(installation).is_file()


def test_the_ledger_is_only_readable_by_its_owner(installation):
    with instance_gate(installation):
        pass
    assert stat.S_IMODE(gate_path(installation).stat().st_mode) == 0o600


def test_opening_the_gate_writes_nothing_to_any_archive(installation):
    with instance_gate(installation) as gate:
        assert gate.try_acquire(route="retain", holder="generation-1",
                                resource="local-gpu", priority=1) is not None
    with EvidenceStore(installation.db_path) as store:
        assert store.db.execute("SELECT count(*) FROM gate_reservations").fetchone()[0] \
            == 0, "admission through the instance ledger left no row in the archive"


def test_the_gate_ledger_holds_no_evidence_tables(installation):
    with GateStore(gate_path(installation)) as ledger:
        tables = {row[0] for row in ledger.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"gate_reservations", "gate_ledger", "budget_usage",
            "runtime_controls"} <= tables
    assert "records" not in tables and "assertions" not in tables, \
        "the admission ledger is not a second archive"


def test_a_ledger_from_a_newer_build_is_refused(installation):
    """Reading a newer admission ledger with an older build would miscount a queue."""
    with GateStore(gate_path(installation)) as ledger:
        ledger.db.execute("PRAGMA user_version=99")
    with pytest.raises(GateError, match="newer admission ledger"):
        GateStore(gate_path(installation))


# -- the one slot ---------------------------------------------------------------

def test_a_second_profile_cannot_take_a_slot_the_first_is_holding(installation, tmp_path):
    """The whole invariant, stated as the failure it prevents."""
    other = second_profile(installation, tmp_path)
    with instance_gate(installation) as julia, instance_gate(other) as work:
        assert julia.try_acquire(route="retain", holder="julia:generation-1",
                                 resource="local-gpu", priority=1) is not None
        assert work.try_acquire(route="retain", holder="work:generation-1",
                                resource="local-gpu", priority=0) is None, \
            "one GPU, one holder, whoever asked"


def test_a_different_resource_is_still_available(installation, tmp_path):
    other = second_profile(installation, tmp_path)
    with instance_gate(installation) as julia, instance_gate(other) as work:
        assert julia.try_acquire(route="retain", holder="julia:1", resource="remote-9b",
                                 priority=1)
        assert work.try_acquire(route="embed", holder="work:1", resource="local-gpu",
                                priority=0) is not None
        assert work.blocked_resources() == ["local-gpu", "remote-9b"], \
            "both slots are occupied, one by each profile"


def test_releasing_for_one_profile_admits_the_other(installation, tmp_path):
    other = second_profile(installation, tmp_path)
    with instance_gate(installation) as julia, instance_gate(other) as work:
        held = julia.try_acquire(route="retain", holder="julia:1", resource="local-gpu",
                                 priority=1)
        assert work.try_acquire(route="retain", holder="work:1", resource="local-gpu",
                                priority=0) is None
        julia.release(held, outcome="succeeded", tokens=40)
        assert work.try_acquire(route="retain", holder="work:1", resource="local-gpu",
                                priority=0) is not None


def test_an_unresolved_reservation_blocks_the_resource_for_everybody(installation,
                                                                     tmp_path):
    other = second_profile(installation, tmp_path)
    with instance_gate(installation) as julia, instance_gate(other) as work:
        held = julia.try_acquire(route="retain", holder="julia:1", resource="remote-9b",
                                 priority=1)
        julia.mark_uncertain(held, reason="the server stopped answering")
        assert work.try_acquire(route="retain", holder="work:1", resource="remote-9b",
                                priority=0) is None
        assert work.blocked_resources() == ["remote-9b"], \
            "an uncertain lease is not evidence that a model is free"


# -- the hold -------------------------------------------------------------------

def test_a_pause_by_one_profile_stops_the_models_for_all_of_them(installation, tmp_path):
    other = second_profile(installation, tmp_path)
    with instance_gate(installation) as julia:
        julia.pause(actor=OWNER, reason="overnight maintenance")
    with instance_gate(other) as work:
        assert work.paused is True
        with pytest.raises(Exception, match="paused"):
            work.try_acquire(route="retain", holder="work:1", resource="local-gpu",
                             priority=1)


def test_the_hold_outlives_the_process_that_set_it(installation):
    with instance_gate(installation) as gate:
        gate.pause(actor=OWNER, reason="model update")
    with instance_gate(installation) as reopened:
        assert reopened.paused is True
        reopened.resume(actor=OWNER, reason="the update finished")
    with instance_gate(installation) as after:
        assert after.paused is False


def test_the_hold_keeps_the_actor_who_set_it(installation):
    with instance_gate(installation) as gate:
        gate.pause(actor=OWNER, reason="model update")
    with GateStore(gate_path(installation)) as ledger:
        row = ledger.control("global", "inference")
    assert row["actor"] == OWNER and row["state"] == "paused"
    assert "model update" in row["reason"]


def test_an_unattributed_hold_is_refused(installation):
    with GateStore(gate_path(installation)) as ledger:
        with pytest.raises(GateError, match="attributed"):
            ledger.set_control("global", "inference", "paused", actor="  ",
                               reason="nobody", policy_version="v1")
        with pytest.raises(GateError, match="active"):
            ledger.set_control("global", "inference", "maybe", actor=OWNER,
                               reason="typo", policy_version="v1")


# -- lifetime and reading -------------------------------------------------------

def test_a_gate_never_closes_an_archive_it_was_only_handed(installation):
    with EvidenceStore(installation.db_path) as store:
        gate = ResourceGate(store)
        gate.close()
        assert store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 0, \
            "the store is still open, as its owner left it"


def test_a_gate_over_its_own_ledger_does_close_it(installation):
    gate = instance_gate(installation)
    gate.close()
    with pytest.raises(sqlite3.ProgrammingError):
        gate.store.db.execute("SELECT 1")


def test_status_reports_no_ledger_as_nothing_queued_rather_than_creating_one(
        installation):
    assert status_gate(installation) is None
    with ReadOnlyGate(installation) as reporter:
        stage = reporter.resource_gate()
    assert stage.state == UNCONFIGURED
    assert "nothing has been queued" in stage.detail
    assert not gate_path(installation).exists()


def test_a_reading_gate_cannot_change_the_ledger_it_reads(installation):
    """Read-only is the connection's promise, not the reader's good intention."""
    with instance_gate(installation) as gate:
        gate.try_acquire(route="retain", holder="generation-1", resource="local-gpu",
                         priority=1)
    reading = status_gate(installation)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reading.db.execute("DELETE FROM gate_reservations")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reading.store.set_control("global", "inference", "active", actor=OWNER,
                                      reason="a reading lifted the owner's hold",
                                      policy_version="v1")
    finally:
        reading.close()


def test_the_gate_refuses_a_missing_ledger_rather_than_making_one(installation):
    with pytest.raises(GateError, match="nothing has been queued"):
        GateStore(gate_path(installation), create=False)
    assert not gate_path(installation).exists()


def test_status_names_the_holder_it_found(installation):
    with instance_gate(installation) as gate:
        gate.try_acquire(route="retain", holder="generation-3", resource="local-gpu",
                         priority=1)
    with ReadOnlyGate(installation) as reporter:
        stage = reporter.resource_gate()
    assert [row["holder"] for row in stage.evidence["held"]] == ["generation-3"]
    assert stage.evidence["blocked"] == ["local-gpu"]


class ReadOnlyGate:
    """A status reporter over the installation's own ledgers, opened and closed here."""

    def __init__(self, settings):
        self.settings = settings
        self.reporter = None

    def __enter__(self):
        self.store = EvidenceStore(self.settings.db_path)
        self.reporter = StatusReporter(self.store, settings=self.settings)
        return self.reporter

    def __exit__(self, *exc_info):
        gate = self.reporter.gate
        if gate is not None:
            gate.close()
        self.store.close()
