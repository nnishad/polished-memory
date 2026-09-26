"""C14 upgrade plan (§10.6): a plan that reads the machine and touches nothing.

The claim these tests hold is a negative one, and it is the whole point of the module:
an operation called "upgrade" that performs an upgrade while answering "what would
change" is how a hand-edited unit and an unrehearsed schema migration get into a
running installation. So every test here asks either "did the disk stay as it was" or
"did it name the exact thing in its way".
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import namedtuple
from pathlib import Path

import pytest

from conftest import envelope
from hermes_memory.config import load_settings
from hermes_memory.install import inventory
from hermes_memory.install.profiles import ProfileRegistry
from hermes_memory.install.services import apply as write_units
from hermes_memory.install.services import plan as units_plan
from hermes_memory.install.upgrade import (BACKUP_AGE_DAYS, UpgradeError,
                                           profile_stores, upgrade_plan)
from hermes_memory.lifecycle.snapshots import Snapshots
from hermes_memory.processing.resource_gate import ResourceGate
from hermes_memory.storage.evidence import EvidenceStore

OWNER = "jugaadu"


def staged(home: Path, name: str = "release-0.2.0") -> Path:
    """A release tree as §10.3 ships it: an executable and a plugin artifact."""
    release = home / name
    (release / "bin").mkdir(parents=True, exist_ok=True)
    (release / "bin" / "hermes-memory").write_text("#!/bin/sh\n", encoding="utf-8")
    plugin = release / "integrations" / "hermes-memory"
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.py").write_text("NAME = 'hermes-memory'\n", encoding="utf-8")
    return release


@pytest.fixture()
def installation(tmp_path, monkeypatch):
    """An instance with one initialised store, a staged release, and no units written."""
    home = tmp_path / "instance"
    (home / "data").mkdir(parents=True)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    release = staged(tmp_path)
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n", encoding="utf-8")
    settings = load_settings()
    with EvidenceStore(settings.db_path):
        pass
    environ = {"HERMES_MEMORY_RELEASE": str(release),
               "HERMES_HOME": str(tmp_path / "homes" / "work"),
               "XDG_CONFIG_HOME": os.environ["XDG_CONFIG_HOME"]}
    (tmp_path / "homes" / "work").mkdir(parents=True)
    return settings, environ, release, tmp_path


def said(report) -> str:
    return " ".join(report["blocking"] + report["advisory"])


# -- the promise that it writes nothing ---------------------------------------

def test_the_plan_changes_no_single_file_on_disk(installation):
    """Read-only in the sense that can be tested: the archive's bytes do not move.

    A read-only connection to a WAL database is still allowed to leave an empty
    write-ahead log beside it — SQLite will not let a connection that cannot write
    clean one up — so the claim is about content, and about nothing being created
    except what the storage engine itself needs to read.
    """
    settings, environ, release, tmp_path = installation
    before = _contents(tmp_path)
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["blocking"] or report["advisory"], "it read something real"
    after = _contents(tmp_path)
    assert set(before) - set(after) == set(), "the plan deleted something"
    assert {key: value for key, value in after.items() if key in before} == before
    assert all(_is_sidecar(path) for path in set(after) - set(before))


def _contents(root: Path) -> dict[Path, str]:
    return {path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file() and not _is_sidecar(path)}


def _is_sidecar(path: Path) -> bool:
    return path.name.endswith(("-wal", "-shm", "-journal"))


def test_the_plan_names_the_release_it_refuses_to_guess_at(installation):
    settings, environ, _release, _tmp = installation
    report = upgrade_plan(settings, environ=environ)
    assert report["target_release"] is None
    assert "no target release was named" in said(report)


def test_a_release_argument_that_is_not_a_path_is_refused(installation):
    settings, environ, _release, _tmp = installation
    with pytest.raises(UpgradeError, match="path"):
        upgrade_plan(settings, release=17, environ=environ)


def test_the_plan_says_plainly_which_things_it_did_not_do(installation):
    settings, environ, release, _tmp = installation
    report = upgrade_plan(settings, release=release, environ=environ)
    joined = " ".join(report["not_performed"])
    assert "paused" in joined and "backup was taken" in joined and "migrated" in joined
    assert "this framework has no operation that performs them" in report["note"]


def test_every_line_of_that_list_denies_something(installation):
    """A list of what was not done that states one thing that was is worse than empty."""
    settings, environ, release, _tmp = installation
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["not_performed"]
    for line in report["not_performed"]:
        assert any(word in line.lower() for word in ("not ", "no ", "never", "still")), line


def test_advice_about_the_machine_does_not_invalidate_an_approval(installation,
                                                                  monkeypatch):
    """Two gigabytes more or less is not a different switch.

    The digest covers the files and stores a switch would act on. Folding the advisory
    lists in would make every approval expire on the next poll of a busy disk.
    """
    settings, environ, release, _tmp = installation
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(inventory.shutil, "disk_usage",
                        lambda path: Usage(50_000_000_000, 49_000_000_000, 1_000_000_000))
    first = upgrade_plan(settings, release=release, environ=environ)
    monkeypatch.setattr(inventory.shutil, "disk_usage",
                        lambda path: Usage(50_000_000_000, 10_000_000_000,
                                           40_000_000_000))
    second = upgrade_plan(settings, release=release, environ=environ)
    assert first["review_digest"] == second["review_digest"]
    assert any("side-by-side" in line for line in first["advisory"])
    assert not any("side-by-side" in line for line in second["advisory"])


# -- the target tree ----------------------------------------------------------

def test_a_tree_with_no_executable_would_stop_the_installation_coming_up(installation,
                                                                         tmp_path):
    settings, environ, _release, _tmp = installation
    empty = tmp_path / "half-a-release"
    empty.mkdir()
    report = upgrade_plan(settings, release=empty, environ=environ)
    assert str(empty / "bin" / "hermes-memory") in said(report)


def test_a_tree_that_is_not_a_directory_is_named_as_one(installation, tmp_path):
    settings, environ, _release, _tmp = installation
    stray = tmp_path / "not-a-tree"
    stray.write_text("a wheel is not a tree\n", encoding="utf-8")
    report = upgrade_plan(settings, release=stray, environ=environ)
    assert "is not a directory" in said(report)


def test_a_tree_that_omits_the_plugin_would_move_only_half_the_installation(installation,
                                                                           tmp_path):
    settings, environ, _release, _tmp = installation
    runtime_only = tmp_path / "runtime-only"
    (runtime_only / "bin").mkdir(parents=True)
    (runtime_only / "bin" / "hermes-memory").write_text("#!/bin/sh\n", encoding="utf-8")
    report = upgrade_plan(settings, release=runtime_only, environ=environ)
    assert "older contract" in said(report)
    assert report["target"]["runnable"] is False


def test_a_good_tree_is_reported_runnable_with_its_digest(installation, tmp_path):
    settings, environ, _release, _tmp = installation
    other = staged(tmp_path, "release-0.3.0")
    report = upgrade_plan(settings, release=other, environ=environ)
    assert report["target"]["runnable"] is True
    assert report["target"]["plugin_files"] == 1
    assert len(report["target"]["plugin_digest"]) == 64
    assert "no target release" not in said(report)


def test_naming_the_release_already_in_place_is_called_a_restart(installation):
    settings, environ, release, _tmp = installation
    report = upgrade_plan(settings, release=release, environ=environ)
    assert "wearing a better word" in said(report)
    assert report["switches"][0].endswith("the pointer would not move")


def test_the_plugin_digest_says_whether_the_host_registration_must_move(installation,
                                                                       tmp_path):
    """Identical plugin files mean the host's pin does not have to be repinned."""
    settings, environ, _release, _tmp = installation
    other = staged(tmp_path, "release-0.3.0")
    changed = other / "integrations" / "hermes-memory" / "plugin.py"
    changed.write_text("NAME = 'something-else'\n", encoding="utf-8")
    report = upgrade_plan(settings, release=other, environ=environ)
    assert report["revisions"]["plugin_files"] != report["revisions"]["target_plugin_files"]


# -- what the machine is doing ------------------------------------------------

def test_a_store_on_a_newer_schema_blocks_the_switch(installation, monkeypatch):
    settings, environ, release, _tmp = installation
    from hermes_memory.storage import migrations

    # Pretend this build is older than the one that wrote the store.
    monkeypatch.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS[:5])
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["schema"] == {"state": "newer", "now": 12, "head": 5,
                               "store": str(settings.db_path)}
    assert "older binary against a newer schema" in said(report)


def test_a_store_behind_head_is_said_but_not_blocked(installation, monkeypatch):
    settings, environ, release, _tmp = installation
    from hermes_memory.storage import migrations

    monkeypatch.setattr(migrations, "MIGRATIONS",
                        (*migrations.MIGRATIONS,
                         migrations.Migration("0013_probe", ("SELECT 1;",))))
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["schema"]["state"] == "behind"
    assert "rehearsal copy" in said(report)
    assert [line for line in report["blocking"] if "migration" in line] == []


def test_an_unreadable_store_stops_the_plan_rather_than_claiming_a_schema(installation,
                                                                         tmp_path):
    settings, environ, release, _tmp = installation
    (settings.data_dir / "canonical.db").write_text("not a database\n", encoding="utf-8")
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["schema"]["state"] == "unreadable"
    assert "could not be read" in said(report)


def test_a_store_that_does_not_exist_yet_is_reported_as_absent(installation):
    settings, environ, release, _tmp = installation
    (settings.data_dir / "canonical.db").unlink()
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["schema"]["state"] == "absent"
    assert "newer schema" not in said(report)


def test_a_held_gate_slot_blocks_the_switch_and_names_its_holder(installation):
    settings, environ, release, _tmp = installation
    with EvidenceStore(settings.db_path) as store:
        held = ResourceGate(store).try_acquire(route="retain", holder="generation-7",
                                               resource="local-gpu", priority=1)
    assert held is not None
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["in_flight"]["held"] == ["local-gpu for the default profile"]
    assert "hold a gate slot" in said(report)


def test_an_uncertain_lease_still_counts_as_work_the_machine_is_inside(installation):
    settings, environ, release, _tmp = installation
    with EvidenceStore(settings.db_path) as store:
        store.db.execute("INSERT INTO gate_reservations(id, resource, route, holder, "
                         "priority, state, acquired_at, lease_until) VALUES "
                         "('r1','remote-9b','consolidate','generation-7',1,'uncertain',"
                         "0,0)")
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["in_flight"]["held"] == ["remote-9b for the default profile"]


def test_the_plan_leaves_no_ledger_behind(installation):
    settings, environ, release, _tmp = installation
    upgrade_plan(settings, release=release, environ=environ)
    assert not (settings.home / "installation.db").exists(), \
        "reading the profile map is not a reason to create one"


def test_a_slot_held_in_a_second_profile_store_is_not_invisible(installation, tmp_path):
    """One installation, three people: the upgrade has to see all three stores."""
    settings, environ, release, _tmp = installation
    work = tmp_path / "homes" / "work"
    registry = ProfileRegistry.open(settings)
    try:
        proposal = registry.plan("work", work)
        registry.enroll("work", work, actor=OWNER, review_digest=proposal["review_digest"])
    finally:
        registry.db.close()
    with EvidenceStore(Path(proposal["data_dir"]) / "canonical.db") as store:
        assert ResourceGate(store).try_acquire(route="retain", holder="generation-8",
                                              resource="remote-9b", priority=1)
    report = upgrade_plan(settings, release=release, environ=environ)
    assert [name for name, _ in profile_stores(settings)] == [
        "work", f"the unenrolled store at {settings.db_path}"]
    assert report["in_flight"]["held"] == ["remote-9b for work"]
    assert report["in_flight"]["stores"] == 2


# -- backups ------------------------------------------------------------------

def test_no_backup_means_no_way_back(installation):
    settings, environ, release, _tmp = installation
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["backups"]["missing"] == ["the default profile"]
    assert "no backup to fall back to" in " ".join(report["blocking"])


def test_a_recent_backup_clears_the_block(installation):
    settings, environ, release, _tmp = installation
    with EvidenceStore(settings.db_path) as store:
        made = Snapshots(store, directory=settings.data_dir / "snapshots").create(
            reason="before the switch", actor=OWNER)
    report = upgrade_plan(settings, release=release, environ=environ, at=time.time())
    assert report["backups"]["missing"] == []
    assert report["backups"]["newest"]["the default profile"]["id"] == made["snapshot"].id
    assert report["backups"]["worst_age_days"] == 0
    assert "no backup to fall back to" not in " ".join(report["blocking"])


def test_a_stale_backup_is_said_loudly_without_stopping(installation):
    settings, environ, release, _tmp = installation
    with EvidenceStore(settings.db_path) as store:
        Snapshots(store, directory=settings.data_dir / "snapshots").create(
            reason="an age ago", actor=OWNER)
    future = time.time() + (BACKUP_AGE_DAYS + 20) * 86400
    report = upgrade_plan(settings, release=release, environ=environ, at=future)
    assert report["backups"]["worst_age_days"] > BACKUP_AGE_DAYS
    assert "day(s) old" in said(report)
    assert report["blocking"] == [line for line in report["blocking"] if "backup" not in line]


def test_the_profile_with_the_oldest_backup_is_the_one_named(installation, tmp_path):
    """The worst age is the worst, not the last one read.

    Two profiles, one fresh copy and one from before the year the owner cares about:
    reporting the good news as the installation's would be exactly the false assurance
    this check exists to refuse.
    """
    settings, environ, release, _tmp = installation
    with EvidenceStore(settings.db_path) as store:
        Snapshots(store, directory=settings.data_dir / "snapshots").create(
            reason="default kept", actor=OWNER)
    work = tmp_path / "homes" / "work"
    registry = ProfileRegistry.open(settings)
    proposal = registry.plan("work", work)
    registry.enroll("work", work, actor=OWNER, review_digest=proposal["review_digest"])
    registry.db.close()
    directory = Path(proposal["data_dir"]) / "snapshots"
    with EvidenceStore(Path(proposal["data_dir"]) / "canonical.db") as store:
        Snapshots(store, directory=directory).create(reason="work kept", actor=OWNER)
    manifest = next(path for path in directory.rglob("*.json"))
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    recorded["created_at"] = "2020-01-01T00:00:00+00:00"
    manifest.write_text(json.dumps(recorded), encoding="utf-8")

    report = upgrade_plan(settings, release=release, environ=environ, at=time.time())
    assert report["backups"]["missing"] == []
    assert report["backups"]["worst_profile"] == "work"
    assert report["backups"]["worst_age_days"] > BACKUP_AGE_DAYS
    assert "the newest backup for work is" in said(report)


def test_a_directory_of_nothing_but_old_files_is_not_a_backup(installation, tmp_path):
    """A snapshot is its manifest, so a directory of files is not a way back."""
    settings, environ, release, _tmp = installation
    directory = settings.data_dir / "snapshots"
    (directory / "20260101T000000000000Z-notes").mkdir(parents=True)
    (directory / "20260101T000000000000Z-notes" / "canonical.db").write_text(
        "a copy with nothing to identify it\n", encoding="utf-8")
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["backups"]["missing"] == ["the default profile"]


def test_a_second_profile_without_a_backup_blocks_even_when_the_first_has_one(installation,
                                                                             tmp_path):
    settings, environ, release, _tmp = installation
    with EvidenceStore(settings.db_path) as store:
        Snapshots(store, directory=settings.data_dir / "snapshots").create(
            reason="kept", actor=OWNER)
    work = tmp_path / "homes" / "work"
    registry = ProfileRegistry.open(settings)
    try:
        proposal = registry.plan("work", work)
        registry.enroll("work", work, actor=OWNER,
                        review_digest=proposal["review_digest"])
        Path(proposal["data_dir"]).mkdir(parents=True, exist_ok=True)
        with EvidenceStore(Path(proposal["data_dir"]) / "canonical.db"):
            pass
    finally:
        registry.db.close()
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["backups"]["missing"] == ["work"]
    assert report["backups"]["present"] == 1


# -- the units and the rest ---------------------------------------------------

def test_a_foreign_unit_blocks_the_switch_and_an_owned_one_does_not(installation):
    settings, environ, release, tmp_path = installation
    directory = Path(units_plan(settings, environ=environ)["unit_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "hermes-memory.service").write_text("[Unit]\n# somebody else's\n",
                                                     encoding="utf-8")
    report = upgrade_plan(settings, release=release, environ=environ)
    assert "hermes-memory.service" in " ".join(report["blocking"])
    assert [entry["state"] for entry in report["units"]][0] == "collision"
    (directory / "hermes-memory.service").unlink()
    assert "were not written by this installation" not in said(
        upgrade_plan(settings, release=release, environ=environ))


def test_writing_the_units_ourselves_changes_the_plan_we_would_approve(installation):
    settings, environ, release, _tmp = installation
    first = upgrade_plan(settings, release=release, environ=environ)
    proposal = units_plan(settings, environ=environ)
    write_units(settings, actor=OWNER, review=proposal["review_digest"], environ=environ)
    second = upgrade_plan(settings, release=release, environ=environ)
    assert first["review_digest"] != second["review_digest"]
    assert [entry["state"] for entry in second["units"]] == [
        "unchanged", "not-wanted", "not-wanted"], "only the runtime is wanted here"
    assert "a switch would rewrite" not in said(second)


def test_little_free_space_is_an_advisory_about_room_not_a_refusal(installation,
                                                                   monkeypatch):
    settings, environ, release, _tmp = installation
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(inventory.shutil, "disk_usage",
                        lambda path: Usage(50_000_000_000, 49_999_000_000, 1_000_000))
    report = upgrade_plan(settings, release=release, environ=environ)
    assert "side-by-side stage" in said(report)
    assert report["disk"]["free_bytes"] == 1_000_000
    assert "side-by-side" not in " ".join(report["blocking"])


def test_unreadable_free_space_admits_that_room_is_unproven(installation, monkeypatch):
    settings, environ, release, _tmp = installation
    def refuse(path):
        raise OSError("no such device")

    monkeypatch.setattr(inventory.shutil, "disk_usage", refuse)
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["disk"]["free_bytes"] is None
    assert "room for the second tree is unproven" in said(report)


def test_a_host_not_selecting_this_memory_is_said(installation, tmp_path):
    settings, environ, release, _tmp = installation
    host = Path(environ["HERMES_HOME"])
    assert "the host's memory provider reads" in said(
        upgrade_plan(settings, release=release, environ=environ))
    (host / "config.yaml").write_text("model:\n  provider: openai\n\nmemory:\n"
                                      "  provider: hermes-memory\n", encoding="utf-8")
    assert "the host's memory provider reads" not in said(
        upgrade_plan(settings, release=release, environ=environ))


def test_an_unreadable_store_leaves_the_in_flight_claim_falsifiable(installation, tmp_path):
    settings, environ, release, _tmp = installation
    work = tmp_path / "homes" / "work"
    registry = ProfileRegistry.open(settings)
    try:
        proposal = registry.plan("work", work)
        registry.enroll("work", work, actor=OWNER, review_digest=proposal["review_digest"])
    finally:
        registry.db.close()
    broken = Path(proposal["data_dir"]) / "canonical.db"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("not a database\n", encoding="utf-8")
    report = upgrade_plan(settings, release=release, environ=environ)
    assert report["in_flight"]["held"] == []
    assert "could not be asked about" in said(report)


# -- the digest ---------------------------------------------------------------

def test_the_digest_covers_only_what_a_switch_would_act_on(installation):
    settings, environ, release, _tmp = installation
    first = upgrade_plan(settings, release=release, environ=environ, at=time.time())
    second = upgrade_plan(settings, release=release, environ=environ, at=time.time())
    assert first["review_digest"] == second["review_digest"]


def test_moving_the_target_moves_the_approval(installation, tmp_path):
    settings, environ, release, _tmp = installation
    first = upgrade_plan(settings, release=release, environ=environ)
    second = upgrade_plan(settings, release=staged(tmp_path, "release-0.9.9"),
                         environ=environ)
    assert first["review_digest"] != second["review_digest"]


def test_the_plan_is_serialisable_as_json_without_a_secret_in_it(installation):
    settings, environ, release, _tmp = installation
    text = json.dumps(upgrade_plan(settings, release=release, environ=environ),
                      default=str, sort_keys=True)
    assert "API_KEY" not in text.upper()
    assert str(settings.data_dir) in text


def test_the_kept_list_is_a_promise_about_the_data(installation):
    settings, environ, release, _tmp = installation
    report = upgrade_plan(settings, release=release, environ=environ)
    joined = " ".join(report["kept"])
    assert str(settings.data_dir) in joined
    assert "erasure ledger" in joined and "backup" in joined


def test_the_envelope_shape_a_capture_would_use_is_untouched(installation):
    """A plan that opened the store for writes would move the journal position."""
    settings, environ, release, _tmp = installation
    with EvidenceStore(settings.db_path) as store:
        store.commit(envelope(source_id="before-1"))
        before = store.watermark()
    upgrade_plan(settings, release=release, environ=environ)
    with EvidenceStore(settings.db_path) as store:
        assert store.watermark() == before
