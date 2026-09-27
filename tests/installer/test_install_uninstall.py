"""C14 uninstall (§10.7): the software leaves, the memory stays.

The failures this guards against are all ones that have happened somewhere: an uninstall
that took the archive with it, one that deleted a unit file its owner had spent an
afternoon hand-tuning, and one that "restored" the previous provider by overwriting a
choice the owner made afterwards. So the module is organized around two questions — did
we write this, and is it still what we wrote — and around a refusal to reach for anything
that is not an artefact of its own installation.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from hermes_memory.config import load_settings
from hermes_memory.install.profiles import ProfileRegistry
from hermes_memory.install.services import RECORD_FILENAME, apply as write_units
from hermes_memory.install.services import plan as units_plan
from hermes_memory.install.uninstall import (PROVIDER_RECEIPT, UninstallError,
                                             record_provider_selection, uninstall_apply,
                                             uninstall_plan)
from hermes_memory.storage.evidence import EvidenceStore

OWNER = "jugaadu"
PLUGIN = "hermes-memory"


class Hostes:
    """The host's own plugin and configuration writers, faked and remembered."""

    def __init__(self, *, fail_on=None):
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, argv):
        self.calls.append([str(part) for part in argv])
        joined = " ".join(self.calls[-1])
        if self.fail_on and self.fail_on in joined:
            return 1, "the host refused"
        return 0, ""

    @property
    def verbs(self):
        return [" ".join(call[1:3]) for call in self.calls]


def host_home(tmp_path, *, provider=None) -> Path:
    home = tmp_path / "homes" / "work"
    home.mkdir(parents=True)
    body = "model:\n  provider: openai\n  model: deepseek-chat\n\nmemory:\n"
    body += f"  provider: {provider}\n" if provider else "  memory_enabled: true\n"
    (home / "config.yaml").write_text(body, encoding="utf-8")
    return home


@pytest.fixture()
def installation(tmp_path, monkeypatch):
    """An installed instance: one store, one ledger row, its units written by us."""
    home = tmp_path / "instance"
    (home / "data").mkdir(parents=True)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n", encoding="utf-8")
    settings = load_settings()
    with EvidenceStore(settings.db_path):
        pass
    environ = {"HERMES_MEMORY_RELEASE": str(tmp_path / "release"),
               "XDG_CONFIG_HOME": os.environ["XDG_CONFIG_HOME"]}
    return settings, host_home(tmp_path), environ, tmp_path


def install_units(settings, environ):
    proposal = units_plan(settings, environ=environ)
    return write_units(settings, actor=OWNER, review=proposal["review_digest"],
                       environ=environ)


def approve(settings, hermes_home, environ, *, runner=None, **kwargs):
    """Uninstall the plan that was just shown, as the owner who asked for it."""
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ, **kwargs)
    return uninstall_apply(settings, actor=OWNER, review=proposal["review_digest"],
                           keep_data=True, hermes_home=hermes_home, environ=environ,
                           runner=runner or Hostes(), **kwargs)


def kept_of(receipt) -> str:
    return " ".join(receipt["kept"])


# -- what the plan says -------------------------------------------------------

def test_the_plan_writes_nothing_and_deletes_nothing(installation, tmp_path):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    assert proposal["removals"], "the owned units are what it found"
    assert {path: path.read_bytes() for path in tmp_path.rglob("*")
            if path.is_file()} == before


def test_nothing_installed_leaves_nothing_to_remove(installation):
    settings, hermes_home, environ, _tmp = installation
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    assert proposal["removals"] == [] and proposal["record"] is None


def test_the_owned_units_are_the_ones_named_for_removal(installation):
    settings, hermes_home, environ, _tmp = installation
    receipt = install_units(settings, environ)
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    assert sorted(item["unit"] for item in proposal["removals"]) == \
        sorted(receipt["units_written"])
    assert all(item["why"].startswith("written by this installation")
               for item in proposal["removals"])


def test_the_record_of_writing_them_goes_too(installation):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    assert proposal["record"].endswith(RECORD_FILENAME)


def test_a_unit_somebody_else_wrote_is_refused_by_name(installation):
    settings, hermes_home, environ, _tmp = installation
    directory = Path(units_plan(settings, environ=environ)["unit_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "hermes-memory.service").write_text("[Unit]\nDescription=something else\n",
                                                     encoding="utf-8")
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    assert proposal["removals"] == []
    assert [item["unit"] for item in proposal["refused"]] == ["hermes-memory.service"]


def test_a_unit_we_wrote_and_the_owner_edited_is_reported_not_destroyed(installation):
    settings, hermes_home, environ, tmp_path = installation
    install_units(settings, environ)
    edited = Path(units_plan(settings, environ=environ)["unit_dir"]) / "hermes-memory.service"
    edited.write_text(edited.read_text(encoding="utf-8") + "\n# tuned by hand\n",
                      encoding="utf-8")
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    assert [item["unit"] for item in proposal["modified"]] == ["hermes-memory.service"]
    assert "hermes-memory.service" not in [item["unit"] for item in proposal["removals"]]
    receipt = approve(settings, hermes_home, environ)
    assert str(edited) in receipt["left_behind"]
    assert edited.exists() and "tuned by hand" in edited.read_text(encoding="utf-8")
    assert "MemoryMax" in edited.read_text(encoding="utf-8")


def test_the_host_commands_are_the_named_three_in_the_safe_order(installation):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    (hermes_home / "config.yaml").write_text(
        (hermes_home / "config.yaml").read_text(encoding="utf-8").replace(
            "  memory_enabled: true", "  provider: hermes-memory"), encoding="utf-8")
    record_provider_selection(hermes_home=hermes_home, settings=settings,
                              prior="pg0-memory", actor=OWNER)
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    assert proposal["host_commands"] == [
        ["hermes", "config", "set", "memory.provider", "pg0-memory"],
        ["hermes", "plugins", "disable", PLUGIN],
        ["hermes", "plugins", "remove", PLUGIN]]


def test_the_plan_says_it_registers_no_scheduled_job(installation):
    settings, hermes_home, environ, _tmp = installation
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    assert proposal["jobs"]["owned"] == []
    assert "none for it to unregister" in proposal["jobs"]["note"]


# -- what the approval refuses ------------------------------------------------

def test_an_uninstall_without_keep_data_is_not_an_uninstall(installation):
    settings, hermes_home, environ, _tmp = installation
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    with pytest.raises(UninstallError, match="no flag here that removes them"):
        uninstall_apply(settings, actor=OWNER, review=proposal["review_digest"],
                        keep_data=False, hermes_home=hermes_home, environ=environ,
                        runner=Hostes())


def test_the_call_says_where_purging_belongs(installation):
    settings, hermes_home, environ, _tmp = installation
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    with pytest.raises(UninstallError, match="separate owner operation") as caught:
        uninstall_apply(settings, actor=OWNER, review=proposal["review_digest"],
                        keep_data=False, hermes_home=hermes_home, environ=environ,
                        runner=Hostes())
    assert "manifest" in str(caught.value)


def test_an_unnamed_actor_is_not_enough_of_a_witness(installation):
    settings, hermes_home, environ, _tmp = installation
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    for actor in ("", "   ", None, 7):
        with pytest.raises(UninstallError, match="actor"):
            uninstall_apply(settings, actor=actor, review=proposal["review_digest"],
                            keep_data=True, hermes_home=hermes_home, environ=environ,
                            runner=Hostes())


def test_no_executor_means_no_half_removal(installation):
    """Leaving the registration without the units, or the reverse, is worse than neither."""
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    with pytest.raises(UninstallError, match="no host command executor"):
        uninstall_apply(settings, actor=OWNER, review=proposal["review_digest"],
                        keep_data=True, hermes_home=hermes_home, environ=environ,
                        runner=None)
    assert all(Path(item["path"]).exists() for item in proposal["removals"])


def test_an_approval_spent_on_a_different_list_is_refused(installation):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    with pytest.raises(UninstallError, match="does not match"):
        uninstall_apply(settings, actor=OWNER, review="0" * 64, keep_data=True,
                        hermes_home=hermes_home, environ=environ, runner=Hostes())


def test_editing_a_unit_after_the_plan_made_the_approval_stale(installation):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    edited = Path(proposal["removals"][0]["path"])
    edited.write_text(edited.read_text(encoding="utf-8") + "\n# later\n", encoding="utf-8")
    with pytest.raises(UninstallError, match="does not match"):
        uninstall_apply(settings, actor=OWNER, review=proposal["review_digest"],
                        keep_data=True, hermes_home=hermes_home, environ=environ,
                        runner=Hostes())


def test_a_selection_that_cannot_be_given_back_stops_the_removal(installation, tmp_path):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    (hermes_home / "config.yaml").write_text(
        "model:\n  provider: openai\n\nmemory:\n  provider: hermes-memory\n",
        encoding="utf-8")
    record_provider_selection(hermes_home=hermes_home, settings=settings,
                              prior="unreadable", actor=OWNER)
    with pytest.raises(UninstallError, match="cannot be written back"):
        approve(settings, hermes_home, environ)
    assert all(Path(item["path"]).exists()
               for item in uninstall_plan(settings, hermes_home=hermes_home,
                                          environ=environ)["removals"])


# -- what an uninstalled machine looks like -----------------------------------

def test_the_units_and_their_record_are_gone_and_the_memory_is_not(installation):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    written = [Path(item["path"]) for item in proposal["removals"]]
    receipt = approve(settings, hermes_home, environ)
    assert sorted(str(path) for path in written) == sorted(
        path for path in receipt["removed"] if path.endswith(".service"))
    assert all(not path.exists() for path in written)
    assert settings.data_dir.exists() and settings.db_path.is_file()
    assert (settings.home / "hermes-memory.env").is_file()
    assert proposal["record"] in receipt["removed"]
    assert not Path(proposal["record"]).exists(), \
        "a record of files that no longer exist is a claim about nothing"


def test_the_host_is_told_before_a_single_file_is_removed(installation):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    host = Hostes(fail_on="plugins remove")
    with pytest.raises(UninstallError, match="the host refused"):
        uninstall_apply(settings, actor=OWNER, review=proposal["review_digest"],
                        keep_data=True, hermes_home=hermes_home, environ=environ,
                        runner=host)
    assert host.verbs == ["plugins disable", "plugins remove"]
    assert all(Path(item["path"]).exists() for item in proposal["removals"]), \
        "the destructive step goes last, so a refusal leaves the machine as it was"
    assert proposal["record"] and Path(proposal["record"]).exists()


def test_the_receipt_says_what_was_kept_and_that_nothing_was_purged(installation):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    registry = ProfileRegistry.open(settings)
    registry.db.close()
    receipt = approve(settings, hermes_home, environ)
    assert receipt["purging_data"] is False
    assert str(settings.data_dir) in kept_of(receipt)
    assert str(settings.home / "installation.db") in kept_of(receipt)
    assert receipt["actor"] == OWNER


def test_the_pre_activation_snapshots_of_the_host_file_are_recovery_material(installation):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    snapshot = settings.home / "config.yaml.before-0123456789ab"
    snapshot.write_text("model:\n  provider: openai\n", encoding="utf-8")
    receipt = approve(settings, hermes_home, environ)
    assert str(snapshot) in kept_of(receipt)
    assert snapshot.exists()


def test_the_backups_of_every_profile_are_named_as_kept(installation, tmp_path):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    (settings.data_dir / "snapshots").mkdir()
    personal = tmp_path / "homes" / "personal"
    personal.mkdir()
    registry = ProfileRegistry.open(settings)
    proposal = registry.plan("work", personal)
    registry.enroll("work", personal, actor=OWNER,
                    review_digest=proposal["review_digest"])
    Path(proposal["data_dir"], "snapshots").mkdir(parents=True, exist_ok=True)
    registry.db.close()
    receipt = approve(settings, hermes_home, environ)
    assert str(settings.data_dir / "snapshots") in kept_of(receipt)
    assert str(Path(proposal["data_dir"]) / "snapshots") in kept_of(receipt)


def test_a_second_uninstall_of_the_same_installation_has_nothing_left_to_do(installation):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    first = approve(settings, hermes_home, environ)
    assert first["removed"]
    second = approve(settings, hermes_home, environ)
    assert second["removed"] == []
    assert second["host_commands"], "the registration is still the host's to drop"


def test_the_unit_directory_is_the_only_place_anything_is_deleted(installation):
    settings, hermes_home, environ, tmp_path = installation
    install_units(settings, environ)
    elsewhere = settings.data_dir / "hermes-memory.service"
    elsewhere.write_text("[Unit]\n", encoding="utf-8")
    approve(settings, hermes_home, environ)
    assert elsewhere.exists()
    assert not any(elsewhere.name in path for path in
                   uninstall_plan(settings, hermes_home=hermes_home,
                                  environ=environ)["removals"])


# -- the provider selection ----------------------------------------------------

def test_no_receipt_leaves_the_selection_alone(installation):
    settings, hermes_home, environ, _tmp = installation
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    assert proposal["provider"]["command"] is None
    assert "no activation receipt" in proposal["provider"]["why"]
    assert [[str(part) for part in argv][1:3] for argv in proposal["host_commands"]] == \
        [["plugins", "disable"], ["plugins", "remove"]]


def test_a_selection_the_owner_changed_since_is_left_as_they_left_it(installation):
    settings, hermes_home, environ, _tmp = installation
    record_provider_selection(hermes_home=hermes_home, settings=settings,
                              prior="pg0-memory", actor=OWNER)
    (hermes_home / "config.yaml").write_text(
        "model:\n  provider: openai\n\nmemory:\n  provider: something-else\n",
        encoding="utf-8")
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    assert proposal["provider"]["command"] is None
    assert "owner's choice is left alone" in proposal["provider"]["why"]


def test_a_receipt_for_another_home_does_not_describe_this_one(installation, tmp_path):
    settings, hermes_home, environ, _tmp = installation
    record_provider_selection(hermes_home=tmp_path / "homes" / "other",
                              settings=settings, prior="pg0-memory", actor=OWNER)
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    assert proposal["provider"]["command"] is None
    assert "describes" in proposal["provider"]["why"]


def test_a_prior_value_that_is_not_a_provider_name_is_never_written_back(installation):
    """The receipt is a file, and a file can hold anything. Only a name gets through."""
    settings, hermes_home, environ, _tmp = installation
    (hermes_home / "config.yaml").write_text(
        "model:\n  provider: openai\n\nmemory:\n  provider: hermes-memory\n",
        encoding="utf-8")
    for prior in ("two words", "", "--force", "x" * 200, None, 7):
        (settings.home / PROVIDER_RECEIPT).write_text(json.dumps(
            {"config": str(hermes_home / "config.yaml"), "prior": prior,
             "written": PLUGIN}), encoding="utf-8")
        proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
        assert proposal["provider"]["command"] is None, prior
        assert proposal["provider"]["restore_unsupported"], prior
        with pytest.raises(UninstallError, match="cannot be written back"):
            approve(settings, hermes_home, environ)


def test_a_fresh_host_has_no_selection_to_restore_only_one_to_take_back(installation):
    """The usual case: there was no memory.provider before setup, so unset is the undo."""
    settings, hermes_home, environ, _tmp = installation
    (hermes_home / "config.yaml").write_text(
        "model:\n  provider: openai\n\nmemory:\n  provider: hermes-memory\n",
        encoding="utf-8")
    record_provider_selection(hermes_home=hermes_home, settings=settings,
                              prior="unknown", actor=OWNER)
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    assert proposal["provider"]["command"] == ["hermes", "config", "unset",
                                               "memory.provider"]
    assert proposal["provider"]["prior"] == "unknown"
    assert "removed rather than renamed" in proposal["provider"]["why"]
    assert proposal["host_commands"][0][:3] == ["hermes", "config", "unset"]


def test_a_written_prior_is_never_replaced_by_the_value_we_wrote(installation):
    """A second activation must not record hermes-memory as the thing it replaced."""
    settings, hermes_home, environ, _tmp = installation
    first = record_provider_selection(hermes_home=hermes_home, settings=settings,
                                      prior="pg0-memory", actor=OWNER)
    again = record_provider_selection(hermes_home=hermes_home, settings=settings,
                                      prior=PLUGIN, actor=OWNER)
    assert first["written"] is True and again["written"] is False
    assert again["receipt"]["prior"] == "pg0-memory"


def test_the_receipt_is_owner_only_and_holds_no_secret(installation):
    settings, hermes_home, environ, _tmp = installation
    made = record_provider_selection(hermes_home=hermes_home, settings=settings,
                                     prior="pg0-memory", actor=OWNER)
    path = Path(made["path"])
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "API_KEY" not in path.read_text(encoding="utf-8").upper()


def test_the_plan_is_json_and_the_whole_report_survives_printing(installation):
    settings, hermes_home, environ, _tmp = installation
    install_units(settings, environ)
    proposal = uninstall_plan(settings, hermes_home=hermes_home, environ=environ)
    text = json.dumps(proposal, sort_keys=True)
    assert json.loads(text)["review_digest"] == proposal["review_digest"]
    assert "--review" in proposal["next"]
    assert str(settings.home / PROVIDER_RECEIPT) not in text
    assert "8888" not in text
