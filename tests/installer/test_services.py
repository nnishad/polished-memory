"""C14 owned user services: render, refuse, reload only on a real change.

The guards that matter here are the ones with the longest half-life. A unit file runs
for months with no one looking at it, so: it may name no path outside this installation,
a file somebody else wrote is never overwritten, a start never resumes a pause the
owner set, nothing reaches outside ``systemctl --user``, and the manager is only told
about a change when one of our own files actually changed.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from hermes_memory.config import load_settings
from hermes_memory.install import services
from hermes_memory.install.profiles import InstallationError
from hermes_memory.install.services import (BACKEND_UNIT, RUNTIME_UNIT, UNITS,
                                            WORKER_UNIT, Services, apply, executables,
                                            layout, plan, render, unit_directory,
                                            wanted_units)

OWNER = "jugaadu"
RELEASE = "/srv/releases/hermes-memory/0.1.0"


def environment(**extra):
    """The environment the installer is handed: our release pointer, our config home.

    Read from the process rather than written as a constant because a frozen dict would
    silently point the unit directory at the real account when the fixture that moves it
    is not in play. A missing key here is a loud test failure instead.
    """
    values = {"HERMES_MEMORY_RELEASE": RELEASE,
              "XDG_CONFIG_HOME": os.environ["XDG_CONFIG_HOME"]}
    values.update(extra)
    return values


@pytest.fixture()
def instance(tmp_path, monkeypatch):
    """An installation with a Hindsight backend configured, and a unit directory of its own."""
    home = tmp_path / "instance"
    home.mkdir()
    # The directories the backend unit binds. `initialize` makes these on a real machine, and
    # `services` now refuses a unit that names one which is absent — under
    # `ProtectSystem=strict` that is a mount-namespace failure at exec time, not a warning.
    for directory in (home / "hindsight", home / "pg0", home / "cache" / "huggingface",
                      home / "data"):
        directory.mkdir(parents=True)
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_HINDSIGHT_URL=http://127.0.0.1:8888\n"
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS=127.0.0.1\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return home, load_settings()


@pytest.fixture()
def units(instance):
    """The unit directory this test will use.

    Declared after ``instance`` on purpose: that fixture is what points
    ``XDG_CONFIG_HOME`` at the temporary home, and asking for the path first would hand
    these tests the real one.
    """
    return unit_directory()


def approve(settings, **kwargs):
    proposal = plan(settings, **kwargs)
    if proposal["blocked"]:
        raise AssertionError(proposal["blocked"])
    return apply(settings, actor=OWNER, review=proposal["review_digest"], **kwargs)


class Runner:
    """A systemctl that answers and remembers, so ordering is a checked claim."""

    def __init__(self, code=0, stdout=""):
        self.calls: list[list[str]] = []
        self.code = code
        self.stdout = stdout

    def __call__(self, argv):
        self.calls.append(list(argv))
        return self.code, self.stdout

    @property
    def verbs(self):
        return [" ".join(call[2:]) for call in self.calls]


# -- the rendering ------------------------------------------------------------

def test_every_wanted_unit_renders_with_nothing_left_to_fill_in(instance):
    _, settings = instance
    rendered = render(settings, environ=environment())
    assert set(rendered) == set(UNITS)
    for name, text in rendered.items():
        assert "@" not in text, name
        assert "Documentation=file:///home/" not in text
        assert f"Environment=HERMES_MEMORY_HOME={settings.home}" in text


def test_a_capture_only_installation_offers_nothing_to_start_a_backend_for(instance,
                                                                          monkeypatch):
    home, settings = instance
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    bare = load_settings()
    assert wanted_units(bare) == (RUNTIME_UNIT,)
    assert set(render(bare, environ=environment())) == {RUNTIME_UNIT}


def test_the_release_pointer_names_the_code_and_the_home_names_the_data(instance):
    _, settings = instance
    placed = layout(settings, environ=environment())
    assert str(placed.release) == RELEASE
    assert str(placed.instance_home) == str(settings.home)
    # The embedded database gets a location of its own, not an implicit ~/.pg0.
    assert placed.pg0_dir == settings.home / "pg0"
    without = layout(settings, environ={})
    assert without.release == settings.home / "runtime" / "current"


def test_the_programs_named_off_the_units_are_the_ones_the_release_ships(instance):
    """Three units, three absolute programs — and one of them is our launcher.

    The staging step asks this question instead of keeping its own list, so the answer
    has to include the worker's ``python -m`` program: a check that only looked at
    ``ExecStart`` lines with a single word would report a staged release whose worker
    cannot start at all.
    """
    _, settings = instance
    programs = executables(render(settings, environ=environment()))
    assert programs == [Path(RELEASE) / "bin" / "hermes-memory",
                        Path(RELEASE) / "hindsight" / "bin" / "hindsight-api",
                        Path(RELEASE) / "hindsight" / "bin" / "python"]


def test_a_program_written_as_a_relative_path_is_not_one_anybody_installed(tmp_path):
    """A unit that ExecStarts ``hindsight-api`` starts it out of systemd's own PATH.

    Treating that as staged would approve a release that does not contain the binary and
    then watch the service fail on a machine its owner believed was working.
    """
    programs = executables({"hermes-memory-hindsight.service":
                            "[Service]\nExecStart=hindsight-api\n"})
    assert programs == []


def test_the_prefixes_systemd_strips_are_not_part_of_the_program_name():
    """``-``/``+``/``!``/``:`` change how a program runs, not which program it is.

    A prefix left on the path would be reported missing even in a perfectly staged
    release, and the installer would refuse a machine that works.
    """
    programs = executables({
        "a.service": "[Service]\nExecStart=-/srv/bin/a\n",
        "b.service": "[Service]\nExecStart=+!/srv/bin/b arg\n",
        "c.service": "[Service]\nExecStart=/srv/bin/a\n",
    })
    assert programs == [Path("/srv/bin/a"), Path("/srv/bin/b")]


def test_the_unit_directory_is_the_owner_own_manager_config():
    assert unit_directory(environ={}).name == "user"
    assert unit_directory(environ={}).parent.name == "systemd"
    assert unit_directory(environ={"XDG_CONFIG_HOME": "/srv/xdg"}) == \
        Path("/srv/xdg/systemd/user")


def test_the_ordering_in_the_templates_agrees_with_the_order_we_start_in(instance):
    """Two units that each wait for the other is a cycle the manager breaks at random."""
    rendered = render(instance[1], environ=environment())
    assert "After=network-online.target hermes-memory.service" in rendered[BACKEND_UNIT]
    assert "After=" in rendered[WORKER_UNIT]
    assert BACKEND_UNIT not in rendered[RUNTIME_UNIT].split("[Unit]")[1]
    assert WORKER_UNIT not in rendered[RUNTIME_UNIT]


def test_the_backend_is_told_where_its_own_database_lives(instance):
    """An unstated pg0 location means whichever old database is already there.

    §10.5: the backend gets a data location of its own, and the worker reads the same
    one rather than starting a second embedded instance somewhere in the account. The
    engine's own cluster directory is not configurable — pg0 takes it as a flag the
    engine never passes and otherwise writes `~/.pg0` — so what the unit can do is give
    the process a home inside the installation, which is also the only place its
    read-only-`ProtectHome` sandbox is allowed to write.
    """
    _, settings = instance
    rendered = render(settings, environ=environment())
    for name in (BACKEND_UNIT, WORKER_UNIT):
        pg0 = str(settings.home / "pg0")
        assert f"Environment=HOME={pg0}" in rendered[name], name
        assert "ReadWritePaths=" in rendered[name]
        assert pg0 in rendered[name].split("ReadWritePaths=")[1]
        assert pg0.startswith(str(settings.home)), "the cluster stays inside the installation"


FORGED = [("ReadWritePaths=@INSTANCE_HOME@", "ReadWritePaths=/home/other/.hermes"),
          ("ExecStart=@RELEASE@/bin/hermes-memory serve",
           "ExecStart=/usr/local/bin/something-else"),
          ("EnvironmentFile=@ENV_FILE@", "EnvironmentFile=/etc/hermes-other.env")]


@pytest.fixture(params=FORGED, ids=("a-writable-path", "an-executable", "a-config-file"))
def forged(request):
    """A location line the runtime template really carries, and one to forge it with."""
    return request.param


def test_a_template_naming_a_path_outside_the_installation_is_refused(instance, tmp_path,
                                                                     forged):
    """Every line that names a location, not just the obvious one.

    A unit whose ExecStart was quietly pointed elsewhere would still be started by the
    owner, reading the owner's own configuration, running somebody else's program.
    """
    _, settings = instance
    wanted, replacement = forged
    forged_dir = tmp_path / "templates"
    forged_dir.mkdir()
    for name in UNITS:
        text = (services.template_root() / name).read_text(encoding="utf-8")
        if name == RUNTIME_UNIT:
            assert wanted in text, f"the fixture and the template have drifted: {wanted}"
            text = text.replace(wanted, replacement, 1)
        (forged_dir / name).write_text(text, encoding="utf-8")
    with pytest.raises(InstallationError, match="outside"):
        render(settings, environ=environment(), templates=forged_dir)


def test_a_token_nobody_answered_is_an_error_not_an_installed_unit(instance, tmp_path):
    _, settings = instance
    forged = tmp_path / "templates"
    forged.mkdir()
    for name in UNITS:
        (forged / name).write_text("[Service]\nExecStart=@SOMETHING_ELSE@/bin/x\n",
                                   encoding="utf-8")
    with pytest.raises(InstallationError, match="SOMETHING_ELSE"):
        render(settings, environ=environment(), templates=forged)


def test_a_release_tree_is_asked_before_this_checkout(tmp_path):
    shipped = tmp_path / "release" / "deployment" / "systemd"
    shipped.mkdir(parents=True)
    assert services.template_root(
        environ={"HERMES_MEMORY_RELEASE": str(tmp_path / "release")}) == shipped
    # Nothing shipped beside the release still means a real directory to read, and the
    # failure when there is none is a refusal rather than a silent fallback.
    assert services.template_root(environ={}).is_dir()


def test_a_wanted_unit_the_release_does_not_ship_is_a_refusal(instance, tmp_path):
    _, settings = instance
    partial = tmp_path / "templates"
    partial.mkdir()
    shutil.copy2(services.template_root() / RUNTIME_UNIT, partial / RUNTIME_UNIT)
    with pytest.raises(InstallationError, match=BACKEND_UNIT):
        render(settings, environ=environment(), templates=partial)


# -- the plan: nothing changes until it is approved --------------------------

def test_a_first_plan_says_what_would_change_and_changes_nothing(instance, units):
    _, settings = instance
    proposal = plan(settings, environ=environment())
    assert [entry["state"] for entry in proposal["units"]] == ["written"] * 3
    assert proposal["would_change"] == list(UNITS)
    assert proposal["reload_needed"] is True
    assert not units.exists(), "planning is not doing"


def test_the_plan_digest_covers_exactly_the_paths_that_would_be_written(instance, units):
    _, settings = instance
    first = plan(settings, environ=environment())
    assert first["review_digest"] == plan(settings, environ=environment())["review_digest"]
    other = plan(settings, environ=environment(), unit_dir=units.parent / "elsewhere")
    assert other["review_digest"] != first["review_digest"]


def test_apply_refuses_a_digest_that_is_not_the_plan_now_shown(instance, units):
    _, settings = instance
    with pytest.raises(InstallationError, match="does not match"):
        apply(settings, actor=OWNER, review="0" * 64, environ=environment())
    assert not units.exists()


def test_apply_refuses_an_actor_who_will_not_be_named_in_the_record(instance, units):
    _, settings = instance
    proposal = plan(settings, environ=environment())
    for blank in (None, "", "   "):
        with pytest.raises(InstallationError, match="actor"):
            apply(settings, actor=blank, review=proposal["review_digest"],
                  environ=environment())
    assert not units.exists()


def test_written_units_are_readable_by_the_manager_and_nothing_else_is_granted(instance,
                                                                             units):
    _, settings = instance
    receipt = approve(settings, environ=environment())
    assert sorted(receipt["units_written"]) == sorted(UNITS)
    assert receipt["reload_needed"] is True
    assert units.stat().st_mode & 0o777 == 0o700
    for name in UNITS:
        assert (units / name).stat().st_mode & 0o777 == 0o644
    logged = json.loads((units / "services.json").read_text(encoding="utf-8"))
    assert logged["written_by"] == OWNER
    assert set(logged["units"]) == set(UNITS)
    assert (units / "services.json").stat().st_mode & 0o077 == 0


def test_approving_the_same_plan_twice_changes_nothing_and_asks_for_no_reload(instance,
                                                                             units):
    _, settings = instance
    first = approve(settings, environ=environment())
    stamps = {name: (units / name).stat().st_mtime_ns for name in UNITS}
    again = plan(settings, environ=environment())
    assert [entry["state"] for entry in again["units"]] == ["unchanged"] * 3
    assert again["would_change"] == []
    assert again["reload_needed"] is False
    approve(settings, environ=environment())
    assert {name: (units / name).stat().st_mtime_ns for name in UNITS} == stamps
    assert first["unit_dir"] == again["unit_dir"]


def test_a_unit_somebody_else_wrote_is_reported_and_left_byte_for_byte_alone(instance,
                                                                            units):
    _, settings = instance
    units.mkdir(parents=True, mode=0o700)
    (units / BACKEND_UNIT).write_text("[Service]\nExecStart=/opt/theirs/api\n",
                                      encoding="utf-8")
    proposal = plan(settings, environ=environment())
    states = {entry["unit"]: entry["state"] for entry in proposal["units"]}
    assert states[BACKEND_UNIT] == "collision"
    assert states[RUNTIME_UNIT] == "written"
    assert proposal["blocked"] == [BACKEND_UNIT]
    with pytest.raises(InstallationError, match=BACKEND_UNIT):
        apply(settings, actor=OWNER, review=proposal["review_digest"], environ=environment())
    assert (units / BACKEND_UNIT).read_text(encoding="utf-8") == \
        "[Service]\nExecStart=/opt/theirs/api\n"
    assert not (units / RUNTIME_UNIT).exists(), "half a topology is worse than none"


def test_a_file_we_wrote_that_has_since_been_edited_by_hand_is_not_silently_ours(instance,
                                                                                units):
    """An owner edit is a decision. Losing it to a regeneration is the failure to prevent."""
    _, settings = instance
    approve(settings, environ=environment())
    (units / RUNTIME_UNIT).write_text("[Service]\nMemoryMax=1G\n", encoding="utf-8")
    proposal = plan(settings, environ=environment())
    entry = {item["unit"]: item for item in proposal["units"]}[RUNTIME_UNIT]
    assert entry["state"] == "collision" and entry["ours"] is False
    assert proposal["blocked"] == [RUNTIME_UNIT]


def test_a_corrupt_ownership_record_claims_nothing(instance, units):
    _, settings = instance
    approve(settings, environ=environment())
    (units / "services.json").write_text("{not json", encoding="utf-8")
    (units / RUNTIME_UNIT).write_text("[Service]\nMemoryMax=1G\n", encoding="utf-8")
    assert plan(settings, environ=environment())["blocked"] == [RUNTIME_UNIT]


def test_an_existing_file_we_would_have_written_is_adopted_without_being_touched(instance,
                                                                               units):
    _, settings = instance
    ours = render(settings, environ=environment())[RUNTIME_UNIT]
    units.mkdir(parents=True, mode=0o700)
    (units / RUNTIME_UNIT).write_text(ours, encoding="utf-8")
    stamp = (units / RUNTIME_UNIT).stat().st_mtime_ns
    entry = {item["unit"]: item for item in plan(settings, environ=environment())["units"]}
    assert entry[RUNTIME_UNIT]["state"] == "adopted"
    assert entry[RUNTIME_UNIT]["ours"] is False
    approve(settings, environ=environment())
    assert (units / RUNTIME_UNIT).stat().st_mtime_ns == stamp
    assert plan(settings, environ=environment())["units"][0]["state"] == "unchanged"


def test_moving_the_release_rewrites_our_own_units_and_only_ours(instance, units):
    _, settings = instance
    approve(settings, environ=environment())
    moved = environment(HERMES_MEMORY_RELEASE=RELEASE + "-next")
    proposal = plan(settings, environ=moved)
    assert [entry["state"] for entry in proposal["units"]] == ["rewritten"] * 3
    assert proposal["reload_needed"] is True
    receipt = approve(settings, environ=moved)
    assert sorted(receipt["units_written"]) == sorted(UNITS)
    assert f"{RELEASE}-next/bin/hermes-memory serve" in (units / RUNTIME_UNIT).read_text()


def test_switching_the_backend_off_removes_the_units_this_installation_wrote(instance,
                                                                            units,
                                                                            monkeypatch):
    _, settings = instance
    approve(settings, environ=environment())
    (settings.home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={settings.home / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(settings.home))
    bare = load_settings()
    proposal = plan(bare, environ=environment())
    assert {entry["unit"]: entry["state"] for entry in proposal["units"]} == {
        RUNTIME_UNIT: "unchanged", BACKEND_UNIT: "removed", WORKER_UNIT: "removed"}
    receipt = apply(bare, actor=OWNER, review=proposal["review_digest"], environ=environment())
    assert sorted(receipt["units_removed"]) == sorted([BACKEND_UNIT, WORKER_UNIT])
    assert not (units / WORKER_UNIT).exists()
    assert set(json.loads((units / "services.json").read_text())["units"]) == {RUNTIME_UNIT}


def test_a_foreign_unit_survives_the_removal_of_a_unit_we_owed(instance, units):
    _, settings = instance
    approve(settings, environ=environment())
    (units / WORKER_UNIT).write_text("[Service]\nExecStart=/opt/theirs/worker\n",
                                     encoding="utf-8")
    settings.home.joinpath("hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={settings.home / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n", encoding="utf-8")
    from hermes_memory.config import load_settings as load
    bare = load()
    entry = {item["unit"]: item for item in plan(bare, environ=environment())["units"]}
    assert entry[WORKER_UNIT]["state"] == "left-alone"
    apply(bare, actor=OWNER, review=plan(bare, environ=environment())["review_digest"],
          environ=environment())
    assert "theirs" in (units / WORKER_UNIT).read_text(encoding="utf-8")


# -- the controller ------------------------------------------------------------

def test_starting_follows_the_top_order_and_reloads_nothing(instance):
    _, settings = instance
    runner = Runner()
    assert Services(settings, runner=runner).start() == list(UNITS)
    assert runner.calls == [["systemctl", "--user", "start", name] for name in UNITS]


def test_a_capture_only_installation_starts_only_the_runtime(instance, monkeypatch):
    home, settings = instance
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    runner = Runner()
    Services(load_settings(), runner=runner).start()
    assert runner.calls == [["systemctl", "--user", "start", RUNTIME_UNIT]]


def test_stopping_persists_the_pause_before_touching_a_process(instance):
    _, settings = instance
    events = []
    runner = Runner()
    services_object = Services(settings, runner=runner)
    stopped = services_object.stop(pause=lambda: events.append("paused"))
    assert events == ["paused"]
    assert stopped == list(reversed(UNITS))
    assert runner.calls[0][2:] == ["stop", WORKER_UNIT]


def test_stopping_without_a_pause_resumes_nothing_of_its_own_accord(instance):
    _, settings = instance
    runner = Runner()
    Services(settings, runner=runner).stop()
    assert not [call for call in runner.calls
                if any(word in call for word in ("unpause", "enable", "daemon-reload"))]


def test_autostart_is_a_separate_decision_and_can_be_taken_back(instance):
    _, settings = instance
    runner = Runner()
    Services(settings, runner=runner).autostart()
    assert runner.calls[0] == ["systemctl", "--user", "enable", RUNTIME_UNIT]
    assert runner.calls[-1] == ["systemctl", "--user", "enable", WORKER_UNIT]
    off = Runner()
    Services(settings, runner=off).autostart(enable=False)
    assert off.calls[0] == ["systemctl", "--user", "disable", WORKER_UNIT]
    now = Runner()
    Services(settings, runner=now).autostart(start_now=True)
    assert now.calls[0][:4] == ["systemctl", "--user", "enable", "--now"]


def test_the_reloader_is_only_asked_after_files_actually_changed(instance, units):
    _, settings = instance
    runner = Runner()
    controller = Services(settings, runner=runner)
    proposal = plan(settings, environ=environment())
    if proposal["reload_needed"]:
        approve(settings, environ=environment())
        controller.daemon_reload()
    assert runner.calls == [["systemctl", "--user", "daemon-reload"]]
    runner.calls.clear()
    assert plan(settings, environ=environment())["reload_needed"] is False
    assert runner.calls == []


@pytest.mark.parametrize("argument", [["--system", "start", RUNTIME_UNIT],
                                      ["sudo", "systemctl", "start", RUNTIME_UNIT],
                                      ["mask", RUNTIME_UNIT],
                                      ["enable-linger", OWNER],
                                      ["--global", "enable", RUNTIME_UNIT]])
def test_no_amount_of_asking_reaches_beyond_the_owners_own_manager(instance, argument):
    _, settings = instance
    runner = Runner()
    with pytest.raises(InstallationError):
        Services(settings, runner=runner)._run(argument)
    assert runner.calls == []


def test_a_unit_that_is_not_ours_cannot_be_named_even_by_accident(instance):
    _, settings = instance
    runner = Runner()
    controller = Services(settings, runner=runner)
    with pytest.raises(InstallationError, match="not this installation's units"):
        controller._run(["stop", "postgresql.service"])
    with pytest.raises(InstallationError, match="not one of this installation"):
        controller.restart("hermes-other.service")
    assert runner.calls == []


def test_a_failing_host_command_is_reported_with_the_command_that_failed(instance):
    _, settings = instance
    runner = Runner(code=1, stdout="Failed to start hermes-memory.service: Permission denied")
    with pytest.raises(InstallationError) as refused:
        Services(settings, runner=runner).start()
    assert "systemctl --user start " + RUNTIME_UNIT in str(refused.value)
    assert "Permission denied" in str(refused.value)


def test_status_reads_the_manager_rather_than_guessing_from_files(instance, units):
    _, settings = instance
    approve(settings, environ=environment())
    stdout = "\n".join([
        f"Id={RUNTIME_UNIT}", "LoadState=loaded", "ActiveState=active", "SubState=running",
        "UnitFileState=enabled", "",
        f"Id={BACKEND_UNIT}", "LoadState=loaded", "ActiveState=inactive", "SubState=dead",
        "UnitFileState=disabled", "",
        f"Id={WORKER_UNIT}", "LoadState=loaded", "ActiveState=activating",
        "SubState=start-pre", "UnitFileState=enabled", ""])
    report = Services(settings, runner=Runner(stdout=stdout)).status()
    assert report["units"][RUNTIME_UNIT]["ActiveState"] == "active"
    assert report["units"][BACKEND_UNIT]["UnitFileState"] == "disabled"
    assert report["units"][WORKER_UNIT]["SubState"] == "start-pre"
    assert all(report["installed"].values())
    assert report["expected"] == list(UNITS)


def test_status_says_what_the_manager_does_not_know_about(instance):
    _, settings = instance
    report = Services(settings, runner=Runner(stdout="")).status()
    assert all(report["units"][name] == {} for name in UNITS)


def test_status_answers_for_the_units_this_installation_would_start(instance, monkeypatch):
    """A capture-only account has no backend to ask about, and must not report one.

    The manager is sent the units we expect; answering for the others from a table we
    never queried would invent a state that has no source.
    """
    home, settings = instance
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    bare = load_settings()
    runner = Runner(stdout=f"Id={RUNTIME_UNIT}\nActiveState=active\n")
    report = Services(bare, runner=runner).status()
    assert list(report["units"]) == [RUNTIME_UNIT]
    assert report["units"][RUNTIME_UNIT]["ActiveState"] == "active"
    assert runner.calls == [["systemctl", "--user", "--no-legend", "show", RUNTIME_UNIT,
                             "-p", "Id", "-p", "LoadState", "-p", "ActiveState",
                             "-p", "SubState", "-p", "UnitFileState"]]


def test_a_unit_block_without_a_separating_blank_line_still_stays_separate(instance):
    """The output shape differs between manager versions; mixing two units does not.

    One dictionary shared by two blocks would report the worker's state beside the
    runtime's name, which is the kind of wrong answer that settles an argument.
    """
    _, settings = instance
    stdout = (f"Id={RUNTIME_UNIT}\nActiveState=active\n"
              f"Id={BACKEND_UNIT}\nActiveState=failed\n"
              f"Id={WORKER_UNIT}\nActiveState=inactive\n")
    units = Services(settings, runner=Runner(stdout=stdout)).status()["units"]
    assert units[RUNTIME_UNIT]["ActiveState"] == "active"
    assert units[BACKEND_UNIT]["ActiveState"] == "failed"
    assert units[WORKER_UNIT]["ActiveState"] == "inactive"
    assert all(units[name].get("Id") == name for name in UNITS)


def test_a_unit_naming_a_directory_that_was_never_made_is_refused_before_it_is_written(
        tmp_path, monkeypatch):
    """The failure this prevents was real: exit 226/NAMESPACE, with the unit never starting.

    An installation whose backend directories were not created used to have its units written
    and approved, and then discover on the first `start` that the manager could not spawn the
    process at all.
    """
    home = tmp_path / "never-initialised"
    home.mkdir()
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n"
        "HERMES_MEMORY_HINDSIGHT_URL=http://127.0.0.1:8888\n"
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS=127.0.0.1\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    settings = load_settings()
    proposal = plan(settings, environ=environment())
    assert str(home / "pg0") in proposal["missing_paths"], proposal["missing_paths"]
    with pytest.raises(InstallationError, match="would fail to spawn"):
        apply(settings, actor=OWNER, review=proposal["review_digest"],
              environ=environment())
