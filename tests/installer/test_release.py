"""§10.3 staging: two environments from one wheel, or nothing at all.

The installer has always refused to fetch a release, which left the release itself produced by
an unrecorded sequence of commands that no test could contradict. These hold the new door to
the three rules the rest of the installer lives by: a plan writes nothing, an approval dies
when the tree under it changes, and a failure half-way through leaves no half-staged tree for
a later command to mistake for a release.

The package commands run through a stand-in that keeps ``uv``'s shape: this suite proves the
staging step's promises, not the network's availability, and the real thing is exercised once
by hand on a machine that is allowed to be slow.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from hermes_memory.backend.capabilities import PINNED_VERSION
from hermes_memory.config import load_settings
from hermes_memory.ids import content_digest
from hermes_memory.install.release import (BACKEND_SPEC, MANIFEST, ReleaseError, apply,
                                           plan, verify)

OWNER = "jugaadu"
TEMPLATES = Path(__file__).resolve().parents[2] / "deployment" / "systemd"


class Fake:
    """A ``uv`` that creates the files ``uv`` would create, and remembers being asked."""

    def __init__(self, fail_on=None):
        self.calls: list[list[str]] = []
        self.fail_on = set(fail_on or ())

    def __call__(self, argv, **kwargs):
        argv = [str(part) for part in argv]
        self.calls.append(argv)
        for marker in self.fail_on:
            if marker in " ".join(argv):
                return subprocess.CompletedProcess(argv, 1, "", f"{marker} refused")
        if argv[:2] == ["uv", "venv"]:
            # As strict as the real command: a half-filled directory is refused, which is
            # what makes the order of the staging steps observable here at all.
            venv = Path(argv[2])
            if venv.exists() and any(venv.iterdir()):
                return subprocess.CompletedProcess(argv, 1, "",
                                                   "A directory already exists")
            (venv / "bin").mkdir(parents=True, exist_ok=True)
            (venv / "bin" / "python").write_text("#!/bin/sh\n")
        elif argv[1:3] == ["pip", "install"]:
            environment = Path(argv[argv.index("--python") + 1]).parent.parent
            name = "hermes-memory" if any(part.endswith(".whl") for part in argv) \
                else "hindsight-api"
            (environment / "bin" / name).write_text("#!/bin/sh\n")
        elif argv[:2] == ["uv", "build"]:
            out = Path(argv[argv.index("--out-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "hermes_memory-0.1.0-py3-none-any.whl").write_text("wheel\n")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def mentioning(self, fragment: str) -> list[list[str]]:
        return [call for call in self.calls if any(fragment in part for part in call)]


@pytest.fixture()
def instance(tmp_path, monkeypatch):
    """An installation with a backend route, because that is the case with two venvs."""
    home = tmp_path / "data"
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HERMES_MEMORY_HINDSIGHT_URL", "http://127.0.0.1:8888")
    monkeypatch.setenv("HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS", "127.0.0.1")
    return home, load_settings()


@pytest.fixture()
def source(tmp_path):
    """A checkout that looks like a release source. The service templates are the real ones,
    because a staged tree has to satisfy what this build's units actually start."""
    root = tmp_path / "checkout"
    (root / "deployment" / "systemd").mkdir(parents=True)
    (root / "integrations" / "hermes-memory").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "hermes-memory"\n')
    (root / "deployment" / "compatibility.json").write_text("{}\n")
    for template in TEMPLATES.glob("*.service"):
        (root / "deployment" / "systemd" / template.name).write_text(
            template.read_text(encoding="utf-8"))
    (root / "integrations" / "hermes-memory" / "provider.py").write_text(
        "def register():\n    return None\n")
    return root


def staged(instance, source, tmp_path, *, wheel=None, runner=None, review=None,
           backend=True, actor=OWNER, into=None):
    settings = instance[1]
    into = Path(into or tmp_path / "release")
    shown = plan(settings=settings, into=into, source=source, wheel=wheel, backend=backend)
    return apply(settings=settings, into=into, source=source, wheel=wheel, backend=backend,
                 actor=actor, review=review or shown["review_digest"], runner=runner)


def a_wheel(tmp_path, name="given-0.1.0-py3-none-any.whl"):
    wheel = tmp_path / name
    wheel.write_text("wheel\n")
    return wheel


# -- the plan is a reading ----------------------------------------------------

def test_a_plan_writes_nothing_and_asks_for_no_packages(instance, source, tmp_path):
    report = plan(settings=instance[1], into=tmp_path / "never", source=source)
    assert not (tmp_path / "never").exists()
    assert report["blocking"] == []
    assert any("hindsight-api-slim" in line for line in report["actions"])


def test_a_tree_that_already_holds_something_is_not_a_place_to_stage(instance, source,
                                                                     tmp_path):
    occupied = tmp_path / "release"
    (occupied / "bin").mkdir(parents=True)
    (occupied / "bin" / "keep-me").write_text("someone's work\n")
    report = plan(settings=instance[1], into=occupied, source=source)
    assert any("side by side" in line for line in report["blocking"])


def test_a_pointer_is_refused_because_staging_through_it_writes_through_it(instance, source,
                                                                          tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    pointer = tmp_path / "current"
    pointer.symlink_to(real)
    report = plan(settings=instance[1], into=pointer, source=source)
    assert any("is a pointer" in line for line in report["blocking"])
    assert not (real / "bin").exists()


def test_a_source_that_does_not_look_like_one_stops_the_plan(instance, tmp_path):
    report = plan(settings=instance[1], into=tmp_path / "release", source=tmp_path / "nowhere")
    assert any("no source checkout" in line for line in report["blocking"])


def test_a_release_missing_its_plugin_files_would_ship_two_halves_of_one_contract(instance,
                                                                                 tmp_path,
                                                                                 source):
    for path in (source / "integrations" / "hermes-memory").glob("*"):
        path.unlink()
    report = plan(settings=instance[1], into=tmp_path / "release", source=source)
    assert any("plugin are missing" in line for line in report["blocking"])


# -- the approval is for a stated tree ---------------------------------------

def test_staging_names_who_approved_it(instance, source, tmp_path):
    shown = plan(settings=instance[1], into=tmp_path / "release", source=source)
    with pytest.raises(ReleaseError, match="who approved"):
        apply(settings=instance[1], into=tmp_path / "release", source=source, actor="",
              review=shown["review_digest"], runner=Fake())


def test_an_approval_does_not_survive_the_tree_changing_underneath_it(instance, source,
                                                                     tmp_path):
    shown = plan(settings=instance[1], into=tmp_path / "release", source=source)
    (source / "integrations" / "hermes-memory" / "provider.py").write_text("changed\n")
    with pytest.raises(ReleaseError, match="plan changed"):
        apply(settings=instance[1], into=tmp_path / "release", source=source, actor=OWNER,
              review=shown["review_digest"], runner=Fake())
    assert not (tmp_path / "release").exists()


def test_a_wheel_named_by_hand_is_the_one_staged(instance, source, tmp_path):
    wheel = a_wheel(tmp_path)
    fake = Fake()
    report = staged(instance, source, tmp_path, wheel=wheel, runner=fake)
    assert report["performed"] is True
    assert fake.mentioning("uv build") == []
    assert fake.mentioning(str(wheel))


# -- the pin is one fact, not a copy -----------------------------------------

def test_the_pin_is_the_one_the_routes_are_pinned_to():
    assert BACKEND_SPEC == f"hindsight-api-slim[embedded-db]=={PINNED_VERSION}"


def test_backend_packages_go_into_the_backend_environment_only(instance, source, tmp_path):
    fake = Fake()
    staged(instance, source, tmp_path, runner=fake)
    asked = fake.mentioning(BACKEND_SPEC)
    assert len(asked) == 1
    assert asked[0][asked[0].index("--python") + 1].endswith("hindsight/bin/python")


def test_both_environments_run_the_same_framework_code(instance, source, tmp_path):
    wheel = a_wheel(tmp_path, name="w-0.1.0-py3-none-any.whl")
    fake = Fake()
    report = staged(instance, source, tmp_path, wheel=wheel, runner=fake)
    assert [call[call.index("--python") + 1] for call in fake.mentioning(str(wheel))] == [
        str(Path(report["staged"]) / "bin" / "python"),
        str(Path(report["staged"]) / "hindsight" / "bin" / "python")]


def test_a_release_carries_the_units_and_the_plugin_beside_the_runtime(instance, source,
                                                                      tmp_path):
    into = Path(staged(instance, source, tmp_path, runner=Fake())["staged"])
    assert (into / "deployment" / "compatibility.json").is_file()
    assert (into / "deployment" / "systemd" / "hermes-memory.service").is_file()
    assert (into / "integrations" / "hermes-memory" / "provider.py").is_file()
    assert not list(into.rglob("__pycache__"))


def test_the_manifest_says_which_pin_and_whose_decision_this_is(instance, source, tmp_path):
    into = Path(staged(instance, source, tmp_path, runner=Fake())["staged"])
    written = json.loads((into / MANIFEST).read_text(encoding="utf-8"))
    assert written["backend_spec"] == BACKEND_SPEC
    assert written["staged_by"] == OWNER
    assert written["review_digest"]


# -- what a release has to be ------------------------------------------------

def test_a_staged_tree_is_measured_against_what_the_units_would_start(instance, source,
                                                                     tmp_path):
    report = staged(instance, source, tmp_path, runner=Fake())
    assert report["complete"] is True, report["missing"]
    assert {Path(path).name for path in report["starts"]} == {
        "hermes-memory", "hindsight-api", "python"}


def test_a_tree_whose_runtime_binary_vanished_is_not_reported_as_a_release(instance, source,
                                                                          tmp_path):
    into = Path(staged(instance, source, tmp_path, runner=Fake())["staged"])
    (into / "bin" / "hermes-memory").unlink()
    reading = verify(settings=instance[1], into=into)
    assert reading["complete"] is False
    assert str(into / "bin" / "hermes-memory") in reading["missing"]


def test_a_tree_staged_without_the_backend_is_incomplete_for_a_configured_route(instance,
                                                                                source,
                                                                                tmp_path):
    report = staged(instance, source, tmp_path, runner=Fake(), backend=False)
    assert report["complete"] is False
    assert any("hindsight-api" in path for path in report["missing"])


def test_a_backend_only_request_stages_one_environment_and_says_so(instance, source,
                                                                   tmp_path):
    fake = Fake()
    report = staged(instance, source, tmp_path, runner=fake, backend=False)
    assert fake.mentioning(BACKEND_SPEC) == []
    assert not (Path(report["staged"]) / "hindsight").exists()
    assert any("will then refuse" in line for line in
               plan(settings=instance[1], into=tmp_path / "other", source=source,
                    backend=False)["actions"])


# -- a failure leaves nothing behind -----------------------------------------

@pytest.mark.parametrize("marker", ["uv build", "hindsight", "pip"])
def test_a_failure_half_way_leaves_no_tree_for_a_pointer_to_be_flipped_to(instance, source,
                                                                        tmp_path, marker):
    with pytest.raises(ReleaseError):
        staged(instance, source, tmp_path, runner=Fake(fail_on={marker}))
    assert not (tmp_path / "release").exists()


def test_a_wheel_build_that_produced_nothing_is_reported_as_a_failure(instance, source,
                                                                     tmp_path):
    class BuildLie(Fake):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            if argv[:2] == ["uv", "build"]:
                for path in Path(argv[argv.index("--out-dir") + 1]).glob("*.whl"):
                    path.unlink()
            return result

    with pytest.raises(ReleaseError, match="left 0 wheels"):
        staged(instance, source, tmp_path, runner=BuildLie())


def test_a_build_that_left_two_wheels_installs_neither_of_them(instance, source, tmp_path):
    """Choosing "the newest" is how a release ends up holding something else's bytes."""
    class TwoWheels(Fake):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            if argv[:2] == ["uv", "build"]:
                out = Path(argv[argv.index("--out-dir") + 1])
                (out / "hermes_memory-0.0.9-py3-none-any.whl").write_text("the old one\n")
            return result

    with pytest.raises(ReleaseError, match="left 2 wheels"):
        staged(instance, source, tmp_path, runner=TwoWheels())
    assert not (tmp_path / "release").exists(), "an ambiguous build leaves no release behind"


# -- which revision a release is ---------------------------------------------

def a_repository(source):
    """The source fixture as a real checkout standing at a real commit."""
    def git(*arguments):
        return subprocess.run(["git", "-c", "user.email=release@example.invalid",
                               "-c", "user.name=Release Test", *arguments],
                              cwd=source, check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    return commit_all(source)


def commit_all(source):
    """Commit whatever the tree holds now, and say which commit that made.

    The identity and signing flags are given rather than inherited: a test that wrote
    commits with the machine owner's key, or refused because their global config demands
    one, would be a function of the user rather than of this build.
    """
    def git(*arguments):
        return subprocess.run(["git", "-c", "user.email=release@example.invalid",
                               "-c", "user.name=Release Test", "-c", "commit.gpgsign=false",
                               *arguments],
                              cwd=source, check=True, capture_output=True, text=True).stdout

    git("add", "-A")
    git("commit", "-q", "--allow-empty", "-m", "staged as shown")
    return git("rev-parse", "HEAD").strip()


def test_an_unrevisioned_source_says_so_rather_than_being_refused(instance, source, tmp_path):
    report = plan(settings=instance[1], into=tmp_path / "release", source=source)
    assert report["source_commit"] == "" and report["blocking"] == [], \
        "a checkout that cannot be asked is reported as unasked, not as broken"


def test_a_tree_inside_someone_elses_repository_is_not_staged_as_that_one(instance, source,
                                                                         tmp_path):
    """The upward search is the trap: one directory up is not the revision in hand."""
    a_repository(tmp_path)
    report = plan(settings=instance[1], into=tmp_path / "release", source=source)
    assert report["source_commit"] == "" and report["blocking"] == []


def test_the_plan_names_the_commit_it_would_stage(instance, source, tmp_path):
    commit = a_repository(source)
    report = plan(settings=instance[1], into=tmp_path / "release", source=source)
    assert report["source_commit"] == commit
    assert any(f"commit {commit[:12]}" in line for line in report["actions"])


def test_the_record_says_which_commit_and_exactly_which_wheel_it_is(instance, source,
                                                                   tmp_path):
    commit = a_repository(source)
    report = staged(instance, source, tmp_path, runner=Fake())
    record = json.loads((Path(report["staged"]) / MANIFEST).read_text(encoding="utf-8"))
    assert record["source_commit"] == commit and record["source_dirty"] is False
    assert Path(record["wheel"]).parent == Path(report["staged"]) / "wheel", \
        "the artefact lives with the release that installed it"
    assert record["wheel_digest"] == content_digest(Path(record["wheel"]).read_bytes())


def test_a_wheel_named_by_hand_is_still_kept_with_the_release(instance, source, tmp_path):
    wheel = a_wheel(tmp_path)
    report = staged(instance, source, tmp_path, wheel=wheel, runner=Fake())
    record = json.loads((Path(report["staged"]) / MANIFEST).read_text(encoding="utf-8"))
    assert Path(record["wheel"]).parent == Path(report["staged"]) / "wheel"
    assert Path(record["wheel"]).read_bytes() == wheel.read_bytes(), \
        "the record names the bytes the two environments actually installed"


def test_a_dirty_checkout_is_not_staged_under_a_clean_commit_name(instance, source, tmp_path):
    a_repository(source)
    (source / "not_committed.py").write_text("x = 1\n", encoding="utf-8")
    report = plan(settings=instance[1], into=tmp_path / "release", source=source)
    assert report["source_dirty"] is True, "the reading says the tree differs before it refuses"
    assert any("uncommitted changes" in line for line in report["blocking"]), \
        "the wheel would be built from bytes no revision names"


def test_a_tree_named_for_another_commit_is_not_staged_there(instance, source, tmp_path):
    a_repository(source)
    report = plan(settings=instance[1], into=tmp_path / ("f" * 40), source=source)
    assert any("does not hold" in line for line in report["blocking"])


def test_a_different_revision_is_a_different_approval(instance, source, tmp_path):
    """An approval is for the revision that was shown, not for the directory named."""
    commit = a_repository(source)
    shown = plan(settings=instance[1], into=tmp_path / "release", source=source)
    (source / "later.py").write_text("y = 2\n", encoding="utf-8")
    assert commit_all(source) != commit, "a source change this build cannot see in its digests"
    moved = plan(settings=instance[1], into=tmp_path / "release", source=source)
    assert moved["review_digest"] != shown["review_digest"]
    with pytest.raises(ReleaseError, match="the plan changed"):
        apply(settings=instance[1], into=tmp_path / "release", source=source, actor=OWNER,
              review=shown["review_digest"], runner=Fake())


# -- the door itself ---------------------------------------------------------

def asked(*argv):
    """One CLI call, and the answer it printed. Nothing here reaches a package index."""
    import json

    from hermes_memory.cli import main
    out: list[str] = []
    patch = pytest.MonkeyPatch()
    patch.setattr("builtins.print", lambda *a, **k: out.append(a[0] if a else ""))
    try:
        code = main(list(argv))
    finally:
        patch.undo()
    return code, (json.loads(out[-1]) if out and out[-1].startswith("{") else out)


def written(*argv):
    import sys

    from hermes_memory.cli import main
    messages: list[str] = []
    patch = pytest.MonkeyPatch()
    patch.setattr("sys.stderr.write", messages.append)
    try:
        code = main(list(argv))
    except SystemExit:
        code = 2
    finally:
        patch.undo()
    return code, "".join(messages)


def test_the_door_plans_without_touching_a_disk_or_an_index(instance, source, tmp_path):
    code, report = asked("release", "--into", str(tmp_path / "release"),
                         "--source", str(source))
    assert code == 0
    assert report["review_digest"] and not (tmp_path / "release").exists()


def test_staging_without_the_digest_that_was_shown_is_refused_before_anything_runs(instance,
                                                                                  source,
                                                                                  tmp_path):
    code, message = written("release", "--into", str(tmp_path / "release"),
                            "--source", str(source), "--apply", "--actor", OWNER)
    assert code == 2 and "digest" in message
    assert not (tmp_path / "release").exists()


def test_an_approval_taken_for_one_tree_cannot_be_spent_on_another(instance, source, tmp_path):
    shown = plan(settings=instance[1], into=tmp_path / "one", source=source)
    code, message = written("release", "--into", str(tmp_path / "two"), "--source", str(source),
                            "--apply", "--actor", OWNER, "--review", shown["review_digest"])
    assert code == 2 and "plan changed" in message
