"""Which copy of the framework the host runs, and who gets told.

Hermes imports the plugin with the host's interpreter and its own `sys.path`. The plan forbids
installing the engine into that environment, so the release the installation points at has to be
loaded by path — and when it cannot be, the host has to be told which copy is answering, because
"memory behaves like an older build" is otherwise indistinguishable from a bug in this one.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

from plugin_loader import INTEGRATIONS

RUNTIME = INTEGRATIONS / "hermes-memory" / "runtime.py"
REVISION = "a" * 40


def loader():
    spec = importlib.util.spec_from_file_location("hm_runtime_under_test", RUNTIME)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def a_release(tmp_path, *, marker="from-the-release", revision=REVISION,
              python="python3.11", with_package=True):
    release = tmp_path / "runtime" / "releases" / revision
    site = release / "lib" / python / "site-packages"
    if with_package:
        package = site / "hermes_memory"
        (package / "storage").mkdir(parents=True)
        (package / "__init__.py").write_text(f"MARKER = {marker!r}\n", encoding="utf-8")
        (package / "storage" / "__init__.py").write_text("FROM_RELEASE = True\n", encoding="utf-8")
    else:
        site.mkdir(parents=True)
    (release / "RELEASE.json").write_text(json.dumps({"source_commit": revision}), encoding="utf-8")
    return release


# -- which release is this installation running? --------------------------------

def test_the_release_named_by_the_environment_wins(tmp_path):
    runtime, release = loader(), a_release(tmp_path)
    assert runtime.release_root({"HERMES_MEMORY_RELEASE": str(release),
                                 "HERMES_MEMORY_HOME": str(tmp_path)}) == release


def test_the_current_pointer_under_the_instance_home_is_the_fallback(tmp_path):
    runtime, release = loader(), a_release(tmp_path)
    current = tmp_path / "runtime" / "current"
    current.symlink_to(release, target_is_directory=True)
    assert runtime.release_root({"HERMES_MEMORY_HOME": str(tmp_path)}) == current
    assert runtime.package_directory(current).name == "hermes_memory"
    assert runtime.carried_revision(current) == REVISION, "the pointer still names a commit"


def test_an_installation_with_neither_pointer_is_answered_with_none(tmp_path):
    assert loader().release_root({"HERMES_MEMORY_HOME": str(tmp_path)}) is None


def test_a_release_built_on_another_python_is_still_found(tmp_path):
    release = a_release(tmp_path, python="python3.13")
    assert loader().package_directory(release) is not None


# -- adopting it ----------------------------------------------------------------

def test_the_release_is_loaded_by_path_in_a_clean_interpreter(tmp_path):
    """A subprocess, because this is about what one interpreter holds in `sys.modules`.

    The point is not that `hermes_memory` imports — it is that the *release's* copy imports, with
    its subpackages, ahead of any copy the host's own environment happens to hold.
    """
    release = a_release(tmp_path)
    script = (
        "import importlib.util, json\n"
        f"spec = importlib.util.spec_from_file_location('hm', {str(RUNTIME)!r})\n"
        "runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)\n"
        f"said = runtime.adopt({{'HERMES_MEMORY_RELEASE': {str(release)!r}}})\n"
        "import hermes_memory, hermes_memory.storage\n"
        "print(json.dumps({'said': said, 'marker': hermes_memory.MARKER, "
        "'subpackage': hermes_memory.storage.FROM_RELEASE, "
        "'path': list(hermes_memory.__path__)[0]}))\n")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          cwd=str(tmp_path), env={"PATH": "/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr
    said = json.loads(done.stdout)
    assert said["marker"] == "from-the-release"
    assert said["subpackage"] is True
    assert str(release) in said["path"], "submodules come from the release too"
    assert said["said"]["matches_release"] is True
    assert said["said"]["from_release"] is True
    assert said["said"]["revision"] == REVISION
    assert said["said"]["warning"] == ""


def test_a_copy_already_loaded_is_reported_and_left_alone(tmp_path):
    """Two module trees of one package in one process is worse than the copy that got there first."""
    import hermes_memory

    runtime, release = loader(), a_release(tmp_path)
    said = runtime.adopt({"HERMES_MEMORY_RELEASE": str(release)})
    assert said["matches_release"] is False
    assert "already loaded" in said["warning"]
    assert said["loaded_from"] == str(hermes_memory.__file__)
    assert sys.modules["hermes_memory"] is hermes_memory


def test_a_host_holding_its_own_copy_is_named_without_importing_it(tmp_path):
    """Neither in `sys.modules` nor from the release, so the copy Python would reach is the
    whole answer — and naming it must not import it, because an unavailable provider that
    imports the framework to explain itself has proved nothing.
    """
    release = a_release(tmp_path, with_package=False, revision="c" * 40)
    script = (
        "import importlib.util, json\n"
        f"spec = importlib.util.spec_from_file_location('hm', {str(RUNTIME)!r})\n"
        "runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)\n"
        f"said = runtime.adopt({{'HERMES_MEMORY_RELEASE': {str(release)!r}}})\n"
        "print(json.dumps(said))\n")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          cwd=str(tmp_path), env={"PATH": "/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr
    said = json.loads(done.stdout)
    assert said["loaded_from"], "the host is told which copy answers, not that something did"
    assert said["loaded_from"].endswith("hermes_memory/__init__.py")
    assert str(release) not in said["loaded_from"]
    assert said["warning"].startswith(f"{release} carries no hermes_memory package")


def test_a_release_carrying_no_package_says_so_rather_than_blaming_a_copy(tmp_path):
    """Isolated, because a host with no copy of its own is the case worth hearing plainly."""
    release = a_release(tmp_path, with_package=False, revision="b" * 40)
    script = (
        "import importlib.util, json\n"
        f"spec = importlib.util.spec_from_file_location('hm', {str(RUNTIME)!r})\n"
        "runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)\n"
        f"said = runtime.adopt({{'HERMES_MEMORY_RELEASE': {str(release)!r}}})\n"
        "print(json.dumps(said))\n")
    done = subprocess.run([sys.executable, "-I", "-S", "-c", script],
                          capture_output=True, text=True,
                          cwd=str(tmp_path), env={"PATH": "/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr
    said = json.loads(done.stdout)
    assert said["revision"] == "b" * 40
    assert said["warning"].startswith(f"{release} carries no hermes_memory package")
    assert said["loaded_from"] is None, "nothing answers, and the report says so"
    assert said["from_release"] is False


def test_no_release_at_all_is_said_loudly(tmp_path):
    said = loader().adopt({"HERMES_MEMORY_HOME": str(tmp_path)})
    assert said["release"] is None
    assert "runtime/current" in said["warning"]


def test_the_adoption_is_kept_for_whoever_asks_later(tmp_path):
    runtime, release = loader(), a_release(tmp_path)
    made = runtime.adopt({"HERMES_MEMORY_RELEASE": str(release)})
    assert runtime.report() == made, "the record a reader sees is the record adopt wrote"
    runtime.report()["release"] = "edited by a reader"
    assert runtime.report()["release"] == str(release), "a report is handed out as a copy"


# -- the order that makes it work -----------------------------------------------

def test_the_package_adopts_the_release_before_it_imports_the_framework():
    """`adopt()` has to run first, and a tidy import block is exactly how it stops running first."""
    text = (INTEGRATIONS / "hermes-memory" / "__init__.py").read_text(encoding="utf-8")
    assert text.index("RUNTIME = adopt()") < text.index("from .provider import")
