"""Which copy of the framework does the host run? The release's, or none.

Hermes imports this plugin with the host's own interpreter and the host's own `sys.path`, and
the plan forbids installing the engine into that environment. Left to plain import resolution the
host answers out of whichever copy happens to sit in its site-packages — which on a live
installation was a days-old snapshot, so switching `runtime/current` changed the CLI, the
services and the backend while the conversation kept running the old code. A release switch the
host does not obey is not a switch.

So the release this installation names is loaded, by path, before anything imports
`hermes_memory` — and only our package: the framework needs nothing outside the standard library,
so the host's own dependencies stay as the host left them. What was resolved is reported rather
than assumed, because the difference between "this release" and "some copy" is exactly what
somebody needs to see when memory behaves like an older build.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["adopt", "release_root", "report"]

_PACKAGE = "hermes_memory"
_REPORT: dict[str, Any] = {}


def release_root(environ: dict[str, str] | None = None) -> Path | None:
    """The release tree, named the way the service layout and the manifest name it."""
    source = os.environ if environ is None else environ
    named = str(source.get("HERMES_MEMORY_RELEASE") or "").strip()
    if named:
        return Path(named)
    home = Path(source.get("HERMES_MEMORY_HOME") or Path.home() / "data" / "hermes-memory")
    current = home / "runtime" / "current"
    return current if current.exists() else None


def package_directory(release: Path) -> Path | None:
    """Where ``hermes_memory`` lives inside a release venv, whatever Python version built it."""
    for site in sorted(release.glob("lib/python*/site-packages")):
        candidate = site / _PACKAGE
        if (candidate / "__init__.py").is_file():
            return candidate
    return None


def resolved_from() -> str | None:
    """The file ``hermes_memory`` would be imported from, without importing it."""
    origin = getattr(sys.modules.get(_PACKAGE), "__file__", None)
    if origin:
        return str(origin)
    try:
        spec = importlib.util.find_spec(_PACKAGE)
    except (ImportError, ValueError):
        return None
    return str(spec.origin) if spec and spec.origin else None


def carried_revision(release: Path) -> str | None:
    """The commit this release was cut at, from the record the release door left behind."""
    manifest = release / "RELEASE.json"
    if not manifest.is_file():
        return None
    try:
        return str(json.loads(manifest.read_text(encoding="utf-8")).get("source_commit") or "")
    except (OSError, ValueError):
        return None


def adopt(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Make the release's copy of the framework the one this process imports, and say if it did.

    A package already loaded is left alone: two `hermes_memory` trees in one interpreter means two
    classes of one name and an identity check that fails between them. Running the copy the host
    reached first is the lesser harm as long as the report names it.
    """
    report = _decide(environ)
    _REPORT.clear()
    _REPORT.update(report)
    return report


def report() -> dict[str, Any]:
    """What :func:`adopt` decided, for whoever asks the provider what it is running."""
    return dict(_REPORT)


def _decide(environ: dict[str, str] | None) -> dict[str, Any]:
    decided = {"release": None, "revision": None, "loaded_from": None,
               "from_release": False, "matches_release": False, "warning": ""}
    release = release_root(environ)
    if release is None:
        decided["loaded_from"] = resolved_from()
        decided["warning"] = (
            "no release is named by HERMES_MEMORY_RELEASE and there is no runtime/current under "
            "the instance home, so the host runs whatever copy its own environment holds: "
            f"{decided['loaded_from'] or 'nothing'}")
        return decided

    decided["release"] = str(release)
    decided["revision"] = carried_revision(release)
    if _PACKAGE in sys.modules:
        decided["loaded_from"] = resolved_from()
        decided["matches_release"] = _under(release, decided["loaded_from"])
        if not decided["matches_release"]:
            decided["warning"] = (
                f"hermes_memory was already loaded from {decided['loaded_from']} before this "
                f"plugin was imported, and that is not inside {decided['release']}; a release "
                "switch does not change what this process runs")
        return decided

    directory = package_directory(release)
    if directory is None:
        decided["loaded_from"] = resolved_from()
        decided["warning"] = (
            f"{decided['release']} carries no {_PACKAGE} package, so the host is running "
            f"{decided['loaded_from'] or 'nothing'} while the installation points at this release")
        return decided

    spec = importlib.util.spec_from_file_location(
        _PACKAGE, directory / "__init__.py", submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{directory} is not an importable package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = module
    spec.loader.exec_module(module)
    decided["loaded_from"] = resolved_from()
    decided["from_release"] = True
    decided["matches_release"] = _under(release, decided["loaded_from"])
    if not decided["matches_release"]:
        del sys.modules[_PACKAGE]
        raise RuntimeError(f"{_PACKAGE} resolved outside {decided['release']} after "
                          "being loaded from it")
    return decided


def _under(release: Path, origin: str | None) -> bool:
    if not origin:
        return False
    try:
        return Path(origin).resolve().is_relative_to(release.resolve())
    except OSError:
        return False
