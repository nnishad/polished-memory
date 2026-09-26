"""The three user services this installation owns, and nothing else.

A unit file is the most durable thing an installer writes: it outlives the shell that
created it, it runs with no interactive environment to fall back on, and a second
installation that regenerates it can quietly take over a process the owner is still
using. So this module is deliberately narrow. It renders from shipped templates, it
refuses to touch a unit it did not write, it reloads the user manager only when one of
its own files actually changed, and it can name every path a unit is allowed to see.

Nothing here runs ``sudo``, edits a system-wide unit, or turns on lingering. ``systemctl
--user`` against the owner's own manager is the entire privilege this has, and the
runner is injected so that every claim below is testable without a machine running
systemd.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..ids import content_digest, digest, now
from .profiles import InstallationError

__all__ = ["UNITS", "RUNTIME_UNIT", "BACKEND_UNIT", "WORKER_UNIT", "Layout", "layout",
           "render", "plan", "apply", "Services", "unit_directory", "wanted_units"]

# Kebab-case, the host convention, and the only names these functions will ever pass to
# systemctl. A unit outside this set is somebody else's service.
RUNTIME_UNIT = "hermes-memory.service"
BACKEND_UNIT = "hermes-memory-hindsight.service"
WORKER_UNIT = "hermes-memory-worker.service"
UNITS: tuple[str, ...] = (RUNTIME_UNIT, BACKEND_UNIT, WORKER_UNIT)

RECORD_FILENAME = "services.json"
PLAN_VERSION = "service-plan-v1"
_TEMPLATE_TOKEN = re.compile(r"@([A-Z][A-Z0-9_]*)@")
_ABSOLUTE = re.compile(r"(?:^|\s)(/\S+)")

# Every line in a template that names a location. A token that is never answered would
# otherwise install a unit pointing at a directory nobody chose.
LOCATION_KEYS = frozenset({"WorkingDirectory", "EnvironmentFile", "ExecStart", "ExecReload",
                           "ReadWritePaths", "StateDirectory", "CacheDirectory",
                           "LogsDirectory", "BindPaths"})


@dataclass(frozen=True)
class Layout:
    """Where the pieces of this installation actually are."""

    release: Path
    instance_home: Path
    data_dir: Path
    env_file: Path
    hindsight_dir: Path
    pg0_dir: Path
    hindsight_env: Path
    templates: Path

    @property
    def tokens(self) -> dict[str, str]:
        return {"RELEASE": str(self.release), "INSTANCE_HOME": str(self.instance_home),
                "DATA_DIR": str(self.data_dir), "ENV_FILE": str(self.env_file),
                "HINDSIGHT_DATA_DIR": str(self.hindsight_dir),
                "PG0_DIR": str(self.pg0_dir), "HINDSIGHT_ENV": str(self.hindsight_env)}

    @property
    def roots(self) -> tuple[str, ...]:
        """The only prefixes a rendered unit may name.

        ``/bin`` is here for the one fixed executable the reload line uses; it is a
        program path, not somewhere a service can keep data.
        """
        return str(self.release), str(self.instance_home), "/bin"


def template_root(*, environ: dict[str, str] | None = None) -> Path:
    """The shipped template directory: the release tree first, this checkout second.

    A wheel has no ``deployment/`` beside it, so the release pointer is asked first and
    the source tree is the fallback rather than the assumption.
    """
    environment = dict(os.environ if environ is None else environ)
    release = environment.get("HERMES_MEMORY_RELEASE")
    candidates = []
    if release:
        candidates.append(Path(release) / "deployment" / "systemd")
    candidates.append(Path(__file__).resolve().parents[3] / "deployment" / "systemd")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise InstallationError("no service templates found. Looked in: "
                            + ", ".join(str(path) for path in candidates)
                            + ". Point HERMES_MEMORY_RELEASE at an unpacked release, "
                              "which ships them.")


def layout(settings, *, environ: dict[str, str] | None = None,
           templates: str | Path | None = None) -> Layout:
    """The installation's own paths.

    ``HERMES_MEMORY_RELEASE`` is read from the process environment because it names
    which code is running, which is a fact about the process. Where the installation
    keeps its state is read from the configuration instead, because that is a fact about
    the installation and not about the shell that happens to be running the report.
    """
    home = Path(settings.home).expanduser()
    environment = dict(os.environ if environ is None else environ)
    release = environment.get("HERMES_MEMORY_RELEASE")
    return Layout(release=Path(release).expanduser() if release else home / "runtime" / "current",
                  instance_home=home,
                  data_dir=Path(settings.data_dir),
                  env_file=home / "hermes-memory.env",
                  hindsight_dir=home / "hindsight",
                  pg0_dir=home / "pg0",
                  hindsight_env=home / "hindsight.env",
                  templates=Path(templates) if templates
                  else template_root(environ=environment))


def unit_directory(*, environ: dict[str, str] | None = None) -> Path:
    """The owner's own user unit directory; never a system one.

    ``$XDG_CONFIG_HOME`` is honoured because that is where the user manager actually
    looks, and a unit written anywhere else is a file that silently does nothing.
    """
    environment = os.environ if environ is None else environ
    base = environment.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "systemd" / "user"


def wanted_units(settings) -> tuple[str, ...]:
    """Which units this installation would start, in the order they may be started.

    The backend and its worker exist only once a Hindsight endpoint is configured. An
    installation in capture-only mode has no reason to run a process with nothing to
    dispatch, and offering to start one would claim a service that is not wanted.
    """
    if not settings.hindsight_url:
        return (RUNTIME_UNIT,)
    return UNITS


def render(settings, *, environ: dict[str, str] | None = None,
           templates: str | Path | None = None) -> dict[str, str]:
    """The unit text this installation would write, keyed by unit name."""
    placed = layout(settings, environ=environ, templates=templates)
    tokens = placed.tokens
    rendered: dict[str, str] = {}
    for name in wanted_units(settings):
        path = placed.templates / name
        if not path.is_file():
            raise InstallationError(f"the release ships no template for {name}")
        text = path.read_text(encoding="utf-8")
        unknown = sorted(set(_TEMPLATE_TOKEN.findall(text)) - set(tokens))
        if unknown:
            raise InstallationError(
                f"{name} uses tokens this installation cannot answer: {', '.join(unknown)}")
        for token, value in tokens.items():
            text = text.replace(f"@{token}@", value)
        _check_locations(name, text, placed)
        rendered[name] = text
    return rendered


def _check_locations(name: str, text: str, placed: Layout) -> None:
    """No unit this installer writes may reach outside the installation.

    A template edit that reintroduced an absolute home path would otherwise be written
    into a process that runs for months with whatever permissions that directory has.
    """
    for line in text.splitlines():
        key, separator, value = line.strip().partition("=")
        if not separator or key.strip() not in LOCATION_KEYS or not value.startswith("/"):
            continue
        for candidate in _ABSOLUTE.findall(value):
            if not candidate.startswith(placed.roots):
                raise InstallationError(
                    f"{name} would let a service touch {candidate}, which is outside "
                    + " or ".join(placed.roots))


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def record(*, unit_dir: str | Path) -> dict[str, Any]:
    """What this installation last wrote.

    The record is not evidence of the present — the file beside it is — only of the
    answer to "did we write this, or did somebody else?".
    """
    raw = _read(Path(unit_dir) / RECORD_FILENAME)
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"unreadable": True}
    return parsed if isinstance(parsed, dict) else {"unreadable": True}


def _write_record(values: dict[str, Any], *, unit_dir: Path) -> None:
    unit_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    path = unit_dir / RECORD_FILENAME
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def plan(settings, *, environ: dict[str, str] | None = None,
         unit_dir: str | Path | None = None,
         templates: str | Path | None = None) -> dict[str, Any]:
    """What would change, stated before anything changes.

    The distinction that matters is between a file we wrote and a file we did not. A
    unit we own that we would now render differently is ours to rewrite; a unit already
    there under somebody else's name is theirs, and this reports the collision instead
    of overwriting it.
    """
    target = Path(unit_dir or unit_directory(environ=environ))
    previous = record(unit_dir=target)
    owned = previous.get("units") if isinstance(previous.get("units"), dict) else {}
    rendered = render(settings, environ=environ, templates=templates)
    entries: list[dict[str, Any]] = []
    for name in UNITS:
        path = target / name
        current = _read(path)
        on_disk = None if current is None else content_digest(current.encode())
        ours = on_disk is not None and owned.get(name) == on_disk
        wanted = rendered.get(name)
        if wanted is None:
            state = "not-wanted" if current is None else ("removed" if ours else "left-alone")
        elif on_disk is None:
            state = "written"
        elif on_disk == content_digest(wanted.encode()):
            # Identical to what we would write. Ours to keep either way, but there is
            # nothing to change and no reason to pretend the manager needs a reload.
            state = "unchanged" if ours else "adopted"
        else:
            state = "rewritten" if ours else "collision"
        entries.append({"unit": name, "path": str(path), "state": state, "ours": ours,
                        "present": current is not None})
    changing = [entry["unit"] for entry in entries
                if entry["state"] in ("written", "rewritten", "removed")]
    proposal = {"unit_dir": str(target),
                "changes": sorted(f"{entry['unit']}:{entry['state']}" for entry in entries
                                  if entry["unit"] in changing)}
    return {"unit_dir": str(target), "units": entries, "would_change": changing,
            "reload_needed": bool(changing),
            "blocked": [entry["unit"] for entry in entries if entry["state"] == "collision"],
            "start_order": list(wanted_units(settings)),
            "review_digest": digest([PLAN_VERSION,
                                     {key: proposal[key] for key in ("unit_dir", "changes")}])}


def apply(settings, *, actor: str, review: str, environ: dict[str, str] | None = None,
          unit_dir: str | Path | None = None) -> dict[str, Any]:
    """Write the plan that was shown, for the owner who approved that exact plan.

    The digest has to match, because a plan regenerated after the unit directory moved
    is a different plan. A collision stops the whole call: half a service topology is
    worse than none, since the units name each other.
    """
    proposal = plan(settings, environ=environ, unit_dir=unit_dir)
    if proposal["blocked"]:
        raise InstallationError(
            "these units already exist and were not written by this installation, so they "
            "are not ours to overwrite: " + ", ".join(proposal["blocked"]))
    if review != proposal["review_digest"]:
        raise InstallationError(
            "the review digest does not match what would change now. Run "
            "`hermes-memory services` again and approve the plan it prints.")
    if not isinstance(actor, str) or not actor.strip():
        raise InstallationError("an actor must be named for the record of who changed this")
    target = Path(proposal["unit_dir"])
    target.mkdir(parents=True, mode=0o700, exist_ok=True)
    rendered = render(settings, environ=environ)
    written, removed = [], []
    for entry in proposal["units"]:
        path = Path(entry["path"])
        if entry["state"] in ("written", "rewritten"):
            path.write_text(rendered[entry["unit"]], encoding="utf-8")
            # 0644 inside a 0700 directory: the user manager has to read it, and nobody
            # outside the account can reach the directory to try.
            path.chmod(0o644)
            written.append(entry["unit"])
        elif entry["state"] == "removed":
            path.unlink()
            removed.append(entry["unit"])
    _write_record({"units": {name: content_digest(text.encode())
                             for name, text in rendered.items()},
                   "unit_dir": str(target), "written_by": actor.strip(), "at": now()},
                  unit_dir=target)
    return {"ok": True, "units_written": written, "units_removed": removed,
            "reload_needed": bool(written or removed), "actor": actor.strip(),
            "review_digest": proposal["review_digest"], "unit_dir": str(target),
            "at": now()}


def subprocess_runner(argv: Sequence[str]) -> tuple[int, str]:
    """The only real way to run a host command from here; tests pass their own."""
    completed = subprocess.run(list(argv), capture_output=True, text=True, timeout=120)
    return completed.returncode, completed.stdout


class Services:
    """Start, stop and ask about the units this installation owns.

    ``runner`` is the host command executor, injected so the ordering, the permission
    ceiling and the refusal to name a foreign unit are all claims a test can check.
    """

    def __init__(self, settings, *, environ: dict[str, str] | None = None,
                 unit_dir: str | Path | None = None,
                 runner: Callable[[Sequence[str]], tuple[int, str]] | None = None) -> None:
        self.settings = settings
        self.unit_dir = Path(unit_dir or unit_directory(environ=environ))
        self.runner = runner or subprocess_runner
        self.order = list(wanted_units(settings))

    # -- the ceiling ---------------------------------------------------------

    def _run(self, arguments: Sequence[str]) -> tuple[int, str]:
        argv = ["systemctl", "--user", *[str(item) for item in arguments]]
        for forbidden in ("sudo", "--system", "--global", "enable-linger",
                          "disable-linger", "link", "mask", "reboot", "poweroff", "halt"):
            if forbidden in argv:
                raise InstallationError(
                    f"`{forbidden}` is not something this installer does; user units only")
        foreign = sorted({item for item in argv if item.endswith(".service")} - set(UNITS))
        if foreign:
            raise InstallationError(
                "these are not this installation's units: " + ", ".join(foreign))
        code, stdout = self.runner(argv)
        if code != 0:
            raise InstallationError(
                f"`{' '.join(argv)}` failed ({code}): {stdout.strip()[:300]}")
        return code, stdout

    # -- the operations ------------------------------------------------------

    def daemon_reload(self) -> bool:
        """Tell the manager the files changed. Only ever called when they did."""
        self._run(["daemon-reload"])
        return True

    def start(self) -> list[str]:
        """Runtime first, then the backend, then the worker. Never resumes a pause.

        A paused fence is the owner's decision and outlives a restart on purpose; a
        start that unpaused anything would turn "put this on hold" into "the machine
        rebooted, so we changed our mind".
        """
        for unit in self.order:
            self._run(["start", unit])
        return list(self.order)

    def stop(self, *, pause: Callable[[], Any] | None = None) -> list[str]:
        """Persist the pause before the last process goes away.

        The order is the shutdown contract: stop submitting, then stop the submitters.
        A worker killed mid-retain with no recorded pause looks like an idle
        installation and resumes formation on the next boot.
        """
        if pause is not None:
            pause()
        stopped = []
        for unit in reversed(self.order):
            self._run(["stop", unit])
            stopped.append(unit)
        return stopped

    def restart(self, unit: str) -> str:
        if unit not in UNITS:
            raise InstallationError(f"{unit} is not one of this installation's units")
        self._run(["restart", unit])
        return unit

    def autostart(self, *, enable: bool = True, start_now: bool = False) -> list[str]:
        """Autostart, offered separately from start and never decided by a start.

        ``start_now`` is the one place a unit may be started as a side effect, and only
        because the owner asked for it in the same breath.
        """
        verb = "enable" if enable else "disable"
        order = self.order if enable else list(reversed(self.order))
        for unit in order:
            self._run([verb, *(["--now"] if start_now else []), unit])
        return list(order)

    def status(self) -> dict[str, Any]:
        """What the manager says, read rather than inferred from a pid file."""
        if not self.order:
            return {"units": {}, "expected": [], "installed": {}}
        _, stdout = self._run(["--no-legend", "show", *self.order,
                               "-p", "Id", "-p", "LoadState", "-p", "ActiveState",
                               "-p", "SubState", "-p", "UnitFileState"])
        units = _parse_show(stdout)
        return {"units": {name: units.get(name, {}) for name in self.order},
                "expected": list(self.order),
                "installed": {name: (self.unit_dir / name).is_file() for name in UNITS}}


def _parse_show(stdout: str) -> dict[str, dict[str, str]]:
    """``systemctl show`` output is one block per unit, keys in order."""
    units: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in stdout.splitlines():
        if not line.strip():
            current = None
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        if key == "Id":
            current = {}
            units[value] = current
        if current is not None:
            current[key] = value
    return units
