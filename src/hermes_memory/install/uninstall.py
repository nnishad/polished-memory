"""Removing this installation without removing anybody's memory (§10.7).

The default uninstall is data-preserving, and that is the whole point of it. What leaves
is the plugin registration, the owned unit files and this installation's own record of
having written them. What stays is every byte of evidence, the erasure ledger that proves
what was forgotten, the logs a recovery needs, and the backups — because an uninstall is
a change of software, and the owner's memory outlives software.

So the two questions this module asks about every file are "did we write this" and "is it
still what we wrote". A file answering yes to both is deleted. A file we wrote that
somebody edited since is reported and left alone: destroying a hand-edited unit because
its name is familiar is the exact act this design exists to refuse. A file we never wrote
is not ours either way.

Purging evidence is a different operation, with a different name and a displayed
manifest, and there is deliberately no flag here that reaches it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import DEFAULT_ENV_FILENAME
from ..ids import digest, now
from .inventory import provider_selection
from .profiles import STATE_FILENAME
from .services import RECORD_FILENAME, plan as service_plan, record as service_record, \
    unit_directory

__all__ = ["uninstall_plan", "uninstall_apply", "UninstallError",
           "PROVIDER_RECEIPT", "record_provider_selection"]


class UninstallError(ValueError):
    """The owner asked for something this installation will not do."""


PLAN_VERSION = "uninstall-plan-v1"
PROVIDER_RECEIPT = "provider-selection.json"
PLUGIN = "hermes-memory"
# What the host's narrow writer will accept as a provider name. The value is passed as
# one argv element to the host's own writer, so this is not shell safety — it is the
# refusal to write a nonsense selection read out of a receipt file.
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
# The scanner's word for "I read the file and there was no plain selection in it".
_ABSENT = ("unknown", "unset")
# Its words for "I could not tell". Neither may be acted on by writing something.
_UNREADABLE = ("unreadable", "no config.yaml to read")


def uninstall_plan(settings, *, hermes_home: str | Path | None = None,
                   unit_dir: str | Path | None = None,
                   environ: dict[str, str] | None = None) -> dict[str, Any]:
    """What would be deleted, what is refused, and what is kept because it must be.

    Nothing is written here. The digest it returns is the one ``uninstall_apply`` needs,
    so the list an operator reads is the list that gets acted on.
    """
    target = Path(unit_dir or unit_directory(environ=environ))
    units = service_plan(settings, environ=environ, unit_dir=target)
    owned = service_record(unit_dir=target).get("units")
    owned = owned if isinstance(owned, dict) else {}
    removals, modified, refused = [], [], []
    for entry in units["units"]:
        path = Path(entry["path"])
        if not path.is_file():
            continue
        if entry["ours"]:
            removals.append({"path": str(path), "unit": entry["unit"],
                             "why": "written by this installation and unchanged since"})
        elif entry["unit"] in owned:
            modified.append({"path": str(path), "unit": entry["unit"],
                             "why": "written by this installation and changed since, so it "
                                    "is reported rather than destroyed"})
        else:
            refused.append({"path": str(path), "unit": entry["unit"],
                            "why": "not written by this installation"})
    record_file = target / RECORD_FILENAME
    provider = _provider_state(settings, hermes_home=hermes_home, environ=environ)

    home = Path(settings.home)
    kept = []
    for path, why in ((home / STATE_FILENAME,
                       "the instance ledger is the map to every profile's memory"),
                      (Path(settings.data_dir),
                       "the evidence itself, one store per enrolled profile"),
                      (home / DEFAULT_ENV_FILENAME,
                       "the configuration the owner wrote, which no uninstall owns")):
        if path.exists():
            kept.append({"path": str(path), "why": why})
    for name in provider["snapshots"]:
        kept.append({"path": str(home / name),
                     "why": "a snapshot of the host configuration taken before this "
                            "installation changed it"})

    proposal = {"unit_dir": str(target),
                "removals": sorted(item["path"] for item in removals),
                "record": str(record_file) if record_file.is_file() else None,
                "provider_restore": provider["command"]}
    return {"unit_dir": str(target), "removals": removals, "modified": modified,
            "refused": refused,
            "record": str(record_file) if record_file.is_file() else None,
            "retained": kept + _retained_backups(settings),
            "provider": provider,
            "host_commands": _host_commands(provider),
            "jobs": {"owned": [], "note": "this installation registered no scheduled job, "
                                          "so there is none for it to unregister"},
            "purging_data": False,
            "review_digest": digest([PLAN_VERSION, proposal]),
            "next": f"hermes-memory uninstall --keep-data --review "
                    f"{digest([PLAN_VERSION, proposal])} --actor <owner-principal>"}


def uninstall_apply(settings, *, actor: str, review: str, keep_data: bool,
                    hermes_home: str | Path | None = None,
                    unit_dir: str | Path | None = None,
                    environ: dict[str, str] | None = None,
                    runner: Callable[[Sequence[str]], tuple[int, str]] | None = None
                    ) -> dict[str, Any]:
    """Delete the verified-owned artefacts and nothing else.

    ``keep_data`` is a required argument rather than a default so that the call reading
    like an uninstall cannot secretly be a purge.
    """
    if not keep_data:
        raise UninstallError(
            "uninstall keeps the evidence, the erasure ledger and the backups; there is "
            "no flag here that removes them, and there will not be one: purging is a "
            "separate owner operation with a manifest of what it will destroy")
    if not isinstance(actor, str) or not actor.strip():
        raise UninstallError("an actor must be named for the record of who removed this")
    if runner is None:
        # Deleting the units while the host still has the plugin registered and the
        # provider selected is half an uninstall, which is worse than none: the memory
        # reads as configured and every call to it then fails.
        raise UninstallError(
            "no host command executor was provided, so the plugin registration could be "
            "removed and this installation's units left behind, or the reverse")
    proposal = uninstall_plan(settings, hermes_home=hermes_home, unit_dir=unit_dir,
                              environ=environ)
    if review != proposal["review_digest"]:
        raise UninstallError(
            "the review digest does not match what would be removed now. Run "
            "`hermes-memory uninstall --keep-data` again and approve the list it prints.")
    if proposal["provider"]["restore_unsupported"]:
        raise UninstallError("the provider selection cannot be restored, so it must not be "
                             "left pointing at a plugin about to be removed: "
                             + proposal["provider"]["restore_unsupported"])
    removed = []
    # The host commands go first: every one of them is reversible by the owner running
    # setup again, while a deleted unit file is not. Failing here therefore leaves the
    # machine untouched rather than half-removed.
    for argv in proposal["host_commands"]:
        code, output = runner(argv)
        if code != 0:
            raise UninstallError(
                f"`{' '.join(str(part) for part in argv)}` failed ({code}): "
                f"{output.strip()[:300]}. Nothing was removed.")
    for item in proposal["removals"]:
        Path(item["path"]).unlink()
        removed.append(item["path"])
    if proposal["record"]:
        Path(proposal["record"]).unlink()
        removed.append(proposal["record"])
    return {"ok": True, "removed": removed,
            "left_behind": [item["path"] for item in proposal["modified"]],
            "refused": [item["path"] for item in proposal["refused"]],
            "kept": [item["path"] for item in proposal["retained"]],
            "host_commands": [" ".join(str(part) for part in argv)
                              for argv in proposal["host_commands"]],
            "actor": actor.strip(), "purging_data": False,
            "next": "the profile data, backups and instance ledger are still on disk, and "
                    "`hermes-memory status` will report them as unconfigured until they "
                    "are enrolled again"}


def record_provider_selection(*, hermes_home: str | Path, settings, prior: str,
                              actor: str) -> dict[str, Any]:
    """Keep the receipt that makes §10.7's conditional restore possible.

    Written once, before the first change: an activation that ran twice would otherwise
    record the provider it wrote as the provider it replaced, and an uninstall would
    "restore" this plugin. That is the reason a second call leaves the first answer
    standing rather than refreshing it.
    """
    home = Path(settings.home)
    path = home / PROVIDER_RECEIPT
    config = Path(hermes_home) / "config.yaml"
    if path.is_file():
        existing = _read(path)
        if isinstance(existing, dict) and existing.get("config") == str(config):
            return {"receipt": existing, "written": False, "path": str(path)}
    value = {"config": str(config), "prior": prior, "written": PLUGIN,
             "actor": actor.strip(), "at": now()}
    home.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return {"receipt": value, "written": True, "path": str(path)}


def _provider_state(settings, *, hermes_home: str | Path | None,
                    environ: dict[str, str] | None) -> dict[str, Any]:
    """Whether the current selection is still the one this installation wrote.

    The condition matters more than the command. An owner who chose a different provider
    after setup is not an accident to be undone, and the only honest reading of a receipt
    whose value no longer matches the file is "leave what they picked alone".
    """
    home = Path(settings.home)
    root = Path(hermes_home) if hermes_home else Path(
        (environ or {}).get("HERMES_HOME") or Path.home() / ".hermes")
    config = root / "config.yaml"
    raw = _read(home / PROVIDER_RECEIPT)
    receipt = raw if isinstance(raw, dict) else {}
    snapshots = sorted(item.name for item in home.glob("config.yaml.before-*")) \
        if home.is_dir() else []
    current = provider_selection(config) if config.is_file() else "no config.yaml to read"
    state = {"hermes_home": str(root), "config": str(config), "current": current,
             "receipt": receipt, "snapshots": snapshots, "prior": receipt.get("prior"),
             "command": None, "restore_unsupported": None}
    if not receipt:
        state["why"] = "no activation receipt exists for this installation, so there is no " \
                       "prior selection known to restore"
        return state
    if receipt.get("config") != str(config):
        state["why"] = f"the receipt describes {receipt.get('config')}, not {config}"
        return state
    if current != receipt.get("written"):
        state["why"] = f"the selection now reads {current!r}, which is not what this " \
                       "installation wrote, so the owner's choice is left alone"
        return state
    prior = receipt.get("prior")
    if isinstance(prior, str) and prior in _ABSENT:
        # The usual case, and worth stating: a fresh host has no memory.provider at all,
        # so the selection to restore is the absence of one. Taking our own key back is
        # what "prior" means here; writing the word "unknown" into the file is not.
        state["command"] = ["hermes", "config", "unset", "memory.provider"]
        state["why"] = "the selection this installation replaced did not exist, so the key " \
                       "this installation wrote is removed rather than renamed"
        return state
    if not isinstance(prior, str) or prior in _UNREADABLE or not _SAFE_NAME.fullmatch(prior):
        # The one branch that stops the removal. The selection is ours to give back and
        # there is no trustworthy thing to give it to: the receipt could not read the
        # file when it wrote it, so nothing here knows what the owner would lose.
        state["restore_unsupported"] = (
            f"the selection still reads {PLUGIN!r} but the recorded prior value "
            f"({prior!r}) cannot be written back; set memory.provider by hand first")
        return state
    state["command"] = ["hermes", "config", "set", "memory.provider", prior]
    state["why"] = "the selection still reads what setup wrote, so the receipt's value " \
                   "goes back"
    return state


def _host_commands(provider: dict[str, Any]) -> list[list[str]]:
    """Restore the selection first: the host should never be left naming a provider it
    has just had uninstalled."""
    commands = []
    if provider["command"]:
        commands.append(list(provider["command"]))
    commands.append(["hermes", "plugins", "disable", PLUGIN])
    commands.append(["hermes", "plugins", "remove", PLUGIN])
    return commands


def _retained_backups(settings) -> list[dict[str, str]]:
    from .upgrade import profile_stores

    found = []
    for profile, path in profile_stores(settings):
        directory = path.parent / "snapshots"
        if directory.is_dir():
            found.append({"path": str(directory),
                          "why": f"every backup of {profile}'s memory"})
    return found


def _read(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
