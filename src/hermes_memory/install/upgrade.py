"""What switching this installation to another release would involve (§10.6).

Read only, and deliberately weaker than it looks. §10.6 makes the switch a quiesced,
backed-up, rehearsed, validated operation: pause formation and delivery, reconcile the
accepted work, take coherent backups, run migrations and the contract tests against an
isolated restored copy, and only then move the release pointer and restart owned
components. A module that performed half of that while answering "what would change"
would be the exact failure that design exists to prevent, so this one writes nothing,
stops nothing and migrates nothing. What it produces is the list of things that are in
the way, in the order an operator would deal with them.

Each question it answers is one the retired installation learned the hard way: an upgrade
that clobbered a hand-edited unit, a schema a reverted binary could no longer read, and a
"recent backup" that turned out to be a document left open in another folder.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ..ids import content_digest, digest
from .inventory import survey
from .profiles import InstallationError, ProfileRegistry
from .services import plan as service_plan

__all__ = ["upgrade_plan", "UpgradeError", "BACKUP_AGE_DAYS",
           "profile_stores"]


class UpgradeError(ValueError):
    """The owner asked for something this installation will not do."""


PLAN_VERSION = "upgrade-plan-v1"
# How stale the newest backup may be before an upgrade says so. Not a promise about
# recovery time, only a refusal to call an untested machine safe to switch under.
BACKUP_AGE_DAYS = 7
# Side-by-side staging needs room for two trees plus a rehearsal copy of the store.
MINIMUM_FREE_BYTES = 5_000_000_000
RELEASE_EXECUTABLE = ("bin", "hermes-memory")


def upgrade_plan(settings, *, release: str | Path | None = None,
                 hermes_home: str | Path | None = None,
                 environ: dict[str, str] | None = None,
                 at: float | None = None) -> dict[str, Any]:
    """The two releases, what stands between them, and what a switch would touch.

    ``release`` names the staged tree to switch *to*; it is never defaulted, because
    "upgrade" without a target is a hope about the current one. ``at`` is the moment the
    backup age is measured from, so a plan can be shown to be the same plan tomorrow.
    """
    from time import time

    if release is not None and not isinstance(release, (str, Path)):
        raise UpgradeError("release must be a path to a staged tree, or nothing")
    moment = float(at) if at is not None else time()
    target = Path(release).expanduser() if release else None
    current = Path((environ or {}).get("HERMES_MEMORY_RELEASE")
                   or Path(settings.home) / "runtime" / "current")
    report = survey(settings, hermes_home=hermes_home, environ=environ, proc=None)
    units = service_plan(settings, environ=environ)
    schema = _schema_state(settings)
    profiles = profile_stores(settings)
    occupied = _in_flight(profiles)
    backups = _backups(profiles, moment=moment)

    blocking: list[str] = []
    advisory: list[str] = []
    if target is None:
        blocking.append("no target release was named; an upgrade is a switch between two "
                        "known trees, not a hope about the current one")
    staged = _check_target(target, blocking)
    same = staged["staged"] and target.resolve() == current.resolve()
    if same:
        advisory.append("the named release is already the one this installation runs, so "
                        "switching to it would be a restart wearing a better word")
    if schema["state"] == "newer":
        blocking.append(f"the store is at schema {schema['now']} and this build only knows "
                        f"{schema['head']}; §10.6 forbids running an older binary against a "
                        "newer schema, so code rollback is not available here")
    elif schema["state"] == "behind":
        advisory.append(f"{schema['head'] - schema['now']} migration(s) are waiting; they "
                        "run against a rehearsal copy first, not against the live store")
    elif schema["state"] == "unreadable":
        blocking.append(f"{schema['store']} could not be read ({schema['detail']}), so no "
                        "claim about its schema can be made")
    if occupied["held"]:
        blocking.append(f"{', '.join(occupied['held'])} hold a gate slot right now; §10.6 "
                        "reconciles active work before switching code")
    if occupied["unreadable"]:
        advisory.append(f"{occupied['unreadable']} store(s) could not be asked about "
                        "in-flight work, so none is claimed to be absent there")
    if backups["missing"]:
        blocking.append("there is no backup to fall back to for "
                        + ", ".join(backups["missing"])
                        + "; a bad switch would have no way back, so run "
                        "`hermes-memory backup` first")
    elif backups["worst_age_days"] is not None and \
            backups["worst_age_days"] > BACKUP_AGE_DAYS:
        advisory.append(f"the newest backup for {backups['worst_profile']} is "
                        f"{backups['worst_age_days']} day(s) old, past the "
                        f"{BACKUP_AGE_DAYS} this plan will sign for")
    if units["blocked"]:
        blocking.append(f"{', '.join(units['blocked'])} exist and were not written by this "
                        "installation, so an upgrade cannot rewrite them")
    elif units["would_change"]:
        advisory.append("a switch would rewrite: " + ", ".join(units["would_change"]))
    free = report["disk"]["free_bytes"]
    if free is not None and free < MINIMUM_FREE_BYTES:
        advisory.append(f"{free / 1e9:.1f}GB free beside the data directory; a side-by-side "
                        "stage needs room for two trees and a rehearsal copy")
    elif free is None:
        advisory.append("free disk space could not be read, so room for the second tree is "
                        "unproven")
    provider = report["host"]["memory_provider"]
    if provider != "hermes-memory":
        advisory.append(f"the host's memory provider reads {provider!r}, so this "
                        "installation is not the one a switch would be restarting")
    current_plugin = _plugin_digest(_source_plugin_files())
    if staged["plugin_digest"] == current_plugin:
        advisory.append("the plugin files in the two trees are byte-identical, so the "
                        "host's registration would not need repinning")

    proposal = {"current_release": str(current), "target_release": str(target) if target
                else None, "target": staged,
                "units": sorted(f"{entry['unit']}:{entry['state']}"
                                for entry in units["units"]),
                "schema": {key: schema[key] for key in ("state", "now", "head")},
                "held": occupied["held"],
                "plugin_digest": current_plugin,
                "backups": {key: value for key, value in sorted(backups["per_profile"].items())}}
    return {"current_release": str(current),
            "target_release": str(target) if target else None,
            "target": staged,
            "revisions": {"build": report["release"]["framework_version"],
                          "package_source": report["release"]["package_source"],
                          "python": report["release"]["python"],
                          "hermes_home": report["host"]["hermes_home"],
                          "host": report["host"]["hermes_version"],
                          "provider": provider,
                          "plugin_files": (current_plugin or "")[:12] or None,
                          "target_plugin_files": (staged["plugin_digest"] or "")[:12] or None},
            "switches": _switches(current, target, units, same=same),
            "kept": _kept(settings),
            "not_performed": _NOT_PERFORMED,
            "blocking": blocking, "advisory": advisory,
            "schema": schema, "in_flight": occupied, "backups": backups,
            "disk": {"free_bytes": free, "measured_against": report["disk"]["measured_against"]},
            "units": units["units"],
            "review_digest": digest([PLAN_VERSION, proposal]),
            "note": "this plan writes, stops, migrates and switches nothing; §10.6's "
                    "switch needs a rehearsal on a restored copy and a validated final "
                    "state, and this framework has no operation that performs them"}


# Stated as part of the plan rather than left to the reader: an upgrade report that
# lists what it checked is also a claim about what it did not do.
_NOT_PERFORMED = [
    "formation and delivery are still running: nothing here paused them",
    "no backup was taken, only the existing ones were read",
    "no schema was migrated, and no rehearsal ran on a restored copy",
    "the release pointer and the owned units were not rewritten, and nothing restarted",
    "the previous release is not retained by this plan, so it cannot be reverted to by it",
]


def _check_target(target: Path | None, blocking: list[str]) -> dict[str, Any]:
    """Is the named tree a release this installation could run?"""
    if target is None:
        return {"path": None, "staged": False, "executable": None, "runnable": False,
                "plugin_files": 0, "plugin_digest": None}
    executable = target.joinpath(*RELEASE_EXECUTABLE)
    if not target.is_dir():
        blocking.append(f"{target} is not a directory; §10.3 stages a release tree and "
                        "nothing here downloads or unpacks one")
        return {"path": str(target), "staged": False, "executable": str(executable),
                "runnable": False, "plugin_files": 0, "plugin_digest": None}
    if not executable.is_file():
        blocking.append(f"{executable} is not a file; the owned units start exactly that "
                        "path, so pointing the release at this tree would stop the "
                        "installation coming up")
    files = sorted((target / "integrations" / "hermes-memory").glob("*.py"))
    if not files:
        blocking.append(f"{target / 'integrations' / 'hermes-memory'} holds no plugin "
                        "files; the host's registration and the runtime have to move "
                        "together or the plugin keeps calling an older contract")
    return {"path": str(target), "staged": True, "executable": str(executable),
            "runnable": executable.is_file() and bool(files),
            "plugin_files": len(files), "plugin_digest": _plugin_digest(files)}


def _plugin_digest(files: list[Path]) -> str | None:
    """A digest of the plugin artifact, so "do the two trees agree" is answerable.

    Names and bytes only: the same files with the same content give the same digest,
    which is what §10.3 wants when it pairs the wheel and the plugin from one revision.
    """
    if not files:
        return None
    return digest([[path.name, content_digest(path.read_bytes())] for path in files])


def _source_plugin_files() -> list[Path]:
    root = Path(__file__).resolve().parents[3]
    return sorted((root / "integrations" / "hermes-memory").glob("*.py"))


def _switches(current: Path, target: Path | None, units: dict[str, Any], *,
              same: bool) -> list[str]:
    """What a switch would change, as opposed to what this plan changes."""
    if target is None:
        out = ["name a release to switch to"]
    elif same:
        out = [f"{current} is the release already in place, so the pointer would not move"]
    else:
        out = [f"point {current} at {target} once the rehearsal on a restored copy has "
               "passed"]
    out += [f"rewrite {entry['path']}" for entry in units["units"]
            if entry["state"] in ("written", "rewritten")]
    if units["reload_needed"]:
        out.append("systemctl --user daemon-reload, then restart only the owned units")
    return out


def _kept(settings) -> list[str]:
    return [f"{Path(settings.data_dir)} — the evidence, one store per enrolled profile",
            "the erasure ledger and the tombstones it proves",
            "the outboxes and their delivery proofs",
            "the profile bindings in the instance ledger",
            "every backup taken so far, under each profile's own snapshots directory"]


def profile_stores(settings) -> list[tuple[str, Path]]:
    """One (name, store) pair per profile this installation serves.

    Read from the ledger rather than assumed from the default configuration: an
    installation serving three people has three stores to back up, and checking only
    the default one is how an upgrade gets switched under somebody's live work. The
    default store is included as well when nothing in the ledger points at it, because
    evidence left outside the ledger still belongs to the owner.
    """
    pairs: list[tuple[str, Path]] = []
    registry = None
    try:
        registry = ProfileRegistry.reading(settings)
    except InstallationError:
        registry = None
    try:
        if registry is not None:
            pairs = [(profile.profile, Path(profile.db_path))
                     for profile in registry.profiles()]
    finally:
        if registry is not None:
            registry.db.close()
    default = Path(settings.db_path)
    if pairs:
        if default.is_file() and default not in [path for _name, path in pairs]:
            # Evidence nobody's ledger points at. Not enrolled is not the same as gone:
            # an upgrade still switches the code out from under it.
            pairs.append((f"the unenrolled store at {default}", default))
    else:
        pairs = [("the default profile", default)]
    return pairs


def _schema_state(settings) -> dict[str, Any]:
    from ..storage.evidence import ReadOnlyStore
    from ..storage.migrations import MIGRATIONS, current_version

    path = Path(settings.db_path)
    head = len(MIGRATIONS)
    if not path.is_file():
        return {"state": "absent", "now": None, "head": head, "store": str(path)}
    try:
        # Read only, and not merely as a manner of speaking: the writable store applies
        # pending migrations when it is opened, and a plan that migrated the archive on
        # its way past is the failure this module exists to avoid.
        with ReadOnlyStore(path) as store:
            applied = current_version(store.db)
    except Exception as error:
        return {"state": "unreadable", "now": None, "head": head, "store": str(path),
                "detail": str(error)[:200]}
    if applied > head:
        return {"state": "newer", "now": applied, "head": head, "store": str(path)}
    return {"state": "current" if applied == head else "behind",
            "now": applied, "head": head, "store": str(path)}


def _in_flight(profiles: list[tuple[str, Path]]) -> dict[str, Any]:
    """Which resources are occupied right now, across every store this installation serves.

    Counted rather than inferred from a PID: an upgrade that swapped the code out from
    under a running retain is the failure §10.6 calls quiescing for, and a generation
    still holding a lease is the evidence that one is.
    """
    from ..processing.resource_gate import HELD, UNCERTAIN
    from ..storage.evidence import ReadOnlyStore

    held: list[str] = []
    unreadable = 0
    scanned = 0
    for profile, path in profiles:
        if not path.is_file():
            continue
        scanned += 1
        try:
            with ReadOnlyStore(path) as store:
                rows = store.db.execute(
                    "SELECT resource FROM gate_reservations WHERE state IN (?,?)",
                    (HELD, UNCERTAIN)).fetchall()
        except Exception:
            unreadable += 1
            continue
        held.extend(f"{row[0]} for {profile}" for row in rows)
    return {"stores": scanned, "held": sorted(held), "unreadable": unreadable}


def _backups(profiles: list[tuple[str, Path]], *, moment: float) -> dict[str, Any]:
    """The newest real backup per profile, read from the manifests rather than the names.

    A snapshot is only evidence if it can be identified, so the answer is the recorded
    id and digest rather than "the directory exists".
    """
    from ..lifecycle.snapshots import Snapshots
    from ..storage.evidence import ReadOnlyStore

    per_profile: dict[str, str] = {}
    newest: dict[str, Any] = {}
    missing: list[str] = []
    worst_age, worst_profile = None, None
    for profile, path in profiles:
        directory = path.parent / "snapshots"
        found = None
        if directory.is_dir() and path.is_file():
            with ReadOnlyStore(path) as store:
                listed = Snapshots(store, directory=directory).list(limit=1)
            found = listed[0].as_dict() if listed else None
        if found is None:
            missing.append(profile)
            per_profile[profile] = "none"
            continue
        age = _age_days(found["created_at"], moment)
        per_profile[profile] = str(found["id"])
        newest[profile] = found
        if age is not None and (worst_age is None or age > worst_age):
            worst_age, worst_profile = age, profile
    return {"profiles": len(per_profile), "present": len(newest),
            "missing": missing, "newest": newest, "per_profile": per_profile,
            "worst_age_days": worst_age, "worst_profile": worst_profile}


def _age_days(stamp: Any, moment: float) -> int | None:
    try:
        created = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if created.tzinfo is None:
        return None
    return max(0, int((moment - created.timestamp()) // 86400))
