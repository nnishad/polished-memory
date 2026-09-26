"""Operator CLI.

Readiness is reported per stage; 'configured' is never reported as 'operational'. Every
reading here opens the canonical store in a mode that cannot change it, so running the
tool against a live installation is not an action.
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import SettingError, load_settings
from .install.services import Services, subprocess_runner
from .storage.evidence import EvidenceError, EvidenceStore, ReadOnlyStore

__all__ = ["main"]


def _configuration(settings) -> dict[str, Any]:
    return {
        "data_dir": str(settings.data_dir),
        "database": str(settings.db_path),
        "database_present": settings.db_path.exists(),
        "inference_enabled": settings.inference_enabled,
        "capture_only": settings.capture_only,
        "hindsight_route": settings.hindsight_url or "unset",
        "approved_inference_hosts": sorted(settings.allowed_inference_hosts),
        "background_budget_tokens": settings.background_budget_tokens,
        "foreground_deadline_s": settings.foreground_deadline_s,
        "owner_principal": settings.owner_principal or "unset",
        # An absent backend package means degraded capture and lexical recall, and is
        # never a reason for the framework to refuse to start.
        "hindsight_client_importable": find_spec("hindsight_client") is not None,
    }


def _capabilities(settings, store_present: bool) -> dict[str, bool]:
    """Which jobs this installation can do, said one at a time.

    A store that only captures is working, and one whose formation is switched off is
    not broken; a single readiness flag made both of those look like failures.
    """
    return {
        "capture": store_present,
        "local_recall": store_present,
        "formation": store_present and not settings.capture_only,
        # Nothing in this installation drains the queue on its own. Formation is a
        # bounded pass an operator runs (`hermes-memory form`), so a store that *can*
        # form observations still forms none until somebody asks and approves the list.
        # §8.2 keeps the always-on worker behind that decision rather than pretending it
        # is already running.
        "formation_unattended": False,
        "forgetting": store_present and bool(settings.owner_principal),
        # Delivery is not a capability an installation has merely because somebody is
        # named: it needs an owner, an explicit switch and one concrete destination.
        "delivery": bool(store_present and settings.owner_principal
                         and settings.delivery_enabled and settings.delivery_target),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-memory")
    parser.add_argument("--env-file",
                        help="owned env file (default: $HERMES_MEMORY_HOME/hermes-memory.env)")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status",
                            help="print configuration and every stage's own account")
    status.add_argument("--hermes-home",
                        help="report the memory enrolled for this Hermes profile home")
    init = sub.add_parser("init", help="create and migrate the canonical store")
    init.add_argument("--dry-run", action="store_true")
    doctor = sub.add_parser("doctor", help="read-only checks; performs no inference")
    doctor.add_argument("--hermes-home",
                        help="examine the memory enrolled for this Hermes profile home")
    doctor.add_argument("--probe", action="store_true",
                        help="ask the configured backend whether it is answering")
    doctor.add_argument("--synthetic-probe", action="store_true",
                        help="run one bounded, synthetic retain and recall round trip")

    setup = sub.add_parser("setup",
                           help="the installation transaction; run without --review to "
                                "see every step first")
    setup.add_argument("--hermes-home", required=True,
                       help="the Hermes profile home to set this memory up for")
    setup.add_argument("--profile", help="short name; defaults to the home's directory")
    setup.add_argument("--ref", help="the 40-character release commit to pin the plugin to")
    setup.add_argument("--actor")
    setup.add_argument("--review", metavar="DIGEST",
                       help="the digest of the plan that was actually shown")
    setup.add_argument("--start", action="store_true",
                       help="start the owned services once the units are installed")

    sub.add_parser("profiles", help="list the Hermes profiles this installation serves")
    inventory = sub.add_parser("inventory",
                               help="read what is already here; opens no socket and runs "
                                    "no model")
    inventory.add_argument("--hermes-home",
                           help="the Hermes profile home to report the host facts of")
    inventory.add_argument("--conflicts", action="store_true",
                           help="print only the reasons a setup run would be a bad idea")
    enroll = sub.add_parser("enroll",
                            help="map one Hermes profile to its own memory; run without "
                                 "--review to see what would change")
    enroll.add_argument("--hermes-home", required=True,
                        help="the profile home Hermes passes to an activity")
    enroll.add_argument("--profile", help="short name; defaults to the home's directory")
    enroll.add_argument("--actor", help="the owner principal approving this")
    enroll.add_argument("--review", metavar="DIGEST",
                        help="the digest of the plan that was actually shown")
    retire = sub.add_parser("retire",
                            help="unlink a profile; its evidence stays exactly where it is")
    retire.add_argument("--profile", required=True)
    retire.add_argument("--actor")
    retire.add_argument("--reason", required=True)

    audit = sub.add_parser("audit", help="read the ledger of what has already happened")
    audit.add_argument("--action")
    audit.add_argument("--object", dest="object_id")
    audit.add_argument("--actor")
    audit.add_argument("--source", help="one connector's own history instead of the ledger")
    audit.add_argument("--limit", type=int, default=50)

    explain = sub.add_parser("explain", help="why an item came back, or why nothing did")
    target = explain.add_mutually_exclusive_group(required=True)
    target.add_argument("--record")
    target.add_argument("--artifact")
    target.add_argument("--goal")
    target.add_argument("--not-told", metavar="TOPIC",
                        help="list the reasons this topic did not interrupt the owner")
    explain.add_argument("--at", help="the moment to judge quiet hours against")
    explain.add_argument("--include-private", action="store_true",
                         help="include payload and goal text, still redacted")
    explain.add_argument("--limit", type=int, default=20)

    sub.add_parser("serve", help="run the admission endpoint the runtime unit expects")
    services = sub.add_parser("services",
                              help="show the owned user units; --install writes the plan "
                                   "that was shown")
    services.add_argument("--install", metavar="DIGEST",
                          help="the digest of the plan to write")
    services.add_argument("--autostart", choices=("enable", "disable"),
                          help="decide whether these units start at login; a separate "
                               "decision from starting them now")
    services.add_argument("--actor")
    sub.add_parser("start", help="start the owned services, gate before backend before worker")
    stop = sub.add_parser("stop", help="persist the pause, then stop the owned services")
    stop.add_argument("--actor")
    stop.add_argument("--reason", default="the operator stopped the services")
    pause = sub.add_parser("pause", help="hold a stage, and keep holding it across a restart")
    pause.add_argument("--scope", required=True, choices=("inference", "delivery"))
    pause.add_argument("--resume", action="store_true",
                       help="lift a hold a previous pause set")
    pause.add_argument("--actor")
    pause.add_argument("--reason")

    backup = sub.add_parser("backup",
                            help="copy each profile's store to a snapshot that can be "
                                 "identified and verified")
    backup.add_argument("--profile", help="one enrolled profile; every of them by default")
    backup.add_argument("--reason", default="operator backup")
    backup.add_argument("--actor")
    backup.add_argument("--list", action="store_true",
                        help="report the backups that exist and take none")

    restore = sub.add_parser(
        "restore", help="take one profile's store back to a snapshot that verifies, "
                        "keeping every decision made since; run without --review to see "
                        "the snapshot's own facts first")
    restore.add_argument("--snapshot", required=True)
    restore.add_argument("--profile", help="whose store; required once one is enrolled")
    restore.add_argument("--actor")
    restore.add_argument("--review", metavar="DIGEST",
                         help="the digest of the restore that was actually shown")

    upgrade = sub.add_parser("upgrade",
                             help="plan a switch to a staged release. It is read only: "
                                  "there is no flag here that performs one")
    upgrade.add_argument("--version", metavar="RELEASE-TREE",
                         help="the staged release tree to switch to")
    upgrade.add_argument("--hermes-home",
                         help="the Hermes profile home whose host facts are reported")

    uninstall = sub.add_parser("uninstall",
                               help="remove this installation and keep every byte of memory")
    uninstall.add_argument("--keep-data", action="store_true",
                           help="required: data-preserving is the only uninstall there is")
    uninstall.add_argument("--hermes-home",
                           help="the profile home whose provider selection setup changed")
    uninstall.add_argument("--actor")
    uninstall.add_argument("--review", metavar="DIGEST",
                           help="the digest of the list that was actually shown")

    sources = sub.add_parser("sources", help="what the connectors know, including what "
                                             "they did not get")
    sources.add_argument("action", choices=("list",))
    sources.add_argument("--gaps", action="store_true", help="include each source's open gaps")
    sources.add_argument("--limit", type=int, default=20)

    importing = sub.add_parser("import",
                               help="read one export directory into the memory that owns it")
    importing.add_argument("--source", required=True, help="the connector id to record it under")
    importing.add_argument("--path", required=True,
                           help="a directory of exports the owner pointed at")
    importing.add_argument("--profile", help="whose memory it goes into")
    importing.add_argument("--policy", default="local-only",
                           choices=("local-only", "private-api", "disabled"),
                           help="the scope declared for a connector new to this store")
    importing.add_argument("--dry-run", action="store_true",
                           help="ask whether the export can be read, and read nothing else")

    form = sub.add_parser("form",
                          help="project evidence into the derived backend; run without "
                               "--review to see the bounded list and perform nothing")
    form.add_argument("--limit", type=int, default=None,
                      help="the most records one pass may select")
    form.add_argument("--max-jobs", type=int, default=None,
                      help="the most queued jobs to attempt in this invocation")
    form.add_argument("--actor")
    form.add_argument("--hermes-home",
                      help="form the memory enrolled for this Hermes profile home")
    form.add_argument("--review", metavar="DIGEST",
                      help="the digest of the list that was actually shown")

    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.env_file)
    except SettingError as error:
        print(f"configuration refused: {error}", file=sys.stderr)
        return 2

    if args.command == "status":
        return _status_command(settings, args)
    if args.command == "init":
        return _init_command(settings, args)
    if args.command == "doctor":
        return _doctor_command(settings, args)
    if args.command == "audit":
        return _audit_command(settings, args)
    if args.command == "profiles":
        return _profiles_command(settings)
    if args.command == "inventory":
        return _inventory_command(settings, args)
    if args.command == "setup":
        return _setup_command(settings, args)
    if args.command == "enroll":
        return _enroll_command(settings, args)
    if args.command == "retire":
        return _retire_command(settings, args)
    if args.command == "serve":
        return _serve_command(settings)
    if args.command == "services":
        return _services_command(settings, args)
    if args.command == "start":
        return _start_command(settings)
    if args.command == "stop":
        return _stop_command(settings, args)
    if args.command == "pause":
        return _pause_command(settings, args)
    if args.command == "backup":
        return _backup_command(settings, args)
    if args.command == "restore":
        return _restore_command(settings, args)
    if args.command == "upgrade":
        return _upgrade_command(settings, args)
    if args.command == "uninstall":
        return _uninstall_command(settings, args)
    if args.command == "sources":
        return _sources_command(settings, args)
    if args.command == "import":
        return _import_command(settings, args)
    if args.command == "form":
        return _form_command(settings, args)
    return _explain_command(settings, args)


# -- the profile map -----------------------------------------------------------

def _registry(settings, *, create: bool = True):
    """The instance ledger. A read never brings one into being."""
    from .install.profiles import ProfileRegistry

    return ProfileRegistry.open(settings) if create else ProfileRegistry.reading(settings)


def _profiles_command(settings) -> int:
    registry = _registry(settings, create=False)
    try:
        enrolled = [item.as_dict(private=True) for item in registry.profiles()]
        retired = [item.as_dict(private=True) for item in registry.profiles(include_retired=True)
                   if item.state != "enrolled"]
        stores = {item.profile: item.db_path.exists() for item in registry.profiles()}
    finally:
        registry.db.close()
    return _emit({"instance_home": str(settings.home), "profiles": enrolled,
                  "retired": retired,
                  "store_present": stores,
                  "note": "each profile has its own store, bank and credential scope; "
                          "nothing here is shared but the machine"})


def _enroll_command(settings, args) -> int:
    """Two steps, because an enrollment is a decision and not a side effect.

    Without ``--review`` this prints what would change and nothing else. With it, the
    digest has to be the one that was shown, and the actor has to be the owner the
    configuration names — a caller that can name a directory does not thereby own it.
    """
    from .install.profiles import InstallationError

    name = args.profile or _profile_name(args.hermes_home)
    reading = _registry(settings, create=False)
    try:
        proposal = reading.plan(name, args.hermes_home)
    finally:
        reading.db.close()
    if not args.review:
        return _emit({**proposal,
                      "next": f"hermes-memory enroll --profile {name} "
                              f"--hermes-home {args.hermes_home} "
                              f"--review {proposal['review_digest']}"})
    registry = _registry(settings)
    try:
        actor = args.actor or settings.owner_principal
        if not actor:
            print("refused: no owner principal is configured, so nobody may approve an "
                  "enrollment (set HERMES_MEMORY_OWNER_PRINCIPAL)", file=sys.stderr)
            return 2
        try:
            result = registry.enroll(name, args.hermes_home, actor=actor,
                                     review_digest=args.review)
        except InstallationError as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
        return _emit({**result,
                      "store": str(Path(result["data_dir"]) / "canonical.db")})
    finally:
        registry.db.close()


def _retire_command(settings, args) -> int:
    from .install.profiles import InstallationError

    actor = args.actor or settings.owner_principal
    if not actor:
        print("refused: retiring a profile is the owner's decision and no owner is "
              "configured", file=sys.stderr)
        return 2
    registry = _registry(settings)
    try:
        try:
            return _emit(registry.retire(args.profile, actor=actor, reason=args.reason))
        except InstallationError as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
    finally:
        registry.db.close()


def _setup_command(settings, args) -> int:
    """The transaction, from the operator's chair.

    Without ``--review`` this only reads: the eleven steps, their actions, and the host
    commands any of them would run. With it, the digest must be the one that was shown,
    the actor must be the owner the configuration names, and the host commands go through
    the one executor this process uses.
    """
    from .install.profiles import InstallationError
    from .install.setup import SetupError, plan

    arguments = {"hermes_home": args.hermes_home, "profile": args.profile, "ref": args.ref,
                 "runner": _HOST_RUNNER, "actor": args.actor, "start_services": args.start}
    try:
        proposal = plan(settings, **arguments)
    except (SetupError, InstallationError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not args.review:
        return _emit({**proposal,
                      "next": f"hermes-memory setup --hermes-home {args.hermes_home} "
                              f"--ref {args.ref or '<tested-release-commit>'} "
                              f"--review {proposal['review_digest']}"})
    from .install.setup import run as apply_transaction

    try:
        return _emit(apply_transaction(settings, review=args.review,
                                       **{**arguments,
                                          "actor": args.actor
                                          or settings.owner_principal or ""}))
    except (SetupError, InstallationError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


def _inventory_command(settings, args) -> int:
    """What is on this machine, read without changing it.

    ``--conflicts`` exists because the useful question is not everything an inventory
    can see but whether any of it should stop an install, and a script that gates on it
    should not have to re-derive the answer from a 60-key report.
    """
    from .install.inventory import conflicts, survey

    report = survey(settings, hermes_home=args.hermes_home)
    if args.conflicts:
        blocking = conflicts(report)
        print(json.dumps(blocking, indent=2, sort_keys=True))
        return 1 if blocking else 0
    return _emit({**report, "conflicts": conflicts(report)})


def _profile_name(hermes_home: str) -> str:
    """The home's own directory name, or ``default`` for a bare account home."""
    name = Path(hermes_home).expanduser().name.strip().lower()
    return name if name and name not in {"home", ".hermes", "hermes"} else "default"


# -- the services and the fence ------------------------------------------------

def _serve_command(settings) -> int:
    """The process the runtime unit starts. It refuses before it binds.

    A gate that cannot name its address, or that would bind somewhere other than
    loopback, exits non-zero here rather than starting a service that quietly listens
    on a network.
    """
    from .config import SettingError
    from .service import serve

    try:
        return serve(settings)
    except SettingError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


def _services_command(settings, args) -> int:
    from .install.profiles import InstallationError
    from .install.services import apply, plan

    proposal = plan(settings)
    if args.autostart:
        # A separate verb, because "start it now" and "start it at every login" are
        # different decisions and only one of them is reversible by a reboot.
        try:
            units = _controller(settings).autostart(
                enable=args.autostart == "enable")
        except InstallationError as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
        return _emit({"autostart": args.autostart, "units": units,
                      "note": "a start never lifts a pause, so an enabled unit that finds "
                              "inference or delivery held comes up holding it"})
    if not args.install:
        return _emit({**proposal,
                      "next": (f"hermes-memory services --install {proposal['review_digest']}"
                               if proposal["would_change"] else
                               "nothing to write: every owned unit already says this")})
    try:
        receipt = apply(settings, actor=args.actor or settings.owner_principal or "",
                        review=args.install)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({**receipt,
                  "next": "systemctl --user daemon-reload, then hermes-memory start"
                          if receipt["reload_needed"] else "no reload needed"})


def _controller(settings) -> Services:
    """The service controller, with whatever host command executor this process uses."""
    return Services(settings, runner=_HOST_RUNNER)


# Every command that reaches outside this process - systemctl, the host's plugin and
# configuration writers - goes through one executor, so a test can watch the ordering
# and refusal claims instead of starting real services on the machine running them.
_HOST_RUNNER: Callable[[Sequence[str]], tuple[int, str]] = subprocess_runner


def _start_command(settings) -> int:
    from .install.profiles import InstallationError
    from .install.services import plan

    stale = plan(settings)
    if stale["blocked"]:
        print("refused: these units exist but were not written by this installation, so "
              f"they are not ours to start: {', '.join(stale['blocked'])}", file=sys.stderr)
        return 2
    try:
        started = _controller(settings).start()
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({"ok": True, "started": started,
                  "note": "a pause the owner set is still in force; starting does not "
                          "resume formation or delivery",
                  "units_not_as_described": stale["would_change"]})


def _stop_command(settings, args) -> int:
    """Pause first, then stop. The order is the shutdown contract, not a preference.

    A worker killed mid-retain with no recorded pause looks like an idle installation
    to the next boot, which resumes exactly the formation the owner was trying to halt.
    """
    from .install.profiles import InstallationError
    from .processing.instance_gate import instance_gate
    from .storage.evidence import EvidenceStore

    actor = args.actor or settings.owner_principal
    if not actor:
        print("refused: stopping is the owner's decision and no owner principal is "
              "configured", file=sys.stderr)
        return 2
    reason = args.reason
    paused = []
    held_for = []

    def hold() -> None:
        # Inference is held once, for the whole installation: the models are shared
        # hardware, and a stop that only paused the default profile would leave the
        # second one dispatching against a machine its owner just stopped.
        with instance_gate(settings) as gate:
            gate.pause(actor=actor, reason=reason)
        for scoped in _archive_targets(settings, None):
            if not scoped.db_path.is_file():
                continue
            with EvidenceStore(scoped.db_path) as store:
                store.set_control("global", "delivery", "paused", actor=actor,
                                  reason=reason, policy_version="operator-stop")
            held_for.append(scoped.profile)
        paused.extend(["inference", "delivery"])

    try:
        stopped = _controller(settings).stop(pause=hold)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({"ok": True, "stopped": stopped, "paused": paused, "held_for": held_for,
                  "actor": actor,
                  "resumes_with": "hermes-memory pause --scope inference --resume"
                                  " (and --scope delivery --resume)"})


def _pause_command(settings, args) -> int:
    """A hold that outlives the process that set it.

    Both are written to a database rather than to the unit, because a restart is exactly
    the event that must not lift it — and an autostart that resumed formation would be the
    machine deciding against its owner. Inference goes in the instance admission ledger,
    which is the one place every profile's queue can see; delivery is a per-profile
    outbox, so it is held on each profile's own archive.
    """
    from .processing.instance_gate import instance_gate
    from .storage.evidence import EvidenceStore

    actor = args.actor or settings.owner_principal
    if not actor:
        print("refused: a pause is attributed, and with no owner principal configured "
              "there is nobody to attribute it to", file=sys.stderr)
        return 2
    reason = args.reason or ("the owner lifted the hold" if args.resume
                             else "the owner held this stage")
    state = "active" if args.resume else "paused"
    written_to: list[str] = []
    if args.scope == "inference":
        with instance_gate(settings) as gate:
            (gate.resume if args.resume else gate.pause)(actor=actor, reason=reason)
    else:
        for scoped in _archive_targets(settings, None):
            if not scoped.db_path.is_file():
                continue
            with EvidenceStore(scoped.db_path) as store:
                store.set_control("global", "delivery", state, actor=actor,
                                  reason=reason, policy_version="operator-pause")
            written_to.append(scoped.profile)
    return _emit({"ok": True, "scope": args.scope, "state": state, "actor": actor,
                  "reason": reason, "held": _holds(settings), "written_to": written_to,
                  "survives_a_restart": True})


def _holds(settings) -> dict[str, Any]:
    """What is being held right now, read from both ledgers that hold anything.

    Reported rather than echoed from the command that was just run: an operator asking
    "is it stopped" wants the machine's answer, not a restatement of their own request.
    """
    from .processing.instance_gate import instance_gate

    with instance_gate(settings) as gate:
        inference = gate.paused
    delivery = False
    for scoped in _archive_targets(settings, None):
        if not scoped.db_path.is_file():
            continue
        with ReadOnlyStore(scoped.db_path) as store:
            delivery = delivery or store.stage_is_paused("global", "delivery")
    return {"inference": inference, "delivery": delivery}


# -- the archive, the release and the sources ----------------------------------

def _backup_command(settings, args) -> int:
    """A copy of the archive that can be named, verified and taken back.

    Every enrolled profile is backed up unless one is named, because the operator who
    types `backup` means "this installation is now recoverable" — and one store copied
    out of three would make that sentence false.
    """
    from .lifecycle.snapshots import Snapshots
    from .storage.evidence import EvidenceStore

    actor = args.actor or settings.owner_principal
    if not actor:
        print("refused: a backup records who took it, and no owner principal is configured",
              file=sys.stderr)
        return 2
    try:
        targets = _archive_targets(settings, args.profile)
    except Exception as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    taken, present = [], []
    try:
        for scoped in targets:
            directory = Path(scoped.data_dir) / "snapshots"
            if not scoped.db_path.is_file():
                present.append({"profile": scoped.profile, "snapshot": None,
                                "directory": str(directory),
                                "reason": f"no store at {scoped.db_path} to copy"})
                continue
            with EvidenceStore(scoped.db_path) as store:
                snapshots = Snapshots(store, directory=directory)
                if args.list:
                    present.append({"profile": scoped.profile, "directory": str(directory),
                                    "snapshots": [item.as_dict() for item in
                                                  snapshots.list(limit=20)]})
                    continue
                made = snapshots.create(reason=args.reason, actor=actor)["snapshot"]
                taken.append({"profile": scoped.profile, "snapshot": made.as_dict(),
                              "verified": snapshots.verify(made.id)["ok"]})
    except EvidenceError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if args.list:
        return _emit({"listing": True, "profiles": present})
    return _emit({"ok": True, "actor": actor, "backups": taken, "skipped": present,
                  "note": "a snapshot is one consistent copy of one store, taken while "
                          "the store stayed open; take it back with `hermes-memory restore`"})


# The facts a restore is approved against. The live epoch is in it on purpose: a store
# another process is still writing has moved since the operator read it, and the answer to
# that is a refusal and `hermes-memory stop`, never a rollback of a store nobody looked at.
RESTORE_PLAN_VERSION = "restore-plan-v1"


def _restore_command(settings, args) -> int:
    """Show the snapshot's own facts, then take back the store that was shown.

    A restore destroys every record written after the snapshot, so it is the one command
    here that a digest has to gate: the operator sees which snapshot, what it holds, and
    how many forgetting decisions the ledger will re-apply, and approves exactly that
    reading. The carried rows are counted rather than printed — a plan that dumped record
    IDs would put private evidence in the shell's scrollback.
    """
    from .ids import digest
    from .install.profiles import InstallationError
    from .lifecycle.recovery import Recovery
    from .lifecycle.snapshots import Snapshots

    try:
        targets = _archive_targets(settings, args.profile)
    except (InstallationError, EvidenceError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not args.profile and len(targets) != 1:
        print(f"refused: a restore takes back one store and {len(targets)} profiles are "
              "enrolled; name one with --profile", file=sys.stderr)
        return 2
    scoped = targets[0]
    actor = args.actor or scoped.owner_principal
    if not actor:
        print("refused: a restore is confirmed by the owner principal and none is "
              "configured, so nothing here could be sure who asked", file=sys.stderr)
        return 2
    if not scoped.db_path.is_file():
        print(f"refused: no store at {scoped.db_path} to take back", file=sys.stderr)
        return 2
    try:
        with EvidenceStore(scoped.db_path) as store:
            snapshots = Snapshots(store, directory=Path(scoped.data_dir) / "snapshots")
            checked = snapshots.verify(args.snapshot)
            if not checked["ok"]:
                print("refused: " + "; ".join(checked["problems"]), file=sys.stderr)
                return 2
            snapshot = snapshots.resolve(args.snapshot)
            carried = Recovery(store, snapshots=snapshots,
                               owner_principal=scoped.owner_principal).carry()
            proposal = {"profile": scoped.profile, "store": str(scoped.db_path),
                        "snapshot": snapshot.as_dict(), "notes": checked["notes"],
                        "live_epoch": store.epoch(),
                        "decisions_kept": {name: len(rows)
                                           for name, rows in carried.tables.items()}}
            review = digest([RESTORE_PLAN_VERSION, proposal])
            if args.review is None:
                return _emit({**proposal, "review_digest": review,
                              "next": f"hermes-memory restore --snapshot {snapshot.id} "
                                      f"--profile {scoped.profile} --actor {actor} "
                                      f"--review {review}"})
            if args.review != review:
                print("refused: the store has moved since that plan was shown, so what "
                      "would be destroyed is not what was approved. This is the reading "
                      f"now: {review}", file=sys.stderr)
                return 2
            report = Recovery(store, snapshots=snapshots,
                              owner_principal=scoped.owner_principal).restore(
                                  args.snapshot, actor=actor)
    except (EvidenceError, InstallationError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({"ok": True, "profile": scoped.profile, **report})


def _archive_targets(settings, profile: str | None) -> list:
    """The configurations whose stores a backup covers."""
    from .install.profiles import InstallationError, ProfileRegistry

    if not (Path(settings.home) / "installation.db").is_file():
        if profile:
            raise InstallationError(
                f"no instance ledger exists, so profile {profile!r} is not enrolled")
        return [settings]
    registry = ProfileRegistry.reading(settings)
    try:
        if profile:
            return [registry.profile(profile).scoped(settings)]
        enrolled = [item.scoped(settings) for item in registry.profiles()]
    finally:
        registry.db.close()
    return enrolled or [settings]


def _upgrade_command(settings, args) -> int:
    """The plan of §10.6, and deliberately not the switch.

    There is no ``--apply`` here to approve. Performing the switch needs a quiesced
    machine, a rehearsal on a restored copy and a validated final state, and an operator
    who has those would not want a flag that guesses at them.
    """
    from .install.upgrade import UpgradeError, upgrade_plan

    try:
        report = upgrade_plan(settings, release=args.version, hermes_home=args.hermes_home)
    except UpgradeError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({**report,
                  "next": "this plan performs nothing. Read the blockers, take a backup, "
                          "rehearse the migrations on a restored copy, and switch the "
                          "release pointer only when the rehearsal passed"})


def _uninstall_command(settings, args) -> int:
    """Two steps again: the list first, then the removal of exactly that list."""
    from .install.uninstall import UninstallError, uninstall_apply, uninstall_plan

    try:
        proposal = uninstall_plan(settings, hermes_home=args.hermes_home)
    except UninstallError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not args.review:
        return _emit({**proposal, "next": proposal["next"].replace(
            "--actor <owner-principal>",
            f"--actor {args.actor or settings.owner_principal or '<owner-principal>'}")})
    try:
        return _emit(uninstall_apply(
            settings, actor=args.actor or settings.owner_principal or "",
            review=args.review, keep_data=args.keep_data,
            hermes_home=args.hermes_home, runner=_HOST_RUNNER))
    except UninstallError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


def _sources_command(settings, args) -> int:
    """Each connector's own account: where it stopped, and what it could not get."""
    from .sources.sync import SyncController

    def read(store):
        sync = SyncController(store)
        listed = []
        for row in store.db.execute("SELECT source FROM connectors ORDER BY source"):
            state = sync.state(row["source"])
            entry = {"source": row["source"], "generation": state["generation"],
                     "policy": state["policy_version"], "cursor": state["cursor"],
                     "coverage": state["coverage_state"],
                     "last_success_at": state["last_success_at"],
                     "paused_stages": sync.paused_stages(row["source"]),
                     "lease": "held" if state["lease_active"] else "free"}
            if args.gaps:
                entry["open_gaps"] = sync.gaps(row["source"], limit=args.limit)
            listed.append(entry)
        return {"sources": listed,
                "registered": len(listed),
                "note": "coverage is what the source says it has; a gap is what it could "
                        "not hand over, and neither is a claim about what is missing"}

    return _with_store(settings, read)


def _import_command(settings, args) -> int:
    """Read a directory of exports the owner named into the store that owns it.

    The adapter is chosen from a fixed map of ids to file readers: no path here is ever
    turned into a command, a URL or an import of somebody's code. A connector already
    registered keeps its own policy version, because re-declaring a scope from a shell
    command is how a private source starts looking public.
    """
    from .sources.runtime import ConnectorRuntime
    from .sources.sync import SyncController
    from .storage.evidence import EvidenceStore

    adapter_class = _export_readers().get(args.source)
    if adapter_class is None:
        print("refused: no export reader for source "
              f"{args.source!r}; this command reads {', '.join(sorted(_export_readers()))} "
              "from a directory the owner points at", file=sys.stderr)
        return 2
    try:
        targets = _archive_targets(settings, args.profile)
    except Exception as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if len(targets) != 1:
        print("refused: --profile must name which memory the export is being read into",
              file=sys.stderr)
        return 2
    scoped = targets[0]
    if not scoped.db_path.is_file():
        print(f"refused: no canonical store at {scoped.db_path}; an import writes into a "
              "memory that exists, so run `hermes-memory init` (or enroll) first",
              file=sys.stderr)
        return 2
    adapter = adapter_class(args.path, source=args.source)
    if not args.dry_run:
        try:
            with EvidenceStore(scoped.db_path) as store:
                sync = SyncController(store)
                try:
                    declared = sync.state(args.source)["policy_version"]
                except EvidenceError:
                    declared = args.policy
                    sync.register(args.source, policy_version=declared)
                runtime = ConnectorRuntime(store, sync,
                                           holder=f"cli import from {args.path}")
                outcome = runtime.run(adapter)
        except EvidenceError as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
        return _emit({**outcome.as_dict(), "profile": scoped.profile, "policy": declared,
                      "store": str(scoped.db_path),
                      "recheck": "a run that stopped for a reason says so; the source's "
                                 "coverage claim is its own"})
    return _emit({"dry_run": True, "source": args.source, "path": str(args.path),
                  "profile": scoped.profile, "reachable": adapter.check(),
                  "note": "nothing was read; run the same command without --dry-run to "
                          "record what is in the export"})


def _form_command(settings, args) -> int:
    """The one place memory-originated inference is dispatched from a shell.

    Planning is a reading. Performing is a write to the queue, a slot in a device the
    whole machine shares, and a spend from a daily budget, so it requires the digest of
    the list that was actually shown and the name of the person who authorised it. There
    is no ``--yes`` and no default: an unattended pass is a worker that has not been
    commissioned yet, not a flag on this command.
    """
    from .install.profiles import InstallationError
    from .processing.formation import (DEFAULT_BATCH, MAX_JOBS, FormationError,
                                       formation_apply, formation_plan)

    try:
        settings = _memory_for_home(settings, args.hermes_home)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    limit = DEFAULT_BATCH if args.limit is None else args.limit
    max_jobs = MAX_JOBS if args.max_jobs is None else args.max_jobs
    try:
        proposal = formation_plan(settings, limit=limit)
    except FormationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not args.review:
        return _emit({**proposal, "next": "nothing left this process. Approve this exact "
                                          "list with --review <digest> and an --actor; a "
                                          "pass that costs tokens names who authorised it"})
    try:
        return _emit(formation_apply(
            settings, review=args.review,
            actor=args.actor or settings.owner_principal or "",
            limit=limit, max_jobs=max_jobs))
    except FormationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


# The file readers only. A live mailbox, an MCP server or the host's own event spool
# each need an authorisation this command cannot carry, so they are not in the map.
def _export_readers() -> dict[str, Any]:
    from .sources.email import EmailSource
    from .sources.files import FileSource
    from .sources.structured import StructuredSource
    from .sources.whatsapp_export import WhatsAppExport

    return {"email": EmailSource, "files": FileSource, "structured": StructuredSource,
            "whatsapp": WhatsAppExport}


def _memory_for_home(settings, hermes_home: str | None):
    """One enrolled profile's configuration, or this installation's own when none is named.

    The lookup is the ledger's, so an unenrolled or retired home is refused rather than
    answered from the default profile. A report about one person that quietly came out of
    another person's memory is the exact failure the profile map exists to prevent.
    """
    if not hermes_home:
        return settings
    from .install.profiles import ProfileRegistry

    reading = ProfileRegistry.reading(settings)
    try:
        return reading.resolve(hermes_home).scoped(settings)
    finally:
        reading.db.close()


# -- the commands ------------------------------------------------------------

def _status_command(settings, args) -> int:
    from .install.profiles import InstallationError
    from .operations.status import StatusReporter

    try:
        settings = _memory_for_home(settings, args.hermes_home)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    present = settings.db_path.exists()
    report = {**_configuration(settings),
              "profile": settings.profile,
              "capabilities": _capabilities(settings, present),
              "stages": None}
    if not present:
        report["note"] = "no canonical store yet; run `hermes-memory init`"
        return _emit(report)
    try:
        with ReadOnlyStore(settings.db_path) as store:
            report["stages"] = StatusReporter(store, settings=settings).report(
                profile=settings.profile)
    except EvidenceError as error:
        report["note"] = str(error)
        return _emit(report, 1)
    return _emit(report)


def _init_command(settings, args) -> int:
    if args.dry_run:
        return _emit({"would_create": [str(settings.data_dir), str(settings.blob_dir)],
                      "store": str(settings.db_path),
                      "migrations": "applied when the store is opened for writing"})
    settings.data_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    settings.blob_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    with EvidenceStore(settings.db_path) as store:
        store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return _emit({"initialized": str(settings.db_path)})


def _doctor_command(settings, args) -> int:
    from .install.profiles import InstallationError
    from .operations.doctor import Doctor, unreachable_store_report

    try:
        settings = _memory_for_home(settings, args.hermes_home)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not settings.db_path.exists():
        return _emit(unreachable_store_report(
            settings.db_path, "the canonical store does not exist yet"), 1)
    try:
        with ReadOnlyStore(settings.db_path) as store:
            report = Doctor(store, settings=settings).examine(
                connectivity=args.probe, synthetic=args.synthetic_probe,
                profile=settings.profile)
    except EvidenceError as error:
        return _emit(unreachable_store_report(settings.db_path, str(error)), 1)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return int(report["exit_code"])


def _audit_command(settings, args) -> int:
    from .operations.audit import AuditTrail

    def read(store):
        trail = AuditTrail(store)
        if args.source:
            return trail.source_history(args.source, limit=args.limit)
        return trail.recent(action=args.action, object_id=args.object_id,
                            actor=args.actor, limit=args.limit)

    return _with_store(settings, read)


def _explain_command(settings, args) -> int:
    from .operations.explanations import Explanations

    def read(store):
        explained = Explanations(store, settings=settings)
        if args.record:
            return explained.retrieval(args.record, limit=args.limit)
        if args.artifact:
            return explained.notification(args.artifact, include_private=args.include_private)
        if args.goal:
            return explained.goal(args.goal, include_private=args.include_private)
        return explained.suppressed(topic=args.not_told, at=args.at, limit=args.limit)

    return _with_store(settings, read)


def _with_store(settings, read: Callable[[Any], Any]) -> int:
    try:
        with ReadOnlyStore(settings.db_path) as store:
            payload = read(store)
    except (EvidenceError, ValueError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit(payload)


def _emit(payload, code: int = 0) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
