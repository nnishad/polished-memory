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

    sub.add_parser("status", help="print configuration and every stage's own account")
    init = sub.add_parser("init", help="create and migrate the canonical store")
    init.add_argument("--dry-run", action="store_true")
    doctor = sub.add_parser("doctor", help="read-only checks; performs no inference")
    doctor.add_argument("--probe", action="store_true",
                        help="ask the configured backend whether it is answering")
    doctor.add_argument("--synthetic-probe", action="store_true",
                        help="run one bounded, synthetic retain and recall round trip")

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

    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.env_file)
    except SettingError as error:
        print(f"configuration refused: {error}", file=sys.stderr)
        return 2

    if args.command == "status":
        return _status_command(settings)
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


def _controller(settings):
    """The service controller, with whatever host command executor this process uses.

    ``_SERVICE_RUNNER`` exists so the tests can watch the ordering claims instead of
    running them: the alternative is a test suite that starts and stops real services.
    """
    from .install.services import Services

    return Services(settings, runner=_SERVICE_RUNNER)


_SERVICE_RUNNER: Callable[[Sequence[str]], tuple[int, str]] | None = None


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
    from .processing.resource_gate import ResourceGate
    from .storage.evidence import EvidenceStore

    actor = args.actor or settings.owner_principal
    if not actor:
        print("refused: stopping is the owner's decision and no owner principal is "
              "configured", file=sys.stderr)
        return 2
    reason = args.reason
    paused = []

    def hold() -> None:
        with EvidenceStore(settings.db_path) as store:
            ResourceGate(store).pause(actor=actor, reason=reason)
            store.set_control("global", "delivery", "paused", actor=actor, reason=reason,
                              policy_version="operator-stop")
        paused.extend(["inference", "delivery"])

    try:
        stopped = _controller(settings).stop(pause=hold)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({"ok": True, "stopped": stopped, "paused": paused, "actor": actor,
                  "resumes_with": "hermes-memory pause --scope inference --resume"
                                  " (and --scope delivery --resume)"})


def _pause_command(settings, args) -> int:
    """A hold that outlives the process that set it.

    Both scopes are recorded in the canonical store rather than in the unit, because a
    restart is exactly the event that must not lift it — and an autostart that resumed
    formation would be the machine deciding against its owner.
    """
    from .processing.resource_gate import ResourceGate
    from .storage.evidence import EvidenceStore

    actor = args.actor or settings.owner_principal
    if not actor:
        print("refused: a pause is attributed, and with no owner principal configured "
              "there is nobody to attribute it to", file=sys.stderr)
        return 2
    reason = args.reason or ("the owner lifted the hold" if args.resume
                             else "the owner held this stage")
    state = "active" if args.resume else "paused"
    with EvidenceStore(settings.db_path) as store:
        gate = ResourceGate(store)
        if args.scope == "inference":
            if args.resume:
                gate.resume(actor=actor, reason=reason)
            else:
                gate.pause(actor=actor, reason=reason)
        else:
            store.set_control("global", "delivery", state, actor=actor, reason=reason,
                              policy_version="operator-pause")
        held = {"inference": gate.paused,
                "delivery": store.stage_is_paused("global", "delivery")}
    return _emit({"ok": True, "scope": args.scope, "state": state, "actor": actor,
                  "reason": reason, "held": held,
                  "survives_a_restart": True})


# -- the commands ------------------------------------------------------------

def _status_command(settings) -> int:
    from .operations.status import StatusReporter

    present = settings.db_path.exists()
    report = {**_configuration(settings),
              "capabilities": _capabilities(settings, present),
              "stages": None}
    if not present:
        report["note"] = "no canonical store yet; run `hermes-memory init`"
        return _emit(report)
    try:
        with ReadOnlyStore(settings.db_path) as store:
            report["stages"] = StatusReporter(store, settings=settings).report()
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
    from .operations.doctor import Doctor, unreachable_store_report

    if not settings.db_path.exists():
        return _emit(unreachable_store_report(
            settings.db_path, "the canonical store does not exist yet"), 1)
    try:
        with ReadOnlyStore(settings.db_path) as store:
            report = Doctor(store, settings=settings).examine(
                connectivity=args.probe, synthetic=args.synthetic_probe,
                profile=str(settings.home))
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
