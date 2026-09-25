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
from typing import Any, Callable

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
        "delivery": store_present and bool(settings.owner_principal),
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
    return _explain_command(settings, args)


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
