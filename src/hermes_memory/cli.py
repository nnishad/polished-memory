"""Operator CLI.

Readiness is reported per stage; 'configured' is never reported as 'operational'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import SettingError, load_settings
from .storage.evidence import EvidenceStore

__all__ = ["main"]


def _status(settings) -> dict:
    from importlib.util import find_spec

    return {
        "data_dir": str(settings.data_dir),
        "database": str(settings.db_path),
        "database_present": settings.db_path.exists(),
        "inference_enabled": settings.inference_enabled,
        "capture_only": settings.capture_only,
        "hindsight_route": settings.hindsight_url or "unset",
        "approved_inference_hosts": sorted(settings.allowed_inference_hosts),
        "background_budget_tokens": settings.background_budget_tokens,
        # Absent backend means degraded capture/lexical memory, never a failed startup.
        "hindsight_client_importable": find_spec("hindsight_client") is not None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-memory")
    parser.add_argument("--env-file", help="owned env file (default: $HERMES_MEMORY_HOME/hermes-memory.env)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="print effective configuration and component readiness")
    init = sub.add_parser("init", help="create and migrate the canonical store")
    init.add_argument("--dry-run", action="store_true")
    sub.add_parser("doctor", help="read-only checks; performs no inference")

    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.env_file)
    except SettingError as error:
        print(f"configuration refused: {error}", file=sys.stderr)
        return 2

    if args.command == "status":
        print(json.dumps(_status(settings), indent=2, sort_keys=True))
        return 0

    if args.command == "init":
        if args.dry_run:
            print(json.dumps({"would_create": [str(settings.data_dir), str(settings.db_path)]}, indent=2))
            return 0
        settings.data_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        settings.blob_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        with EvidenceStore(settings.db_path) as store:
            store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        print(json.dumps({"initialized": str(settings.db_path)}, indent=2))
        return 0

    # doctor: read-only, no network, no model calls.
    problems: list[str] = []
    if not Path(settings.data_dir).exists():
        problems.append("data directory does not exist; run `hermes-memory init`")
    report = {**_status(settings), "problems": problems}
    report["ready_for_capture"] = not problems
    report["ready_for_formation"] = not problems and not settings.capture_only
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not problems else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
