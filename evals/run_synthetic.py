#!/usr/bin/env python3
"""§12.3/§12.4: the synthetic evaluation run, in one command.

    uv run python evals/run_synthetic.py                 # the feature-quality report
    uv run python evals/run_synthetic.py --json out.json  # the same thing, machine-shaped

Exit status is the release gate: zero only when every *measured* row passed. Rows that cannot
be measured on this build are printed, counted and named — they do not pass and they do not
fail, because the honest answer to "did the extraction quality gate hold?" on a machine with
inference switched off is "that was not measured here", and a report that said either yes or no
would be the liar.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

sys.path[:0] = [str(Path(__file__).resolve().parent / "cases"),
                str(Path(__file__).resolve().parent / "fixtures")]

import checks                      # noqa: E402 - the question set, beside this runner
import corpus                      # noqa: E402 - the generator that answers to it

# Order matters: the corpus is ingested once and later rows read the store the earlier ones
# wrote, so a reorder is a different measurement rather than a faster one.
SUITE: list[Callable[[checks.Environment], list[dict[str, Any]]]] = [
    checks.check_corpus,
    checks.check_capture,
    checks.check_corrections,
    checks.check_measurements,
    checks.check_retrieval,
    checks.check_retrieval_ablation,
    checks.check_latency,
    checks.check_proactivity,
    checks.check_privacy,
    checks.check_compute,
    checks.check_operations,
    checks.check_install,
    checks.check_recovery,
]


def evaluate(root: Path) -> dict[str, Any]:
    """Run every check over one scratch installation and hand back the whole report."""
    env = checks.Environment(root=root)
    rows: list[dict[str, Any]] = []
    try:
        env.timings["ingest"] = checks.ingest(env)["seconds"]
        for check in SUITE:
            rows.extend(check(env))
        rows.extend(checks.not_measured_rows())
    finally:
        env.close()
    measured = [row for row in rows if row["status"] == checks.MEASURED]
    failed = [row for row in measured if not row["pass"]]
    by_row: dict[str, dict[str, int]] = {}
    for row in rows:
        tally = by_row.setdefault(row["row"], {"measured": 0, "passed": 0, "failed": 0,
                                               "not-measured": 0})
        key = "not-measured" if row["status"] == checks.NOT_MEASURED else (
            "passed" if row.get("pass") else "failed")
        tally["measured"] += 1 if row["status"] == checks.MEASURED else 0
        tally[key] += 1
    return {"harness_version": "synthetic-1", "corpus": corpus.totals(),
            "summary": {"checks": len(rows), "measured": len(measured),
                        "passed": len(measured) - len(failed), "failed": len(failed),
                        "not_measured": len(rows) - len(measured),
                        "seconds": env.timings.get("ingest")},
            "rows": by_row,
            "failures": [{"row": row["row"], "check": row["check"], "value": row["value"],
                          "criterion": row["criterion"], "detail": row["detail"]}
                         for row in failed],
            "results": rows,
            "ok": not failed}


def table(report: dict[str, Any]) -> str:
    """The same numbers, for a human reading a terminal.

    Grouped by the §12.4 area rather than by the order the checks ran, because the plan is
    written as a table of release criteria and that is what this has to be checked against.
    """
    lines = [f"synthetic evaluation — {report['summary']['measured']} measured, "
             f"{report['summary']['not_measured']} not measured, "
             f"{report['summary']['failed']} failed"]
    order: list[str] = []
    for row in report["results"]:
        if row["row"] not in order:
            order.append(row["row"])
    for area in order:
        tally = report["rows"][area]
        aside = (f", {tally['not-measured']} not measured" if tally["not-measured"] else "")
        lines.append(f"\n{area}  [{tally['passed']}/{tally['measured']} measured{aside}]")
        for row in report["results"]:
            if row["row"] != area:
                continue
            if row["status"] == checks.NOT_MEASURED:
                lines.append(f"   --   {row['check']}")
                lines.append(f"        waits on: {row['requires']}")
                continue
            mark = "ok  " if row["pass"] else "FAIL"
            value = row["value"]
            shown = f"{value:.4g}" if isinstance(value, float) else str(value)
            lines.append(f"   {mark} {row['check']} = {shown} "
                         f"(of {row['denominator']}) {row['criterion']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", metavar="PATH", type=Path,
                        help="write the machine-readable report here")
    parser.add_argument("--root", metavar="DIR", type=Path,
                        help="scratch directory to build the installation in; a temporary "
                             "one is removed afterwards unless this is given")
    parser.add_argument("--quiet", action="store_true", help="print only the summary line")
    arguments = parser.parse_args(argv)
    temporary = arguments.root is None
    root = arguments.root or Path(tempfile.mkdtemp(prefix="hermes-evals-"))
    if not temporary and root.exists() and any(root.iterdir()):
        # Refused rather than wiped: a rerun over a dirty archive would measure last run's
        # tombstones, and "rm -rf" on a path somebody else typed is not this command's call.
        print(f"refused: {root} is not empty. Point --root somewhere new, or clear it "
              "yourself — a report built over a previous run's store is not this run's "
              "measurement.", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)
    report = evaluate(root)
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str)
                                  + "\n", encoding="utf-8")
    if not arguments.quiet:
        print(table(report))
    summary = report["summary"]
    print(f"\n{summary['passed']} of {summary['measured']} measured checks passed; "
          f"{summary['not_measured']} named as not measured"
          f"{'' if arguments.quiet else ' — report: ' + str(root)}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
