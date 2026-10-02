"""Activate an explicitly named, verified release with a quiescent private host backup.

Operational procedure, not an installed-package patch. Uses the official plugin installer
and immutable release builder output. Never restores or deletes canonical records.
"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path


UNITS = ("hermes-gateway", "hermes-memory", "hermes-memory-worker", "hermes-memory-hindsight")


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--hermes-source", type=Path, required=True)
    parser.add_argument("--hermes-home", type=Path, required=True)
    parser.add_argument("--build-source", type=Path, required=True)
    args = parser.parse_args()
    release, backup = args.release.resolve(), args.backup.resolve()
    record = json.loads((release / "RELEASE.json").read_text())
    assert record["source_commit"] == release.name and (release / "bin/hermes-memory").is_file()
    instance = release.parent.parent
    pointer = instance / "runtime/current"
    assert pointer.is_symlink() and not backup.exists()
    previous = str(pointer.resolve())
    env = dict(os.environ, HERMES_MEMORY_RELEASE=str(release), HERMES_MEMORY_HOME=str(instance),
               HERMES_HOME=str(args.hermes_home.resolve()))
    run([str(release / "bin/hermes-memory"), "compatibility", "--digests"], env=env)
    proposal = subprocess.run([str(release / "bin/hermes-memory"), "upgrade", "--version", str(release),
                               "--hermes-home", str(args.hermes_home)], check=True,
                              capture_output=True, text=True, env=env)
    assert not json.loads(proposal.stdout)["blocking"], "upgrade preflight is blocked"
    backup.mkdir(mode=0o700, parents=True)
    stopped = False
    try:
        # The gateway's normal shutdown drains capture before the canonical writer stops.
        run(["systemctl", "--user", "stop", "hermes-gateway"])
        stopped = True
        run(["systemctl", "--user", "stop", *UNITS[1:]])
        for unit in UNITS:
            state = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True)
            assert state.stdout.strip() in ("inactive", "failed"), (unit, state.stdout)
        shutil.copytree(args.hermes_home, backup / "hermes-home", symlinks=True)
        shutil.copytree(instance, backup / "memory-instance", symlinks=True,
                        ignore=shutil.ignore_patterns("runtime", "build-source-*", "__pycache__"))
        canonical = instance / "data/canonical.db"
        with sqlite3.connect(f"file:{canonical}?mode=ro", uri=True) as db:
            before = {"records": db.execute("SELECT count(*) FROM records WHERE deleted=0").fetchone()[0],
                      "tombstones": db.execute("SELECT count(*) FROM tombstones").fetchone()[0]}
        (backup / "activation.json").write_text(json.dumps({"previous": previous,
            "target": str(release), "before": before, "canonical_schema": "requires coherent backup restore for downgrade"}, indent=2))
        source = "file://" + str(args.build_source.resolve()) + "#integrations/hermes-memory"
        run([str(args.hermes_source / "venv/bin/hermes"), "plugins", "install", source,
             "--ref", release.name, "--force", "--enable"], env=env)
        wheel = list((release / "wheel").glob("*.whl"))
        assert len(wheel) == 1
        run(["uv", "pip", "install", "--python", str(args.hermes_source / "venv/bin/python"),
             "--reinstall", "--no-deps", str(wheel[0])])
        # No file is patched inside a release. Publish the already built target atomically.
        temporary = pointer.with_name("current-clarification-next")
        assert not temporary.exists() and not temporary.is_symlink()
        temporary.symlink_to(release)
        os.replace(temporary, pointer)
        migrate = (
            "import json; from hermes_memory.storage.evidence import EvidenceStore; "
            f"s=EvidenceStore({str(canonical)!r}); "
            "assert s.db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'; "
            "print(json.dumps({'records':s.db.execute('SELECT count(*) FROM records WHERE deleted=0').fetchone()[0],"
            "'tombstones':s.db.execute('SELECT count(*) FROM tombstones').fetchone()[0]})); s.close()"
        )
        migrated = subprocess.run([str(release / "bin/python"), "-c", migrate], check=True,
                                   capture_output=True, text=True, env=env)
        assert json.loads(migrated.stdout) == before
        print(json.dumps({"activated": str(release), "private_backup": str(backup), "preserved": before}))
    finally:
        # After migration an old binary must not be started against the new schema.
        # A failure after publication leaves the new pointer in place for diagnosis.
        if stopped:
            run(["systemctl", "--user", "start", *UNITS])


if __name__ == "__main__":
    main()
