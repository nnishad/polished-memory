"""Verify and restore a real canonical snapshot into an isolated rehearsal only.

Run with the target release's interpreter. Never opens the live canonical store.
The rehearsal directory holds private evidence and is intentionally retained 0700.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from hermes_memory.lifecycle.snapshots import Snapshots
from hermes_memory.lifecycle.recovery import Recovery
from hermes_memory.storage.evidence import EvidenceStore

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--into", type=Path, required=True)
args = parser.parse_args()
source = args.snapshot.resolve()
target = args.into.resolve()
if target.exists():
    raise SystemExit("refusing an existing rehearsal directory")
manifest_files = list(source.glob("snap_*.json"))
assert len(manifest_files) == 1
manifest = json.loads(manifest_files[0].read_text())
assert hashlib.sha256((source / "canonical.db").read_bytes()).hexdigest() == manifest["checksum"]
target.mkdir(mode=0o700, parents=True)
copied = target / "snapshots" / source.name
shutil.copytree(source, copied)
with EvidenceStore(target / "canonical.db") as store:
    snapshots = Snapshots(store, directory=target / "snapshots")
    checked = snapshots.verify(manifest["id"])
    assert checked["ok"], checked["problems"]
    recovery = Recovery(store, snapshots=snapshots, owner_principal="upgrade-rehearsal")
    recovery.restore(manifest["id"], actor="upgrade-rehearsal")
    integrity = recovery.integrity()
    assert integrity["ok"], integrity["problems"]
    assert integrity["records"] == manifest["records"]
    assert integrity["tombstones"] == manifest["tombstones"]
    print(json.dumps({"ok": True, "isolated_directory": str(target),
                      "snapshot": manifest["id"], "records": integrity["records"],
                      "tombstones": integrity["tombstones"],
                      "migrations": store.db.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]},
                     indent=2))
