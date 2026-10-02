"""Reviewed, quiescent release activation through owned services and host installers.

An unsuccessful publication is left stopped for forward repair. Never automatically
start an older binary against a migrated store or silently restore an older corpus.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

from ..ids import content_digest, digest, now
from .release import MANIFEST, _revision, _source_fingerprints, verify
from .services import Services, plan as services_plan, UNITS
from .upgrade import UpgradeError, profile_stores, _in_flight

GATEWAY = "hermes-gateway.service"
PROBE = """import json
from hermes_memory.install.compatibility import _framework_digest
from hermes_memory.storage.migrations import MIGRATIONS
from importlib.metadata import version
print(json.dumps({'digest':_framework_digest(),'schema':len(MIGRATIONS)}))
"""
MIGRATE = """import sys,json
from hermes_memory.storage.evidence import EvidenceStore
rows=[]
for path in sys.argv[1:]:
 with EvidenceStore(path) as store:
  integrity=store.db.execute('PRAGMA integrity_check').fetchone()[0]
  foreign=store.db.execute('PRAGMA foreign_key_check').fetchall()
  if integrity!='ok' or foreign: raise RuntimeError('migration integrity failed')
  rows.append({'schema':store.db.execute('SELECT count(*) FROM schema_migrations').fetchone()[0]})
print(json.dumps(rows))
"""


def _run(argv, *, env, runner=None, timeout=120):
    result = (runner or subprocess.run)(list(map(str, argv)), env=env,
                                       capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        # Command failures can contain tokens; preserve diagnostics locally, not in a
        # tool-facing exception. The failing executable/phase is enough for the caller.
        raise UpgradeError(f"activation command failed: {Path(str(argv[0])).name} ({result.returncode})")
    return result.stdout


def _environment(settings, target, host_home):
    return {**os.environ, "HERMES_MEMORY_HOME": str(settings.home),
            "HERMES_MEMORY_RELEASE": str(target), "HERMES_HOME": str(host_home)}


def activation_plan(settings, *, release, hermes_home, hermes_source, runner=None):
    target = Path(release).expanduser().resolve()
    home = Path(settings.home).resolve()
    host = Path(hermes_home).expanduser().resolve()
    host_source = Path(hermes_source).expanduser().resolve()
    pointer = home / "runtime/current"
    blocking = []
    if not target.is_relative_to(home / "runtime") or target == pointer.resolve():
        blocking.append("target must be a distinct staged release under the instance runtime directory")
    if not pointer.is_symlink():
        blocking.append("current release must be an existing managed pointer")
    if not (host / "config.yaml").is_file() or not (host_source / "venv/bin/hermes").is_file():
        blocking.append("explicit Hermes home and installed source/venv are required")
    env = _environment(settings, target, host)
    checked = verify(settings=settings, into=target, environ=env)
    if not checked["complete"]:
        blocking.extend(checked["missing"])
    record = json.loads((target / MANIFEST).read_text()) if (target / MANIFEST).is_file() else {}
    registration = Path(record.get("registration_source") or target / "source")
    revision = record.get("registration_commit")
    if (not revision or _revision(registration) != (revision, False)
            or _source_fingerprints(registration) != record.get("source_fingerprints")):
        blocking.append("release has no clean, matching immutable host-registration source")
    units_env = {**env, "HERMES_MEMORY_RELEASE": str(pointer)}
    units = services_plan(settings, environ=units_env,
                          templates=target / "deployment/systemd")
    if units["blocked"] or units["would_change"] or units["missing_paths"]:
        blocking.append("owned service topology must already match the reviewed target templates")
    inflight = _in_flight(settings)
    if inflight["held"] or inflight["unreadable"]:
        blocking.append("model admission is occupied or unreadable; reconcile before activation")
    probes = []
    if not blocking:
        for interpreter in (target / "bin/python", target / "hindsight/bin/python"):
            probes.append(json.loads(_run([interpreter, "-c", PROBE], env=env, runner=runner)))
        shipped = json.loads((target / "deployment/compatibility.json").read_text())
        if probes[0] != probes[1] or probes[0]["digest"] != shipped["framework_digest"]:
            blocking.append("runtime, worker and shipped compatibility digests disagree")
    stores = [str(path.resolve()) for _, path in profile_stores(settings) if path.is_file()]
    if not stores:
        blocking.append("no canonical store exists for migration rehearsal")
    schemas = []
    for path in stores:
        with sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True) as db:
            schemas.append(db.execute("SELECT count(*) FROM schema_migrations").fetchone()[0])
    if probes and any(schema > probes[0]["schema"] for schema in schemas):
        blocking.append("target cannot open all current schemas; code downgrade is forbidden")
    gateway = (runner or subprocess.run)(["systemctl", "--user", "show", GATEWAY,
        "--property=Environment", "--property=WorkingDirectory", "--property=ExecStart"],
        capture_output=True, text=True, timeout=30)
    if gateway.returncode or (f"HERMES_HOME={host}" not in gateway.stdout
                             or str(host_source / "venv/bin/python") not in gateway.stdout):
        blocking.append("gateway ownership does not match the explicit Hermes home/source")
    files = [home / "hermes-memory.env", home / "hindsight.env", host / "config.yaml",
             host / "plugins/.install-metadata.json"]
    fingerprints = [[str(path), content_digest(path.read_bytes())] for path in files if path.is_file()]
    proposal = {"target": str(target), "previous": str(pointer.resolve()), "host": str(host),
                "host_source": str(host_source), "registration": str(registration),
                "registration_commit": revision, "wheel_digest": record.get("wheel_digest"),
                "source_digest": record.get("source_digest"), "probes": probes,
                "stores": stores, "schemas": schemas, "fingerprints": fingerprints,
                "gateway_fingerprint": content_digest(gateway.stdout.encode())}
    return {**proposal, "blocking": blocking, "review_digest": digest(["activation-v1", proposal])}


def activate_release(settings, *, release, hermes_home, hermes_source, actor, review, runner=None):
    if not settings.owner_principal or actor != settings.owner_principal:
        raise UpgradeError("only the configured owner may activate a release")
    home = Path(settings.home).resolve()
    lock = home / "activation.lock"
    with lock.open("a+") as handle:
        os.chmod(lock, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise UpgradeError("another activation is in progress") from None
        proposal = activation_plan(settings, release=release, hermes_home=hermes_home,
                                   hermes_source=hermes_source, runner=runner)
        if proposal["blocking"] or proposal["review_digest"] != review:
            raise UpgradeError("activation refused: blockers or changed review; re-plan")
        target, host = Path(proposal["target"]), Path(proposal["host"])
        source = Path(proposal["host_source"])
        env = _environment(settings, target, host)
        backup_root = home / "activation-backups"
        backup_root.mkdir(mode=0o700, exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix="activation-", dir=backup_root))
        receipt = {"started_at": now(), "actor": actor, "plan": proposal,
                   "state": "planned", "backup": str(backup), "published": False}

        def record(state):
            receipt["state"] = state
            temporary = backup / "receipt.next.json"
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.chmod(0o600)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, backup / "receipt.json")

        def service_run(argv):
            completed = (runner or subprocess.run)(list(argv), capture_output=True,
                                                   text=True, timeout=120)
            return completed.returncode, completed.stdout

        services = Services(settings, runner=service_run)
        record("quiescing")
        try:
            _run(["systemctl", "--user", "stop", GATEWAY], env=env, runner=runner)
            services.stop()
            for unit in (*UNITS, GATEWAY):
                state = (runner or subprocess.run)(["systemctl", "--user", "is-active", unit],
                    capture_output=True, text=True, timeout=30)
                if state.stdout.strip() not in {"inactive", "failed"}:
                    raise UpgradeError("owned services did not quiesce")
            inflight = _in_flight(settings)
            if inflight["held"] or inflight["unreadable"]:
                raise UpgradeError("unsettled inference remains after quiescence")
            copied = []
            for index, path in enumerate(proposal["stores"]):
                copy = backup / f"canonical-{index}.db"
                with sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True) as donor:
                    with sqlite3.connect(copy) as destination:
                        donor.backup(destination)
                copy.chmod(0o600)
                rehearsal = backup / f"rehearsal-{index}.db"
                shutil.copy2(copy, rehearsal)
                copied.append(str(rehearsal))
            for path in (home / "gate.db", home / "installation.db"):
                if path.is_file():
                    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as donor:
                        with sqlite3.connect(backup / path.name) as destination:
                            donor.backup(destination)
                    (backup / path.name).chmod(0o600)
            for relative in ("config.yaml", ".env", "plugins/.install-metadata.json"):
                path = host / relative
                if path.is_file():
                    destination = backup / "hermes-home" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    shutil.copy2(path, destination)
                    destination.chmod(0o600)
            plugin = host / "plugins/hermes-memory"
            if plugin.is_dir():
                shutil.copytree(plugin, backup / "prior-plugin", symlinks=True)
            record("backed-up")
            _run([target / "bin/python", "-c", MIGRATE, *copied], env=env, runner=runner)
            record("rehearsed")
            wheel = Path(json.loads((target / MANIFEST).read_text())["wheel"])
            if content_digest(wheel.read_bytes()) != proposal["wheel_digest"]:
                raise UpgradeError("release wheel changed after review")
            _run(["uv", "pip", "install", "--python", source / "venv/bin/python",
                  "--reinstall", "--no-deps", wheel], env=env, runner=runner)
            _run([source / "venv/bin/hermes", "plugins", "install",
                  "file://" + proposal["registration"] + "#integrations/hermes-memory",
                  "--ref", proposal["registration_commit"], "--force", "--no-enable"],
                 env=env, runner=runner)
            _run([source / "venv/bin/hermes", "plugins", "enable", "hermes-memory",
                  "--no-allow-tool-override"], env=env, runner=runner)
            host_probe = json.loads(_run([source / "venv/bin/python", "-c", PROBE], env=env, runner=runner))
            if host_probe != proposal["probes"][0]:
                raise UpgradeError("installed host framework differs from reviewed target")
            metadata = json.loads((host / "plugins/.install-metadata.json").read_text())
            if metadata.get("hermes-memory", {}).get("revision") != proposal["registration_commit"]:
                raise UpgradeError("host plugin registration did not preserve the reviewed git pin")
            record("host-paired")
            _run([target / "bin/python", "-c", MIGRATE, *proposal["stores"]], env=env, runner=runner)
            pointer = home / "runtime/current"
            temporary = home / "runtime/current-activation-next"
            if temporary.exists() or temporary.is_symlink():
                raise UpgradeError("an unfinished pointer publication requires reconciliation")
            temporary.symlink_to(target)
            os.replace(temporary, pointer)
            receipt["published"] = True
            record("published")
            services.start()
            _run(["systemctl", "--user", "start", GATEWAY], env=env, runner=runner)
            record("started-unverified")
            return receipt
        except BaseException:
            try:
                _run(["systemctl", "--user", "stop", GATEWAY], env=env, runner=runner)
                services.stop()
            except Exception as stop_error:
                receipt["stop_failure"] = type(stop_error).__name__
            record("failed-stopped-forward-repair-required")
            # In particular, do not restart the older release against a newer schema.
            raise
