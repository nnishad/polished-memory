"""C4 snapshots: a copy of the store that can be taken back without taking back.

A snapshot is one SQLite file written through the backup API, so it is consistent
without the application being quiescent, and it carries its attachment bytes with it —
blobs live in the database, so there is no second place for a restore to disagree
with. Each one gets a sidecar manifest holding the epoch, the journal position and a
digest of the file, which is what makes "is this snapshot the one I was told about"
an answerable question rather than a hope.

The part that is not a file copy is what a restore must *not* undo. Forgetting is an
irrevocable act, so the erasure ledger, its outstanding obligations and the tombstones
are lifted out of the live database and re-applied over the restored copy before it is
ever opened. A rollback that quietly restored a forgotten email would not be a
recovery; it would be a leak with a receipt.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..ids import digest, now
from ..storage.evidence import EvidenceError
from ..storage.migrations import MIGRATIONS

__all__ = ["Snapshots", "Snapshot", "MANIFEST_SUFFIX"]

MANIFEST_SUFFIX = ".json"
SNAPSHOT_NAME = "canonical.db"
MIN_KEEP = 1
EXPECTED_SCHEMA = len(MIGRATIONS)


@dataclass(frozen=True)
class Snapshot:
    id: str
    path: Path
    manifest: Path
    created_at: str
    reason: str
    actor: str
    epoch: int
    seq: int
    records: int
    tombstones: int
    size_bytes: int
    checksum: str

    @property
    def database(self) -> Path:
        return self.path / SNAPSHOT_NAME

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "created_at": self.created_at, "reason": self.reason,
                "actor": self.actor, "epoch": self.epoch, "seq": self.seq,
                "records": self.records, "tombstones": self.tombstones,
                "size_bytes": self.size_bytes, "checksum": self.checksum,
                "directory": str(self.path)}


class Snapshots:
    """Owned files under ``<data>/snapshots``. Nothing outside it is ever touched."""

    def __init__(self, store, *, directory: str | Path):
        self.store = store
        self.db = store.db
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    # -- writing -------------------------------------------------------------

    def create(self, *, reason: str, actor: str) -> dict[str, Any]:
        """Copy the live store into its own directory and record what it was."""
        words = _text(reason, "reason", 300)
        teller = _text(actor, "actor", 120)
        stamp = now()
        snapshot_id = "snap_" + digest([stamp, words, id(self)])[:16]
        target = self._path_for(snapshot_id)
        if target.exists():
            raise EvidenceError(f"snapshot {target} already exists")
        target.mkdir(parents=True, mode=0o700)
        database = target / SNAPSHOT_NAME
        # The backup API rather than a file copy: the application keeps writing
        # while this runs, and a torn copy is worse than no copy at all.
        destination = _sealed(database)
        try:
            self.db.backup(destination)
            _seal(destination)
            head = _head(destination)
        finally:
            destination.close()
        # A snapshot holds every private thing the store knew, in plain text, in a
        # directory an operator may well point a backup tool at.
        database.chmod(0o600)
        # After the writer has closed: a digest of a file still being written
        # describes a state the snapshot was never in, and would fail its own
        # verify a moment later.
        checksum = _checksum(database)
        manifest = {
            "id": snapshot_id, "created_at": stamp, "reason": words, "actor": teller,
            "epoch": head["epoch"], "seq": head["seq"], "records": head["records"],
            "tombstones": head["tombstones"], "size_bytes": database.stat().st_size,
            "checksum": checksum, "schema": head["schema"],
            "journal_tail": _tail(head["seq"]),
        }
        _write_json(target / (snapshot_id + MANIFEST_SUFFIX), manifest)
        self.store._audit("snapshot_create", snapshot_id,
                          {"actor": teller, "epoch": head["epoch"],
                           "records": head["records"], "reason": words[:200]})
        return {"snapshot": _snapshot(target, manifest), "created": True}

    # -- reading -------------------------------------------------------------

    def list(self, *, limit: int = 20) -> list[Snapshot]:
        """Every snapshot this directory actually holds, newest first.

        Found by reading the manifests rather than by name, so a directory that is
        not a snapshot — the pre-restore guards, for one — is skipped on evidence
        rather than on a naming convention.
        """
        found: list[Snapshot] = []
        for candidate in sorted((path for path in self.root.iterdir() if path.is_dir()),
                                reverse=True):
            manifest = _read_manifest(candidate)
            if manifest is None:
                continue
            found.append(_snapshot(candidate, manifest))
            if len(found) >= _bounded(limit):
                break
        return found

    def resolve(self, snapshot_id: str) -> Snapshot:
        for item in self.list(limit=500):
            if item.id == snapshot_id:
                return item
        raise EvidenceError(
            f"no snapshot {snapshot_id!r} in {self.root}; a restore has to name a "
            "snapshot this directory actually holds")

    def verify(self, snapshot_id: str) -> dict[str, Any]:
        """Is this snapshot whole, and is it the one the manifest describes?

        Checks the file digest, SQLite's own integrity pass, the foreign-key sweep,
        and that the epoch and journal position match the record made at creation.

        Anything in ``problems`` blocks a restore. A ``note`` is something worth
        knowing that does not: an older schema is a perfectly good thing to roll back
        to, this build just has to bring it forward first.
        """
        item = self.resolve(snapshot_id)
        problems: list[str] = []
        notes: list[str] = []
        head: dict[str, Any] = {}
        if not item.database.exists():
            problems.append("the snapshot database file is missing")
        elif _checksum(item.database) != item.checksum:
            problems.append("the file no longer matches the digest recorded at creation; "
                            "it has been altered since")
        else:
            connection = open_read_only(item.database)
            try:
                head = _head(connection)
                integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
                if integrity != "ok":
                    problems.append(f"sqlite's own integrity pass said {integrity!r}")
                foreign = connection.execute("PRAGMA foreign_key_check").fetchall()
                if foreign:
                    problems.append(f"{len(foreign)} foreign-key violation(s) inside")
                applied = int(connection.execute(
                    "SELECT count(*) FROM schema_migrations").fetchone()[0])
                if applied < EXPECTED_SCHEMA:
                    notes.append(f"the snapshot carries {applied} of {EXPECTED_SCHEMA} "
                                 "migrations; restoring brings it forward to this build")
            except sqlite3.Error as error:
                problems.append(f"unreadable: {str(error)[:160]}")
                head = {}
            finally:
                connection.close()
            if head:
                for key, expected in (("epoch", item.epoch), ("seq", item.seq),
                                      ("records", item.records),
                                      ("tombstones", item.tombstones)):
                    if head[key] != expected:
                        problems.append(f"{key} is {head[key]}, the manifest said "
                                        f"{expected}")
        return {"id": item.id, "ok": not problems, "problems": problems, "notes": notes,
                **head}

    # -- removing ------------------------------------------------------------

    def prune(self, *, keep: int = MIN_KEEP, actor: str) -> dict[str, Any]:
        """Drop old snapshots, newest first, never the only recent one.

        The live database is not touched here: a snapshot directory is the only thing
        this class was given, and deleting anything else would be an uninstall
        pretending to be housekeeping.
        """
        retained = keep if isinstance(keep, int) and keep >= MIN_KEEP else MIN_KEEP
        items = self.list(limit=500)
        removed: list[str] = []
        for item in items[retained:]:
            # A directory listed here can be a symlink out of the owned tree, and a
            # recursive delete follows it. Resolve before handing anything to one.
            if item.path.resolve().parent != self.root.resolve():
                raise EvidenceError(f"snapshot {item.id} is outside {self.root}; refusing")
            _remove_tree(item.path)
            removed.append(item.id)
        self.store._audit("snapshot_prune", f"keep:{retained}",
                          {"actor": _text(actor, "actor", 120), "removed": removed})
        return {"kept": retained, "removed": removed}


    def _path_for(self, snapshot_id: str) -> Path:
        # Microseconds, not seconds: pruning keeps "the newest N by directory name",
        # so two snapshots taken in the same second would otherwise order by digest.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        return self.root / f"{stamp}-{snapshot_id}"


def _snapshot(path: Path, manifest: dict[str, Any]) -> Snapshot:
    return Snapshot(id=str(manifest["id"]), path=path,
                    manifest=path / (str(manifest["id"]) + MANIFEST_SUFFIX),
                    created_at=str(manifest["created_at"]), reason=str(manifest["reason"]),
                    actor=str(manifest["actor"]), epoch=int(manifest["epoch"]),
                    seq=int(manifest["seq"]), records=int(manifest["records"]),
                    tombstones=int(manifest["tombstones"]),
                    size_bytes=int(manifest["size_bytes"]),
                    checksum=str(manifest["checksum"]))


def _head(connection: sqlite3.Connection) -> dict[str, Any]:
    """The four numbers that say which moment a database is."""
    def count(sql: str, default: int = 0) -> int:
        try:
            return int(connection.execute(sql).fetchone()[0] or default)
        except sqlite3.Error:
            return default

    return {"epoch": count("SELECT value FROM memory_epoch WHERE id=1", 1),
            "seq": count("SELECT COALESCE(max(seq), 0) FROM change_journal"),
            "records": count("SELECT count(*) FROM records WHERE deleted=0"),
            "tombstones": count("SELECT count(*) FROM tombstones"),
            "schema": count("SELECT count(*) FROM schema_migrations")}


def _tail(seq: int) -> str:
    return f"journal through {seq}"


def _checksum(path: Path) -> str:
    hash_ = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hash_.update(block)
    return hash_.hexdigest()


def _sealed(path: Path) -> sqlite3.Connection:
    """Open a new file to receive a snapshot copy."""
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _seal(connection: sqlite3.Connection) -> None:
    """Make a copied database rollback-journal, which is what a snapshot must be.

    The copy arrives carrying page 1 of the live store, and page 1 is where the
    journal mode lives: an unsealed snapshot is a WAL database whose committed pages
    sit in a sidecar the recorded digest never covered.
    """
    connection.execute("PRAGMA journal_mode=DELETE")


def open_read_only(path: Path) -> sqlite3.Connection:
    """Open a sealed snapshot without being able to alter it.

    Every read of a snapshot — verifying it, or putting it into the live store —
    has to leave the bytes alone, because the recorded digest is the only proof
    that the copy is the one that was taken.
    """
    try:
        connection = sqlite3.connect(Path(path).absolute().as_uri() + "?mode=ro",
                                     uri=True, timeout=10)
    except sqlite3.Error as error:
        raise EvidenceError(f"{path} could not be opened for reading: "
                            f"{str(error)[:160]}") from None
    connection.row_factory = sqlite3.Row
    return connection


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=1), encoding="utf-8")
    path.chmod(0o600)


def _read_manifest(directory: Path) -> dict[str, Any] | None:
    for candidate in directory.glob("*" + MANIFEST_SUFFIX):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # An unreadable manifest is not a snapshot. Restoring from a file whose
            # provenance cannot be checked is how the wrong data comes back.
            return None
        if isinstance(payload, dict) and {"id", "checksum", "epoch"} <= set(payload):
            return payload
    return None


def _remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        child.unlink(missing_ok=True)
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_dir():
            child.rmdir()
    if path.exists():
        path.rmdir()


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise EvidenceError(f"{label} must be nonempty text of at most {maximum} characters")
    return " ".join(value.split())


def _bounded(value: int) -> int:
    return value if isinstance(value, int) and 1 <= value <= 500 else 20
