"""Instance-level installation state: which Hermes profile owns which memory.

One installation serves several Hermes profiles. They share a physical machine, a GPU
and — if it is ever admitted twice — the same model, so admission must be coordinated
once for the instance; but their evidence, banks, policies and outboxes must not be.
This module is the mapping that makes both of those statements true at the same time,
and it deliberately holds no evidence: it says where a profile's memory lives, not
what is in it.

Two rules here are the ones the retired installation got wrong. A profile is resolved
from the home of the *activity* that is asking, never from a home cached at process
start — one gateway serves many profiles, and the first one to connect is not the
owner of the rest. And an unknown home is an error rather than a fallback: reading the
default profile's configuration after a scoped lookup misses is exactly how a private
conversation ends up filed under somebody else's identity.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..config import scoped_settings
from ..ids import digest, now
from ..storage.migrations import MIGRATION_LEDGER, Migration, connect

__all__ = ["ProfileRegistry", "Profile", "InstallationError", "open_installation",
           "apply_installation_migrations", "INSTALLATION_MIGRATIONS", "PROPOSAL_KEYS",
           "STATE_FILENAME", "DEFAULT_PROFILE"]


class InstallationError(ValueError):
    """Installation state is absent, disputed, or being asked for by the wrong party."""


STATE_FILENAME = "installation.db"
DEFAULT_PROFILE = "default"
_SLUG = re.compile(r"[^a-z0-9-]+")
MAX_LABEL = 200

# What a review approves, and therefore what a receipt records. Advisory fields
# like "does the store already exist" travel alongside the plan but stay out of
# both the digest and the receipt, or a rerun after a half-finished enrollment
# would look like a new decision requiring a new approval.
PROPOSAL_KEYS = ("profile", "hermes_home", "data_dir", "bank_id", "credential_scope")

INSTALLATION_MIGRATIONS: Sequence[Migration] = (
    Migration("0001_profiles", ("""
    -- One row per enrolled Hermes profile. ``hermes_home`` is unique because two
    -- profiles pointing at the same host home would mean two capture owners for one
    -- conversation, which is the failure this table exists to prevent.
    CREATE TABLE profiles(
        profile TEXT PRIMARY KEY,
        hermes_home TEXT NOT NULL UNIQUE,
        data_dir TEXT NOT NULL,
        bank_id TEXT NOT NULL UNIQUE,
        credential_scope TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('enrolled', 'retired')),
        review_digest TEXT NOT NULL,
        enrolled_by TEXT NOT NULL,
        enrolled_at TEXT NOT NULL,
        retired_by TEXT,
        retired_at TEXT,
        retire_reason TEXT
    )""",)),
    Migration("0002_enrollment_receipts", ("""
    -- Enrollment is a reviewed action, so the review is recorded as its own append-only
    -- receipt: what was displayed, who approved it, and what was written as a result.
    -- A rerun that changed nothing must not look like a fresh approval.
    CREATE TABLE enrollment_receipts(
        id TEXT PRIMARY KEY,
        profile TEXT NOT NULL,
        action TEXT NOT NULL,
        actor TEXT NOT NULL,
        review_digest TEXT NOT NULL,
        changes TEXT NOT NULL,
        result TEXT NOT NULL,
        at TEXT NOT NULL
    )""",
     "CREATE INDEX enrollment_receipts_profile ON enrollment_receipts(profile, at)")),
)


def open_installation(path: str | Path) -> sqlite3.Connection:
    """Open the instance ledger, creating it with owner-only permissions.

    A missing parent directory is created at 0700 rather than at the process umask, and
    the file is re-sealed to 0600 on every open: this ledger names where each profile's
    private evidence lives, so a umask that left it group-readable would leak the map
    even though it holds no evidence itself.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    db = connect(target)
    target.chmod(0o600)
    apply_installation_migrations(db)
    return db


def apply_installation_migrations(db: sqlite3.Connection) -> int:
    """Bring the instance ledger to head. Idempotent, and refuses a newer one."""
    db.execute("BEGIN IMMEDIATE")
    try:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND "
                          "name='schema_migrations'").fetchone():
            db.execute(MIGRATION_LEDGER)
        applied = {row[0] for row in db.execute("SELECT name FROM schema_migrations")}
        known = {migration.name for migration in INSTALLATION_MIGRATIONS}
        for name in sorted(applied - known):
            raise InstallationError(
                f"the installation ledger carries migration {name!r} unknown to this "
                "build; upgrade the framework rather than reading a newer ledger with "
                "an older one")
        for migration in INSTALLATION_MIGRATIONS:
            if migration.name in applied:
                continue
            for statement in migration.statements:
                db.execute(statement)
            db.execute("INSERT INTO schema_migrations(name, applied_at) VALUES(?, ?)",
                       (migration.name, now()))
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")
    return len(INSTALLATION_MIGRATIONS)


@dataclass(frozen=True)
class Profile:
    """One enrolled Hermes profile and the memory that belongs to it."""

    profile: str
    hermes_home: Path
    data_dir: Path
    bank_id: str
    credential_scope: str
    state: str
    review_digest: str
    enrolled_by: str
    enrolled_at: str

    @property
    def db_path(self) -> Path:
        return self.data_dir / "canonical.db"

    @property
    def blob_dir(self) -> Path:
        return self.data_dir / "blobs"

    def scoped(self, settings):
        """The instance configuration as this profile's memory.

        Paths, bank and credential scope come from the ledger row; everything the
        operator set about routes and budgets stays as it is.
        """
        return scoped_settings(settings, profile=self.profile, data_dir=self.data_dir,
                               bank_id=self.bank_id, credential_scope=self.credential_scope)

    def as_dict(self, *, private: bool = False) -> dict[str, Any]:
        out = {"profile": self.profile, "bank_id": self.bank_id, "state": self.state,
               "data_dir": str(self.data_dir), "credential_scope": self.credential_scope,
               "enrolled_by": self.enrolled_by, "enrolled_at": self.enrolled_at}
        if private:
            out["hermes_home"] = str(self.hermes_home)
        return out


class ProfileRegistry:
    """The instance's profile map, with enrollment as a reviewed transaction."""

    def __init__(self, db: sqlite3.Connection, *, root: Path, owner_principal: str | None,
                 default_home: Path | None = None, detached: bool = False):
        self.db = db
        self.root = Path(root)
        self.owner_principal = owner_principal
        self.default_home = default_home
        # A detached ledger is a read of an installation that has no ledger yet. It
        # holds nothing, so writing to it would report a success that never happened.
        self.detached = detached

    @classmethod
    def open(cls, settings, *, home: str | Path | None = None) -> "ProfileRegistry":
        root = Path(home or settings.home)
        return cls(open_installation(root / STATE_FILENAME), root=root,
                   owner_principal=settings.owner_principal,
                   default_home=Path(settings.data_dir))

    @classmethod
    def reading(cls, settings, *, home: str | Path | None = None) -> "ProfileRegistry":
        """The ledger as it stands, for a call that has no business creating it.

        An installation that has never enrolled anything has an empty map, not a
        missing one, and a status or a plan that wrote a ledger into being would have
        changed the installation it was only asked about.
        """
        root = Path(home or settings.home)
        if (root / STATE_FILENAME).exists():
            return cls.open(settings, home=root)
        return cls.detached(root=root, owner_principal=settings.owner_principal,
                            default_home=settings.data_dir)

    @classmethod
    def detached(cls, *, root: str | Path, owner_principal: str | None = None,
                 default_home: str | Path | None = None) -> "ProfileRegistry":
        """An empty ledger held in memory, carrying no rows and touching no disk."""
        db = sqlite3.connect(":memory:", isolation_level=None)
        db.row_factory = sqlite3.Row
        apply_installation_migrations(db)
        return cls(db, root=Path(root), owner_principal=owner_principal,
                   default_home=Path(default_home) if default_home else None,
                   detached=True)

    def _writable(self) -> None:
        if self.detached:
            raise InstallationError(
                "this installation has no ledger to write to yet; run "
                "`hermes-memory init` or pass --env-file for the installation that owns "
                f"{self.root}")

    # -- resolution ----------------------------------------------------------

    def resolve(self, hermes_home: str | Path) -> Profile:
        """The *enrolled* profile for this activity's home. There is no fallback.

        An unenrolled home is refused rather than served from the default profile: a
        launch-time cache and a silent default are the two ways the old installation
        answered a question about one person with another person's memory. A retired
        home is refused too — retirement is the owner taking the mapping away, so it
        has to take the access away as well, or the row would only have been renamed.
        """
        home = _home(hermes_home)
        row = self.db.execute("SELECT * FROM profiles WHERE hermes_home=?",
                              (str(home),)).fetchone()
        if row is None:
            raise InstallationError(
                f"no profile is enrolled for {home}; run `hermes-memory enroll "
                "--hermes-home <profile-home>` — the default profile is not a fallback")
        if row["state"] != "enrolled":
            raise InstallationError(
                f"profile {row['profile']!r} was retired for {home} by "
                f"{row['retired_by']} on {row['retired_at']}; its memory stays put but "
                "is not served until the owner enrolls it again")
        return _profile(row)

    def profile(self, name: str) -> Profile:
        """Lookup for the operator, including retired profiles.

        Inventory has to be able to say a profile was retired; only
        :meth:`resolve`, which grants access, treats retirement as terminal.
        """
        wanted = _label(name, "profile")
        row = self.db.execute("SELECT * FROM profiles WHERE profile=?", (wanted,)).fetchone()
        if row is None:
            raise InstallationError(f"profile {wanted!r} is not enrolled")
        return _profile(row)

    def profiles(self, *, include_retired: bool = False) -> list[Profile]:
        where = "" if include_retired else " WHERE state='enrolled'"
        return [_profile(row) for row in self.db.execute(
            f"SELECT * FROM profiles{where} ORDER BY profile").fetchall()]

    def names(self) -> list[str]:
        return [item.profile for item in self.profiles()]

    # -- the plan and its review --------------------------------------------

    def plan(self, profile: str, hermes_home: str | Path) -> dict[str, Any]:
        """What enrolling this profile would change. Nothing is written by this call.

        The digest covers the *proposed state* only: the paths, the bank and the
        credential scope. Whether the store already exists, and whether anything would
        actually move, is reported alongside it but kept out of the approval, or an
        interrupted enrollment could never be resumed — the second run would be asked
        to present an approval of a proposal that had already taken effect.
        """
        name = _label(profile, "profile")
        home = _home(hermes_home)
        data_dir = self._data_dir(name)
        proposal = {"profile": name,
                    "hermes_home": str(home),
                    "data_dir": str(data_dir),
                    "bank_id": self._bank(name),
                    "credential_scope": f"profile-{name}"}
        existing = self.db.execute("SELECT * FROM profiles WHERE profile=?",
                                   (name,)).fetchone()
        return {
            **proposal,
            "review_digest": digest(["profile-enroll-v1", proposal]),
            "store": "created" if not (data_dir / "canonical.db").exists() else "existing",
            "would_write": bool(existing is None or existing["state"] == "retired"
                                or existing["hermes_home"] != str(home)
                                or existing["data_dir"] != str(data_dir)),
            "main_model_unchanged": True,
            "shared_state_untouched": True,
            "note": "enrollment maps a profile to its own memory; it does not "
                    "install a plugin, start a service or select a provider",
        }

    def enroll(self, profile: str, hermes_home: str | Path, *, actor: str,
               review_digest: str) -> dict[str, Any]:
        """Enroll a profile, given a review that matches the proposal exactly.

        Owner-only, and the review digest has to be the one :meth:`plan` produced for
        these arguments. A model that can name a directory is not thereby able to
        enroll it.
        """
        self._owner(actor, "enroll a profile")
        self._writable()
        proposal = self.plan(profile, hermes_home)
        if not isinstance(review_digest, str) or not review_digest.strip():
            raise InstallationError("enrollment needs the digest of the review that was "
                                    "shown; there is no default approval")
        if review_digest != proposal["review_digest"]:
            raise InstallationError(
                "the approval does not match what would change now; show the current "
                "plan and have it approved again")
        if not proposal["would_write"]:
            return {**proposal, "changed": False,
                    "note": "this profile is already enrolled with these paths; nothing "
                            "was rewritten and no second receipt was recorded"}
        self._claim(proposal)
        stamp = now()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                """
                INSERT INTO profiles(profile, hermes_home, data_dir, bank_id,
                    credential_scope, state, review_digest, enrolled_by, enrolled_at)
                VALUES(?,?,?,?,?, 'enrolled', ?, ?, ?)
                ON CONFLICT(profile) DO UPDATE SET
                    hermes_home=excluded.hermes_home, data_dir=excluded.data_dir,
                    bank_id=excluded.bank_id, credential_scope=excluded.credential_scope,
                    state='enrolled', review_digest=excluded.review_digest,
                    enrolled_by=excluded.enrolled_by, enrolled_at=excluded.enrolled_at,
                    retired_by=NULL, retired_at=NULL, retire_reason=NULL
                """,
                (proposal["profile"], proposal["hermes_home"], proposal["data_dir"],
                 proposal["bank_id"], proposal["credential_scope"], review_digest,
                 actor, stamp))
            receipt = self._receipt(proposal, actor=actor, action="enroll", stamp=stamp)
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {**proposal, "changed": True, "receipt_id": receipt}

    def retire(self, profile: str, *, actor: str, reason: str) -> dict[str, Any]:
        """Unlink a profile. Its evidence stays exactly where it is.

        Disabling a profile is not forgetting: the store, its tombstones and the
        erasure ledger outlive the mapping, and a purge is a separate owner action with
        a manifest of its own.
        """
        self._owner(actor, "retire a profile")
        self._writable()
        name = _label(profile, "profile")
        _text(reason, "reason")
        existing = self.db.execute("SELECT * FROM profiles WHERE profile=?",
                                   (name,)).fetchone()
        if existing is None:
            raise InstallationError(f"profile {name!r} is not enrolled")
        if existing["state"] == "retired":
            return {"profile": name, "changed": False, "state": "retired",
                    "note": "already retired; the data was left where it is"}
        stamp = now()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE profiles SET state='retired', retired_by=?, retired_at=?, "
                "retire_reason=? WHERE profile=?", (actor, stamp, reason[:1000], name))
            receipt = self._receipt({"profile": name, "hermes_home": existing["hermes_home"],
                                     "data_dir": existing["data_dir"],
                                     "bank_id": existing["bank_id"],
                                     "credential_scope": existing["credential_scope"],
                                     "review_digest": existing["review_digest"]},
                                    actor=actor, action="retire", stamp=stamp,
                                    result=reason[:200])
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"profile": name, "changed": True, "state": "retired",
                "data_dir": existing["data_dir"], "bank_id": existing["bank_id"],
                "receipt_id": receipt,
                "note": "the mapping is retired; its evidence, tombstones and erasure "
                        "ledger are untouched"}

    # -- receipts ------------------------------------------------------------

    def receipts(self, profile: str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or not 1 <= limit <= 500:
            raise InstallationError("limit must be between 1 and 500 rows")
        clause, params = ((" WHERE profile=?", [_label(profile, "profile")]) if profile
                         else ("", []))
        rows = self.db.execute(
            f"SELECT * FROM enrollment_receipts{clause} ORDER BY at DESC, id LIMIT ?",
            [*params, limit]).fetchall()
        return [{"id": row["id"], "profile": row["profile"], "action": row["action"],
                 "actor": row["actor"], "review_digest": row["review_digest"],
                 "at": row["at"], "result": row["result"],
                 "changes": json.loads(row["changes"])} for row in rows]

    def receipt_for(self, review_digest: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM enrollment_receipts WHERE review_digest=? "
                              "ORDER BY at DESC, id LIMIT 1",
                              (_text(review_digest, "review digest", 200),)).fetchone()
        return None if row is None else {"id": row["id"], "profile": row["profile"],
                                         "action": row["action"], "actor": row["actor"],
                                         "at": row["at"]}

    # -- internals -----------------------------------------------------------

    def _claim(self, proposal: dict[str, Any]) -> None:
        """Refuse a plan that would put two profiles on one home or one bank."""
        clash = self.db.execute(
            "SELECT profile, state FROM profiles WHERE (hermes_home=? OR bank_id=?) "
            "AND profile!=?",
            (proposal["hermes_home"], proposal["bank_id"], proposal["profile"])).fetchone()
        if clash is not None:
            raise InstallationError(
                f"profile {clash[0]!r} already owns this home or bank; two capture owners "
                "for one conversation is the failure this refuses")

    def _receipt(self, proposal: dict[str, Any], *, actor: str, action: str, stamp: str,
                 result: str = "enrolled") -> str:
        receipt = "enr_" + digest([action, proposal["profile"], proposal["review_digest"],
                                   stamp])[:32]
        self.db.execute(
            "INSERT INTO enrollment_receipts(id, profile, action, actor, review_digest, "
            "changes, result, at) VALUES(?,?,?,?,?,?,?,?)",
            (receipt, proposal["profile"], action, actor, proposal["review_digest"],
             json.dumps({key: proposal[key] for key in PROPOSAL_KEYS},
                        ensure_ascii=False, sort_keys=True)[:4000], result[:400], stamp))
        return receipt

    def _owner(self, actor: str, action: str) -> None:
        if not isinstance(actor, str) or not actor.strip():
            raise InstallationError(f"only the owner may {action}, and none is named")
        if self.owner_principal is None or actor != self.owner_principal:
            raise InstallationError(
                f"only the owner principal may {action}; the configured owner is "
                f"{self.owner_principal or 'unset — set HERMES_MEMORY_OWNER_PRINCIPAL'}")

    def _data_dir(self, profile: str) -> Path:
        if profile == DEFAULT_PROFILE and self.default_home is not None:
            return Path(self.default_home)
        return self.root / "profiles" / profile

    @staticmethod
    def _bank(profile: str) -> str:
        """The bank holding this profile's derived memories.

        ``default`` keeps the name the single-profile installation already used, so the
        document map's default agrees with the ledger. Every other profile gets its own
        bank: two profiles sharing one bank is one recall away from answering a question
        about one person out of another person's memories.
        """
        return "hermes" if profile == DEFAULT_PROFILE else "hermes-" + profile


def _profile(row) -> Profile:
    return Profile(profile=row["profile"], hermes_home=Path(row["hermes_home"]),
                   data_dir=Path(row["data_dir"]), bank_id=row["bank_id"],
                   credential_scope=row["credential_scope"], state=row["state"],
                   review_digest=row["review_digest"], enrolled_by=row["enrolled_by"],
                   enrolled_at=row["enrolled_at"])


def _home(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise InstallationError("a Hermes home must be a path")
    return Path(value).expanduser().resolve()


def _label(value: Any, what: str) -> str:
    """A profile name, never a path fragment.

    Whatever the caller meant, the result is one slug that can be a directory name, a
    bank id and a credential scope without carrying separators from one to the next.
    """
    if not isinstance(value, str) or not value.strip():
        raise InstallationError(f"{what} must be nonempty text")
    cleaned = _SLUG.sub("-", value.strip().lower()).strip("-")
    if not cleaned or len(cleaned) > MAX_LABEL:
        raise InstallationError(f"{what} must be 1-{MAX_LABEL} name characters")
    return cleaned


def _text(value: Any, what: str, maximum: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InstallationError(f"{what} must be nonempty text")
    return value.strip()[:maximum]
