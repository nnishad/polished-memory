"""Whose memory is this? Answered per activity, from the home that activity runs in.

Hermes runs one gateway for several profiles. Each of them may hold a conversation
whose answers must come from a different archive, under a different credential, and the
process is shared. So the only safe rule is that every activity names its own home and
that home is looked up fresh: a provider that resolved the profile once at startup would
serve the first caller's memory to everybody who connected afterwards.

This module holds that lookup and nothing else. It opens no evidence and sends no
request — it says which store and which scope a call should use, and refuses when the
answer is not on the ledger.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hermes_memory.config import load_settings, scoped_secret
from hermes_memory.install.profiles import InstallationError, Profile, ProfileRegistry

__all__ = ["Activity", "bind", "BindingError", "unenrolled_reason"]

_SPOOL = Path("hermes-memory") / "capture-spool.db"


class BindingError(RuntimeError):
    """This activity's home is not enrolled, so it has no memory to be served from."""


def unenrolled_reason(home: Any) -> str:
    """The operator-facing sentence, shared with the provider's availability check.

    Kept here rather than in the provider so that ``is_available()`` can explain a
    missing binding without a session to bind to.
    """
    return (f"no memory profile is enrolled for {home}; run "
            "`hermes-memory enroll --hermes-home <profile-home>` from the instance "
            "home. The default profile is not used as a fallback: one gateway serves "
            "many profiles and guessing which one is asking is how one person's "
            "question gets answered out of another person's memory")


class Activity:
    """One bound activity: its profile, its configuration, its paths, its scope."""

    def __init__(self, registry: ProfileRegistry, profile: Profile, settings: Any):
        self._registry = registry
        self.profile = profile
        self.settings = settings
        self.hermes_home = profile.hermes_home

    @property
    def name(self) -> str:
        return self.profile.profile

    @property
    def bank_id(self) -> str:
        return self.profile.bank_id

    @property
    def credential_scope(self) -> str:
        return self.profile.credential_scope

    @property
    def data_dir(self) -> Path:
        return self.profile.data_dir

    @property
    def db_path(self) -> Path:
        return self.profile.db_path

    @property
    def spool_path(self) -> Path:
        """The capture spool lives with the activity's own memory, not the gateway's."""
        return self.profile.data_dir / _SPOOL

    def secret(self, name: str | None) -> str | None:
        """This profile's credential, or None. Never another profile's."""
        return scoped_secret(self.settings, name)

    def as_dict(self) -> dict[str, Any]:
        return {"profile": self.profile.profile, "bank_id": self.profile.bank_id,
                "credential_scope": self.profile.credential_scope,
                "data_dir": str(self.profile.data_dir),
                "store_present": self.profile.db_path.exists(),
                "model_config_untouched": True}

    def close(self) -> None:
        if self._registry is not None:
            self._registry.db.close()
            self._registry = None


def bind(hermes_home: str | Path, *, settings: Any = None,
         registry: ProfileRegistry | None = None) -> Activity:
    """Resolve *hermes_home* to the profile that owns it.

    Raises :class:`BindingError` when nothing is enrolled there. There is no argument
    that means "use the default": an activity that cannot say whose home it is in has
    no business reading anybody's memory. A registry passed in by the caller is not
    closed here — the caller owns that connection.
    """
    base = settings if settings is not None else load_settings()
    owns = registry is None
    ledger = registry if registry is not None else ProfileRegistry.open(base)
    try:
        profile = ledger.resolve(hermes_home)
    except InstallationError as error:
        if owns:
            ledger.db.close()
        raise BindingError(unenrolled_reason(hermes_home)) from error
    return Activity(ledger if owns else None, profile, profile.scoped(base))
