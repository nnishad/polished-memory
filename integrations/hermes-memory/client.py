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
from hermes_memory.install.profiles import (InstallationError, Profile,
                                            ProfileRegistry)

__all__ = ["Activity", "bind", "BindingError", "unenrolled_reason"]

_SPOOL = Path("hermes-memory") / "capture-spool.db"


class BindingError(RuntimeError):
    """This activity's home is not enrolled, so it has no memory to be served from."""


def unenrolled_reason(home: Any, *, fresh: bool = False) -> str:
    """The operator-facing sentence, shared with the provider's availability check.

    Kept here rather than in the provider so that ``is_available()`` can explain a
    missing binding without a session to bind to.
    """
    first_step = ("there is no installation ledger yet, so run setup first" if fresh
                  else "run `hermes-memory enroll --hermes-home <profile-home>` from the "
                       "instance home")
    return (f"no memory profile is enrolled for {home}; {first_step}. The default "
            "profile is not used as a fallback: one gateway serves many profiles and "
            "guessing which one is asking is how one person's question gets answered "
            "out of another person's memory")


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
    def data_dir(self) -> Path:
        return self.profile.data_dir

    @property
    def db_path(self) -> Path:
        return self.profile.db_path

    @property
    def spool_path(self) -> Path:
        """The capture spool lives with the activity's own memory, not the gateway's."""
        return self.profile.data_dir / _SPOOL

    @property
    def instance_db_path(self) -> Path:
        """The installation's own canonical store, where a machine-wide fence lives.

        A hold on delivery or on inference is not one profile's business: it covers every
        outbox on the machine, so it is read from the store the operator's command wrote
        it to rather than from each profile's copy of the question.
        """
        home = getattr(self._registry, "default_home", None) if self._registry else None
        return Path(home) / "canonical.db" if home else Path(self.settings.db_path)

    def secret(self, name: str | None) -> str | None:
        """This profile's credential, or None. Never another profile's."""
        return scoped_secret(self.settings, name)

    def close(self) -> None:
        if self._registry is not None:
            self._registry.db.close()
            self._registry = None


def bind(hermes_home: str | Path, *, settings: Any = None) -> Activity:
    """Resolve *hermes_home* to the profile that owns it.

    Raises :class:`BindingError` when nothing is enrolled there. There is no argument
    that means "use the default": an activity that cannot say whose home it is in has
    no business reading anybody's memory. And a lookup never brings the ledger into
    being — a refused read that leaves state behind has changed the installation it
    was only supposed to consult.
    """
    base = settings if settings is not None else load_settings()
    ledger = ProfileRegistry.reading(base)
    try:
        profile = ledger.resolve(hermes_home)
    except InstallationError as error:
        ledger.db.close()
        raise BindingError(unenrolled_reason(hermes_home,
                                            fresh=ledger.detached)) from error
    return Activity(ledger, profile, profile.scoped(base))
