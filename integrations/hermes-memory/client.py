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
from dataclasses import replace
import json
from typing import Any

from hermes_memory.config import (load_settings, scoped_secret, env_file_values,
                                 validate_inference_route, _deadline)
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
        if not name:
            return None
        try:
            from agent.secret_scope import get_secret, serves_routed_profile
        except ImportError:
            return scoped_secret(self.settings, name)
        prefix = self.settings.credential_scope.strip().upper().replace("-", "_")
        scoped = get_secret(f"{prefix}_{name}") if prefix else None
        if scoped:
            return scoped
        # Under a host-bound secret scope, a bare name is already profile-local.
        # Without that scope, a named profile may not borrow process credentials.
        if self.settings.profile == "default" or serves_routed_profile():
            return get_secret(name)
        return None

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
    profile_config = Path(hermes_home) / "hermes-memory.json"
    legacy = Path(hermes_home) / "hermes-memory.env"
    values = {}
    if profile_config.exists():
        values = json.loads(profile_config.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise BindingError("hermes-memory.json must contain a settings object")
        allowed = {"data_dir", "hindsight_url", "allowed_inference_hosts", "foreground_deadline_s"}
        if set(values) - allowed:
            raise BindingError("unrecognized profile memory settings")
    elif legacy.exists():
        allowed = {"HERMES_MEMORY_DATA_DIR", "HERMES_MEMORY_HINDSIGHT_URL",
                   "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS", "HERMES_MEMORY_FOREGROUND_DEADLINE_S"}
        values = {key.removeprefix("HERMES_MEMORY_").lower(): value
                  for key, value in env_file_values(legacy).items() if key in allowed}
    ledger = ProfileRegistry.reading(base)
    try:
        profile = ledger.resolve(hermes_home)
        overrides = {}
        if "allowed_inference_hosts" in values:
            hosts = frozenset(host.strip().lower() for host in
                              str(values["allowed_inference_hosts"]).split(",") if host.strip())
            if not hosts <= base.allowed_inference_hosts:
                raise BindingError("profile settings cannot widen the instance's approved hosts")
            overrides["allowed_inference_hosts"] = hosts
        if "hindsight_url" in values:
            url = str(values["hindsight_url"]).strip()
            validate_inference_route(url, overrides.get("allowed_inference_hosts",
                                                        base.allowed_inference_hosts))
            if url.rstrip("/") == str(base.admission_url or "").rstrip("/"):
                raise BindingError("the Hindsight endpoint cannot be the model admission gate")
            overrides["hindsight_url"] = url
        if "foreground_deadline_s" in values:
            overrides["foreground_deadline_s"] = _deadline(str(values["foreground_deadline_s"]))
        # data_dir is an enrollment proposal, not authority to move a profile's store.
        base = replace(base, **overrides)
    except InstallationError as error:
        ledger.db.close()
        raise BindingError(unenrolled_reason(hermes_home,
                                            fresh=ledger.detached)) from error
    except Exception:
        ledger.db.close()
        raise
    return Activity(ledger, profile, profile.scoped(base))
