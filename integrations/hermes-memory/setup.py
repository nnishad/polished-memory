"""Enrolling a profile: a reviewed diff, never a side effect of importing a plugin.

``hermes memory setup`` writes this plugin's configuration and selects it as the memory
provider. That is the host's decision to make, and it makes it. What is *not* the host's
decision — and not this module's either — is which archive a conversation is answered
from. That is the instance ledger, and the only way it changes is an owner approving the
exact proposal :func:`plan` produced.

So the wizard ends with a command rather than a completed enrollment, and that is the
point. A digest computed by the code that consumes it proves nothing about what a person
was shown; a digest handed back by the owner, after the paths were printed, does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hermes_memory.install.profiles import (DEFAULT_PROFILE, InstallationError,
                                            ProfileRegistry, open_installation)

__all__ = ["plan", "enroll", "retire", "pending", "report", "enrollment_command",
           "registry", "UNTOUCHED"]

# What setup is allowed to claim it left alone. Each of these is a thing an
# installation has previously been observed to do by accident.
UNTOUCHED = (
    "model configuration: the main model, its fallbacks and any auxiliary route",
    "the native Hermes memory store and the notes in it",
    "installed packages, running services or occupied ports",
    "source connectors, their policies and their credentials",
    "which memory provider Hermes has selected",
)


def registry(instance_home: Any = None) -> ProfileRegistry:
    """The instance ledger, opened at the installation's own home.

    ``instance_home`` is the *installation's* home, not a profile's. Pointing this at a
    profile home would give that profile a private ledger nobody reviews — the same
    failure seen from the other side. With no argument the installation is the one the
    environment names.
    """
    from hermes_memory.config import DEFAULT_ENV_FILENAME, load_settings

    if instance_home is None:
        return ProfileRegistry.open(load_settings())
    root = Path(instance_home).expanduser().resolve()
    # That env file is the authority for two things: who may approve a change here,
    # and where the default profile's memory lives.
    settings = load_settings(root / DEFAULT_ENV_FILENAME)
    return ProfileRegistry(open_installation(root / "installation.db"), root=root,
                           owner_principal=settings.owner_principal,
                           default_home=Path(settings.data_dir))


def plan(hermes_home: Any, *, profile: str | None = None,
         instance_home: Any = None) -> dict[str, Any]:
    """What enrolling this profile would change, and what it would leave alone."""
    ledger = registry(instance_home)
    try:
        proposal = ledger.plan(profile or _guess_profile(hermes_home), hermes_home)
    finally:
        ledger.db.close()
    return {**proposal,
            "adds": [f"profile {proposal['profile']} is recognised by the instance ledger",
                     f"{proposal['data_dir']} becomes this profile's memory",
                     f"derived memories for it go to the {proposal['bank_id']} bank",
                     f"its credentials resolve under the {proposal['credential_scope']} scope"],
            "unchanged": list(UNTOUCHED),
            "already_enrolled": not proposal["would_write"],
            "command": enrollment_command(hermes_home, profile=proposal["profile"])}


def enroll(hermes_home: Any, *, actor: str, review_digest: str, profile: str | None = None,
           instance_home: Any = None) -> dict[str, Any]:
    """Apply a proposal the owner has seen. Raises when it is not the one they saw."""
    ledger = registry(instance_home)
    try:
        return ledger.enroll(profile or _guess_profile(hermes_home), hermes_home,
                             actor=actor, review_digest=review_digest)
    finally:
        ledger.db.close()


def retire(hermes_home: Any, *, actor: str, reason: str, profile: str | None = None,
           instance_home: Any = None) -> dict[str, Any]:
    """Unlink a profile. Its evidence, tombstones and erasure ledger stay where they are."""
    ledger = registry(instance_home)
    try:
        name = profile or ledger.resolve(hermes_home).profile
        return ledger.retire(name, actor=actor, reason=reason)
    finally:
        ledger.db.close()


def pending(hermes_home: Any, *, instance_home: Any = None) -> dict[str, Any]:
    """Is this home enrolled? If not, the exact command that would finish setup."""
    ledger = registry(instance_home)
    try:
        try:
            profile = ledger.resolve(hermes_home)
        except InstallationError as error:
            return {"enrolled": False, "profile": _guess_profile(hermes_home),
                    "command": enrollment_command(hermes_home),
                    "reason": str(error)}
        return {"enrolled": True, "profile": profile.profile, "bank_id": profile.bank_id,
                "data_dir": str(profile.data_dir), "command": None, "reason": None}
    finally:
        ledger.db.close()


def report(hermes_home: Any, *, instance_home: Any = None) -> dict[str, Any]:
    """What the wizard should print at the end of a setup run."""
    state = pending(hermes_home, instance_home=instance_home)
    return {"provider": "hermes-memory", "config_written": True,
            "enrolled": state["enrolled"], "profile": state["profile"],
            "next_command": state["command"],
            "why": None if state["enrolled"] else state["reason"],
            "model_config_unchanged": True, "main_model_unchanged": True}


def enrollment_command(hermes_home: Any, *, profile: str = DEFAULT_PROFILE) -> str:
    """The command, spelled out, because a summary of it would be a guess."""
    return (f"hermes-memory enroll --profile {profile} "
            f"--hermes-home {Path(str(hermes_home)).expanduser()} --actor <your principal>")


def _guess_profile(hermes_home: Any) -> str:
    """The profile a home is named after, or ``default`` for a bare account home.

    Guessing is admissible here only because the result is a *label* inside a proposal
    an owner then reviews. Nothing in this module lets a guessed name decide an access
    path without that review.
    """
    path = Path(str(hermes_home)).expanduser()
    name = path.name.strip().lower()
    return DEFAULT_PROFILE if not name or name in {"", "home", ".hermes", "hermes"} else name
