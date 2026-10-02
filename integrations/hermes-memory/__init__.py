"""hermes-memory plugin entry point.

Hermes's memory-provider loader imports this file, calls ``register(ctx)``, and
collects the instance passed to ``ctx.register_memory_provider``. The literal
name must stay within the loader's first-8192-byte scan window of this file.
"""
from __future__ import annotations

from .runtime import adopt

# Before anything below imports `hermes_memory`: the copy the host must run is the one the
# installation's release pointer names, not whichever copy the host's environment happens to
# hold. Import order here is the whole mechanism, so it is written out.
RUNTIME = adopt()

from .provider import (PROVIDER_NAME, HermesMemoryProvider, post_setup,  # noqa: E402
                       save_profile_config)


def register(ctx) -> None:
    """Hand Hermes a provider instance.

    Registering the class instead of an instance is a silent downgrade: the
    loader's subclass fallback yields a bare second instance with no state.
    """
    ctx.register_memory_provider(HermesMemoryProvider())


__all__ = ["register", "post_setup", "PROVIDER_NAME", "HermesMemoryProvider", "save_profile_config"]
