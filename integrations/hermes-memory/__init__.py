"""hermes-memory plugin entry point.

Hermes's memory-provider loader imports this file, calls ``register(ctx)``, and
collects the instance passed to ``ctx.register_memory_provider``. The literal
name must stay within the loader's first-8192-byte scan window of this file.
"""
from __future__ import annotations

from .provider import PROVIDER_NAME, HermesMemoryProvider, post_setup, write_env_file


def register(ctx) -> None:
    """Hand Hermes a provider instance.

    Registering the class instead of an instance is a silent downgrade: the
    loader's subclass fallback yields a bare second instance with no state.
    """
    ctx.register_memory_provider(HermesMemoryProvider())


__all__ = ["register", "post_setup", "PROVIDER_NAME", "HermesMemoryProvider", "write_env_file"]
