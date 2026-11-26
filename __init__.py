"""Hermes plugin entry point for Shiko.

This file is the plugin door, not the library. It only exists so Hermes can load
the repository directory as a package and hand it a plugin context. Everything
else lives in the ``shiko`` subpackage, whose relative imports work both here and
in an installed wheel.

Discovery note: Hermes decides whether a user plugin directory is a memory
provider by reading this module (a cheap text match, then an AST pass for
``register_memory_provider`` or a ``MemoryProvider`` subclass). Keep the literal
call in ``register`` so both passes recognise us.
"""

from __future__ import annotations

from typing import Any

__all__ = ["register"]


def register(ctx: Any):
    """Register Shiko with a Hermes plugin context."""
    from .shiko.adapters.hermes import ShikoProvider, register_hook_fallback

    if hasattr(ctx, "register_memory_provider"):
        provider = ShikoProvider()
        ctx.register_memory_provider(provider)
        return provider

    # Older or reduced hosts: capture conservatively through hooks instead of
    # failing to load. The fallback records what it can and never raises.
    return register_hook_fallback(ctx)
