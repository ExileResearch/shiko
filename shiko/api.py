"""Public entry points for ordinary Python.

Nothing in this module imports a host. A custom runtime uses exactly this::

    from shiko import open_engine, Actor, ActorKind

    with open_engine("/var/lib/shiko/jiko", owner_id="jiko") as engine:
        engine.capture_text(
            "The wrist stalled with the M2000 camera.",
            session_id="design-2026-11-20",
            source_kind="harness.turn",
        )
"""

from __future__ import annotations

from pathlib import Path

from .engine import Engine, EngineConfig, default_root, resolve_owner_id, resolve_root

__all__ = [
    "Engine",
    "EngineConfig",
    "default_root",
    "open_engine",
    "resolve_owner_id",
    "resolve_root",
]


def open_engine(
    root: str | Path | None = None,
    owner_id: str | None = None,
    *,
    display_name: str = "",
    config: EngineConfig | None = None,
) -> Engine:
    """Open (or provision) one deployment's engine.

    Raises :class:`shiko.errors.OwnershipError` if the root belongs to another
    owner and :class:`shiko.errors.StorageError` if the store cannot be opened.
    """
    return Engine.open(root, owner_id, display_name=display_name, config=config)
