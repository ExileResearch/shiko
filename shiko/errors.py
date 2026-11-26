"""Typed failures for the Shiko public API.

Failures a caller must be able to distinguish are exceptions with stable codes;
routine outcomes (empty recall, pending work, degraded backends) are values in
:mod:`shiko.results`. The split matters: an exception means the call did not
complete, a result means it did and the caller can inspect how.
"""

from __future__ import annotations

from typing import Any, Mapping


class ShikoError(Exception):
    """Base class for Shiko failures, carrying a stable machine-readable code."""

    code = "shiko.error"

    def __init__(self, message: str, *, detail: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = dict(detail or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }


class EnvelopeError(ShikoError, ValueError):
    """A source envelope is malformed or internally inconsistent."""

    code = "envelope.invalid"


class OwnershipError(ShikoError):
    """A data root belongs to a different owner, or its manifest is unreadable."""

    code = "owner.mismatch"


class StorageError(ShikoError):
    """The canonical store could not be opened, migrated or written."""

    code = "storage.error"


class EngineClosedError(ShikoError):
    """The engine was used after :meth:`shiko.Engine.close`."""

    code = "engine.closed"
