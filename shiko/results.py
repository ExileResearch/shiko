"""Typed operation results.

Every public operation returns a result whose ``status`` says what happened, so a
caller can tell an empty recall from an unavailable retrieval path and a durable
write from a failed one without parsing prose. Diagnostics travel in the result
and are also appended to the deployment's JSONL log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Status(str, Enum):
    """Outcome of a public operation."""

    OK = "ok"
    """Completed; payload fields carry the answer."""

    EMPTY = "empty"
    """Completed and the answer is genuinely nothing. Never used for a broken path."""

    PENDING = "pending"
    """Accepted, durable work still outstanding; the caller may proceed."""

    DEGRADED = "degraded"
    """Completed with a reduced path; diagnostics name what was missing."""

    UNSUPPORTED = "unsupported"
    """Recognised but not implemented in this build or milestone."""

    FAILED = "failed"
    """Attempted and did not complete. Diagnostics name the failure."""


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Diagnostic:
    """One inspectable event attached to a result or written to the log."""

    code: str
    message: str
    severity: Severity = Severity.INFO
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", dict(self.detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class Result:
    """Base result; subclasses add their own payload fields."""

    operation: str
    status: Status
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    @property
    def empty(self) -> bool:
        return self.status is Status.EMPTY

    @property
    def degraded(self) -> bool:
        return self.status is Status.DEGRADED

    @property
    def unsupported(self) -> bool:
        return self.status is Status.UNSUPPORTED

    @property
    def failed(self) -> bool:
        return self.status is Status.FAILED

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity is Severity.ERROR)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operation": self.operation,
            "status": self.status.value,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }
        payload.update(self.payload())
        return payload

    def payload(self) -> dict[str, Any]:
        """Subclass hook for the operation's own fields."""
        return {}


@dataclass(frozen=True)
class OpenResult(Result):
    root: str = ""
    owner_id: str = ""
    schema_version: int = 0
    created: bool = False

    @classmethod
    def success(
        cls, *, root: str, owner_id: str, schema_version: int, created: bool,
        diagnostics: tuple[Diagnostic, ...] = (),
    ) -> "OpenResult":
        return cls(
            operation="open", status=Status.OK, diagnostics=diagnostics,
            root=root, owner_id=owner_id, schema_version=schema_version, created=created,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "owner_id": self.owner_id,
            "schema_version": self.schema_version,
            "created": self.created,
        }


@dataclass(frozen=True)
class CaptureResult(Result):
    event_id: str = ""
    stored: bool = False
    duplicate: bool = False
    conflict: bool = False

    @classmethod
    def success(
        cls, event_id: str, *, duplicate: bool = False, conflict: bool = False,
        diagnostics: tuple[Diagnostic, ...] = (),
    ) -> "CaptureResult":
        return cls(
            operation="capture",
            status=Status.OK,
            diagnostics=diagnostics,
            event_id=event_id,
            stored=not duplicate,
            duplicate=duplicate,
            conflict=conflict,
        )

    @classmethod
    def failure(cls, event_id: str, diagnostics: tuple[Diagnostic, ...]) -> "CaptureResult":
        return cls(operation="capture", status=Status.FAILED, diagnostics=diagnostics, event_id=event_id)

    def payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stored": self.stored,
            "duplicate": self.duplicate,
            "conflict": self.conflict,
        }


@dataclass(frozen=True)
class RecallResult(Result):
    context: str = ""
    memory_ids: tuple[str, ...] = ()
    elapsed_ms: float = 0.0
    budget_tokens: int | None = None

    @classmethod
    def unavailable(cls, *, reason: str, code: str = "recall.unavailable") -> "RecallResult":
        """An empty answer from a path that did not run, or could not."""
        return cls(
            operation="recall",
            status=Status.UNSUPPORTED,
            diagnostics=(Diagnostic(code, reason, Severity.WARNING),),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "memory_ids": list(self.memory_ids),
            "elapsed_ms": self.elapsed_ms,
            "budget_tokens": self.budget_tokens,
        }


@dataclass(frozen=True)
class OutcomeResult(Result):
    memory_id: str = ""
    recorded: bool = False

    def payload(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "recorded": self.recorded}


@dataclass(frozen=True)
class MaintenanceResult(Result):
    jobs_run: int = 0
    jobs_pending: int = 0
    elapsed_ms: float = 0.0

    def payload(self) -> dict[str, Any]:
        return {
            "jobs_run": self.jobs_run,
            "jobs_pending": self.jobs_pending,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class CloseResult(Result):
    closed: bool = False

    def payload(self) -> dict[str, Any]:
        return {"closed": self.closed}
