"""The engine: one owner, one data root, one canonical store.

``Engine.open`` provisions or adopts a deployment root, verifies ownership and
migrates the store. The remaining methods are the public contract named in
milestone 1.1 — capture, recall, outcome feedback, maintenance and close — and
each returns a typed result. Operations that arrive in later milestones answer
``UNSUPPORTED`` with a diagnostic rather than pretending to be empty.

Root resolution is explicit, then config, then ``SHIKO_HOME``, then the host's
``HERMES_HOME/shiko``, then ``~/.hermes/shiko``. One runtime, one root.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .diagnostics import Diagnostics
from .envelope import (
    Actor,
    ActorKind,
    SourceEnvelope,
    Subject,
    Visibility,
    build_envelope,
)
from .errors import EngineClosedError, ShikoError, StorageError
from .owner import OwnerManifest, ensure_manifest, slugify_owner
from .results import (
    CaptureResult,
    CloseResult,
    Diagnostic,
    MaintenanceResult,
    OpenResult,
    OutcomeResult,
    RecallResult,
    Severity,
    Status,
)
from .storage import DB_NAME, SCHEMA_VERSION, Store

DEFAULT_OWNER = "hermes"
_SAFE_ROOT = re.compile(r"^[a-zA-Z0-9._/\\:@ -]+$")


@dataclass
class EngineConfig:
    """Optional overrides; every value has a host-derived default."""

    root: str | Path | None = None
    owner_id: str | None = None
    display_name: str = ""
    visibility: Visibility = Visibility.PRIVATE
    log_diagnostics: bool = True


def default_root() -> Path:
    """Where a deployment lives when the caller names nothing."""
    explicit = os.environ.get("SHIKO_HOME", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    if hermes_home:
        return Path(hermes_home).expanduser() / "shiko"
    return Path.home() / ".hermes" / "shiko"


def resolve_root(root: str | Path | None = None, config: EngineConfig | None = None) -> Path:
    candidate = root if root is not None else (config.root if config else None)
    if candidate is None:
        return default_root().resolve()
    return Path(candidate).expanduser().resolve()


def resolve_owner_id(
    owner_id: str | None = None, config: EngineConfig | None = None, *, root: Path | None = None,
) -> str:
    candidate = owner_id if owner_id is not None else (config.owner_id if config else None)
    if not candidate:
        candidate = os.environ.get("SHIKO_OWNER", "").strip()
    if not candidate and root is not None:
        candidate = ""
    return slugify_owner(candidate or DEFAULT_OWNER)


class Engine:
    """A deployment's memory engine. Construct through :func:`shiko.open_engine`."""

    def __init__(
        self,
        *,
        root: Path,
        manifest: OwnerManifest,
        store: Store,
        diagnostics: Diagnostics,
        created: bool,
        config: EngineConfig,
    ) -> None:
        self._root = root
        self._manifest = manifest
        self._store = store
        self._diagnostics = diagnostics
        self._created = created
        self._config = config
        self._closed = False

    # -- construction -------------------------------------------------------

    @classmethod
    def open(
        cls,
        root: str | Path | None = None,
        owner_id: str | None = None,
        *,
        display_name: str = "",
        config: EngineConfig | None = None,
    ) -> "Engine":
        config = config or EngineConfig()
        resolved_root = resolve_root(root, config)
        resolved_owner = resolve_owner_id(owner_id, config, root=resolved_root)
        try:
            resolved_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as exc:
            raise StorageError(
                f"cannot create data root: {exc}", detail={"root": str(resolved_root)}
            ) from exc

        manifest, created = ensure_manifest(
            resolved_root, resolved_owner, display_name=display_name or config.display_name
        )
        store = Store(resolved_root / DB_NAME, resolved_owner)
        schema_version = store.open()
        log_path = resolved_root / "logs" / "shiko.jsonl" if config.log_diagnostics else None
        diagnostics = Diagnostics(log_path, owner_id=resolved_owner)
        engine = cls(
            root=resolved_root,
            manifest=manifest,
            store=store,
            diagnostics=diagnostics,
            created=created,
            config=config,
        )
        engine._diagnostics.emit(
            Diagnostic(
                "engine.opened",
                "deployment root opened",
                Severity.INFO,
                {
                    "root": str(resolved_root),
                    "owner_id": resolved_owner,
                    "created": created,
                    "schema_version": schema_version,
                    "engine_version": __version__,
                },
            )
        )
        return engine

    # -- identity -----------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def owner_id(self) -> str:
        return self._manifest.owner_id

    @property
    def manifest(self) -> OwnerManifest:
        return self._manifest

    @property
    def schema_version(self) -> int:
        return self._store.schema_version

    @property
    def diagnostics(self) -> Diagnostics:
        return self._diagnostics

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def event_count(self) -> int:
        self._ensure_open()
        return self._store.count_events()

    def open_result(self) -> OpenResult:
        return OpenResult.success(
            root=str(self._root),
            owner_id=self.owner_id,
            schema_version=self.schema_version,
            created=self._created,
        )

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """Read one captured event back, for inspection and recovery checks."""
        self._ensure_open()
        return self._store.get_event(event_id)

    def events(self, *, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Read recent events in capture order, for diagnostics and tests."""
        self._ensure_open()
        return self._store.list_events(session_id=session_id, limit=limit)

    # -- operations ---------------------------------------------------------

    def capture(self, envelope: SourceEnvelope) -> CaptureResult:
        """Accept one observation durably. The only write in milestone 1."""
        self._ensure_open()
        if not isinstance(envelope, SourceEnvelope):
            raise TypeError("capture() expects a SourceEnvelope")
        try:
            result = self._store.capture(envelope)
        except StorageError as exc:
            diagnostic = Diagnostic(
                "event.write_failed", str(exc), Severity.ERROR, {"event_id": envelope.event_id}
            )
            self._diagnostics.emit(diagnostic)
            return CaptureResult.failure(envelope.event_id, (diagnostic,))
        self._diagnostics.extend(result.diagnostics)
        return result

    def capture_text(
        self,
        content: str,
        *,
        actor: Actor | None = None,
        session_id: str,
        source_kind: str,
        event_id: str | None = None,
        source_id: str = "",
        source_detail: dict[str, Any] | None = None,
        visibility: Visibility | None = None,
        delegation_parent: str | None = None,
        subjects: tuple[Subject, ...] = (),
        metadata: dict[str, Any] | None = None,
        content_type: str = "text/plain",
    ) -> CaptureResult:
        """Convenience for hosts and harnesses: build an envelope, then capture it."""
        envelope = build_envelope(
            content,
            event_id=event_id or f"shiko:{uuid.uuid4().hex}",
            actor=actor or Actor(id=self.owner_id, kind=ActorKind.OWNER),
            session_id=session_id,
            source_kind=source_kind,
            source_id=source_id,
            source_detail=source_detail,
            visibility=visibility or self._config.visibility,
            delegation_parent=delegation_parent,
            subjects=subjects,
            metadata=metadata,
            content_type=content_type,
        )
        return self.capture(envelope)

    def recall(self, query: str, *, limit: int = 8, budget_tokens: int = 800) -> RecallResult:
        """No candidate path exists yet; say so explicitly instead of returning empty."""
        self._ensure_open()
        diagnostic = Diagnostic(
            "recall.not_implemented",
            "retrieval arrives in milestone 4; no lexical, semantic or graph path is configured",
            Severity.WARNING,
            {"query_length": len(query or ""), "limit": limit, "budget_tokens": budget_tokens},
        )
        self._diagnostics.emit(diagnostic)
        return RecallResult(
            operation="recall",
            status=Status.UNSUPPORTED,
            diagnostics=(diagnostic,),
            budget_tokens=budget_tokens,
        )

    def record_outcome(
        self, memory_id: str, *, signal: str, value: float | None = None, note: str = "",
    ) -> OutcomeResult:
        """Feedback lands in milestone 5; the call is visible but does nothing yet."""
        self._ensure_open()
        diagnostic = Diagnostic(
            "outcome.not_implemented",
            "outcome feedback arrives in milestone 5",
            Severity.WARNING,
            {"memory_id": memory_id, "signal": signal, "value": value, "note": note},
        )
        self._diagnostics.emit(diagnostic)
        return OutcomeResult(
            operation="record_outcome",
            status=Status.UNSUPPORTED,
            diagnostics=(diagnostic,),
            memory_id=memory_id,
            recorded=False,
        )

    def maintain(self, *, budget_seconds: float = 1.0) -> MaintenanceResult:
        """Idle maintenance arrives in milestone 7."""
        self._ensure_open()
        started = time.monotonic()
        diagnostic = Diagnostic(
            "maintenance.not_implemented",
            "background consolidation arrives in milestone 7",
            Severity.WARNING,
            {"budget_seconds": budget_seconds},
        )
        self._diagnostics.emit(diagnostic)
        return MaintenanceResult(
            operation="maintain",
            status=Status.UNSUPPORTED,
            diagnostics=(diagnostic,),
            elapsed_ms=(time.monotonic() - started) * 1000.0,
        )

    def close(self) -> CloseResult:
        if self._closed:
            return CloseResult(operation="close", status=Status.OK, closed=False)
        self._store.close()
        self._closed = True
        self._diagnostics.emit(
            Diagnostic("engine.closed", "deployment root closed", Severity.INFO, {"root": str(self._root)})
        )
        return CloseResult(operation="close", status=Status.OK, closed=True)

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise EngineClosedError("engine is closed", detail={"root": str(self._root)})
