"""The engine: one owner, one data root, one canonical store.

``Engine.open`` provisions or adopts a deployment root, verifies ownership and
migrates the store. Milestone 1 defined the public contract (open, capture,
recall, outcome feedback, maintenance, close). Milestone 2 makes capture atomic
with its follow-up work, materialises the JSONL archive with a recoverable
checkpoint, and adds canonical memories, revisions, provenance and artifacts.

Operations that arrive in later milestones still answer ``UNSUPPORTED`` with a
diagnostic rather than pretending to be empty.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ._version import __version__
from .archive import Archive
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
from .queue import complete_by_kind, failed_jobs, is_pending, pending_count
from .records import (
    Affect,
    ClassificationInput,
    EvidenceInput,
    MemoryDraft,
    RecordError,
    attach_artifact,
    create_memory,
    get_memory,
    list_memories,
    revise_memory,
)
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
from .worker import Worker

DEFAULT_OWNER = "hermes"
DEFAULT_POLICY = "m2.policy.1"
DEFAULT_JOB_KINDS = ("archive.event", "index.event")


@dataclass
class EngineConfig:
    """Optional overrides; every value has a host-derived default."""

    root: str | Path | None = None
    owner_id: str | None = None
    display_name: str = ""
    visibility: Visibility = Visibility.PRIVATE
    log_diagnostics: bool = True
    auto_archive: bool = True
    archive_dir: str | Path | None = None


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
        self._archive = Archive(store, config.archive_dir or root)

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
    def archive(self) -> Archive:
        return self._archive

    @property
    def store(self) -> Store:
        return self._store

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
        self._ensure_open()
        return self._store.get_event(event_id)

    def events(self, *, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        self._ensure_open()
        return self._store.list_events(session_id=session_id, limit=limit)

    # -- capture ------------------------------------------------------------

    def capture(self, envelope: SourceEnvelope) -> CaptureResult:
        """Accept one observation: event + follow-up work commit together.

        The archive append is attempted immediately so a caller learns whether
        the derived copy is current. A failure there leaves the canonical write
        committed, the work queued, and the result visibly ``DEGRADED``.
        """
        self._ensure_open()
        if not isinstance(envelope, SourceEnvelope):
            raise TypeError("capture() expects a SourceEnvelope")
        try:
            result = self._store.capture(envelope, job_kinds=DEFAULT_JOB_KINDS)
        except StorageError as exc:
            diagnostic = Diagnostic(
                "event.write_failed", str(exc), Severity.ERROR, {"event_id": envelope.event_id}
            )
            self._diagnostics.emit(diagnostic)
            return CaptureResult.failure(envelope.event_id, (diagnostic,))
        self._diagnostics.extend(result.diagnostics)
        if not result.ok or result.duplicate or not self._config.auto_archive:
            return result

        try:
            self._archive.sync_event(envelope.event_id)
        except Exception as exc:  # an unavailable archive must not lose the event
            diagnostic = Diagnostic(
                "archive.write_failed",
                f"event is committed but the JSONL archive was not written: {exc}",
                Severity.ERROR,
                {"event_id": envelope.event_id, "capture_seq": result.capture_seq},
            )
            self._diagnostics.emit(diagnostic)
            return CaptureResult.with_archive_failure(
                envelope.event_id, capture_seq=result.capture_seq, diagnostics=(diagnostic,)
            )
        complete_by_kind(self._store, "archive.event", envelope.event_id)
        return replace(result, archived=True)

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

    # -- canonical memories -------------------------------------------------

    def create_memory(self, draft: MemoryDraft, *, policy_version: str = DEFAULT_POLICY) -> dict[str, Any]:
        """Record an interpretation with its evidence, classifications and affect."""
        self._ensure_open()
        return create_memory(self._store, draft, policy_version=policy_version)

    def revise_memory(
        self, memory_id: str, content: str, *, policy_version: str = DEFAULT_POLICY,
        source_event_id: str | None = None, event_time: datetime | None = None,
        valid_from: datetime | None = None, valid_to: datetime | None = None,
    ) -> dict[str, Any]:
        """Append a revision; earlier revisions and their sources stay readable."""
        self._ensure_open()
        revision = revise_memory(
            self._store, memory_id, content, policy_version=policy_version,
            source_event_id=source_event_id, event_time=event_time,
            valid_from=valid_from, valid_to=valid_to,
        )
        memory = get_memory(self._store, memory_id) or {}
        memory["appended_revision"] = revision
        return memory

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        self._ensure_open()
        return get_memory(self._store, memory_id)

    def memories(self, *, limit: int = 100) -> list[dict[str, Any]]:
        self._ensure_open()
        return list_memories(self._store, limit=limit)

    def attach_artifact(
        self, event_id: str, source: str | Path | bytes, *, media_type: str, name: str = "",
    ) -> dict[str, Any]:
        """Copy non-text input into the deployment and record its integrity metadata."""
        self._ensure_open()
        return self._store_artifact(event_id, source, media_type=media_type, name=name)

    def capture_artifact(
        self, envelope: SourceEnvelope, source: str | Path | bytes, *, media_type: str, name: str = "",
    ) -> dict[str, Any]:
        """Capture a non-text observation: event, copied artifact, archive line.

        The artifact row exists before the archive line is written, so the line
        carries the reference; the event is committed first, as always.
        """
        self._ensure_open()
        payload = source if isinstance(source, bytes) else Path(source).read_bytes()
        filename = name or (Path(source).name if not isinstance(source, bytes) else "artifact.bin")
        try:
            result = self._store.capture(envelope, job_kinds=DEFAULT_JOB_KINDS)
        except StorageError as exc:
            diagnostic = Diagnostic(
                "event.write_failed", str(exc), Severity.ERROR, {"event_id": envelope.event_id}
            )
            self._diagnostics.emit(diagnostic)
            return {"capture": CaptureResult.failure(envelope.event_id, (diagnostic,)).to_dict(), "artifact": None}
        self._diagnostics.extend(result.diagnostics)
        if not result.ok or result.duplicate:
            return {"capture": result.to_dict(), "artifact": None}
        artifact = self._store_artifact(envelope.event_id, payload, media_type=media_type, name=filename)
        if self._config.auto_archive:
            try:
                self._archive.sync_event(envelope.event_id)
                complete_by_kind(self._store, "archive.event", envelope.event_id)
                result = replace(result, archived=True)
            except Exception as exc:
                diagnostic = Diagnostic(
                    "archive.write_failed",
                    f"event is committed but the JSONL archive was not written: {exc}",
                    Severity.ERROR,
                    {"event_id": envelope.event_id, "capture_seq": result.capture_seq},
                )
                self._diagnostics.emit(diagnostic)
                result = CaptureResult.with_archive_failure(
                    envelope.event_id, capture_seq=result.capture_seq, diagnostics=(diagnostic,)
                )
        return {"capture": result.to_dict(), "artifact": artifact}

    def _store_artifact(
        self, event_id: str, source: str | Path | bytes, *, media_type: str, name: str = "",
    ) -> dict[str, Any]:
        if isinstance(source, bytes):
            payload = source
            filename = name or "artifact.bin"
        else:
            path = Path(source)
            payload = path.read_bytes()
            filename = name or path.name
        digest = hashlib.sha256(payload).hexdigest()
        artifact_id = f"art_{uuid.uuid4().hex}"
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in filename)[-80:] or "artifact.bin"
        target_dir = self._root / "artifacts"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{artifact_id}-{safe_name}"
        target.write_bytes(payload)
        uri = str(target.relative_to(self._root))
        artifact_id = attach_artifact(
            self._store, event_id, uri=uri, media_type=media_type,
            size_bytes=len(payload), sha256=digest,
        )
        return {
            "artifact_id": artifact_id, "event_id": event_id, "uri": uri,
            "media_type": media_type, "size_bytes": len(payload), "sha256": digest,
        }

    def verify_artifacts(self) -> list[dict[str, Any]]:
        self._ensure_open()
        return self._archive.verify(base_dir=self._root)

    # -- ordering and recovery ----------------------------------------------

    def recent_context(
        self, *, session_id: str, exclude_event_id: str | None = None, limit: int = 6,
    ) -> dict[str, Any]:
        """Bounded fallback over committed events, with the indexed boundary.

        Milestone 4 replaces this with ranked recall; here it is the honest way a
        next turn can see a correction that has not been indexed yet.
        """
        self._ensure_open()
        boundary = self._store.indexed_through()
        events = self._store.recent_events_for_session(
            session_id, exclude_event_id=exclude_event_id, limit=limit
        )
        shaped = [
            {
                "event_id": event["event_id"],
                "capture_seq": event["capture_seq"],
                "content": event["content"],
                "source_kind": event["source_kind"],
                "actor_kind": event["actor_kind"],
                "observed_at": event["observed_at"],
                "indexed": int(event["capture_seq"]) <= boundary,
                "pending": is_pending(self._store, event["event_id"]),
            }
            for event in events
        ]
        return {
            "session_id": session_id,
            "boundary": boundary,
            "events": shaped,
            "pending_jobs": pending_count(self._store),
        }

    def recover(self) -> dict[str, Any]:
        """Re-materialize the archive from canonical state; idempotent."""
        self._ensure_open()
        recovered = self._archive.recover()
        diagnostic = Diagnostic(
            "archive.recovered", "archive recovery pass", Severity.INFO, dict(recovered)
        )
        self._diagnostics.emit(diagnostic)
        return {"archive": recovered, "integrity": self._store.integrity_check()}

    def failed_jobs(self) -> list[dict[str, Any]]:
        self._ensure_open()
        return failed_jobs(self._store)

    def pending_jobs(self) -> int:
        self._ensure_open()
        return pending_count(self._store)

    def create_worker(self, **kwargs: Any) -> Worker:
        """A worker bound to this deployment's store and archive."""
        self._ensure_open()
        return Worker(self._store, self._archive, **kwargs)

    # -- later milestones ---------------------------------------------------

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
