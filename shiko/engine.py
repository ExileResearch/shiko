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
import json
import os
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
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
from .extraction import (
    Candidate,
    ExtractionError,
    ExtractionResult,
    GenerativeExtractor,
    Rejection,
    StructuredExtractor,
    validate_candidate,
)
from .owner import OwnerManifest, ensure_manifest, slugify_owner
from .pipeline import IngestionPipeline
from .queue import complete_by_kind, failed_jobs, is_pending, pending_count
from .records import (
    Affect,
    ClassificationInput,
    EvidenceInput,
    MemoryDraft,
    RecordError,
    artifacts_for_event,
    attach_artifact,
    create_memory,
    derive_memory,
    effective_activation_for,
    entities_for_memory,
    evidence_roots,
    get_dynamics,
    get_memory,
    link_alias,
    list_memories,
    memory_event_rollups,
    memory_events,
    note_evidence_root,
    prune_memory_events,
    record_memory_event,
    register_entity,
    resolve_entity,
    revise_memory,
    set_dynamics,
)
from .schema_map import PROPERTY_GROUPS, schema_map
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
from .storage import DB_NAME, SCHEMA_VERSION, Store, new_id
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
    pipeline_debounce_ms: int = 200
    pipeline_max_delay_ms: int = 1500
    pipeline_coalesce_at: int = 50
    pipeline_batch_size: int = 20


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
        self._pipeline = IngestionPipeline(
            store,
            debounce_ms=config.pipeline_debounce_ms,
            max_delay_ms=config.pipeline_max_delay_ms,
            coalesce_at=config.pipeline_coalesce_at,
            batch_size=config.pipeline_batch_size,
        )
        self._extractor: object = StructuredExtractor()

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
        # Any session with unprocessed events gets its coalesced job back after a
        # crash, a paused pipeline or a reopened deployment.
        engine._pipeline.ensure_all()
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
    def pipeline(self) -> IngestionPipeline:
        return self._pipeline

    @property
    def extractor(self) -> object:
        return self._extractor

    def set_extractor(self, extractor: object) -> None:
        """Choose the extractor the pipeline and direct calls use."""
        if not hasattr(extractor, "extract"):
            raise TypeError("extractor must expose extract(event)")
        self._extractor = extractor

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
            result = self._store.capture(
                envelope, job_kinds=DEFAULT_JOB_KINDS, on_insert=self._pipeline.note_capture,
            )
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

    # -- extraction ---------------------------------------------------------

    def extract_event(
        self, event_id: str, *, extractor: object | None = None,
    ) -> ExtractionResult:
        """Run one extractor over one source event and write validated candidates.

        A failed extraction never removes or mutates the source event. A replay
        of a succeeded (event, extractor, version) returns the existing memories
        without calling the backend again.
        """
        self._ensure_open()
        event = self._store.get_event(event_id)
        if event is None:
            raise RecordError("extraction references an unknown event", detail={"event_id": event_id})
        active = extractor or self._extractor
        name = str(getattr(active, "name", "structured"))
        version = str(getattr(active, "version", "1"))

        previous = self._store.read_one(
            "SELECT run_id FROM extraction_runs WHERE event_id=? AND extractor=? AND extractor_version=? "
            "AND status='succeeded' ORDER BY started_at LIMIT 1",
            (event_id, name, version),
        )
        if previous is not None:
            memory_ids = tuple(
                row[0]
                for row in self._store.read(
                    "SELECT memory_id FROM extraction_candidates WHERE run_id=? "
                    "AND status='accepted' AND memory_id IS NOT NULL ORDER BY created_at",
                    (previous[0],),
                )
            )
            return ExtractionResult(
                run_id=str(previous[0]), event_id=event_id, extractor=name,
                extractor_version=version, status="replayed", accepted=memory_ids,
            )

        run_id = new_id("run")
        started = _utc_iso()
        with self._store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO extraction_runs
                    (run_id, event_id, extractor, extractor_version, status, calls, started_at, detail_json)
                VALUES (?, ?, ?, ?, 'running', 0, ?, '{}')
                """,
                (run_id, event_id, name, version, started),
            )
        try:
            outcome = active.extract(event)  # type: ignore[attr-defined]
        except Exception as exc:
            diagnostic = Diagnostic(
                "extraction.failed", f"extractor failed: {exc}", Severity.ERROR,
                {"event_id": event_id, "extractor": name, "extractor_version": version},
            )
            self._diagnostics.emit(diagnostic)
            with self._store.transaction() as conn:
                conn.execute(
                    """
                    UPDATE extraction_runs
                    SET status='failed', error=?, model_id=?, calls=?, tokens=?, finished_at=?
                    WHERE run_id=?
                    """,
                    (
                        str(exc)[:2000], _model_id(active), _calls(active), _tokens(active),
                        _utc_iso(), run_id,
                    ),
                )
            return ExtractionResult(
                run_id=run_id, event_id=event_id, extractor=name, extractor_version=version,
                status="failed", diagnostics=(diagnostic,), calls=_calls(active),
            )

        artifacts = artifacts_for_event(self._store, event_id)
        accepted: list[str] = []
        rejected: list[Rejection] = list(outcome.rejected)
        model_id = (outcome.detail or {}).get("model_id") or _model_id(active)
        for candidate in outcome.candidates:
            reason = validate_candidate(event, candidate, artifacts=artifacts)
            if reason is not None:
                rejected.append(Rejection(candidate.content, reason))
                continue
            duplicate = self._store.read_one(
                "SELECT memory_id FROM extraction_candidates WHERE event_id=? AND extractor_version=? "
                "AND content_sha256=? AND span_start IS ? AND span_end IS ? AND status='accepted'",
                (
                    event_id, version, candidate.content_sha256(),
                    candidate.span[0] if candidate.span else None,
                    candidate.span[1] if candidate.span else None,
                ),
            )
            if duplicate is not None:
                if duplicate[0]:
                    accepted.append(str(duplicate[0]))
                continue

            classifications = [
                ClassificationInput("epistemic", candidate.epistemic, method=f"extractor:{name}", model_id=model_id),
                ClassificationInput("modality", candidate.modality, method=f"extractor:{name}", model_id=model_id),
                ClassificationInput("scope", candidate.scope, method=f"extractor:{name}", model_id=model_id),
                ClassificationInput("extraction", name, method="deterministic", model_id=model_id),
            ]
            if candidate.negated:
                classifications.append(ClassificationInput("polarity", "negated", method=f"extractor:{name}"))
            if candidate.quoted_speaker:
                classifications.append(
                    ClassificationInput("attribution", candidate.quoted_speaker, method=f"extractor:{name}")
                )
            draft = MemoryDraft(
                kind=candidate.kind,
                content=candidate.content,
                evidence=(EvidenceInput(event_id, "supports", candidate.span),),
                classifications=tuple(classifications),
                subject_kind=candidate.subject_kind,
                subject_id=candidate.subject_id,
                event_time=_parse_iso(event["observed_at"]),
                metadata={
                    "extractor": name, "extractor_version": version, "run_id": run_id,
                    "quoted_speaker": candidate.quoted_speaker, "negated": candidate.negated,
                    "confidence": candidate.confidence,
                },
            )
            memory = create_memory(self._store, draft, policy_version=f"extract:{name}.{version}")
            memory_id = str(memory["memory_id"])
            accepted.append(memory_id)
            self._record_candidate(run_id, event_id, name, version, candidate, memory_id, None)

        for rejection in rejected:
            self._record_candidate(
                run_id, event_id, name, version,
                Candidate(content=rejection.content or "(empty)", kind="semantic"), None, rejection.reason,
            )

        detail = dict(outcome.detail or {})
        detail.update({"accepted": len(accepted), "rejected": len(rejected)})
        with self._store.transaction() as conn:
            conn.execute(
                """
                UPDATE extraction_runs
                SET status='succeeded', prompt_version=?, model_id=?, calls=?, tokens=?,
                    finished_at=?, detail_json=?
                WHERE run_id=?
                """,
                (
                    detail.get("prompt_version"), model_id, _calls(active), _tokens(active),
                    _utc_iso(), json.dumps(detail, sort_keys=True), run_id,
                ),
            )
        if rejected:
            self._diagnostics.emit(Diagnostic(
                "extraction.rejected", "some candidates were rejected", Severity.WARNING,
                {"event_id": event_id, "count": len(rejected), "reasons": [r.reason for r in rejected]},
            ))
        return ExtractionResult(
            run_id=run_id, event_id=event_id, extractor=name, extractor_version=version,
            status="succeeded", accepted=tuple(accepted), rejected=tuple(rejected),
            model_id=model_id, calls=_calls(active),
        )

    def _record_candidate(
        self, run_id: str, event_id: str, extractor: str, version: str,
        candidate: Candidate, memory_id: str | None, reason: str | None,
    ) -> None:
        with self._store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO extraction_candidates
                    (candidate_id, run_id, event_id, memory_id, extractor, extractor_version,
                     content, content_sha256, kind, epistemic, modality, scope, subject_kind,
                     subject_id, span_start, span_end, negated, quoted_speaker, confidence,
                     status, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("cand"), run_id, event_id, memory_id, extractor, version,
                    candidate.content, candidate.content_sha256(), candidate.kind,
                    candidate.epistemic, candidate.modality, candidate.scope,
                    candidate.subject_kind, candidate.subject_id,
                    candidate.span[0] if candidate.span else None,
                    candidate.span[1] if candidate.span else None,
                    int(candidate.negated), candidate.quoted_speaker, candidate.confidence,
                    "accepted" if memory_id else "rejected", reason, _utc_iso(),
                ),
            )

    def extraction_runs(self, *, event_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        self._ensure_open()
        if event_id is None:
            rows = self._store.read(
                "SELECT run_id, event_id, extractor, extractor_version, status, model_id, calls, "
                "error, started_at, detail_json FROM extraction_runs ORDER BY started_at DESC LIMIT ?",
                (int(limit),),
            )
        else:
            rows = self._store.read(
                "SELECT run_id, event_id, extractor, extractor_version, status, model_id, calls, "
                "error, started_at, detail_json FROM extraction_runs WHERE event_id=? "
                "ORDER BY started_at DESC LIMIT ?",
                (event_id, int(limit)),
            )
        return [
            {
                "run_id": row[0], "event_id": row[1], "extractor": row[2],
                "extractor_version": row[3], "status": row[4], "model_id": row[5],
                "calls": int(row[6]), "error": row[7], "started_at": row[8],
                "detail": json.loads(row[9] or "{}"),
            }
            for row in rows
        ]

    # -- identity and lineage -----------------------------------------------

    def register_entity(
        self, kind: str, name: str, *, entity_id: str | None = None,
        aliases: tuple[str, ...] = (), metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_open()
        return register_entity(
            self._store, kind, name, entity_id=entity_id, aliases=aliases, metadata=metadata,
        )

    def resolve_entity(self, name: str) -> dict[str, Any]:
        self._ensure_open()
        return resolve_entity(self._store, name)

    def link_alias(self, alias: str, entity_id: str, *, source_event_id: str | None = None) -> None:
        self._ensure_open()
        link_alias(self._store, alias, entity_id, source_event_id=source_event_id)

    def entities_for_memory(self, memory_id: str) -> list[dict[str, Any]]:
        self._ensure_open()
        return entities_for_memory(self._store, memory_id)

    def note_evidence_root(
        self, origin_event_id: str, *, root_id: str | None = None,
        kind: str = "observation", parent_root_id: str | None = None,
    ) -> str:
        self._ensure_open()
        return note_evidence_root(
            self._store, origin_event_id, root_id=root_id, kind=kind, parent_root_id=parent_root_id,
        )

    def evidence_roots(self, *, limit: int = 200) -> list[dict[str, Any]]:
        self._ensure_open()
        return evidence_roots(self._store, limit=limit)

    def derive_memory(
        self, parent_memory_ids: list[str], *, content: str, kind: str = "reflection",
        method: str = "summary", policy_version: str = DEFAULT_POLICY,
        classifications: tuple[ClassificationInput, ...] = (),
        subject_kind: str | None = None, subject_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a derived memory that inherits its parents' evidence roots."""
        self._ensure_open()
        return derive_memory(
            self._store, parent_memory_ids, content=content, kind=kind, method=method,
            policy_version=policy_version, classifications=classifications,
            subject_kind=subject_kind, subject_id=subject_id,
        )

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
            result = self._store.capture(
                envelope, job_kinds=DEFAULT_JOB_KINDS, on_insert=self._pipeline.note_capture,
            )
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
        """A worker bound to this deployment's store, archive and pipeline."""
        self._ensure_open()
        kwargs.setdefault("pipeline", self._pipeline)
        kwargs.setdefault("gate", self._pipeline.can_run)
        return Worker(self._store, self._archive, **kwargs)

    # -- ingestion pipeline -------------------------------------------------

    def pipeline_status(self) -> dict[str, Any]:
        self._ensure_open()
        return self._pipeline.status()

    def set_pipeline_paused(self, paused: bool) -> dict[str, Any]:
        self._ensure_open()
        self._pipeline.set_paused(bool(paused))
        return self._pipeline.status()

    def ingestion_state(self, session_id: str | None = None) -> list[dict[str, Any]]:
        self._ensure_open()
        if session_id is None:
            rows = self._store.read(
                "SELECT session_id, watermark_seq, last_enqueued_seq, last_batch_at "
                "FROM ingestion_state ORDER BY session_id"
            )
        else:
            rows = self._store.read(
                "SELECT session_id, watermark_seq, last_enqueued_seq, last_batch_at "
                "FROM ingestion_state WHERE session_id=?",
                (session_id,),
            )
        return [
            {
                "session_id": row[0], "watermark_seq": int(row[1]),
                "last_enqueued_seq": int(row[2]), "last_batch_at": row[3],
            }
            for row in rows
        ]

    def sweep(self, *, budget_seconds: float = 2.0, max_jobs: int = 200) -> dict[str, Any]:
        """Drain queued work in-process; the cron/systemd entry point."""
        self._ensure_open()
        worker = self.create_worker(worker_id="sweep")
        summary = worker.run_until_idle(budget_seconds=budget_seconds, max_jobs=max_jobs)
        self._diagnostics.emit(
            Diagnostic("pipeline.sweep", "sweep drained queued work", Severity.INFO, dict(summary))
        )
        return summary

    # -- lifecycle history and dynamics -------------------------------------

    def record_memory_event(
        self, memory_id: str, kind: str, **kwargs: Any,
    ) -> None:
        self._ensure_open()
        record_memory_event(self._store, memory_id, kind, **kwargs)

    def memory_events(
        self, *, memory_id: str | None = None, kind: str | None = None, limit: int = 200,
    ) -> list[dict[str, Any]]:
        self._ensure_open()
        return memory_events(self._store, memory_id=memory_id, kind=kind, limit=limit)

    def prune_memory_events(self, *, retention_days: int = 30) -> dict[str, Any]:
        self._ensure_open()
        result = prune_memory_events(self._store, retention_days=retention_days)
        self._diagnostics.emit(
            Diagnostic("memory_events.pruned", "lifecycle history pruned", Severity.INFO, dict(result))
        )
        return result

    def memory_event_rollups(self, *, memory_id: str | None = None) -> list[dict[str, Any]]:
        self._ensure_open()
        return memory_event_rollups(self._store, memory_id=memory_id)

    def set_dynamics(self, memory_id: str, **kwargs: Any) -> None:
        self._ensure_open()
        set_dynamics(self._store, memory_id, **kwargs)

    def get_dynamics(self, memory_id: str) -> dict[str, Any] | None:
        self._ensure_open()
        return get_dynamics(self._store, memory_id)

    def effective_activation(self, memory_id: str, *, at: datetime | None = None) -> float | None:
        self._ensure_open()
        return effective_activation_for(self._store, memory_id, at=at)

    def schema_map(self) -> list[dict[str, Any]]:
        self._ensure_open()
        return schema_map(self._store)

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


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _calls(extractor: object) -> int:
    backend = getattr(extractor, "backend", None)
    return int(getattr(backend, "calls", 0) or 0)


def _tokens(extractor: object) -> int | None:
    backend = getattr(extractor, "backend", None)
    tokens = getattr(backend, "tokens", None)
    return int(tokens) if isinstance(tokens, int) else None


def _model_id(extractor: object) -> str | None:
    backend = getattr(extractor, "backend", None)
    model = getattr(backend, "model_id", None)
    return str(model) if model else None
