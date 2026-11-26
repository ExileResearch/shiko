"""Shiko — an adaptive memory engine for AI agents.

The import surface is deliberately small and host-free. Hermes integration lives
in :mod:`shiko.adapters.hermes` and is imported only by the plugin entry point.
"""

from __future__ import annotations

from ._version import __version__
from .api import open_engine
from .engine import Engine, EngineConfig, default_root, resolve_owner_id, resolve_root
from .envelope import (
    Actor,
    ActorKind,
    SourceEnvelope,
    SourceRef,
    Subject,
    Visibility,
    build_envelope,
)
from .errors import (
    EngineClosedError,
    EnvelopeError,
    OwnershipError,
    ShikoError,
    StorageError,
)
from .owner import OwnerManifest, read_manifest, slugify_owner
from .pipeline import IngestionPipeline
from .worker import Worker
from .records import (
    Affect,
    ClassificationInput,
    EvidenceInput,
    MemoryDraft,
    RecordError,
)
from .extraction import (
    Candidate,
    ExtractionError,
    ExtractionResult,
    GenerativeExtractor,
    Rejection,
    StructuredExtractor,
)
from .episodes import EpisodeError
from .backends import OpenAIBackend, RecordedBackend, backend_from_env
from .llm_config import (
    ConfigError,
    LLMSettings,
    build_backend,
    load_settings,
)
from .results import (
    CaptureResult,
    CloseResult,
    Diagnostic,
    MaintenanceResult,
    OpenResult,
    OutcomeResult,
    RecallResult,
    Result,
    Severity,
    Status,
)

__all__ = [
    "__version__",
    # api / engine
    "open_engine",
    "Engine",
    "EngineConfig",
    "default_root",
    "resolve_root",
    "resolve_owner_id",
    # evidence
    "Actor",
    "ActorKind",
    "SourceEnvelope",
    "SourceRef",
    "Subject",
    "Visibility",
    "build_envelope",
    "OwnerManifest",
    "read_manifest",
    "slugify_owner",
    # canonical records
    "MemoryDraft",
    "EvidenceInput",
    "ClassificationInput",
    "Affect",
    "RecordError",
    "Worker",
    "IngestionPipeline",
    "Candidate",
    "ExtractionError",
    "ExtractionResult",
    "GenerativeExtractor",
    "Rejection",
    "StructuredExtractor",
    "EpisodeError",
    "OpenAIBackend",
    "RecordedBackend",
    "backend_from_env",
    "ConfigError",
    "LLMSettings",
    "build_backend",
    "load_settings",
    # results
    "Result",
    "Status",
    "Severity",
    "Diagnostic",
    "OpenResult",
    "CaptureResult",
    "RecallResult",
    "OutcomeResult",
    "MaintenanceResult",
    "CloseResult",
    # errors
    "ShikoError",
    "EnvelopeError",
    "OwnershipError",
    "StorageError",
    "EngineClosedError",
]
