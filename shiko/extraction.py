"""Candidate extraction: turn one observation into attributable claims.

Two extractors share one contract:

* ``StructuredExtractor`` reads typed claims a host or tool already produced. It
  is deterministic, free, and the fallback when no model is available.
* ``GenerativeExtractor`` asks a completion backend for JSON claims and validates
  every field before anything durable is written.

Validation is the point. A candidate needs a kind, an epistemic type, a modality,
a scope and either a span inside the source text or an artifact reference. A
quoted claim must be marked reported and keeps its speaker. Malformed output is
recorded as a rejection; a failed backend records a failed run while the source
event stays intact.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .backends import BackendError, CompletionBackend
from .errors import ShikoError
from .records import MEMORY_KINDS, artifacts_for_event
from .results import Diagnostic, Severity

EPISTEMIC_TYPES = frozenset({
    "observation", "fact", "belief", "inference", "hypothesis", "simulation", "reported",
})
MODALITIES = frozenset({"text", "vision", "audio", "sensor", "mixed"})
SCOPES = frozenset({"internal", "user", "shared", "environmental", "external"})
MAX_CONTENT = 4000
PROMPT_VERSION = "extract.v1"

PROMPT = """You extract memory candidates from one observation.
Return ONLY a JSON array. Each element is an object with these fields:
  content: one self-contained claim, copied or minimally normalised
  kind: semantic | episodic | procedural | commitment | profile | reflection
  epistemic: observation | fact | belief | inference | hypothesis | simulation | reported
  modality: text | vision | audio | sensor | mixed
  scope: internal | user | shared | environmental | external
  subject: {"kind": "person|device|project|concept|system", "id": "..."} or null
  span: [start, end] character offsets into the observation text
  negated: true when the claim is denied rather than asserted
  quoted_speaker: the person or system being quoted, or null
  confidence: 0.0 to 1.0, or null

Rules:
- One claim per object. Do not merge two devices, two people or two tenses.
- A proposal is a hypothesis; an installed or measured value is a fact.
- A quoted statement is "reported" and keeps its speaker; do not restate it as fact.
- A denial is "negated" with the denied claim, not the opposite claim.
- Spans must lie inside the observation.
"""


class ExtractionError(ShikoError):
    code = "extraction.error"


@dataclass(frozen=True)
class Candidate:
    content: str
    kind: str
    epistemic: str = "observation"
    modality: str = "text"
    scope: str = "internal"
    subject_kind: str | None = None
    subject_id: str | None = None
    span: tuple[int, int] | None = None
    negated: bool = False
    quoted_speaker: str | None = None
    confidence: float | None = None
    extraction: str = "generative"

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", str(self.content or "").strip())
        if self.span is not None:
            start, end = self.span
            object.__setattr__(self, "span", (int(start), int(end)))
        object.__setattr__(self, "negated", bool(self.negated))
        if self.quoted_speaker is not None:
            object.__setattr__(self, "quoted_speaker", str(self.quoted_speaker).strip() or None)
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))

    def content_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content, "kind": self.kind, "epistemic": self.epistemic,
            "modality": self.modality, "scope": self.scope,
            "subject_kind": self.subject_kind, "subject_id": self.subject_id,
            "span": list(self.span) if self.span else None,
            "negated": self.negated, "quoted_speaker": self.quoted_speaker,
            "confidence": self.confidence, "extraction": self.extraction,
        }


@dataclass(frozen=True)
class Rejection:
    content: str
    reason: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content, "reason": self.reason, "detail": dict(self.detail)}


@dataclass
class ExtractionOutcome:
    candidates: list[Candidate]
    rejected: list[Rejection]
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionResult:
    run_id: str
    event_id: str
    extractor: str
    extractor_version: str
    status: str
    accepted: tuple[str, ...] = ()
    rejected: tuple[Rejection, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    model_id: str | None = None
    calls: int = 0

    @property
    def ok(self) -> bool:
        return self.status in {"succeeded", "replayed", "skipped"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "event_id": self.event_id,
            "extractor": self.extractor, "extractor_version": self.extractor_version,
            "status": self.status, "accepted": list(self.accepted),
            "rejected": [item.to_dict() for item in self.rejected],
            "model_id": self.model_id, "calls": self.calls,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


# -- extractors --------------------------------------------------------------


class StructuredExtractor:
    """Read typed claims the host already produced."""

    name = "structured"
    version = "1"

    def extract(self, event: Mapping[str, Any]) -> ExtractionOutcome:
        metadata = _load_json(event.get("metadata_json"))
        claims = metadata.get("claims")
        if claims is not None and not isinstance(claims, list):
            return ExtractionOutcome([], [Rejection("", "claims_not_a_list", {"type": type(claims).__name__})])
        if not claims and event.get("actor_kind") == "tool" and event.get("content"):
            return ExtractionOutcome(
                [Candidate(
                    content=str(event["content"]),
                    kind="semantic",
                    epistemic="observation",
                    modality=_modality_for(str(event.get("content_type", "text/plain"))),
                    scope="internal",
                    span=(0, len(str(event["content"]))),
                    extraction=self.name,
                )],
                [],
                {"source": "tool-event"},
            )
        candidates: list[Candidate] = []
        rejected: list[Rejection] = []
        for item in claims or []:
            try:
                candidates.append(_candidate_from_mapping(item, extraction=self.name))
            except (TypeError, ValueError) as exc:
                rejected.append(Rejection(str(item)[:200], "malformed_claim", {"error": str(exc)}))
        return ExtractionOutcome(candidates, rejected, {"source": "typed-claims"})


class GenerativeExtractor:
    """Ask a completion backend for claims, then validate every one."""

    name = "generative"
    version = "1"

    def __init__(self, backend: CompletionBackend, *, timeout: float = 30.0) -> None:
        self.backend = backend
        self.timeout = timeout

    def extract(self, event: Mapping[str, Any]) -> ExtractionOutcome:
        prompt = (
            PROMPT
            + f"\nObservation ({event.get('content_type', 'text/plain')}, from "
              f"{event.get('actor_kind', 'user')}):\n{event.get('content', '')}\n"
        )
        try:
            raw = self.backend.complete(prompt, timeout=self.timeout)
        except BackendError as exc:
            raise ExtractionError(f"backend failed: {exc}", detail={"code": exc.code}) from exc
        claims = parse_claims(raw)
        candidates: list[Candidate] = []
        rejected: list[Rejection] = []
        for item in claims:
            try:
                candidates.append(_candidate_from_mapping(item, extraction=self.name))
            except (TypeError, ValueError) as exc:
                rejected.append(Rejection(str(item)[:200], "malformed_candidate", {"error": str(exc)}))
        return ExtractionOutcome(
            candidates, rejected,
            {"prompt_version": PROMPT_VERSION, "raw_chars": len(raw), "model_id": getattr(self.backend, "model_id", None)},
        )


# -- validation --------------------------------------------------------------


def validate_candidate(event: Mapping[str, Any], candidate: Candidate, *, artifacts: list[dict] | None = None) -> str | None:
    """Return a rejection reason, or ``None`` when the candidate is writable."""
    if not candidate.content:
        return "empty_content"
    if len(candidate.content) > MAX_CONTENT:
        return "content_too_long"
    if candidate.kind not in MEMORY_KINDS:
        return "unknown_kind"
    if candidate.epistemic not in EPISTEMIC_TYPES:
        return "unknown_epistemic"
    if candidate.modality not in MODALITIES:
        return "unknown_modality"
    if candidate.scope not in SCOPES:
        return "unknown_scope"
    if candidate.confidence is not None and not 0.0 <= candidate.confidence <= 1.0:
        return "confidence_out_of_range"
    if candidate.quoted_speaker and candidate.epistemic != "reported":
        return "quoted_claim_must_be_reported"

    text = str(event.get("content") or "")
    is_text = str(event.get("content_type", "text/plain")).startswith("text/")
    if candidate.span is not None:
        start, end = candidate.span
        if start < 0 or end <= start:
            return "span_invalid"
        if not is_text:
            return "span_on_non_text_event"
        if end > len(text):
            return "span_out_of_range"
        if text[start:end].strip() and candidate.content[:24].strip() and candidate.content[:24].strip() not in text:
            # Normalised claims are allowed; a claim that neither appears nor is
            # supported by its span is not.
            if not _overlaps(candidate.content, text[start:end]):
                return "span_does_not_support_content"
    else:
        if is_text:
            return "span_required"
        if artifacts is None:
            artifacts = []
        if not artifacts:
            return "artifact_required"

    if candidate.subject_id and not candidate.subject_kind:
        return "subject_kind_required"
    return None


def parse_claims(raw: str) -> list[dict]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ExtractionError("response contained no JSON array", detail={"preview": text[:200]})
    try:
        data = json.loads(text[start:end + 1])
    except ValueError as exc:
        raise ExtractionError(f"response JSON is malformed: {exc}") from exc
    if not isinstance(data, list):
        raise ExtractionError("response JSON is not an array")
    return [item for item in data if isinstance(item, dict)]


def _candidate_from_mapping(data: Mapping[str, Any], *, extraction: str) -> Candidate:
    if not isinstance(data, Mapping):
        raise TypeError("candidate must be an object")
    subject = data.get("subject")
    subject_kind = subject_id = None
    if isinstance(subject, Mapping):
        subject_kind = subject.get("kind")
        subject_id = subject.get("id")
    span = data.get("span")
    if span is not None:
        if not (isinstance(span, (list, tuple)) and len(span) == 2):
            raise ValueError("span must be [start, end]")
        span = (int(span[0]), int(span[1]))
    return Candidate(
        content=str(data.get("content") or ""),
        kind=str(data.get("kind") or ""),
        epistemic=str(data.get("epistemic") or "observation"),
        modality=str(data.get("modality") or "text"),
        scope=str(data.get("scope") or "internal"),
        subject_kind=str(subject_kind) if subject_kind else None,
        subject_id=str(subject_id) if subject_id else None,
        span=span,
        negated=bool(data.get("negated") or False),
        quoted_speaker=str(data["quoted_speaker"]) if data.get("quoted_speaker") else None,
        confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
        extraction=extraction,
    )


def _modality_for(content_type: str) -> str:
    if content_type.startswith("text/"):
        return "text"
    if content_type.startswith("image/"):
        return "vision"
    if content_type.startswith("audio/"):
        return "audio"
    return "sensor"


def _load_json(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _overlaps(claim: str, span_text: str) -> bool:
    """Cheap support check: at least half the claim's words appear in its span."""
    words = [word for word in re.findall(r"[a-z0-9]+", claim.lower()) if len(word) > 2]
    if not words:
        return True
    span_words = set(re.findall(r"[a-z0-9]+", span_text.lower()))
    return sum(1 for word in words if word in span_words) >= max(1, len(words) // 2)


__all__ = [
    "Candidate",
    "ExtractionError",
    "ExtractionOutcome",
    "ExtractionResult",
    "GenerativeExtractor",
    "PROMPT_VERSION",
    "Rejection",
    "StructuredExtractor",
    "parse_claims",
    "validate_candidate",
]
