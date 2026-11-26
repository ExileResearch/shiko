"""Source envelopes: what arrived, from whom, and how it reached Shiko.

An envelope is the evidence side of the engine. It records the actor, the event
ID the source chose, the delegation chain, both clocks (when the source observed
the event and when Shiko received it), a visibility scope and a pointer back to
the original source. Interpretation happens later and never mutates this record.

Two deployments may legitimately receive the same incoming event ID; identity is
``(owner_id, event_id)``, and the owner is fixed by the data root, not the
envelope. See :mod:`shiko.owner`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from .errors import EnvelopeError

MAX_EVENT_ID = 256
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class ActorKind(str, Enum):
    OWNER = "owner"
    """The main agent that owns this deployment."""

    SUBAGENT = "subagent"
    """A delegated actor; ``delegation_parent`` names who delegated."""

    USER = "user"
    """A person talking to the agent."""

    TOOL = "tool"
    """A tool, sensor or service reporting an observation."""

    SYSTEM = "system"
    """The host runtime itself (session lifecycle, compaction, ...)."""

    EXTERNAL = "external"
    """Another system or deployment relaying evidence."""


class Visibility(str, Enum):
    PRIVATE = "private"
    """Only this deployment's owner may recall it."""

    DEPLOYMENT = "deployment"
    """Any runtime sharing this data root may recall it."""

    SHAREABLE = "shareable"
    """The owner marked it as exportable; still not public by default."""


@dataclass(frozen=True)
class Actor:
    id: str
    kind: ActorKind
    display_name: str = ""

    def __post_init__(self) -> None:
        actor_id = _clean(self.id, "actor.id")
        object.__setattr__(self, "id", actor_id)
        if not isinstance(self.kind, ActorKind):
            object.__setattr__(self, "kind", ActorKind(str(self.kind)))
        object.__setattr__(self, "display_name", (self.display_name or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind.value, "display_name": self.display_name}


@dataclass(frozen=True)
class SourceRef:
    """A pointer to the original material this envelope came from."""

    kind: str
    id: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _clean(self.kind, "source.kind"))
        object.__setattr__(self, "id", (self.id or "").strip())
        object.__setattr__(self, "detail", dict(self.detail))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "id": self.id, "detail": dict(self.detail)}


@dataclass(frozen=True)
class Subject:
    """Someone or something the content is *about*, not the author."""

    id: str
    kind: str = "entity"
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _clean(self.id, "subject.id"))
        object.__setattr__(self, "kind", _clean(self.kind, "subject.kind"))
        object.__setattr__(self, "label", (self.label or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "label": self.label}


@dataclass(frozen=True)
class SourceEnvelope:
    """One immutable observation as it reached this deployment."""

    event_id: str
    actor: Actor
    session_id: str
    observed_at: datetime
    received_at: datetime
    source: SourceRef
    content: str
    content_type: str = "text/plain"
    visibility: Visibility = Visibility.PRIVATE
    delegation_parent: str | None = None
    subjects: tuple[Subject, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        event_id = _clean(self.event_id, "event_id")
        if len(event_id) > MAX_EVENT_ID:
            raise EnvelopeError(f"event_id exceeds {MAX_EVENT_ID} characters", detail={"event_id": event_id})
        if _CONTROL.search(event_id):
            raise EnvelopeError("event_id contains control characters", detail={"event_id": event_id})
        object.__setattr__(self, "event_id", event_id)

        if not isinstance(self.actor, Actor):
            raise EnvelopeError("actor must be an Actor", detail={"event_id": event_id})
        if not isinstance(self.source, SourceRef):
            raise EnvelopeError("source must be a SourceRef", detail={"event_id": event_id})

        object.__setattr__(self, "session_id", _clean(self.session_id, "session_id"))
        object.__setattr__(self, "observed_at", _as_utc(self.observed_at, "observed_at"))
        object.__setattr__(self, "received_at", _as_utc(self.received_at, "received_at"))

        if not isinstance(self.visibility, Visibility):
            object.__setattr__(self, "visibility", Visibility(str(self.visibility)))
        parent = (self.delegation_parent or "").strip() or None
        object.__setattr__(self, "delegation_parent", parent)
        object.__setattr__(self, "content", self.content if isinstance(self.content, str) else str(self.content))
        object.__setattr__(self, "content_type", _clean(self.content_type, "content_type"))
        object.__setattr__(self, "subjects", tuple(self.subjects))
        object.__setattr__(self, "metadata", dict(self.metadata))
        _ensure_jsonable(self.metadata, "metadata")
        for subject in self.subjects:
            if not isinstance(subject, Subject):
                raise EnvelopeError("subjects must contain Subject values", detail={"event_id": event_id})

    # -- derived values -----------------------------------------------------

    @property
    def clock_skew_seconds(self) -> float:
        """Positive when the source clock reported the event after Shiko received it."""
        return (self.observed_at - self.received_at).total_seconds()

    def content_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "actor": self.actor.to_dict(),
            "session_id": self.session_id,
            "observed_at": _iso(self.observed_at),
            "received_at": _iso(self.received_at),
            "source": self.source.to_dict(),
            "content": self.content,
            "content_type": self.content_type,
            "visibility": self.visibility.value,
            "delegation_parent": self.delegation_parent,
            "subjects": [s.to_dict() for s in self.subjects],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceEnvelope":
        try:
            actor = data["actor"]
            source = data["source"]
            return cls(
                event_id=data["event_id"],
                actor=Actor(id=actor["id"], kind=actor["kind"], display_name=actor.get("display_name", "")),
                session_id=data["session_id"],
                observed_at=_parse_time(data["observed_at"]),
                received_at=_parse_time(data["received_at"]),
                source=SourceRef(kind=source["kind"], id=source.get("id", ""), detail=source.get("detail", {})),
                content=data.get("content", ""),
                content_type=data.get("content_type", "text/plain"),
                visibility=data.get("visibility", Visibility.PRIVATE.value),
                delegation_parent=data.get("delegation_parent"),
                subjects=tuple(Subject(**s) for s in data.get("subjects", ())),
                metadata=data.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EnvelopeError(f"cannot rebuild envelope: {exc}", detail={"data": dict(data)}) from exc

    def canonical_json(self) -> str:
        """Stable serialisation for hashing and replay comparison."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_envelope(
    content: str,
    *,
    event_id: str,
    actor: Actor,
    session_id: str,
    source_kind: str,
    source_id: str = "",
    source_detail: Mapping[str, Any] | None = None,
    observed_at: datetime | None = None,
    received_at: datetime | None = None,
    visibility: Visibility = Visibility.PRIVATE,
    delegation_parent: str | None = None,
    subjects: tuple[Subject, ...] = (),
    metadata: Mapping[str, Any] | None = None,
    content_type: str = "text/plain",
) -> SourceEnvelope:
    """Construct an envelope with sane clocks, for hosts and Python callers."""
    now = datetime.now(timezone.utc)
    return SourceEnvelope(
        event_id=event_id,
        actor=actor,
        session_id=session_id,
        observed_at=observed_at or now,
        received_at=received_at or now,
        source=SourceRef(kind=source_kind, id=source_id, detail=source_detail or {}),
        content=content,
        content_type=content_type,
        visibility=visibility,
        delegation_parent=delegation_parent,
        subjects=subjects,
        metadata=metadata or {},
    )


def _clean(value: Any, field_name: str) -> str:
    text = (value or "").strip() if isinstance(value, str) else str(value).strip()
    if not text:
        raise EnvelopeError(f"{field_name} must be a non-empty string", detail={"field": field_name})
    return text


def _as_utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise EnvelopeError(f"{field_name} must be a datetime", detail={"field": field_name})
    if value.tzinfo is None:
        raise EnvelopeError(
            f"{field_name} must be timezone-aware (naive timestamps are ambiguous)",
            detail={"field": field_name},
        )
    return value.astimezone(timezone.utc)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_jsonable(value: Mapping[str, Any], field_name: str) -> None:
    try:
        json.dumps(dict(value))
    except (TypeError, ValueError) as exc:
        raise EnvelopeError(f"{field_name} must be JSON-serialisable: {exc}", detail={"field": field_name}) from exc
