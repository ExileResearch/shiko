"""Canonical memories, revisions, evidence and optional property groups.

A memory is an interpretation, not an observation. Its identity stays stable
while revisions accumulate; every revision points at the source event that
produced it. Evidence links carry the role, the span in the source and an
independence root, so a paraphrase cannot masquerade as a second observation.

Optional properties are grouped by behaviour. ``None`` means unknown; ``0.0`` is
a measured value and survives a round trip as one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from .dynamics import effective_activation as _effective_activation
from .errors import ShikoError
from .storage import Store, new_id


class RecordError(ShikoError, ValueError):
    """A memory, revision or evidence record is malformed or inconsistent."""

    code = "record.invalid"


MEMORY_KINDS = frozenset({"episodic", "semantic", "procedural", "commitment", "profile", "reflection"})
EVIDENCE_ROLES = frozenset({"supports", "contradicts", "derives", "describes"})
AFFECT_SOURCES = frozenset({"attributed", "estimated", "unknown"})
MEMORY_STATUSES = frozenset({"active", "superseded", "retracted", "draft"})


@dataclass(frozen=True)
class EvidenceInput:
    event_id: str
    role: str = "supports"
    span: tuple[int, int] | None = None
    independence_root: str | None = None

    def __post_init__(self) -> None:
        if not str(self.event_id or "").strip():
            raise RecordError("evidence.event_id is required")
        role = str(self.role or "supports").strip()
        if role not in EVIDENCE_ROLES:
            raise RecordError(f"unknown evidence role '{role}'", detail={"role": role})
        object.__setattr__(self, "role", role)
        if self.span is not None:
            start, end = self.span
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                raise RecordError("evidence span must be a (start, end) pair of non-negative ints")
            object.__setattr__(self, "span", (start, end))


@dataclass(frozen=True)
class ClassificationInput:
    axis: str
    value: str
    confidence: float | None = None
    method: str = "deterministic"
    model_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("axis", self.axis), ("value", self.value), ("method", self.method)):
            if not str(value or "").strip():
                raise RecordError(f"classification.{name} is required")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise RecordError("classification.confidence must be within [0, 1]")


@dataclass(frozen=True)
class Affect:
    """Attributed affect. Every field defaults to unknown (SQL NULL)."""

    valence: float | None = None
    arousal: float | None = None
    dominance: float | None = None
    confidence: float | None = None
    source: str = "attributed"
    model_id: str | None = None

    def __post_init__(self) -> None:
        if self.source not in AFFECT_SOURCES:
            raise RecordError(f"unknown affect source '{self.source}'", detail={"source": self.source})
        for name in ("valence", "arousal", "dominance", "confidence"):
            value = getattr(self, name)
            if value is not None and not -1.0 <= float(value) <= 1.0 and name != "confidence":
                raise RecordError(f"affect.{name} must be within [-1, 1]")
            if name == "confidence" and value is not None and not 0.0 <= float(value) <= 1.0:
                raise RecordError("affect.confidence must be within [0, 1]")


@dataclass(frozen=True)
class MemoryDraft:
    kind: str
    content: str
    evidence: tuple[EvidenceInput, ...]
    classifications: tuple[ClassificationInput, ...] = ()
    affect: Affect | None = None
    subject_kind: str | None = None
    subject_id: str | None = None
    event_time: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    time_granularity: str = "unknown"
    status: str = "active"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in MEMORY_KINDS:
            raise RecordError(f"unknown memory kind '{self.kind}'", detail={"kind": self.kind})
        if not str(self.content or "").strip():
            raise RecordError("memory content is required")
        if not self.evidence:
            raise RecordError("a canonical memory needs at least one evidence link")
        if self.status not in MEMORY_STATUSES:
            raise RecordError(f"unknown memory status '{self.status}'", detail={"status": self.status})
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "classifications", tuple(self.classifications))
        try:
            json.dumps(dict(self.metadata))
        except (TypeError, ValueError) as exc:
            raise RecordError(f"metadata must be JSON-serialisable: {exc}") from exc


def create_memory(
    store: Store,
    draft: MemoryDraft,
    *,
    policy_version: str,
    learned_at: datetime | None = None,
) -> dict[str, Any]:
    """Insert a memory, its first revision and every attached observation."""
    memory_id = new_id("mem")
    now = learned_at or datetime.now(timezone.utc)
    learned = _iso(now)
    event_times: list[str] = []
    roots: dict[str, str] = {}
    for item in draft.evidence:
        row = store.read_one(
            "SELECT observed_at, metadata_json FROM source_events WHERE event_id=?", (item.event_id,)
        )
        if row is None:
            raise RecordError(
                "evidence references an event that is not in this store",
                detail={"event_id": item.event_id, "memory_id": memory_id},
            )
        event_times.append(str(row[0]))
        root = item.independence_root
        if root is None:
            try:
                metadata = json.loads(row[1] or "{}")
            except ValueError:
                metadata = {}
            root = str(metadata.get("independence_root") or item.event_id)
        roots[item.event_id] = root

    event_time = _iso(draft.event_time) if draft.event_time else (event_times[0] if event_times else None)
    valid_from = _iso(draft.valid_from) if draft.valid_from else event_time
    valid_to = _iso(draft.valid_to) if draft.valid_to else None

    with store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO canonical_memories
                (memory_id, owner_id, kind, subject_kind, subject_id, status, current_revision,
                 created_at, learned_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                memory_id, store.owner_id, draft.kind, draft.subject_kind, draft.subject_id,
                draft.status, learned, learned, json.dumps(dict(draft.metadata), sort_keys=True),
            ),
        )
        conn.execute(
            """
            INSERT INTO memory_revisions
                (memory_id, revision, content, status, event_time, valid_from, valid_to,
                 learned_at, policy_version, source_event_id, created_at)
            VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id, draft.content, draft.status, event_time, valid_from, valid_to,
                learned, policy_version, draft.evidence[0].event_id, learned,
            ),
        )
        for item in draft.evidence:
            conn.execute(
                """
                INSERT OR IGNORE INTO evidence_links
                    (link_id, memory_id, event_id, role, span_start, span_end, independence_root, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("evl"), memory_id, item.event_id, item.role,
                    item.span[0] if item.span else None, item.span[1] if item.span else None,
                    roots[item.event_id], learned,
                ),
            )
        if draft.subject_id:
            conn.execute(
                """
                INSERT INTO entity_refs
                    (entity_ref_id, entity_id, entity_kind, display_name, event_id, memory_id, created_at)
                VALUES (?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    new_id("eref"), draft.subject_id, draft.subject_kind or "entity",
                    draft.subject_id, memory_id, learned,
                ),
            )
        for classification in draft.classifications:
            conn.execute(
                """
                INSERT OR IGNORE INTO classifications
                    (classification_id, memory_id, axis, value, confidence, method, model_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("cls"), memory_id, classification.axis, classification.value,
                    classification.confidence, classification.method, classification.model_id, learned,
                ),
            )
        if draft.affect is not None:
            conn.execute(
                """
                INSERT INTO memory_affect
                    (memory_id, valence, arousal, dominance, confidence, source, model_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id, draft.affect.valence, draft.affect.arousal, draft.affect.dominance,
                    draft.affect.confidence, draft.affect.source, draft.affect.model_id, learned,
                ),
            )
        conn.execute(
            """
            INSERT INTO memory_temporal
                (memory_id, event_time, valid_from, valid_to, time_granularity, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (memory_id, event_time, valid_from, valid_to, draft.time_granularity, learned),
        )
        _insert_event(
            conn, memory_id, "created", at=learned, policy_version=policy_version,
            detail={"kind": draft.kind, "revision": 1},
        )
    return get_memory(store, memory_id) or {}


def revise_memory(
    store: Store,
    memory_id: str,
    content: str,
    *,
    policy_version: str,
    source_event_id: str | None = None,
    event_time: datetime | None = None,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> int:
    """Add a revision; the previous one stays readable and attributed."""
    if not str(content or "").strip():
        raise RecordError("revision content is required")
    now = _iso(datetime.now(timezone.utc))
    with store.transaction() as conn:
        row = conn.execute(
            "SELECT current_revision, status FROM canonical_memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        if row is None:
            raise RecordError("memory not found", detail={"memory_id": memory_id})
        revision = int(row[0]) + 1
        event_time_s = _iso(event_time) if event_time else None
        if event_time_s is None:
            event_time_s = conn.execute(
                "SELECT event_time FROM memory_temporal WHERE memory_id=?", (memory_id,)
            ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO memory_revisions
                (memory_id, revision, content, status, event_time, valid_from, valid_to,
                 learned_at, policy_version, source_event_id, created_at)
            VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id, revision, content, event_time_s,
                _iso(valid_from) if valid_from else None, _iso(valid_to) if valid_to else None,
                now, policy_version, source_event_id, now,
            ),
        )
        conn.execute(
            "UPDATE canonical_memories SET current_revision=?, status='active' WHERE memory_id=?",
            (revision, memory_id),
        )
        conn.execute(
            "UPDATE memory_temporal SET event_time=?, updated_at=? WHERE memory_id=?",
            (event_time_s, now, memory_id),
        )
        _insert_event(
            conn, memory_id, "revised", at=now, policy_version=policy_version,
            detail={"revision": revision, "source_event_id": source_event_id},
        )
    return revision


def get_memory(store: Store, memory_id: str) -> dict[str, Any] | None:
    row = store.read_one(
        """
        SELECT memory_id, owner_id, kind, subject_kind, subject_id, status, current_revision,
               created_at, learned_at, metadata_json
        FROM canonical_memories WHERE memory_id=?
        """,
        (memory_id,),
    )
    if row is None:
        return None
    revisions = [
        {
            "revision": int(r[0]), "content": r[1], "status": r[2], "event_time": r[3],
            "valid_from": r[4], "valid_to": r[5], "learned_at": r[6],
            "policy_version": r[7], "source_event_id": r[8], "created_at": r[9],
        }
        for r in store.read(
            """
            SELECT revision, content, status, event_time, valid_from, valid_to, learned_at,
                   policy_version, source_event_id, created_at
            FROM memory_revisions WHERE memory_id=? ORDER BY revision
            """,
            (memory_id,),
        )
    ]
    evidence = [
        {
            "event_id": r[0], "role": r[1],
            "span": [int(r[2]), int(r[3])] if r[2] is not None and r[3] is not None else None,
            "independence_root": r[4],
        }
        for r in store.read(
            "SELECT event_id, role, span_start, span_end, independence_root "
            "FROM evidence_links WHERE memory_id=? ORDER BY created_at, event_id",
            (memory_id,),
        )
    ]
    classifications = [
        {"axis": r[0], "value": r[1], "confidence": r[2], "method": r[3], "model_id": r[4]}
        for r in store.read(
            "SELECT axis, value, confidence, method, model_id FROM classifications "
            "WHERE memory_id=? ORDER BY axis, value",
            (memory_id,),
        )
    ]
    affect_row = store.read_one(
        "SELECT valence, arousal, dominance, confidence, source, model_id "
        "FROM memory_affect WHERE memory_id=?",
        (memory_id,),
    )
    affect = None
    if affect_row is not None:
        affect = {
            "valence": affect_row[0], "arousal": affect_row[1], "dominance": affect_row[2],
            "confidence": affect_row[3], "source": affect_row[4], "model_id": affect_row[5],
            "unknown": [name for name, value in zip(
                ("valence", "arousal", "dominance"), affect_row[:3]
            ) if value is None],
        }
    temporal_row = store.read_one(
        "SELECT event_time, valid_from, valid_to, time_granularity FROM memory_temporal WHERE memory_id=?",
        (memory_id,),
    )
    temporal = None
    if temporal_row is not None:
        temporal = {
            "event_time": temporal_row[0], "valid_from": temporal_row[1],
            "valid_to": temporal_row[2], "granularity": temporal_row[3],
        }
    derived_from = [
        {"parent_memory_id": row[0], "method": row[1]}
        for row in store.read(
            "SELECT parent_memory_id, method FROM memory_derivations "
            "WHERE memory_id=? ORDER BY created_at, parent_memory_id",
            (memory_id,),
        )
    ]
    return {
        "memory_id": row[0],
        "owner_id": row[1],
        "kind": row[2],
        "subject": {"kind": row[3], "id": row[4]} if row[3] or row[4] else None,
        "status": row[5],
        "current_revision": int(row[6]),
        "created_at": row[7],
        "learned_at": row[8],
        "metadata": json.loads(row[9] or "{}"),
        "revisions": revisions,
        "evidence": evidence,
        "classifications": classifications,
        "affect": affect,
        "temporal": temporal,
        "derived_from": derived_from,
    }


def list_memories(store: Store, *, limit: int = 100) -> list[dict[str, Any]]:
    rows = store.read(
        "SELECT memory_id FROM canonical_memories ORDER BY created_at, memory_id LIMIT ?", (int(limit),)
    )
    return [memory for row in rows if (memory := get_memory(store, row[0])) is not None]


def attach_artifact(
    store: Store, event_id: str, *, uri: str, media_type: str, size_bytes: int, sha256: str
) -> str:
    """Record an artifact reference and its integrity metadata."""
    if store.read_one("SELECT 1 FROM source_events WHERE event_id=?", (event_id,)) is None:
        raise RecordError("artifact references an event that is not in this store", detail={"event_id": event_id})
    artifact_id = new_id("art")
    with store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO artifacts (artifact_id, event_id, uri, media_type, size_bytes, sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (artifact_id, event_id, uri, media_type, int(size_bytes), sha256, _iso(datetime.now(timezone.utc))),
        )
    return artifact_id


def artifacts_for_event(store: Store, event_id: str) -> list[dict[str, Any]]:
    return [
        {"artifact_id": r[0], "uri": r[1], "media_type": r[2], "size_bytes": int(r[3]), "sha256": r[4]}
        for r in store.read(
            "SELECT artifact_id, uri, media_type, size_bytes, sha256 FROM artifacts "
            "WHERE event_id=? ORDER BY created_at",
            (event_id,),
        )
    ]


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise RecordError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# -- entity identity, roots and derivation -----------------------------------


def normalize_name(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def register_entity(
    store: Store,
    kind: str,
    name: str,
    *,
    entity_id: str | None = None,
    aliases: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or update an entity and its aliases in deployment scope."""
    if not str(name or "").strip():
        raise RecordError("entity name is required")
    if not str(kind or "").strip():
        raise RecordError("entity kind is required")
    entity = entity_id or new_id("ent")
    now = _iso(datetime.now(timezone.utc))
    with store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO entities (entity_id, entity_kind, canonical_name, canonical_norm, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                entity_kind=excluded.entity_kind, canonical_name=excluded.canonical_name,
                canonical_norm=excluded.canonical_norm, metadata_json=excluded.metadata_json
            """,
            (entity, kind, str(name).strip(), normalize_name(name), json.dumps(dict(metadata or {})), now),
        )
        for alias in (str(name), *aliases):
            conn.execute(
                """
                INSERT OR IGNORE INTO entity_aliases (alias_norm, entity_id, alias, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (normalize_name(alias), entity, str(alias).strip(), now),
            )
    return get_entity(store, entity) or {}


def get_entity(store: Store, entity_id: str) -> dict[str, Any] | None:
    row = store.read_one(
        "SELECT entity_id, entity_kind, canonical_name, metadata_json FROM entities WHERE entity_id=?",
        (entity_id,),
    )
    if row is None:
        return None
    aliases = [
        alias for (alias,) in store.read(
            "SELECT alias FROM entity_aliases WHERE entity_id=? ORDER BY alias", (entity_id,)
        )
    ]
    return {
        "entity_id": row[0], "kind": row[1], "name": row[2],
        "aliases": aliases, "metadata": json.loads(row[3] or "{}"),
    }


def link_alias(store: Store, alias: str, entity_id: str, *, source_event_id: str | None = None) -> None:
    if store.read_one("SELECT 1 FROM entities WHERE entity_id=?", (entity_id,)) is None:
        raise RecordError("alias references an unknown entity", detail={"entity_id": entity_id})
    with store.transaction() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO entity_aliases (alias_norm, entity_id, alias, source_event_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (normalize_name(alias), entity_id, str(alias).strip(), source_event_id,
             _iso(datetime.now(timezone.utc))),
        )


def resolve_entity(store: Store, name: str) -> dict[str, Any]:
    """Resolve a name or alias, preserving ambiguity instead of guessing."""
    norm = normalize_name(name)
    rows = store.read(
        """
        SELECT DISTINCT e.entity_id, e.entity_kind, e.canonical_name
        FROM entities e LEFT JOIN entity_aliases a ON a.entity_id = e.entity_id
        WHERE e.canonical_norm = ? OR a.alias_norm = ?
        ORDER BY e.entity_id
        """,
        (norm, norm),
    )
    matches = [
        {
            "entity_id": row[0], "kind": row[1], "name": row[2],
            "aliases": [alias for (alias,) in store.read(
                "SELECT alias FROM entity_aliases WHERE entity_id=? ORDER BY alias", (row[0],)
            )],
        }
        for row in rows
    ]
    return {"query": name, "normalized": norm, "matches": matches, "ambiguous": len(matches) > 1}


def entities_for_memory(store: Store, memory_id: str) -> list[dict[str, Any]]:
    return [
        {"entity_id": row[0], "kind": row[1], "display_name": row[2]}
        for row in store.read(
            "SELECT entity_id, entity_kind, display_name FROM entity_refs "
            "WHERE memory_id=? ORDER BY created_at, entity_id",
            (memory_id,),
        )
    ]


def note_evidence_root(
    store: Store,
    origin_event_id: str,
    *,
    root_id: str | None = None,
    kind: str = "observation",
    parent_root_id: str | None = None,
) -> str:
    """Record where an independent evidence root came from (echo, summary, import)."""
    if store.read_one("SELECT 1 FROM source_events WHERE event_id=?", (origin_event_id,)) is None:
        raise RecordError("evidence root references an unknown event", detail={"event_id": origin_event_id})
    root = root_id or new_id("root")
    with store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO evidence_roots (root_id, origin_event_id, kind, parent_root_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(root_id) DO UPDATE SET
                origin_event_id=excluded.origin_event_id, kind=excluded.kind,
                parent_root_id=excluded.parent_root_id
            """,
            (root, origin_event_id, kind, parent_root_id, _iso(datetime.now(timezone.utc))),
        )
    return root


def evidence_roots(store: Store, *, limit: int = 200) -> list[dict[str, Any]]:
    return [
        {"root_id": row[0], "origin_event_id": row[1], "kind": row[2], "parent_root_id": row[3]}
        for row in store.read(
            "SELECT root_id, origin_event_id, kind, parent_root_id FROM evidence_roots "
            "ORDER BY created_at, root_id LIMIT ?",
            (int(limit),),
        )
    ]


def derive_memory(
    store: Store,
    parent_memory_ids: list[str],
    *,
    content: str,
    kind: str = "reflection",
    method: str = "summary",
    policy_version: str,
    classifications: tuple[ClassificationInput, ...] = (),
    subject_kind: str | None = None,
    subject_id: str | None = None,
) -> dict[str, Any]:
    """Create a derived memory from parents, inheriting their evidence roots."""
    if not parent_memory_ids:
        raise RecordError("derivation needs at least one parent memory")
    evidence: list[EvidenceInput] = []
    seen: set[tuple[str, str]] = set()
    for parent in parent_memory_ids:
        if get_memory(store, parent) is None:
            raise RecordError("derivation references an unknown memory", detail={"memory_id": parent})
        for event_id, root in store.read(
            "SELECT event_id, independence_root FROM evidence_links WHERE memory_id=? ORDER BY created_at",
            (parent,),
        ):
            key = (str(event_id), str(root))
            if key in seen:
                continue
            seen.add(key)
            evidence.append(EvidenceInput(str(event_id), "derives", independence_root=str(root)))
    if not evidence:
        raise RecordError("parents have no evidence to derive from")
    memory = create_memory(
        store,
        MemoryDraft(
            kind=kind, content=content, evidence=tuple(evidence),
            classifications=classifications, subject_kind=subject_kind, subject_id=subject_id,
            metadata={"derived": True, "method": method},
        ),
        policy_version=policy_version,
    )
    with store.transaction() as conn:
        for parent in parent_memory_ids:
            conn.execute(
                "INSERT OR IGNORE INTO memory_derivations (memory_id, parent_memory_id, method, created_at) "
                "VALUES (?, ?, ?, ?)",
                (memory["memory_id"], parent, method, _iso(datetime.now(timezone.utc))),
            )
    return get_memory(store, memory["memory_id"]) or {}


# -- lifecycle history -------------------------------------------------------


def _insert_event(
    conn,
    memory_id: str,
    kind: str,
    *,
    at: str,
    policy_version: str | None = None,
    model_id: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO memory_events (memory_id, kind, at, policy_version, model_id, detail_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (memory_id, kind, at, policy_version, model_id, json.dumps(dict(detail or {}), sort_keys=True)),
    )


def record_memory_event(
    store: Store,
    memory_id: str,
    kind: str,
    *,
    at: datetime | None = None,
    policy_version: str | None = None,
    model_id: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> None:
    """Append one lifecycle event. Only real state changes should call this."""
    if store.read_one("SELECT 1 FROM canonical_memories WHERE memory_id=?", (memory_id,)) is None:
        raise RecordError("event references an unknown memory", detail={"memory_id": memory_id})
    moment = at or datetime.now(timezone.utc)
    with store.transaction() as conn:
        _insert_event(
            conn, memory_id, kind, at=_iso(moment),
            policy_version=policy_version, model_id=model_id, detail=detail,
        )


def memory_events(
    store: Store, *, memory_id: str | None = None, kind: str | None = None, limit: int = 200,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if memory_id is not None:
        clauses.append("memory_id=?")
        params.append(memory_id)
    if kind is not None:
        clauses.append("kind=?")
        params.append(kind)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(int(limit))
    rows = store.read(
        f"SELECT seq, memory_id, kind, at, policy_version, model_id, detail_json "
        f"FROM memory_events {where} ORDER BY seq DESC LIMIT ?",
        tuple(params),
    )
    return [
        {
            "seq": row[0], "memory_id": row[1], "kind": row[2], "at": row[3],
            "policy_version": row[4], "model_id": row[5], "detail": json.loads(row[6] or "{}"),
        }
        for row in reversed(rows)
    ]


def prune_memory_events(
    store: Store, *, retention_days: int = 30, now: datetime | None = None,
) -> dict[str, Any]:
    """Move old raw events into daily rollups, then delete them."""
    if retention_days < 0:
        raise RecordError("retention_days must be non-negative")
    moment = now or datetime.now(timezone.utc)
    cutoff = _iso(moment - timedelta(days=int(retention_days)))
    with store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO memory_event_rollups (day, memory_id, kind, count)
            SELECT substr(at, 1, 10), memory_id, kind, COUNT(*)
            FROM memory_events WHERE at < ?
            GROUP BY 1, 2, 3
            ON CONFLICT(day, memory_id, kind) DO UPDATE SET count = count + excluded.count
            """,
            (cutoff,),
        )
        deleted = conn.execute("DELETE FROM memory_events WHERE at < ?", (cutoff,)).rowcount
    return {"cutoff": cutoff, "deleted": int(deleted or 0)}


def memory_event_rollups(
    store: Store, *, memory_id: str | None = None, limit: int = 200,
) -> list[dict[str, Any]]:
    if memory_id is None:
        rows = store.read(
            "SELECT day, memory_id, kind, count FROM memory_event_rollups "
            "ORDER BY day DESC, memory_id LIMIT ?",
            (int(limit),),
        )
    else:
        rows = store.read(
            "SELECT day, memory_id, kind, count FROM memory_event_rollups "
            "WHERE memory_id=? ORDER BY day DESC LIMIT ?",
            (memory_id, int(limit)),
        )
    return [{"day": r[0], "memory_id": r[1], "kind": r[2], "count": int(r[3])} for r in rows]


# -- dynamics ----------------------------------------------------------------


def set_dynamics(
    store: Store,
    memory_id: str,
    *,
    activation: float | None = None,
    baseline_activation: float | None = None,
    storage_strength: float | None = None,
    retrieval_strength: float | None = None,
    reinforcement: float | None = None,
    decay_rate: float | None = None,
    at: datetime | None = None,
) -> None:
    if store.read_one("SELECT 1 FROM canonical_memories WHERE memory_id=?", (memory_id,)) is None:
        raise RecordError("dynamics reference an unknown memory", detail={"memory_id": memory_id})
    moment = _iso(at or datetime.now(timezone.utc))
    with store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO memory_dynamics
                (memory_id, activation, baseline_activation, storage_strength, retrieval_strength,
                 reinforcement, decay_rate, last_updated, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                activation=COALESCE(excluded.activation, memory_dynamics.activation),
                baseline_activation=COALESCE(excluded.baseline_activation, memory_dynamics.baseline_activation),
                storage_strength=COALESCE(excluded.storage_strength, memory_dynamics.storage_strength),
                retrieval_strength=COALESCE(excluded.retrieval_strength, memory_dynamics.retrieval_strength),
                reinforcement=COALESCE(excluded.reinforcement, memory_dynamics.reinforcement),
                decay_rate=COALESCE(excluded.decay_rate, memory_dynamics.decay_rate),
                last_updated=excluded.last_updated,
                updated_at=excluded.updated_at
            """,
            (
                memory_id, activation, baseline_activation, storage_strength, retrieval_strength,
                reinforcement, decay_rate, moment, moment,
            ),
        )


def get_dynamics(store: Store, memory_id: str) -> dict[str, Any] | None:
    row = store.read_one(
        "SELECT activation, baseline_activation, storage_strength, retrieval_strength, "
        "reinforcement, decay_rate, last_updated FROM memory_dynamics WHERE memory_id=?",
        (memory_id,),
    )
    if row is None:
        return None
    return {
        "activation": row[0], "baseline_activation": row[1], "storage_strength": row[2],
        "retrieval_strength": row[3], "reinforcement": row[4], "decay_rate": row[5],
        "last_updated": row[6],
    }


def effective_activation_for(store: Store, memory_id: str, *, at: datetime | None = None) -> float | None:
    """Current activation from stored state, without writing anything."""
    record = get_dynamics(store, memory_id)
    if record is None:
        return None
    moment = at or datetime.now(timezone.utc)
    last = datetime.fromisoformat(str(record["last_updated"]).replace("Z", "+00:00")) if record["last_updated"] else None
    return _effective_activation(record["activation"], record["decay_rate"], last, moment)

