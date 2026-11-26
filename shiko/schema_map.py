"""The property-group map: every group, its tables and the phase that owns it.

`memory-layers-v2.md` describes roughly four hundred properties. They are not
columns on one table; they are groups with different read/write behaviour,
different growth rates and different phases. This map is the contract tests
assert against, so a group cannot be added without an owner and a table cannot
appear without a group.
"""

from __future__ import annotations

from .storage import Store

PROPERTY_GROUPS: tuple[dict, ...] = (
    {
        "group": "identity", "tables": ("canonical_memories",), "phase": "2.1", "status": "present",
        "note": "stable id, kind, subject, status, revision pointer, learned time",
    },
    {
        "group": "classification", "tables": ("classifications",), "phase": "2.1", "status": "present",
        "note": "axis/value rows; one memory can be many things",
    },
    {
        "group": "provenance", "tables": ("evidence_links", "artifacts"), "phase": "2.1", "status": "present",
        "note": "role, span, independence root, artifact hash and size",
    },
    {
        "group": "temporal", "tables": ("memory_temporal",), "phase": "2.1", "status": "present",
        "note": "event time, validity interval, granularity",
    },
    {
        "group": "revisions", "tables": ("memory_revisions",), "phase": "2.1", "status": "present",
        "note": "knowledge evolution with policy and source event",
    },
    {
        "group": "lifecycle", "tables": ("memory_events", "memory_event_rollups"), "phase": "2.5", "status": "present",
        "note": "bounded raw events plus daily rollups",
    },
    {
        "group": "dynamics", "tables": ("memory_dynamics",), "phase": "2.5", "status": "present",
        "note": "activation, strength, reinforcement, decay; evaluated lazily",
    },
    {
        "group": "affect", "tables": ("memory_affect",), "phase": "8.1", "status": "present",
        "note": "VAD with NULL for unknown and 0.0 for a measurement",
    },
    {
        "group": "entities", "tables": ("entity_refs",), "phase": "7.2", "status": "partial",
        "note": "entity references exist; profiles and memberships arrive in 7.2",
    },
    {
        "group": "associations", "tables": (), "phase": "5.1", "status": "planned",
        "note": "typed weighted edges with their own dynamics",
    },
    {
        "group": "vectors", "tables": (), "phase": "4.1", "status": "planned",
        "note": "semantic, structural and sensory representations",
    },
    {
        "group": "context", "tables": (), "phase": "8.1", "status": "planned",
        "note": "encoding circumstances, goals, state",
    },
    {
        "group": "predictions", "tables": (), "phase": "8.2", "status": "planned",
        "note": "expectations, prediction error, novelty, surprise",
    },
    {
        "group": "intentions", "tables": (), "phase": "8.2", "status": "planned",
        "note": "prospective triggers, prerequisites, completion evidence",
    },
    {
        "group": "profiles", "tables": (), "phase": "7.2", "status": "planned",
        "note": "derived entity representations and memberships",
    },
)


def schema_map(store: Store) -> list[dict]:
    """Return the map with a live check for each present table."""
    present: set[str] = {
        str(row[0])
        for row in store.read(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    result: list[dict] = []
    for group in PROPERTY_GROUPS:
        tables = tuple(group["tables"])
        result.append({
            **group,
            "tables": tables,
            "tables_present": all(table in present for table in tables),
        })
    return result


def missing_present_tables(store: Store) -> list[str]:
    """Tables that a present group claims but the store lacks."""
    return [
        table
        for group in schema_map(store)
        if group["status"] == "present"
        for table in group["tables"]
        if not group["tables_present"]
    ]
