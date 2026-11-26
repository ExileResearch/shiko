"""Episodes: ordered experience with goals, actions and delayed outcomes.

An episode is a narrative container over canonical events and memories. A step
points at its source event when one exists; a reflection may have no event at
all. Actions carry a stable correlation id so an outcome that arrives in another
session attaches to the right attempt, and a missing result stays ``pending``
rather than being guessed.

Two simultaneous experiments have different correlation ids even in one session.
A late result reopens nothing by itself; it appends an outcome step and refreshes
the episode status from its actions.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .errors import ShikoError
from .storage import Store, new_id

EPISODE_ROLES = frozenset({"observation", "action", "outcome", "reflection", "note"})
EPISODE_STATUSES = frozenset({"open", "completed", "failed", "abandoned"})
ACTION_STATUSES = frozenset({"pending", "completed", "failed", "unknown"})


class EpisodeError(ShikoError, ValueError):
    code = "episode.invalid"


def start_episode(
    store: Store,
    title: str,
    *,
    goal: str | None = None,
    participants: Iterable[str] = (),
    tags: Iterable[str] = (),
    session_id: str | None = None,
    at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not str(title or "").strip():
        raise EpisodeError("episode title is required")
    episode_id = new_id("ep")
    moment = _iso(at or datetime.now(timezone.utc))
    with store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO episodes
                (episode_id, owner_id, title, goal, status, session_id, started_at,
                 tags_json, participants_json, metadata_json)
            VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?)
            """,
            (
                episode_id, store.owner_id, str(title).strip(),
                str(goal).strip() if goal else None, session_id, moment,
                json.dumps([str(item) for item in tags]),
                json.dumps([str(item) for item in participants]),
                json.dumps(dict(metadata or {}), sort_keys=True),
            ),
        )
    return get_episode(store, episode_id) or {}


def add_step(
    store: Store,
    episode_id: str,
    role: str,
    content: str,
    *,
    event_id: str | None = None,
    memory_id: str | None = None,
    outcome: str | None = None,
    correlation_id: str | None = None,
    occurred_at: datetime | None = None,
    session_id: str | None = None,
    scope: str = "internal",
    actor_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if role not in EPISODE_ROLES:
        raise EpisodeError(f"unknown episode role '{role}'", detail={"role": role})
    if not (str(content or "").strip() or event_id or memory_id):
        raise EpisodeError("a step needs content, an event or a memory")
    if outcome is not None and outcome not in {"success", "failure", "unknown"}:
        raise EpisodeError(f"unknown outcome '{outcome}'", detail={"outcome": outcome})
    if role == "outcome" and not correlation_id:
        raise EpisodeError("an outcome step needs a correlation id")
    if correlation_id and role not in {"action", "outcome"}:
        raise EpisodeError("only actions and outcomes carry a correlation id")
    if event_id and store.read_one("SELECT 1 FROM source_events WHERE event_id=?", (event_id,)) is None:
        raise EpisodeError("step references an unknown event", detail={"event_id": event_id})
    if memory_id and store.read_one("SELECT 1 FROM canonical_memories WHERE memory_id=?", (memory_id,)) is None:
        raise EpisodeError("step references an unknown memory", detail={"memory_id": memory_id})

    moment = _iso(occurred_at or datetime.now(timezone.utc))
    step_id = new_id("step")
    with store.transaction() as conn:
        row = conn.execute("SELECT started_at, ended_at FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        if row is None:
            raise EpisodeError("unknown episode", detail={"episode_id": episode_id})
        sequence = int(conn.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM episode_steps WHERE episode_id=?", (episode_id,)
        ).fetchone()[0]) + 1

        if role == "action":
            correlation_id = correlation_id or new_id("corr")
            conn.execute(
                """
                INSERT INTO episode_actions
                    (correlation_id, episode_id, action_step_id, status, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(correlation_id) DO NOTHING
                """,
                (correlation_id, episode_id, step_id, moment, moment),
            )
        elif role == "outcome":
            action = conn.execute(
                "SELECT action_step_id, status FROM episode_actions WHERE correlation_id=? AND episode_id=?",
                (correlation_id, episode_id),
            ).fetchone()
            if action is None:
                raise EpisodeError("outcome references an unknown correlation",
                                   detail={"correlation_id": correlation_id})
            resolved = _action_status(outcome)
            conn.execute(
                """
                UPDATE episode_actions
                SET outcome_step_id=?, status=?, updated_at=?
                WHERE correlation_id=?
                """,
                (step_id, resolved, moment, correlation_id),
            )
            conn.execute("UPDATE episode_steps SET status=? WHERE step_id=?", (resolved, action[0]))

        step_status = "pending" if role == "action" else "recorded"
        conn.execute(
            """
            INSERT INTO episode_steps
                (step_id, episode_id, sequence, role, content, event_id, memory_id,
                 correlation_id, outcome, status, occurred_at, session_id, scope, actor_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_id, episode_id, sequence, role, str(content or ""), event_id, memory_id,
                correlation_id, outcome, step_status, moment, session_id, scope, actor_id,
                json.dumps(dict(metadata or {}), sort_keys=True),
            ),
        )
        if row[1] is not None:  # a late step reopens a manually closed episode
            conn.execute(
                "UPDATE episodes SET ended_at=NULL, duration_seconds=NULL, status='open' WHERE episode_id=?",
                (episode_id,),
            )
        _refresh_episode(conn, episode_id, moment)
    return get_episode(store, episode_id) or {}


def record_outcome(
    store: Store,
    episode_id: str,
    correlation_id: str,
    content: str,
    *,
    outcome: str = "success",
    event_id: str | None = None,
    memory_id: str | None = None,
    occurred_at: datetime | None = None,
    session_id: str | None = None,
    actor_id: str | None = None,
    scope: str = "internal",
) -> dict[str, Any]:
    return add_step(
        store, episode_id, "outcome", content,
        event_id=event_id, memory_id=memory_id, outcome=outcome,
        correlation_id=correlation_id, occurred_at=occurred_at,
        session_id=session_id, actor_id=actor_id, scope=scope,
    )


def link_memory(store: Store, episode_id: str, memory_id: str, *, role: str = "supports") -> None:
    if store.read_one("SELECT 1 FROM canonical_memories WHERE memory_id=?", (memory_id,)) is None:
        raise EpisodeError("link references an unknown memory", detail={"memory_id": memory_id})
    with store.transaction() as conn:
        if conn.execute("SELECT 1 FROM episodes WHERE episode_id=?", (episode_id,)).fetchone() is None:
            raise EpisodeError("unknown episode", detail={"episode_id": episode_id})
        conn.execute(
            "INSERT OR IGNORE INTO episode_memories (episode_id, memory_id, role) VALUES (?, ?, ?)",
            (episode_id, memory_id, str(role)),
        )


def close_episode(
    store: Store, episode_id: str, *, status: str = "completed", at: datetime | None = None,
) -> dict[str, Any]:
    if status not in {"completed", "failed", "abandoned"}:
        raise EpisodeError(f"cannot close an episode as '{status}'")
    moment = at or datetime.now(timezone.utc)
    with store.transaction() as conn:
        row = conn.execute("SELECT started_at FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        if row is None:
            raise EpisodeError("unknown episode", detail={"episode_id": episode_id})
        duration = (moment - _parse(row[0])).total_seconds()
        conn.execute(
            "UPDATE episodes SET status=?, ended_at=?, duration_seconds=? WHERE episode_id=?",
            (status, _iso(moment), max(0.0, duration), episode_id),
        )
    return get_episode(store, episode_id) or {}


def get_episode(store: Store, episode_id: str) -> dict[str, Any] | None:
    row = store.read_one(
        """
        SELECT episode_id, title, goal, status, session_id, started_at, ended_at,
               duration_seconds, tags_json, participants_json, metadata_json
        FROM episodes WHERE episode_id=?
        """,
        (episode_id,),
    )
    if row is None:
        return None
    steps = [
        {
            "step_id": item[0], "sequence": int(item[1]), "role": item[2],
            "content": item[3], "event_id": item[4], "memory_id": item[5],
            "correlation_id": item[6], "outcome": item[7], "status": item[8],
            "occurred_at": item[9], "session_id": item[10], "scope": item[11],
            "actor_id": item[12], "metadata": json.loads(item[13] or "{}"),
        }
        for item in store.read(
            """
            SELECT step_id, sequence, role, content, event_id, memory_id, correlation_id,
                   outcome, status, occurred_at, session_id, scope, actor_id, metadata_json
            FROM episode_steps WHERE episode_id=? ORDER BY sequence
            """,
            (episode_id,),
        )
    ]
    actions = [
        {
            "correlation_id": item[0], "action_step_id": item[1],
            "outcome_step_id": item[2], "status": item[3],
        }
        for item in store.read(
            "SELECT correlation_id, action_step_id, outcome_step_id, status "
            "FROM episode_actions WHERE episode_id=? ORDER BY created_at",
            (episode_id,),
        )
    ]
    memories = [
        {"memory_id": item[0], "role": item[1]}
        for item in store.read(
            "SELECT memory_id, role FROM episode_memories WHERE episode_id=? ORDER BY memory_id",
            (episode_id,),
        )
    ]
    return {
        "episode_id": row[0], "title": row[1], "goal": row[2], "status": row[3],
        "session_id": row[4], "started_at": row[5], "ended_at": row[6],
        "duration_seconds": row[7], "tags": json.loads(row[8] or "[]"),
        "participants": json.loads(row[9] or "[]"), "metadata": json.loads(row[10] or "{}"),
        "steps": steps, "actions": actions, "memories": memories,
    }


def list_episodes(store: Store, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    if status is None:
        rows = store.read(
            "SELECT episode_id FROM episodes ORDER BY started_at DESC LIMIT ?", (int(limit),)
        )
    else:
        if status not in EPISODE_STATUSES:
            raise EpisodeError(f"unknown episode status '{status}'")
        rows = store.read(
            "SELECT episode_id FROM episodes WHERE status=? ORDER BY started_at DESC LIMIT ?",
            (status, int(limit)),
        )
    return [episode for row in rows if (episode := get_episode(store, row[0])) is not None]


def episode_for_session(store: Store, session_id: str) -> dict[str, Any] | None:
    row = store.read_one(
        "SELECT episode_id FROM episodes WHERE session_id=? AND status='open' "
        "ORDER BY started_at DESC LIMIT 1",
        (session_id,),
    )
    return get_episode(store, row[0]) if row else None


def _refresh_episode(conn, episode_id: str, now: str) -> None:
    rows = conn.execute(
        "SELECT status FROM episode_actions WHERE episode_id=?", (episode_id,)
    ).fetchall()
    if not rows:
        return
    statuses = [row[0] for row in rows]
    if "pending" in statuses:
        status = "open"
    elif "failed" in statuses:
        status = "failed"
    else:
        status = "completed"
    if status == "open":
        conn.execute("UPDATE episodes SET status='open', ended_at=NULL, duration_seconds=NULL WHERE episode_id=?",
                     (episode_id,))
        return
    started = conn.execute("SELECT started_at FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()[0]
    duration = max(0.0, (_parse(now) - _parse(started)).total_seconds())
    conn.execute(
        "UPDATE episodes SET status=?, ended_at=?, duration_seconds=? WHERE episode_id=?",
        (status, now, duration, episode_id),
    )


def _action_status(outcome: str | None) -> str:
    if outcome == "success":
        return "completed"
    if outcome == "failure":
        return "failed"
    if outcome == "unknown":
        return "unknown"
    raise EpisodeError("an outcome step needs success, failure or unknown")


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise EpisodeError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
