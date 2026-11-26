"""Durable work queue with leases, bounded retries and visible failures.

Jobs are enqueued in the same transaction as the event they follow, so an
accepted capture cannot lose its follow-up work. A worker leases one job with a
short deadline; a crash leaves the lease to expire rather than losing the job.
Retries are bounded and the final failure stays in the table for inspection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from .storage import Store, new_id

JOB_KINDS = frozenset({"archive.event", "index.event", "ingest.session"})


@dataclass(frozen=True)
class Job:
    job_id: str
    kind: str
    event_id: str | None
    payload: dict[str, Any]
    state: str
    attempts: int
    max_attempts: int
    not_before: str
    lease_owner: str | None
    lease_expires: str | None
    last_error: str | None
    created_at: str


def enqueue(
    conn,
    kind: str,
    event_id: str | None,
    *,
    payload: dict[str, Any] | None = None,
    max_attempts: int = 3,
    not_before: datetime | None = None,
) -> str:
    """Insert a job inside the caller's transaction. Idempotent per (kind, event)."""
    if kind not in JOB_KINDS:
        raise ValueError(f"unknown job kind '{kind}'")
    now = _now()
    job_id = new_id("job")
    conn.execute(
        """
        INSERT INTO pending_work
            (job_id, kind, event_id, payload_json, state, attempts, max_attempts, not_before,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)
        ON CONFLICT(kind, event_id) DO NOTHING
        """,
        (
            job_id, kind, event_id, json.dumps(payload or {}, sort_keys=True),
            int(max_attempts), _iso(not_before) if not_before else now, now, now,
        ),
    )
    row = conn.execute(
        "SELECT job_id FROM pending_work WHERE kind=? AND event_id IS ?", (kind, event_id)
    ).fetchone()
    return str(row[0]) if row else job_id


def lease_next(
    store: Store,
    worker_id: str,
    *,
    lease_seconds: float = 15.0,
    now: datetime | None = None,
    kinds: Iterable[str] | None = None,
) -> Job | None:
    """Lease one ready job, or an expired lease abandoned by a dead worker."""
    moment = now or datetime.now(timezone.utc)
    now_s = _iso(moment)
    expires_s = _iso(moment + timedelta(seconds=lease_seconds))
    allowed = tuple(kinds) if kinds else tuple(JOB_KINDS)
    placeholders = ",".join("?" for _ in allowed)
    with store.transaction() as conn:
        row = conn.execute(
            f"""
            UPDATE pending_work
            SET state='leased', lease_owner=?, lease_expires=?, attempts=attempts + 1,
                updated_at=?
            WHERE job_id = (
                SELECT job_id FROM pending_work
                WHERE kind IN ({placeholders})
                  AND not_before <= ?
                  AND (state='pending' OR (state='leased' AND lease_expires < ?))
                ORDER BY created_at, rowid
                LIMIT 1
            )
            RETURNING job_id, kind, event_id, payload_json, state, attempts, max_attempts,
                      not_before, lease_owner, lease_expires, last_error, created_at
            """,
            (worker_id, expires_s, now_s, *allowed, now_s, now_s),
        ).fetchone()
    return _job(row) if row else None


def complete(store: Store, job_id: str, *, worker_id: str | None = None) -> bool:
    now = _now()
    with store.transaction() as conn:
        cursor = conn.execute(
            """
            UPDATE pending_work
            SET state='done', lease_owner=NULL, lease_expires=NULL, updated_at=?
            WHERE job_id=? AND state='leased' AND (? IS NULL OR lease_owner=?)
            """,
            (now, job_id, worker_id, worker_id),
        )
        return cursor.rowcount > 0


def fail(
    store: Store,
    job_id: str,
    error: str,
    *,
    worker_id: str | None = None,
    backoff_seconds: float = 0.5,
    now: datetime | None = None,
) -> str:
    """Record a failed attempt. Returns the resulting state: ``pending`` or ``failed``."""
    moment = now or datetime.now(timezone.utc)
    with store.transaction() as conn:
        row = conn.execute(
            "SELECT attempts, max_attempts FROM pending_work WHERE job_id=?", (job_id,)
        ).fetchone()
        if row is None:
            return "missing"
        attempts, max_attempts = int(row[0]), int(row[1])
        if attempts >= max_attempts:
            conn.execute(
                """
                UPDATE pending_work
                SET state='failed', lease_owner=NULL, lease_expires=NULL,
                    last_error=?, updated_at=?
                WHERE job_id=?
                """,
                (str(error)[:2000], _iso(moment), job_id),
            )
            return "failed"
        retry_at = moment + timedelta(seconds=backoff_seconds * max(1, attempts))
        conn.execute(
            """
            UPDATE pending_work
            SET state='pending', lease_owner=NULL, lease_expires=NULL, last_error=?,
                not_before=?, updated_at=?
            WHERE job_id=? AND (? IS NULL OR lease_owner=?)
            """,
            (str(error)[:2000], _iso(retry_at), _iso(moment), job_id, worker_id, worker_id),
        )
        return "pending"


def defer(
    store: Store, job_id: str, *, seconds: float = 0.0, now: datetime | None = None,
) -> bool:
    """Return a job to the queue without consuming an attempt.

    Used when a worker declines to run it (pipeline paused) or when a handler
    finished one bounded batch and more work is already waiting.
    """
    moment = now or datetime.now(timezone.utc)
    retry_at = moment + timedelta(seconds=max(0.0, seconds))
    with store.transaction() as conn:
        cursor = conn.execute(
            """
            UPDATE pending_work
            SET state='pending', lease_owner=NULL, lease_expires=NULL,
                attempts=0, not_before=?, updated_at=?
            WHERE job_id=?
            """,
            (_iso(retry_at), _iso(moment), job_id),
        )
        return cursor.rowcount > 0


def complete_by_kind(store: Store, kind: str, event_id: str) -> bool:
    """Mark an inline-completed job done, if it is still pending."""
    with store.transaction() as conn:
        cursor = conn.execute(
            """
            UPDATE pending_work SET state='done', lease_owner=NULL, lease_expires=NULL, updated_at=?
            WHERE kind=? AND event_id=? AND state='pending'
            """,
            (_now(), kind, event_id),
        )
        return cursor.rowcount > 0


def job_for(store: Store, kind: str, event_id: str | None) -> Job | None:
    row = store.read_one(
        """
        SELECT job_id, kind, event_id, payload_json, state, attempts, max_attempts,
               not_before, lease_owner, lease_expires, last_error, created_at
        FROM pending_work WHERE kind=? AND event_id IS ?
        """,
        (kind, event_id),
    )
    return _job(row) if row else None


def pending_count(store: Store) -> int:
    row = store.read_one("SELECT COUNT(*) FROM pending_work WHERE state IN ('pending','leased')")
    return int(row[0]) if row else 0


def failed_jobs(store: Store, *, limit: int = 50) -> list[dict[str, Any]]:
    return [
        dict(zip(
            ("job_id", "kind", "event_id", "attempts", "max_attempts", "last_error", "updated_at"),
            row,
        ))
        for row in store.read(
            "SELECT job_id, kind, event_id, attempts, max_attempts, last_error, updated_at "
            "FROM pending_work WHERE state='failed' ORDER BY updated_at LIMIT ?",
            (int(limit),),
        )
    ]


def is_pending(store: Store, event_id: str) -> bool:
    row = store.read_one(
        "SELECT 1 FROM pending_work WHERE event_id=? AND state IN ('pending','leased') LIMIT 1",
        (event_id,),
    )
    return row is not None


def _job(row: tuple) -> Job:
    return Job(
        job_id=row[0], kind=row[1], event_id=row[2], payload=json.loads(row[3] or "{}"),
        state=row[4], attempts=int(row[5]), max_attempts=int(row[6]), not_before=row[7],
        lease_owner=row[8], lease_expires=row[9], last_error=row[10], created_at=row[11],
    )


def _now() -> str:
    return _iso(datetime.now(timezone.utc))


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
