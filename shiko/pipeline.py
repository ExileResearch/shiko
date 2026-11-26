"""Continuous ingestion: coalesce a bursty conversation into bounded work.

A turn is cheap to capture and expensive to interpret. The pipeline therefore
keeps capture on the hot path (one insert plus a job upsert in the same
transaction) and puts interpretation behind a debounced, per-session batch.

Rules:

* One in-flight batch per session. The job row is keyed by session, so a second
  worker cannot lease it concurrently.
* A burst extends the debounce window up to a maximum delay; it never creates
  one job per turn.
* Work is never lost. The batch advances an ingestion watermark; a crash leaves
  the lease to expire and the retry resumes from the watermark.
* Pause is an operator switch, not a backpressure guess: capture and recall keep
  working while extraction waits.
* Every accepted event is processed exactly once, in capture order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .queue import JOB_KINDS, enqueue
from .storage import Store, _now

INGEST_JOB = "ingest.session"


@dataclass(frozen=True)
class Batch:
    session_id: str
    from_seq: int
    through_seq: int
    event_count: int
    job_id: str


class IngestionPipeline:
    """Debounce, coalesce, throttle and track one deployment's ingestion."""

    def __init__(
        self,
        store: Store,
        *,
        debounce_ms: int = 200,
        max_delay_ms: int = 1500,
        coalesce_at: int = 50,
        batch_size: int = 20,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.debounce_ms = max(0, int(debounce_ms))
        self.max_delay_ms = max(self.debounce_ms, int(max_delay_ms))
        self.coalesce_at = max(1, int(coalesce_at))
        self.batch_size = max(1, int(batch_size))
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    # -- capture side -------------------------------------------------------

    def note_capture(self, conn, envelope, seq: int) -> None:
        """Upsert session state and the coalesced job inside the capture transaction."""
        now = self.clock()
        now_s = _iso(now)
        session = envelope.session_id
        conn.execute(
            """
            INSERT INTO ingestion_state (session_id, watermark_seq, last_enqueued_seq, updated_at)
            VALUES (?, 0, 0, ?)
            ON CONFLICT(session_id) DO NOTHING
            """,
            (session, now_s),
        )
        conn.execute(
            "UPDATE ingestion_state SET last_enqueued_seq=MAX(last_enqueued_seq, ?), updated_at=? "
            "WHERE session_id=?",
            (int(seq), now_s, session),
        )
        row = conn.execute(
            "SELECT job_id, state, created_at FROM pending_work WHERE kind=? AND event_id=?",
            (INGEST_JOB, session),
        ).fetchone()
        if row is None:
            enqueue(
                conn, INGEST_JOB, session,
                payload={"session_id": session},
                not_before=now + timedelta(milliseconds=self.debounce_ms),
            )
            return
        job_id, state, created_at = row
        if state == "pending":
            anchor = _parse(created_at) + timedelta(milliseconds=self.max_delay_ms)
            target = min(now + timedelta(milliseconds=self.debounce_ms), anchor)
            conn.execute(
                "UPDATE pending_work SET not_before=?, updated_at=? WHERE job_id=?",
                (_iso(target), now_s, job_id),
            )
            conn.execute(
                "UPDATE pipeline_state SET value=CAST(CAST(value AS INTEGER) + 1 AS TEXT), updated_at=? "
                "WHERE key='coalesced'",
                (now_s,),
            )
        elif state in {"done", "failed"}:
            # A finished batch followed by new traffic re-arms the same row:
            # the unique (kind, event_id) index allows exactly one session job.
            conn.execute(
                """
                UPDATE pending_work
                SET state='pending', attempts=0, not_before=?, lease_owner=NULL,
                    lease_expires=NULL, last_error=NULL, updated_at=?
                WHERE job_id=?
                """,
                (now_s, now_s, job_id),
            )
        # A leased job will pick this event up when it finishes; finish_batch()
        # re-arms the row if the watermark is still behind.

    # -- recovery -----------------------------------------------------------

    def ensure_all(self) -> int:
        """Create a job for any session whose events are not fully processed."""
        now = self.clock()
        created = 0
        with self.store.transaction() as conn:
            rows = conn.execute(
                "SELECT session_id, last_enqueued_seq, watermark_seq FROM ingestion_state "
                "WHERE last_enqueued_seq > watermark_seq"
            ).fetchall()
            for session, last_enqueued, watermark in rows:
                exists = conn.execute(
                    "SELECT 1 FROM pending_work WHERE kind=? AND event_id=? "
                    "AND state IN ('pending','leased')",
                    (INGEST_JOB, session),
                ).fetchone()
                if exists:
                    continue
                conn.execute(
                    """
                    INSERT INTO pending_work
                        (job_id, kind, event_id, payload_json, state, attempts, max_attempts,
                         not_before, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'pending', 0, 5, ?, ?, ?)
                    ON CONFLICT(kind, event_id) DO UPDATE SET
                        state='pending', attempts=0, not_before=excluded.not_before,
                        lease_owner=NULL, lease_expires=NULL, last_error=NULL,
                        updated_at=excluded.updated_at
                    """,
                    (
                        f"job_{session[:32]}", INGEST_JOB, session,
                        json.dumps({"session_id": session, "recovered": True}),
                        _iso(now), _iso(now), _iso(now),
                    ),
                )
                created += 1
        return created

    # -- worker side --------------------------------------------------------

    def can_run(self, kind: str) -> bool:
        return not (kind == INGEST_JOB and self.paused)

    @property
    def paused(self) -> bool:
        row = self.store.read_one("SELECT value FROM pipeline_state WHERE key='paused'")
        return bool(row and row[0] == "1")

    def set_paused(self, paused: bool) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO pipeline_state (key, value, updated_at) VALUES ('paused', ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                ("1" if paused else "0", _now()),
            )

    def process(self, job) -> dict[str, Any]:
        """Advance one bounded batch and return the worker's outcome."""
        session = str(job.event_id or (job.payload or {}).get("session_id") or "")
        if not session:
            raise ValueError("ingest job has no session")
        now = self.clock()
        state = self.store.read_one(
            "SELECT watermark_seq, last_enqueued_seq FROM ingestion_state WHERE session_id=?",
            (session,),
        )
        watermark, last_enqueued = (int(state[0]), int(state[1])) if state else (0, 0)
        events = self.store.read(
            "SELECT event_id, capture_seq FROM source_events "
            "WHERE session_id=? AND capture_seq > ? ORDER BY capture_seq LIMIT ?",
            (session, watermark, self.batch_size),
        )
        new_watermark = int(events[-1][1]) if events else last_enqueued
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE ingestion_state SET watermark_seq=?, last_batch_at=?, updated_at=? "
                "WHERE session_id=?",
                (new_watermark, _iso(now), _iso(now), session),
            )
            conn.execute(
                "UPDATE pipeline_state SET value=CAST(CAST(value AS INTEGER) + 1 AS TEXT), updated_at=? "
                "WHERE key='processed_batches'",
                (_iso(now),),
            )
        keep_pending = last_enqueued > new_watermark
        return {
            "session_id": session,
            "processed": len(events),
            "from_seq": watermark,
            "through_seq": new_watermark,
            "keep_pending": keep_pending,
        }

    # -- observability ------------------------------------------------------

    def status(self) -> dict[str, Any]:
        now = self.clock()
        paused = self.paused
        backlog_row = self.store.read_one(
            "SELECT COALESCE(SUM(MAX(last_enqueued_seq - watermark_seq, 0)), 0) FROM ingestion_state"
        )
        pending_events = int(backlog_row[0]) if backlog_row else 0
        oldest_row = self.store.read_one(
            "SELECT MIN(e.captured_at) FROM source_events e "
            "JOIN ingestion_state s ON s.session_id = e.session_id "
            "WHERE e.capture_seq > s.watermark_seq"
        )
        oldest = oldest_row[0] if oldest_row else None
        oldest_age = None
        if oldest:
            oldest_age = max(0.0, (now - _parse(oldest)).total_seconds())
        jobs_row = self.store.read_one(
            "SELECT COUNT(*) FROM pending_work WHERE kind=? AND state IN ('pending','leased')",
            (INGEST_JOB,),
        )
        counters = {
            row[0]: int(row[1])
            for row in self.store.read(
                "SELECT key, value FROM pipeline_state WHERE key IN ('coalesced','processed_batches')"
            )
        }
        mode = "pause" if paused else ("coalesce" if pending_events >= self.coalesce_at else "normal")
        return {
            "mode": mode,
            "paused": paused,
            "pending_events": pending_events,
            "pending_jobs": int(jobs_row[0]) if jobs_row else 0,
            "oldest_pending_seconds": oldest_age,
            "coalesced": counters.get("coalesced", 0),
            "processed_batches": counters.get("processed_batches", 0),
            "batch_size": self.batch_size,
        }


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# The queue knows the kind; importing the pipeline must not make it required.
JOB_KINDS = frozenset(set(JOB_KINDS) | {INGEST_JOB})
