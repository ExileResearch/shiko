"""The background worker: lease one job, run it, record the outcome.

Handlers are idempotent because a crash can leave a lease to expire and the job
will be retried. The archive handler converges through its checkpoint; the index
handler advances a watermark that later retrieval treats as its committed read
boundary.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from .archive import Archive
from .queue import Job, complete, defer, fail, lease_next, pending_count
from .storage import Store

logger = logging.getLogger("shiko.worker")


class Worker:
    """A single-threaded job runner for one deployment."""

    def __init__(
        self,
        store: Store,
        archive: Archive,
        *,
        worker_id: str | None = None,
        lease_seconds: float = 15.0,
        backoff_seconds: float = 0.5,
        clock: Callable[[], datetime] | None = None,
        gate: Callable[[str], bool] | None = None,
        pipeline: Any | None = None,
    ) -> None:
        self.store = store
        self.archive = archive
        self.pipeline = pipeline
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.lease_seconds = lease_seconds
        self.backoff_seconds = backoff_seconds
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._gate = gate
        self._stopped = False
        self._handlers: dict[str, Callable[[Job], Any]] = {
            "archive.event": self._archive_event,
            "index.event": self._index_event,
            "ingest.session": self._ingest_session,
        }

    # -- control ------------------------------------------------------------

    def stop(self) -> None:
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    def run_once(self, *, kinds: tuple[str, ...] | None = None) -> dict[str, Any] | None:
        """Lease and execute at most one job. Returns a record of what happened."""
        if self._stopped:
            return None
        allowed = tuple(k for k in kinds if self._gate(k)) if kinds and self._gate else kinds
        if allowed is not None and not allowed:
            return None
        job = lease_next(
            self.store, self.worker_id, lease_seconds=self.lease_seconds,
            now=self._clock(), kinds=allowed,
        )
        if job is None:
            return None
        if self._gate and not self._gate(job.kind):
            defer(self.store, job.job_id, now=self._clock())
            return None
        handler = self._handlers.get(job.kind)
        try:
            if handler is None:
                raise RuntimeError(f"no handler for job kind '{job.kind}'")
            outcome = handler(job)
        except Exception as exc:  # a failed job is data, not a crash
            state = fail(
                self.store, job.job_id, f"{type(exc).__name__}: {exc}",
                worker_id=self.worker_id, backoff_seconds=self.backoff_seconds, now=self._clock(),
            )
            logger.warning("job %s (%s) failed: %s -> %s", job.job_id, job.kind, exc, state)
            return {"job_id": job.job_id, "kind": job.kind, "state": state, "error": str(exc)}
        if isinstance(outcome, dict) and outcome.get("keep_pending"):
            defer(self.store, job.job_id, now=self._clock())
            return {
                "job_id": job.job_id, "kind": job.kind, "event_id": job.event_id,
                "state": "pending", **{k: v for k, v in outcome.items() if k != "keep_pending"},
            }
        complete(self.store, job.job_id, worker_id=self.worker_id)
        record = {"job_id": job.job_id, "kind": job.kind, "event_id": job.event_id, "state": "done"}
        if isinstance(outcome, dict):
            record.update({k: v for k, v in outcome.items() if k != "keep_pending"})
        return record

    def run_until_idle(
        self, *, budget_seconds: float = 2.0, max_jobs: int = 100,
        kinds: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Drain ready work within a wall-clock budget. Never blocks on leases."""
        started = time.monotonic()
        processed: list[dict[str, Any]] = []
        failures = 0
        while len(processed) < max_jobs and time.monotonic() - started < budget_seconds:
            if self._stopped:
                break
            record = self.run_once(kinds=kinds)
            if record is None:
                break
            processed.append(record)
            if record["state"] == "failed":
                failures += 1
        return {
            "worker_id": self.worker_id,
            "processed": len(processed),
            "failures": failures,
            "pending": pending_count(self.store),
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
            "records": processed,
        }

    # -- handlers -----------------------------------------------------------

    def _archive_event(self, job: Job) -> None:
        if job.event_id:
            self.archive.sync_event(job.event_id)
        else:
            self.archive.recover()

    def _index_event(self, job: Job) -> None:
        """Advance the committed read boundary.

        Milestone 2 has no lexical or vector index yet; the watermark records
        that the event has been processed and is the boundary milestone 4
        retrieval will read up to.
        """
        with self.store.transaction() as conn:
            row = conn.execute("SELECT COALESCE(MAX(capture_seq), 0) FROM source_events").fetchone()
            target = int(row[0]) if row else 0
            if target > self.store.indexed_through():
                self.store.set_indexed_through(conn, target)

    def _ingest_session(self, job: Job) -> dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("worker has no ingestion pipeline")
        return self.pipeline.process(job)
