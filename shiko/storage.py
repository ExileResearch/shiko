"""Canonical SQLite storage with versioned migrations.

Schema 1 stored one row per accepted source event. Schema 2 adds the canonical
layer — memories, revisions, evidence links, classifications, entity references,
optional property groups, artifacts, a durable work queue and archive
checkpoints — plus a per-store capture sequence that gives every reader a single
committed boundary.

Rules that hold across versions:

* One transaction per logical write; an acknowledged capture is committed.
* Migrations run inside ``BEGIN IMMEDIATE``. SQLite DDL is transactional, so an
  interrupted migration leaves the previous schema intact and can be retried.
* A reused event ID is never overwritten: identical content is an idempotent
  duplicate, different content is a visible conflict.
* Optional properties use SQL ``NULL`` for "unknown". ``0.0`` is a measurement.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .envelope import SourceEnvelope
from .errors import StorageError
from .results import CaptureResult, Diagnostic, Severity

SCHEMA_VERSION = 3
DB_NAME = "shiko.db"

# Columns every event read shares, so get_event() and list_events() cannot drift.
EVENT_FIELDS = (
    "event_id", "capture_seq", "owner_id", "actor_id", "actor_kind", "actor_name",
    "session_id", "delegation_parent", "observed_at", "received_at", "visibility",
    "source_kind", "source_id", "subjects_json", "content", "content_type",
    "metadata_json", "content_sha256", "captured_at",
)
_EVENT_COLUMNS = ", ".join(EVENT_FIELDS)


def _migration_1(conn: sqlite3.Connection) -> None:
    """Evidence layer (schema 1)."""
    for statement in (
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            checksum   TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS source_events (
            event_id          TEXT PRIMARY KEY,
            owner_id          TEXT NOT NULL,
            actor_id          TEXT NOT NULL,
            actor_kind        TEXT NOT NULL,
            actor_name        TEXT NOT NULL DEFAULT '',
            session_id        TEXT NOT NULL,
            delegation_parent TEXT,
            observed_at       TEXT NOT NULL,
            received_at       TEXT NOT NULL,
            visibility        TEXT NOT NULL,
            source_kind       TEXT NOT NULL,
            source_id         TEXT NOT NULL DEFAULT '',
            source_json       TEXT NOT NULL,
            content           TEXT NOT NULL,
            content_type      TEXT NOT NULL,
            subjects_json     TEXT NOT NULL,
            metadata_json     TEXT NOT NULL,
            content_sha256    TEXT NOT NULL,
            captured_at       TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS source_events_session ON source_events (session_id, observed_at)",
        "CREATE INDEX IF NOT EXISTS source_events_actor ON source_events (actor_id, observed_at)",
    ):
        conn.execute(statement)


def _migration_2(conn: sqlite3.Connection) -> None:
    """Canonical records, revisions, provenance, artifacts and work queue."""
    for statement in (
        "ALTER TABLE source_events ADD COLUMN capture_seq INTEGER",
        "UPDATE source_events SET capture_seq = rowid WHERE capture_seq IS NULL",
        "CREATE UNIQUE INDEX IF NOT EXISTS source_events_capture_seq ON source_events (capture_seq)",
        """
        CREATE TABLE IF NOT EXISTS canonical_memories (
            memory_id        TEXT PRIMARY KEY,
            owner_id         TEXT NOT NULL,
            kind             TEXT NOT NULL,
            subject_kind     TEXT,
            subject_id       TEXT,
            status           TEXT NOT NULL,
            current_revision INTEGER NOT NULL DEFAULT 1,
            created_at       TEXT NOT NULL,
            learned_at       TEXT NOT NULL,
            metadata_json    TEXT NOT NULL DEFAULT '{}'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_revisions (
            memory_id       TEXT NOT NULL REFERENCES canonical_memories(memory_id) ON DELETE CASCADE,
            revision        INTEGER NOT NULL,
            content         TEXT NOT NULL,
            status          TEXT NOT NULL,
            event_time      TEXT,
            valid_from      TEXT,
            valid_to        TEXT,
            learned_at      TEXT NOT NULL,
            policy_version  TEXT NOT NULL,
            source_event_id TEXT,
            created_at      TEXT NOT NULL,
            PRIMARY KEY (memory_id, revision)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS evidence_links (
            link_id           TEXT PRIMARY KEY,
            memory_id         TEXT NOT NULL REFERENCES canonical_memories(memory_id) ON DELETE CASCADE,
            event_id          TEXT NOT NULL,
            role              TEXT NOT NULL,
            span_start        INTEGER,
            span_end          INTEGER,
            independence_root TEXT NOT NULL,
            created_at        TEXT NOT NULL,
            UNIQUE (memory_id, event_id, role)
        )
        """,
        "CREATE INDEX IF NOT EXISTS evidence_links_event ON evidence_links (event_id)",
        """
        CREATE TABLE IF NOT EXISTS classifications (
            classification_id TEXT PRIMARY KEY,
            memory_id         TEXT NOT NULL REFERENCES canonical_memories(memory_id) ON DELETE CASCADE,
            axis              TEXT NOT NULL,
            value             TEXT NOT NULL,
            confidence        REAL,
            method            TEXT NOT NULL,
            model_id          TEXT,
            created_at        TEXT NOT NULL,
            UNIQUE (memory_id, axis, value, method)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS entity_refs (
            entity_ref_id TEXT PRIMARY KEY,
            entity_id     TEXT NOT NULL,
            entity_kind   TEXT NOT NULL,
            display_name  TEXT NOT NULL DEFAULT '',
            event_id      TEXT,
            memory_id     TEXT,
            created_at    TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_affect (
            memory_id  TEXT PRIMARY KEY REFERENCES canonical_memories(memory_id) ON DELETE CASCADE,
            valence    REAL,
            arousal    REAL,
            dominance  REAL,
            confidence REAL,
            source     TEXT NOT NULL,
            model_id   TEXT,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_temporal (
            memory_id        TEXT PRIMARY KEY REFERENCES canonical_memories(memory_id) ON DELETE CASCADE,
            event_time       TEXT,
            valid_from       TEXT,
            valid_to         TEXT,
            time_granularity TEXT NOT NULL DEFAULT 'unknown',
            updated_at       TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            event_id    TEXT NOT NULL,
            uri         TEXT NOT NULL,
            media_type  TEXT NOT NULL,
            size_bytes  INTEGER NOT NULL,
            sha256      TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS artifacts_event ON artifacts (event_id)",
        """
        CREATE TABLE IF NOT EXISTS pending_work (
            job_id        TEXT PRIMARY KEY,
            kind          TEXT NOT NULL,
            event_id      TEXT,
            payload_json  TEXT NOT NULL DEFAULT '{}',
            state         TEXT NOT NULL,
            attempts      INTEGER NOT NULL DEFAULT 0,
            max_attempts  INTEGER NOT NULL DEFAULT 3,
            not_before    TEXT NOT NULL,
            lease_owner   TEXT,
            lease_expires TEXT,
            last_error    TEXT,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            UNIQUE (kind, event_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS pending_work_ready ON pending_work (state, not_before)",
        """
        CREATE TABLE IF NOT EXISTS archive_streams (
            stream           TEXT PRIMARY KEY,
            path             TEXT NOT NULL,
            byte_length      INTEGER NOT NULL DEFAULT 0,
            last_capture_seq INTEGER NOT NULL DEFAULT 0,
            updated_at       TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS index_state (
            name            TEXT PRIMARY KEY,
            indexed_through INTEGER NOT NULL DEFAULT 0,
            updated_at      TEXT NOT NULL
        )
        """,
        "INSERT OR IGNORE INTO index_state (name, indexed_through, updated_at) "
        "VALUES ('events', 0, '1970-01-01T00:00:00Z')",
    ):
        conn.execute(statement)


def _migration_3(conn: sqlite3.Connection) -> None:
    """Ingestion state, pipeline counters and bounded lifecycle history."""
    for statement in (
        """
        CREATE TABLE IF NOT EXISTS ingestion_state (
            session_id        TEXT PRIMARY KEY,
            watermark_seq     INTEGER NOT NULL DEFAULT 0,
            last_enqueued_seq INTEGER NOT NULL DEFAULT 0,
            last_batch_at     TEXT,
            updated_at        TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS pipeline_state (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "INSERT OR IGNORE INTO pipeline_state (key, value, updated_at) VALUES ('paused', '0', '1970-01-01T00:00:00Z')",
        "INSERT OR IGNORE INTO pipeline_state (key, value, updated_at) VALUES ('coalesced', '0', '1970-01-01T00:00:00Z')",
        "INSERT OR IGNORE INTO pipeline_state (key, value, updated_at) VALUES ('processed_batches', '0', '1970-01-01T00:00:00Z')",
        """
        CREATE TABLE IF NOT EXISTS memory_events (
            seq            INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id      TEXT NOT NULL REFERENCES canonical_memories(memory_id) ON DELETE CASCADE,
            kind           TEXT NOT NULL,
            at             TEXT NOT NULL,
            policy_version TEXT,
            model_id       TEXT,
            detail_json    TEXT NOT NULL DEFAULT '{}'
        )
        """,
        "CREATE INDEX IF NOT EXISTS memory_events_memory ON memory_events (memory_id, at)",
        "CREATE INDEX IF NOT EXISTS memory_events_at ON memory_events (at)",
        """
        CREATE TABLE IF NOT EXISTS memory_event_rollups (
            day       TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            kind      TEXT NOT NULL,
            count     INTEGER NOT NULL,
            PRIMARY KEY (day, memory_id, kind)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_dynamics (
            memory_id          TEXT PRIMARY KEY REFERENCES canonical_memories(memory_id) ON DELETE CASCADE,
            activation         REAL,
            baseline_activation REAL,
            storage_strength   REAL,
            retrieval_strength REAL,
            reinforcement      REAL,
            decay_rate         REAL,
            last_updated       TEXT,
            updated_at         TEXT NOT NULL
        )
        """,
        """
        INSERT OR IGNORE INTO ingestion_state (session_id, watermark_seq, last_enqueued_seq, updated_at)
        SELECT session_id, 0, 0, '1970-01-01T00:00:00Z' FROM source_events GROUP BY session_id
        """,
    ):
        conn.execute(statement)


MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = (_migration_1, _migration_2, _migration_3)


class Store:
    """One deployment's canonical database."""
    def __init__(
        self,
        path: str | Path,
        owner_id: str,
        *,
        migrations: tuple[Callable[[sqlite3.Connection], None], ...] = MIGRATIONS,
    ) -> None:
        self.path = Path(path)
        self.owner_id = owner_id
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self.schema_version = 0
        self._migrations = migrations

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(
                str(self.path), timeout=10.0, isolation_level=None, check_same_thread=False
            )
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open store: {exc}", detail={"path": str(self.path)}) from exc
        self._conn = conn
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=10000")
            self.schema_version = self._migrate()
        except sqlite3.Error as exc:
            self.close()
            raise StorageError(f"cannot prepare store: {exc}", detail={"path": str(self.path)}) from exc
        return self.schema_version

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None

    @property
    def is_open(self) -> bool:
        return self._conn is not None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StorageError("store is closed", detail={"path": str(self.path)})
        return self._conn

    # -- transactions -------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One write transaction. Any exception rolls the whole unit back."""
        conn = self._connection()
        with self._lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise StorageError(f"cannot begin transaction: {exc}") from exc
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise

    def read(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple]:
        with self._lock:
            return self._connection().execute(query, params).fetchall()

    def read_one(self, query: str, params: tuple[Any, ...] = ()) -> tuple | None:
        with self._lock:
            return self._connection().execute(query, params).fetchone()

    # -- schema -------------------------------------------------------------

    def _migrate(self) -> int:
        conn = self._connection()
        current = self._read_schema_version(conn)
        if current > SCHEMA_VERSION:
            raise StorageError(
                "store schema is newer than this build; refusing to downgrade",
                detail={"path": str(self.path), "store_schema": current, "supported": SCHEMA_VERSION},
            )
        for target in range(current + 1, SCHEMA_VERSION + 1):
            migration = self._migrations[target - 1]
            try:
                conn.execute("BEGIN IMMEDIATE")
                migration(conn)
                self._write_schema_version(conn, target)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_migrations (version, applied_at, checksum) VALUES (?, ?, ?)",
                    (target, _now(), f"{target}:{migration.__name__}"),
                )
                conn.execute("COMMIT")
            except BaseException as exc:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                if isinstance(exc, StorageError):
                    raise
                raise StorageError(
                    f"migration to schema {target} failed: {exc}",
                    detail={"path": str(self.path), "from_schema": target - 1},
                ) from exc
        return SCHEMA_VERSION

    @staticmethod
    def _read_schema_version(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if row is None:
            return 0
        stored = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if stored is None:
            return 0
        try:
            return int(stored[0])
        except (TypeError, ValueError) as exc:
            raise StorageError("store schema version is not an integer") from exc

    @staticmethod
    def _write_schema_version(conn: sqlite3.Connection, version: int) -> None:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(version),),
        )

    # -- events -------------------------------------------------------------

    def next_capture_seq(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COALESCE(MAX(capture_seq), 0) FROM source_events").fetchone()
        return int(row[0]) + 1

    def insert_event(self, conn: sqlite3.Connection, envelope: SourceEnvelope, capture_seq: int) -> None:
        conn.execute(
            """
            INSERT INTO source_events (
                event_id, capture_seq, owner_id, actor_id, actor_kind, actor_name, session_id,
                delegation_parent, observed_at, received_at, visibility,
                source_kind, source_id, source_json, content, content_type,
                subjects_json, metadata_json, content_sha256, captured_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                envelope.event_id, capture_seq, self.owner_id, envelope.actor.id,
                envelope.actor.kind.value, envelope.actor.display_name, envelope.session_id,
                envelope.delegation_parent, _iso(envelope.observed_at), _iso(envelope.received_at),
                envelope.visibility.value, envelope.source.kind, envelope.source.id,
                _json(envelope.source.to_dict()), envelope.content, envelope.content_type,
                _json([s.to_dict() for s in envelope.subjects]), _json(dict(envelope.metadata)),
                envelope.content_sha256(), _now(),
            ),
        )

    def capture(
        self,
        envelope: SourceEnvelope,
        *,
        job_kinds: tuple[str, ...] = (),
        on_insert: Callable[[sqlite3.Connection, SourceEnvelope, int], None] | None = None,
    ) -> CaptureResult:
        """Insert one event (and its follow-up jobs) in a single transaction."""
        sha = envelope.content_sha256()
        result: CaptureResult | None = None
        seq = 0
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT content_sha256, capture_seq FROM source_events WHERE event_id=?",
                (envelope.event_id,),
            ).fetchone()
            if row is not None:
                seq = int(row[1] or 0)
                result = self._duplicate_result(envelope, row[0], sha, seq)
            else:
                seq = self.next_capture_seq(conn)
                self.insert_event(conn, envelope, seq)
                if job_kinds:
                    from .queue import enqueue

                    for kind in job_kinds:
                        enqueue(conn, kind, envelope.event_id)
                if on_insert is not None:
                    on_insert(conn, envelope, seq)
        if result is not None:
            return result
        diagnostics: tuple[Diagnostic, ...] = ()
        if envelope.clock_skew_seconds > 0:
            diagnostics = (
                Diagnostic(
                    "event.clock_skew",
                    "source observed the event after Shiko received it",
                    Severity.INFO,
                    {"event_id": envelope.event_id, "skew_seconds": envelope.clock_skew_seconds},
                ),
            )
        return CaptureResult.success(envelope.event_id, capture_seq=seq, diagnostics=diagnostics)

    def _duplicate_result(
        self, envelope: SourceEnvelope, stored_sha: str, sha: str, capture_seq: int
    ) -> CaptureResult:
        if stored_sha != sha:
            return CaptureResult.success(
                envelope.event_id,
                capture_seq=capture_seq,
                duplicate=True,
                conflict=True,
                diagnostics=(
                    Diagnostic(
                        "event.id_reuse",
                        "event_id was reused with different content; the original was preserved",
                        Severity.ERROR,
                        {
                            "event_id": envelope.event_id,
                            "stored_sha256": stored_sha,
                            "incoming_sha256": sha,
                        },
                    ),
                ),
            )
        return CaptureResult.success(
            envelope.event_id,
            capture_seq=capture_seq,
            duplicate=True,
            diagnostics=(
                Diagnostic(
                    "event.duplicate",
                    "event_id already captured with identical content; nothing written",
                    Severity.INFO,
                    {"event_id": envelope.event_id},
                ),
            ),
        )

    def event_sha(self, event_id: str) -> str | None:
        row = self.read_one("SELECT content_sha256 FROM source_events WHERE event_id=?", (event_id,))
        return str(row[0]) if row else None

    # -- reads --------------------------------------------------------------

    def count_events(self) -> int:
        return int(self.read_one("SELECT COUNT(*) FROM source_events")[0])

    def count_events_for_session(self, session_id: str) -> int:
        row = self.read_one("SELECT COUNT(*) FROM source_events WHERE session_id=?", (session_id,))
        return int(row[0]) if row else 0

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        row = self.read_one(f"SELECT {_EVENT_COLUMNS} FROM source_events WHERE event_id=?", (event_id,))
        return dict(zip(EVENT_FIELDS, row)) if row is not None else None

    def list_events(self, *, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if session_id is None:
            rows = self.read(
                f"SELECT {_EVENT_COLUMNS} FROM source_events ORDER BY capture_seq LIMIT ?", (int(limit),)
            )
        else:
            rows = self.read(
                f"SELECT {_EVENT_COLUMNS} FROM source_events WHERE session_id=? "
                "ORDER BY capture_seq LIMIT ?",
                (session_id, int(limit)),
            )
        return [dict(zip(EVENT_FIELDS, row)) for row in rows]

    def events_after(self, capture_seq: int, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.read(
            f"SELECT {_EVENT_COLUMNS} FROM source_events WHERE capture_seq > ? "
            "ORDER BY capture_seq LIMIT ?",
            (int(capture_seq), int(limit)),
        )
        return [dict(zip(EVENT_FIELDS, row)) for row in rows]

    def recent_events_for_session(
        self, session_id: str, *, exclude_event_id: str | None = None, limit: int = 6
    ) -> list[dict[str, Any]]:
        rows = self.read(
            f"SELECT {_EVENT_COLUMNS} FROM source_events WHERE session_id=? AND event_id IS NOT ? "
            "ORDER BY capture_seq DESC LIMIT ?",
            (session_id, exclude_event_id, int(limit)),
        )
        return [dict(zip(EVENT_FIELDS, row)) for row in reversed(rows)]

    def integrity_check(self) -> str:
        return str(self.read_one("PRAGMA integrity_check")[0])

    # -- archive streams ----------------------------------------------------

    def stream_state(self, stream: str) -> dict[str, Any] | None:
        row = self.read_one(
            "SELECT stream, path, byte_length, last_capture_seq FROM archive_streams WHERE stream=?",
            (stream,),
        )
        if row is None:
            return None
        return {
            "stream": row[0], "path": row[1], "byte_length": int(row[2]), "last_capture_seq": int(row[3])
        }

    def update_stream(
        self, conn: sqlite3.Connection, stream: str, *, path: str, byte_length: int, last_capture_seq: int
    ) -> None:
        conn.execute(
            """
            INSERT INTO archive_streams (stream, path, byte_length, last_capture_seq, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(stream) DO UPDATE SET
                path=excluded.path, byte_length=excluded.byte_length,
                last_capture_seq=excluded.last_capture_seq, updated_at=excluded.updated_at
            """,
            (stream, path, int(byte_length), int(last_capture_seq), _now()),
        )

    # -- index boundary -----------------------------------------------------

    def indexed_through(self) -> int:
        row = self.read_one("SELECT indexed_through FROM index_state WHERE name='events'")
        return int(row[0]) if row else 0

    def set_indexed_through(self, conn: sqlite3.Connection, value: int) -> None:
        conn.execute(
            "INSERT INTO index_state (name, indexed_through, updated_at) VALUES ('events', ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET indexed_through=excluded.indexed_through, updated_at=excluded.updated_at",
            (int(value), _now()),
        )


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _now() -> str:
    return _iso(datetime.now(timezone.utc))


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"
