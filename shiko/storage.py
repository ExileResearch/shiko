"""Canonical SQLite storage.

Milestone 1 stores the evidence layer only: one row per accepted source event,
keyed by ``event_id`` within the owning deployment. Later milestones extend this
schema through :data:`MIGRATIONS`; the migration runner and version check exist
now so a store written by a newer build is refused instead of corrupted.

Durability rules already in force:

* WAL journal, ``synchronous=NORMAL`` and a busy timeout, so a reader and the
  capture thread can share the file.
* One transaction per event; absence of an error means the row is committed.
* A reused event ID is never overwritten. Same content is an idempotent
  duplicate; different content is a visible conflict.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .envelope import SourceEnvelope
from .errors import StorageError
from .results import CaptureResult, Diagnostic, Severity

SCHEMA_VERSION = 1
DB_NAME = "shiko.db"

# Columns every read shape shares, so get_event() and list_events() cannot drift.
EVENT_FIELDS = (
    "event_id", "owner_id", "actor_id", "actor_kind", "actor_name", "session_id",
    "delegation_parent", "observed_at", "received_at", "visibility", "source_kind",
    "source_id", "subjects_json", "content", "content_type", "metadata_json",
    "content_sha256", "captured_at",
)
_EVENT_COLUMNS = ", ".join(EVENT_FIELDS)


_MIGRATION_1_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
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
    """
    CREATE INDEX IF NOT EXISTS source_events_session
        ON source_events (session_id, observed_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS source_events_actor
        ON source_events (actor_id, observed_at)
    """,
)


def _migration_1(conn: sqlite3.Connection) -> None:
    # executescript() would implicitly COMMIT the migration transaction, so each
    # statement runs through execute() inside the caller's BEGIN IMMEDIATE.
    for statement in _MIGRATION_1_STATEMENTS:
        conn.execute(statement)


MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = (_migration_1,)


class Store:
    """One deployment's canonical database."""

    def __init__(self, path: str | Path, owner_id: str) -> None:
        self.path = Path(path)
        self.owner_id = owner_id
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self.schema_version = 0

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(
                str(self.path), timeout=5.0, isolation_level=None, check_same_thread=False
            )
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open store: {exc}", detail={"path": str(self.path)}) from exc
        self._conn = conn
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
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
            migration = MIGRATIONS[target - 1]
            try:
                conn.execute("BEGIN IMMEDIATE")
                migration(conn)
                self._write_schema_version(conn, target)
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
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

    # -- writes -------------------------------------------------------------

    def capture(self, envelope: SourceEnvelope) -> CaptureResult:
        """Insert one event. Idempotent for identical content; loud for conflicts."""
        conn = self._connection()
        sha = envelope.content_sha256()
        with self._lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT owner_id, content_sha256 FROM source_events WHERE event_id=?",
                    (envelope.event_id,),
                ).fetchone()
                if row is not None:
                    conn.execute("COMMIT")
                    return self._duplicate_result(envelope, row[1], sha)
                conn.execute(
                    """
                    INSERT INTO source_events (
                        event_id, owner_id, actor_id, actor_kind, actor_name, session_id,
                        delegation_parent, observed_at, received_at, visibility,
                        source_kind, source_id, source_json, content, content_type,
                        subjects_json, metadata_json, content_sha256, captured_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        envelope.event_id,
                        self.owner_id,
                        envelope.actor.id,
                        envelope.actor.kind.value,
                        envelope.actor.display_name,
                        envelope.session_id,
                        envelope.delegation_parent,
                        _iso(envelope.observed_at),
                        _iso(envelope.received_at),
                        envelope.visibility.value,
                        envelope.source.kind,
                        envelope.source.id,
                        _json(envelope.source.to_dict()),
                        envelope.content,
                        envelope.content_type,
                        _json([s.to_dict() for s in envelope.subjects]),
                        _json(dict(envelope.metadata)),
                        sha,
                        _iso(datetime.now(timezone.utc)),
                    ),
                )
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                return CaptureResult.failure(
                    envelope.event_id,
                    (Diagnostic("event.write_failed", str(exc), Severity.ERROR, {"event_id": envelope.event_id}),),
                )
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
        return CaptureResult.success(envelope.event_id, diagnostics=diagnostics)

    def _duplicate_result(self, envelope: SourceEnvelope, stored_sha: str, sha: str) -> CaptureResult:
        if stored_sha != sha:
            return CaptureResult.success(
                envelope.event_id,
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

    # -- reads --------------------------------------------------------------

    def count_events(self) -> int:
        conn = self._connection()
        with self._lock:
            return int(conn.execute("SELECT COUNT(*) FROM source_events").fetchone()[0])

    def count_events_for_session(self, session_id: str) -> int:
        conn = self._connection()
        with self._lock:
            row = conn.execute(
                "SELECT COUNT(*) FROM source_events WHERE session_id=?", (session_id,)
            ).fetchone()
            return int(row[0])

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        conn = self._connection()
        with self._lock:
            row = conn.execute(
                f"SELECT {_EVENT_COLUMNS} FROM source_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
        return dict(zip(EVENT_FIELDS, row)) if row is not None else None

    def list_events(self, *, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        conn = self._connection()
        with self._lock:
            if session_id is None:
                rows = conn.execute(
                    f"SELECT {_EVENT_COLUMNS} FROM source_events ORDER BY observed_at, event_id LIMIT ?",
                    (int(limit),),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {_EVENT_COLUMNS} FROM source_events WHERE session_id=? "
                    "ORDER BY observed_at, event_id LIMIT ?",
                    (session_id, int(limit)),
                ).fetchall()
        return [dict(zip(EVENT_FIELDS, row)) for row in rows]

    def integrity_check(self) -> str:
        conn = self._connection()
        with self._lock:
            return str(conn.execute("PRAGMA integrity_check").fetchone()[0])


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
