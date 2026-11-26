"""Conversation JSONL archive with a recoverable checkpoint.

The database is canonical; the JSONL archive is a derived, human-readable copy.
That makes the recovery rule simple: an append is only acknowledged once the
stream checkpoint (byte length and last capture sequence) is committed. A crash
between writing lines and committing the checkpoint leaves extra bytes; the next
run truncates back to the checkpoint and re-appends, so replay converges to one
line per event.

Artifacts (non-text input) are referenced by metadata — id, URI, media type,
size and sha256 — never embedded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import StorageError
from .records import artifacts_for_event
from .storage import EVENT_FIELDS, Store, _now

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
# Event fields that are already JSON text in the database.
_JSON_FIELDS = {"subjects_json", "metadata_json"}


class Archive:
    """Per-session JSONL materialization for one deployment."""

    def __init__(self, store: Store, root: str | Path) -> None:
        self.store = store
        self.root = Path(root) / "archive"

    # -- paths --------------------------------------------------------------

    def path_for(self, stream: str) -> Path:
        safe = _UNSAFE.sub("_", stream or "unknown")[:96].strip("._-") or "unknown"
        return self.root / f"{safe}.jsonl"

    # -- writes -------------------------------------------------------------

    def sync_event(self, event_id: str) -> int:
        """Append any unarchived events for this event's session."""
        row = self.store.read_one("SELECT session_id FROM source_events WHERE event_id=?", (event_id,))
        if row is None:
            return 0
        return self.sync_stream(str(row[0]))

    def sync_stream(self, stream: str, *, update_checkpoint: bool = True) -> int:
        state = self.store.stream_state(stream)
        path = Path(state["path"]) if state else self.path_for(stream)
        last_seq = int(state["last_capture_seq"]) if state else 0
        byte_length = int(state["byte_length"]) if state else 0
        events = self._events_after(stream, last_seq)
        if not events:
            return 0

        self.root.mkdir(parents=True, exist_ok=True)
        mode = "r+b" if path.exists() else "w+b"
        with path.open(mode) as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() > byte_length:
                # Unacknowledged bytes from an interrupted append.
                handle.truncate(byte_length)
            if handle.tell() < byte_length:
                raise StorageError(
                    "archive is shorter than its checkpoint; refusing to guess at lost lines",
                    detail={"stream": stream, "path": str(path), "expected": byte_length},
                )
            handle.seek(0, os.SEEK_END)
            for event in events:
                handle.write(_line(self.store, event).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
            new_length = handle.tell()

        if update_checkpoint:
            with self.store.transaction() as conn:
                self.store.update_stream(
                    conn, stream, path=str(path), byte_length=new_length,
                    last_capture_seq=int(events[-1]["capture_seq"]),
                )
        return len(events)

    def _events_after(self, stream: str, last_seq: int) -> list[dict[str, Any]]:
        columns = ", ".join(EVENT_FIELDS)
        rows = self.store.read(
            f"SELECT {columns} FROM source_events WHERE session_id=? AND capture_seq > ? "
            "ORDER BY capture_seq",
            (stream, int(last_seq)),
        )
        return [dict(zip(EVENT_FIELDS, row)) for row in rows]

    # -- recovery -----------------------------------------------------------

    def recover(self) -> dict[str, Any]:
        """Bring every stream up to its committed checkpoint. Idempotent."""
        streams = [
            str(row[0]) for row in self.store.read(
                "SELECT DISTINCT session_id FROM source_events ORDER BY session_id"
            )
        ]
        appended = 0
        for stream in streams:
            appended += self.sync_stream(stream)
        return {
            "streams": len(streams),
            "appended": appended,
            "pending": int(self.store.read_one(
                "SELECT COUNT(*) FROM source_events e "
                "LEFT JOIN archive_streams s ON s.stream = e.session_id "
                "WHERE e.capture_seq > COALESCE(s.last_capture_seq, 0)"
            )[0]),
        }

    # -- inspection ---------------------------------------------------------

    def verify(self, *, base_dir: str | Path | None = None) -> list[dict[str, Any]]:
        """Check checkpoint/file agreement and artifact hashes."""
        problems: list[dict[str, Any]] = []
        for stream, path, byte_length, last_seq in self.store.read(
            "SELECT stream, path, byte_length, last_capture_seq FROM archive_streams"
        ):
            file = Path(path)
            if not file.exists():
                problems.append({"kind": "archive_missing", "stream": stream, "path": str(file)})
                continue
            size = file.stat().st_size
            if size != int(byte_length):
                problems.append({
                    "kind": "archive_length", "stream": stream, "path": str(file),
                    "expected": int(byte_length), "actual": size,
                })
        base = Path(base_dir) if base_dir is not None else self.root.parent
        for artifact_id, uri, sha, size in self.store.read(
            "SELECT artifact_id, uri, sha256, size_bytes FROM artifacts"
        ):
            candidate = Path(uri) if Path(uri).is_absolute() else base / uri
            if not candidate.exists():
                problems.append({"kind": "artifact_missing", "artifact_id": artifact_id, "uri": uri})
                continue
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != sha or candidate.stat().st_size != int(size):
                problems.append({
                    "kind": "artifact_mismatch", "artifact_id": artifact_id, "uri": uri,
                    "expected_sha256": sha, "actual_sha256": digest,
                })
        return problems


def _line(store: Store, event: dict[str, Any]) -> str:
    payload: dict[str, Any] = {
        "event_id": event["event_id"],
        "capture_seq": event["capture_seq"],
        "owner_id": event["owner_id"],
        "session_id": event["session_id"],
        "observed_at": event["observed_at"],
        "received_at": event["received_at"],
        "captured_at": event["captured_at"],
        "visibility": event["visibility"],
        "actor": {
            "id": event["actor_id"], "kind": event["actor_kind"], "name": event["actor_name"],
        },
        "delegation_parent": event["delegation_parent"],
        "source": {"kind": event["source_kind"], "id": event["source_id"]},
        "content": event["content"],
        "content_type": event["content_type"],
        "sha256": event["content_sha256"],
        "artifacts": artifacts_for_event(store, event["event_id"]),
        "archived_at": _now(),
    }
    for field in _JSON_FIELDS:
        payload[field.removesuffix("_json")] = json.loads(event[field] or "{}")
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)
