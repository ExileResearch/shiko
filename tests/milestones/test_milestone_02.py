"""Milestone 2 — durable capture and canonical storage.

Three entry points:

* ``test_canonical_records_and_revisions`` round-trips a multi-source memory,
  an unknown affect and a measured zero, upgrades a schema-1 fixture and proves
  an interrupted migration rolls back cleanly.
* ``test_capture_and_archive_recovery`` interrupts capture and archive at each
  boundary and checks that replay converges to one capture and one archive line.
* ``test_ordered_background_work`` holds the worker, shows a correction to the
  next turn before indexing, races two workers and kills one mid-job.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from shiko import (
    Actor,
    ActorKind,
    Affect,
    ClassificationInput,
    Engine,
    EngineConfig,
    EvidenceInput,
    MemoryDraft,
    RecordError,
    build_envelope,
    open_engine,
)
from shiko.errors import StorageError
from shiko.owner import ensure_manifest
from shiko.queue import job_for
from shiko.storage import DB_NAME, MIGRATIONS, Store, _migration_1
from shiko.worker import Worker

DEFAULT_POLICY = "m2.policy.1"


# -- helpers -----------------------------------------------------------------


def _engine(root: Path, name: str = "jiko", **config) -> Engine:
    return open_engine(root / name, name, config=EngineConfig(**config))


def _capture(engine, content: str, *, event_id: str, session="s1", source_kind="user.message",
             actor=None, delegation_parent=None):
    return engine.capture_text(
        content, event_id=event_id, session_id=session, source_kind=source_kind,
        actor=actor, delegation_parent=delegation_parent,
    )


def _v1_fixture(root: Path) -> None:
    """A schema-1 store with one event, as milestone 1 wrote it."""
    root.mkdir(parents=True, exist_ok=True)
    ensure_manifest(root, "jiko")
    conn = sqlite3.connect(root / DB_NAME)
    try:
        _migration_1(conn)
        conn.execute("INSERT INTO meta VALUES ('schema_version', '1')")
        conn.execute(
            "INSERT INTO schema_migrations VALUES (1, '2026-01-01T00:00:00Z', '1:_migration_1')"
        )
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
                "evt-v1", "jiko", "u1", "user", "Robin", "legacy-session", None,
                "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "private",
                "user.message", "", '{"kind": "user.message", "id": "", "detail": {}}',
                "legacy reading", "text/plain", "[]", "{}", "0" * 64,
                "2026-01-01T00:00:01Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()


class Clock:
    def __init__(self) -> None:
        # Slightly ahead of wall time: jobs are enqueued at the real clock, and a
        # worker is expected to run after the capture that queued them.
        self.now = datetime.now(timezone.utc) + timedelta(seconds=5)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


# -- 2.1 ---------------------------------------------------------------------


def test_canonical_records_and_revisions(tmp_path):
    """Phase 2.1 — provenance survives reopen, migration and unknown values."""
    engine = _engine(tmp_path)
    assert engine.schema_version == 2
    tool = _capture(engine, "scale reports 82 g", event_id="evt-tool", source_kind="tool.reading",
                    actor=Actor("kitchen-scale", ActorKind.TOOL))
    user = _capture(engine, "it arrived and it is light", event_id="evt-user")
    assert tool.ok and user.ok

    draft = MemoryDraft(
        kind="semantic",
        content="The wrist camera weighs 82 g.",
        evidence=(
            EvidenceInput(tool.event_id, "supports", (0, 20)),
            EvidenceInput(user.event_id, "describes"),
        ),
        classifications=(
            ClassificationInput("topic", "hardware"),
            ClassificationInput("epistemic", "measured", confidence=0.9),
            ClassificationInput("kind", "semantic"),
        ),
        affect=Affect(valence=None, arousal=0.0, confidence=0.6, source="estimated"),
        subject_kind="device",
        subject_id="wrist-camera",
        event_time=datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc),
    )
    memory = engine.create_memory(draft)
    assert memory["memory_id"].startswith("mem_")
    assert memory["current_revision"] == 1
    assert len(memory["evidence"]) == 2
    spans = {item["event_id"]: item["span"] for item in memory["evidence"]}
    assert spans[tool.event_id] == [0, 20]
    assert spans[user.event_id] is None
    assert len(memory["classifications"]) == 3
    assert memory["affect"]["valence"] is None
    assert memory["affect"]["arousal"] == 0.0
    assert set(memory["affect"]["unknown"]) == {"valence", "dominance"}
    assert memory["temporal"]["event_time"] == "2026-11-25T09:00:00Z"

    revised = engine.revise_memory(
        memory["memory_id"], "The wrist camera weighs 82 g on the kitchen scale.",
        source_event_id=tool.event_id,
    )
    assert revised["current_revision"] == 2
    assert len(revised["revisions"]) == 2
    assert revised["revisions"][0]["content"] == "The wrist camera weighs 82 g."
    assert revised["revisions"][0]["policy_version"] == DEFAULT_POLICY
    assert revised["revisions"][1]["source_event_id"] == tool.event_id

    memory_id, created_at = memory["memory_id"], memory["created_at"]
    engine.close()

    reopened = _engine(tmp_path)
    again = reopened.get_memory(memory_id)
    assert again is not None
    assert again["memory_id"] == memory_id
    assert again["created_at"] == created_at
    assert again["current_revision"] == 2
    assert len(again["revisions"]) == 2
    assert len(again["evidence"]) == 2
    assert again["affect"]["valence"] is None and again["affect"]["arousal"] == 0.0
    assert reopened.recover()["integrity"] == "ok"
    assert [m["memory_id"] for m in reopened.memories()] == [memory_id]

    with pytest.raises(RecordError):
        reopened.create_memory(MemoryDraft(
            kind="semantic", content="orphan",
            evidence=(EvidenceInput("evt-does-not-exist"),),
        ))
    with pytest.raises(RecordError):
        MemoryDraft(kind="semantic", content="no evidence", evidence=())
    reopened.close()

    # A schema-1 fixture upgrades without losing its event.
    legacy_root = tmp_path / "legacy"
    _v1_fixture(legacy_root)

    def interrupted(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE canonical_memories (memory_id TEXT PRIMARY KEY)")
        raise RuntimeError("simulated interruption")

    broken = Store(legacy_root / DB_NAME, "jiko", migrations=(_migration_1, interrupted))
    with pytest.raises(StorageError) as failure:
        broken.open()
    assert "migration to schema 2 failed" in str(failure.value)
    broken.close()
    raw = sqlite3.connect(legacy_root / DB_NAME)
    try:
        assert raw.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "1"
        assert raw.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='canonical_memories'"
        ).fetchone()[0] == 0
    finally:
        raw.close()

    upgraded = open_engine(legacy_root, "jiko")
    assert upgraded.schema_version == 2
    legacy_event = upgraded.get_event("evt-v1")
    assert legacy_event is not None
    assert legacy_event["content"] == "legacy reading"
    assert legacy_event["capture_seq"] == 1
    assert upgraded.open_result().created is False
    upgraded.close()


# -- 2.2 ---------------------------------------------------------------------


def test_capture_and_archive_recovery(tmp_path):
    """Phase 2.2 — commit, append and checkpoint each fail safe."""
    engine = _engine(tmp_path / "crash", auto_archive=False)
    result = _capture(engine, "correction: the camera is 82 g", event_id="ck-1", session="ck")
    assert result.ok and result.archived is False
    archive_file = engine.root / "archive" / "ck.jsonl"
    assert not archive_file.exists()
    assert engine.event_count == 1
    assert engine.pending_jobs() == 2

    first = engine.recover()
    assert first["archive"]["appended"] == 1
    assert first["archive"]["pending"] == 0
    assert first["integrity"] == "ok"
    assert len(archive_file.read_text().splitlines()) == 1
    assert engine.recover()["archive"]["appended"] == 0  # idempotent

    # Crash between the append and the checkpoint: extra bytes are truncated.
    engine.capture_text("second event", event_id="ck-2", session_id="ck", source_kind="tool.reading")
    store = engine.store
    original_update = store.update_stream
    store.update_stream = lambda *args, **kwargs: None  # type: ignore[method-assign]
    assert engine.archive.sync_stream("ck") == 1
    assert len(archive_file.read_text().splitlines()) == 2
    state = store.stream_state("ck")
    assert state is not None and state["last_capture_seq"] == 1
    store.update_stream = original_update  # type: ignore[method-assign]
    engine.archive.sync_stream("ck")
    lines = [json.loads(line) for line in archive_file.read_text().splitlines()]
    assert [line["capture_seq"] for line in lines] == [1, 2]

    # Replay converges: one logical capture, one archive entry.
    duplicate = _capture(engine, "correction: the camera is 82 g", event_id="ck-1", session="ck")
    assert duplicate.duplicate and duplicate.stored is False
    assert len(archive_file.read_text().splitlines()) == 2
    assert engine.event_count == 2

    # A failed archive write is visible at the call site and leaves work queued.
    failing = _engine(tmp_path / "fail", auto_archive=True)
    original_sync = failing.archive.sync_event

    def explode(event_id: str) -> int:
        raise OSError("disk full")

    failing.archive.sync_event = explode  # type: ignore[method-assign]
    degraded = _capture(failing, "camera frame metadata", event_id="f-1", session="f")
    assert degraded.status.value == "degraded"
    assert degraded.diagnostics[0].code == "archive.write_failed"
    assert failing.event_count == 1
    assert failing.pending_jobs() >= 1
    failing.archive.sync_event = original_sync  # type: ignore[method-assign]
    drained = failing.create_worker().run_until_idle()
    assert drained["failures"] == 0
    assert (failing.root / "archive" / "f.jsonl").read_text().count("\n") == 1

    # Tool, sensor, assistant and delegated reports are first-class sources.
    sources = _engine(tmp_path / "sources")
    source_specs = (
        ("tool.result", Actor("terminal", ActorKind.TOOL), None),
        ("sensor.reading", Actor("camera", ActorKind.TOOL), None),
        ("assistant.message", Actor("jiko", ActorKind.OWNER), None),
        ("subagent.report", Actor("helper-1", ActorKind.SUBAGENT), "parent-session"),
    )
    for index, (kind, actor, parent) in enumerate(source_specs):
        _capture(sources, f"payload {index}", event_id=f"src-{index}", session="src",
                 source_kind=kind, actor=actor, delegation_parent=parent)
    rows = sources.events(session_id="src")
    assert [(row["actor_kind"], row["source_kind"]) for row in rows] == [
        ("tool", "tool.result"), ("tool", "sensor.reading"),
        ("owner", "assistant.message"), ("subagent", "subagent.report"),
    ]
    assert rows[3]["delegation_parent"] == "parent-session"
    archived = [json.loads(line) for line in
                (sources.root / "archive" / "src.jsonl").read_text().splitlines()]
    assert [line["capture_seq"] for line in archived] == [1, 2, 3, 4]
    assert archived[3]["actor"]["kind"] == "subagent"

    # Non-text input: reference plus integrity metadata, never embedded bytes.
    image_bytes = b"\xff\xd8\xff\xe0shiko-test-jpeg"
    envelope = build_envelope(
        "", event_id="img-1", actor=Actor("camera", ActorKind.TOOL), session_id="img",
        source_kind="sensor.frame", content_type="image/jpeg",
    )
    captured = sources.capture_artifact(envelope, image_bytes, media_type="image/jpeg", name="wrist.jpg")
    assert captured["capture"]["status"] == "ok"
    assert captured["artifact"]["sha256"] == hashlib.sha256(image_bytes).hexdigest()
    assert captured["artifact"]["size_bytes"] == len(image_bytes)
    line = json.loads((sources.root / "archive" / "img.jsonl").read_text().splitlines()[0])
    assert line["artifacts"][0]["artifact_id"] == captured["artifact"]["artifact_id"]
    assert line["content"] == ""
    assert sources.verify_artifacts() == []
    artifact_path = sources.root / captured["artifact"]["uri"]
    artifact_path.write_bytes(b"tampered")
    problems = sources.verify_artifacts()
    assert any(problem["kind"] == "artifact_mismatch" for problem in problems)
    sources.close()
    failing.close()


# -- 2.3 ---------------------------------------------------------------------


def test_ordered_background_work(tmp_path):
    """Phase 2.3 — leases, retries, ordering and a committed read boundary."""
    clock = Clock()
    engine = _engine(tmp_path / "order")
    _capture(engine, "we measured the camera at 120 g", event_id="old", session="s1")
    _capture(engine, "correction: the camera is 82 g", event_id="new", session="s1")
    _capture(engine, "how heavy is it?", event_id="follow", session="s1")

    context = engine.recent_context(session_id="s1", exclude_event_id="follow")
    ids = [entry["event_id"] for entry in context["events"]]
    assert "new" in ids and "follow" not in ids
    correction = next(entry for entry in context["events"] if entry["event_id"] == "new")
    assert correction["pending"] is True and correction["indexed"] is False
    assert context["boundary"] == 0
    assert [entry["capture_seq"] for entry in context["events"]] == sorted(
        entry["capture_seq"] for entry in context["events"]
    )

    worker = engine.create_worker(worker_id="drain", clock=clock)
    summary = worker.run_until_idle()
    assert summary["processed"] == 3 and summary["failures"] == 0
    drained = engine.recent_context(session_id="s1", exclude_event_id="follow")
    assert all(entry["indexed"] for entry in drained["events"])
    assert not any(entry["pending"] for entry in drained["events"])
    assert drained["boundary"] == 3

    # Two workers race one job: exactly one wins the lease.
    raced = _engine(tmp_path / "race")
    _capture(raced, "only one", event_id="race-1", session="race")
    outcomes: list[dict | None] = []
    lock = threading.Lock()

    def take(worker_id: str) -> None:
        record = raced.create_worker(worker_id=worker_id, clock=clock).run_once(kinds=("index.event",))
        with lock:
            outcomes.append(record)

    threads = [threading.Thread(target=take, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(1 for outcome in outcomes if outcome) == 1
    assert sum(1 for outcome in outcomes if outcome is None) == 1

    # A killed worker leaves a lease; it expires and the next worker finishes once.
    killed = _engine(tmp_path / "kill", auto_archive=False)
    _capture(killed, "camera arrived", event_id="k-1", session="k")

    class BoomWorker(Worker):
        def _archive_event(self, job) -> None:
            raise SystemExit("killed mid-job")

    dead = BoomWorker(killed.store, killed.archive, worker_id="dead", clock=clock, lease_seconds=30)
    with pytest.raises(SystemExit):
        dead.run_once(kinds=("archive.event",))
    job = job_for(killed.store, "archive.event", "k-1")
    assert job is not None and job.state == "leased" and job.lease_owner == "dead"
    assert killed.create_worker(worker_id="b", clock=clock).run_once(kinds=("archive.event",)) is None
    clock.advance(seconds=31)
    recovered = killed.create_worker(worker_id="b", clock=clock, lease_seconds=5).run_once(
        kinds=("archive.event",)
    )
    assert recovered is not None and recovered["state"] == "done"
    assert (killed.root / "archive" / "k.jsonl").read_text().count("\n") == 1

    # Bounded retries end in a visible failed item; recovery then converges.
    retrying = _engine(tmp_path / "retry", auto_archive=False)
    _capture(retrying, "retry me", event_id="r-1", session="r")

    def explode(event_id: str) -> int:
        raise OSError("disk full")

    retrying.archive.sync_event = explode  # type: ignore[method-assign]
    worker = retrying.create_worker(worker_id="w", backoff_seconds=0.0, clock=clock)
    states = [worker.run_once(kinds=("archive.event",)) for _ in range(3)]
    assert [record["state"] for record in states if record] == ["pending", "pending", "failed"]
    failures = retrying.failed_jobs()
    assert len(failures) == 1 and failures[0]["kind"] == "archive.event"
    assert "disk full" in failures[0]["last_error"]

    from shiko.archive import Archive

    retrying.archive.sync_event = Archive.sync_event.__get__(retrying.archive, Archive)  # type: ignore[method-assign]
    retrying.recover()
    assert (retrying.root / "archive" / "r.jsonl").read_text().count("\n") == 1

    # Stop drains nothing and leaves queued work for the next worker.
    stopping = _engine(tmp_path / "stop")
    _capture(stopping, "later", event_id="s-1", session="s")
    stopped = stopping.create_worker(worker_id="stopped")
    stopped.stop()
    assert stopped.run_until_idle()["processed"] == 0
    assert stopping.pending_jobs() >= 1
    assert stopping.create_worker(worker_id="later").run_until_idle()["processed"] >= 1

    for item in (engine, raced, killed, retrying, stopping):
        item.close()
