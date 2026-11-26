"""Milestone 1 — engine ownership and host contracts.

Three entry points, one per phase:

* ``test_installable_package`` builds both archives, installs each into a fresh
  interpreter outside the checkout, and exercises the public contract. It also
  loads the plugin door the way Hermes does.
* ``test_one_owner_per_deployment`` provisions two owners side by side, reopens
  them, and proves that ownership and delegation survive.
* ``test_hermes_lifecycle_probe`` drives a real Hermes session and compares the
  observed lifecycle to the adapter mapping.

The first two are hermetic. The third needs a live host and a model call, so it
is gated by ``SHIKO_LIVE=1`` and records its trace under
``notes/plans/evidence/artifacts/milestone-1/``.
"""

from __future__ import annotations

import configparser
import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from shiko import (
    Actor,
    ActorKind,
    EnvelopeError,
    OwnershipError,
    SourceEnvelope,
    Subject,
    Visibility,
    build_envelope,
    open_engine,
    read_manifest,
)
from tests.support import (
    REPO_ROOT,
    last_json_line,
    make_venv,
    pip_install,
    read_jsonl,
    run,
    run_python,
)
from tests.milestones.host_facts import hermes_versions as host_versions

SMOKE_SCRIPT = """
    import json, os, sys, tempfile
    import shiko

    assert "agent" not in sys.modules, "core import pulled in the Hermes host"
    assert "hermes" not in sys.modules, "core import pulled in the Hermes host"
    assert "shiko.adapters" not in sys.modules, "core import eagerly imported adapters"

    base = tempfile.mkdtemp(prefix="shiko-smoke-")
    root = os.path.join(base, "deploy")
    engine = shiko.open_engine(root, "smoke")
    capture = engine.capture_text(
        "The wrist stalled with the M2000 camera.",
        session_id="probe",
        source_kind="smoke.turn",
    )
    recall = engine.recall("wrist")
    outcome = engine.record_outcome("mem-1", signal="exposure")
    maintenance = engine.maintain()
    closed = engine.close()
    print("RESULT:" + json.dumps({
        "version": shiko.__version__,
        "root": root,
        "owner": engine.owner_id,
        "schema": engine.schema_version,
        "capture_status": capture.status.value,
        "capture_stored": capture.stored,
        "recall_status": recall.status.value,
        "recall_empty": recall.empty,
        "recall_code": recall.diagnostics[0].code if recall.diagnostics else "",
        "outcome_status": outcome.status.value,
        "maintenance_status": maintenance.status.value,
        "closed": closed.closed,
        "manifest": os.path.exists(os.path.join(root, "manifest.json")),
        "hermes_contract": sys.modules["shiko"].__dict__.get("HOST_CONTRACT"),
    }))
"""

_FORBIDDEN_ARCHIVE_PARTS = ("notes/", "tests/", ".venv/", "dist/", "build/", "__pycache__/", ".DS_Store")


def _assert_clean(names: list[str], label: str) -> None:
    for name in names:
        assert not any(part in name for part in _FORBIDDEN_ARCHIVE_PARTS), (
            f"{label} contains private or build-only path: {name}"
        )


def _wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def _sdist_names(sdist: Path) -> list[str]:
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
    prefix = names[0].split("/", 1)[0] + "/"
    return [name[len(prefix):] if name.startswith(prefix) else name for name in names]


def _wheel_entry_points(wheel: Path) -> configparser.ConfigParser:
    with zipfile.ZipFile(wheel) as archive:
        entry = next(n for n in archive.namelist() if n.endswith(".dist-info/entry_points.txt"))
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(entry).decode("utf-8"))
        return parser


def _wheel_metadata(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        return archive.read(name).decode("utf-8")


def _alt_python() -> Path:
    """A second interpreter when one exists; the current one otherwise."""
    configured = os.environ.get("SHIKO_ALT_PYTHON", "").strip()
    if configured and Path(configured).exists():
        return Path(configured)
    for candidate in sorted(Path.home().glob(".hermes/tools/python-3.1[34]*/bin/python3")):
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def _load_plugin_door():
    """Import the repository's ``__init__.py`` exactly as Hermes does: as a package."""
    module_name = "shiko_plugin_door_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeCtx:
    def __init__(self) -> None:
        self.provider = None
        self.hooks: list[str] = []

    def register_memory_provider(self, provider) -> None:
        self.provider = provider

    def register_hook(self, name, callback, **kwargs) -> None:
        self.hooks.append(name)


@pytest.mark.slow
def test_installable_package(built_distributions, tmp_path, monkeypatch):
    """Phase 1.1 — both distributions install clean and expose one contract."""
    outdir, wheel, sdist = built_distributions

    wheel_names = _wheel_names(wheel)
    sdist_names = _sdist_names(sdist)
    _assert_clean(wheel_names, "wheel")
    _assert_clean(sdist_names, "sdist")

    # The wheel is the library; the sdist is the plugin tree as well.
    assert "shiko/__init__.py" in wheel_names
    assert "shiko/engine.py" in wheel_names
    assert "shiko/adapters/hermes.py" in wheel_names
    for required in ("plugin.yaml", "__init__.py", "LICENSE", "pyproject.toml", "shiko/__init__.py"):
        assert required in sdist_names, f"sdist is missing {required}"

    # Metadata: ASCII constraint, matching versions, memory-provider entry point.
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert (REPO_ROOT / "pyproject.toml").read_bytes().isascii(), "pyproject.toml must stay ASCII"
    assert project["project"]["requires-python"] == ">=3.11"
    assert project["project"]["name"] == "shiko-memory"

    import shiko as repo_shiko

    metadata = _wheel_metadata(wheel)
    assert "Requires-Python: >=3.11" in metadata
    assert "Name: shiko-memory" in metadata
    assert f"Version: {repo_shiko.__version__}" in metadata

    plugin_yaml = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")
    assert re.search(rf'^version:\s*"{re.escape(repo_shiko.__version__)}"', plugin_yaml, re.M)

    entry_points = _wheel_entry_points(wheel)
    assert entry_points["hermes_agent.memory_providers"]["shiko"] == "shiko.adapters.hermes:provider"

    # Install each archive into its own fresh interpreter, outside the checkout.
    primary = make_venv(tmp_path / "venv-wheel")
    pip_install(primary, str(wheel), offline=True)
    wheel_run = run_python(primary, SMOKE_SCRIPT, cwd=tmp_path)
    assert wheel_run.returncode == 0, wheel_run.stderr[-4000:]
    wheel_result = last_json_line(wheel_run.stdout)

    alt_python = _alt_python()
    alt_venv = tmp_path / "venv-sdist"
    subprocess.run([str(alt_python), "-m", "venv", str(alt_venv)], check=True, capture_output=True, text=True)
    secondary = alt_venv / "bin" / "python"
    pip_install(secondary, str(sdist))
    sdist_run = run_python(secondary, SMOKE_SCRIPT, cwd=tmp_path)
    assert sdist_run.returncode == 0, sdist_run.stderr[-4000:]
    sdist_result = last_json_line(sdist_run.stdout)

    expected_version = repo_shiko.__version__
    for label, result in (("wheel", wheel_result), ("sdist", sdist_result)):
        assert result["version"] == expected_version, label
        assert result["owner"] == "smoke", label
        assert result["schema"] == 1, label
        assert result["manifest"] is True, label
        assert result["capture_status"] == "ok" and result["capture_stored"] is True, label
        # An unavailable path is not an empty answer.
        assert result["recall_status"] == "unsupported", label
        assert result["recall_empty"] is False, label
        assert result["recall_code"] == "recall.not_implemented", label
        assert result["outcome_status"] == "unsupported", label
        assert result["maintenance_status"] == "unsupported", label
        assert result["closed"] is True, label

    # The plugin door: Hermes loads the repo directory as a package and calls register(ctx).
    door = _load_plugin_door()
    assert "register_memory_provider" in (REPO_ROOT / "__init__.py").read_text(encoding="utf-8")
    ctx = _FakeCtx()
    door.register(ctx)
    assert ctx.provider is not None and ctx.provider.name == "shiko"

    for key in ("SHIKO_HOME", "SHIKO_OWNER", "HERMES_HOME"):
        monkeypatch.delenv(key, raising=False)
    home = tmp_path / "host-home"
    home.mkdir()
    ctx.provider.initialize(
        "session-1", hermes_home=str(home), agent_identity={"name": "Jiko Prime"},
        agent_context="primary", platform="unit",
    )
    engine = ctx.provider.engine
    assert engine is not None and engine.owner_id == "jiko-prime"
    assert engine.root == (home / "shiko").resolve()
    assert ctx.provider.prefetch("wrist") == ""
    assert ctx.provider.recall_status() is None
    assert ctx.provider.get_tool_schemas() == []

    ctx.provider.sync_turn(
        "I found a camera for the wrist", "How heavy is it?",
        session_id="session-1", turn_author={"id": "u1", "name": "Robin"},
    )
    assert engine.event_count == 2
    ctx.provider.on_delegation("weigh the camera", "measured 82 g", child_session_id="child-1")
    assert engine.event_count == 3
    delegation = [e for e in engine.events(session_id="session-1") if e["source_kind"] == "hermes.delegation.report"]
    assert len(delegation) == 1
    assert delegation[0]["actor_kind"] == "subagent"
    assert delegation[0]["actor_id"] == "child-1"
    assert delegation[0]["delegation_parent"] == "session-1"
    assert delegation[0]["owner_id"] == "jiko-prime"
    ctx.provider.shutdown()
    assert engine.closed is True


def test_one_owner_per_deployment(tmp_path):
    """Phase 1.2 — one store per owner; identical event IDs stay independent."""
    jiko = open_engine(tmp_path / "jiko", "jiko")
    kumi = open_engine(tmp_path / "kumi", "kumi")
    assert jiko.owner_id == "jiko" and kumi.owner_id == "kumi"
    assert jiko.open_result().created is True

    jiko_result = jiko.capture(build_envelope(
        "payload measured 1.24 kg", actor=Actor("scale-a", ActorKind.TOOL),
        event_id="evt-0001", session_id="shared-design",
        source_kind="tool.reading", source_id="scale-1",
        subjects=(Subject("wrist", "device", "camera wrist"),),
    ))
    kumi_result = kumi.capture(build_envelope(
        "payload estimated 1.5 kg", actor=Actor("kumi-scale", ActorKind.TOOL),
        event_id="evt-0001", session_id="shared-design",
        source_kind="tool.reading", source_id="scale-1",
    ))
    assert jiko_result.ok and kumi_result.ok
    assert jiko.event_count == 1 and kumi.event_count == 1

    # The same incoming event ID resolves to a different row in each owner's store.
    jiko_row = jiko.get_event("evt-0001")
    kumi_row = kumi.get_event("evt-0001")
    assert jiko_row is not None and kumi_row is not None
    assert jiko_row["owner_id"] == "jiko" and kumi_row["owner_id"] == "kumi"
    assert jiko_row["content"] != kumi_row["content"]
    assert jiko_row["subjects_json"] == json.dumps([{"id": "wrist", "kind": "device", "label": "camera wrist"}], sort_keys=True, separators=(",", ":"))
    jiko.close()
    kumi.close()

    # Reopen: ownership is verified, not assumed.
    reopened = open_engine(tmp_path / "jiko", "jiko")
    assert reopened.open_result().created is False
    assert reopened.event_count == 1

    with pytest.raises(OwnershipError) as refusal:
        open_engine(tmp_path / "jiko", "kumi")
    assert "jiko" in str(refusal.value) and "kumi" in str(refusal.value)
    assert refusal.value.detail["expected_owner"] == "jiko"

    # A delegated report keeps its actor and parent; the owner stays Jiko.
    report = build_envelope(
        "measured 82 g on the kitchen scale",
        event_id="evt-0002",
        actor=Actor("researcher-1", ActorKind.SUBAGENT, "helper"),
        session_id="parent-session",
        source_kind="subagent.report",
        delegation_parent="parent-session",
        visibility=Visibility.PRIVATE,
    )
    assert reopened.capture(report).ok
    stored = reopened.get_event("evt-0002")
    assert stored is not None
    assert stored["actor_id"] == "researcher-1"
    assert stored["actor_kind"] == "subagent"
    assert stored["delegation_parent"] == "parent-session"
    assert stored["owner_id"] == "jiko"

    # Idempotent duplicate: same ID and content writes nothing.
    again = reopened.capture(build_envelope(
        "measured 82 g on the kitchen scale",
        event_id="evt-0002", actor=Actor("researcher-1", ActorKind.SUBAGENT),
        session_id="parent-session", source_kind="subagent.report",
    ))
    assert again.duplicate is True and again.conflict is False and again.stored is False
    assert again.diagnostics[0].code == "event.duplicate"
    assert reopened.event_count == 2

    # Reused ID with different content is visible and does not overwrite.
    conflict = reopened.capture(build_envelope(
        "measured 96 g with the new battery",
        event_id="evt-0002", actor=Actor("researcher-1", ActorKind.SUBAGENT),
        session_id="parent-session", source_kind="subagent.report",
    ))
    assert conflict.duplicate is True and conflict.conflict is True
    assert conflict.diagnostics[0].code == "event.id_reuse"
    assert conflict.diagnostics[0].severity.value == "error"
    preserved = reopened.get_event("evt-0002")
    assert preserved is not None and preserved["content"] == "measured 82 g on the kitchen scale"
    assert reopened.event_count == 2

    # Envelope round trip and validation.
    envelope = build_envelope(
        "camera arrived", event_id="evt-0003", actor=Actor("u1", ActorKind.USER),
        session_id="s3", source_kind="user.message", visibility=Visibility.DEPLOYMENT,
        metadata={"note": "unverified"},
    )
    assert SourceEnvelope.from_dict(envelope.to_dict()) == envelope
    assert reopened.capture(envelope).ok
    deployment_event = reopened.get_event("evt-0003")
    assert deployment_event is not None and deployment_event["visibility"] == "deployment"

    with pytest.raises(EnvelopeError):
        build_envelope("naive", event_id="evt-0004", actor=Actor("u1", ActorKind.USER),
                       session_id="s4", source_kind="user.message",
                       observed_at=datetime.now(), received_at=datetime.now(timezone.utc))
    with pytest.raises(EnvelopeError):
        build_envelope("bad id", event_id="  ", actor=Actor("u1", ActorKind.USER),
                       session_id="s4", source_kind="user.message")

    # A store without a manifest is refused rather than adopted.
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "shiko.db").write_bytes(b"")
    with pytest.raises(OwnershipError):
        open_engine(orphan, "jiko")

    manifest = read_manifest(tmp_path / "jiko")
    assert manifest is not None and manifest.owner_id == "jiko"
    assert oct((tmp_path / "jiko" / "manifest.json").stat().st_mode & 0o777) == "0o600"
    reopened.close()


def _hermes_python() -> Path:
    """The interpreter the installed Hermes runtime uses."""
    configured = os.environ.get("SHIKO_HERMES_PYTHON", "").strip()
    if configured and Path(configured).exists():
        return Path(configured)
    candidates = sorted(
        Path.home().glob(".hermes/installs/*/environments/*/venv/bin/python"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    assert candidates, "no Hermes runtime interpreter found under ~/.hermes/installs"
    return candidates[0]


def _first(group: list[dict], name: str) -> int | None:
    for record in group:
        if record["event"] == name:
            return record["seq"]
    return None


@pytest.mark.host
def test_hermes_lifecycle_probe(tmp_path, artifact_dir):
    """Phase 1.3 — observe the pinned host's real lifecycle and compare it to the adapter.

    The host's gateway is a host-wide singleton: swapping a memory provider in a
    live deployment would require restarting it and would interrupt active
    sessions. Instead this runs the real ``AIAgent`` + ``MemoryManager`` from the
    pinned install in its own process against a loopback fake model, in a
    throwaway HERMES_HOME. Discovery, initialization, prefetch, capture,
    compaction hooks and shutdown are the host's own code paths.
    """
    if os.environ.get("SHIKO_LIVE") != "1":
        pytest.skip("live host probe: set SHIKO_LIVE=1 after recording the target host revision")

    python = _hermes_python()
    agent_root = Path(os.environ.get("SHIKO_HERMES_AGENT_ROOT") or Path.home() / ".hermes" / "hermes-agent")
    assert (agent_root / "run_agent.py").exists(), f"no Hermes checkout at {agent_root}"

    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "tests" / "milestones" / "hermes_probe", home / "plugins" / "shiko_probe")

    log = home / "probe.jsonl"
    history = home / "history.json"
    requests_path = home / "requests.json"
    env = {
        "HERMES_HOME": str(home),
        "SHIKO_PROBE_LOG": str(log),
        "PYTHONPATH": str(agent_root),
        "HERMES_SKIP_UPDATE_CHECK": "1",
        "NO_COLOR": "1",
    }

    def drive(extra: list[str]) -> subprocess.CompletedProcess[str]:
        return run(
            [str(python), str(REPO_ROOT / "tests" / "milestones" / "host_driver.py"),
             "--session-id", "probe-session", *extra],
            cwd=agent_root, env=env, timeout=300,
        )

    first = drive(["--turn", "pair", "--history-out", str(history), "--requests-out", str(requests_path)])
    second = drive(["--turn", "second", "--history-in", str(history)])

    events = read_jsonl(log)
    assert events, "probe wrote no lifecycle events"

    # Two processes: a session with two turns, then a restart that replays the transcript.
    pids: list[int] = []
    for record in events:
        if record["pid"] not in pids:
            pids.append(record["pid"])
    assert len(pids) == 2, f"expected two host processes, saw {pids}"
    first_group = [record for record in events if record["pid"] == pids[0]]
    second_group = [record for record in events if record["pid"] == pids[1]]

    observed = {record["event"] for record in events}
    provider_events = {record["event"] for record in events if record["event"].startswith("provider.")}
    hook_events = {record["event"] for record in events if record["event"].startswith("hook.")}

    # -- capture path ------------------------------------------------------
    for group, label in ((first_group, "session"), (second_group, "restart")):
        names = [record["event"] for record in group]
        assert "plugin.register" in names, label
        assert "plugin.provider_registered" in names, label
        assert "provider.initialize" in names, label
        assert "provider.sync_turn" in names, label
        assert "provider.shutdown" in names, label

    initialize = next(record for record in first_group if record["event"] == "provider.initialize")
    init_keys = set(initialize["payload"]["keys"])
    assert {"hermes_home", "agent_context", "platform", "message_count", "resumed"} <= init_keys, init_keys

    # -- recall path: prefetch runs inside the turn, before the model answers --
    pre_llm = _first(first_group, "hook.pre_llm_call")
    turn_start = _first(first_group, "provider.on_turn_start")
    prefetch = _first(first_group, "provider.prefetch")
    post_llm = _first(first_group, "hook.post_llm_call")
    sync_turn = _first(first_group, "provider.sync_turn")
    queue_prefetch = _first(first_group, "provider.queue_prefetch")
    assert pre_llm is not None and turn_start is not None and prefetch is not None
    assert post_llm is not None and sync_turn is not None and queue_prefetch is not None
    assert pre_llm < turn_start < prefetch < post_llm, (pre_llm, turn_start, prefetch, post_llm)
    assert post_llm < sync_turn < queue_prefetch, (post_llm, sync_turn, queue_prefetch)

    # Prefetch output really reaches the model request, not just the callback.
    request_bodies = json.loads(requests_path.read_text(encoding="utf-8"))
    assert any(
        "SHIKO-PROBE-RECALL-SENTINEL" in json.dumps(body)
        for body in request_bodies
        if body is not None
    ), "prefetch output never reached a model request"

    # -- tool results are visible ------------------------------------------
    tool_events = [record for record in first_group if record["event"] == "hook.post_tool_call"]
    assert tool_events, "no post_tool_call hook fired"
    assert tool_events[0]["payload"]["tool_name"]["type"] == "str"

    # -- compaction hook ---------------------------------------------------
    pre_compress = next(record for record in first_group if record["event"] == "provider.on_pre_compress")
    assert pre_compress["payload"]["messages"]["type"] == "list"
    assert "provider.on_pre_compress" in provider_events

    # -- restart saw the earlier conversation ------------------------------
    restart_sync = next(record for record in second_group if record["event"] == "provider.sync_turn")
    assert restart_sync["payload"]["messages"]["len"] >= 4, restart_sync["payload"]

    # -- session end and shutdown -----------------------------------------
    assert "provider.on_session_end" in provider_events
    assert "hook.on_session_end" in hook_events
    # The host asks for the recall indicator on this path, so the adapter's
    # None return is an observed decision rather than an assumption.
    assert "provider.recall_status" in provider_events

    # -- every observed callback exists on the shipped adapter -------------
    from shiko.adapters.hermes import ShikoProvider

    provider = ShikoProvider()
    for event in sorted(provider_events):
        method = event.split(".", 1)[1]
        assert hasattr(provider, method), f"adapter is missing observed callback {method}"
    sync_signature = inspect.signature(ShikoProvider.sync_turn)
    assert {"user_content", "assistant_content", "session_id", "messages", "turn_author"} <= set(sync_signature.parameters)
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(ShikoProvider.initialize).parameters.values())

    # -- documented absences ----------------------------------------------
    absent = {
        "provider.on_session_switch": "fires only on /resume, /branch or /reset inside one process",
        "provider.on_delegation": "requires a real subagent delegation",
        "provider.on_memory_write": "requires a native memory-tool write",
        "hook.on_session_open": "registered, not emitted by this construction path",
    }
    for name in absent:
        assert name not in observed, f"{name} unexpectedly fired; update the mapping"

    mapping = {
        "host": {
            "python": str(python),
            "agent_root": str(agent_root),
            **{k: v for k, v in host_versions().items() if k in {"hermes_version", "hermes_revision"}},
        },
        "processes": [pids[0], pids[1]],
        "provider_events": sorted(provider_events),
        "hook_events": sorted(hook_events),
        "initialize_keys": sorted(init_keys),
        "turn_order": {
            "pre_llm_call": pre_llm,
            "on_turn_start": turn_start,
            "prefetch": prefetch,
            "post_llm_call": post_llm,
            "sync_turn": sync_turn,
            "queue_prefetch": queue_prefetch,
        },
        "sentinel_reached_model_request": True,
        "absent": absent,
        "restart_messages_seen": restart_sync["payload"]["messages"]["len"],
        "driver": {
            "first_requests": last_json_line(first.stdout).get("requests"),
            "second_requests": last_json_line(second.stdout).get("requests"),
        },
    }
    (artifact_dir / "probe-trace.json").write_text(
        json.dumps(events, indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifact_dir / "probe-mapping.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifact_dir / "probe-requests.json").write_text(
        json.dumps(request_bodies, indent=2), encoding="utf-8"
    )

