"""Milestone 3 — acquisition, episodes and provenance.

Phase 3.1 is implemented here: structured and generative extraction with span,
modality, subject, epistemic type, scope, negation and quotation attribution.
Phases 3.2–3.4 add their entry points alongside it as they are implemented.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from shiko import (
    Actor,
    ActorKind,
    Engine,
    EngineConfig,
    GenerativeExtractor,
    MemoryDraft,
    StructuredExtractor,
    build_envelope,
    open_engine,
)
from shiko.backends import BackendTimeout, OpenAIBackend, RecordedBackend
from shiko.llm_config import ConfigError, build_backend, load_settings
from shiko.records import EvidenceInput


def _engine(root: Path) -> Engine:
    return open_engine(root / "jiko", "jiko", config=EngineConfig(
        pipeline_debounce_ms=20, pipeline_max_delay_ms=200,
    ))


def _capture(engine, content: str, *, event_id: str, session="s1",
             source_kind="user.message", actor=None, metadata=None, content_type="text/plain"):
    return engine.capture_text(
        content, event_id=event_id, session_id=session, source_kind=source_kind,
        actor=actor, metadata=metadata, content_type=content_type,
    )


def _live_backend() -> OpenAIBackend | None:
    """Resolve the live backend from config/env; the key is never printed."""
    env = dict(os.environ)
    env.setdefault("SHIKO_ENV_FILE", str(Path.home() / ".hermes" / ".env"))
    env.setdefault("SHIKO_LLM_PROVIDER", "opencode")
    env.setdefault("SHIKO_LLM_MODEL", "deepseek-v4.1-flash")
    env.setdefault("SHIKO_LLM_TEMPERATURE", "0.3")
    if str(env["SHIKO_LLM_MODEL"]).startswith("deepseek"):
        env.setdefault("SHIKO_LLM_REASONING_EFFORT", "none")
        env.setdefault("SHIKO_LLM_EXTRA_BODY", '{"thinking": {"type": "disabled"}}')
    try:
        settings = load_settings(env=env)
    except ConfigError:
        return None
    if not settings.api_key:
        return None
    return build_backend(settings)


TEXT = (
    "I don't think the NEMA17 will be strong enough. Robin said 'the NEMA17 is fine'. "
    "We proposed the M2000; the 7020 currently has a GTX 960."
)


def _span(phrase: str) -> list[int]:
    start = TEXT.index(phrase)
    return [start, start + len(phrase)]


def _claims() -> list[dict]:
    return [
        {
            "content": "The NEMA17 will not be strong enough.",
            "quote": "I don't think the NEMA17 will be strong enough.",
            "kind": "semantic", "epistemic": "belief", "modality": "text", "scope": "internal",
            "subject": {"kind": "device", "id": "nema17"},
            "span": _span("I don't think the NEMA17 will be strong enough."),
            "negated": True, "confidence": 0.7,
        },
        {
            "content": "The NEMA17 is fine.",
            "quote": "the NEMA17 is fine",
            "kind": "semantic", "epistemic": "reported", "modality": "text", "scope": "user",
            "subject": {"kind": "person", "id": "robin"},
            "span": _span("the NEMA17 is fine"),
            "quoted_speaker": "Robin", "confidence": 0.9,
        },
        {
            "content": "We proposed the M2000.",
            "quote": "We proposed the M2000",
            "kind": "semantic", "epistemic": "hypothesis", "modality": "text", "scope": "shared",
            "subject": {"kind": "device", "id": "m2000"},
            "span": _span("We proposed the M2000"), "confidence": 0.8,
        },
        {
            "content": "The 7020 currently has a GTX 960.",
            "quote": "the 7020 currently has a GTX 960",
            "kind": "semantic", "epistemic": "fact", "modality": "text", "scope": "shared",
            "subject": {"kind": "device", "id": "7020"},
            "span": _span("the 7020 currently has a GTX 960"), "confidence": 0.95,
        },
        {
            "content": "The camera is heavy.",
            "quote": "The camera is heavy",
            "kind": "semantic", "epistemic": "fact", "modality": "text", "scope": "internal",
            "span": [0, 18],
        },
    ]


def _all_classifications(memory: dict) -> set[tuple[str, str]]:
    return {(item["axis"], item["value"]) for item in memory["classifications"]}


def _content(memory: dict) -> str:
    return memory["revisions"][-1]["content"]


def test_candidate_extraction(tmp_path):
    """Phase 3.1 — spurs, negation, quotation and failures stay attributable."""
    engine = _engine(tmp_path)
    event = _capture(engine, TEXT, event_id="turn-1")
    assert event.ok

    backend = RecordedBackend([json.dumps(_claims())])
    extractor = GenerativeExtractor(backend, timeout=5)
    result = engine.extract_event(event.event_id, extractor=extractor)

    assert result.status == "succeeded"
    assert len(result.accepted) == 4
    assert [item.reason for item in result.rejected] == ["quote_not_in_source"]
    assert backend.calls == 1

    memories = engine.memories()
    assert len(memories) == 4
    by_content = {_content(item): item for item in memories}
    assert set(by_content) == {
        "The NEMA17 will not be strong enough.", "The NEMA17 is fine.",
        "We proposed the M2000.", "The 7020 currently has a GTX 960.",
    }

    # The proposal and the installed part are different claims about different devices.
    proposal = by_content["We proposed the M2000."]
    installed = by_content["The 7020 currently has a GTX 960."]
    assert proposal["memory_id"] != installed["memory_id"]
    assert proposal["subject"] == {"kind": "device", "id": "m2000"}
    assert installed["subject"] == {"kind": "device", "id": "7020"}
    assert ("epistemic", "hypothesis") in _all_classifications(proposal)
    assert ("epistemic", "fact") in _all_classifications(installed)

    # Negation is a property of the denied claim, not a separate assertion.
    negated = by_content["The NEMA17 will not be strong enough."]
    assert ("polarity", "negated") in _all_classifications(negated)
    assert ("epistemic", "belief") in _all_classifications(negated)

    # A quotation keeps its speaker and is marked reported, not fact.
    quoted = by_content["The NEMA17 is fine."]
    assert ("epistemic", "reported") in _all_classifications(quoted)
    assert ("attribution", "Robin") in _all_classifications(quoted)
    assert quoted["subject"] == {"kind": "person", "id": "robin"}

    # Spans survive into the evidence links, pointing at each clause.
    for content, phrase in (
        ("We proposed the M2000.", "We proposed the M2000"),
        ("The 7020 currently has a GTX 960.", "the 7020 currently has a GTX 960"),
    ):
        memory = by_content[content]
        assert memory["evidence"][0]["event_id"] == event.event_id
        assert memory["evidence"][0]["span"] == _span(phrase)

    runs = engine.extraction_runs(event_id=event.event_id)
    assert len(runs) == 1 and runs[0]["status"] == "succeeded"
    assert runs[0]["model_id"] == "recorded-model"

    # Replay is free and idempotent: no second backend call, no duplicate memory.
    replay = engine.extract_event(event.event_id, extractor=extractor)
    assert replay.status == "replayed"
    assert set(replay.accepted) == set(result.accepted)
    assert backend.calls == 1
    assert len(engine.memories()) == 4

    # Malformed output fails the run and leaves the source event untouched.
    broken_event = _capture(engine, "A second observation.", event_id="turn-2")
    broken = GenerativeExtractor(RecordedBackend(["this is not json"]), timeout=5)
    failed = engine.extract_event(broken_event.event_id, extractor=broken)
    assert failed.status == "failed"
    assert engine.get_event(broken_event.event_id) is not None
    assert len(engine.memories()) == 4
    assert engine.extraction_runs(event_id=broken_event.event_id)[0]["status"] == "failed"

    # A timeout is a failure, not an empty extraction.
    timeout_event = _capture(engine, "A third observation.", event_id="turn-3")
    timing_out = GenerativeExtractor(RecordedBackend([BackendTimeout()]), timeout=1)
    timed_out = engine.extract_event(timeout_event.event_id, extractor=timing_out)
    assert timed_out.status == "failed"
    assert timed_out.diagnostics[0].code == "extraction.failed"

    # The polarity check rejects both observed live failures: a denial restated
    # as an assertion, and a quote the model invented.
    guard_event = _capture(
        engine, "I don't think the NEMA17 will hold. The camera is light.",
        event_id="turn-4",
    )
    guarded = GenerativeExtractor(RecordedBackend([json.dumps([
        {
            "content": "The NEMA17 will hold.",
            "quote": "I don't think the NEMA17 will hold",
            "kind": "semantic", "epistemic": "belief", "span": [0, 36], "negated": False,
        },
        {
            "content": "The camera is heavy.",
            "quote": "The camera is heavy",
            "kind": "semantic", "epistemic": "fact", "span": [0, 21],
        },
    ])]), timeout=5)
    guarded_result = engine.extract_event(guard_event.event_id, extractor=guarded)
    assert guarded_result.status == "succeeded"
    assert guarded_result.accepted == ()
    assert {item.reason for item in guarded_result.rejected} == {
        "negation_not_marked", "quote_not_in_source", "negation_uncovered",
    }
    assert len(engine.memories()) == 4

    # A model that drops the denial entirely gets one targeted repair pass.
    repair_event = _capture(
        engine, "I doubt the NEMA17 will hold. The camera is light.", event_id="turn-5",
    )
    repair_backend = RecordedBackend([
        json.dumps([{
            "content": "The camera is light.", "quote": "The camera is light.",
            "kind": "semantic", "epistemic": "observation", "span": [37, 57],
        }]),
        json.dumps([{
            "content": "The NEMA17 may not hold.", "quote": "I doubt the NEMA17 will hold",
            "kind": "semantic", "epistemic": "belief", "subject": {"kind": "device", "id": "nema17"},
            "span": [0, 28], "negated": True,
        }]),
    ])
    repaired = engine.extract_event(
        repair_event.event_id, extractor=GenerativeExtractor(repair_backend, timeout=5)
    )
    assert repaired.status == "succeeded"
    assert len(repaired.accepted) == 2
    assert repair_backend.calls == 2
    run_detail = engine.extraction_runs(event_id=repair_event.event_id)[0]["detail"]
    assert run_detail["repair_calls"] == 1
    assert run_detail["negation_gaps_before"] == 1
    assert run_detail["negation_gaps_after"] == 0

    # If the repair also misses it, the gap is recorded and visible.
    unresolved_event = _capture(
        engine, "I doubt the NEMA17 will hold.", event_id="turn-6",
    )
    unresolved_backend = RecordedBackend([
        json.dumps([]),
        json.dumps([{
            "content": "The NEMA17 is fine.", "quote": "the NEMA17 is fine",
            "kind": "semantic", "epistemic": "reported", "span": [0, 20],
        }]),
    ])
    unresolved = engine.extract_event(
        unresolved_event.event_id, extractor=GenerativeExtractor(unresolved_backend, timeout=5)
    )
    assert unresolved.status == "succeeded"
    assert any(item.reason == "negation_uncovered" for item in unresolved.rejected)
    diagnostics = [item.code for item in engine.diagnostics.recent()]
    assert "extraction.negation_uncovered" in diagnostics

    # Typed tool events extract without a model at all.
    tool_event = _capture(
        engine, "scale reports 82 g", event_id="tool-1", source_kind="tool.reading",
        actor=Actor("kitchen-scale", ActorKind.TOOL),
        content_type="text/plain",
    )
    structured = engine.extract_event(tool_event.event_id, extractor=StructuredExtractor())
    assert structured.status == "succeeded"
    assert len(structured.accepted) == 1
    tool_memory = engine.get_memory(structured.accepted[0])
    assert tool_memory is not None
    assert ("epistemic", "observation") in _all_classifications(tool_memory)
    assert tool_memory["evidence"][0]["span"] == [0, len("scale reports 82 g")]

    # Structured claims supplied by a host are honoured, including rejection reasons.
    typed_event = _capture(
        engine, "payload 1.24 kg", event_id="tool-2", source_kind="tool.reading",
        actor=Actor("scale-a", ActorKind.TOOL),
        metadata={"claims": [
            {"content": "payload 1.24 kg", "kind": "semantic", "epistemic": "observation",
             "span": [0, 15]},
            {"content": "bad", "kind": "unknown", "epistemic": "fact", "span": [0, 3]},
        ]},
    )
    typed = engine.extract_event(typed_event.event_id)
    assert typed.status == "succeeded"
    assert len(typed.accepted) == 1
    assert [item.reason for item in typed.rejected] == ["unknown_kind"]

    engine.close()

    # Opt-in live generator: real backend, real response, recorded in evidence.
    if os.environ.get("SHIKO_LIVE") != "1":
        pytest.skip("live generator: set SHIKO_LIVE=1")
    backend = _live_backend()
    if backend is None:
        pytest.skip("live generator: no host key available")
    live_engine = _engine(tmp_path / "live")
    live_event = _capture(
        live_engine,
        "We proposed the M2000 for the wrist; the 7020 still has a GTX 960. "
        "Robin says the NEMA17 is fine, but I doubt it will hold.",
        event_id="live-1",
    )
    live = live_engine.extract_event(
        live_event.event_id,
        extractor=GenerativeExtractor(
            backend, timeout=float(os.environ.get("SHIKO_LLM_TIMEOUT", "180"))
        ),
    )
    assert live.status == "succeeded", live.to_dict()
    assert live.calls == 1
    assert live.accepted or live.rejected
    print("LIVE_RESULT:" + json.dumps({
        "model": live.model_id, "accepted": len(live.accepted),
        "rejected": [item.reason for item in live.rejected],
        "memories": [_content(m)[:80] for m in live_engine.memories()],
    }))
    live_engine.close()


# -- 3.2 ---------------------------------------------------------------------


def test_identity_and_evidence_lineage(tmp_path):
    """Phase 3.2 — roots, delegation and ambiguous entity names survive."""
    engine = _engine(tmp_path)
    report = engine.capture_text(
        "measured 82 g", event_id="rep-1", session_id="parent-session",
        source_kind="subagent.report", actor=Actor("helper-1", ActorKind.SUBAGENT),
        delegation_parent="parent-session", metadata={"independence_root": "root-helper"},
    )
    echo = engine.capture_text(
        "the helper measured 82 g", event_id="echo-1", session_id="parent-session",
        source_kind="assistant.message", actor=Actor("jiko", ActorKind.OWNER),
        metadata={"independence_root": "root-helper"},
    )
    tool = engine.capture_text(
        "scale reading 82.1 g", event_id="tool-1", session_id="parent-session",
        source_kind="tool.reading", actor=Actor("kitchen-scale", ActorKind.TOOL),
    )
    assert report.ok and echo.ok and tool.ok

    def memory(content: str, event_id: str, **kwargs):
        return engine.create_memory(MemoryDraft(
            kind="semantic", content=content,
            evidence=(EvidenceInput(event_id),), **kwargs,
        ))

    measured = memory("82 g measured by the helper", "rep-1")
    paraphrase = memory("the helper measured 82 g", "echo-1")
    independent = memory("82.1 g on the kitchen scale", "tool-1")
    roots = {
        item["memory_id"]: item["evidence"][0]["independence_root"]
        for item in (measured, paraphrase, independent)
    }
    assert roots[measured["memory_id"]] == roots[paraphrase["memory_id"]] == "root-helper"
    assert roots[independent["memory_id"]] == "tool-1"

    stored = engine.get_event("rep-1")
    assert stored is not None
    assert stored["actor_id"] == "helper-1"
    assert stored["actor_kind"] == "subagent"
    assert stored["delegation_parent"] == "parent-session"

    engine.note_evidence_root("rep-1", root_id="root-helper", kind="observation")
    engine.note_evidence_root("echo-1", root_id="root-helper", kind="echo", parent_root_id="root-helper")
    recorded = engine.evidence_roots()
    assert len(recorded) == 1
    assert recorded[0]["root_id"] == "root-helper" and recorded[0]["kind"] == "echo"

    derived = engine.derive_memory(
        [measured["memory_id"], independent["memory_id"]],
        content="Two independent sources agree near 82 g.",
        method="summary",
    )
    assert {item["parent_memory_id"] for item in derived["derived_from"]} == {
        measured["memory_id"], independent["memory_id"],
    }
    assert {item["independence_root"] for item in derived["evidence"]} == {"root-helper", "tool-1"}
    assert derived["metadata"]["derived"] is True
    assert len(engine.get_memory(measured["memory_id"])["revisions"]) == 1

    # Two devices share a nickname: the resolver returns both instead of merging.
    first = engine.register_entity("device", "wrist camera", entity_id="cam-a", aliases=("wrist cam",))
    second = engine.register_entity("device", "wrist camera", entity_id="cam-b")
    assert first["entity_id"] == "cam-a" and second["entity_id"] == "cam-b"
    resolution = engine.resolve_entity("Wrist Camera")
    assert resolution["ambiguous"] is True
    assert {item["entity_id"] for item in resolution["matches"]} == {"cam-a", "cam-b"}
    alias = engine.resolve_entity("wrist cam")
    assert alias["ambiguous"] is False
    assert alias["matches"] and alias["matches"][0]["entity_id"] == "cam-a"
    engine.link_alias("camera", "cam-b", source_event_id="tool-1")
    assert {item["entity_id"] for item in engine.resolve_entity("camera")["matches"]} == {"cam-b"}

    camera_a = memory("camera A has a 6 mm lens", "rep-1", subject_kind="device", subject_id="cam-a")
    camera_b = memory("camera B has an 8 mm lens", "echo-1", subject_kind="device", subject_id="cam-b")
    assert engine.entities_for_memory(camera_a["memory_id"])[0]["entity_id"] == "cam-a"
    assert engine.entities_for_memory(camera_b["memory_id"])[0]["entity_id"] == "cam-b"
    assert camera_a["memory_id"] != camera_b["memory_id"]
    engine.close()


# -- 3.3 ---------------------------------------------------------------------


def test_episodes_across_turns_and_sessions(tmp_path):
    """Phase 3.3 — ordered episodes, two experiments, delayed outcomes."""
    from datetime import datetime, timezone

    engine = _engine(tmp_path)
    started = datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc)
    gear = _capture(engine, "changed the gearing to 3:1", event_id="gear-1", session="s1")
    action_a = _capture(engine, "run the load test", event_id="act-a", session="s1",
                        source_kind="tool.call", actor=Actor("jiko", ActorKind.OWNER))
    action_b = _capture(engine, "run the pulley comparison", event_id="act-b", session="s1",
                        source_kind="tool.call", actor=Actor("jiko", ActorKind.OWNER))
    outcome_b = _capture(engine, "pulley comparison passed", event_id="out-b", session="s1",
                         source_kind="tool.result", actor=Actor("jiko", ActorKind.OWNER))

    first = engine.start_episode(
        "Gearing change and load test", goal="verify the wrist camera mount",
        participants=("robin", "jiko"), tags=("arm", "hardware"),
        session_id="s1", at=started,
    )
    second = engine.start_episode("Pulley comparison", goal="compare pulleys", session_id="s1", at=started)
    first_id, second_id = first["episode_id"], second["episode_id"]

    observation = engine.add_episode_step(
        first_id, "observation", "gearing changed to 3:1",
        event_id=gear.event_id, session_id="s1", scope="user", actor_id="robin",
    )
    step_a1 = engine.add_episode_step(
        first_id, "action", "run the load test", event_id=action_a.event_id,
        session_id="s1", actor_id="jiko", scope="shared",
    )
    step_a2 = engine.add_episode_step(
        second_id, "action", "run the pulley comparison", event_id=action_b.event_id,
        session_id="s1", actor_id="jiko",
    )
    reflection = engine.add_episode_step(
        first_id, "reflection", "the mount may flex under load", actor_id="jiko",
    )

    correlation_a = step_a1["actions"][0]["correlation_id"]
    correlation_b = step_a2["actions"][0]["correlation_id"]
    assert correlation_a != correlation_b
    assert step_a1["steps"][0]["scope"] == "user"
    assert step_a1["steps"][0]["actor_id"] == "robin"
    assert reflection["steps"][-1]["role"] == "reflection"
    assert reflection["steps"][-1]["event_id"] is None

    # The pulley comparison finishes immediately; the load test stays pending.
    second_after = engine.record_episode_outcome(
        second_id, correlation_b, "pulley comparison passed",
        outcome="success", event_id=outcome_b.event_id, session_id="s1",
    )
    assert second_after["status"] == "completed"
    assert second_after["actions"][0]["status"] == "completed"

    pending = engine.get_episode(first_id)
    assert pending is not None
    assert pending["status"] == "open"
    assert pending["actions"][0]["status"] == "pending"
    assert [item["role"] for item in pending["steps"]] == ["observation", "action", "reflection"]
    assert pending["actions"][0]["outcome_step_id"] is None
    assert pending["ended_at"] is None

    # Session ends, the engine reopens, and the result arrives in a new session.
    engine.close()
    engine = _engine(tmp_path)
    outcome_a = _capture(engine, "load test completed", event_id="out-a", session="s2",
                         source_kind="tool.result", actor=Actor("jiko", ActorKind.OWNER))
    completed = engine.record_episode_outcome(
        first_id, correlation_a, "load test completed at 82 g",
        outcome="success", event_id=outcome_a.event_id, session_id="s2",
    )
    assert completed["status"] == "completed"
    assert completed["actions"][0]["status"] == "completed"
    assert completed["steps"][-1]["session_id"] == "s2"
    assert completed["duration_seconds"] is not None and completed["duration_seconds"] > 0
    assert [item["sequence"] for item in completed["steps"]] == [1, 2, 3, 4]
    assert [item["role"] for item in completed["steps"]] == [
        "observation", "action", "reflection", "outcome",
    ]
    assert completed["participants"] == ["robin", "jiko"]
    assert completed["tags"] == ["arm", "hardware"]

    # One source event informed two memory types without cloning the event.
    gear_memory = engine.create_memory(MemoryDraft(
        kind="episodic", content="We changed the arm gearing to 3:1.",
        evidence=(EvidenceInput(gear.event_id),),
    ))
    semantic = engine.create_memory(MemoryDraft(
        kind="semantic", content="The arm uses a 3:1 gearing.",
        evidence=(EvidenceInput(gear.event_id),),
    ))
    assert gear_memory["memory_id"] != semantic["memory_id"]
    assert gear_memory["evidence"][0]["event_id"] == semantic["evidence"][0]["event_id"] == gear.event_id
    engine.link_episode_memory(first_id, gear_memory["memory_id"], role="describes")
    engine.link_episode_memory(first_id, semantic["memory_id"], role="derives")
    linked = engine.get_episode(first_id)
    assert linked is not None
    assert {item["memory_id"] for item in linked["memories"]} == {
        gear_memory["memory_id"], semantic["memory_id"],
    }

    # A missing result stays explicitly pending.
    open_episode = engine.start_episode("Unfinished test", goal="measure flex", session_id="s3")
    open_step = engine.add_episode_step(
        open_episode["episode_id"], "action", "apply 2 kg and measure",
        event_id=None, session_id="s3",
    )
    assert open_step["status"] == "open"
    assert open_step["actions"][0]["status"] == "pending"
    assert open_step["actions"][0]["outcome_step_id"] is None
    assert open_step["steps"][0]["status"] == "pending"
    open_view = engine.get_episode(open_episode["episode_id"])
    assert open_view is not None and open_view["steps"][0]["outcome"] is None

    # A failed attempt is distinct from a missing one.
    failed_episode = engine.start_episode("Failed attempt", session_id="s3")
    failed_step = engine.add_episode_step(
        failed_episode["episode_id"], "action", "tighten the set screw",
    )
    failed = engine.record_episode_outcome(
        failed_episode["episode_id"], failed_step["actions"][0]["correlation_id"],
        "the screw stripped", outcome="failure",
    )
    assert failed["status"] == "failed"
    assert failed["actions"][0]["status"] == "failed"

    engine.close()
    reopened = _engine(tmp_path)
    replay = reopened.get_episode(first_id)
    assert replay is not None
    assert replay["status"] == "completed"
    assert [item["step_id"] for item in replay["steps"]] == [
        item["step_id"] for item in completed["steps"]
    ]
    assert len(reopened.episodes()) == 4
    assert [item["episode_id"] for item in reopened.episodes(status="open")] == [open_episode["episode_id"]]
    reopened.close()
