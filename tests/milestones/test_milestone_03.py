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
    """Read the host's key for an opt-in live run; never printed or logged."""
    env_file = Path.home() / ".hermes" / ".env"
    key = ""
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENCODE_GO_API_KEY="):
                key = line.split("=", 1)[1].strip()
                break
    if not key:
        return None
    return OpenAIBackend(
        base_url=os.environ.get("SHIKO_LLM_BASE_URL", "https://opencode.ai/zen/go/v1"),
        api_key=key,
        model_id=os.environ.get("SHIKO_LLM_MODEL", "qwen3.8-flash"),
        temperature=0.6,
        max_tokens=1600,
        extra_headers={"x-opencode-session": "shiko-live-extraction"},
    )


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
            "kind": "semantic", "epistemic": "belief", "modality": "text", "scope": "internal",
            "subject": {"kind": "device", "id": "nema17"},
            "span": _span("I don't think the NEMA17 will be strong enough."),
            "negated": True, "confidence": 0.7,
        },
        {
            "content": "The NEMA17 is fine.",
            "kind": "semantic", "epistemic": "reported", "modality": "text", "scope": "user",
            "subject": {"kind": "person", "id": "robin"},
            "span": _span("the NEMA17 is fine"),
            "quoted_speaker": "Robin", "confidence": 0.9,
        },
        {
            "content": "We proposed the M2000.",
            "kind": "semantic", "epistemic": "hypothesis", "modality": "text", "scope": "shared",
            "subject": {"kind": "device", "id": "m2000"},
            "span": _span("We proposed the M2000"), "confidence": 0.8,
        },
        {
            "content": "The 7020 currently has a GTX 960.",
            "kind": "semantic", "epistemic": "fact", "modality": "text", "scope": "shared",
            "subject": {"kind": "device", "id": "7020"},
            "span": _span("the 7020 currently has a GTX 960"), "confidence": 0.95,
        },
        {
            "content": "This span is deliberately outside the source.",
            "kind": "semantic", "epistemic": "fact", "modality": "text", "scope": "internal",
            "span": [9999, 10050],
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
    assert [item.reason for item in result.rejected] == ["span_out_of_range"]
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
        live_event.event_id, extractor=GenerativeExtractor(backend, timeout=90)
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
