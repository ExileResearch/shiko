"""Hermes lifecycle probe for Shiko milestone 1.3.

Installed into a throwaway HERMES_HOME by the milestone test, never shipped. It
implements the memory-provider contract and registers observer hooks, appending
one JSON line per observed event to ``$SHIKO_PROBE_LOG``. Payloads record shape,
not content: signatures, keys and counts are what the adapter mapping needs.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time

_LOCK = threading.Lock()
_SEQ = 0
_LOG = os.environ.get("SHIKO_PROBE_LOG", "")
_HOOKS_REGISTERED = False

HOOKS = (
    "on_session_open",
    "on_session_start",
    "pre_llm_call",
    "post_llm_call",
    "pre_tool_call",
    "post_tool_call",
    "post_compaction",
    "on_session_end",
    "on_session_finalize",
)


def _record(event: str, payload: dict | None = None) -> None:
    global _SEQ
    if not _LOG:
        return
    with _LOCK:
        _SEQ += 1
        record = {
            "seq": _SEQ,
            "ts": time.time(),
            "mono": time.monotonic(),
            "pid": os.getpid(),
            "event": event,
            "payload": payload or {},
        }
        line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        fd = os.open(_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)


def _describe(value, *, depth: int = 0):
    """Shape without content, so a trace is safe to share."""
    if depth > 2:
        return type(value).__name__
    if isinstance(value, str):
        return {"type": "str", "len": len(value), "sha256_12": hashlib.sha256(value.encode()).hexdigest()[:12]}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(str(k) for k in value.keys())}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "len": len(value)}
    return {"type": type(value).__name__}


def _summary(kwargs: dict) -> dict:
    payload = {"keys": sorted(kwargs.keys())}
    for key in ("session_id", "parent_session_id", "platform", "tool_name", "turn_number",
                "agent_context", "message_count", "resumed", "caller"):
        if key in kwargs:
            payload[key] = _describe(kwargs[key])
    for key in ("messages", "history", "user_message", "response_text", "prompt", "text", "result"):
        if key in kwargs:
            payload[key] = _describe(kwargs[key])
    return payload


try:  # the probe only ever runs inside Hermes
    from agent.memory_provider import MemoryProvider as _Base
except Exception:  # pragma: no cover - defensive
    _Base = object


class ProbeProvider(_Base):  # type: ignore[misc, valid-type]
    """Logs every provider callback with its signature and timing."""

    def __init__(self) -> None:
        self._session_id = ""

    @property
    def name(self) -> str:
        return "shiko-probe"

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = str(session_id or "")
        _record("provider.initialize", {"session_id": session_id, **_summary(kwargs)})

    def system_prompt_block(self) -> str:
        _record("provider.system_prompt_block")
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        _record("provider.prefetch", {"session_id": session_id, "query": _describe(query)})
        # A recognisable marker proves the host places prefetch output in the
        # next model request; the driver checks the recorded request bodies.
        return "SHIKO-PROBE-RECALL-SENTINEL"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        _record("provider.queue_prefetch", {"session_id": session_id, "query": _describe(query)})

    def recall_status(self):
        _record("provider.recall_status")
        return None

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None, turn_author=None) -> None:
        _record("provider.sync_turn", {
            "session_id": session_id,
            "user": _describe(user_content),
            "assistant": _describe(assistant_content),
            "messages": _describe(messages),
            "turn_author": _describe(turn_author),
        })

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        _record("provider.on_turn_start", {"turn_number": turn_number, "message": _describe(message), **_summary(kwargs)})

    def on_session_end(self, messages) -> None:
        _record("provider.on_session_end", {"messages": _describe(messages)})

    def on_session_switch(self, new_session_id, *, parent_session_id="", reset=False, rewound=False, **kwargs) -> None:
        _record("provider.on_session_switch", {
            "new_session_id": new_session_id, "parent_session_id": parent_session_id,
            "reset": reset, "rewound": rewound, **_summary(kwargs),
        })

    def on_pre_compress(self, messages) -> str:
        _record("provider.on_pre_compress", {"messages": _describe(messages)})
        return ""

    def on_delegation(self, task, result, *, child_session_id="", **kwargs) -> None:
        _record("provider.on_delegation", {
            "child_session_id": child_session_id, "task": _describe(task), "result": _describe(result),
            **_summary(kwargs),
        })

    def on_memory_write(self, action, target, content, metadata=None) -> None:
        _record("provider.on_memory_write", {
            "action": action, "target": target, "content": _describe(content), "metadata": _describe(metadata),
        })

    def get_tool_schemas(self):
        _record("provider.get_tool_schemas")
        return []

    def shutdown(self) -> None:
        _record("provider.shutdown")


def _make_hook(name: str):
    def callback(**kwargs):
        _record(f"hook.{name}", _summary(kwargs))
        return None

    callback.__name__ = f"probe_{name}"
    return callback


def register(ctx):
    global _HOOKS_REGISTERED
    _record("plugin.register", {"ctx": type(ctx).__name__})
    provider = ProbeProvider()
    register_memory_provider = getattr(ctx, "register_memory_provider", None)
    if register_memory_provider is not None:
        try:
            register_memory_provider(provider)
            _record("plugin.provider_registered")
        except Exception as exc:  # pragma: no cover - host specific
            _record("plugin.provider_refused", {"error": str(exc)})
    if not _HOOKS_REGISTERED:
        _HOOKS_REGISTERED = True
        for hook in HOOKS:
            try:
                ctx.register_hook(hook, _make_hook(hook))
                _record("plugin.hook_registered", {"hook": hook})
            except Exception as exc:  # pragma: no cover - host specific
                _record("plugin.hook_failed", {"hook": hook, "error": str(exc)})
    return provider
