"""Hermes memory-provider adapter.

Host-specific behaviour lives here and nowhere else. The adapter implements the
``MemoryProvider`` lifecycle observed on Hermes 0.23.4 (see
``agent/memory_provider.py`` in the host): initialize, prefetch, queue_prefetch,
sync_turn, on_pre_compress, on_session_end, on_memory_write, on_delegation,
on_session_switch and shutdown.

Two rules shape the implementation:

* Importing this module must work without Hermes. The host base class is
  optional, so a plain Python environment can load the adapter for inspection
  and tests.
* A host callback never raises. Every failure becomes a diagnostic on the
  deployment and on ``last_error``, because a memory plugin that breaks a turn
  is worse than one that misses a turn.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from ..diagnostics import Diagnostics
from ..engine import Engine, EngineConfig, resolve_owner_id
from ..envelope import Actor, ActorKind
from ..errors import ShikoError
from ..results import Diagnostic, Severity
from ..owner import slugify_owner

logger = logging.getLogger("shiko.hermes")

try:  # pragma: no cover - depends on the host being importable
    from agent.memory_provider import MemoryProvider as _MemoryProviderBase

    HOST_CONTRACT = True
except Exception:  # ImportError, or a host module that is not importable yet
    _MemoryProviderBase = object
    HOST_CONTRACT = False

_DISABLED_VALUES = {"1", "true", "yes", "on"}
_NON_PRIMARY_CONTEXTS = {"subagent", "cron", "flush", "auxiliary"}


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _DISABLED_VALUES


def _identity_name(identity: Any) -> str:
    if isinstance(identity, str):
        return identity.strip()
    if isinstance(identity, dict):
        for key in ("name", "id", "profile", "profile_name"):
            value = identity.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


class ShikoProvider(_MemoryProviderBase):  # type: ignore[misc, valid-type]
    """One Hermes agent's Shiko engine."""

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._root: Path | None = None
        self._session_id = ""
        self._platform = ""
        self._primary = True
        self._turn_index = 0
        self._initialized = False
        self._last_error: Diagnostic | None = None
        self._owner_hint = ""
        self._closed = False

    # -- identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "shiko"

    @property
    def engine(self) -> Engine | None:
        return self._engine

    @property
    def last_error(self) -> Diagnostic | None:
        return self._last_error

    @property
    def primary_context(self) -> bool:
        return self._primary

    def is_available(self) -> bool:
        """Cheap readiness check: configuration only, no directory or network work."""
        return not _truthy("SHIKO_DISABLED")

    def unavailable_reason(self) -> str:
        if _truthy("SHIKO_DISABLED"):
            return "SHIKO_DISABLED is set; unset it to activate Shiko."
        return ""

    def identity_signature(self) -> dict[str, Any]:
        return {"shiko.owner_hint": os.environ.get("SHIKO_OWNER", "").strip()}

    # -- lifecycle ----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = str(session_id or kwargs.get("session_id") or "hermes")
        self._platform = str(kwargs.get("platform") or "")
        context = str(kwargs.get("agent_context") or "primary")
        self._primary = context not in _NON_PRIMARY_CONTEXTS
        hermes_home = str(kwargs.get("hermes_home") or os.environ.get("HERMES_HOME") or "")
        self._owner_hint = (
            os.environ.get("SHIKO_OWNER", "").strip()
            or _identity_name(kwargs.get("agent_identity"))
            or (Path(hermes_home).name if hermes_home else "")
        )

        explicit_root = os.environ.get("SHIKO_HOME", "").strip()
        root = Path(explicit_root).expanduser() if explicit_root else (
            Path(hermes_home).expanduser() / "shiko" if hermes_home else None
        )
        owner_id = resolve_owner_id(self._owner_hint or None, root=root)
        try:
            self._engine = Engine.open(
                root,
                owner_id,
                display_name=kwargs.get("agent_name") or self._owner_hint,
                config=EngineConfig(log_diagnostics=True),
            )
            self._root = self._engine.root
            self._initialized = True
            self._emit(
                Diagnostic(
                    "adapter.initialized",
                    "Shiko engine bound to Hermes session",
                    Severity.INFO,
                    {
                        "session_id": self._session_id,
                        "root": str(self._root),
                        "owner_id": self._engine.owner_id,
                        "agent_context": context,
                        "primary": self._primary,
                        "host_contract": HOST_CONTRACT,
                    },
                )
            )
        except (ShikoError, OSError) as exc:
            self._initialized = False
            self._report(
                "adapter.init_failed",
                f"Shiko could not open its data root: {exc}",
                {"root": str(root) if root else "", "owner_hint": self._owner_hint},
                Severity.ERROR,
            )

    def on_session_switch(
        self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
        rewound: bool = False, **kwargs,
    ) -> None:
        previous = self._session_id
        self._session_id = str(new_session_id or previous)
        if reset:
            self._turn_index = 0
        self._emit(
            Diagnostic(
                "adapter.session_switch",
                "session rebound",
                Severity.INFO,
                {
                    "previous": previous,
                    "new": self._session_id,
                    "parent": parent_session_id,
                    "reset": reset,
                    "rewound": rewound,
                },
            )
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_index = max(self._turn_index, int(turn_number or 0))

    # -- recall (milestone 4) ----------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def recall_status(self):
        return None

    def system_prompt_block(self) -> str:
        return ""

    # -- capture ------------------------------------------------------------

    def sync_turn(
        self, user_content: str, assistant_content: str, *,
        session_id: str = "", messages: list[dict[str, Any]] | None = None,
        turn_author: dict[str, Any] | None = None,
    ) -> None:
        session = session_id or self._session_id
        self._turn_index += 1
        message_count = len(messages) if messages else 0
        if user_content:
            self._capture(
                user_content,
                actor=self._actor_for_speaker(turn_author),
                session_id=session,
                source_kind="hermes.turn.user",
                metadata={"turn_index": self._turn_index, "role": "user", "message_count": message_count, "platform": self._platform},
            )
        if assistant_content:
            self._capture(
                assistant_content,
                actor=Actor(id=self._owner_id(), kind=ActorKind.OWNER, display_name="assistant"),
                session_id=session,
                source_kind="hermes.turn.assistant",
                metadata={"turn_index": self._turn_index, "role": "assistant", "message_count": message_count, "platform": self._platform},
            )

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        if not result:
            return
        self._capture(
            result,
            actor=Actor(
                id=child_session_id or "subagent",
                kind=ActorKind.SUBAGENT,
                display_name="subagent",
            ),
            session_id=self._session_id,
            source_kind="hermes.delegation.report",
            delegation_parent=self._session_id,
            metadata={"task": task, "child_session_id": child_session_id},
        )

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        roles = [str((m or {}).get("role", "")) for m in (messages or [])]
        self._capture(
            "",
            actor=Actor(id=self._owner_id(), kind=ActorKind.SYSTEM, display_name="hermes"),
            session_id=self._session_id,
            source_kind="hermes.compaction",
            metadata={
                "message_count": len(messages or []),
                "roles": roles,
                "require_checkpoint": bool((messages or []) and isinstance(messages, list)),
            },
        )
        return ""

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        self._capture(
            "",
            actor=Actor(id=self._owner_id(), kind=ActorKind.SYSTEM, display_name="hermes"),
            session_id=self._session_id,
            source_kind="hermes.session_end",
            metadata={"message_count": len(messages or [])},
        )

    def on_memory_write(
        self, action: str, target: str, content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not content:
            return
        self._capture(
            content,
            actor=Actor(id=self._owner_id(), kind=ActorKind.OWNER, display_name="native-memory"),
            session_id=self._session_id,
            source_kind="hermes.native_memory",
            metadata={"action": action, "target": target, **(metadata or {})},
        )

    # -- tools and teardown -------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        # Memory arrives through automatic recall; explicit tools come later.
        return []

    def backup_paths(self) -> list[str]:
        # The root lives inside HERMES_HOME, so `hermes backup` already covers it.
        return []

    def shutdown(self) -> None:
        self._closed = True
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception as exc:  # never raise into host teardown
                self._report("adapter.shutdown_failed", str(exc), severity=Severity.WARNING)

    # -- internals ----------------------------------------------------------

    def _owner_id(self) -> str:
        return self._engine.owner_id if self._engine else slugify_owner(self._owner_hint or "hermes")

    def _actor_for_speaker(self, turn_author: dict[str, Any] | None) -> Actor:
        if not turn_author:
            return Actor(id=self._owner_id(), kind=ActorKind.OWNER, display_name="user")
        author_id = str(turn_author.get("id") or turn_author.get("name") or "user")
        kind = ActorKind.EXTERNAL if turn_author.get("is_bot") else ActorKind.USER
        return Actor(id=author_id, kind=kind, display_name=str(turn_author.get("name") or author_id))

    def _capture(
        self,
        content: str,
        *,
        actor: Actor,
        session_id: str,
        source_kind: str,
        delegation_parent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if not self._primary:
            return None
        if self._engine is None:
            if not self._initialized:
                self._report(
                    "adapter.not_initialized",
                    "capture skipped: the Shiko engine is not open",
                    {"source_kind": source_kind},
                    Severity.WARNING,
                )
            return None
        try:
            result = self._engine.capture_text(
                content,
                event_id=f"hermes:{uuid.uuid4().hex}",
                actor=actor,
                session_id=session_id or self._session_id,
                source_kind=source_kind,
                delegation_parent=delegation_parent,
                metadata=metadata or {},
            )
        except (ShikoError, OSError) as exc:
            self._report(
                "adapter.capture_failed",
                f"Shiko capture failed: {exc}",
                {"source_kind": source_kind},
                Severity.ERROR,
            )
            return None
        for diagnostic in result.diagnostics:
            if diagnostic.severity is Severity.ERROR:
                self._last_error = diagnostic
        return result

    def _emit(self, diagnostic: Diagnostic) -> None:
        if self._engine is not None:
            try:
                self._engine.diagnostics.emit(diagnostic)
                return
            except Exception:
                pass
        logger.debug("%s: %s", diagnostic.code, diagnostic.message)

    def _report(
        self, code: str, message: str, detail: dict[str, Any] | None = None,
        severity: Severity = Severity.WARNING,
    ) -> Diagnostic:
        diagnostic = Diagnostic(code, message, severity, detail or {})
        self._last_error = diagnostic
        self._emit(diagnostic)
        logger.warning("%s: %s", code, message)
        return diagnostic


def provider() -> ShikoProvider:
    """Entry-point factory (``hermes_agent.memory_providers``)."""
    return ShikoProvider()


class HookFallback:
    """Best-effort capture for hosts without ``register_memory_provider``.

    This path is deliberately conservative: it captures only text the hook
    payload actually exposes and never transforms the turn. Hermes 0.23.4 and
    later use the provider API above, so this exists for reduced hosts and for
    the 0.24 transition.
    """

    def __init__(self) -> None:
        self.hooks: tuple[str, ...] = ()
        self.engine: Engine | None = None
        self.last_error: Diagnostic | None = None
        self._session_id = "hermes-hooks"
        self._diagnostics: Diagnostics | None = None

    def _ensure(self) -> Engine | None:
        if self.engine is not None:
            return self.engine
        try:
            hermes_home = os.environ.get("HERMES_HOME", "").strip()
            root = Path(os.environ["SHIKO_HOME"]).expanduser() if os.environ.get("SHIKO_HOME") else (
                Path(hermes_home).expanduser() / "shiko" if hermes_home else None
            )
            self.engine = Engine.open(
                root,
                resolve_owner_id(os.environ.get("SHIKO_OWNER", "").strip() or None, root=root),
            )
            self._diagnostics = self.engine.diagnostics
            self._diagnostics.emit(
                Diagnostic(
                    "adapter.hook_fallback",
                    "host has no memory-provider API; capturing through hooks",
                    Severity.WARNING,
                    {"hooks": list(self.hooks)},
                )
            )
        except (ShikoError, OSError) as exc:
            self.last_error = Diagnostic("adapter.init_failed", str(exc), Severity.ERROR, {})
            logger.warning("Shiko hook fallback could not open its root: %s", exc)
        return self.engine

    def _text(self, payload: dict[str, Any], *names: str) -> str:
        for name in names:
            value = payload.get(name)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def on_post_llm_call(self, **kwargs) -> None:
        engine = self._ensure()
        if engine is None:
            return
        session = str(kwargs.get("session_id") or self._session_id)
        user_text = self._text(kwargs, "user_message", "user_content", "prompt", "query")
        assistant_text = self._text(kwargs, "response_text", "assistant_content", "content", "text")
        try:
            if user_text:
                engine.capture_text(user_text, session_id=session, source_kind="hermes.hook.user")
            if assistant_text:
                engine.capture_text(
                    assistant_text,
                    actor=Actor(id=engine.owner_id, kind=ActorKind.OWNER, display_name="assistant"),
                    session_id=session,
                    source_kind="hermes.hook.assistant",
                )
        except (ShikoError, OSError) as exc:
            self.last_error = Diagnostic("adapter.capture_failed", str(exc), Severity.ERROR, {})
            logger.warning("Shiko hook capture failed: %s", exc)

    def on_session_end(self, **kwargs) -> None:
        engine = self.engine or self._ensure()
        if engine is not None:
            try:
                engine.close()
            finally:
                self.engine = None


def register_hook_fallback(ctx: Any) -> HookFallback:
    """Register the conservative hook path and report what actually attached."""
    fallback = HookFallback()
    registered: list[str] = []
    for hook_name, callback in (
        ("post_llm_call", fallback.on_post_llm_call),
        ("on_session_end", fallback.on_session_end),
    ):
        register = getattr(ctx, "register_hook", None)
        if register is None:
            continue
        try:
            register(hook_name, callback)
            registered.append(hook_name)
        except Exception as exc:  # pragma: no cover - host-specific refusal
            logger.warning("Shiko could not register %s: %s", hook_name, exc)
    fallback.hooks = tuple(registered)
    if not registered:
        fallback.last_error = Diagnostic(
            "adapter.no_host_surface",
            "host exposes neither register_memory_provider nor usable hooks",
            Severity.ERROR,
            {},
        )
    return fallback
