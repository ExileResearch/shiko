"""Completion backends for generative extraction.

The core never reads a host's credential store. A backend is injected: a
recorded list for deterministic tests, or an OpenAI-compatible HTTP client
configured from ``SHIKO_LLM_*`` environment variables. Keys are never logged or
included in diagnostics.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


class BackendError(RuntimeError):
    """A completion backend could not produce a response."""

    def __init__(self, message: str, *, code: str = "backend.error") -> None:
        super().__init__(message)
        self.code = code


class BackendTimeout(BackendError):
    def __init__(self, message: str = "completion timed out") -> None:
        super().__init__(message, code="backend.timeout")


class CompletionBackend(Protocol):
    name: str
    model_id: str
    calls: int

    def complete(self, prompt: str, *, timeout: float = 30.0) -> str: ...


@dataclass
class RecordedBackend:
    """Replay scripted responses; an ``Exception`` entry is raised instead."""

    responses: list[str | Exception]
    name: str = "recorded"
    model_id: str = "recorded-model"
    calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str, *, timeout: float = 30.0) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if not self.responses:
            raise BackendError("no recorded response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@dataclass
class OpenAIBackend:
    """Minimal OpenAI-compatible chat client (stdlib only)."""

    base_url: str
    api_key: str
    model_id: str
    temperature: float = 0.6
    max_tokens: int = 900
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    reasoning_effort: str | None = None
    name: str = "openai-compatible"
    calls: int = 0
    tokens: int | None = None
    timeout: float = 90.0

    def complete(self, prompt: str, *, timeout: float | None = None) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        payload.update(self.extra_body)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "shiko-memory/0.1 (+https://github.com/exileresearch/shiko)",
                "Authorization": f"Bearer {self.api_key}",
                **self.extra_headers,
            },
            method="POST",
        )
        self.calls += 1
        effective_timeout = self.timeout if timeout is None else timeout
        try:
            with urllib.request.urlopen(request, timeout=effective_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise BackendError(f"backend HTTP {exc.code}: {detail}", code="backend.http") from exc
        except TimeoutError as exc:
            raise BackendTimeout() from exc
        except urllib.error.URLError as exc:
            if "timed out" in str(exc).lower():
                raise BackendTimeout() from exc
            raise BackendError(f"backend unreachable: {exc.reason}", code="backend.unreachable") from exc
        except ValueError as exc:
            raise BackendError(f"backend returned invalid JSON: {exc}", code="backend.malformed") from exc
        try:
            choice = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("backend response has no message content", code="backend.malformed") from exc
        usage = data.get("usage") or {}
        if isinstance(usage.get("total_tokens"), int):
            self.tokens = usage["total_tokens"]
        if not isinstance(choice, str) or not choice.strip():
            raise BackendError(
                "backend returned empty content (a reasoning model may have spent the "
                "completion budget; raise max_tokens or disable thinking)",
                code="backend.empty",
            )
        return choice


def backend_from_env() -> OpenAIBackend | None:
    """Build a backend from ``SHIKO_LLM_*``/config; ``None`` when incomplete."""
    from .llm_config import ConfigError, build_backend, load_settings

    try:
        settings = load_settings()
        if not settings.api_key:
            return None
        return build_backend(settings)
    except ConfigError:
        return None
