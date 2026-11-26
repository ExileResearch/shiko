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
from typing import Protocol


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
    name: str = "openai-compatible"
    calls: int = 0
    tokens: int | None = None

    def complete(self, prompt: str, *, timeout: float = 30.0) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps({
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
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
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
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
            choice = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("backend response has no message content", code="backend.malformed") from exc
        usage = payload.get("usage") or {}
        if isinstance(usage.get("total_tokens"), int):
            self.tokens = usage["total_tokens"]
        return str(choice)


def backend_from_env() -> OpenAIBackend | None:
    """Build a backend from ``SHIKO_LLM_*``; ``None`` when unconfigured."""
    base_url = os.environ.get("SHIKO_LLM_BASE_URL", "").strip()
    api_key = os.environ.get("SHIKO_LLM_API_KEY", "").strip()
    model = os.environ.get("SHIKO_LLM_MODEL", "").strip()
    if not (base_url and api_key and model):
        return None
    temperature = float(os.environ.get("SHIKO_LLM_TEMPERATURE", "0.6") or 0.6)
    return OpenAIBackend(base_url=base_url, api_key=api_key, model_id=model, temperature=temperature)
