"""Model backend configuration.

Three layers, lowest to highest precedence:

1. A TOML config (``SHIKO_CONFIG`` or ``~/.config/shiko/config.toml``) with an
   ``[llm]`` table. Provider presets supply base URLs and key variable names.
2. Environment variables (``SHIKO_LLM_*``), so a shell or a supervisor can
   switch providers without editing a file.
3. Explicit arguments to :func:`load_settings`.

Keys are never written to the config file by Shiko. A key is read from the
provider's environment variable, from ``SHIKO_LLM_API_KEY``, or from an opt-in
``env_file``. Python does not load ``.env`` automatically; setting
``SHIKO_ENV_FILE`` (or ``[llm] env_file``) makes Shiko read one explicitly.

OpenCode Go needs a browser-like User-Agent and an ``x-opencode-session``
header; the preset adds the session header, and the backend sends the UA.
Reasoning models that spend the completion budget before answering can be
switched off per deployment with ``reasoning_effort`` and ``extra_body``.
"""

from __future__ import annotations

import json
import os
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Model configuration is incomplete or contradictory."""


PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "opencode": {
        "base_url": "https://opencode.ai/zen/go/v1",
        "api_key_env": "OPENCODE_GO_API_KEY",
        "model": "deepseek-v4.1-flash",
        "session_header": "x-opencode-session",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        # An explicit model is required in practice: openrouter/auto can route
        # to a reasoning model that returns empty content on extraction prompts.
        "model": "qwen/qwen3.8-flash",
    },
    "custom": {
        "api_key_env": "SHIKO_LLM_API_KEY",
    },
}

DEFAULT_CONFIG_PATH = "~/.config/shiko/config.toml"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1600
DEFAULT_TIMEOUT = 90.0


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "opencode"
    model: str = ""
    base_url: str = ""
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT
    api_key: str = field(default="", repr=False)
    api_key_env: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    reasoning_effort: str | None = None
    env_file: str = ""
    config_path: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        """Settings safe to log: no key, no env file contents."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "reasoning_effort": self.reasoning_effort,
            "headers": dict(self.headers),
            "extra_body": dict(self.extra_body),
            "api_key_env": self.api_key_env,
            "has_api_key": bool(self.api_key),
            "config_path": self.config_path,
        }


def load_settings(
    *,
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
    env_file: str | Path | None = None,
) -> LLMSettings:
    environment = dict(os.environ if env is None else env)
    path = _config_path(config_path, environment)
    table = _read_config(path) if path else {}

    chosen_provider = (
        provider
        or environment.get("SHIKO_LLM_PROVIDER")
        or table.get("provider")
        or "opencode"
    ).strip()
    preset = PROVIDER_PRESETS.get(chosen_provider, {})

    chosen_model = (
        model
        or environment.get("SHIKO_LLM_MODEL")
        or table.get("model")
        or preset.get("model")
        or ""
    ).strip()
    chosen_base = (
        base_url
        or environment.get("SHIKO_LLM_BASE_URL")
        or table.get("base_url")
        or preset.get("base_url")
        or ""
    ).strip()
    chosen_temperature = _number(
        temperature,
        environment.get("SHIKO_LLM_TEMPERATURE"),
        table.get("temperature"),
        default=DEFAULT_TEMPERATURE,
    )
    chosen_max_tokens = int(_number(
        None,
        environment.get("SHIKO_LLM_MAX_TOKENS"),
        table.get("max_tokens"),
        default=DEFAULT_MAX_TOKENS,
    ))
    chosen_timeout = _number(
        None,
        environment.get("SHIKO_LLM_TIMEOUT"),
        table.get("timeout"),
        default=DEFAULT_TIMEOUT,
    )

    key_env = (
        environment.get("SHIKO_LLM_API_KEY_ENV")
        or table.get("api_key_env")
        or preset.get("api_key_env")
        or "SHIKO_LLM_API_KEY"
    ).strip()
    chosen_env_file = str(
        env_file
        or environment.get("SHIKO_ENV_FILE")
        or table.get("env_file")
        or ""
    )
    file_values = _read_env_file(chosen_env_file) if chosen_env_file else {}
    resolved_key = (
        api_key
        or environment.get("SHIKO_LLM_API_KEY")
        or environment.get(key_env)
        or file_values.get("SHIKO_LLM_API_KEY")
        or file_values.get(key_env)
        or ""
    ).strip()

    headers = dict(table.get("headers") or {})
    session_header = preset.get("session_header")
    if session_header and session_header not in headers:
        session = environment.get("SHIKO_LLM_SESSION") or f"shiko-{uuid.uuid4().hex[:12]}"
        headers[session_header] = session

    extra_body = dict(table.get("extra_body") or {})
    raw_extra = environment.get("SHIKO_LLM_EXTRA_BODY", "").strip()
    if raw_extra:
        try:
            parsed = json.loads(raw_extra)
        except ValueError as exc:
            raise ConfigError(f"SHIKO_LLM_EXTRA_BODY is not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ConfigError("SHIKO_LLM_EXTRA_BODY must be a JSON object")
        extra_body.update(parsed)

    reasoning = (
        environment.get("SHIKO_LLM_REASONING_EFFORT")
        or table.get("reasoning_effort")
        or None
    )
    if reasoning is None and chosen_model.lower().startswith("deepseek"):
        # DeepSeek's reasoning can consume the whole completion budget before
        # the JSON answer starts; extraction needs the answer, not the thoughts.
        reasoning = "none"
        if "thinking" not in extra_body:
            extra_body["thinking"] = {"type": "disabled"}

    return LLMSettings(
        provider=chosen_provider,
        model=chosen_model,
        base_url=chosen_base,
        temperature=float(chosen_temperature),
        max_tokens=int(chosen_max_tokens),
        timeout=float(chosen_timeout),
        api_key=resolved_key,
        api_key_env=key_env,
        headers=headers,
        extra_body=extra_body,
        reasoning_effort=str(reasoning).strip() if reasoning else None,
        env_file=chosen_env_file,
        config_path=str(path) if path else "",
    )


def build_backend(settings: LLMSettings):
    """Build an OpenAI-compatible backend from resolved settings."""
    from .backends import OpenAIBackend

    if not settings.base_url:
        raise ConfigError(
            f"no base_url for provider '{settings.provider}'; set SHIKO_LLM_BASE_URL "
            "or a [llm] base_url in the config"
        )
    if not settings.model:
        raise ConfigError("no model configured; set SHIKO_LLM_MODEL or [llm] model")
    if not settings.api_key:
        raise ConfigError(
            f"no API key for provider '{settings.provider}'; set {settings.api_key_env}, "
            "SHIKO_LLM_API_KEY, or an env_file"
        )
    return OpenAIBackend(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model_id=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        extra_headers=dict(settings.headers),
        extra_body=dict(settings.extra_body),
        reasoning_effort=settings.reasoning_effort,
    )


def _config_path(explicit: str | Path | None, env: Mapping[str, str]) -> Path | None:
    candidate = explicit or env.get("SHIKO_CONFIG") or DEFAULT_CONFIG_PATH
    path = Path(candidate).expanduser()
    return path if path.exists() else None


def _read_config(path: Path) -> dict[str, Any]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    table = data.get("llm")
    if table is None:
        return {}
    if not isinstance(table, dict):
        raise ConfigError("[llm] must be a table")
    return table


def _read_env_file(path: str | Path) -> dict[str, str]:
    target = Path(path).expanduser()
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip("'\"")
    return values


def _number(*values: Any, default: float) -> float:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"expected a number, got {value!r}") from exc
    return default


__all__ = [
    "ConfigError",
    "DEFAULT_TEMPERATURE",
    "LLMSettings",
    "PROVIDER_PRESETS",
    "build_backend",
    "load_settings",
]
