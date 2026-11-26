"""Offline tests for model-backend configuration and request building."""

from __future__ import annotations

import json
import urllib.request

import pytest

from shiko.backends import BackendError, OpenAIBackend
from shiko.llm_config import ConfigError, build_backend, load_settings


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_exc) -> bool:
        return False


def test_provider_switch_uses_openrouter_preset(tmp_path) -> None:
    settings = load_settings(config_path=tmp_path / "missing.toml", env={
        "SHIKO_LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "test-key",
    })
    assert settings.base_url == "https://openrouter.ai/api/v1"
    assert settings.model == "qwen/qwen3.8-flash"
    assert settings.temperature == 0.3
    assert "x-opencode-session" not in settings.headers
    assert "test-key" not in json.dumps(settings.to_public_dict())


def test_toml_table_and_env_precedence(tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[llm]
provider = "opencode"
model = "deepseek-v4.1-flash"
temperature = 0.3
reasoning_effort = "none"

[llm.extra_body]
thinking = { type = "disabled" }

[llm.headers]
x-custom = "1"
""",
        encoding="utf-8",
    )
    settings = load_settings(config_path=config, env={
        "OPENCODE_GO_API_KEY": "test-key",
        "SHIKO_LLM_MODEL": "qwen3.8-flash",
    })
    assert settings.model == "qwen3.8-flash"  # env wins over TOML
    assert settings.reasoning_effort == "none"
    assert settings.extra_body == {"thinking": {"type": "disabled"}}
    assert settings.headers["x-custom"] == "1"
    assert settings.headers["x-opencode-session"].startswith("shiko-")
    assert settings.temperature == 0.3


def test_env_file_supplies_key_and_env_wins(tmp_path) -> None:
    env_file = tmp_path / "host.env"
    env_file.write_text("OPENCODE_GO_API_KEY=from-file\n", encoding="utf-8")
    from_file = load_settings(config_path=tmp_path / "missing.toml", env={"SHIKO_ENV_FILE": str(env_file)})
    assert from_file.api_key == "from-file"
    from_env = load_settings(config_path=tmp_path / "missing.toml", env={
        "SHIKO_ENV_FILE": str(env_file),
        "OPENCODE_GO_API_KEY": "from-env",
    })
    assert from_env.api_key == "from-env"


def test_incomplete_settings_fail_loudly(tmp_path) -> None:
    settings = load_settings(config_path=tmp_path / "missing.toml", env={"SHIKO_LLM_PROVIDER": "custom"})
    with pytest.raises(ConfigError):
        build_backend(settings)


def test_request_carries_reasoning_body_and_session(monkeypatch, tmp_path) -> None:
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = {key.lower(): value for key, value in request.headers.items()}
        captured["timeout"] = timeout
        return FakeResponse({
            "choices": [{"message": {"content": '[{"content": "x"}]'}}],
            "usage": {"total_tokens": 7},
        })

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = tmp_path / "config.toml"
    config.write_text(
        '[llm]\nprovider = "opencode"\nmodel = "deepseek-v4.1-flash"\n'
        'reasoning_effort = "none"\n\n[llm.extra_body]\nthinking = { type = "disabled" }\n',
        encoding="utf-8",
    )
    settings = load_settings(config_path=config, env={"OPENCODE_GO_API_KEY": "test-key"})
    backend = build_backend(settings)
    assert backend.complete("hello", timeout=5) == '[{"content": "x"}]'
    assert captured["body"]["reasoning_effort"] == "none"
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert captured["body"]["temperature"] == 0.3
    assert captured["headers"]["x-opencode-session"]
    assert "shiko-memory" in captured["headers"]["user-agent"]
    assert captured["timeout"] == 5
    assert backend.tokens == 7


def test_empty_completion_is_a_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda request, timeout=None: FakeResponse({"choices": [{"message": {"content": ""}}]}),
    )
    backend = OpenAIBackend(base_url="https://example.invalid/v1", api_key="test", model_id="m")
    with pytest.raises(BackendError) as failure:
        backend.complete("hello")
    assert failure.value.code == "backend.empty"
