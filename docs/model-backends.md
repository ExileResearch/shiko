# Model backends

Shiko's extraction and arbitration stages call an OpenAI-compatible chat
endpoint. The backend is configuration, not code: switch providers without
editing Python.

## Precedence

1. A TOML file: `$SHIKO_CONFIG`, or `~/.config/shiko/config.toml` if it exists.
2. Environment variables (`SHIKO_LLM_*`).
3. Explicit arguments to `shiko.llm_config.load_settings`.

## Quick switch

```sh
# OpenCode Go (default preset)
export OPENCODE_GO_API_KEY=...
export SHIKO_LLM_PROVIDER=opencode
export SHIKO_LLM_MODEL=qwen3.8-flash
export SHIKO_LLM_TEMPERATURE=0.3

# OpenRouter when OpenCode credits run out
export OPENROUTER_API_KEY=...
export SHIKO_LLM_PROVIDER=openrouter
export SHIKO_LLM_MODEL=qwen/qwen3.8-flash
```

Choose an explicit OpenRouter model. `openrouter/auto` can route an extraction
prompt to a reasoning model that returns empty content.

## Config file

`shiko.example.toml` is a complete example. The `[llm]` table is what matters:

```toml
[llm]
provider = "opencode"
model = "qwen3.8-flash"
temperature = 0.3
max_tokens = 1600
# env_file = "~/.hermes/.env"
```

Switch providers by changing `provider` and `model`, or override both with
environment variables. A provider preset supplies the base URL and the name of
the key variable:

| Provider | Base URL | Key variable |
| --- | --- | --- |
| `opencode` | `https://opencode.ai/zen/go/v1` | `OPENCODE_GO_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `custom` | `SHIKO_LLM_BASE_URL` | `SHIKO_LLM_API_KEY` |

Keys never belong in the TOML file. Shiko reads the provider variable,
`SHIKO_LLM_API_KEY`, or an opt-in env file.

## `.env` files

Python does not load `.env` automatically. Shiko reads one only when asked:

```sh
export SHIKO_ENV_FILE=~/.hermes/.env
```

or `env_file = "~/.hermes/.env"` in `[llm]`. The OS environment wins over the
file, so a shell export still overrides a stale file value.

## Reasoning models

A reasoning model shares `max_tokens` with its reasoning, so a long prompt can
return empty content before the answer starts. Two remedies, both configurable:

```toml
[llm]
model = "deepseek-v4.1-flash"
reasoning_effort = "none"

[llm.extra_body]
thinking = { type = "disabled" }
```

`reasoning_effort` is sent at the top level; `extra_body` is merged into the
request JSON, matching the OpenAI SDK's `extra_body`. Either may also be set
with `SHIKO_LLM_REASONING_EFFORT` and `SHIKO_LLM_EXTRA_BODY` (a JSON object).
Raising `max_tokens` remains the fallback when a model has no disable switch.

An empty completion is a failed run (`backend.empty`), never zero claims.

## Provider quirks

OpenCode Go sits behind an edge check: a bare stdlib client gets HTTP 403
(error 1010), and requests without `x-opencode-session` are refused. The preset
adds a session header (`SHIKO_LLM_SESSION` overrides its value); the backend
sends a browser-like `User-Agent`. OpenRouter needs neither.

## Inspecting the resolved settings

```python
from shiko.llm_config import load_settings, build_backend

settings = load_settings()
print(settings.to_public_dict())  # no key, no secret
backend = build_backend(settings)
```

`to_public_dict()` is safe to log; `LLMSettings` hides the key from `repr`.
