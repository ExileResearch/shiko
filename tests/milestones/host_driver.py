"""Run one real Hermes agent session for the milestone 1.3 probe.

Executed by the milestone test with the Hermes interpreter (not the project
venv), cwd ``~/.hermes/hermes-agent``, so it imports the pinned host. The agent
talks to a loopback fake model, which keeps the trace reproducible and costs no
tokens; the lifecycle under observation is the real host's.

Usage (from the milestone test):

    python tests/milestones/host_driver.py --session-id probe-1 --turn first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERMES_AGENT_ROOT = Path(os.environ.get("SHIKO_HERMES_AGENT_ROOT") or Path.home() / ".hermes" / "hermes-agent")

EXTRA_CONFIG = """\
updates:
  check: false
models_dev:
  url: http://127.0.0.1:9/api.json
skills:
  creation_nudge_interval: 0
memory:
  provider: shiko_probe
  nudge_interval: 0
plugins:
  enabled:
    - shiko_probe
compression:
  enabled: true
  threshold: 0.50
  threshold_tokens: 50000
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--turn", required=True, choices=["first", "second", "pair"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--history-in", default="")
    parser.add_argument("--history-out", default="")
    parser.add_argument("--requests-out", default="")
    args = parser.parse_args()

    sys.path.insert(0, str(HERMES_AGENT_ROOT))
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

    home = Path(os.environ["HERMES_HOME"])
    first_script = [ToolCall("todo_list", {}), Text("DONE", prompt_tokens=60_000, completion_tokens=10)]
    script = (
        first_script + [Text("second answer", prompt_tokens=60_000, completion_tokens=10)]
        if args.turn == "pair"
        else first_script if args.turn == "first"
        else [Text("second answer", prompt_tokens=60_000, completion_tokens=10)]
    )

    result: dict = {"turn": args.turn, "session_id": args.session_id}
    with FakeLLMServer(script) as server:
        write_hermes_home(home, server.base_url, extra_config=EXTRA_CONFIG)
        # The host enforces a 64K floor. Reported usage above the (floored) trigger
        # makes the preflight compressor reachable without a huge transcript.
        config_path = home / "config.yaml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace("context_length: 128000", "context_length: 64000"),
            encoding="utf-8",
        )
        from run_agent import AIAgent

        agent = AIAgent(
            provider="custom",
            base_url=server.base_url,
            api_key="sk-fake",
            model="fake-model",
            session_id=args.session_id,
            quiet_mode=True,
            platform="cli",
        )
        try:
            prompts = (
                ["List the open questions on the arm project, then answer DONE.",
                 "What did we decide about the wrist camera?"]
                if args.turn == "pair"
                else ["List the open questions on the arm project, then answer DONE."]
                if args.turn == "first"
                else ["What did we decide about the wrist camera?"]
            )
            history = None if args.resume else []
            if args.history_in:
                history = json.loads(Path(args.history_in).read_text(encoding="utf-8"))
            last_outcome = None
            for index, prompt in enumerate(prompts):
                outcome = agent.run_conversation(
                    prompt, conversation_history=None if index else history, task_id="probe"
                )
                last_outcome = outcome
                result[f"outcome_{index}_type"] = type(outcome).__name__
                if isinstance(outcome, dict):
                    result[f"outcome_{index}_keys"] = sorted(outcome.keys())
                    result[f"outcome_{index}_tokens"] = outcome.get("prompt_tokens")
            result["requests"] = len(server.main_requests())
            if args.requests_out:
                Path(args.requests_out).write_text(
                    json.dumps([request.get("messages") for request in server.main_requests()]),
                    encoding="utf-8",
                )

            messages = last_outcome.get("messages") if isinstance(last_outcome, dict) else None
            if args.history_out and messages is not None:
                Path(args.history_out).write_text(json.dumps(messages), encoding="utf-8")

            # The host's own pre-compaction hook site: the function the compressor
            # calls before it summarizes. This is the observed path, not a mock.
            from agent.conversation_compression import _pre_compress_memory_context

            memory_context = _pre_compress_memory_context(agent, list(messages or []), False)
            result["pre_compress_returned"] = bool(memory_context)
        finally:
            agent.close()

    print("RESULT:" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
