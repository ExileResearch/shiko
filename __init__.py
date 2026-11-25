"""Minimal Hermes plugin — registers a tool and a hook."""

import json

def register(ctx):
    def launch_bombs_baby(tool_name, args, result, **kwargs):
        printf(f"BOOM")

    ctx.register_hook('pre_llm_call', launch_bombs_baby)

    # # --- Hook: log every tool call ---
    # def on_tool_call(tool_name, args, result, **kwargs):
    #     print(f"[hello-world] tool called: {tool_name}")

    # ctx.register_hook("post_tool_call", on_tool_call) 

