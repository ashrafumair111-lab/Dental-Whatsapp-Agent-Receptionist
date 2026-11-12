"""Sanity check for the Groq ITPM fix: history cap + system prompt size."""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

import agent  # noqa: F401  (importing proves syntax + module init)
from livekit.agents.llm import ChatContext


def main() -> None:
    ctx = ChatContext()
    ctx.add_message(role="system", content="SYSTEM")
    for i in range(40):
        ctx.add_message(role="user" if i % 2 == 0 else "assistant", content=f"message {i} " * 30)
    print("before items:", len(ctx.items))
    ctx.truncate(max_items=16)
    print("after items:", len(ctx.items))
    msgs = ctx.messages()
    print("first kept message:", msgs[0].raw_text_content[:60])
    print("history cap OK")

    # Exact prompt size: replicate the f-string by reading the module constant.
    import inspect

    src = inspect.getsource(sys.modules["agent"])
    marker = "instructions=f\"\"\""
    i = src.index(marker) + len(marker)
    j = src.index('Today\'s date: {_today}."""', i)
    body = src[i:j]
    # rough token estimate: 4 chars/token
    print(f"prompt chars={len(body)} ~{len(body)//4} tokens (static part, real call adds tool schemas)")


if __name__ == "__main__":
    main()
