"""Check Groq tool-calling through the OpenAI-compatible plugin (booking tools).

Uses the real Assistant toolset, so it fails if the model cannot call
`check_availability` the way the agent needs it to.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import agent  # noqa: E402
from livekit.agents import utils  # noqa: E402
from livekit.agents.llm import ChatContext  # noqa: E402


async def main() -> None:
    assistant = agent.Assistant()
    llm = agent._build_llm()
    ctx = ChatContext()
    ctx.add_message(role="system", content="You are Mike, a dental receptionist. Use tools for availability.")
    ctx.add_message(role="user", content="do you have anything free tomorrow morning?")
    calls = 0
    text: list[str] = []
    async with utils.http_context.open():
        async for chunk in llm.chat(chat_ctx=ctx, tools=list(assistant.tools)):
            if chunk.delta and chunk.delta.tool_calls:
                calls += len(chunk.delta.tool_calls)
            if chunk.delta and chunk.delta.content:
                text.append(chunk.delta.content)
    print("tool_call_deltas:", calls)
    print("text:", "".join(text)[:160])
    await llm.aclose()
    if calls == 0:
        raise SystemExit("FAIL: model did not call a tool")


if __name__ == "__main__":
    asyncio.run(main())
