"""Real-LLM probe: 'I want to book on Tuesday' must yield one question."""

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
    ctx.add_message(role="system", content=assistant.instructions)
    ctx.add_message(role="assistant", content="السلام علیکم! یہ عمر اشرف ڈینٹل کلینک ہے، میں مائیک ہوں۔")
    ctx.add_message(role="user", content="mujhe Tuesday ki booking karni hai")

    raw = ""
    async with utils.http_context.open():
        async for chunk in llm.chat(chat_ctx=ctx, tools=list(assistant.tools)):
            delta = getattr(chunk, "delta", None)
            if delta and delta.content:
                raw += delta.content
            for tc in (getattr(delta, "tool_calls", None) or []) if delta else []:
                print(f"[TOOL] {tc.name}({tc.arguments})")

    async def replay():
        for i in range(0, len(raw), 4):
            yield raw[i : i + 4]

    guarded = "".join([c async for c in agent._first_question_only(replay())])
    print("\nraw    :", raw.strip())
    print("spoken :", guarded.strip())
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
