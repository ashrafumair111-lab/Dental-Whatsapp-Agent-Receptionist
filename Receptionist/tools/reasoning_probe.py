"""Show the raw delta structure from Groq models to see if reasoning leaks into content."""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

from livekit.agents import utils
from livekit.agents.llm import ChatContext
from livekit.plugins import openai

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen3.8-27b"


async def main() -> None:
    llm = openai.LLM(
        model=MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        max_completion_tokens=120,
        temperature=0.4,
    )
    ctx = ChatContext()
    ctx.add_message(role="system", content="You are Mike, a dental receptionist. Reply in 1 short sentence.")
    ctx.add_message(role="user", content="theek hai")
    async with utils.http_context.open():
        async for chunk in llm.chat(chat_ctx=ctx):
            d = chunk.delta
            parts = []
            if d and d.content:
                parts.append(f"content={d.content!r}")
            rc = getattr(d, "reasoning_content", None) if d else None
            if rc:
                parts.append(f"reasoning={rc!r}")
            if d and d.tool_calls:
                parts.append("tool_calls=yes")
            if parts:
                print(" | ".join(parts))
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
