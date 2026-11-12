"""Quick Groq sanity check for the voice agent (latency + non-empty reply)."""

from __future__ import annotations

import asyncio
import os
import sys
import time

from dotenv import load_dotenv

from livekit.agents import utils
from livekit.agents.llm import ChatContext
from livekit.plugins import openai

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def run(model: str, budget: int) -> None:
    llm = openai.LLM(
        model=model,
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        max_completion_tokens=budget,
        temperature=0.4,
    )
    ctx = ChatContext()
    ctx.add_message(role="system", content="You are Mike, a dental clinic receptionist. Reply briefly.")
    ctx.add_message(role="user", content="I need an appointment tomorrow at 9")
    parts: list[str] = []
    t0 = time.perf_counter()
    first = None
    try:
        async for chunk in llm.chat(chat_ctx=ctx):
            if chunk.delta and chunk.delta.content:
                if first is None:
                    first = time.perf_counter() - t0
                parts.append(chunk.delta.content)
        total = time.perf_counter() - t0
        ttft = f"{first:.2f}s" if first else "n/a"
        print(f"{model} budget={budget} ttft={ttft} total={total:.2f}s")
        print("  reply:", "".join(parts)[:120])
    except Exception as exc:
        print(f"{model} FAIL {type(exc).__name__}: {str(exc)[:200]}")
    finally:
        await llm.aclose()


async def main() -> None:
    async with utils.http_context.open():
        await run("openai/gpt-oss-120b", 300)
        await run("openai/gpt-oss-20b", 300)


if __name__ == "__main__":
    asyncio.run(main())
