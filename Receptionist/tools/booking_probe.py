"""Capture raw chunks for the production booking turn (exact prod prompt + tools)."""

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


def build_ctx(assistant) -> ChatContext:
    ctx = ChatContext()
    ctx.add_message(role="system", content=assistant.instructions)
    ctx.add_message(
        role="assistant",
        content="السلام علیکم! یہ عمر اشرف ڈینٹل کلینک ہے، میں مائیک ہوں۔ بتائیے، کیا مدد کر سکتا ہوں؟",
    )
    ctx.add_message(role="user", content="مجھے پرسوں صبح نو بجے اپوائنٹمنٹ چاہیے")
    return ctx


async def probe(name, llm, tools) -> None:
    assistant = agent.Assistant()
    ctx = build_ctx(assistant)
    print("=" * 25, name)
    async with utils.http_context.open():
        try:
            async for chunk in llm.chat(chat_ctx=ctx, tools=tools):
                d = chunk.delta
                if not d:
                    continue
                if d.content:
                    print("C:", d.content, end="", flush=True)
                if getattr(d, "reasoning_content", None):
                    print("\n[REASON]:", d.reasoning_content, end="", flush=True)
                for tc in d.tool_calls or []:
                    print(f"\n[TOOL] {tc}", end="", flush=True)
        except Exception as exc:
            print(f"FAIL: {type(exc).__name__}: {str(exc)[:200]}")
    print("\n")


async def main() -> None:
    assistant = agent.Assistant()
    tools = list(assistant.tools)

    from livekit.plugins import openai

    def groq(model: str, **kwargs):
        return openai.LLM(
            model=model,
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            max_completion_tokens=200,
            temperature=0.4,
            **kwargs,
        )

    await probe("qwen primary", groq("qwen/qwen3.8-27b"), tools)
    await probe("gpt-oss-120b fallback", groq("openai/gpt-oss-120b"), tools)


if __name__ == "__main__":
    asyncio.run(main())
