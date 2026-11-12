"""Reproduce the production Urdu booking turn raw to see everything the LLM emits."""

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

URDU_TURN = "السلام علیکم، مجھے کل صبح نو بجے اپوائنٹمنٹ چاہیے"
SHORT_ACK = "جی"


async def run(model: str, llm_tools) -> None:
    llm = agent._build_llm()
    ctx = ChatContext()
    # emulate a production-like Urdu opening
    ctx.add_message(
        role="system",
        content=(
            "You are Mike, a warm dental receptionist at Umair Ashraf Dental Clinic. "
            "Mirror the caller: Urdu in, Urdu out, Urdu script only. Never Roman Urdu. "
            "Open replies with a 4-8 word opener. 1-2 short sentences."
        ),
    )
    ctx.add_message(role="assistant", content="السلام علیکم! یہ عمر اشرف ڈینٹل کلینک ہے، میں مائیک ہوں۔ بتائیے، کیا مدد کر سکتا ہوں؟")
    ctx.add_message(role="user", content=URDU_TURN)
    print("=" * 20, model)
    async with utils.http_context.open():
        try:
            async for chunk in llm.chat(chat_ctx=ctx, tools=llm_tools):
                d = chunk.delta
                if not d:
                    continue
                if d.content:
                    print("C:", d.content, end="", flush=True)
                if getattr(d, "reasoning_content", None):
                    print("\nR:", d.reasoning_content, end="", flush=True)
                if d.tool_calls:
                    for tc in d.tool_calls:
                        print(f"\nTOOL: {tc.function.name}({tc.function.arguments})", end="", flush=True)
        except Exception as exc:
            print(f"\nFAIL: {type(exc).__name__}: {str(exc)[:200]}")
    print()
    await llm.aclose()


async def main() -> None:
    assistant = agent.Assistant()
    print("tools:", [getattr(t, "info", "") and getattr(getattr(t, "info", None), "name", type(t).__name__) for t in assistant.tools])
    await run("qwen primary", list(assistant.tools))


if __name__ == "__main__":
    asyncio.run(main())
