"""Production-path check: build the real LLM, run the Urdu booking turn, print raw chunks."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.production" if os.path.exists(".env.production") else ".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import agent  # noqa: E402
from livekit.agents import utils  # noqa: E402
from livekit.agents.llm import ChatContext  # noqa: E402


async def main() -> None:
    assistant = agent.Assistant()
    tools = list(assistant.tools)
    llm = agent._build_llm()
    ctx = ChatContext()
    ctx.add_message(role="system", content=assistant.instructions)
    ctx.add_message(
        role="assistant",
        content="السلام علیکم! یہ عمر اشرف ڈینٹل کلینک ہے، میں مائیک ہوں۔ بتائیے، کیا مدد کر سکتا ہوں؟",
    )
    ctx.add_message(role="user", content="مجھے پرسوں صبح نو بجے اپوائنٹمنٹ چاہیے")
    async with utils.http_context.open():
        try:
            async for chunk in llm.chat(chat_ctx=ctx, tools=tools):
                d = getattr(chunk, "delta", None)
                if d and d.content:
                    print("C:", d.content, end="", flush=True)
                for tc in (getattr(d, "tool_calls", None) or []) if d else []:
                    print(f"\n[TOOL] {tc.name}({tc.arguments})", flush=True)
        except Exception as exc:
            print(f"\nFAIL: {type(exc).__name__}: {str(exc)[:250]}")
    print()
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
