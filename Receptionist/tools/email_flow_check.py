"""Check the email flow against the real production LLM.

The point of save_email is that the model never has to carry a spelled-out
address in its own context, so it can never "fix" it or ask for it again.
These probes confirm the model actually calls it.

Run:  .venv\\Scripts\\python.exe tools\\email_flow_check.py
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


async def probe(llm, tools, messages: list[tuple[str, str]]) -> list[str]:
    ctx = ChatContext()
    for role, content in messages:
        ctx.add_message(role=role, content=content)
    calls: list[str] = []
    async for chunk in llm.chat(chat_ctx=ctx, tools=tools):
        delta = getattr(chunk, "delta", None)
        for tc in (getattr(delta, "tool_calls", None) or []) if delta else []:
            calls.append(tc.name)
            print(f"   [TOOL] {tc.name}({(tc.arguments or '')[:120]})")
    return calls


async def main() -> None:
    assistant = agent.Assistant()
    tools = list(assistant.tools)
    llm = agent._build_llm()
    base = [
        ("system", assistant.instructions),
        ("assistant", "Hi, this is Mike from Umair Ashraf Dental Clinic — how can I help?"),
    ]
    fails = 0

    print("probe 1: caller spells the email on its own")
    calls = await probe(
        llm,
        tools,
        base
        + [
            ("user", "I want to book for tomorrow at 9."),
            ("assistant", "Sure. Can you spell your email for me?"),
            ("user", "s m i t h at gmail dot com"),
        ],
    )
    if "save_email" not in calls:
        fails += 1
        print("   FAIL: save_email was not called")

    print("\nprobe 2: everything in one breath")
    calls = await probe(
        llm,
        tools,
        base
        + [
            ("user", "Please book me tomorrow at nine, my name is Ali Khan, phone 0300 1234567, email is a l i at gmail dot com"),
        ],
    )
    if "save_email" not in calls:
        fails += 1
        print("   FAIL: save_email was not called")

    await llm.aclose()
    print("\nresult:", "PASS" if fails == 0 else f"{fails} FAILED")


if __name__ == "__main__":
    asyncio.run(main())
