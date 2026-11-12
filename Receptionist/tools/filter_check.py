"""Unit test for the reasoning-envelope filter (_without_reasoning_envelopes)."""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "src")
import agent  # noqa: E402
from livekit.agents import llm as llm_mod  # noqa: E402


def chunk(text: str) -> llm_mod.ChatChunk:
    return llm_mod.ChatChunk(
        id="c1",
        delta=llm_mod.ChoiceDelta(role="assistant", content=text),
    )


async def collect(stream):
    out = []
    async for item in stream:
        if isinstance(item, llm_mod.ChatChunk):
            out.append(item.delta.content or "")
        elif isinstance(item, str):
            out.append(item)
    return "".join(out)


async def aiter(chunks):
    for c in chunks:
        yield chunk(c) if isinstance(c, str) else c


async def main() -> None:
    cases = [
        (["<think>user wants booking</think>Sure, I can help."], "Sure, I can help."),
        (["Sure, <think>check calendar", " first</think>let me see."], "Sure, let me see."),
        (["<|start|>assistant<|channel|>analysis<|message|>internal plan<|end|><|channel|>final<|message|>Done.<|end|>"], "Done."),
        (["Just a normal reply."], "Just a normal reply."),
        (["پہلی<think>hidden</think>دوسری"], "پہلیدوسری"),
        (["Book now<think>unclosed tail"], "Book now"),
    ]
    for chunks, want in cases:
        got = await collect(agent._without_reasoning_envelopes(aiter(chunks)))
        ok = got == want
        print(("OK  " if ok else "FAIL"), f"{got!r}")
        assert ok
    print("filter OK")


if __name__ == "__main__":
    asyncio.run(main())
