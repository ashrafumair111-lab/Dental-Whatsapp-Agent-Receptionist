import sys
import asyncio

sys.path.insert(0, "src")
import agent  # noqa: E402
from livekit.agents import llm as llm_mod  # noqa: E402


async def main():
    print("ChatChunk path:", llm_mod.ChatChunk.__module__)
    d = llm_mod.ChoiceDelta(role="assistant", content="<think>x</think>Hi")
    c = llm_mod.ChatChunk(id="c1", delta=d)
    print("delta content:", repr(c.delta.content))
    print("isinstance:", isinstance(c, llm_mod.ChatChunk))

    async def aiter():
        yield c

    async for x in agent._without_reasoning_envelopes(aiter()):
        print(
            "OUT:",
            type(x).__name__,
            repr(x) if isinstance(x, str) else repr(getattr(getattr(x, "delta", None), "content", None)),
        )


asyncio.run(main())
