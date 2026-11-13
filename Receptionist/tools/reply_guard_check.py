"""Check the one-question-per-reply guard on the LLM stream."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import agent  # noqa: E402


async def collect(text: str) -> str:
    async def stream():
        for i in range(0, len(text), 4):
            yield text[i : i + 4]

    return "".join([c async for c in agent._one_question_only(stream())])


CASES = [
    (
        "Theek hai. Kis din ki booking chahiye? Kis time? Email? Phone? Shukriya!",
        "Theek hai. Kis din ki booking chahiye?",
    ),
    ("Kitne baje?", "Kitne baje?"),
    (
        "آپ کی email s m i t h ہے، sahi hai؟ Aap ka phone number?",
        "آپ کی email s m i t h ہے، sahi hai؟",
    ),
    ("No question here, just a short statement.", "No question here, just a short statement."),
]

fails = 0
for text, want in CASES:
    got = asyncio.run(collect(text))
    ok = got == want
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {got!r}")

# Tool-call chunks must pass straight through, untouched.
class FakeDelta:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = ["TOOL"]


class FakeItem:
    def __init__(self, content=None):
        self.delta = FakeDelta(content)


async def passthrough() -> int:
    async def stream():
        yield FakeItem("partial? " * 40)  # question-containing content: gets cut
        yield FakeItem(None)              # tool call item: must not be touched

    kept = [item async for item in agent._one_question_only(stream())]
    return sum(
        1
        for item in kept
        if item.delta.content is None and getattr(item.delta, "tool_calls", None)
    )

try:
    got = asyncio.run(passthrough())
    ok = got == 1
    print(f"{'OK  ' if ok else 'FAIL'} tool-call chunks pass through ({got} tool item(s))")
except Exception as exc:  # noqa: BLE001
    print(f"FAIL: {type(exc).__name__}: {exc}")
    ok = False
fails += 0 if ok else 1
print(f"{'OK  ' if ok else 'FAIL'} tool-call chunks pass through")

print("\nresult:", "PASS" if fails == 0 else f"{fails} FAILED")
