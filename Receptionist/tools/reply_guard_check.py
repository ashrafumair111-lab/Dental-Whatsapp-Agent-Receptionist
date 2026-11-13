"""Check the one-question-per-reply guard."""

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
        # emit the way an LLM does: a few characters at a time
        for i in range(0, len(text), 4):
            yield text[i : i + 4]

    return "".join([c async for c in agent._first_question_only(stream())])


CASES = [
    (
        "Theek hai. Kis din ki booking chahiye? Kis time? Email? Phone? Shukriya!",
        "Theek hai. Kis din ki booking chahiye?",
    ),
    ("Kitne baje?", "Kitne baje?"),
    (
        "آپ کی email smith@gmail.com ہے، sahi hai؟ Aap ka phone number?",
        "آپ کی email smith@gmail.com ہے، sahi hai؟",
    ),
    ("No question here, just a short statement.", "No question here, just a short statement."),
]

fails = 0
for text, want in CASES:
    got = asyncio.run(collect(text))
    ok = got == want
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {got!r}")

print("\nresult:", "PASS" if fails == 0 else f"{fails} FAILED")
