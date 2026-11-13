"""End-to-end check of the Deepgram/Edge/Groq pipeline without a phone call.

Builds the same objects the agent uses, synthesizes one Urdu (Edge) and one
English (Deepgram) line, and checks the language routing logic.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import agent  # noqa: E402
from livekit.agents import utils  # noqa: E402


async def synth(tts, text: str, label: str) -> None:
    t0 = time.perf_counter()
    first = None
    samples = 0
    async for ev in tts.synthesize(text):
        if first is None:
            first = time.perf_counter() - t0
        samples += ev.frame.samples_per_channel
    total = time.perf_counter() - t0
    print(f"  {label}: first_audio={first:.2f}s total={total:.2f}s audio={samples / 24000:.2f}s")


async def main() -> None:
    print("import OK | START_LANGUAGE =", agent.START_LANGUAGE, "| STT start =", agent.STT_START_LANGUAGE)

    cases = [
        ("کل صبح نو بجے appointment چاہیے", "ur"),
        ("मुझे कल अपॉइंटमेंट चाहिए", "ur"),
        ("mujhe kal appointment chahiye", "ur"),
        ("I need an appointment tomorrow at nine", "multi"),
    ]
    print("\nSTT switch target:")
    for text, want in cases:
        got = agent._stt_switch_target(text)
        print(f"  {'OK  ' if got == want else 'FAIL'} {got!r:8} {text[:40]}")

    tts = agent._build_tts()
    stt = agent._build_stt()
    llm = agent._build_llm()
    print(f"\nengines: tts={type(tts).__name__} stt={type(stt).__name__} llm={type(llm).__name__}")

    async with utils.http_context.open():
        print("\nTTS routing:")
        tts.update_options(language="ur", voice=agent.TTS_VOICE_UR)
        await synth(tts, "جی بالکل، میں دیکھتا ہوں۔", "ur → Cartesia sonic-3.6")
        tts.update_options(language="en", voice=agent.TTS_VOICE_EN)
        await synth(tts, "Sure, let me check that for you.", "en → Deepgram Aura")

    await tts.aclose()
    await stt.aclose()
    await llm.aclose()
    print("\nOK")


if __name__ == "__main__":
    asyncio.run(main())
