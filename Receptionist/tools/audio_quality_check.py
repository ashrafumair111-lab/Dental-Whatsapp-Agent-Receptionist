"""Measure Urdu TTS audio for clipping / discontinuities (the "crackling" report)."""

from __future__ import annotations

import asyncio
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import agent  # noqa: E402
from livekit.agents import utils  # noqa: E402

TEXT_UR = "السلام علیکم! یہ عمر اشرف ڈینٹل کلینک ہے، میں مائیک ہوں۔ بتائیے، کیا مدد کر سکتا ہوں؟"
TEXT_EN = "Hi, this is Mike from Umair Ashraf Dental Clinic. How can I help you today?"


async def synth(tts, text: str, name: str) -> np.ndarray:
    chunks = []
    sr = 24000
    async for ev in tts.synthesize(text):
        frame = ev.frame
        sr = frame.sample_rate
        chunks.append(np.frombuffer(bytes(frame.data), dtype=np.int16))
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    path = os.path.join(os.path.dirname(__file__), "voice_samples", f"quality_{name}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())

    peak = int(np.max(np.abs(audio))) if audio.size else 0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
    clipped = int(np.sum(np.abs(audio) >= 32760))
    diffs = np.abs(np.diff(audio.astype(np.int32))) if audio.size > 1 else np.zeros(1)
    jumps = int(np.sum(diffs > 20000))
    print(
        f"{name:6} sr={sr} dur={audio.size / sr:5.2f}s peak={peak:6d} ({peak / 32767:.0%}) "
        f"rms={rms:7.0f} clipped={clipped:6d} hard_jumps={jumps:4d} -> {os.path.basename(path)}"
    )
    return audio


async def main() -> None:
    tts = agent._build_tts()
    async with utils.http_context.open():
        await synth(tts, TEXT_UR, "ur")
        await synth(tts, TEXT_EN, "en")
    # same Urdu line straight from Cartesia, no fallback wrapper
    from livekit.plugins import cartesia

    direct = cartesia.TTS(
        model=agent.TTS_UR_CARTESIA_MODEL,
        voice=agent.TTS_UR_CARTESIA_VOICE,
        language="ur",
        speed=1.0,
        sample_rate=24000,
        word_timestamps=False,
        api_key=agent.CARTESIA_API_KEY,
    )
    async with utils.http_context.open():
        await synth(direct, TEXT_UR, "ur_speed1")
    await direct.aclose()
    await tts.aclose()


if __name__ == "__main__":
    asyncio.run(main())
