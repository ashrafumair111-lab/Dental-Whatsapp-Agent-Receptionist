"""Local voice audition for the bilingual (English + Urdu) receptionist.

Synthesizes the same Urdu lines with a few cheap TTS candidates so they can be
listened to side by side before touching the deployed agent.

Run:  .venv\\Scripts\\python.exe tools\\voice_lab.py
Out:  tools\\voice_samples\\*.wav
"""

from __future__ import annotations

import asyncio
import os
import wave
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import inference, utils

load_dotenv(".env.local")

OUT_DIR = Path(__file__).parent / "voice_samples"
OUT_DIR.mkdir(exist_ok=True)

# Line 1 is the phone greeting; line 2 has a date/time + an English dental word.
URDU_GREETING = (
    "السلام علیکم! یہ عمر اشرف ڈینٹل کلینک ہے، میں مائیک ہوں۔ "
    "آپ کی کیا مدد کر سکتا ہوں؟"
)
URDU_BOOKING = (
    "جی بالکل، کل صبح نو بجے appointment خالی ہے۔ "
    "کیا میں آپ کی booking کر دوں؟"
)

CASES = [
    ("01_inworld_flash_alex_ur", "inworld/inworld-tts-2-flash", "Alex", "ur-PK", URDU_GREETING),
    ("02_inworld_flash_edward_ur", "inworld/inworld-tts-2-flash", "Edward", "ur-PK", URDU_GREETING),
    ("03_inworld_tts2_alex_ur", "inworld/inworld-tts-2", "Alex", "ur-PK", URDU_GREETING),
    (
        "04_cartesia_blake_ur",
        "cartesia/sonic-3.6",
        "a167e0f3-df7e-4d52-a9c3-f949145efdab",
        "ur",
        URDU_GREETING,
    ),
    ("05_inworld_flash_alex_ur_mixed", "inworld/inworld-tts-2-flash", "Alex", "ur-PK", URDU_BOOKING),
    ("06_inworld_flash_alex_en", "inworld/inworld-tts-2-flash", "Alex", "en-US", "Hi, this is Mike from Umair Ashraf Dental Clinic. How can I help?"),
]


def _write_wav(path: Path, frames: list) -> None:
    if not frames:
        raise RuntimeError("no audio frames returned")
    first = frames[0]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(first.num_channels)
        wf.setsampwidth(2)
        wf.setframerate(first.sample_rate)
        for frame in frames:
            wf.writeframes(bytes(frame.data))


async def run_case(name: str, model: str, voice: str, language: str, text: str) -> None:
    tts = inference.TTS(model=model, voice=voice, language=language)
    frames = []
    try:
        async for ev in tts.synthesize(text):
            frames.append(ev.frame)
        path = OUT_DIR / f"{name}.wav"
        _write_wav(path, frames)
        seconds = sum(f.samples_per_channel for f in frames) / frames[0].sample_rate
        print(f"OK   {name}.wav  ({seconds:.1f}s, {len(frames)} frames)")
    except Exception as exc:  # keep going so one bad combo doesn't stop the rest
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    finally:
        await tts.aclose()


async def main() -> None:
    if not os.environ.get("LIVEKIT_API_KEY"):
        raise SystemExit("LIVEKIT_API_KEY missing — run from the Receptionist folder")
    for case in CASES:
        await run_case(*case)
    print(f"\nSamples in {OUT_DIR}")


if __name__ == "__main__":
    async def _main() -> None:
        # Plugins need an HTTP session context when run outside the worker.
        async with utils.http_context.open():
            await main()

    asyncio.run(_main())
