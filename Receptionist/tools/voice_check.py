"""Round-trip check: transcribe the generated voice samples back to text.

If the Urdu TTS is intelligible and correctly accented, the Urdu STT model
should return roughly the same words in Urdu script. Run after voice_lab.py.
"""

from __future__ import annotations

import asyncio
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import inference, stt as stt_mod, utils

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SAMPLES = Path(__file__).parent / "voice_samples"


def _read_wav(path: Path) -> list[rtc.AudioFrame]:
    with wave.open(str(path), "rb") as wf:
        assert wf.getsampwidth() == 2, "expects 16-bit PCM"
        rate, channels = wf.getframerate(), wf.getnchannels()
        frames = []
        while True:
            data = wf.readframes(int(rate * 0.1))
            if not data:
                break
            frames.append(
                rtc.AudioFrame(
                    data=data,
                    sample_rate=rate,
                    num_channels=channels,
                    samples_per_channel=len(data) // (2 * channels),
                )
            )
    return frames


async def _transcribe(stt, frames: list[rtc.AudioFrame]) -> tuple[str, str | None]:
    stream = stt.stream()

    async def _push() -> None:
        for frame in frames:
            stream.push_frame(frame)
            await asyncio.sleep(frame.samples_per_channel / frame.sample_rate)
        stream.end_input()

    pusher = asyncio.create_task(_push())
    text, language = "", None
    try:
        iterator = stream.__aiter__()
        while True:
            try:
                ev = await asyncio.wait_for(iterator.__anext__(), timeout=8)
            except (TimeoutError, StopAsyncIteration):
                break
            if ev.type is stt_mod.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
                alt = ev.alternatives[0]
                text = f"{text} {alt.text}".strip()
                language = str(alt.language) if alt.language else language
    finally:
        pusher.cancel()
        await stream.aclose()
    return text, language


async def main() -> None:
    stt = inference.STT(model="assemblyai/universal-3-6-pro")
    try:
        for path in sorted(SAMPLES.glob("*.wav")):
            if sys.argv[1:] and not any(path.name.startswith(p) for p in sys.argv[1:]):
                continue
            frames = _read_wav(path)
            try:
                text, language = await asyncio.wait_for(_transcribe(stt, frames), timeout=60)
                print(f"\n{path.name}  [lang={language}]\n  {text}")
            except Exception as exc:
                print(f"\n{path.name}  FAIL: {type(exc).__name__}: {exc}")
            await asyncio.sleep(3)
    finally:
        await stt.aclose()


if __name__ == "__main__":
    async def _main() -> None:
        async with utils.http_context.open():
            await main()

    asyncio.run(_main())
