"""Microsoft Edge (Read Aloud) text-to-speech, keyed by voice name.

Why this exists: the clinic agent needs a native Urdu voice, and no mainstream
TTS on a free key speaks Urdu. Azure's ``ur-PK-AsadNeural`` is the voice we
want; the Edge Read Aloud endpoint serves that same voice without an API key,
so it works as the interim Urdu voice until the Azure key is added.

The public ``edge-tts`` package handles the signed websocket + token dance and
returns 24 kHz mono MP3. We decode it to 16-bit PCM with PyAV and stream frames
straight into LiveKit's emitter, so audio starts before synthesis finishes.

Swap-out plan: when an Azure Speech key exists, replace this class with
``livekit.plugins.azure.TTS`` — the rest of the agent stays untouched.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator

import av
import edge_tts
from livekit import rtc
from livekit.agents import APIConnectionError, DEFAULT_API_CONNECT_OPTIONS, tts

logger = logging.getLogger("edge-tts")


class EdgeTTS(tts.TTS):
    """Streaming TTS over the free Edge Read Aloud endpoint."""

    def __init__(
        self,
        *,
        voice: str = "ur-PK-AsadNeural",
        rate: float = 1.0,
        sample_rate: int = 24000,
        fallback: tts.TTS | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._voice = voice
        # Edge takes a percentage: 1.10 -> "+10%".
        self._rate_pct = max(-50, min(100, round((rate - 1.0) * 100)))
        # Used if Edge is unreachable: the agent keeps talking in another voice
        # instead of going silent mid-call.
        self._fallback = fallback

    @property
    def model(self) -> str:
        return f"edge:{self._voice}"

    @property
    def provider(self) -> str:
        return "microsoft-edge"

    def synthesize(
        self, text: str, *, conn_options=DEFAULT_API_CONNECT_OPTIONS  # type: ignore[assignment]
    ) -> tts.ChunkedStream:
        return _EdgeStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        if self._fallback is not None:
            await self._fallback.aclose()


class _EdgeStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=uuid.uuid4().hex,
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        try:
            async for pcm in _edge_pcm(
                self._input_text,
                voice=self._tts._voice,  # type: ignore[attr-defined]
                rate_pct=self._tts._rate_pct,  # type: ignore[attr-defined]
                sample_rate=self._tts.sample_rate,
            ):
                output_emitter.push(pcm)
            return
        except Exception as exc:  # network / endpoint change / decode error
            fallback = getattr(self._tts, "_fallback", None)
            if fallback is None or output_emitter.pushed_duration() > 0:
                raise APIConnectionError(f"edge-tts failed: {exc}") from exc
            logger.warning("edge-tts failed (%s) — using fallback voice", exc)

        # Nothing was pushed yet, so the whole line can be re-synthesized by the
        # backup provider without audio overlap.
        stream = fallback.synthesize(self._input_text, conn_options=self._conn_options)
        async for event in stream:
            output_emitter.push_frame(event.frame)


async def _edge_pcm(
    text: str, *, voice: str, rate_pct: int, sample_rate: int
) -> AsyncGenerator[bytes, None]:
    """Yield 16-bit mono PCM frames for ``text``."""
    if not text.strip():
        return

    communicate = edge_tts.Communicate(text, voice, rate=f"{rate_pct:+d}%")
    decoder = av.CodecContext.create("mp3", "r")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)

    async for chunk in communicate.stream():
        if chunk["type"] != "audio":
            continue
        for packet in decoder.parse(chunk["data"]):
            for frame in decoder.decode(packet):
                for out in resampler.resample(frame):
                    data = _plane_bytes(out)
                    if data:
                        yield data

    # Flush whatever the resampler is still holding.
    for out in resampler.resample(None):
        data = _plane_bytes(out)
        if data:
            yield data


def _plane_bytes(frame: av.AudioFrame) -> bytes:
    return bytes(frame.planes[0])


__all__ = ["EdgeTTS"]
