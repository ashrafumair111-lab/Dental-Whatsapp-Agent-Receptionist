"""Synthesize an Urdu line with candidate Cartesia voices.

Run:  .venv\\Scripts\\python.exe tools\\cartesia_ur_check.py
Writes WAV samples to tools/voice_samples/cartesia_ur_<name>.wav so the voice
can be judged by ear before it is wired into the agent.
"""

from __future__ import annotations

import json
import os
import pathlib
import urllib.request

from dotenv import load_dotenv

load_dotenv(".env.local")

API_KEY = os.environ.get("CARTESIA_API_KEY", "").strip()
URL = "https://api.cartesia.ai/tts/bytes"
MODEL = "sonic-3.6"
LANG = "ur"

TEXT = (
    "السلام علیکم! یہ عمر اشرف ڈینٹل کلینک ہے، میں مائیک ہوں۔ "
    "جی بتائیے، کیا مدد کر سکتا ہوں؟"
)

VOICES = {
    "rohan_hi_male": "4877b818-c7fe-4c89-b1cf-eadf8e23da72",
    "sagar_hi_male": "6303e5fb-a0a7-48f9-bb1a-dd42c216dc5d",
    "rahul_hi_male": "393dd459-f8d8-4c3e-a86b-ec43a1113d0b",
    "walid_ar_male": "f1cdfb4a-bf7d-4e83-916e-8f0802278315",
    "daniel_en_male": "47c38ca4-5f35-497b-b1a3-415245fb35e1",
}

OUT = pathlib.Path(__file__).parent / "voice_samples"
OUT.mkdir(exist_ok=True)


def synth(name: str, voice: str) -> None:
    body = json.dumps(
        {
            "model_id": MODEL,
            "transcript": TEXT,
            "voice": {"mode": "id", "id": voice},
            "language": LANG,
            "output_format": {"container": "wav", "sample_rate": 24000, "encoding": "pcm_s16le"},
            "generation_config": {"speed": 1.15},
        }
    ).encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={
            "X-API-Key": API_KEY,
            "Cartesia-Version": "2025-04-16",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            audio = resp.read()
        path = OUT / f"cartesia_ur_{name}.wav"
        path.write_bytes(audio)
        print(f"OK   {name:16s} {len(audio):7d} bytes -> {path.name}")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        detail = ""
        if hasattr(exc, "read"):
            detail = exc.read()[:300].decode("utf-8", "replace")
        print(f"FAIL {name:16s} {exc} {detail}")


if __name__ == "__main__":
    for label, voice_id in VOICES.items():
        synth(label, voice_id)
