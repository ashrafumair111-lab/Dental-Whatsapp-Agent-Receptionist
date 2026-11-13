"""Confirm the per-sentence TTS router points Urdu at Cartesia and English at Deepgram."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import agent  # noqa: E402

tts = agent._build_tts()
print("LANG_MODE      =", agent.LANG_MODE, "| START =", agent.START_LANGUAGE)
print("deepgram       =", bool(agent._USE_DEEPGRAM))
print("cartesia urdu  =", agent._USE_CARTESIA_URDU, "| model =", agent.TTS_UR_CARTESIA_MODEL)
print("azure fallback =", agent._USE_AZURE_TTS)

cases = [
    ("السلام علیکم! یہ عمر اشرف ڈینٹل کلینک ہے۔", "ur"),
    ("mujhe kal appointment chahiye", "ur"),
    ("Sure, let me check that for you.", "en"),
]
print("\nrouter:")
bad = 0
for text, want in cases:
    active = tts._active(text)
    chain = getattr(active, "_tts_instances", [active])
    names = [type(t).__name__ for t in chain]
    ok = (want == "ur" and names[0] == "TTS" and "cartesia" in str(type(chain[0])).lower()) or (
        want == "en" and "deepgram" in str(type(chain[0])).lower()
    )
    bad += 0 if ok else 1
    print(f"  {'OK  ' if ok else 'FAIL'} {want} -> {names} {text[:40]}")
print("\nresult:", "PASS" if bad == 0 else f"{bad} FAILED")
