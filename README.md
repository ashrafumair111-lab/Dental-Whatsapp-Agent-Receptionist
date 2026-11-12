# Dental AI Voice Agent & Receptionist

AI-powered front desk for **Umair Ashraf Dental Clinic** — a bilingual (English + Urdu) voice agent for WhatsApp calls, a WhatsApp text chatbot, and a web demo. Patients can ask about the clinic, get their availability checked, and book, reschedule, or cancel appointments over the phone — in the language they speak.

## What's inside

| Piece | Path | What it does |
| --- | --- | --- |
| Voice agent "Mike" | `Receptionist/` | LiveKit voice agent: Urdu greeting, language-follows-caller pipeline (English ⇄ Urdu), booking tools via Cal.com. |
| WhatsApp chatbot | `Receptionist/webhook/functions/` | Netlify serverless functions: WhatsApp voice-call leg + text chat handling, verify/signature validation. |

(The browser web demo is deployed separately on Netlify and is not part of this repository.)

## Voice agent pipeline

- **STT** — Deepgram `nova-3` (`multi` at call start; the agent switches to `ur` as soon as it hears Urdu — `ur` transcribes Urdu/Hindi script but cannot do English).
- **TTS** — English: Deepgram Aura-2 (`aura-2-arcas-en`); Urdu: Azure Speech neural `ur-PK-AsadNeural` (24 kHz clean stream) with automatic fallback to Edge `ur-PK-AsadNeural` (free) — so a call is never silent.
- **LLM** — Groq (OpenAI-compatible) `openai/gpt-oss-120b`, fallback `openai/gpt-oss-20b`. Runs `reasoning_effort=low`. A reasoning-envelope filter strips any hidden analysis (`<think>`, harmony analysis channels) from the model stream before TTS.
- **Turn detection** — VAD (Silero) + dynamic endpointing; semantic TurnDetector is available via `TURN_DETECTION=model` but needs LiveKit Inference credit.
- **Reply routing** — per-reply text decides the voice: Urdu/Devanagari script and Roman-Urdu hint words get the Urdu voice; only clearly-English text goes to the English voice. Roman Urdu can never be read with an English accent.

## Booking tools (Cal.com)

`check_availability(date)`, `create_booking(...)`, `reschedule_booking(...)`, `cancel_booking(...)` — availability answers come only from tool results, and spoken confirmations happen only after the tool confirms. Slot caching included.

## Repo layout

```
Receptionist/
  src/agent.py            # LiveKit agent: pipeline, bilingual routing, bookings
  src/cal_tools.py        # Cal.com booking toolset (@llm.function_tool)
  src/edge_tts_plugin.py  # Free Edge TTS (Urdu fallback)
  tools/                  # local checks: pipeline smoke test, LLM/TTS probes
  webhook/functions/      # WhatsApp (voice + chat + webhooks)
```

## Setup

1. `cd Receptionist`
2. Copy `.env.example` → `.env.local` and fill in keys (see the file for the full reference). Never commit real keys — `.gitignore` covers env files.
3. `uv sync` (or `uv pip install -e .`)
4. Run locally: `uv run python src/agent.py console` — or deploy to LiveKit Cloud: `lk agent deploy --secrets-file .env.production --ignore-empty-secrets`
5. Local pipeline smoke check: `.venv\Scripts\python.exe tools\pipeline_check.py`

## Cost posture

Runs on Deepgram usage (STT nova-3 + Aura-2), Azure Speech F0 (Urdu, 0.5M chars/month free), Groq free tier (LLM), and LiveKit Cloud (hosting). LiveKit Inference credits are not used. Edge TTS is kept as a zero-cost fallback path for Urdu.
