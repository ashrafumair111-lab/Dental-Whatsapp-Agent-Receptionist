<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=flat&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/LiveKit_Agents-1.8.11-00D4AA?style=flat&logo=livekit" alt="LiveKit Agents">
  <img src="https://img.shields.io/badge/Deepgram-Nova_3-13EF93?style=flat&logo=deepgram" alt="Deepgram Nova-3">
  <img src="https://img.shields.io/badge/Groq-gpt--oss--120b-00875A?style=flat&logo=groq" alt="Groq">
  <img src="https://img.shields.io/badge/Azure-Speech_Neural-0078D4?style=flat&logo=microsoft-azure" alt="Azure Speech">
  <img src="https://img.shields.io/badge/Cal.com-API_v2-292929?style=flat&logo=cal.com" alt="Cal.com">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat" alt="License">
</p>

<h1 align="center">🎙️ Dental AI Voice Agent & Receptionist</h1>

<p align="center">
  <em>A bilingual (English + Urdu), latency-tuned LiveKit voice receptionist — demonstrated as "Mike" at Umair Ashraf Dental Clinic — that answers WhatsApp calls, talks in the caller's language, and books, reschedules, or cancels appointments through natural conversation.</em>
</p>

---

## ✨ Features

- **📞 Voice AI Receptionist** — "Mike" handles inbound WhatsApp calls with a warm, human voice; greeting starts in Urdu and follows the caller's language
- **🗣️ Bilingual EN ⇄ Urdu** — English caller gets the English voice (Deepgram Aura-2), Urdu caller gets a native Urdu voice (Azure `ur-PK-AsadNeural`); per-reply text routing so Roman-Urdu is never read with an English accent
- **📅 Smart Booking** — Cal.com API v2: check availability, book, reschedule, cancel
- **🔊 Production Pipeline** — Deepgram `nova-3` STT → Groq `openai/gpt-oss-120b` → Deepgram Aura-2 (EN) / Azure Speech (UR), running on LiveKit Cloud (ap-south)
- **🎯 VAD + Dynamic Endpointing** — Silero VAD (prewarmed), adaptive interruption handling, preemptive TTS
- **⚡ Latency-Tuned** — warm STT/LLM/TTS connections, tool-filler audio so the agent never promises and goes silent
- **🔇 Noise Cancellation** — ai-coustics Quail VF-S (currently off for cost; one env var to re-enable)
- **📊 Built-in Metrics** — per-call TTFT, tokens/second, TTS byte/time metrics in the logs
- **🧵 Reasoning Envelope Filter** — hidden `<think>`/analysis channels are stripped before TTS, so internal planning is never spoken aloud

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 LiveKit Cloud (jarvis project)               │
│            (WebRTC signaling, rooms, ap-south)               │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                AgentServer (MetaByte Solutions Receptionist)  │
│  - Prewarmed Silero VAD at worker start                     │
│  - Job dispatch: "MetaByte Solutions Receptionist"           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  AgentSession (Assistant)                    │
│  - STT: Deepgram nova-3 (en: multi → ur on Urdu evidence)   │
│  - LLM: Groq openai/gpt-oss-120b (effort=low) +             │
│         gpt-oss-20b fallback (separate rate-limit bucket)   │
│  - TTS EN: Deepgram Aura-2 (arcas)                         │
│  - TTS UR: Azure Speech ur-PK-AsadNeural (F0 free tier)    │
│            └─ fallback: free Edge ur-PK-AsadNeural          │
│  - VAD: Silero (prewarmed) │ Turn: VAD (adaptive endpoint) │
│  - Interruption: Adaptive │ Preemptive TTS: on              │
│  - History capped at 16 items (Groq ITPM limit)             │
└──────────┬───────────────────────────┬─────────────────────┘
           ▼                           ▼
   ┌───────────────┐          ┌─────────────────┐
   │ Assistant     │◄────────►│  CalToolset      │
   │ (Agent)       │          │  check_avail.    │
   │ - Conversation│          │  create_booking  │
   │ - Per-reply   │          │  reschedule      │
   │   TTS routing │          │  cancel_booking  │
   └───────────────┘          └────────┬─────────┘
                                       ▼
                              ┌─────────────────┐
                              │  Cal.com API v2 │
                              └─────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Agent Framework** | [LiveKit Agents](https://docs.livekit.io/agents/) v1.8.11 |
| **Speech-to-Text** | Deepgram Nova-3 (`multi` start → `ur` on Urdu) — own key |
| **Language Model** | Groq (OpenAI-compatible) `openai/gpt-oss-120b`, low reasoning effort; `gpt-oss-20b` fallback |
| **Text-to-Speech (EN)** | Deepgram Aura-2 `aura-2-arcas-en` |
| **Text-to-Speech (UR)** | Azure Speech `ur-PK-AsadNeural` (free F0, 0.5M chars/mo); Edge TTS fallback |
| **Voice Activity** | Silero VAD (prewarmed) |
| **Turn Detection** | VAD (`TURN_DETECTION=vad`); semantic detector available via `model` (needs LiveKit Inference) |
| **Noise Cancellation** | ai-coustics Quail VF-S (plumb line remains) |
| **Booking** | Cal.com API v2 |
| **Deployment** | Docker via `lk agent deploy` (LiveKit Cloud, ap-south) |
| **Language** | Python 3.12+ |

## 📦 Key Dependencies

Declared in `Receptionist/pyproject.toml`:

| Package | Purpose |
|---------|---------|
| `livekit-agents>=1.8.11` | Session orchestration, endpointing, tool plumbing, metrics |
| `livekit-plugins-silero` / `livekit-plugins-ai-coustics` | VAD + noise handling |
| `livekit-plugins-deepgram` | Nova-3 STT + Aura-2 TTS |
| `livekit-plugins-azure` | Urdu TTS (Azure Speech) |
| `livekit-plugins-openai` | Groq-compatible LLM client |
| `edge-tts` + `av` | Free Urdu TTS fallback (Edge Read Aloud + PCM decode) |
| `httpx` | Cal.com API calls |

## 📁 Project Structure

```
Receptionist/
├── src/
│   ├── agent.py            # LiveKit agent: pipeline, bilingual routing (EN⇄UR), tools wiring
│   ├── cal_tools.py        # Cal.com API v2 function tools (availability, book, reschedule, cancel)
│   └── edge_tts_plugin.py  # Free Edge TTS plugin (Urdu fallback path)
├── tools/                  # Local checks (pipeline smoke test, LLM/TTS probes)
├── webhook/functions/      # WhatsApp calling + text chatbot (Netlify)
├── pyproject.toml / uv.lock
├── Dockerfile
└── livekit.toml            # LiveKit Cloud project config (agent name, dispatch)
```

## 🚀 Getting Started

Prerequisites: Python 3.12+, [UV](https://docs.astral.sh/uv/), a Cal.com key, and:

- **Deepgram API key** (STT + English TTS)
- **Groq API key** (LLM — free tier, see note below)
- **Azure Speech** resource (`AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION`, F0 free tier) — for Urdu
- **LiveKit Cloud** credentials (project hosting)

### Environment variables

`Receptionist/.env.example` is the full reference. The essential ones:

```bash
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
DEEPGRAM_API_KEY=...
GROQ_API_KEY=...
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=centralindia   # or your region
CAL_API_KEY=...
CAL_USERNAME=...
COMPANY_NAME="Umair Ashraf Dental Clinic"
```

### Local development

```bash
cd Receptionist
uv sync
uv run python src/agent.py start
# smoke test (no phone call needed):
python tools/pipeline_check.py
```

### Deploy to LiveKit Cloud

```bash
cd Receptionist
lk agent deploy --secrets-file .env.production --ignore-empty-secrets
```

## 🧠 Agent Behavior

Mike sounds human, not scripted:

- **Language follows the caller** — Urdu turn in → Urdu reply (native voice), English turn in → English reply; mixed dental terms (appointment, root canal, X-ray…) stay in English inside Urdu sentences; per-reply text routing keeps it clean
- **Warm, terse flow** — 1–2 sentences per reply, one question at a time, direct answers, no filler
- **Tool fillers** — Cal.com calls run inside `ctx.with_filler()` with an immediate status update, so the agent never goes silent mid-booking
- **Safety-first speech** — hidden reasoning/envelope text from the LLM is stripped before TTS (verified by unit tests)
- **Urdu-first greeting** — "السلام علیکم! یہ عمر اشرف ڈینٹل کلینک ہے، میں مائیک ہوں۔"

## 🔧 Tools (Cal.com)

| Tool | Purpose |
|------|---------|
| `check_availability(date)` | Slots for a date (YYYY-MM-DD), with caching |
| `create_booking(start_time, attendee_name, attendee_email, attendee_phone, ...)` | Book an appointment |
| `reschedule_booking(uid, new_start)` | Reschedule by booking UID |
| `cancel_booking(uid)` | Cancel by booking UID |

All booking actions go through tools — the model speaks a confirmation only after the tool confirms success.

## 📊 Metrics

The agent logs per-call metrics:
- **LLM**: TTFT, duration, prompt/completion tokens, tok/s
- **TTS**: TTFB, audio duration, characters, cancellation status
- **STT**: transcript events, language switches, final latency

## 🗺️ Roadmap

- **✅ WhatsApp Telephony** — user-initiated WhatsApp calling via the Meta Business Platform (webhook in `Receptionist/webhook/functions/`)
- **✅ Bilingual EN ⇄ Urdu** with native Azure Urdu voice
- **🗣️ Patient memory** — return-call recognition (PostgreSQL)
- **📋 Conversation summaries** — per-call notes into the clinic's system
- **🏥 Multi-clinic support**

## 📄 License

MIT
