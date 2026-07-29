<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=flat&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/LiveKit_Agents-1.6.7%2B-00D4AA?style=flat&logo=livekit" alt="LiveKit Agents">
  <img src="https://img.shields.io/badge/OpenAI-GPT_4o--mini-412991?style=flat&logo=openai" alt="OpenAI GPT-4o-mini">
  <img src="https://img.shields.io/badge/Deepgram-Nova_3-13EF93?style=flat&logo=deepgram" alt="Deepgram Nova 3">
  <img src="https://img.shields.io/badge/Cartesia-Sonic_3-FF6B35?style=flat" alt="Cartesia Sonic 3">
  <img src="https://img.shields.io/badge/Cal.com-API_v2-292929?style=flat&logo=cal.com" alt="Cal.com API">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat" alt="License">
</p>

<h1 align="center">🤖 Highly Latency-Optimized Voice Agent</h1>

<p align="center">
  <em>A production-tuned LiveKit voice AI pipeline (STT → LLM → TTS, semantic turn detection, adaptive interruption, reliable async tool fillers), demonstrated here as "Jassey" — a clinic receptionist that answers phone calls, books appointments, and handles rescheduling and cancellations through natural conversation.</em>
</p>

---

## ✨ Features

- **📞 Voice AI Receptionist** — Named "Jassey", handles inbound calls with warm, human-like conversation
- **🗣️ Natural Speech** — Sparse, well-placed SSML pauses (at most one `<break/>` per response) and adaptive conversation flow
- **📅 Smart Booking** — Powered by Cal.com API v2: check availability, book, reschedule, or cancel appointments
- **🔊 Best-in-Class AI Pipeline** — Deepgram Nova-3 (STT) → GPT-4o-mini (LLM) → Cartesia Sonic-3 (TTS), all via LiveKit Inference
- **🎯 Semantic Turn Detection** — LiveKit's audio Turn Detector v1 (combines speech meaning + acoustic cues) with dynamic endpointing and adaptive interruption handling
- **🔇 Noise Cancellation** — ai-coustics Quail VF-S for clear audio in noisy environments
- **⚡ Pre-warmed VAD** — Silero VAD loaded at startup for low-latency voice activity detection
- **🧵 Reliable Tool Fillers** — Long-running Cal.com calls emit a code-driven status update and acoustic filler audio, so the agent can never "promise and go silent"
- **📊 Built-in Metrics** — Tracks LLM token usage and TTS performance per call
- **🐳 Docker-Ready** — Multi-stage UV-based build for production deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LiveKit Cloud                             │
│  (WebRTC signaling, room management, participant handling)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Agent Server (AgentServer)                 │
│  - Prewarms Silero VAD model at job start                   │
│  - Manages agent lifecycle and job dispatch                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  AgentSession                               │
│  - STT: Deepgram Nova-3 (multilingual)                      │
│  - LLM: GPT-4o-mini (priority inference, max 150 tokens)   │
│  - TTS: Cartesia Sonic-3 (voice: custom preset)            │
│  - VAD: Silero (prewarmed, 0.5 threshold)                  │
│  - Noise cancellation: ai-coustics Quail VF-S              │
│  - Turn: Audio Turn Detector v1 (semantic+acoustic) + dynamic endpointing │
│  - Interruption: Adaptive (0.3s min, backchannel aware)    │
│  - Preemptive TTS enabled for faster response              │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
      ┌─────────────────┐          ┌─────────────────────┐
      │  Assistant      │          │  CalToolset         │
      │  (Agent)        │◄────────►│  (Function Tools)   │
      │  - Conversation │          │  - list_event_types │
      │  - SSML speech  │          │  - check_availability│
      │  - Turn control │          │  - create_booking   │
      └─────────────────┘          │  - reschedule_booking│
                                   │  - cancel_booking   │
                                   └────────┬────────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │   Cal.com API v2 │
                                   │  (Booking System)│
                                   └─────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Agent Framework** | [LiveKit Agents](https://docs.livekit.io/agents/) v1.6.7+ |
| **Speech-to-Text** | Deepgram Nova-3 (multilingual), via LiveKit Inference |
| **Language Model** | OpenAI GPT-4o-mini (priority inference, 150 max tokens), via LiveKit Inference |
| **Text-to-Speech** | Cartesia Sonic-3 (custom voice preset), via LiveKit Inference |
| **Voice Detection** | Silero VAD (prewarmed) |
| **Turn Detection** | LiveKit Audio Turn Detector v1 — semantic + acoustic end-of-turn model, built into `livekit-agents` |
| **Interruption Handling** | LiveKit Adaptive Interruption (barge-in vs. backchannel detection) |
| **Noise Cancellation** | ai-coustics Quail VF-S |
| **Booking System** | Cal.com API v2 |
| **Deployment** | Docker (UV multi-stage build) |
| **Language** | Python 3.12+ |

## 📦 Voice Pipeline Dependencies

Package dependencies declared in `Receptionist/pyproject.toml`:

| Package | Version | Purpose |
|---------|---------|---------|
| `livekit-agents` | `>=1.6.7` | Core agent framework — session orchestration, `inference.*` gateway (STT/LLM/TTS), the built-in audio Turn Detector, adaptive interruption, async tools (`ctx.update()` / `ctx.with_filler()`) |
| `livekit-plugins-silero` | `>=0.10` | Local Silero VAD model (voice activity detection), prewarmed at worker startup |
| `livekit-plugins-ai-coustics` | `>=0.2` | Input noise/voice-isolation enhancement (Quail VF-S model) |
| `python-dotenv` | `>=1.0` | Loads `.env.local` for local development |
| `httpx` | `>=0.27` | Async HTTP client used by `cal_tools.py` to call the Cal.com API |

**No separate `livekit-plugins-turn-detector` package is needed** — the audio Turn Detector shipped directly in `livekit-agents` 1.6.1+ and replaces the deprecated `MultilingualModel` text turn detector plugin.

**No separate OpenAI, Deepgram, or Cartesia API keys are required either.** STT, LLM, and TTS are all accessed through `livekit.agents.inference` (LiveKit Inference), which authenticates with your `LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET` and proxies/bills usage through LiveKit Cloud. Only Cal.com and ai-coustics need their own keys — see [Environment Variables](#environment-variables) below.

## 📁 Project Structure

```
Receptionist_For_Clinics/
├── Receptionist/
│   ├── src/
│   │   ├── agent.py              # Main entry point + Assistant agent
│   │   └── cal_tools.py          # Cal.com API function tools
│   ├── pyproject.toml            # Python dependencies & metadata
│   ├── uv.lock                   # Locked dependencies (reproducible builds)
│   ├── Dockerfile                # Multi-stage production Docker image
│   ├── start.sh                  # Local development runner
│   ├── livekit.toml              # LiveKit Cloud project config
│   ├── .dockerignore
│   └── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) package manager (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- LiveKit Cloud account (or self-hosted LiveKit server) — STT, LLM, and TTS are billed through this via LiveKit Inference
- Cal.com account with API key
- ai-coustics API key (noise cancellation plugin)

### Environment Variables

Create a `.env.local` file in the `Receptionist/` directory:

```bash
# LiveKit — also authenticates STT/LLM/TTS via LiveKit Inference, no separate
# OpenAI/Deepgram/Cartesia keys needed
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Cal.com — CAL_USERNAME must match the account that owns CAL_API_KEY
# (verify at https://api.cal.com/v2/me if unsure); the agent fails fast at
# startup if either is missing
CAL_API_KEY=your_cal_api_key
CAL_USERNAME=your_cal_username

# ai-coustics (noise cancellation plugin — separate provider, not proxied
# through LiveKit Inference)
AI_COUSTICS_API_KEY=...

# Optional — defaults to "Hamza Hospital". Also used to derive the
# placeholder patient email domain when no email is given.
COMPANY_NAME="Your Clinic Name"
```

### Local Development

```bash
cd Receptionist

# Install dependencies
uv sync

# Start the agent
chmod +x start.sh
./start.sh
```

Or directly:

```bash
cd Receptionist
uv run python3 src/agent.py start
```

### Docker Deployment

```bash
cd Receptionist

# Build the image
docker build -t receptionist-agent .

# Run with environment variables
docker run --env-file .env.local receptionist-agent
```

## 🧠 Agent Behavior

Jassey is designed to sound human, not robotic. Key behavior patterns:

- **Light SSML** — At most one `<break time="500ms"/>`–`<break time="1s"/>` tag per response, only at a genuinely long pause. Cartesia's own docs warn that stacking several break tags in one reply can cause the model to hallucinate and slows synthesis, so replies stay sparse by design.
- **Adaptive Flow** — No rigid step sequences. If the patient asks about slots, checks immediately. Only asks for name/email when booking.
- **Reliable Tool Fillers** — Each Cal.com tool call (`check_availability`, `create_booking`, etc.) sends an immediate status update via `ctx.update()` and, if the API is slow, plays acoustic filler audio via `ctx.with_filler()` — both driven by code inside the tool, not by hoping the LLM improvises a filler line before deciding to call the tool.
- **Self-Correction** — Drops the first version mid-sentence and restarts naturally
- **Warm Tone** — Calm, professional, conversational
- **Schedule Awareness** — Converts relative dates ("tomorrow", "next Monday") automatically

### Appointment Types

| Type | Slug | Duration |
|------|------|----------|
| General Consultation | `30min` | 30 min |
| Routine Checkup | `checkup` | 30 min |
| Follow-up Consultation | `follow-up` | 15 min |
| Urgent Consultation | `secret` | 15 min |
| New Patient Registration | `new-patient` | 45 min |

## 🔧 Tools

All booking operations go through Cal.com API function tools:

| Tool | Description |
|------|-------------|
| `list_event_types()` | List all available appointment types |
| `check_availability(slug, date)` | Check available time slots for a given date |
| `create_booking(slug, start, name, email, ...)` | Book a new appointment |
| `reschedule_booking(uid, new_start)` | Reschedule an existing appointment |
| `cancel_booking(uid)` | Cancel an appointment |

Features: 30-second slot caching, automatic retry with exponential backoff (max 3), human-readable event type names synced to Cal.com on startup.

## 📊 Metrics

The agent logs LLM and TTS metrics per call:
- **LLM**: Time to first token, duration, prompt/completion/total tokens, tokens/second
- **TTS**: Time to first byte, duration, audio duration, characters generated, cancelled status

## 🗺️ Roadmap

Planned enhancements:

- **🧠 Patient Memory** — PostgreSQL database for returning patient recognition
- **📞 WhatsApp Telephony** — Full WhatsApp call integration with webhooks
- **📋 Booking History** — Persistent conversation summaries per patient
- **🏥 Multi-clinic Support** — Handle multiple clinic locations

## 📄 License

MIT
