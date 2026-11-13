"""Mike — the dental clinic receptionist (Urdu + English).

Small and deterministic where it matters:

- STT: Deepgram nova-3 multilingual (English + Urdu) for the whole call.
- TTS: ElevenLabs eleven_multilingual_v2 — Daniel's voice, native Urdu and English.
- The LLM (Groq gpt-oss-120b) handles the greeting, clinic questions and the
  day/time. Every stage after the time is saved (name -> phone -> email ->
  booking) is spoken by code — one question, exactly once, never repeated.
- One question per reply, enforced on the model stream.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import timedelta

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import (
    AgentServer,
    AgentSession,
    Agent,
    JobProcess,
    TurnHandlingOptions,
    room_io,
    llm as llm_mod,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics
from livekit.plugins import ai_coustics, deepgram, elevenlabs, openai, silero

from cal_tools import CalToolset, _spell_back, now_local, format_when

logger = logging.getLogger("clinic")
logger.setLevel(logging.INFO)

load_dotenv(".env.local")

_company_name = os.environ.get("COMPANY_NAME", "Dental Clinic")
_company_legal = os.environ.get("COMPANY_LEGAL_NAME", _company_name)
_company_address = os.environ.get("COMPANY_ADDRESS", "")
_company_phone = os.environ.get("COMPANY_PHONE", "")
_company_email = os.environ.get("COMPANY_EMAIL", "")

_company_services_raw = os.environ.get("COMPANY_SERVICES", "")
_company_services = [s.strip() for s in _company_services_raw.split(";") if s.strip()]
_company_services_block = (
    "\n".join(f"- {s}" for s in _company_services)
    if _company_services
    else "- (No service list configured; be honest about what you don't know.)"
)

ENDPOINTING_MAX_DELAY = float(os.environ.get("ENDPOINTING_MAX_DELAY", "2.5"))
STT_MODEL = os.environ.get("STT_MODEL", "nova-3").split("/")[-1] or "nova-3"
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "onwK4e9ZLuTAKqWW03F9")  # Daniel
ELEVEN_MODEL = os.environ.get("ELEVEN_MODEL", "eleven_multilingual_v2")

_EN_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _week_ahead() -> str:
    today = now_local()
    rows = []
    for i in range(8):
        d = today + timedelta(days=i)
        tag = " (today)" if i == 0 else " (tomorrow)" if i == 1 else ""
        rows.append(f"{_EN_DAYS[d.weekday()]} = {d.strftime('%Y-%m-%d')}{tag}")
    return "; ".join(rows)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
async def _display_llm_metrics(metrics: LLMMetrics) -> None:
    logger.info(
        "LLM | TTFT=%.2fs | %d in / %d out | tok/s=%.1f",
        getattr(metrics, "ttft", 0.0) or 0.0,
        metrics.prompt_tokens,
        metrics.completion_tokens,
        metrics.tokens_per_second,
    )


async def _display_stt_metrics(metrics: STTMetrics) -> None:
    logger.info("STT | text=%s", getattr(metrics, "text", "")[:120])


async def _display_tts_metrics(metrics: TTSMetrics) -> None:
    logger.info("TTS | TTFB=%.2fs | chars=%d", getattr(metrics, "ttfb", 0.0) or 0.0, metrics.characters_count)


def _build_stt():
    return deepgram.STT(model=STT_MODEL, language="multi")


def _build_tts():
    return elevenlabs.TTS(voice_id=ELEVEN_VOICE_ID, model=ELEVEN_MODEL)


def _build_llm():
    def make(name: str) -> openai.LLM:
        return openai.LLM(
            model=name,
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            reasoning_effort="low",
            max_completion_tokens=80,
        )

    primary = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
    fallback = os.environ.get("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b")
    return llm_mod.FallbackAdapter([make(primary), make(fallback)], attempt_timeout=8.0)


# --------------------------------------------------------------------------- #
# Stripping gpt-oss envelopes                                                    #
# --------------------------------------------------------------------------- #
_THINK_OPEN = ("<think>", "<|start|>assistant<|channel|>analysis", "<|channel|>analysis<|message|>")
_THINK_CLOSE = ("</think>", "<|end|>")
_STRAY_CLEAN = ("<|start|>", "<|channel|>final<|message|>", "<|message|>", "<|end|>", "<|start|>assistant")


async def _without_reasoning_envelopes(stream):
    """Strip gpt-oss <think>/channel envelopes from the model output."""
    in_think = False
    async for item in stream:
        delta = getattr(item, "delta", None)
        if delta is None or delta.content is None:
            yield item
            continue
        text = delta.content
        for tag in _THINK_OPEN:
            if tag in text:
                in_think = True
                text = text.replace(tag, "")
        for tag in _THINK_CLOSE:
            if tag in text:
                in_think = False
                text = text.replace(tag, "")
        for tag in _STRAY_CLEAN:
            text = text.replace(tag, "")
        if in_think or not text:
            continue
        delta.content = text
        yield item


def _cut_after_question(text: str, spoken: int) -> tuple[str, bool]:
    min_cut = -1
    for marker in ("?", "؟"):
        i = text.find(marker)
        if i >= 0 and (min_cut < 0 or i < min_cut):
            min_cut = i
    if min_cut >= 0:
        return text[: min_cut + 1], True
    return text, spoken > 300 and bool(re.search(r"[.!۔\n]", text))


async def _one_question_only(stream):
    """Let at most one question through; drop the rest of the reply."""
    spoken = 0
    finished = False
    async for item in stream:
        if isinstance(item, str):
            if finished:
                continue
            text, finished = _cut_after_question(item, spoken)
            spoken += len(text)
            if text:
                yield text
            continue
        delta = getattr(item, "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        if delta is None or not content:
            yield item
            continue
        if finished:
            text = ""
        else:
            text, finished = _cut_after_question(content, spoken)
            spoken += len(text)
        if text:
            if text != content:
                item.delta.content = text
            yield item
        elif getattr(delta, "tool_calls", None):
            item.delta.content = None
            yield item


# --------------------------------------------------------------------------- #
# Fixed lines spoken by code — the flow decides which one                       #
# --------------------------------------------------------------------------- #
_LINES: dict[str, str] = {
    "ask_name": "Okay. May I have your name, please?",
    "reask_name": "Sorry, I didn't catch that. What's your name?",
    "ask_phone": "Thank you. And your phone number?",
    "reask_phone": "Sorry, could you say the full number once more?",
    "more_phone": "Go on, I'm listening.",
    "ask_email": "Thanks. Now your email, letter by letter? Say no if you don't have one.",
    "reask_email": "Sorry, please say your email again, letter by letter?",
    "go_on_email": "Okay, go on?",
    "readback_again": "Please say yes or no. Is the email correct?",
    "booking": "Okay, booking it now.",
    "failed": "Sorry, that time just got taken. Could you pick another time?",
}


def _scripted_line(key: str, flow) -> str:
    if key == "readback":
        spelled = _spell_back(flow.pending_email or "")
        return f"Your email is {spelled}. Is that correct?"
    return _LINES[key]


def _booked_line(when_iso: str) -> str:
    date, clock = when_iso.split(" ", 1)
    return f"All set! Your appointment is booked for {format_when(date, clock)}. Anything else?"


# Stages where the caller answers with a name / digits / letters; anything
# said there tells the flow (not the language engine) what is happening.
_STRUCTURED_STAGES = ("name", "phone", "email", "confirm_email")


class Assistant(Agent):
    def __init__(self) -> None:
        end_call = EndCallTool()
        self.cal_tools = CalToolset(fallback_email=_company_email)
        self.flow = self.cal_tools.flow
        today = now_local().strftime("%A, %Y-%m-%d")
        super().__init__(
            instructions=f"""You are Mike, the friendly receptionist at {_company_name}, on a live call. Keep replies warm, short (1–2 sentences) and in the same language the caller uses: Urdu in, Urdu out; English in, English out. Common dental words can stay in English. One question at a time.

BOOKING:
- First find the day and the time. Convert "kal", "parso", "Tuesday" etc. using this table (Asia/Karachi is UTC+5): {_week_ahead()}
- Then ask their name and call set_appointment_time and say nothing else in that turn. Times are clinic-local 'YYYY-MM-DD HH:MM' (3 baje = 15:00). A slot is only real if set_appointment_time says it is free.
- If it says NOT_AVAILABLE, offer two of the free times in one short question.
- Never ask for name, phone or email yourself — the system collects them right after the time is saved.
- Only say "booked" after the system confirms. The dental visit is 30 minutes.

NOT_BOOKING: Stay on dental clinic topics. Answer prices/hours/services honestly; never invent. If something needs a human, say the clinic team will call back.

Company: {_company_legal} — {_company_address}; Phone {_company_phone}; Email {_company_email}. Services: {_company_services_block}
Today: {today}.""",
            tools=[*end_call.tools, self.cal_tools],
        )

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Capture booking data exactly once per committed caller turn."""
        text = (new_message.text_content or "").strip()
        if not text:
            return
        before = self.flow.stage
        self.flow.handle_user_text(text)
        if self.flow.stage != before or self.flow.reask_key or self.flow.deferred:
            logger.info("FLOW | %s -> %s | reask=%s | %s", before, self.flow.stage, self.flow.reask_key, self.flow.describe())

    async def _booking_lines(self):
        if self.flow.stage == "booking" and not self.flow.booking_announced:
            self.flow.booking_announced = True
            yield _scripted_line("booking", self.flow) + " "
        await self.cal_tools.run_booking()
        notice, self.flow.notice = self.flow.notice, None
        if notice:
            kind, when = notice
            yield _booked_line(when) if kind == "booked" else _scripted_line("failed", self.flow)

    async def llm_node(self, chat_ctx, tools, model_settings):
        # 1) Code owns the booking questions: name, phone, email, read-back and
        #    the booking itself. The LLM never sees these turns.
        key = self.flow.next_directive()
        if key in ("booking", "notice"):
            async for line in self._booking_lines():
                yield line
            return
        if key:
            logger.info("FLOW | say %s", key)
            yield _scripted_line(key, self.flow)
            return

        # 2) Everything else is the LLM's job, with a one-line note of what is
        #    already captured so it never asks twice.
        chat_ctx.truncate(max_items=20)
        note = self.flow.state_note()
        if note:
            chat_ctx.add_message(role="system", content=note)

        stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        if asyncio.iscoroutine(stream):
            stream = await stream
        async for item in _one_question_only(_without_reasoning_envelopes(stream)):
            yield item

    async def on_enter(self) -> None:
        self.session.userdata = {}

        def _llm_wrapper(metrics: LLMMetrics):
            asyncio.create_task(_display_llm_metrics(metrics))

        def _stt_wrapper(metrics: STTMetrics):
            asyncio.create_task(_display_stt_metrics(metrics))

        def _tts_wrapper(metrics: TTSMetrics):
            asyncio.create_task(_display_tts_metrics(metrics))

        self.session.llm.on("metrics_collected", _llm_wrapper)
        self.session.stt.on("metrics_collected", _stt_wrapper)
        self.session.tts.on("metrics_collected", _tts_wrapper)

        self._prewarm_task = asyncio.create_task(self.cal_tools.prewarm())
        self.session.say(f"السلام علیکم! Hi, this is Mike from {_company_name}. How may I help you? بتائیے؟")

    async def on_exit(self) -> None:
        task = getattr(self, "_prewarm_task", None)
        if task is not None and not task.done():
            task.cancel()


def _prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.3,
        min_silence_duration=0.65,
        prefix_padding_duration=0.4,
    )


server = AgentServer(setup_fnc=_prewarm)


@server.rtc_session(agent_name="MetaByte Solutions Receptionist")
async def my_agent(ctx: agents.JobContext):
    audio_input = room_io.AudioInputOptions()
    if os.environ.get("NOISE_CANCELLATION", "on").lower() not in {"off", "none", "false"}:
        audio_input.noise_cancellation = ai_coustics.audio_enhancement(
            model=ai_coustics.EnhancerModel.QUAIL_VF_S,
        )

    session = AgentSession(
        stt=_build_stt(),
        llm=_build_llm(),
        tts=_build_tts(),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing={
                "mode": "dynamic",
                "min_delay": 0.6,
                "max_delay": ENDPOINTING_MAX_DELAY,
            },
            interruption={
                "enabled": True,
                "mode": "adaptive",
                "min_duration": 0.3,
                "min_words": 0,
                "backchannel_boundary": (0.5, 1.0),
                "false_interruption_timeout": 1.5,
            },
            preemptive_generation={"preemptive_tts": True},
        ),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(audio_input=audio_input),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
