"""Mike — the dental clinic receptionist.

On purpose this is a small file:

- STT: Deepgram nova-3 multilingual, the whole call in one model.
- TTS: one engine per language, picked per reply by the script of the text:
  Urdu -> Edge TTS (ur-PK-AsadNeural, native voice), anything else -> Deepgram Aura.
- The agent asks exactly one question per reply, never invents a booking, and
  the only two tools are check_availability and create_booking. create_booking
  collects and confirms email + phone itself.
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
    llm as llm_mod,
    tts as tts_mod,
    room_io,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics
from livekit.plugins import ai_coustics, deepgram, openai, silero

from cal_tools import CalToolset, now_local
from edge_tts_plugin import EdgeTTS

logger = logging.getLogger("clinic")
logger.setLevel(logging.INFO)

load_dotenv(".env.local")

_company_name = os.environ.get("COMPANY_NAME", "Dental Clinic")
_company_legal = os.environ.get("COMPANY_LEGAL_NAME", _company_name)
_company_address = os.environ.get("COMPANY_ADDRESS", "")
_company_phone = os.environ.get("COMPANY_PHONE", "")
_company_email = os.environ.get("COMPANY_EMAIL", "")

# Services, one per line in COMPANY_SERVICES (";"-separated).
_company_services_raw = os.environ.get("COMPANY_SERVICES", "")
_company_services = [s.strip() for s in _company_services_raw.split(";") if s.strip()]
_company_services_block = (
    "\n".join(f"- {s}" for s in _company_services)
    if _company_services
    else "- (No service list configured; be honest about what you don't know.)"
)

ENDPOINTING_MAX_DELAY = float(os.environ.get("ENDPOINTING_MAX_DELAY", "2.5"))
STT_MODEL = os.environ.get("STT_MODEL", "nova-3").split("/")[-1] or "nova-3"
TTS_EN_MODEL = os.environ.get("TTS_EN_MODEL", "aura-2-arcas-en")
TTS_UR_VOICE = os.environ.get("TTS_UR_VOICE", "ur-PK-AsadNeural")
TTS_UR_SPEED = float(os.environ.get("TTS_UR_SPEED", "1.15") or "1.15")

_URDU_SCRIPT = re.compile(r"[؀-ۿ\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

# A few Roman Urdu words so a Latin-script Urdu reply still picks the Urdu voice.
_ROMAN_URDU_HINTS = {
    "hai", "hain", "hoon", "hun", "haan", "ji", "aap", "ap", "apko", "aapko", "mera", "meri",
    "mein", "main", "ka", "ki", "ke", "kya", "kitne", "kitna", "kisne", "kab", "kahan",
    "kal", "aaj", "thik", "theek", "nam", "naam", "bata", "batain", "batayein", "karna",
    "karni", "karo", "booking", "milne", "shukriya", "shukria", "hoga", "hogi",
}


def _looks_urdu(text: str) -> bool:
    if _URDU_SCRIPT.search(text or ""):
        return True
    words = re.findall(r"[a-z']+", (text or "").lower())
    return sum(w in _ROMAN_URDU_HINTS for w in words) >= 2

_EN_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_UR_DAYS = ["پیر", "منگل", "بدھ", "جمعرات", "جمعہ", "ہفتہ", "اتوار"]


def _week_ahead() -> str:
    today = now_local()
    rows = []
    for i in range(8):
        d = today + timedelta(days=i)
        tag = " (today)" if i == 0 else " (tomorrow)" if i == 1 else ""
        rows.append(f"{_EN_DAYS[d.weekday()]}/{_UR_DAYS[d.weekday()]} = {d.strftime('%Y-%m-%d')}{tag}")
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


# --------------------------------------------------------------------------- #
# Text -> voice routing                                                        #
# --------------------------------------------------------------------------- #
class MixedTTS(tts_mod.TTS):
    """Urdu script -> Edge (ur-PK-AsadNeural), anything else -> Deepgram Aura."""

    def __init__(self, english: tts_mod.TTS, urdu: tts_mod.TTS) -> None:
        super().__init__(
            capabilities=tts_mod.TTSCapabilities(streaming=False),
            sample_rate=english.sample_rate,
            num_channels=1,
        )
        self._english = english
        self._urdu = urdu

    @property
    def model(self) -> str:
        return "aura + edge"

    @property
    def provider(self) -> str:
        return "deepgram + microsoft-edge"

    def _active(self, text: str) -> tts_mod.TTS:
        return self._urdu if _looks_urdu(text) else self._english

    def synthesize(
        self,
        text: str,
        *,
        conn_options=None,
    ) -> tts_mod.ChunkedStream:
        kwargs = {} if conn_options is None else {"conn_options": conn_options}
        return self._active(text).synthesize(text, **kwargs)

    def prewarm(self) -> None:
        self._english.prewarm()
        if hasattr(self._urdu, "prewarm"):
            self._urdu.prewarm()

    async def aclose(self) -> None:
        await self._english.aclose()
        await self._urdu.aclose()


def _build_tts() -> MixedTTS:
    english = deepgram.TTS(model=TTS_EN_MODEL)
    urdu = EdgeTTS(voice=TTS_UR_VOICE, rate=TTS_UR_SPEED)
    return MixedTTS(english, urdu)


def _build_stt():
    return deepgram.STT(model=STT_MODEL, language="multi")


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
# One question per reply                                                       #
# --------------------------------------------------------------------------- #
def _cut_after_question(text: str) -> tuple[str, bool]:
    cut = min((i for i in (text.find("?"), text.find("؟")) if i >= 0), default=-1)
    return (text[: cut + 1], True) if cut >= 0 else (text, False)


async def _one_question_only(stream):
    """Let at most one question through; drop the rest of the reply."""
    finished = False
    async for item in stream:
        if isinstance(item, str):
            if finished:
                continue
            text, finished = _cut_after_question(item)
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
            text, finished = _cut_after_question(content)
        if text:
            if text != content:
                item.delta.content = text
            yield item
        elif getattr(delta, "tool_calls", None):
            item.delta.content = None
            yield item


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
            continue  # drain the stream, but never speak reasoning
        delta.content = text
        yield item


GREETING = "السلام علیکم! یہ عمر اشرف ڈینٹل کلینک ہے، میں مائیک ہوں۔ بتائیے، کیا مدد کر سکتا ہوں؟"


class Assistant(Agent):
    def __init__(self) -> None:
        end_call = EndCallTool()
        self.cal_tools = CalToolset()
        today = now_local().strftime("%A, %Y-%m-%d")
        super().__init__(
            instructions=f"""You are Mike, a warm, friendly receptionist at {_company_name}, on a live phone call. Keep it short and human.

- Reply in the language the caller uses (Urdu in, Urdu reply; English in, English reply). Common dental words can stay in English inside Urdu sentences (appointment, booking, root canal, X-ray, check-up).
- One question at a time; 1–2 short sentences. Never list steps or say "waiting", "saved" or "checking".

BOOKING:
- First find the day and the time they want. Convert "Tuesday" and "kal"/"parso" using this table (Asia/Karachi is UTC+5, no DST): {_week_ahead()}
- Then ask their name.
- Then call create_booking(start_time=..., attendee_name=...) and say nothing else in that turn. `start_time` is clinic-local as 'YYYY-MM-DD HH:MM' ("2026-11-20 15:30" = 3:30 PM). create_booking asks for and confirms the caller's email and phone itself — never ask for them yourself, never invent them.
- A dental visit is 30 minutes with the doctor.
- Only say "booked" after create_booking confirms.

CANCEL OR RESCHEDULE:
- There is no self-service cancel or reschedule. Ask for the booking UID and their name, then say the clinic team will help and they should call us again. Never invent a booking.

- Stay on dental clinic topics. If you don't know something, say so honestly. Never invent services, prices or hours.
Services: {_company_services_block}
Company: {_company_legal} (trading as {_company_name}) — {_company_address}; Phone {_company_phone}; Email {_company_email}. Share only when asked.

Today: {today}.""",
            tools=[*end_call.tools, self.cal_tools],
        )

    async def llm_node(self, chat_ctx, tools, model_settings):
        chat_ctx.truncate(max_items=16)
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

        # Warm today's slots while the greeting plays.
        self._prewarm_task = asyncio.create_task(self.cal_tools.prewarm())

        self.session.say(GREETING)

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
    tts_instance = _build_tts()
    stt_instance = _build_stt()
    llm_instance = _build_llm()

    audio_input = room_io.AudioInputOptions()
    if os.environ.get("NOISE_CANCELLATION", "on").lower() not in {"off", "none", "false"}:
        audio_input.noise_cancellation = ai_coustics.audio_enhancement(
            model=ai_coustics.EnhancerModel.QUAIL_VF_S,
        )

    session = AgentSession(
        stt=stt_instance,
        llm=llm_instance,
        tts=tts_instance,
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
