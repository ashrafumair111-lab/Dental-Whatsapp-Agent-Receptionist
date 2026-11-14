"""Mike, the dental clinic receptionist (Urdu + English).

The LLM runs the whole conversation and sees the full history, so it never
asks twice. Code only does the things that must be exact: check the slot,
read the email back, and book exactly once.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobProcess, RunContext, TurnHandlingOptions, function_tool, room_io
from livekit.agents import llm as llm_mod
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.llm.tool_context import ToolError
from livekit.plugins import ai_coustics, deepgram, openai, silero

from cal_tools import TZ, Cal, clean_email, fmt_time, now, say_when, spell_email

logger = logging.getLogger("clinic")
logger.setLevel(logging.INFO)
load_dotenv(".env.local")

COMPANY = os.environ.get("COMPANY_NAME", "Dental Clinic")
LEGAL = os.environ.get("COMPANY_LEGAL_NAME", COMPANY)
ADDRESS = os.environ.get("COMPANY_ADDRESS", "")
PHONE = os.environ.get("COMPANY_PHONE", "")
EMAIL = os.environ.get("COMPANY_EMAIL", "")
SERVICES = "\n".join(f"- {s.strip()}" for s in os.environ.get("COMPANY_SERVICES", "").split(";") if s.strip()) \
    or "- (No service list configured; be honest about what you don't know.)"


def _cut_after_question(text: str, spoken: int) -> tuple[str, bool]:
    cut = text.find("?")
    if cut >= 0:
        return text[: cut + 1], True
    return text, spoken > 300 and bool(re.search(r"[.!\n]", text))


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


EN_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def week_ahead() -> str:
    today = now()
    rows = []
    for i in range(8):
        d = today + timedelta(days=i)
        tag = " (today)" if i == 0 else " (tomorrow)" if i == 1 else ""
        rows.append(f"{EN_DAYS[d.weekday()]} = {d:%Y-%m-%d}{tag}")
    return "; ".join(rows)


def instructions() -> str:
    return f"""You are Mike, the friendly receptionist at {COMPANY}, on a live phone call.
Reply in English only. Keep every reply to 1-2 short sentences and ask only ONE question at a time.

BOOKING, in this order:
1. Find the day and time. Dates: {week_ahead()}. Times are 24-hour in tools (3 PM = 15:00).
   If the caller only gives one, ask for the other. Use check_availability if they ask what is free.
2. Ask the name. Then ask the phone number.
3. Ask the email and tell the caller to say it slowly, letter by letter. If they have none, skip it.
4. If an email was given, call read_back_email and read EXACTLY what it returns, then wait for yes/no.
   If no: ask them to say it again, and read it back again.
5. Call book_appointment, then tell the caller the booked time and ask if they need anything else.

RULES:
- Never ask for something the caller already told you. Read the whole conversation before each question — if the answer is already there, use it instead of asking again.
- Do NOT repeat or confirm the name or the phone number. Just say "Thank you" and move on.
- Never say "booked" until book_appointment returns BOOKED.
- Once book_appointment has returned BOOKED (or ALREADY BOOKED), the booking is finished. Never ask for the day, time, name, phone or email again, and never call book_appointment again, unless the caller clearly asks for a NEW appointment.
- If the caller's last message is only a few letters of an email, they are still spelling it: reply only "Okay, go on." and wait.
- If a time is not free, offer two nearby free times in one short question.
- Stay on dental clinic topics. Answer prices, hours and services only from the info below, never invent;
  if you don't know, say the clinic team will call back.

Clinic: {LEGAL}, {ADDRESS}. Phone {PHONE}. Email {EMAIL}.
Services:
{SERVICES}
Today: {now():%A, %Y-%m-%d}. A visit is 30 minutes."""


class Assistant(Agent):
    def __init__(self) -> None:
        end_call = EndCallTool()
        super().__init__(instructions=instructions(), tools=[*end_call.tools])
        self.cal = Cal()
        self._lock = asyncio.Lock()
        self._pending_email: str | None = None
        self._booked: tuple | None = None

    # ------------------------------- tools -------------------------------- #
    @function_tool
    async def check_availability(self, ctx: RunContext, date: str) -> str:
        """List free appointment times for a date (YYYY-MM-DD, clinic time)."""
        try:
            day = datetime.strptime(date.strip(), "%Y-%m-%d").date()
        except ValueError:
            raise ToolError("date must be YYYY-MM-DD.")
        free = [t for t in await self.cal.free_slots(day) if t > now()]
        if not free:
            return f"No free times on {date}."
        shown = ", ".join(fmt_time(t) for t in free[:6])
        return f"Free on {date}: {shown}" + (f" and {len(free) - 6} more." if len(free) > 6 else ".")

    @function_tool
    async def read_back_email(self, ctx: RunContext, email: str) -> str:
        """Call this once the caller has said their email. Returns the exact words to read back."""
        clean = clean_email(email)
        if not clean:
            raise ToolError("That doesn't look like a valid email. Ask the caller to say it again letter by letter.")
        self._pending_email = clean
        return f'Say exactly: "Your email is {spell_email(clean)}. Is that correct?" then wait for yes or no.'

    @function_tool
    async def book_appointment(
        self,
        ctx: RunContext,
        date: str,
        time: str,
        name: str,
        phone: str,
        email: str = "",
    ) -> str:
        """Book the appointment. date=YYYY-MM-DD, time=24h HH:MM, email only if the caller confirmed it."""
        async with self._lock:
            digits = re.sub(r"\D", "", phone or "")
            if not 7 <= len(digits) <= 15:
                raise ToolError("The phone number looks incomplete. Ask for it once more.")

            email_ok = None
            if email:
                email_ok = clean_email(email)
                if not email_ok or email_ok != self._pending_email:
                    raise ToolError("Call read_back_email first and get the caller's yes.")

            try:
                start = datetime.strptime(f"{date.strip()} {time.strip()}", "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
            except ValueError:
                raise ToolError("date must be YYYY-MM-DD and time 24-hour HH:MM.")

            # Check "already booked" BEFORE the "time has passed" / availability
            # checks, so a repeated call right after booking never errors or re-books.
            if self._booked == (start, digits):
                return f"ALREADY BOOKED for {say_when(start)}. Don't book again. Just ask if they need anything else."

            if start <= now():
                raise ToolError("That time has passed. Ask for a later time.")

            free = await self.cal.free_slots(start.date())
            if start not in free:
                near = sorted(free, key=lambda t: abs((t - start).total_seconds()))[:2]
                if not near:
                    return "NOT_AVAILABLE: no free times that day. Ask for another day."
                return f"NOT_AVAILABLE. Offer these instead: {', '.join(fmt_time(t) for t in sorted(near))}."

            await self.cal.book(start, name.strip() or "Patient", email_ok or self._fallback_email(digits), digits)
            self._booked = (start, digits)
            self._pending_email = None  # booking done: forget the email state
            logger.info("BOOKED %s %s %s", start, name, digits)
            return (
                f"BOOKED for {say_when(start)}. Tell the caller the time, then ask if they need anything else. "
                "Do not ask for any details again."
            )

    @staticmethod
    def _fallback_email(digits: str) -> str:
        if "@" in EMAIL:
            local, _, domain = EMAIL.partition("@")
            return f"{local.split('+')[0]}+{digits}@{domain}"
        return f"noemail+{digits}@example.com"

    async def llm_node(self, chat_ctx, tools, model_settings):
        # NOTE: no chat_ctx.truncate() here. Cutting the history made the LLM
        # forget the name/phone/BOOKED result and ask for them again (the loop).
        # A phone call is short; the full history is small for gpt-oss.
        stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        if asyncio.iscoroutine(stream):
            stream = await stream
        async for item in _one_question_only(stream):
            yield item

    # ----------------------------- lifecycle ------------------------------ #
    async def on_enter(self) -> None:
        self.session.say(f"Hi, this is Mike from {COMPANY}. How may I help you today?")

    async def on_exit(self) -> None:
        await self.cal.aclose()


# --------------------------------- setup ---------------------------------- #
def _build_llm():
    def make(name: str) -> openai.LLM:
        return openai.LLM(
            model=name,
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            reasoning_effort="low",
            max_completion_tokens=600,  # reasoning tokens count too; too low = silent agent
        )

    return llm_mod.FallbackAdapter(
        [make(os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")),
         make(os.environ.get("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b"))],
        attempt_timeout=10.0,
    )


def _prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.5,
        min_silence_duration=0.8,  # was 0.65: callers pause while spelling an email
        prefix_padding_duration=0.4,
    )


def _turn_handling() -> TurnHandlingOptions:
    kwargs = dict(
        turn_detection="vad",
        # More patient: wait longer before deciding the caller has finished.
        endpointing={"mode": "dynamic", "min_delay": 1.0, "max_delay": 4.0},
        interruption={
            "enabled": True,
            "mode": "adaptive",
            "min_duration": 0.5,
            "min_words": 1,
            "false_interruption_timeout": 1.5,
        },
    )
    try:
        # Don't start replying on partial transcripts (causes overlapping replies).
        return TurnHandlingOptions(**kwargs, preemptive_generation=False)
    except Exception:
        logger.warning("preemptive_generation option not supported in this livekit version; skipping it")
        return TurnHandlingOptions(**kwargs)


server = AgentServer(setup_fnc=_prewarm)


@server.rtc_session(agent_name="MetaByte Solutions Receptionist")
async def my_agent(ctx: agents.JobContext):
    audio_input = room_io.AudioInputOptions()
    if os.environ.get("NOISE_CANCELLATION", "on").lower() not in {"off", "none", "false"}:
        audio_input.noise_cancellation = ai_coustics.audio_enhancement(
            model=ai_coustics.EnhancerModel.QUAIL_VF_S
        )

    tts_instance = deepgram.TTS(model=os.environ.get("TTS_EN_MODEL", "aura-2-arcas-en"))

    session = AgentSession(
        stt=deepgram.STT(model=os.environ.get("STT_MODEL", "nova-3").split("/")[-1], language="en"),
        llm=_build_llm(),
        tts=tts_instance,
        vad=ctx.proc.userdata["vad"],
        turn_handling=_turn_handling(),
    )

    # Debug logs: see exactly what the agent hears and says.
    @session.on("user_input_transcribed")
    def _heard(ev):
        if getattr(ev, "is_final", False):
            logger.info("HEARD: %s", ev.transcript)

    @session.on("conversation_item_added")
    def _item(ev):
        item = ev.item
        logger.info("%s: %s", getattr(item, "role", "?"), getattr(item, "text_content", ""))

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(audio_input=audio_input),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)