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
from livekit.plugins import ai_coustics, deepgram, elevenlabs, openai, silero

from cal_tools import DAYS, TZ, Cal, clean_email, fmt_time, now, say_when, spell_email

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


UR_DAYS = ["پیر", "منگل", "بدھ", "جمعرات", "جمعہ", "ہفتہ", "اتوار"]


def week_ahead() -> str:
    today = now()
    rows = []
    for i in range(8):
        d = today + timedelta(days=i)
        tag = " (آج)" if i == 0 else " (کل)" if i == 1 else ""
        rows.append(f"{UR_DAYS[d.weekday()]} = {d:%Y-%m-%d}{tag}")
    return "; ".join(rows)


def instructions() -> str:
    return f"""آپ ڈینٹل کلینک کے فرینڈلی ریسپشنسٹ مائیک ہیں، ایک لائیو فون کال پر۔ پورے جواب صرف اردو میں دیں — انگریزی الفاظ استعمال نہ کریں (صرف اشد ضروری بات، جیسے دوائی کے نام یا ویسے عام نام)۔ہر جواب 1-2 چھوٹی جملوں کا ہو اور ہمیشہ صرف ایک ہی سوال کریں۔

بکنگ کا ترتیب یہ ہے:
1. پہلے دن اور وقت پوچھیں۔ تاریخیں: {week_ahead()}۔ ٹائم 24 گھنٹے ٹائم میں tools میں (3 بجے = 15:00)۔
   اگر ایک بات ملی ہے تو دوسری پوچھیں۔ اگر مَریز پوچھے "کون سے ٹائم فارغ ہے" تو check_availability استعمال کریں۔
2. نام پوچھیں۔ پھر فون نمبر پوچھیں۔
3. ای میل پوچھیں، ہجے کے ساتھ۔ اگر مَریز کے پاس ای میل نہ ہو تو چھوڑ دیں۔
4. اگر ای میل دی گئی ہے تو read_back_email کال کریں اور بالکل وہی الفاظ پڑھ کر سنائیں جو ٹول دے، پھر ہاں یا نہیں کا انتظار کریں۔
   اگر نہیں تو ای میل دوبارہ کہلوائیں اور جواب دوبارہ پڑھ کر سنائیں۔
5. phir book_appointment کال کریں اور مَریز کو بکنگ کا وقت بتا کر پوچھیں کچھ اور مدد چاہیے یا نہیں۔

اصول:
- جو چیز مَریز پہلے بتا چکا ہو، وہ کبھی دوبارہ نہ پوچھیں۔ ہر سوال کرنے سے پہلے پوری بات چیت پڑھیں — اگر جواب پہلے ہی موجود ہے تو اس کا استعمال کریں، دوبارہ نہ پوچھیں۔
- نام اور فون نمبر کبھی دہرائیے نہیں۔ صرف "شکریہ" کہہ کر آگے بڑھیں۔
- "بکنگ ہو گئی" جب ہی کہیں جب book_appointment BOOKED دے۔
- اگر جو ٹائم مانگا ہے وہ فارغ نہیں تو دو نزدیک کے وقت ایک مختصر سوال میں پیش کریں۔
- صرف ڈینٹل کلینک کے موضوع پر رہیں۔ قیمتیں، گھنٹے اور سروسز صرف نیچے دی معلومات سے؛ کبھی ایجاد نہ کریں۔
  اگر معلومات نہ ہوں تو کہیں کہ کلینک ٹیم واپس کال کرے گی۔

کلینک: {LEGAL}, {ADDRESS}. فون {PHONE}. ای میل {EMAIL}.
خدمات:
{SERVICES}
آج: {now():%A, %Y-%m-%d}۔ وزٹ 30 منٹ کا ہے۔"""


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
            if start <= now():
                raise ToolError("That time has passed. Ask for a later time.")

            if self._booked == (start, digits):
                return f"ALREADY BOOKED for {say_when(start)}. Don't book again."

            free = await self.cal.free_slots(start.date())
            if start not in free:
                near = sorted(free, key=lambda t: abs((t - start).total_seconds()))[:2]
                if not near:
                    return "NOT_AVAILABLE: no free times that day. Ask for another day."
                return f"NOT_AVAILABLE. Offer these instead: {', '.join(fmt_time(t) for t in sorted(near))}."

            await self.cal.book(start, name.strip() or "Patient", email_ok or self._fallback_email(digits), digits)
            self._booked = (start, digits)
            logger.info("BOOKED %s %s %s", start, name, digits)
            return f"BOOKED for {say_when(start)}. Tell the caller, then ask if they need anything else."

    @staticmethod
    def _fallback_email(digits: str) -> str:
        if "@" in EMAIL:
            local, _, domain = EMAIL.partition("@")
            return f"{local.split('+')[0]}+{digits}@{domain}"
        return f"noemail+{digits}@example.com"

    # ----------------------------- lifecycle ------------------------------ #
    async def on_enter(self) -> None:
        self.session.say("السلام علیکم! یہ عمر اشرف ڈینٹل کلینک ہے، میں مائیک ہوں۔ بتائیے، کیا مدد کر سکتا ہوں؟")

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
            max_completion_tokens=400,  # was 80: reasoning tokens ate the whole budget -> silence
        )

    return llm_mod.FallbackAdapter(
        [make(os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")),
         make(os.environ.get("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b"))],
        attempt_timeout=8.0,
    )


def _prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.4, min_silence_duration=0.65, prefix_padding_duration=0.4
    )


server = AgentServer(setup_fnc=_prewarm)


@server.rtc_session(agent_name="MetaByte Solutions Receptionist")
async def my_agent(ctx: agents.JobContext):
    audio_input = room_io.AudioInputOptions()
    if os.environ.get("NOISE_CANCELLATION", "on").lower() not in {"off", "none", "false"}:
        audio_input.noise_cancellation = ai_coustics.audio_enhancement(
            model=ai_coustics.EnhancerModel.QUAIL_VF_S
        )

    tts_instance = elevenlabs.TTS(
        voice_id=os.environ.get("ELEVEN_VOICE_ID", "onwK4e9ZLuTAKqWW03F9"),
        model=os.environ.get("ELEVEN_MODEL", "eleven_multilingual_v2"),
        language="ur",
    )
    # Open the ElevenLabs websocket now so the first reply doesn't wait on a cold connect.
    tts_instance.prewarm()

    session = AgentSession(
        stt=deepgram.STT(model=os.environ.get("STT_MODEL", "nova-3").split("/")[-1], language="ur"),
        llm=_build_llm(),
        tts=tts_instance,
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing={"mode": "dynamic", "min_delay": 0.6, "max_delay": 2.5},
            interruption={
                "enabled": True,
                "mode": "adaptive",
                "min_duration": 0.5,
                "min_words": 1,  # was 0: any noise cut the agent off mid-sentence
                "false_interruption_timeout": 1.5,
            },
        ),
    )
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(audio_input=audio_input),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)