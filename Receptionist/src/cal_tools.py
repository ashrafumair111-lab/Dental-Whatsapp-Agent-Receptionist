"""Cal.com integration for the dental receptionist (English).

Design rule: the LLM never *holds* booking data. It only finds the day and
time ("Tuesday 3 baje") through ``set_appointment_time``. Everything after that
(name -> phone -> email -> read-back -> booking) is a small state machine in
``BookingFlow``, driven by code, so no question can ever be asked twice and
the final booking is always made by code, exactly once, with any phone format.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Annotated
from zoneinfo import ZoneInfo

import httpx
from livekit.agents import RunContext, llm
from livekit.agents.llm.tool_context import ToolError

logger = logging.getLogger("clinic")

CAL_API_BASE = "https://api.cal.com/v2"
CAL_API_VERSION = "2026-02-25"
HTTP_TIMEOUT = 15
MAX_RETRIES = 3
BASE_BACKOFF = 0.4

APPOINTMENT_SLUG = "30min"
APPOINTMENT_DISPLAY = "Dental Appointments"
DEFAULT_TZ = "Asia/Karachi"

MAX_FIELD_TRIES = 3
MAX_EMAIL_TRIES = 4


def now_local(tz_name: str | None = None) -> datetime:
    return datetime.now(_tz(tz_name))


def _tz(name: str | None = None) -> tzinfo:
    name = (name or DEFAULT_TZ).strip() or DEFAULT_TZ
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone(timedelta(hours=5)) if name == DEFAULT_TZ else timezone.utc


def _parse_iso(iso_str: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat((iso_str or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _format_time(iso_str: str) -> str:
    dt = _parse_iso(iso_str)
    if dt is None:
        return iso_str
    return dt.astimezone(_tz(DEFAULT_TZ)).strftime("%I:%M %p").lstrip("0")


_EN_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_EN_MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]


def format_when(date: str, clock: str) -> str:
    """Spoken 'tomorrow at 3:00 PM' / 'Tuesday, 17 November'."""
    try:
        dt = datetime.strptime(f"{date} {clock}", "%Y-%m-%d %H:%M")
    except ValueError:
        return f"{date} {clock}"
    delta = (dt.date() - now_local().date()).days
    if delta == 0:
        day = "today"
    elif delta == 1:
        day = "tomorrow"
    else:
        day = f"{_EN_DAYS[dt.weekday()]}, {dt.day} {_EN_MONTHS[dt.month - 1]}"
    hour, minute = dt.hour % 12 or 12, dt.minute
    return f"{day} at {hour}:{minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


# --------------------------------------------------------------------------- #
# Slots cache                                                                 #
# --------------------------------------------------------------------------- #
_slots_cache: dict[str, tuple[float, list]] = {}
SLOTS_CACHE_TTL = 30


def _get_cached_slots(key: str) -> list | None:
    entry = _slots_cache.get(key)
    if entry and (time.time() - entry[0]) < SLOTS_CACHE_TTL:
        return entry[1]
    return None


def _set_cached_slots(key: str, slots: list) -> None:
    _slots_cache[key] = (time.time(), slots)


def _drop_cached_slots(date: str) -> None:
    for key in [k for k in _slots_cache if k.startswith(f"{date}|")]:
        _slots_cache.pop(key, None)


# --------------------------------------------------------------------------- #
# Spoken text helpers                                                          #
# --------------------------------------------------------------------------- #
_EASTERN_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹०१२३४५६७८९", "0123456789" * 3
)

_DIGIT_WORDS = {
    "zero": "0", "sifar": "0", "shunya": "0", "ek": "1", "one": "1", "aik": "1",
    "two": "2", "do": "2", "teen": "3", "three": "3",
    "four": "4", "char": "4", "chaar": "4", "paanch": "5", "five": "5", "panch": "5",
    "six": "6", "ch": "6", "che": "6", "chhe": "6",
    "seven": "7", "saat": "7", "aath": "8", "eight": "8",
    "nine": "9", "nau": "9", "nau": "9",
    "double": "2", "triple": "3",
}
_MULTIPLIERS = {"double": 2, "dabal": 2, "triple": 3}
_MULTIPLIER_RECOGNIZER = re.compile(r"^(double|dabal|triple|ڈبل|ٹرپل)$")

_WORD_SPLIT = re.compile(r"[^\s.,!?؟۔;:()\",\"\u201C\u201D,/\-–—_]+")


def _split_words(text: str) -> list[str]:
    return _WORD_SPLIT.findall((text or "").lower())


def _digits_from_speech(text: str, *, allow_words: bool = True) -> str:
    out: list[str] = []
    repeat = 1
    for tok in _split_words((text or "").translate(_EASTERN_DIGITS)):
        if allow_words and tok in _MULTIPLIERS:
            repeat = _MULTIPLIERS[tok]
            continue
        if tok.isascii() and tok.isdigit():
            out.append(tok * (repeat if len(tok) == 1 else 1))
        elif allow_words and tok in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[tok] * repeat)
        repeat = 1
    return "".join(out)


def _digits_from_phone_text(text: str) -> str:
    return re.sub(r"\D", "", (text or "").translate(_EASTERN_DIGITS))


def _loose_phone(digits: str) -> str | None:
    """Any reasonable phone number, any country code."""
    d = re.sub(r"\D", "", digits or "")
    if len(d) < 7 or len(d) > 15:
        return None
    return d


def _normalised_e164_default(digits: str) -> str:
    return digits  # keep as spoken — the desk will dial whatever came in


# --------------------------------------------------------------------------- #
# Name parsing + yes/no                                                        #
# --------------------------------------------------------------------------- #
_NAME_LEAD = re.compile(r"^(?:(?:ji|jee|haan|han|ok|okay|so|well|hello|hi)\b[\s,.]*)+", re.IGNORECASE)
_NAME_PREFIX = re.compile(
    r"^(?:my\s+name\s+is|my\s+name's|name\s+is|this\s+is|i\s+am|i'm|im|"
    r"mera\s+naam|mera\s+nam|naam|main|mera)\s+",
    re.IGNORECASE,
)
_NAME_SUFFIX = re.compile(
    r"[\s,.]*(?:\b(?:hai|hain|he|hoon|hun|sir|bhai|sahab|sahib|madam|maam|please|plz|shukriya|thanks|thank you)\b|ہے|ہوں|ہیں|شکریہ)[\s.!?۔]*$",
    re.IGNORECASE,
)

_NOT_NAME = {
    "kya", "kyun", "kyon", "kab", "kahan", "kaise", "kitna", "kitne", "kitni",
    "what", "how", "when", "where", "why", "which", "the", "is", "are", "can",
    "price", "fee", "fees", "cost", "charges", "appointment", "booking", "book",
    "doctor", "dr", "tooth", "teeth", "dant", "dard", "pain", "hai", "hain",
    "nahi", "nahin", "nhi", "no", "yes", "haan", "han", "ji", "jee", "ok", "okay",
    "theek", "thik", "acha", "achha", "sure", "please", "hello", "hi", "salam",
    "assalam", "shukriya", "thanks", "thank", "you", "phone", "number", "email",
    "mail", "wait", "ruko", "minute", "second", "hold", "sorry", "pardon",
    "again", "repeat", "dobara", "phir", "kal", "aaj", "parso", "tomorrow", "today",
    "hmm", "hm", "hmmm", "um", "umm", "uh", "ah", "oh", "aha", "haha",
}

_QUESTION_WORDS = {
    "kya", "kyun", "kyon", "kaise", "kitna", "kitne", "kitni", "kab", "kahan",
    "kaun", "kon", "what", "how", "why", "when", "where", "which", "who",
    "price", "fee", "fees", "cost", "charges", "timing", "timings",
}
_YES_WORDS = {
    "yes", "yeah", "yep", "yup", "correct", "right", "sahi", "sahih", "theek",
    "thik", "haan", "han", "ha", "hn", "ji", "jee", "bilkul", "ok", "okay",
    "sure", "exactly", "yes", "yeah", "yup",
}
_NO_WORDS = {"no", "nope", "nahi", "nahin", "nhi", "nai", "galat", "wrong", "incorrect"}
_SKIP_EMAIL_WORDS = {"no", "none", "skip", "nahi", "nahin", "nhi", "nai"}


def looks_like_question(text: str) -> bool:
    if "?" in text or "؟" in text:
        return True
    return bool(set(_split_words(text)) & _QUESTION_WORDS)


def _name_candidate(text: str) -> str | None:
    """A plausible caller name from one utterance, else ``None``."""
    t = (text or "").strip()
    t = _NAME_LEAD.sub("", t)
    t = _NAME_PREFIX.sub("", t)
    t = _NAME_SUFFIX.sub("", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s'\-؀-ۿऀ-ॿ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" -'")
    letters = re.findall(r"[A-Za-z؀-ۿऀ-ॿ]", t)
    tokens = t.split()
    if len(letters) < 2 or not 1 <= len(tokens) <= 4 or len(t) > 60:
        return None
    if any(tok.lower() in _NOT_NAME for tok in tokens):
        return None
    if t == t.lower() and re.search(r"[a-z]", t):
        t = t.title()
    return t


# --------------------------------------------------------------------------- #
# Email parsing                                                                 #
# --------------------------------------------------------------------------- #
_SPOKEN_SYMBOLS = {
    "at": "@", "at the rate": "@", "dot": ".", "period": ".", "point": ".",
    "underscore": "_", "under score": "_", "dash": "-", "hyphen": "-",
    "minus": "-", "plus": "+",
}
_EN_DIGITS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_EMAIL_FILLER = re.compile(r"\b(?:my|the|this|email|e-mail|address|is|it'?s|that'?s|its|spelled|spell)\b", re.IGNORECASE)
_SPELLED_RUN = re.compile(r"(?<![\w.\-])((?:[a-z0-9][.\-]){2,}[a-z0-9])(?![\w.\-])")
_EMAIL_STOP = {
    "hai", "haan", "han", "hn", "ha", "ji", "jee", "please", "plz", "bata",
    "bataye", "batayein", "batado", "batadein", "bol", "bolo", "raha",
    "rahi", "hun", "hoon", "hain", "mera", "meri", "wala", "wali", "kar",
    "karna", "karni", "dena", "de", "do", "dijiye", "sir", "maam", "madam",
    "sahab", "sahib", "aap", "ap", "ka", "ki", "ke", "ye", "yeh", "yah",
    "this", "that", "is", "are", "was", "for", "appointment", "booking",
    "thank", "thanks", "shukriya", "shukria", "ok", "okay", "theek", "thik",
    "aur", "and", "phone", "number", "email", "mail", "id", "my", "meri",
}
_EMAIL_LEAD = {
    "hai", "haan", "han", "hn", "is", "the", "meri", "mera", "my",
    "ye", "yeh", "that", "this", "please", "ji", "wo", "woh",
}
_TLDS = (
    "co.uk", "com.pk", "com.au", "co.in", "co.jp", "co.za", "com.sa", "com.bd",
    "com", "net", "org", "edu", "gov", "mil", "int", "info", "biz", "name",
    "pro", "io", "co", "me", "tv", "xyz", "online", "site", "store", "live",
    "app", "dev", "cloud", "pk", "uk", "us", "ca", "au", "de", "fr", "ae",
    "sa", "in", "bd", "my", "sg", "it", "es", "nl", "ch", "se", "no", "ru",
)
_TLD_CUT = re.compile(r"^(.*?\.(?:" + "|".join(re.escape(t) for t in _TLDS) + r"))")
_EMAIL_HINT = re.compile(r"@|\b(?:at|dot|point|period|email|e-mail|mail)\b", re.IGNORECASE)
_TYPED_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+")

EMAIL_REGEX = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._%+\-]*@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$"
)


def _collapse_spelled_letters(part: str) -> str:
    return _SPELLED_RUN.sub(lambda m: re.sub(r"[.\-]", "", m.group(1)), part)


def _normalize_spoken_email(raw: str) -> str:
    """Turn anything a caller says into a real address."""
    text = (raw or "").strip().lower()
    if not text:
        return ""
    if "@" in text:
        return re.sub(r"\s+", "", text).rstrip(".,;:")
    text = _EMAIL_FILLER.sub(" ", text)
    for spoken in sorted(_SPOKEN_SYMBOLS, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(spoken)}\b", _SPOKEN_SYMBOLS[spoken], text)
    text = re.sub(r"\s*([@._\-+])\s*", r"\1", text)
    text = re.sub(r"\s+", "", text)
    if "@" not in text:
        return text
    local, _, domain = text.partition("@")
    local = _collapse_spelled_letters(local)
    domain = _collapse_spelled_letters(domain)
    return f"{local}@{domain}".rstrip(".,;:")


def email_from_speech(text: str) -> str | None:
    """Pull a real email address out of one caller transcript, else ``None``."""
    raw = (text or "").strip()
    if not raw or not _EMAIL_HINT.search(raw):
        return None

    typed = _TYPED_EMAIL.search(raw)
    if typed:
        candidate = typed.group(0).rstrip(".,;:")
    else:
        lowered = raw.lower().replace("at the rate", "at")
        tokens = re.sub(r"[^a-z0-9@._+\- ]", " ", lowered).split()
        tokens = [_EN_DIGITS.get(t, t) for t in tokens]
        for i, tok in enumerate(tokens):
            if tok in {"email", "mail", "id"}:
                tokens = tokens[i + 1:]
                break
        while tokens and tokens[0] in _EMAIL_LEAD:
            tokens.pop(0)
        picked: list[str] = []
        for tok in tokens:
            if tok in _EMAIL_STOP and picked:
                break
            picked.append(tok)
            if len(picked) >= 40:
                break
        candidate = " ".join(picked)

    candidate = _normalize_spoken_email(candidate)
    if "@" not in candidate:
        return None

    local, _, domain = candidate.partition("@")
    cut = _TLD_CUT.match(domain)
    if cut:
        candidate = f"{local}@{cut.group(1)}"

    return candidate if EMAIL_REGEX.match(candidate) else None


def _spell_back(email: str) -> str:
    """Read an address letter by letter, the way a receptionist does."""
    local, _, domain = email.partition("@")

    def _letters(part: str) -> str:
        spoken = []
        for ch in part:
            if ch == ".":
                spoken.append("dot")
            elif ch == "_":
                spoken.append("underscore")
            elif ch == "-":
                spoken.append("dash")
            elif ch == "+":
                spoken.append("plus")
            else:
                spoken.append(ch)
        return " ".join(spoken)

    domain_spoken = " dot ".join(domain.split("."))
    return f"{_letters(local)} at {domain_spoken}"


# --------------------------------------------------------------------------- #
# The booking state machine                                                     #
# --------------------------------------------------------------------------- #
class BookingFlow:
    """What has been captured and what to speak next. Pure state, no I/O.

    time -> name -> phone -> email -> confirm_email -> booking -> done
    The ``time`` stage is the LLM's; every other stage is spoken here.
    """

    def __init__(self, fallback_email: str = "") -> None:
        self.fallback_email = fallback_email
        self.last_booking_uid: str | None = None
        self.reset()

    def reset(self) -> None:
        self.date: str | None = None
        self.time: str | None = None
        self.name: str | None = None
        self.phone: str | None = None
        self.email: str | None = None
        self.email_skipped = False
        self.phone_skipped = False
        self.stage = "time"
        self.prompted = ""
        self.reask_key: str | None = None
        self.deferred = False
        self.attempts = {"name": 0, "phone": 0, "confirm": 0}
        self.email_tries = 0
        self.phone_buf = ""
        self.email_buf = ""
        self.pending_email: str | None = None
        self.notice: tuple[str, str] | None = None
        self.booking_announced = False

    def _first_missing_stage(self) -> str:
        if not self.name:
            return "name"
        if not (self.phone or self.phone_skipped):
            return "phone"
        if not (self.email or self.email_skipped):
            return "email"
        return "booking"

    def set_slot(self, date: str, clock: str) -> None:
        if self.stage == "done":
            self.reset()
        self.date, self.time = date, clock
        self.reask_key = None
        if self.stage == "time":
            self.stage = self._first_missing_stage()

    def booking_email(self) -> str:
        if self.email:
            return self.email
        digits = re.sub(r"\D", "", self.phone or "") or "caller"
        base = (self.fallback_email or "").strip()
        if "@" in base:
            local, _, domain = base.partition("@")
            return f"{local.split('+')[0]}+{digits}@{domain}"
        return f"noemail+{digits}@example.com"

    def handle_user_text(self, text: str) -> None:
        self.deferred = False
        self.reask_key = None
        stage = self.stage
        if stage == "name":
            self._take_name(text)
        elif stage == "phone":
            self._take_phone(text)
        elif stage == "email":
            self._take_email(text)
        elif stage == "confirm_email":
            self._take_confirm(text)

    def _take_name(self, text: str) -> None:
        cand = _name_candidate(text)
        if cand:
            self.name = cand
            self.phone = self._extract_phone(text)
            self.stage = self._first_missing_stage()
            return
        if looks_like_question(text):
            self.deferred = True
            return
        self.attempts["name"] += 1
        if self.attempts["name"] >= MAX_FIELD_TRIES:
            self.name = "Patient"
            self.stage = self._first_missing_stage()
            return
        self.reask_key = "reask_name"

    def _extract_phone(self, text: str) -> str | None:
        digits = _digits_from_phone_text(text)
        return _loose_phone(digits)

    def _take_phone(self, text: str) -> None:
        found = self._extract_phone(text)
        if found:
            self.phone = found
            self.phone_buf = ""
            self.stage = self._first_missing_stage()
            return
        if looks_like_question(text):
            self.deferred = True
            return
        buffer_digits = self.phone_buf + _digits_from_phone_text(text)
        if buffer_digits:
            if len(buffer_digits) < 7:
                self.phone_buf = buffer_digits
                self.reask_key = "more_phone"
                return
            self.phone_buf = ""
            self.phone = buffer_digits[:15]
            self.stage = self._first_missing_stage()
            return
        self.attempts["phone"] += 1
        if self.attempts["phone"] >= MAX_FIELD_TRIES:
            self.phone_skipped = True
            self.stage = self._first_missing_stage()
            return
        self.reask_key = "reask_phone"

    def _skip_email(self) -> None:
        self.email_skipped = True
        self.pending_email = None
        self.email_buf = ""
        self.stage = "booking"

    def _take_email(self, text: str) -> None:
        buf = f"{self.email_buf} {text}".strip()
        found = email_from_speech(buf) or email_from_speech(text)
        if found:
            self.email_buf = ""
            self.pending_email = found
            self.email_tries += 1
            self.stage = "confirm_email"
            return
        words = set(_split_words(text))
        if words & _SKIP_EMAIL_WORDS or re.search(r"don'?t have|do not have", text.lower()):
            self._skip_email()
            return
        spelling = len(re.findall(r"\b[a-z0-9]\b", text.lower())) >= 2 or bool(
            words & {"at", "dot", "gmail", "yahoo", "hotmail", "outlook", "icloud"}
        )
        if looks_like_question(text) and not spelling:
            self.deferred = True
            return
        self.email_tries += 1
        if self.email_tries >= MAX_EMAIL_TRIES:
            self._skip_email()
            return
        if spelling:
            self.email_buf = buf[-160:]
            self.reask_key = "go_on_email"
        else:
            self.email_buf = ""
            self.reask_key = "reask_email"

    def _take_confirm(self, text: str) -> None:
        words = set(_split_words(text))
        found = email_from_speech(text)
        if found and found == self.pending_email:
            found = None
            words |= {"yes"}
        if found:
            self.pending_email = found
            self.email_tries += 1
            if self.email_tries >= MAX_EMAIL_TRIES:
                self.email, self.pending_email = found, None
                self.stage = "booking"
                return
            self.reask_key = "readback"
            return
        if words & _NO_WORDS:
            self.pending_email = None
            self.email_tries += 1
            if self.email_tries >= MAX_EMAIL_TRIES:
                self._skip_email()
                return
            self.stage = "email"
            self.prompted = "email"
            self.reask_key = "reask_email"
            return
        if words & _YES_WORDS:
            self.email, self.pending_email = self.pending_email, None
            self.stage = "booking"
            return
        if looks_like_question(text):
            self.deferred = True
            return
        self.attempts["confirm"] += 1
        if self.attempts["confirm"] >= 2:
            self.email, self.pending_email = self.pending_email, None
            self.stage = "booking"
            return
        self.reask_key = "readback_again"

    def next_directive(self) -> str | None:
        """The fixed line code must speak now, or None to let the LLM answer."""
        if self.notice:
            return "notice"
        if self.stage == "booking":
            return "booking"
        if self.reask_key:
            key, self.reask_key = self.reask_key, None
            self.prompted = self.stage
            return key
        if self.stage in ("name", "phone", "email") and self.prompted != self.stage:
            self.prompted = self.stage
            return f"ask_{self.stage}"
        if self.stage == "confirm_email" and self.prompted != "confirm_email":
            self.prompted = "confirm_email"
            return "readback"
        return None

    def state_note(self) -> str | None:
        if self.stage == "done":
            return "BOOKING STATE: already booked. Don't ask booking questions again. Help with anything else, or say a short goodbye."
        if not self.date:
            return None
        nxt = {
            "name": "the caller's name",
            "phone": "the caller's phone number",
            "email": "the caller's email, spelled letter by letter",
            "confirm_email": "whether the email read back is correct (yes or no)",
        }.get(self.stage)
        have = (
            f"date={self.date} time={self.time} name={'captured' if self.name else 'missing'} "
            f"phone={'captured' if self.phone else 'missing'} "
            f"email={'captured' if self.email else 'missing'}"
        )
        note = f"BOOKING STATE: {have}."
        if nxt:
            note += f" NEXT: ask only for {nxt}, once, in one short question."
            if self.deferred:
                note += " The caller asked something else first: answer it in one short sentence, then ask."
        return note

    def describe(self) -> str:
        return (
            f"stage={self.stage} date={self.date} time={self.time} name={self.name!r} "
            f"phone={self.phone!r} email={self.email!r} pending={self.pending_email!r}"
        )


# --------------------------------------------------------------------------- #
# The toolset                                                                 #
# --------------------------------------------------------------------------- #
class CalToolset(llm.Toolset):
    def __init__(self, fallback_email: str = "") -> None:
        super().__init__(id="cal")
        api_key = os.environ.get("CAL_API_KEY")
        if not api_key:
            raise RuntimeError("CAL_API_KEY not set in environment")
        self._api_key = api_key
        username = os.environ.get("CAL_USERNAME")
        if not username:
            raise RuntimeError("CAL_USERNAME not set in environment")
        self._username = username
        self.flow = BookingFlow(fallback_email=fallback_email)
        self._booking_task: asyncio.Task | None = None
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(HTTP_TIMEOUT),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=10,
                keepalive_expiry=120,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        await super().aclose()

    async def prewarm(self) -> None:
        try:
            async with asyncio.timeout(6.0):
                await self._slots_for(now_local().strftime("%Y-%m-%d"))
        except Exception:
            logger.debug("Cal.com prewarm failed", exc_info=True)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
        version: str = CAL_API_VERSION,
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "cal-api-version": version,
            "Content-Type": "application/json",
        }
        last_error: ToolError | None = None
        for attempt in range(MAX_RETRIES):
            try:
                async with asyncio.timeout(HTTP_TIMEOUT):
                    resp = await self._client.request(
                        method,
                        f"{CAL_API_BASE}{path}",
                        headers=headers,
                        json=json,
                        params=params,
                    )
            except TimeoutError as err:
                raise ToolError("Cal.com is responding slowly right now. Ask the caller to hold on a moment.") from err
            except httpx.RequestError as err:
                raise ToolError("I couldn't reach the booking system. Ask the caller to hold on a moment.") from err

            if resp.status_code < 400:
                return resp.json()
            last_error = ToolError(f"Cal.com API error ({resp.status_code}): {resp.text}")
            if resp.status_code in (429, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF * (2**attempt) + random.uniform(0, 0.25)
                await asyncio.sleep(wait)
                continue
            raise last_error
        raise last_error  # type: ignore[misc]

    async def _get(self, path: str, params: dict | None = None, version: str = CAL_API_VERSION) -> dict:
        return await self._request("GET", path, params=params, version=version)

    async def _post(self, path: str, body: dict, version: str = CAL_API_VERSION) -> dict:
        return await self._request("POST", path, json=body, version=version)

    async def _slots_for(self, date: str) -> list:
        key = f"{date}|{DEFAULT_TZ}"
        cached = _get_cached_slots(key)
        if cached is not None:
            return cached
        params = {
            "username": self._username,
            "eventTypeSlug": APPOINTMENT_SLUG,
            "start": date,
            "end": date,
            "timeZone": DEFAULT_TZ,
        }
        data = await self._get("/slots", params=params, version="2024-09-04")
        slots = data.get("data", {}).get(date, [])
        _set_cached_slots(key, slots)
        return slots

    @staticmethod
    def _slot_local(slot: dict) -> datetime | None:
        dt = _parse_iso(slot.get("start", ""))
        if dt is None:
            return None
        return dt.astimezone(_tz(DEFAULT_TZ)).replace(second=0, microsecond=0)

    async def _slot_status(self, date: str, appointment_time: str):
        try:
            day = datetime.strptime(date.strip(), "%Y-%m-%d").date()
        except ValueError:
            raise ToolError("appointment_date must be YYYY-MM-DD. Work out the right date and call again.")
        s = (appointment_time or "").strip().lower().replace(".", ":").replace(" ", "")
        m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$", s)
        if not m:
            raise ToolError("appointment_time must be 24-hour HH:MM, e.g. 15:30. Call again with the right time.")
        hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        clock = f"{hour:02d}:{minute:02d}"
        hour, minute = int(clock[:2]), int(clock[3:])
        start = day.replace(hour=hour, minute=minute, tzinfo=_tz(DEFAULT_TZ))
        if start <= now_local():
            raise ToolError("That time has already passed. Ask the caller for a later day or time.")

        slots = await self._slots_for(day.strftime("%Y-%m-%d"))
        locals_ = [d for d in (self._slot_local(s) for s in slots) if d]
        if any(d == start for d in locals_):
            return start, None
        if not locals_:
            return start, f"NOT_AVAILABLE: no free slots on {day.strftime('%Y-%m-%d')}. Tell the caller and ask for another day."
        nearest = sorted(locals_, key=lambda d: abs((d - start).total_seconds()))[:3]
        nearest.sort()
        listing = ", ".join(d.strftime("%I:%M %p").lstrip("0") for d in nearest)
        return start, (
            f"NOT_AVAILABLE at {start.strftime('%I:%M %p').lstrip('0')}. "
            f"Free times that day: {listing}. Offer two of them in one short question."
        )

    @llm.function_tool
    async def set_appointment_time(
        self,
        ctx: RunContext,
        appointment_date: Annotated[str, "Clinic-local date as YYYY-MM-DD, e.g. '2026-11-17'"],
        appointment_time: Annotated[str, "Clinic-local time, 24-hour HH:MM, e.g. '15:30'"],
        patient_name: Annotated[str | None, "ONLY if the caller already said their name in this call"] = None,
        patient_phone: Annotated[str | None, "ONLY if the caller already said their phone in this call"] = None,
    ) -> str:
        """Record the day and time the caller wants and check the clinic can take it.

        Call this as soon as BOTH the day and the time are known. Do not say
        anything else in that turn: the system asks for the remaining details.
        """
        start, problem = await self._slot_status(appointment_date, appointment_time)
        if problem:
            return problem
        flow = self.flow
        flow.set_slot(start.strftime("%Y-%m-%d"), start.strftime("%H:%M"))
        if patient_name:
            cand = _name_candidate(patient_name)
            if cand:
                flow.name = cand
        if patient_phone:
            digits = _digits_from_phone_text(patient_phone)
            loose = _loose_phone(digits)
            if loose:
                flow.phone = loose
        if flow.stage in ("name", "phone"):
            flow.stage = flow._first_missing_stage()
        logger.info("FLOW | slot set | %s", flow.describe())
        return "OK. The time is free and saved. Say nothing about it; the system asks the next question."

    @llm.function_tool
    async def check_availability(
        self,
        ctx: RunContext,
        date: Annotated[str | None, "Date in YYYY-MM-DD format. Defaults to today if not provided."] = None,
    ) -> str:
        """List free dental appointment times for a clinic-local date (defaults to today)."""
        date = (date or now_local().strftime("%Y-%m-%d")).strip()
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return "Please tell me the date first (YYYY-MM-DD)."
        slots = await self._slots_for(date)
        times = [d.strftime("%I:%M %p").lstrip("0") for d in (self._slot_local(s) for s in slots) if d]
        if not times:
            return f"No available slots on {date}."
        shown = times[:6]
        listing = ", ".join(shown)
        if len(times) > len(shown):
            listing += f", and {len(times) - len(shown)} more"
        return f"Available times on {date}: {listing}."

    async def run_booking(self) -> None:
        """Idempotent — safe to await again if the caller interrupts mid-way."""
        if self.flow.stage != "booking":
            return
        if self._booking_task is None:
            self._booking_task = asyncio.create_task(self._book())
        await asyncio.shield(self._booking_task)

    async def _book(self) -> None:
        flow = self.flow
        try:
            start_local = datetime.strptime(f"{flow.date} {flow.time}", "%Y-%m-%d %H:%M").replace(
                tzinfo=_tz(DEFAULT_TZ)
            )
            attendee = {
                "name": flow.name or "Patient",
                "email": flow.booking_email(),
                "timeZone": DEFAULT_TZ,
            }
            body = {
                "eventTypeSlug": APPOINTMENT_SLUG,
                "username": self._username,
                "start": start_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "attendee": dict(attendee),
            }
            if flow.phone:
                body["attendee"]["phoneNumber"] = flow.phone
                body["metadata"] = {"phone": flow.phone}
            try:
                data = await self._post("/bookings", body=body)
            except ToolError as err:
                if "(400" in str(err) and ("phoneNumber" in body["attendee"] or "metadata" in body):
                    logger.warning("BOOKING | retrying without phone/metadata: %s", err)
                    body["attendee"] = dict(attendee)
                    body.pop("metadata", None)
                    data = await self._post("/bookings", body=body)
                else:
                    raise
            uid = data.get("data", {}).get("uid")
            flow.last_booking_uid = uid
            flow.notice = ("booked", f"{flow.date} {flow.time}")
            logger.info("BOOKING | success uid=%s | %s", uid, flow.describe())
            _drop_cached_slots(flow.date or "")
            flow.stage = "done"
        except Exception as err:
            logger.error("BOOKING | failed: %s", err)
            _drop_cached_slots(flow.date or "")
            flow.notice = ("failed", "")
            flow.date = flow.time = None
            flow.stage = "time"
            flow.prompted = ""
        finally:
            self._booking_task = None


__all__ = ["CalToolset", "BookingFlow", "now_local"]
