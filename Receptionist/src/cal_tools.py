"""Cal.com integration for the dental receptionist.

Two tools, deliberately:

- ``check_availability(date)`` — what slots are free on a date.
- ``create_booking(start_time, attendee_name, ...)`` — the only way to book.

Email and phone are *not* parameters the model guesses. The booking asks for
the address and the number itself — one question, one confirmation — and never
rejects a format, so neither the model nor the caller gets stuck.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Annotated
from zoneinfo import ZoneInfo

import httpx
from livekit.agents import RunContext, AgentTask, llm
from livekit.agents.llm.tool_context import ToolError, ToolFlag, function_tool

CAL_API_BASE = "https://api.cal.com/v2"
CAL_API_VERSION = "2026-02-25"
HTTP_TIMEOUT = 15
MAX_RETRIES = 3
BASE_BACKOFF = 1.0

# One appointment type: a 30-minute dental visit with the dentist.
APPOINTMENT_SLUG = "30min"
APPOINTMENT_DISPLAY = "Dental Appointments"

DEFAULT_TZ = "Asia/Karachi"  # Karachi is fixed UTC+5 all year


def _tz(name: str | None = None) -> tzinfo:
    name = (name or DEFAULT_TZ).strip() or DEFAULT_TZ
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone(timedelta(hours=5)) if name == DEFAULT_TZ else timezone.utc


def now_local(tz_name: str | None = None) -> datetime:
    return datetime.now(_tz(tz_name))


# --------------------------------------------------------------------------- #
# Slots cache                                                                    #
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


def _format_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat((iso_str or "").replace("Z", "+00:00"))
    except ValueError:
        return iso_str
    return dt.astimezone(_tz(DEFAULT_TZ)).strftime("%I:%M %p").lstrip("0")


# --------------------------------------------------------------------------- #
# Loose contact-info capture                                                    #
# --------------------------------------------------------------------------- #
_EMAIL_LOOSE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._%+\-]*@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$")


def _normalize_email_loose(value: str) -> str | None:
    """Best effort: take whatever the model hands over. Never rejects."""
    text = (value or "").strip().lower()
    if not text:
        return None
    typed = re.search(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+", text)
    if typed:
        return typed.group(0).rstrip(".,;")
    if "@" in text:
        text = text.replace(" at ", "@").replace(" dot ", ".")
        text = re.sub(r"\s+", "", text)
        if _EMAIL_LOOSE.match(text):
            return text
        return None
    return None


def _normalize_phone_loose(value: str) -> str | None:
    """Keep the number's digits; leading + wins over everything. Never rejects."""
    text = value or ""
    digits = re.sub(r"\D", "", text)
    if len(digits) < 7:
        return None
    if text.strip().startswith("+"):
        return "+" + digits
    return digits


class LooseCaptureTask(AgentTask[str]):
    """Ask for one piece of contact info and confirm it once.

    Accepts literally anything reasonable — no country code demanded, no format
    rejected — then reads it back and asks "is that correct?" exactly once. If
    the caller says no, it only asks again for that one piece.
    """

    _KINDS = {
        "email": {
            "title": "email address",
            "ask": (
                "Ask the caller for the email to send the confirmation to. If they already "
                "gave one earlier in this call, call update_capture with it — do not ask again."
            ),
            "readback": (
                "Read the address back, slowly, saying it in plain words, then ask 'Is that correct?'"
            ),
        },
        "phone": {
            "title": "phone number",
            "ask": (
                "Ask the caller for their phone number, however they say it — with or without a "
                "country code, any format. If they already gave it earlier in this call, call "
                "update_capture with it — do not ask again. Never ask for a country code."
            ),
            "readback": (
                "Read the number back digit by digit in small groups. Then ask 'Is that correct?'"
            ),
        },
    }

    def __init__(self, kind: str, *, chat_ctx) -> None:
        spec = self._KINDS[kind]
        self._kind = kind
        self._spec = spec
        self._current = ""
        super().__init__(
            instructions=(
                f"You are a single step in a phone system whose only job is capturing the caller's {spec['title']}.\n"
                f"{spec['ask']}\n"
                "Call `update_capture` as soon as the caller gives a value. Never reject formats, "
                "never say the number/address is wrong — take whatever they say.\n"
                f"{spec['readback']}\n"
                "Call `confirm` after the caller confirms. If the caller says no, call `update_capture` "
                "with the corrected value and read it back again.\n"
                "Keep replies to 1-2 short sentences. No lists, no extra questions."
            ),
            chat_ctx=chat_ctx,
            tools=[self._build_update_tool(), self._build_confirm_tool(), self._build_decline_tool()],
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Ask now, briefly.")

    def _normalize(self, value: str) -> str | None:
        return _normalize_email_loose(value) if self._kind == "email" else _normalize_phone_loose(value)

    def _build_update_tool(self):
        kind = self._kind

        @function_tool(flags=ToolFlag.NONE)
        async def update_capture(value: str) -> str | None:
            """Store what the caller just said (email address or phone number).

            Args:
                value: exactly what the caller said, letters and symbols as they spoke them
            """
            normalized = self._normalize(value)
            if normalized is None:
                return f"That doesn't look like a {kind}. Ask the caller to say it once more, slowly — no country code needed, any format is fine."
            self._current = normalized
            return f"Captured {normalized}. {self._spec['readback']}"

        return update_capture

    def _build_confirm_tool(self):
        @function_tool(flags=ToolFlag.NONE)
        async def confirm() -> str | None:
            """Call only after the caller says yes / correct / sahi hai / bilkul."""
            self.complete(getattr(self, "_current", ""))
            return None

        return confirm

    def _build_decline_tool(self):
        @function_tool(flags=ToolFlag.NONE)
        async def decline_capture(reason: str) -> None:
            """Call when the caller refuses to give the email or phone number."""
            if not self.done():
                self.complete("")

        return decline_capture


# --------------------------------------------------------------------------- #
# The toolset                                                                  #
# --------------------------------------------------------------------------- #
class CalToolset(llm.Toolset):
    def __init__(self) -> None:
        super().__init__(id="cal")
        api_key = os.environ.get("CAL_API_KEY")
        if not api_key:
            raise RuntimeError("CAL_API_KEY not set in environment")
        self._api_key = api_key
        username = os.environ.get("CAL_USERNAME")
        if not username:
            raise RuntimeError("CAL_USERNAME not set in environment")
        self._username = username

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
        """Warm DNS/TLS and cache today's slots while the greeting plays."""
        try:
            async with asyncio.timeout(6.0):
                await self._slots_for(now_local().strftime("%Y-%m-%d"))
        except Exception:
            pass

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
                raise ToolError("Cal.com is slow right now. Ask the caller to hold on a moment.") from err
            except httpx.RequestError as err:
                raise ToolError("I couldn't reach the booking system. Ask the caller to hold on a moment.") from err

            if resp.status_code < 400:
                return resp.json()
            last_error = ToolError(f"Cal.com API error ({resp.status_code}): {resp.text}")
            if resp.status_code in (429, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF * (2**attempt)
                await asyncio.sleep(wait)
                continue
            raise last_error
        raise last_error  # type: ignore[misc]

    async def _get(self, path: str, params: dict | None = None, version: str = CAL_API_VERSION) -> dict:
        return await self._request("GET", path, params=params, version=version)

    async def _post(self, path: str, body: dict, version: str = CAL_API_VERSION) -> dict:
        return await self._request("POST", path, json=body, version=version)

    async def _slots_for(self, date: str) -> list:
        cached = _get_cached_slots(date)
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
        _set_cached_slots(date, slots)
        return slots

    @llm.function_tool
    async def check_availability(
        self,
        ctx: RunContext,
        date: Annotated[str, "Clinic-local date as YYYY-MM-DD, e.g. '2026-11-17'"],
    ) -> str:
        """List free dental appointment slots for a date (clinic-local, default today)."""
        try:
            day = datetime.strptime((date or "").strip() or now_local().strftime("%Y-%m-%d"), "%Y-%m-%d").date()
        except ValueError:
            return "Please tell me the date first (YYYY-MM-DD)."
        slots = await self._slots_for(day.strftime("%Y-%m-%d"))
        if not slots:
            return f"No available slots on {day.strftime('%Y-%m-%d')}."
        times = [_format_time(s["start"]) for s in slots]
        shown = times[:6]
        listing = ", ".join(shown)
        if len(times) > len(shown):
            listing += f", and {len(times) - len(shown)} more"
        return f"Available times on {day.strftime('%Y-%m-%d')}: {listing}."

    @llm.function_tool
    async def create_booking(
        self,
        ctx: RunContext,
        start_time: Annotated[
            str,
            "Booking start time. Clinic-local YYYY-MM-DD HH:MM (e.g. '2026-11-17 15:30'), "
            "or a full ISO 8601 instant.",
        ],
        attendee_name: Annotated[str, "Caller's full name, spelled the way they gave it"],
        notes: Annotated[str | None, "One-line reason for the visit (optional)"] = None,
    ) -> str:
        """Book a dental appointment. Use this after the caller confirmed the day, the time and their name.

        Do not ask for the caller's email or phone: this tool asks for both itself,
        confirms the email letter by letter, and never books with a guessed address.
        Only call it when you have a name and a slot both the caller accepted.
        """
        start_utc: datetime | None = None
        cleaned = (start_time or "").strip().replace("T", " ")
        try:  # clinic-local wall time -> UTC
            naive = datetime.strptime(cleaned, "%Y-%m-%d %H:%M")
            start_utc = naive.replace(tzinfo=_tz(DEFAULT_TZ)).astimezone(timezone.utc)
        except ValueError:
            try:  # full ISO instant
                parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
                start_utc = parsed if parsed.tzinfo else parsed.replace(tzinfo=_tz(DEFAULT_TZ)).astimezone(timezone.utc)
            except ValueError:
                raise ToolError(
                    f"Invalid start_time {start_time!r}. Use 'YYYY-MM-DD HH:MM' or ISO 8601."
                ) from None

        if start_utc <= datetime.now(timezone.utc):
            raise ToolError("That time has already passed. Ask the caller for a later day or time.")
        start_iso = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Ask the caller for email and phone, one at a time, and confirm once.
        chat_ctx = ctx.session.current_agent.chat_ctx
        try:
            async with ctx.foreground():
                email_result = await LooseCaptureTask("email", chat_ctx=chat_ctx)
                phone_result = await LooseCaptureTask("phone", chat_ctx=chat_ctx)
        except ToolError:
            return "The caller did not give an email address. Ask them if they would like the clinic to call them instead."

        email_address = email_result if isinstance(email_result, str) else ""
        phone_number = phone_result if isinstance(phone_result, str) else ""

        body = {
            "eventTypeSlug": APPOINTMENT_SLUG,
            "username": self._username,
            "start": start_iso,
            "attendee": {
                "name": (attendee_name or "Patient").strip(),
                "email": email_address or f"noemail+{self._username}@example.com",
                "timeZone": DEFAULT_TZ,
            },
        }
        if phone_number:
            body["attendee"]["phoneNumber"] = phone_number
        if notes:
            body["bookingFieldsResponses"] = {"notes": notes}
        data = await self._post("/bookings", body=body)

        booking = data.get("data", {})
        uid = booking.get("uid", "unknown")
        return (
            f"Appointment booked successfully!\n"
            f"Booking UID: {uid}\n"
            f"Start: {start_iso}\n"
            f"Attendee: {attendee_name}\n"
            f"Confirmation sent to {email_address or 'the clinic'}\n"
            f"Contact number: {phone_number or 'n/a'}"
        )


__all__ = ["CalToolset", "now_local"]
