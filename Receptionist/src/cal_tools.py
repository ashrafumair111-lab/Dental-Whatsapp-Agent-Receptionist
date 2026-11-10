from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Annotated
from zoneinfo import ZoneInfo

import httpx
import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat, country_code_for_region
from livekit.agents import RunContext, llm
from livekit.agents.llm.tool_context import ToolError

logger = logging.getLogger(__name__)

CAL_API_BASE = "https://api.cal.com/v2"
CAL_API_VERSION = "2026-02-25"
HTTP_TIMEOUT = 8.0
MAX_RETRIES = 2
BASE_BACKOFF = 0.4

# Dental Clinic offers a single appointment type: a dental visit with the doctor.
APPOINTMENT_SLUG = "30min"
APPOINTMENT_DISPLAY = "Dental Appointments"

def _default_phone_region() -> str:
    """Region used to turn local trunk numbers (e.g. 0300...) into E.164.

    Read at call time so ``DEFAULT_PHONE_REGION`` set in ``.env.local`` applies
    even though this module is imported before dotenv loads.
    """
    return os.environ.get("DEFAULT_PHONE_REGION", "PK").strip().upper() or "PK"

EMAIL_REGEX = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._%+\-]*@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$"
)

_slots_cache: dict[str, tuple[float, list]] = {}
SLOTS_CACHE_TTL = 30


def _get_cached_slots(key: str) -> list | None:
    entry = _slots_cache.get(key)
    if entry and (time.time() - entry[0]) < SLOTS_CACHE_TTL:
        return entry[1]
    return None


def _set_cached_slots(key: str, slots: list) -> None:
    _slots_cache[key] = (time.time(), slots)


def _validate_email(raw: str) -> str:
    email = (raw or "").strip()
    if "@" not in email:
        # Callers may speak it as "john at example dot com"; normalize that form.
        email = re.sub(r"\s+at\s+", "@", email, flags=re.IGNORECASE)
        email = re.sub(r"\s+dot\s+", ".", email, flags=re.IGNORECASE)
    email = re.sub(r"\s+", "", email).rstrip(".,;:")
    if not EMAIL_REGEX.match(email):
        raise ToolError(
            "That email didn't sound right. Ask the caller to repeat it once, slowly."
        )
    return email


def _normalize_phone(raw: str) -> str:
    """Accept a phone number in any format and return E.164 when possible.

    Local trunk numbers are parsed with DEFAULT_PHONE_REGION, so a caller can
    simply say "0300 1234567" without being asked for a country code.
    """
    cleaned = re.sub(r"[^\d+]", "", raw or "")
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]

    digits = re.sub(r"\D", "", cleaned)
    if not 6 <= len(digits) <= 15:
        raise ToolError(
            "I didn't catch the whole phone number. Ask the caller to say it once more, in one go."
        )

    region = _default_phone_region()
    try:
        parsed = phonenumbers.parse(
            cleaned, None if cleaned.startswith("+") else region
        )
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(parsed, PhoneNumberFormat.E164)
    except NumberParseException:
        pass

    if cleaned.startswith("+"):
        return "+" + digits

    country_code = country_code_for_region(region)
    if country_code:
        cc = str(country_code)
        if digits.startswith(cc) and len(digits) >= len(cc) + 7:
            return "+" + digits

    return digits


def _format_time(iso_str: str, tz_name: str = "Asia/Karachi") -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except ValueError:
        return iso_str
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        dt = dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        pass
    return dt.strftime("%I:%M %p").lstrip("0")


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
        # One pooled client per call: keeps TLS connections warm between tools.
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
        """Warm DNS/TLS and cache today's slots during the greeting.

        A cold Cal.com connection measured ~4.4s here versus ~0.4s warm. Doing
        this in the background while Mike says hello keeps the first real
        availability check fast. Failures are ignored.
        """
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tz = "Asia/Karachi"
        params = {
            "username": self._username,
            "eventTypeSlug": APPOINTMENT_SLUG,
            "start": date,
            "end": date,
            "timeZone": tz,
        }
        try:
            async with asyncio.timeout(6.0):
                data = await self._get("/slots", params=params, version="2024-09-04")
            _set_cached_slots(f"{date}|{tz}", data.get("data", {}).get(date, []))
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
                raise ToolError(
                    "Cal.com is responding slowly right now. Let the caller know and offer to try again in a moment."
                ) from err
            except httpx.RequestError as err:
                raise ToolError(
                    "I couldn't reach the booking system just now. Let the caller know and offer to try again."
                ) from err

            if resp.status_code < 400:
                return resp.json()
            last_error = ToolError(f"Cal.com API error ({resp.status_code}): {resp.text}")
            if resp.status_code in (429, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.25)
                await asyncio.sleep(wait)
                continue
            raise last_error
        raise last_error  # type: ignore[misc]

    async def _get(self, path: str, params: dict | None = None, version: str = CAL_API_VERSION) -> dict:
        return await self._request("GET", path, params=params, version=version)

    async def _post(self, path: str, body: dict, version: str = CAL_API_VERSION) -> dict:
        return await self._request("POST", path, json=body, version=version)

    async def _patch(self, path: str, body: dict, version: str = CAL_API_VERSION) -> dict:
        return await self._request("PATCH", path, json=body, version=version)

    @llm.function_tool
    async def check_availability(
        self,
        ctx: RunContext,
        date: Annotated[str, "Date in YYYY-MM-DD format. Defaults to today if not provided."] | None = None,
        caller_timezone: Annotated[str, "IANA timezone, e.g. 'Asia/Karachi'"] = "Asia/Karachi",
    ) -> str:
        """Check available dental appointment slots for a date (defaults to today)."""
        date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cache_key = f"{date}|{caller_timezone}"
        cached = _get_cached_slots(cache_key)
        if cached is not None:
            slots = cached
        else:
            params = {
                "username": self._username,
                "eventTypeSlug": APPOINTMENT_SLUG,
                "start": date,
                "end": date,
                "timeZone": caller_timezone,
            }
            async with ctx.with_filler("Let me check that for you.", delay=0.5):
                data = await self._get("/slots", params=params, version="2024-09-04")
            slots = data.get("data", {}).get(date, [])
            _set_cached_slots(cache_key, slots)

        if not slots:
            return f"No available slots on {date}."

        times = [_format_time(s["start"], caller_timezone) for s in slots]
        shown = times[:6]
        listing = ", ".join(shown)
        if len(times) > len(shown):
            listing += f", and {len(times) - len(shown)} more"
        return f"Available times on {date}: {listing}."

    @llm.function_tool
    async def create_booking(
        self,
        ctx: RunContext,
        start_time: Annotated[str, "Start time in ISO 8601 UTC format, e.g. '2024-08-13T09:00:00Z'"],
        attendee_name: Annotated[str, "Caller's full name, spelled the way they gave it"],
        attendee_email: Annotated[str, "Caller's email exactly as they spelled it — never correct or normalize the spelling"],
        attendee_phone: Annotated[str, "Caller's phone number exactly as they said it; any format is fine"],
        attendee_timezone: Annotated[str, "IANA timezone, e.g. 'Asia/Karachi'"] = "Asia/Karachi",
        notes: Annotated[str | None, "Additional notes or reason for the visit"] = None,
    ) -> str:
        """Book a dental appointment. Call only after date, time, name, email, and phone are confirmed."""
        resolved_email = _validate_email(attendee_email)
        resolved_phone = _normalize_phone(attendee_phone)

        body = {
            "eventTypeSlug": APPOINTMENT_SLUG,
            "username": self._username,
            "start": start_time,
            "attendee": {
                "name": attendee_name,
                "email": resolved_email,
                "timeZone": attendee_timezone,
                "phoneNumber": resolved_phone,
            },
        }
        if notes:
            body["bookingFieldsResponses"] = {"notes": notes}

        async with ctx.with_filler("One moment, I'm confirming your booking.", delay=0.5):
            data = await self._post("/bookings", body=body)

        booking = data.get("data", {})
        uid = booking.get("uid", "unknown")
        start = booking.get("start", start_time)

        return (
            f"Appointment booked successfully!\n"
            f"Booking UID: {uid}\n"
            f"Start: {start}\n"
            f"Attendee: {attendee_name}\n"
            f"Confirmation sent to {resolved_email}\n"
            f"Contact number: {resolved_phone}"
        )

    @llm.function_tool
    async def reschedule_booking(
        self,
        ctx: RunContext,
        booking_uid: str,
        new_start: Annotated[str, "New start time in ISO 8601 UTC format"],
        reason: Annotated[str | None, "Reason for rescheduling"] = None,
    ) -> str:
        """Reschedule an existing booking to a new ISO 8601 UTC time."""
        body = {"start": new_start}
        if reason:
            body["reschedulingReason"] = reason

        async with ctx.with_filler("One moment, I'm working on that.", delay=0.5):
            data = await self._post(f"/bookings/{booking_uid}/reschedule", body=body)

        booking = data.get("data", {})
        new_start_val = booking.get("start", new_start)
        return f"Booking rescheduled successfully! New time: {new_start_val}"

    @llm.function_tool
    async def cancel_booking(
        self,
        ctx: RunContext,
        booking_uid: str,
        reason: Annotated[str | None, "Reason for cancellation"] = None,
    ) -> str:
        """Cancel an existing booking by UID."""
        body = {}
        if reason:
            body["cancellationReason"] = reason

        async with ctx.with_filler("One moment, I'm working on that.", delay=0.5):
            await self._post(f"/bookings/{booking_uid}/cancel", body=body)

        return "Booking cancelled successfully."
