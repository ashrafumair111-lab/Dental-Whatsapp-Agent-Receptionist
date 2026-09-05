from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Annotated

import httpx
from livekit.agents import RunContext, llm
from livekit.agents.beta.workflows import GetEmailTask, GetPhoneNumberTask
from livekit.agents.llm.tool_context import ToolError

CAL_API_BASE = "https://api.cal.com/v2"
CAL_API_VERSION = "2026-02-25"
HTTP_TIMEOUT = 15
MAX_RETRIES = 3
BASE_BACKOFF = 1.0

# MetaByte Solutions offers a single appointment type: Consultations.
CONSULTATION_SLUG = "30min"
CONSULTATION_DISPLAY = "Consultations"

_slots_cache: dict[str, tuple[float, list]] = {}
SLOTS_CACHE_TTL = 30


def _get_cached_slots(date: str) -> list | None:
    entry = _slots_cache.get(date)
    if entry and (time.time() - entry[0]) < SLOTS_CACHE_TTL:
        return entry[1]
    return None


def _set_cached_slots(date: str, slots: list) -> None:
    _slots_cache[date] = (time.time(), slots)


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
                    async with httpx.AsyncClient() as client:
                        resp = await client.request(
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

            if resp.status_code < 400:
                return resp.json()
            last_error = ToolError(f"Cal.com API error ({resp.status_code}): {resp.text}")
            if resp.status_code == 429 and attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5)
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
        """Check available time slots for a Consultation appointment on a given date.

        Args:
            date: The date to check availability for, in YYYY-MM-DD format. Defaults to today.
            caller_timezone: IANA timezone for slot display (default Asia/Karachi).
        """
        date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cached = _get_cached_slots(date)
        if cached is not None:
            slots = cached
        else:
            await ctx.update(f"Checking availability for {date}.")
            params = {
                "username": self._username,
                "eventTypeSlug": CONSULTATION_SLUG,
                "start": date,
                "end": date,
                "timeZone": caller_timezone,
            }
            async with ctx.with_filler("One moment, still checking.", delay=2.0, interval=6.0, max_steps=2):
                data = await self._get("/slots", params=params, version="2024-09-04")
            slots = data.get("data", {}).get(date, [])
            _set_cached_slots(date, slots)

        if not slots:
            return f"No available slots on {date}."

        times = [s["start"] for s in slots]
        first = _format_time(times[0])
        last = _format_time(times[-1])
        return f"Slots on {date}: {first} to {last}, {len(times)} time(s) available."

    @llm.function_tool
    async def create_booking(
        self,
        ctx: RunContext,
        start_time: Annotated[str, "Start time in ISO 8601 UTC format, e.g. '2024-08-13T09:00:00Z'"],
        attendee_name: str,
        attendee_timezone: Annotated[str, "IANA timezone, e.g. 'Asia/Karachi'"] = "Asia/Karachi",
        notes: Annotated[str | None, "Additional notes or reason for the consultation"] = None,
    ) -> str:
        """Book a Consultation appointment.

        Call this only after the caller has confirmed the date, time, and their
        name. Do not ask for email or phone number before calling — this tool
        collects and confirms both itself as part of running.

        Args:
            start_time: Start time in ISO 8601 UTC format.
            attendee_name: Full name of the caller.
            attendee_timezone: IANA timezone of the caller.
            notes: Additional notes or reason for the consultation (optional).
        """
        chat_ctx = ctx.session.current_agent.chat_ctx
        async with ctx.foreground():
            email_result = await GetEmailTask(
                chat_ctx=chat_ctx,
                require_confirmation=True,
                extra_instructions="Ask for the email to send the booking confirmation to.",
            )
            phone_result = await GetPhoneNumberTask(
                chat_ctx=chat_ctx,
                require_confirmation=True,
                extra_instructions="Ask for a phone number where MetaByte Solutions can reach them about this consultation.",
            )

        resolved_email = email_result.email_address
        resolved_phone = phone_result.phone_number

        body = {
            "eventTypeSlug": CONSULTATION_SLUG,
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

        await ctx.update("Booking your consultation now.")
        async with ctx.with_filler("One moment, still confirming the booking.", delay=2.0, interval=6.0, max_steps=2):
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
        """Reschedule an existing booking to a new time.

        Args:
            booking_uid: The UID of the booking to reschedule.
            new_start: New start time in ISO 8601 UTC format.
            reason: Reason for rescheduling (optional).
        """
        body = {"start": new_start}
        if reason:
            body["reschedulingReason"] = reason

        await ctx.update("Rescheduling the appointment now.")
        async with ctx.with_filler("One moment, still working on that.", delay=2.0, interval=6.0, max_steps=2):
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
        """Cancel an existing booking.

        Args:
            booking_uid: The UID of the booking to cancel.
            reason: Reason for cancellation (optional).
        """
        body = {}
        if reason:
            body["cancellationReason"] = reason

        await ctx.update("Cancelling the appointment now.")
        async with ctx.with_filler("One moment, still working on that.", delay=2.0, interval=6.0, max_steps=2):
            data = await self._post(f"/bookings/{booking_uid}/cancel", body=body)

        return "Booking cancelled successfully."


def _format_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%I:%M %p %Z").lstrip("0")
    except ValueError:
        return iso_str
