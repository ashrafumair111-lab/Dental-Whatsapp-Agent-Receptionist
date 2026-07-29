from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Annotated

import httpx
from livekit.agents import RunContext, llm
from livekit.agents.llm.tool_context import ToolError

CAL_API_BASE = "https://api.cal.com/v2"
CAL_API_VERSION = "2026-02-25"
HTTP_TIMEOUT = 15
MAX_RETRIES = 3
BASE_BACKOFF = 1.0

EVENT_TYPE_DISPLAY = {
    "30min": "General Consultation",
    "checkup": "Routine Checkup",
    "follow-up": "Follow-up Consultation",
    "secret": "Urgent Consultation",
    "new-patient": "New Patient Registration",
}

_slots_cache: dict[str, tuple[float, list]] = {}
SLOTS_CACHE_TTL = 30


def _slots_cache_key(event_type_slug: str, date: str) -> str:
    return f"{event_type_slug}:{date}"


def _get_cached_slots(event_type_slug: str, date: str) -> list | None:
    key = _slots_cache_key(event_type_slug, date)
    entry = _slots_cache.get(key)
    if entry and (time.time() - entry[0]) < SLOTS_CACHE_TTL:
        return entry[1]
    return None


def _set_cached_slots(event_type_slug: str, date: str, slots: list) -> None:
    key = _slots_cache_key(event_type_slug, date)
    _slots_cache[key] = (time.time(), slots)


def _placeholder_email_domain() -> str:
    company = os.environ.get("COMPANY_NAME", "Hamza Hospital")
    slug = re.sub(r"[^a-z0-9]", "", company.lower())
    return f"{slug}.com"


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
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                resp = await client.request(
                    method,
                    f"{CAL_API_BASE}{path}",
                    headers=headers,
                    json=json,
                    params=params,
                )
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

    async def sync_event_type_names(self) -> None:
        data = await self._get("/event-types", {"username": self._username}, version="2024-06-14")
        for et in data.get("data", []):
            slug = et["slug"]
            new_title = EVENT_TYPE_DISPLAY.get(slug)
            if new_title and et.get("title") != new_title:
                await self._patch(f"/event-types/{et['id']}", {"title": new_title}, version="2024-06-14")

    @llm.function_tool
    async def list_event_types(self) -> str:
        """List all available appointment types offered by the clinic."""
        data = await self._get("/event-types", {"username": self._username}, version="2024-06-14")
        types = data.get("data", [])
        if not types:
            return "No event types found for this user."
        lines = []
        for et in types:
            slug = et["slug"]
            display = EVENT_TYPE_DISPLAY.get(slug, et["title"])
            lines.append(f"- {display} ({et['lengthInMinutes']} min)")
        return "Available appointment types:\n" + "\n".join(lines)

    @llm.function_tool
    async def check_availability(
        self,
        ctx: RunContext,
        event_type_slug: Annotated[str, "Slug of the event type to check. Options: '30min' (General Consultation), 'checkup' (Routine Checkup), 'follow-up' (Follow-up), 'secret' (Urgent), 'new-patient' (New Patient Registration)."],
        date: Annotated[str, "Date in YYYY-MM-DD format. Defaults to today if not provided."] | None = None,
        patient_timezone: Annotated[str, "IANA timezone, e.g. 'Asia/Karachi'"] = "Asia/Karachi",
    ) -> str:
        """Check available time slots for a specific appointment type on a given date.

        Args:
            event_type_slug: The slug of the event type.
            date: The date to check availability for, in YYYY-MM-DD format. Defaults to today.
            patient_timezone: IANA timezone for slot display (default Asia/Karachi).
        """
        date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cached = _get_cached_slots(event_type_slug, date)
        if cached is not None:
            slots = cached
        else:
            await ctx.update(f"Checking availability for {event_type_slug} on {date}.")
            params = {
                "username": self._username,
                "eventTypeSlug": event_type_slug,
                "start": date,
                "end": date,
                "timeZone": patient_timezone,
            }
            async with ctx.with_filler("One moment, still checking.", delay=2.0, interval=6.0, max_steps=2):
                data = await self._get("/slots", params=params, version="2024-09-04")
            slots = data.get("data", {}).get(date, [])
            _set_cached_slots(event_type_slug, date, slots)

        if not slots:
            return f"No available slots on {date} for {event_type_slug}."

        times = [s["start"] for s in slots]
        first = _format_time(times[0])
        last = _format_time(times[-1])
        return f"Slots on {date}: {first} to {last}, {len(times)} time(s) available."



    @llm.function_tool
    async def create_booking(
        self,
        ctx: RunContext,
        event_type_slug: Annotated[str, "Slug of the event type to book (e.g. 'checkup', 'follow-up', 'new-patient')"],
        start_time: Annotated[str, "Start time in ISO 8601 UTC format, e.g. '2024-08-13T09:00:00Z'"],
        attendee_name: str,
        attendee_email: Annotated[str | None, "Email address of the patient. Ask for it first. Only omit if the patient truly doesn't have one."] = None,
        attendee_timezone: Annotated[str, "IANA timezone, e.g. 'Asia/Karachi'"] = "Asia/Karachi",
        attendee_phone: Annotated[str | None, "Phone number in international format"] = None,
        notes: Annotated[str | None, "Additional notes or reason for the visit"] = None,
    ) -> str:
        """Book an appointment for the patient.

        Args:
            event_type_slug: The slug of the event type (e.g. 'checkup', 'follow-up').
            start_time: Start time in ISO 8601 UTC format.
            attendee_name: Full name of the patient.
            attendee_email: Email address of the patient. Ask first. If unavailable, a placeholder will be used.
            attendee_timezone: IANA timezone of the patient.
            attendee_phone: Phone number in international format (optional).
            notes: Additional notes or reason for the visit (optional).
        """
        resolved_email = attendee_email or f"{attendee_name.lower().replace(' ', '.')}@{_placeholder_email_domain()}"
        body = {
            "eventTypeSlug": event_type_slug,
            "username": self._username,
            "start": start_time,
            "attendee": {
                "name": attendee_name,
                "email": resolved_email,
                "timeZone": attendee_timezone,
            },
        }
        if attendee_phone:
            body["attendee"]["phoneNumber"] = attendee_phone
        if notes:
            body["bookingFieldsResponses"] = {"notes": notes}

        await ctx.update(f"Booking the {EVENT_TYPE_DISPLAY.get(event_type_slug, event_type_slug)} appointment now.")
        async with ctx.with_filler("One moment, still confirming the booking.", delay=2.0, interval=6.0, max_steps=2):
            data = await self._post("/bookings", body=body)

        booking = data.get("data", {})
        uid = booking.get("uid", "unknown")
        start = booking.get("start", start_time)

        return (
            f"Appointment booked successfully!\n"
            f"Booking UID: {uid}\n"
            f"Start: {start}\n"
            f"Patient: {attendee_name}\n"
            f"Confirmation sent to {resolved_email}"
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
