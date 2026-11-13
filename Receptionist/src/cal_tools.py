"""Cal.com helpers: thin and stateless. The conversation lives in the LLM prompt."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import date as Date, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

import httpx
from livekit.agents.llm.tool_context import ToolError

logger = logging.getLogger("clinic")

API = "https://api.cal.com/v2"
SLUG = "30min"
TZ_NAME = "Asia/Karachi"


def _load_tz() -> tzinfo:
    try:
        return ZoneInfo(TZ_NAME)
    except Exception:
        return timezone(timedelta(hours=5))


TZ = _load_tz()
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def now() -> datetime:
    return datetime.now(TZ)


def fmt_time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def say_when(dt: datetime) -> str:
    return f"{DAYS[dt.weekday()]}, {dt.day} {dt.strftime('%B')} at {fmt_time(dt)}"


# ------------------------------ email helpers ------------------------------ #
EMAIL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._%+\-]*@(?:[A-Za-z0-9\-]+\.)+[A-Za-z]{2,}$")
_SAY = {".": "dot", "_": "underscore", "-": "dash", "+": "plus"}
_COMMON = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"}


def clean_email(raw: str) -> str | None:
    """'m i k e at gmail dot com' or 'mike@gmail.com' -> 'mike@gmail.com', else None."""
    t = (raw or "").strip().lower()
    if "@" not in t:
        t = re.sub(r"\s+(?:at the rate|at)\s+", "@", t)
    t = re.sub(r"\s+dot\s+", ".", t)
    t = re.sub(r"\s+", "", t).rstrip(".,;:")
    return t if EMAIL_RE.match(t) else None


def spell_email(email: str) -> str:
    """Letter by letter, the way a receptionist reads it back."""
    local, domain = email.split("@")

    def spell(s: str) -> str:
        return " ".join(_SAY.get(c, c) for c in s)

    dom = f"{domain.split('.')[0]} dot com" if domain in _COMMON else spell(domain)
    return f"{spell(local)} at {dom}"


# -------------------------------- Cal.com --------------------------------- #
class Cal:
    def __init__(self) -> None:
        self.key = os.environ["CAL_API_KEY"]
        self.user = os.environ["CAL_USERNAME"]
        self.http = httpx.AsyncClient(timeout=15)

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _call(self, method: str, path: str, version: str, tries: int = 3, **kw) -> dict:
        headers = {"Authorization": f"Bearer {self.key}", "cal-api-version": version}
        for attempt in range(tries):
            last = attempt == tries - 1
            try:
                r = await self.http.request(method, API + path, headers=headers, **kw)
            except httpx.HTTPError as err:
                if last:
                    raise ToolError("The booking system isn't responding. Say the team will call back.") from err
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code in (429, 502, 503, 504) and not last:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise ToolError(f"Cal.com error {r.status_code}: {r.text[:200]}")
            return r.json()
        raise ToolError("Booking system error.")

    async def free_slots(self, day: Date) -> list[datetime]:
        d = day.isoformat()
        data = await self._call(
            "GET", "/slots", "2024-09-04",
            params={"username": self.user, "eventTypeSlug": SLUG, "start": d, "end": d, "timeZone": TZ_NAME},
        )
        out = []
        for slot in data.get("data", {}).get(d, []):
            try:
                dt = datetime.fromisoformat(slot["start"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            out.append(dt.astimezone(TZ).replace(second=0, microsecond=0))
        return sorted(out)

    async def book(self, start: datetime, name: str, email: str, phone: str) -> str | None:
        attendee = {"name": name, "email": email, "timeZone": TZ_NAME}
        body = {
            "eventTypeSlug": SLUG,
            "username": self.user,
            "start": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "attendee": {**attendee, "phoneNumber": phone},
            "metadata": {"phone": phone},
        }
        try:
            data = await self._call("POST", "/bookings", "2026-02-25", tries=1, json=body)
        except ToolError as err:
            if "error 400" not in str(err):
                raise
            # Cal.com sometimes rejects the phone format: book without it.
            body["attendee"] = attendee
            body.pop("metadata")
            data = await self._call("POST", "/bookings", "2026-02-25", tries=1, json=body)
        return data.get("data", {}).get("uid")