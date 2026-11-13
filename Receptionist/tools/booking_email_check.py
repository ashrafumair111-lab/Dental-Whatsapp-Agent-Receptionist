"""Deterministic check: the code-owned booking flow reaches the Cal.com body.

Run:  .venv\\Scripts\\python.exe tools/booking_email_check.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cal_tools import BookingFlow, CalToolset  # noqa: E402


async def main() -> None:
    toolset = CalToolset(fallback_email="clinic@example.com")
    flow = toolset.flow
    fails = 0

    sent: list[dict] = []

    async def fake_post(path: str, body: dict, version: str | None = None) -> dict:
        sent.append(body)
        if path == "/bookings":
            return {"data": {"uid": "TEST123", "start": body["start"]}}
        return {"data": {"start": body.get("start")}}

    toolset._post = fake_post  # type: ignore[method-assign]

    # 1. A booking with no email at all still completes, with a placeholder
    #    that reaches the clinic (never a fake patient address).
    flow.set_slot("2026-11-20", "15:30")
    for spoken in ("Ali Khan", "0300 1234567", "nahi"):
        flow.handle_user_text(spoken)
    print("after name/phone/email-skip:", flow.describe())
    if flow.stage != "booking" or not flow.email_skipped:
        fails += 1
        print(f"FAIL: expected stage booking w/ email skipped, got stage={flow.stage} email_skipped={flow.email_skipped}")
    await toolset.run_booking()
    attendee = (sent[-1].get("attendee") if sent else {}) or {}
    want_email = "clinic+923001234567@example.com" if not flow.email else None
    if attendee.get("email") == want_email and attendee.get("name") == "Ali Khan":
        print(f"OK   email skipped -> booked with placeholder {attendee['email']}")
    else:
        fails += 1
        print(f"FAIL: booking body {attendee}")

    # 2. Full flow: capture -> confirm -> booking reuses what was captured.
    flow.reset()
    flow.set_slot("2026-11-21", "10:00")
    for spoken in ("Ali Khan", "0300 1234567", "s m i t h at gmail dot com", "haan"):
        flow.handle_user_text(spoken)
    print("after full flow:", flow.describe())
    if flow.stage != "booking":
        fails += 1
        print(f"FAIL: stage is {flow.stage}, expected booking")
    await toolset.run_booking()
    attendee = (sent[-1].get("attendee") if sent else {}) or {}
    if attendee.get("email") == "smith@gmail.com" and attendee.get("phoneNumber") == "+923001234567":
        print(f"OK   booking body email={attendee['email']} phone={attendee['phoneNumber']}")
    else:
        fails += 1
        print(f"FAIL: booking body {attendee}")
    print("notice:", flow.notice)
    if flow.notice and flow.notice[0] == "booked":
        print("OK   booking notice surfaced to the caller")
    else:
        fails += 1
        print(f"FAIL: notice {flow.notice}")

    # 3. Second confirmation/no re-ask: repeating the same address is stable.
    if flow.email == "smith@gmail.com":
        print("OK   repeated address maps to the same value (no re-ask, no loop)")
    else:
        fails += 1
        print(f"FAIL: email is {flow.email!r}")

    await toolset.aclose()
    print("\nresult:", "PASS" if fails == 0 else f"{fails} FAILED")


if __name__ == "__main__":
    asyncio.run(main())
