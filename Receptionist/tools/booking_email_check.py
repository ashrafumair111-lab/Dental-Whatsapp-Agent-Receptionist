"""Deterministic check: the email captured from speech reaches the booking.

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

from cal_tools import CalToolset, email_from_speech  # noqa: E402


async def main() -> None:
    toolset = CalToolset()
    tools = {t.info.name: t for t in toolset.tools}
    fails = 0

    sent: list[dict] = []

    async def fake_post(path: str, body: dict, version: str | None = None) -> dict:
        sent.append(body)
        return {"data": {"uid": "TEST123", "start": body["start"]}}

    toolset._post = fake_post  # type: ignore[method-assign]

    # 1. Booking never blocks on a missing email (that caused the re-ask loop).
    await tools["create_booking"](
        None,
        start_time="2026-11-14T09:00:00Z",
        attendee_name="Ali Khan",
        attendee_phone="0300 1234567",
    )
    if (sent[-1].get("attendee") or {}).get("email") == "patient@gmail.com":
        print("OK   no captured email -> booked with placeholder, caller not re-asked")
    else:
        fails += 1
        print(f"FAIL: booking body {sent[-1].get('attendee')}")

    # 2. The transcript is enough: code captures the address, no tool call.
    email = email_from_speech("meri email s m i t h at gmail dot com hai")
    toolset.set_email(email)
    stored = toolset._email
    if stored == "smith@gmail.com":
        print(f"OK   transcript captured {stored}")
    else:
        fails += 1
        print(f"FAIL: captured {stored!r}")

    # 3. Booking now reuses that address without asking again.
    result = await tools["create_booking"](
        None,
        start_time="2026-11-14T09:00:00Z",
        attendee_name="Ali Khan",
        attendee_phone="0300 1234567",
    )
    attendee = (sent[-1].get("attendee") or {}) if sent else {}
    if attendee.get("email") == "smith@gmail.com" and attendee.get("phoneNumber", "").startswith("+92"):
        print(f"OK   booking body email={attendee['email']} phone={attendee['phoneNumber']}")
    else:
        fails += 1
        print(f"FAIL: booking body {attendee}")
    if "smith@gmail.com" in str(result):
        print("OK   confirmation mentions the captured email")
    else:
        fails += 1
        print(f"FAIL: confirmation text: {result[:140]}")

    # 4. A later turn that repeats the same address is not captured again.
    again = email_from_speech("haan smith at gmail dot com sahi hai")
    if again == stored:
        print("OK   repeated address maps to the same value (no re-ask, no loop)")
    else:
        fails += 1
        print(f"FAIL: repeated address parsed as {again!r}")

    await toolset.aclose()
    print("\nresult:", "PASS" if fails == 0 else f"{fails} FAILED")


if __name__ == "__main__":
    asyncio.run(main())
