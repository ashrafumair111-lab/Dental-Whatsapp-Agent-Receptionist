"""Deterministic check: the email captured by save_email reaches the booking body.

Run:  .venv\\Scripts\\python.exe tools\\booking_email_check.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cal_tools import CalToolset  # noqa: E402


class FakeCtx:
    """Just enough of RunContext for the tools (ctx.with_filler)."""

    def with_filler(self, *args, **kwargs):
        @asynccontextmanager
        async def _cm():
            yield

        return _cm()


async def main() -> None:
    toolset = CalToolset()
    tools = {t.info.name: t for t in toolset.tools}
    ctx = FakeCtx()
    fails = 0

    sent: list[dict] = []

    async def fake_post(path: str, body: dict, version: str | None = None) -> dict:
        sent.append(body)
        return {"data": {"uid": "TEST123", "start": body["start"]}}

    toolset._post = fake_post  # type: ignore[method-assign]

    # 1. Booking without a saved email must be refused, not silently booked.
    try:
        await tools["create_booking"](
            ctx,
            start_time="2026-11-14T09:00:00Z",
            attendee_name="Ali Khan",
            attendee_phone="0300 1234567",
        )
        fails += 1
        print("FAIL: booking went through with no email")
    except Exception as exc:  # noqa: BLE001
        print(f"OK   no email -> refused ({type(exc).__name__}: {str(exc)[:70]})")

    # 2. save_email accepts a spelled-out address and stores it.
    out = await tools["save_email"](ctx, spoken="s m i t h at gmail dot com")
    stored = toolset._email
    if stored == "smith@gmail.com":
        print(f"OK   save_email stored {stored}")
    else:
        fails += 1
        print(f"FAIL: stored {stored!r}")
    if "s m i t h at gmail dot com" in out:
        print("OK   read-back line present")
    else:
        fails += 1
        print(f"FAIL: read-back missing: {out[:120]}")

    # 3. Booking now reuses that address without asking again.
    result = await tools["create_booking"](
        ctx,
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
        print("OK   confirmation mentions the saved email")
    else:
        fails += 1
        print(f"FAIL: confirmation text: {result[:140]}")

    await toolset.aclose()
    print("\nresult:", "PASS" if fails == 0 else f"{fails} FAILED")


if __name__ == "__main__":
    asyncio.run(main())
