"""End-to-end check of the code-owned booking flow (offline, no LLM, no network).

Run:  .venv\\Scripts\\python.exe tools/booking_flow_check.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cal_tools import BookingFlow, CalToolset, email_from_speech, _digits_from_phone_text  # noqa: E402

fails = 0

print("email capture:")
for text, want in [
    ("meri email s m i t h at gmail dot com hai", "smith@gmail.com"),
    ("my email is ali at yahoo dot com", "ali@yahoo.com"),
    ("0300 1234567", None),
    ("Tuesday ki booking hai", None),
]:
    got = email_from_speech(text)
    ok = got == want
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {text!r:45} -> {got!r}")

print("\nphone digits any format:")
for text in ("0300 1234567", "0300-1234567", "+92 300 123 4567", "00300 123 4567"):
    d = _digits_from_phone_text(text)
    ok = len(d) >= 7
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {text!r:30} -> {d!r}")

print("\nchain: name -> phone -> email, each asked exactly once:")
flow = BookingFlow(fallback_email="clinic@example.com")
flow.set_slot("2026-11-20", "15:30")
flow.handle_user_text("Ali Khan")
ok = flow.name == "Ali Khan" and flow.stage == "phone" and flow.next_directive() == "ask_phone"
print(f"{'OK  ' if ok else 'FAIL'} name turn -> stage={flow.stage}")
fails += 0 if ok else 1

flow.handle_user_text("0300 1234567")
ok = flow.phone == "03001234567" and flow.stage == "email"
print(f"{'OK  ' if ok else 'FAIL'} phone turn -> phone={flow.phone!r} stage={flow.stage}")
fails += 0 if ok else 1

flow.handle_user_text("s m i t h at gmail dot com")
ok = flow.stage == "confirm_email" and flow.pending_email == "smith@gmail.com"
print(f"{'OK  ' if ok else 'FAIL'} email turn -> pending={flow.pending_email} stage={flow.stage}")
fails += 0 if ok else 1

flow.handle_user_text("haan sahi hai")
ok = flow.email == "smith@gmail.com" and flow.stage == "booking"
print(f"{'OK  ' if ok else 'FAIL'} confirm turn -> email={flow.email} stage={flow.stage}")
fails += 0 if ok else 1

print("\nany phone format accepted, no country code asked:")
for text in ("0300-1234567", "+92 (300) 123 4567", "0012300 123 4567"):
    flow = BookingFlow()
    flow.set_slot("2026-11-20", "15:30")
    flow.handle_user_text("Ali")
    flow.handle_user_text(text)
    ok = flow.phone is not None and flow.stage == "email"
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {text!r:22} -> phone={flow.phone!r} stage={flow.stage}")

print("\nbooking body uses what was captured:")

async def check_booking() -> bool:
    toolset = CalToolset(fallback_email="clinic@example.com")
    f = toolset.flow
    f.set_slot("2026-11-21", "15:30")
    for spoken in ("Ali Khan", "0300 1234567", "s m i t h at gmail dot com", "haan"):
        f.handle_user_text(spoken)

    sent: list[dict] = []

    async def fake_post(path: str, body: dict, version: str | None = None) -> dict:
        sent.append(body)
        return {"data": {"uid": "TEST123", "start": body["start"]}}

    toolset._post = fake_post  # type: ignore[method-assign]
    await toolset.run_booking()
    await toolset.aclose()
    if not sent:
        print("FAIL: run_booking did not POST")
        return False
    attendee = sent[0].get("attendee", {})
    return (
        attendee.get("name") == "Ali Khan"
        and attendee.get("email") == "smith@gmail.com"
        and attendee.get("phoneNumber") == "03001234567"
    )

try:
    ok = asyncio.run(check_booking())
except Exception as exc:  # noqa: BLE001
    print(f"FAIL: {type(exc).__name__}: {exc}")
    ok = False
fails += 0 if ok else 1
print(f"{'OK  ' if ok else 'FAIL'} booking body uses stored name/phone/email")

print("\nresult:", "PASS" if fails == 0 else f"{fails} FAILED")
