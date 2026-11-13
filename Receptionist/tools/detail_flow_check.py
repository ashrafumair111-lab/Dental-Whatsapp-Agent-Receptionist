"""Check the code-owned booking details: name -> phone -> email, each once.

Run:  .venv\\Scripts\\python.exe tools/detail_flow_check.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import agent  # noqa: E402
from cal_tools import BookingFlow, CalToolset, phone_from_speech, _name_candidate  # noqa: E402

fails = 0

print("phone_from_speech:")
for text, want in [
    ("0300 1234567", "+923001234567"),
    ("mera number 0300-1234567 hai", "+923001234567"),
    ("0300-123 4567", "+923001234567"),
    ("number zero three hundred one two three four five six seven", None),
    ("no digits here", None),
    ("2026", None),
]:
    got = phone_from_speech(text)
    ok = got == want
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {text!r:56} -> {got!r}")

print("\n_name_candidate:")
for text, want in [
    ("Ali Khan", "Ali Khan"),
    ("mera naam Ali Khan hai", "Ali Khan"),
    ("mera naam Ali hai", "Ali"),
    ("my name is Ali Khan", "Ali Khan"),
    ("Ali 0300 1234567", "Ali"),
    ("haan", None),
    ("0300 1234567", None),
    ("aap ka clinic kahan hai", None),
]:
    got = _name_candidate(text)
    ok = got == want
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {text!r:34} -> {got!r}")

print("\nchain: name -> phone -> email, each asked exactly once:")
flow = BookingFlow()
flow.set_slot("2026-11-21", "10:00")
flow.handle_user_text("Ali Khan")
ok = flow.name == "Ali Khan" and flow.stage == "phone"
print(f"{'OK  ' if ok else 'FAIL'} name captured -> stage={flow.stage} name={flow.name!r}")
fails += 0 if ok else 1

flow.handle_user_text("0300 1234567")
ok = flow.phone == "+923001234567" and flow.stage == "email"
print(f"{'OK  ' if ok else 'FAIL'} phone captured -> stage={flow.stage} phone={flow.phone!r}")
fails += 0 if ok else 1

print("directive after capture:", flow.next_directive())

print("\nname and phone in one breath:")
flow = BookingFlow()
flow.set_slot("2026-11-21", "10:00")
flow.handle_user_text("Ali Khan 0300 1234567")
ok = flow.name == "Ali Khan" and flow.phone == "+923001234567" and flow.stage == "email"
print(f"{'OK  ' if ok else 'FAIL'} captured name={flow.name!r} phone={flow.phone!r} stage={flow.stage}")
fails += 0 if ok else 1

print("\nquestion-style replies are deferred, not consumed as answers:")
flow = BookingFlow()
flow.set_slot("2026-11-21", "10:00")
flow.handle_user_text("aap ka clinic kahan hai")
ok = flow.name is None and flow.stage == "name" and flow.deferred
print(f"{'OK  ' if ok else 'FAIL'} deferred={flow.deferred} stage={flow.stage} name={flow.name!r}")
fails += 0 if ok else 1

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
        and attendee.get("phoneNumber") == "+923001234567"
    )

try:
    ok = asyncio.run(check_booking())
except Exception as exc:  # noqa: BLE001
    print(f"FAIL: {type(exc).__name__}: {exc}")
    ok = False
fails += 0 if ok else 1
print(f"{'OK  ' if ok else 'FAIL'} booking body uses stored name/phone/email")

print("\nresult:", "PASS" if fails == 0 else f"{fails} FAILED")
