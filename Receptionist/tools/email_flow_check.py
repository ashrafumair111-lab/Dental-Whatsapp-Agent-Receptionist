"""Check the email flow inside BookingFlow: capture -> read back -> confirm.

Run:  .venv\\Scripts\\python.exe tools/email_flow_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import agent  # noqa: E402
from cal_tools import BookingFlow  # noqa: E402

fails = 0


def booked_with_email(spelled: str) -> BookingFlow:
    flow = BookingFlow(fallback_email="clinic@example.com")
    flow.set_slot("2026-11-21", "10:00")
    flow.handle_user_text("Ali Khan")            # -> phone
    flow.handle_user_text("0300 1234567")         # -> email
    flow.handle_user_text(spelled)                # -> confirm_email
    return flow


print("read-back line:")
flow = booked_with_email("s m i t h at gmail dot com")
print("pending:", flow.pending_email, "| stage:", flow.stage)
ok = flow.stage == "confirm_email" and flow.pending_email == "smith@gmail.com"
fails += 0 if ok else 1
for lang, marker in (("en", "Your email"), ("ur", "صحیح ہے")):
    line = agent._scripted_line("readback", lang, flow)
    ok = marker in line and "s m i t h" in line
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} [{lang}] {line!r}")

print("\nconfirming/correcting both work:")
flow = booked_with_email("ali at gmail dot com")
flow.handle_user_text("nahi")
ok1 = flow.stage == "email" and flow.pending_email is None and flow.reask_key == "reask_email"
print(f"{'OK  ' if ok1 else 'FAIL'} correction returns to email (reask={flow.reask_key})")
flow = booked_with_email("ali at gmail dot com")
flow.handle_user_text("bilkul sahi hai")
ok2 = flow.email == "ali@gmail.com" and flow.stage == "booking"
print(f"{'OK  ' if ok2 else 'FAIL'} yes confirms and advances to booking (stage={flow.stage} email={flow.email})")
fails += (0 if ok1 else 1) + (0 if ok2 else 1)

print("\nskip email on request:")
flow = BookingFlow(fallback_email="clinic@example.com")
flow.set_slot("2026-11-21", "10:00")
flow.handle_user_text("Ali Khan")
flow.handle_user_text("0300 1234567")
flow.handle_user_text("email nahi hai")
print("stage:", flow.stage, "| skipped:", flow.email_skipped)
ok = flow.email_skipped and flow.stage == "booking"
fails += 0 if ok else 1
print(f"{'OK  ' if ok else 'FAIL'} caller opts out of giving an email")

print("\nemail asked at most MAX_EMAIL_TRIES times, then skipped:")
flow = BookingFlow()
flow.set_slot("2026-11-21", "10:00")
flow.handle_user_text("Ali Khan")
flow.handle_user_text("0300 1234567")
for _ in range(4):
    flow.handle_user_text("blah blah blah")
ok = flow.email_skipped and flow.stage == "booking"
print(f"{'OK  ' if ok else 'FAIL'} after 4 attempts: skipped={flow.email_skipped} stage={flow.stage}")
fails += 0 if ok else 1

print("\nside question is deferred, order is kept:")
flow = BookingFlow()
flow.set_slot("2026-11-21", "10:00")
flow.handle_user_text("aap ka clinic kahan hai")
ok = flow.deferred and flow.stage == "name"
print(f"{'OK  ' if ok else 'FAIL'} deferred={flow.deferred} stage={flow.stage}")
fails += 0 if ok else 1

print("\nresult:", "PASS" if fails == 0 else f"{fails} FAILED")
