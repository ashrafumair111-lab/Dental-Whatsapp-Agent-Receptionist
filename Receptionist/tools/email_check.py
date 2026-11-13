"""Check spoken-email capture (code-owned) and the booking tool surface.

Run:  .venv\\Scripts\\python.exe tools/email_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cal_tools import CalToolset, _spell_back, email_from_speech  # noqa: E402

# Whole caller sentences, the way STT hands them to us. This is what the agent
# actually reads: no tool call, no model, just the words the caller said.
TRANSCRIPTS = [
    ("meri email ali khan 123 at gmail dot com hai", "alikhan123@gmail.com"),
    ("email hai s m i t h at gmail dot com shukriya", "smith@gmail.com"),
    ("ali at gmail dot com", "ali@gmail.com"),
    ("mera email ali123 at gmail dot com hai aur phone 0300 1234567", "ali123@gmail.com"),
    ("email is j dot smith at yahoo dot com", "j.smith@yahoo.com"),
    ("meri email Ali.Khan@Gmail.com hai", "ali.khan@gmail.com"),
    ("email pe details bhej dein", None),
    ("Tuesday ki booking karni hai", None),
    ("mera phone 0300 1234567 hai", None),
    ("haan sahi hai", None),
    ("email hai ali at hotmail dot com", "ali@hotmail.com"),
    ("s dot m dot i dot t dot h at gmail dot com", "smith@gmail.com"),
    ("john dot smith at yahoo dot com", "john.smith@yahoo.com"),
    ("ali underscore khan at hotmail dot com", "ali_khan@hotmail.com"),
    ("Ali.Khan@Gmail.com. ", "ali.khan@gmail.com"),
]

failures = 0
for raw, want in TRANSCRIPTS:
    got = email_from_speech(raw)
    ok = got == want
    failures += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {raw!r:56} -> {got!r} (want {want!r})")

print("\nspell-back:")
for email in ("smith@gmail.com", "ali_khan.1@hotmail.com"):
    print(f"  {email} -> {_spell_back(email)}")

toolset = CalToolset(fallback_email="clinic@example.com")
names = sorted(getattr(t, "info").name for t in toolset.tools)
print("\ntools:", names)
required = ("set_appointment_time", "check_availability", "cancel_booking", "reschedule_booking")
missing = [n for n in required if n not in names]

# The model must have no email tool at all: the code owns that detail now.
leftover = [n for n in ("save_email", "create_booking") if n in names]

if missing:
    failures += 1
    print("MISSING TOOLS:", missing)
if leftover:
    failures += 1
    print("STILL EXPOSED TO THE MODEL:", leftover)

print("\nresult:", "PASS" if failures == 0 else f"{failures} FAILED")
