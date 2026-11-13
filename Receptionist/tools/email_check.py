"""Check spoken-email normalisation, transcript capture and the tool surface.

Run:  .venv\\Scripts\\python.exe tools\\email_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cal_tools import (  # noqa: E402
    CalToolset,
    _spell_back,
    _validate_email,
    email_from_speech,
)

CASES = [
    # letter-by-letter, spaced
    ("s m i t h at gmail dot com", "smith@gmail.com"),
    ("a l i 1 2 3 at gmail dot com", "ali123@gmail.com"),
    # letter-by-letter with "dot" between letters
    ("s dot m dot i dot t dot h at gmail dot com", "smith@gmail.com"),
    # normal speech
    ("john at example dot com", "john@example.com"),
    ("john dot smith at yahoo dot com", "john.smith@yahoo.com"),
    ("j.smith at yahoo dot com", "j.smith@yahoo.com"),
    # filler words
    ("my email is ali123 at gmail dot com", "ali123@gmail.com"),
    ("it's ali at hotmail dot com", "ali@hotmail.com"),
    # symbols
    ("ali underscore khan at hotmail dot com", "ali_khan@hotmail.com"),
    ("ali dash khan at hotmail dot com", "ali-khan@hotmail.com"),
    ("ali plus 1 at gmail dot com", "ali+1@gmail.com"),
    # already typed
    ("ali123@gmail.com", "ali123@gmail.com"),
    ("  Ali.Khan@Gmail.com. ", "ali.khan@gmail.com"),
    # STT punctuation glitch
    ("ali at gmail . com", "ali@gmail.com"),
    # hopeless speech still yields a usable address instead of an error
    ("a l i k h a n", "alikhan@gmail.com"),
    ("", "patient@gmail.com"),
]

# Whole caller sentences, the way STT hands them to us. This is what the agent
# actually reads: no tool call, no model, just the words the caller said.
TRANSCRIPTS = [
    ("meri email ali khan 123 at gmail dot com hai", "alikhan123@gmail.com"),
    ("email hai s m i t h at gmail dot com shukriya", "smith@gmail.com"),
    ("ali at gmail dot com", "ali@gmail.com"),
    ("mera email ali123 at gmail dot com hai aur phone 0300 1234567", "ali123@gmail.com"),
    ("email is j dot smith at yahoo dot com", "j.smith@yahoo.com"),
    ("meri email ali at the rate gmail dot com hai", "ali@gmail.com"),
    ("mera email Ali.Khan@Gmail.com hai", "ali.khan@gmail.com"),
    ("email pe details bhej dein", None),
    ("Tuesday ki booking karni hai", None),
    ("mera phone 0300 1234567 hai", None),
    ("haan sahi hai", None),
]

failures = 0
for raw, want in CASES:
    got = _validate_email(raw)
    ok = got == want
    failures += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {raw!r:52} -> {got!r} (want {want!r})")

print("\ntranscripts:")
for raw, want in TRANSCRIPTS:
    got = email_from_speech(raw)
    ok = got == want
    failures += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {raw!r:56} -> {got!r} (want {want!r})")

print("\nspell-back:")
for email in ("smith@gmail.com", "ali_khan.1@hotmail.com"):
    print(f"  {email} -> {_spell_back(email)}")

toolset = CalToolset()
names = sorted(getattr(t, "info").name for t in toolset.tools)
print("\ntools:", names)
required = ("create_booking", "check_availability", "cancel_booking", "reschedule_booking")
missing = [n for n in required if n not in names]
if missing:
    failures += 1
    print("MISSING TOOLS:", missing)

# The model must have no email tool at all: the code owns that detail now.
if "save_email" in names:
    failures += 1
    print("FAIL: save_email is still exposed to the model")

print("\nresult:", "PASS" if failures == 0 else f"{failures} FAILED")
