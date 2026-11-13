"""Check spoken-email normalisation and the booking tool surface.

Run:  .venv\\Scripts\\python.exe tools\\email_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env.local")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cal_tools import CalToolset, _spell_back, _validate_email  # noqa: E402

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

failures = 0
for raw, want in CASES:
    got = _validate_email(raw)
    ok = got == want
    failures += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {raw!r:52} -> {got!r} (want {want!r})")

print("\nspell-back:")
for email in ("smith@gmail.com", "ali_khan.1@hotmail.com"):
    print(f"  {email} -> {_spell_back(email)}")

toolset = CalToolset()
names = sorted(getattr(t, "info").name for t in toolset.tools)
print("\ntools:", names)
missing = [n for n in ("save_email", "create_booking", "check_availability") if n not in names]
if missing:
    failures += 1
    print("MISSING TOOLS:", missing)

print("\nresult:", "PASS" if failures == 0 else f"{failures} FAILED")
