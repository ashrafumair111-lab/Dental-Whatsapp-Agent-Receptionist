"""Check the code-owned email flow: capture from speech, then one fixed read-back.

No LLM call and no network: this is the exact path the agent takes.

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
from cal_tools import email_from_speech  # noqa: E402
from livekit.agents.llm import ChatContext  # noqa: E402

CASES = [
    (
        "meri email ali at gmail dot com hai",
        "ur",
        "آپ کی email a l i at gmail dot com ہے، sahi hai؟",
    ),
    (
        "my email is ali at gmail dot com",
        "en",
        "Your email is a l i at gmail dot com — is that right?",
    ),
    (
        "email hai s m i t h at hotmail dot com",
        "ur",
        "آپ کی email s m i t h at hotmail dot com ہے، sahi hai؟",
    ),
]

fails = 0
for said, lang, want in CASES:
    email = email_from_speech(said)
    line = agent._readback_line(email, lang) if email else "<no email>"
    ok = line == want
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {said!r}")
    print(f"      {line!r}")

# Non-email turns must not trigger a read-back.
for said in ("Tuesday ki booking karni hai", "0300 1234567", "haan sahi hai"):
    if email_from_speech(said) is None:
        print(f"OK   no read-back for {said!r}")
    else:
        fails += 1
        print(f"FAIL: false email in {said!r}")

# The capture looks at the last two caller turns (a slow, split reply).
ctx = ChatContext()
ctx.add_message(role="assistant", content="Email bata dein?")
ctx.add_message(role="user", content="meri email s m i t h")
ctx.add_message(role="assistant", content="Theek hai.")
ctx.add_message(role="user", content="at gmail dot com")
joined = agent._recent_user_text(ctx)
if "s m i t h" in joined and "at gmail dot com" in joined:
    print(f"OK   recent user text = {joined!r}")
else:
    fails += 1
    print(f"FAIL: recent user text = {joined!r}")

print("\nresult:", "PASS" if fails == 0 else f"{fails} FAILED")
