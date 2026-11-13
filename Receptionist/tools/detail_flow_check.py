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
from cal_tools import CalToolset, phone_from_speech  # noqa: E402
from livekit.agents.llm import ChatContext  # noqa: E402
from livekit.agents.voice.agent import ModelSettings  # noqa: E402


def fresh() -> agent.Assistant:
    a = agent.Assistant()
    a._lang_state["lang"] = "ur"
    return a


def ask(assistant: agent.Assistant, *pairs: tuple[str, str]) -> str:
    ctx = ChatContext()
    ctx.add_message(role="system", content=assistant.instructions)
    for role, content in pairs:
        ctx.add_message(role=role, content=content)
    out = asyncio.run(assistant.llm_node(ctx, [], ModelSettings()))
    return out if isinstance(out, str) else "<STREAM (model called)>"


fails = 0

print("phone_from_speech:")
for text, want in [
    ("0300 1234567", "+923001234567"),
    ("mera number 0300-1234567 hai", "+923001234567"),
    ("0300-123 4567", "+923001234567"),
    ("no digits here", None),
    ("2026", None),
]:
    got = phone_from_speech(text)
    ok = got == want
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {text!r:34} -> {got!r}")

print("\n_name_candidate:")
for text, want in [
    ("Ali Khan", "Ali Khan"),
    ("mera naam Ali Khan hai", "Ali Khan"),
    ("my name is Ali Khan", "Ali Khan"),
    ("Ali 0300 1234567", "Ali"),
    ("haan", None),
    ("0300 1234567", None),
]:
    got = agent._name_candidate(text)
    ok = got == want
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {text!r:34} -> {got!r}")

print("\nchain (each detail asked once, never repeated):")
a = fresh()
out = ask(a, ("assistant", "آپ کا نام بتا دیں؟"), ("user", "Ali Khan"))
ok = out == agent._phone_prompt_line("ur") and a.cal_tools.name == "Ali Khan"
fails += 0 if ok else 1
print(f"{'OK  ' if ok else 'FAIL'} name turn -> {out!r} (name={a.cal_tools.name!r})")

out = ask(a, ("assistant", agent._phone_prompt_line("ur")), ("user", "0300 1234567"))
ok = out == agent._email_spell_line("ur") and a.cal_tools.phone == "+923001234567"
fails += 0 if ok else 1
print(f"{'OK  ' if ok else 'FAIL'} phone turn -> {out!r} (phone={a.cal_tools.phone!r})")

out = ask(a, ("assistant", agent._email_spell_line("ur")), ("user", "s m i t h at gmail dot com"))
ok = out == agent._readback_line("smith@gmail.com", "ur") and a.cal_tools.email == "smith@gmail.com"
fails += 0 if ok else 1
print(f"{'OK  ' if ok else 'FAIL'} email turn -> {out!r}")

print("\nname and phone in one breath:")
b = fresh()
out = ask(b, ("assistant", "آپ کا نام بتا دیں؟"), ("user", "Ali Khan 0300 1234567"))
ok = (
    out == agent._email_spell_line("ur")
    and b.cal_tools.name == "Ali Khan"
    and b.cal_tools.phone == "+923001234567"
)
fails += 0 if ok else 1
print(f"{'OK  ' if ok else 'FAIL'} combined -> {out!r} (name={b.cal_tools.name!r} phone={b.cal_tools.phone!r})")

print("\nbooking uses what was captured:")
async def check_booking() -> bool:
    toolset = CalToolset()
    toolset.set_name("Ali Khan")
    toolset.set_phone("+923001234567")
    toolset.set_email("smith@gmail.com")
    sent: list[dict] = []

    async def fake_post(path: str, body: dict, version: str | None = None) -> dict:
        sent.append(body)
        return {"data": {"uid": "TEST123", "start": body["start"]}}

    toolset._post = fake_post  # type: ignore[method-assign]
    tools = {t.info.name: t for t in toolset.tools}
    out = await tools["create_booking"](None, start_time="2026-11-20T09:00:00Z")
    await toolset.aclose()
    attendee = sent[0]["attendee"]
    return attendee["name"] == "Ali Khan" and attendee["phoneNumber"] == "+923001234567" and attendee["email"] == "smith@gmail.com"

try:
    ok = asyncio.run(check_booking())
except Exception as exc:  # noqa: BLE001
    print(f"FAIL: {exc}")
    ok = False
fails += 0 if ok else 1
print(f"{'OK  ' if ok else 'FAIL'} booking body uses stored name/phone/email")

print("\nresult:", "PASS" if fails == 0 else f"{fails} FAILED")
