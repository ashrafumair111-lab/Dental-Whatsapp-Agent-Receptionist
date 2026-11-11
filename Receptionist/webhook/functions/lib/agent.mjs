// Mike — WhatsApp chatbot driven by Groq (openai/gpt-oss-120b).
//
// The patient messages the appointment line, Mike conducts a short intake
// (service, date, slot, name, phone, email) and uses tools to check Cal.com
// availability and book/cancel appointments. Replies are short,
// Mike-style, and English-friendly.

import { getSlots, createBooking, cancelBooking, formatSlot } from './cal.mjs';

const DEFAULT_SERVICES = [
  'Dental check-up and cleaning',
  'Dental consultation and diagnosis',
  'Tooth filling and restorations',
  'Root canal treatment',
  'Tooth extraction and surgical procedures',
  'Braces and orthodontic consultation',
  'Teeth cleaning and scaling',
];

function servicesList() {
  const raw = process.env.COMPANY_SERVICES;
  const envList = raw ? raw.split(';').map((s) => s.trim()).filter(Boolean) : [];
  return envList.length ? envList : DEFAULT_SERVICES;
}

function systemPrompt(todayLabel) {
  return [
    `You are Mike, the WhatsApp receptionist at Dental Clinic. Today is ${todayLabel} (Asia/Karachi).`,
    `Services: ${servicesList().join('; ')}.`,
    `Duties: (1) Answer dental questions briefly. (2) Help the patient book an appointment end-to-end using the tools. (3) Cancel on request.`,
    `Rules: short replies; one question at a time. ALWAYS call check_availability before quoting times. Never invent slots, confirmations, prices or timings.`,
    `Booking checklist: service -> date -> time slot (use check_availability) -> full name -> phone -> email -> patient confirms -> book.`,
    `Always ask the patient's confirmation before booking. Do NOT book on ambiguous intent. If the tool fails, say so honestly.`,
  ].join('\n');
}

const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'check_availability',
      description:
        'Check bookable appointment slots on a date. The date must be YYYY-MM-DD. ' +
        'Returns the available slots as { start (ISO), local (HH:MM) } pairs.',
      parameters: {
        type: 'object',
        properties: {
          date: { type: 'string', description: 'Date in YYYY-MM-DD format, e.g. 2026-11-12' },
        },
        required: ['date'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'book_appointment',
      description:
        'Book a dental appointment. Call only after the patient has confirmed service, ' +
        'date, time, name, phone and email. Returns a booking confirmation.',
      parameters: {
        type: 'object',
        properties: {
          start_time: {
            type: 'string',
            description: "ISO 8601 start time, e.g. '2026-11-12T09:00:00.000+05:00'",
          },
          attendee_name: { type: 'string' },
          attendee_email: { type: 'string' },
          attendee_phone: { type: 'string' },
          notes: { type: 'string', description: 'Reason for the visit (the service)' },
        },
        required: ['start_time', 'attendee_name', 'attendee_email', 'attendee_phone'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'cancel_booking',
      description: 'Cancel an existing appointment.',
      parameters: {
        type: 'object',
        properties: {
          booking_uid: { type: 'string' },
          reason: { type: 'string' },
        },
        required: ['booking_uid'],
      },
    },
  },
];

async function callGroq(groqApiKey, messages) {
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${groqApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'openai/gpt-oss-120b',
        messages,
        tools: TOOLS,
        tool_choice: 'auto',
        temperature: 0.4,
        max_completion_tokens: 384,
      }),
    });
    if (res.status === 429) {
      // Groq free tier can clear in seconds — honor retry-after when it's small.
      const retryAfter = Number(res.headers.get('retry-after') ?? '1.5');
      if (retryAfter <= 4 && attempt === 0) {
        await new Promise((r) => setTimeout(r, Math.max(0, retryAfter * 1000) + 200));
        continue;
      }
      throw new Error(`rate_limit_exceeded retry_after=${retryAfter}`);
    }
    if (!res.ok) {
      throw new Error(`Groq ${res.status}: ${await res.text()}`);
    }
    return res.json();
  }
  throw new Error('Groq rate limited after retry');
}

export async function runAgent({ waId, text, history, cal }) {
  const today = new Date().toLocaleDateString('en-GB', {
    timeZone: 'Asia/Karachi',
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  const recentHistory = history.slice(-8);
  const messages = [{ role: 'system', content: systemPrompt(today) }, ...recentHistory, { role: 'user', content: text }];

  const senderDigits = String(waId).replace(/\D/g, '');

  for (let step = 0; step < 4; step += 1) {
    const data = await callGroq(cal.groqApiKey, messages);
    const choice = data?.choices?.[0]?.message;
    if (!choice) throw new Error('Groq returned no message');

    messages.push(choice);

    const toolCalls = choice.tool_calls ?? [];
    if (toolCalls.length === 0) {
      return { reply: choice.content?.trim() || 'Sorry, I did not catch that.', messages };
    }

    for (const tc of toolCalls) {
      const name = tc.function?.name;
      let args = {};
      try { args = JSON.parse(tc.function?.arguments ?? '{}'); } catch { args = {}; }

      let result;
      try {
        switch (name) {
          case 'check_availability': {
            const date = String(args.date ?? '');
            const slots = await getSlots(date, cal.apiKey, cal.username);
            result = {
              date,
              slots: slots.slice(0, 20).map((s) => ({ start: s.start, local: formatSlot(s.start) })),
            };
            break;
          }
          case 'book_appointment': {
            const booking = await createBooking({
              apiKey: cal.apiKey,
              username: cal.username,
              startIso: String(args.start_time ?? ''),
              attendee: {
                name: String(args.attendee_name ?? 'Patient'),
                email: String(args.attendee_email ?? `patient${senderDigits}@example.com`),
                phone: String(args.attendee_phone ?? `+${senderDigits}`),
              },
              notes: String(args.notes ?? ''),
            });
            result = { success: true, uid: booking?.uid, start: booking?.start };
            break;
          }
          case 'cancel_booking': {
            await cancelBooking({
              apiKey: cal.apiKey,
              uid: String(args.booking_uid ?? ''),
              reason: String(args.reason ?? 'Cancelled via WhatsApp'),
            });
            result = { success: true };
            break;
          }
          default:
            result = { error: `Unknown tool: ${name}` };
        }
      } catch (err) {
        result = { error: String(err?.message ?? err) };
      }

      messages.push({ role: 'tool', tool_call_id: tc.id, name, content: JSON.stringify(result) });
    }
  }

  return { reply: 'I could not complete that — please try again.', messages };
}
