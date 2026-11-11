// Mike — Dental Clinic appointment chatbot (state machine).
//
// The client demo flow: patient messages the appointment line, Mike answers,
// slots are checked in Cal.com and the booking is created — no voice needed.

import { createBooking, getSlots, cancelBooking, formatSlot } from './cal.mjs';

// (Cal.com blocks bookings on placeholder domains, so email is mandatory.)
const TIMEZONE = 'Asia/Karachi';

function services() {
  const raw = process.env.COMPANY_SERVICES;
  const envList = raw ? raw.split(';').map((s) => s.trim()).filter(Boolean) : [];
  return envList.length
    ? envList
    : [
        'Dental check-up and cleaning',
        'Dental consultation and diagnosis',
        'Tooth filling and restorations',
        'Root canal treatment',
        'Tooth extraction and surgical procedures',
        'Braces and orthodontic consultation',
        'Teeth cleaning and scaling',
      ];
}

function norm(text) {
  return String(text ?? '')
    .trim()
    .toLowerCase()
    .replace(/[!?.,;]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function todayPK() {
  return new Date(Date.now() + 5 * 3600 * 1000).toISOString().slice(0, 10);
}

function addDaysPK(days) {
  return new Date(Date.now() + 5 * 3600 * 1000 + days * 86400000).toISOString().slice(0, 10);
}

function humanDate(iso) {
  const [y, m, d] = iso.split('-').map(Number);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${String(d).padStart(2, '0')} ${months[(m ?? 1) - 1]} ${y}`;
}

const MONTHS = {
  jan: 0, january: 0, feb: 1, february: 1, mar: 2, march: 2, apr: 3, april: 3,
  may: 4, jun: 5, june: 5, jul: 6, july: 6, aug: 7, august: 7, sep: 8, september: 8,
  oct: 9, october: 9, nov: 10, november: 10, dec: 11, december: 11,
};

/** Strip a trailing time phrase ("at 9 am", "at 9:30", "baje") before date parsing. */
function stripTimePhrase(text) {
  return text
    .replace(/\s*(?:at|by)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|baje|hours?)?\s*$/i, '')
    .replace(/\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|baje|hours?)\s*$/i, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Parse a patient's date answer into YYYY-MM-DD. Returns null if unclear. */
export function parseDate(text) {
  const t = stripTimePhrase(norm(text));
  if (!t) return null;
  if (/\b(today|aaj|aj)\b/.test(t)) return todayPK();
  if (/day after tomorrow|parso|purso|2 days/.test(t)) return addDaysPK(2);
  if (/\b(tomorrow|kal)\b/.test(t)) return addDaysPK(1);

  const weekdays = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
  for (let i = 0; i < 7; i++) {
    if (new RegExp(`\\b${weekdays[i]}\\b`).test(t)) {
      const pkNow = new Date(Date.now() + 5 * 3600 * 1000);
      const day = pkNow.getUTCDay();
      let diff = (i - day + 7) % 7;
      if (diff === 0) diff = 7; // bare "monday" means the coming Monday
      return new Date(pkNow.getTime() + diff * 86400000).toISOString().slice(0, 10);
    }
  }

  // DD/MM/YYYY, DD-MM-YYYY or DD.MM.YYYY (European order)
  let m = t.match(/\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b/);
  if (m) {
    let [, d, mo, y] = m;
    y = Number(y);
    if (y < 100) y += 2000;
    return `${y}-${String(mo).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
  }
  // YYYY-MM-DD
  m = t.match(/\b(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\b/);
  if (m) {
    const [, y, mo, d] = m;
    return `${y}-${String(mo).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
  }
  // "15 june" / "june 15"
  m = t.match(/\b(\d{1,2})\s*([a-z]+)\b/);
  if (m && MONTHS[m[2]] !== undefined) {
    const d = Number(m[1]);
    const mo = MONTHS[m[2]];
    const year = Number(todayPK().slice(0, 4));
    const iso = `${year}-${String(mo + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
    return iso < todayPK() ? `${year + 1}-${String(mo + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}` : iso;
  }
  m = t.match(/\b([a-z]+)\s*(\d{1,2})\b/);
  if (m && MONTHS[m[1]] !== undefined) {
    const d = Number(m[2]);
    const mo = MONTHS[m[1]];
    const year = Number(todayPK().slice(0, 4));
    const iso = `${year}-${String(mo + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
    return iso < todayPK() ? `${year + 1}-${String(mo + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}` : iso;
  }
  return null;
}

/** Match a slot choice to a stored slot. Returns the slot ISO or null. */
export function pickSlot(text, slots) {
  const t = norm(text);
  if (!Array.isArray(slots) || slots.length === 0) return null;

  // Plain number = position in the list.
  const num = t.match(/^#?(\d{1,2})$/);
  if (num) {
    const idx = Number(num[1]) - 1;
    if (idx >= 0 && idx < slots.length) return slots[idx].start;
    // A bare small number may be an hour ("3" -> 3 PM or 3 AM).
    const hour = Number(num[1]);
    const match = slots.find((s) => {
      const h = new Date(s.start).getUTCHours();
      const pk = (h + 5) % 24;
      return pk === hour || pk === hour + 12;
    });
    return match ? match.start : null;
  }
  // "3 pm" / "3baje" / "15:30" style.
  const hourMatch = t.match(/(\d{1,2})(?::(\d{2}))?\s*(am|pm|baje)?/);
  if (hourMatch) {
    let hour = Number(hourMatch[1]);
    const mins = Number(hourMatch[2] ?? '0');
    const suffix = hourMatch[3];
    if (suffix === 'pm' && hour < 12) hour += 12;
    if (suffix === 'am' && hour === 12) hour = 0;
    const exact = slots.find((s) => {
      const pk = new Date(new Date(s.start).getTime() + 5 * 3600 * 1000);
      return pk.getUTCHours() === hour && pk.getUTCMinutes() === mins;
    });
    if (exact) return exact.start;
    const sameHour = slots.find((s) => {
      const pk = new Date(new Date(s.start).getTime() + 5 * 3600 * 1000);
      return pk.getUTCHours() === hour;
    });
    return sameHour ? sameHour.start : null;
  }
  return null;
}

function isYes(text) {
  return /^(yes|y|yeah|yep|haan|han|ji|ji han|confirm|ok|okay|theek|bilkul)\b/.test(norm(text));
}

function isNo(text) {
  return /^(no|n|nope|nahi|nahin|mat|cancel)\b/.test(norm(text));
}

function servicesMenu(prefix = '') {
  const lines = services().map((s, i) => `${i + 1}) ${s}`);
  return `${prefix}Which service do you need?\n${lines.join('\n')}\n\nReply with the number, or just write the treatment name.`;
}

export function menuReply() {
  return [
    'Hello! 🦷 I am Mike, the receptionist at Dental Clinic.',
    '',
    "I can help you with questions about our services or book an appointment for you.",
    '',
    'Type **book** to schedule an appointment, **services** to see what we offer, or **cancel** to cancel a booking.',
  ].join('\n');
}

/** Main entry: given the patient message + session, return { reply, session }. */
export async function chat({ text, session, waId, cal }) {
  const t = norm(text);

  // ---- global commands -------------------------------------------------
  if (/^(hi|hello|hey|salaam|salam|assalam|good morning|good evening)\b/.test(t)) {
    return done(session, menuReply(), { step: 'idle', data: {} });
  }
  if (/cancel/.test(t) && session.step !== 'cancel') {
    return done(session, 'I can help cancel that. Please type your booking ID (it looks like "abc123def").', {
      step: 'cancel',
      data: {},
    });
  }

  // ---- cancel flow -----------------------------------------------------
  if (session.step === 'cancel') {
    const uid = t.replace(/[^a-z0-9-]/gi, '');
    if (!uid) return done(session, 'Please type the booking ID from your confirmation message.');
    try {
      await cancelBooking({ apiKey: cal.apiKey, uid, reason: 'Patient requested via WhatsApp' });
      return done(session, `Done — booking ${uid} has been cancelled. Sorry to see you go!`, {
        step: 'idle',
        data: {},
      });
    } catch (err) {
      return done(session, `Sorry, I could not cancel ${uid}. Please try again or contact the clinic.`);
    }
  }

  // ---- booking steps ---------------------------------------------------
  switch (session.step) {
    case 'idle': {
      if (/services?|offer|treatments?|kya (ha|available)/.test(t)) {
        return done(session, servicesMenu(), { step: 'service', data: {} });
      }
      if (/book|appointment|appoint|check ?up|cleaning|filling|root canal|braces|extraction|scaler/.test(t)) {
        return done(session, servicesMenu('Sure! '), { step: 'service', data: {} });
      }
      if (/human|agent|doctor|person/.test(t)) {
        return done(
          session,
          'I can help with booking and services right here in the chat. For a doctor, I can pass a request along or you can book an appointment and they will call back.',
          { step: 'idle', data: {} },
        );
      }
      return done(session, menuReply());
    }

    case 'service': {
      const list = services();
      let choice = null;
      const num = t.match(/^#?(\d{1,2})\b/);
      if (num && Number(num[1]) >= 1 && Number(num[1]) <= list.length) {
        choice = list[Number(num[1]) - 1];
      } else {
        choice = list.find((s) => s.toLowerCase().includes(t) || t.includes(s.toLowerCase().split(' ')[0]));
        if (!choice && t.length > 2) choice = text.trim(); // accept free text
      }
      if (!choice) return done(session, servicesMenu());

      return done(session, 'Great! Which date would you like? You can say "today" or "tomorrow".', {
        step: 'date',
        data: { ...session.data, service: choice },
      });
    }

    case 'date': {
      const iso = parseDate(text);
      if (!iso) {
        return done(session, 'Which date, please? For example: "tomorrow".');
      }
      let slots = [];
      try {
        slots = await getSlots(iso, cal.apiKey, cal.username);
      } catch (err) {
        console.error('getSlots failed:', err?.message ?? err);
        return done(session, 'The availability system is busy right now — please try again in a couple of minutes.', {
          step: 'date',
          data: { ...session.data, date: iso },
        });
      }
      if (!slots.length) {
        return done(session, `Nothing available on ${humanDate(iso)}. Try another date?`, {
          step: 'date',
          data: { ...session.data, date: iso },
        });
      }
      const shown = slots.slice(0, 8);
      const lines = shown.map((s, i) => `${i + 1}) ${formatSlot(s.start)}`);
      return done(
        session,
        `Available times on ${humanDate(iso)}:\n${lines.join('\n')}\n\nReply with a number, or write a time like "3 pm".`,
        { step: 'time', data: { ...session.data, date: iso, slots: shown } },
      );
    }

    case 'time': {
      const slots = session.data.slots ?? [];
      const slotStart = pickSlot(t, slots);
      if (!slotStart) {
        return done(session, timePrompt(session.data.slots ?? []));
      }
      return done(session, 'Almost done — what is your name, please?', {
        step: 'name',
        data: { ...session.data, slot: slotStart },
      });
    }

    case 'name': {
      const name = String(text ?? '').trim();
      if (name.length < 2) return done(session, 'Please tell me your full name.');
      return done(session, `Thanks, ${name}. What email should we use for the confirmation?`, {
        step: 'email',
        data: { ...session.data, name },
      });
    }

    case 'email': {
      const raw = String(text ?? '').trim();
      if (/^(skip|none|no|nah|nahi|na)$/i.test(norm(raw))) {
        // Cal.com refuses placeholder domains, so the address is required.
        return done(session, 'I need an email to send your confirmation — please type it, like "name@gmail.com".');
      }
      const m = raw.match(/[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}/);
      if (!m) {
        return done(session, 'That email did not look right. Please type it like "name@gmail.com".');
      }
      const email = m[0].toLowerCase();
      const summary = [
        `Service: ${session.data.service}`,
        `Date: ${humanDate(session.data.date)}`,
        `Time: ${formatSlot(session.data.slot)}`,
        `Name: ${session.data.name}`,
        `Phone: +${waId.replace(/\D/g, '')}`,
        `Email: ${email}`,
      ].join('\n');
      return done(session, `Please confirm:\n\n${summary}\n\nReply **YES** to book, or **NO** to cancel.`, {
        step: 'confirm',
        data: { ...session.data, email },
      });
    }

    case 'confirm': {
      if (isNo(t)) {
        return done(session, 'No problem — I have not booked anything. Type "book" whenever you are ready.', {
          step: 'idle',
          data: {},
        });
      }
      if (!isYes(t)) {
        return done(session, 'Reply **YES** to confirm the booking or **NO** to cancel.');
      }
      const digits = waId.replace(/\D/g, '');
      try {
        const booking = await createBooking({
          apiKey: cal.apiKey,
          username: cal.username,
          startIso: session.data.slot,
          attendee: {
            name: session.data.name,
            email: session.data.email,
            phone: `+${digits}`,
          },
          notes: session.data.service,
        });
        const when = `${humanDate(session.data.date)} at ${formatSlot(session.data.slot)}`;
        return done(
          session,
          [
            `Booked ✓ — ${session.data.service}`,
            `on ${when}`,
            `See you then, ${session.data.name}!`,
            '',
            'If you need to cancel, just type "cancel".',
          ].join('\n'),
          { step: 'idle', data: {} },
        );
      } catch (err) {
        console.error('createBooking failed:', err?.message ?? err);
        return done(session, 'Sorry — the booking system is busy right now. Please try again in a couple of minutes.', {
          step: 'idle',
          data: {},
        });
      }
    }

    default:
      return done(session, menuReply(), { step: 'idle', data: {} });
  }
}

function timePrompt(slots) {
  if (!slots?.length) {
    return 'No openings that day, unfortunately. Try another date — just write it.';
  }
  const lines = slots.slice(0, 8).map((s, i) => `${i + 1}) ${formatSlot(s.start)}`);
  return `Available times:\n${lines.join('\n')}\n\nReply with a number, or write a time like "3 pm".`;
}

function done(_session, reply, extra) {
  return { reply, ...(extra ? { extra } : {}) };
}
