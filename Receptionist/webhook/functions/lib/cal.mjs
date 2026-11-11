// Minimal Cal.com v2 client — mirrors the agent's Python toolset.
//
// Endpoints used:
//   GET  /slots       (cal-api-version: 2024-09-04)  – availability
//   POST /bookings    (cal-api-version: 2026-02-25)  – create booking
//   POST /bookings/{uid}/cancel                        – cancel booking

const CAL_API_BASE = 'https://api.cal.com/v2';
const APPOINTMENT_SLUG = '30min';
const DEFAULT_TIMEZONE = 'Asia/Karachi';
const HTTP_TIMEOUT_MS = 10000;

function calHeaders(apiVersion, apiKey) {
  return {
    Authorization: `Bearer ${apiKey}`,
    'cal-api-version': apiVersion,
    'Content-Type': 'application/json',
  };
}

function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error('Cal.com timed out')), ms)),
  ]);
}

async function calGet(path, params, apiKey, apiVersion) {
  const qs = new URLSearchParams(params).toString();
  const res = await withTimeout(
    fetch(`${CAL_API_BASE}${path}?${qs}`, {
      method: 'GET',
      headers: calHeaders(apiVersion, apiKey),
    }),
    HTTP_TIMEOUT_MS,
  );
  if (res.status >= 400) {
    throw new Error(`Cal.com ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

async function calPost(path, body, apiKey, apiVersion) {
  const res = await withTimeout(
    fetch(`${CAL_API_BASE}${path}`, {
      method: 'POST',
      headers: calHeaders(apiVersion, apiKey),
      body: JSON.stringify(body),
    }),
    HTTP_TIMEOUT_MS,
  );
  if (res.status >= 400) {
    throw new Error(`Cal.com ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

/** Save a short-lived list of slots for a date (cached in the caller). */
export async function getSlots(date, apiKey, username) {
  const params = {
    username,
    eventTypeSlug: APPOINTMENT_SLUG,
    start: date,
    end: date,
    timeZone: DEFAULT_TIMEZONE,
  };
  const data = await calGet('/slots', params, apiKey, '2024-09-04');
  return data?.data?.[date] ?? [];
}

/** Create a booking for the given UTC ISO start and attendee details. */
export async function createBooking({
  apiKey,
  username,
  startIso,
  attendee,
  notes,
}) {
  const body = {
    eventTypeSlug: APPOINTMENT_SLUG,
    username,
    start: startIso,
    attendee: {
      name: attendee.name,
      email: attendee.email,
      timeZone: DEFAULT_TIMEZONE,
      phoneNumber: attendee.phone,
    },
  };
  if (notes) body.bookingFieldsResponses = { notes };

  const data = await calPost('/bookings', body, apiKey, '2026-02-25');
  return data?.data ?? {};
}

export async function cancelBooking({ apiKey, uid, reason }) {
  const body = reason ? { cancellationReason: reason } : {};
  return calPost(`/bookings/${uid}/cancel`, body, apiKey, '2026-02-25');
}

/** Convert a Karachi-local date+slot ISO (YYYY-MM-DDTHH:mm:ssZ) to a readable label. */
export function formatSlot(iso) {
  const dt = new Date(iso);
  if (Number.isNaN(dt.getTime())) return iso;
  // Karachi is UTC+5 year-round.
  const pk = new Date(dt.getTime() + 5 * 3600 * 1000);
  const hours = String(pk.getUTCHours()).padStart(2, '0');
  const mins = String(pk.getUTCMinutes()).padStart(2, '0');
  return `${hours}:${mins}`;
}
