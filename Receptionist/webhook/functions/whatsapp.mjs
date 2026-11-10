// Meta WhatsApp Cloud API call webhook -> LiveKit WhatsApp Connector.
//
// Flow (inbound call):
//   1. Caller dials the business WhatsApp number.
//   2. Meta POSTs a `calls` webhook with event=connect and an SDP offer.
//   3. We verify the X-Hub-Signature-256 header, then call the LiveKit
//      Connector API (AcceptWhatsAppCall) which joins the media and creates a room.
//   4. The LiveKit agent named in LIVEKIT_AGENT_NAME is dispatched into that room.
//
// Env vars (set in Netlify, never commit them):
//   WHATSAPP_VERIFY_TOKEN     - any string; must match the Meta webhook "Verify token"
//   WHATSAPP_APP_SECRET       - Meta app secret, used to verify webhook signatures
//   WHATSAPP_PHONE_NUMBER_ID  - e.g. 1378038128734333
//   WHATSAPP_ACCESS_TOKEN     - permanent system-user token (whatsapp_business_messaging)
//   WHATSAPP_CLOUD_API_VERSION- default 23.0 (must match the subscribed webhook version)
//   LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET - LiveKit Cloud project
//   LIVEKIT_AGENT_NAME        - defaults to "MetaByte Solutions Receptionist"

import crypto from 'node:crypto';
import { LiveKitAPI, RoomAgentDispatch } from 'livekit-server-sdk';

const VERIFY_TOKEN = process.env.WHATSAPP_VERIFY_TOKEN ?? '';
const APP_SECRET = process.env.WHATSAPP_APP_SECRET ?? '';
const PHONE_NUMBER_ID = process.env.WHATSAPP_PHONE_NUMBER_ID ?? '';
const ACCESS_TOKEN = process.env.WHATSAPP_ACCESS_TOKEN ?? '';
const CLOUD_API_VERSION = process.env.WHATSAPP_CLOUD_API_VERSION ?? '23.0';
const AGENT_NAME = process.env.LIVEKIT_AGENT_NAME ?? 'MetaByte Solutions Receptionist';

/** Constant-time compare of Meta's sha256 signature against the raw body. */
function isValidSignature(rawBody, header) {
  if (!APP_SECRET) {
    console.warn('WHATSAPP_APP_SECRET is not set - skipping signature verification');
    return true;
  }
  if (!header || !header.startsWith('sha256=')) return false;

  const expected = crypto.createHmac('sha256', APP_SECRET).update(rawBody).digest('hex');
  const received = header.slice('sha256='.length);

  const a = Buffer.from(expected, 'hex');
  const b = Buffer.from(received, 'hex');
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}

function reply(statusCode, body, headers = {}) {
  return {
    statusCode,
    headers: { 'content-type': 'application/json', ...headers },
    body: typeof body === 'string' ? body : JSON.stringify(body),
  };
}

async function acceptInboundCall(call) {
  const callId = call?.id;
  const sdp = call?.session?.sdp;
  const sdpType = call?.session?.sdp_type ?? 'offer';

  if (!callId || !sdp) {
    return { callId: callId ?? null, ok: false, error: 'webhook missing call id or sdp' };
  }

  const roomName = `whatsapp-${callId}`.replace(/[^a-zA-Z0-9_-]/g, '-').slice(0, 60);

  try {
    const api = new LiveKitAPI();
    const res = await api.connector.acceptWhatsAppCall({
      whatsappPhoneNumberId: PHONE_NUMBER_ID,
      whatsappApiKey: ACCESS_TOKEN,
      whatsappCloudApiVersion: CLOUD_API_VERSION,
      whatsappCallId: callId,
      sdp: { type: sdpType, sdp },
      roomName,
      agents: [new RoomAgentDispatch({ agentName: AGENT_NAME })],
      // Do not block the webhook response on the agent joining; Meta wants a
      // fast 200 and LiveKit finishes the SDP exchange in the background.
      waitUntilAnswered: false,
    });

    console.log(`accepted WhatsApp call ${callId} in room ${roomName}`, res?.roomName ?? '');
    return { callId, ok: true, room: res?.roomName ?? roomName };
  } catch (err) {
    console.error(`failed to accept WhatsApp call ${callId}:`, err?.message ?? err);
    return { callId, ok: false, error: String(err?.message ?? err) };
  }
}

export const handler = async (event) => {
  const method = (event.httpMethod ?? 'GET').toUpperCase();

  // Meta webhook verification handshake.
  if (method === 'GET') {
    const q = event.queryStringParameters ?? {};
    if (q['hub.mode'] === 'subscribe' && q['hub.verify_token'] === VERIFY_TOKEN) {
      return {
        statusCode: 200,
        headers: { 'content-type': 'text/plain' },
        body: q['hub.challenge'] ?? '',
      };
    }
    return reply(403, { error: 'verification failed' });
  }

  if (method !== 'POST') return reply(405, { error: 'method not allowed' });

  const rawBody = event.isBase64Encoded
    ? Buffer.from(event.body ?? '', 'base64')
    : Buffer.from(event.body ?? '', 'utf8');

  const signature =
    event.headers['x-hub-signature-256'] ?? event.headers['X-Hub-Signature-256'];
  if (!isValidSignature(rawBody, signature)) {
    return reply(401, { error: 'invalid signature' });
  }

  let payload;
  try {
    payload = JSON.parse(rawBody.toString('utf8'));
  } catch {
    return reply(400, { error: 'invalid json' });
  }

  const results = [];
  for (const entry of payload.entry ?? []) {
    for (const change of entry.changes ?? []) {
      if (change.field !== 'calls') continue;
      for (const call of change.value?.calls ?? []) {
        // Meta sends "connect" (with SDP offer), "ringing", "terminate", ...
        if (call.event !== 'connect') continue;
        results.push(await acceptInboundCall(call));
      }
    }
  }

  // Always 200 quickly so Meta does not retry unnecessarily.
  return reply(200, { ok: true, handled: results.length, results });
};
