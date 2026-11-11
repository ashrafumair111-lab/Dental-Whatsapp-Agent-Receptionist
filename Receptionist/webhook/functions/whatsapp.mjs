// Meta WhatsApp Cloud API webhook → LiveKit (voice) + Mike (text chatbot).
//
// Inbound messages flow:
//   1. Patient messages the appointment line.
//   2. Meta POSTs a `messages` webhook (within the 24-hour service window).
//   3. Mike (state machine) asks for the details and books a slot in Cal.com.
//   4. The reply is sent back via Graph API.
//
// Inbound voice calls flow:
//   1. Caller dials the business number.
//   2. Meta POSTs a `calls` webhook with an SDP offer.
//   3. We call the LiveKit Connector API (AcceptWhatsAppCall), which joins the
//      media and dispatches the live voice agent to the room.
//
// Env vars (set in Netlify, never commit them):
//   WHATSAPP_VERIFY_TOKEN     - any string; must match the Meta webhook "Verify token"
//   WHATSAPP_APP_SECRET       - Meta app secret, used to verify webhook signatures
//   WHATSAPP_PHONE_NUMBER_ID  - the Cloud API phone number ID
//   WHATSAPP_ACCESS_TOKEN     - permanent system-user token (whatsapp_business_messaging)
//   WHATSAPP_CLOUD_API_VERSION- default 23.0
//   LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET - LiveKit Cloud project
//   LIVEKIT_AGENT_NAME        - voice agent name (default "MetaByte Solutions Receptionist")
//   CAL_API_KEY, CAL_USERNAME - Cal.com booking credentials
//   COMPANY_SERVICES          - optional, ";"-separated list of treatments

import crypto from 'node:crypto';
import { LiveKitAPI, RoomAgentDispatch } from 'livekit-server-sdk';
import { loadHistory, saveHistory } from './lib/conversation.mjs';
import { runAgent } from './lib/agent.mjs';

const VERIFY_TOKEN = process.env.WHATSAPP_VERIFY_TOKEN ?? '';
const APP_SECRET = process.env.WHATSAPP_APP_SECRET ?? '';
const PHONE_NUMBER_ID = process.env.WHATSAPP_PHONE_NUMBER_ID ?? '';
const ACCESS_TOKEN = process.env.WHATSAPP_ACCESS_TOKEN ?? '';
const CLOUD_API_VERSION = process.env.WHATSAPP_CLOUD_API_VERSION ?? '23.0';
const AGENT_NAME = process.env.LIVEKIT_AGENT_NAME ?? 'MetaByte Solutions Receptionist';
const CAL_API_KEY = process.env.CAL_API_KEY ?? '';
const CAL_USERNAME = process.env.CAL_USERNAME ?? '';
const GROQ_API_KEY = process.env.GROQ_API_KEY ?? '';

const GRAPH_BASE = `https://graph.facebook.com/v${CLOUD_API_VERSION}`;

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

/** Send a plain text WhatsApp message inside the 24-hour service window. */
async function sendText(to, body) {
  try {
    const res = await fetch(`${GRAPH_BASE}/${PHONE_NUMBER_ID}/messages`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${ACCESS_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messaging_product: 'whatsapp',
        recipient_type: 'individual',
        to,
        type: 'text',
        text: { body },
      }),
    });
    if (!res.ok) {
      console.error('sendText rejected:', res.status, await res.text());
      return false;
    }
    return true;
  } catch (err) {
    console.error('sendText failed:', err?.message ?? err);
    return false;
  }
}

/** Mark the inbound message as read (single blue tick) — nice touch for the demo. */
async function markRead(messageId) {
  try {
    await fetch(`${GRAPH_BASE}/${PHONE_NUMBER_ID}/messages`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${ACCESS_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messaging_product: 'whatsapp',
        status: 'read',
        message_id: messageId,
      }),
    });
  } catch (err) {
    console.warn('markRead failed:', err?.message ?? err);
  }
}

/** Handle a `messages` webhook change — Mike answers the patient. */
async function handleInboundMessage(message) {
  const waId = message?.from;
  const text = message?.text?.body ?? '';
  const type = message?.type;
  if (!waId || type !== 'text' || !text.trim()) return null;

  const messageId = message?.id;
  const cal = { apiKey: CAL_API_KEY, username: CAL_USERNAME, groqApiKey: GROQ_API_KEY };

  const history = await loadHistory(waId);
  let reply = 'Sorry, I am having a little trouble right now — please try again in a minute.';
  try {
    const result = await runAgent({ waId, text, history, cal });
    reply = result.reply;
    await saveHistory(waId, result.messages);
  } catch (err) {
    console.error('runAgent failed:', err?.message ?? err);
  }
  if (messageId) await markRead(messageId);

  const ok = await sendText(waId, reply);
  console.log(`mike replied to ${waId} ok=${ok} preview=${reply.slice(0, 80).replace(/\n/g, ' | ')}`);
  return { waId, ok, mode: 'llm' };
}

/** Handle a `calls` webhook change — hand the call to LiveKit. */
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
  // DEBUG: temporary - show what env the runtime actually injects
  console.log('DEBUG_ENV', JSON.stringify({
    siteId: process.env.SITE_ID,
    blobsTokenPresent: Boolean(process.env.NETLIFY_BLOBS_TOKEN),
    keys: Object.keys(process.env).filter((k) => /SITE|TOKEN|NETLIFY|BLOB/i.test(k)),
  }));
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
      const value = change.value ?? {};
      if (change.field === 'messages') {
        for (const message of value.messages ?? []) {
          results.push(await handleInboundMessage(message));
        }
      } else if (change.field === 'calls') {
        for (const call of value?.calls ?? []) {
          // Meta sends "connect" (with SDP offer), "ringing", "terminate", ...
          if (call.event !== 'connect') continue;
          results.push(await acceptInboundCall(call));
        }
      }
    }
  }

  // Always 200 quickly so Meta does not retry unnecessarily.
  return reply(200, { ok: true, handled: results.length, results });
};
