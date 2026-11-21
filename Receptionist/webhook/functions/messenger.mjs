// Facebook Messenger webhook → Mike (the same AI brain as the WhatsApp bot).
//
// Inbound flow:
//   1. Patient messages the clinic's Facebook Page.
//   2. Meta POSTs a `page` webhook event.
//   3. Mike (functions/lib/agent.mjs) answers using Groq + Cal.com tools,
//      with per-user conversation memory (Netlify Blobs).
//   4. The reply is sent back via the Graph API (/me/messages).
//
// Audio attachments are transcribed with Deepgram before they reach Mike.
//
// Env vars (set in Netlify, never commit them):
//   MESSENGER_VERIFY_TOKEN       - must match the Meta webhook "Verify token"
//   MESSENGER_PAGE_ACCESS_TOKEN  - system-user token with pages_messaging
//   MESSENGER_APP_SECRET         - Meta app secret, verifies X-Hub-Signature-256
//   MESSENGER_PAGE_ID            - the Page ID (optional, informational)
//   CAL_API_KEY, CAL_USERNAME    - Cal.com booking credentials
//   GROQ_API_KEY                 - the AI brain
//   DEEPGRAM_API_KEY             - audio transcription (optional)

import crypto from 'node:crypto';
import { loadHistory, saveHistory } from './lib/conversation.mjs';
import { runAgent } from './lib/agent.mjs';

const VERIFY_TOKEN = process.env.MESSENGER_VERIFY_TOKEN ?? '';
const PAGE_TOKEN = process.env.MESSENGER_PAGE_ACCESS_TOKEN ?? '';
const APP_SECRET = process.env.MESSENGER_APP_SECRET ?? '';
const PAGE_ID = process.env.MESSENGER_PAGE_ID ?? process.env.messenger_page_id ?? '';
const CLOUD_API_VERSION = process.env.WHATSAPP_CLOUD_API_VERSION ?? 'v21.0';
const CAL_API_KEY = process.env.CAL_API_KEY ?? '';
const CAL_USERNAME = process.env.CAL_USERNAME ?? '';
const GROQ_API_KEY = process.env.GROQ_API_KEY ?? '';
const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY ?? '';

const GRAPH_BASE = `https://graph.facebook.com/${CLOUD_API_VERSION}`;

function reply(statusCode, body, headers = {}) {
  return {
    statusCode,
    headers: { 'content-type': 'application/json', ...headers },
    body: typeof body === 'string' ? body : JSON.stringify(body),
  };
}

/** Constant-time compare of Meta's sha256 signature against the raw body. */
function isValidSignature(rawBody, header) {
  if (!APP_SECRET) {
    console.warn('MESSENGER_APP_SECRET is not set - skipping signature verification');
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

/** Send a plain text Messenger message. */
async function sendText(psid, body) {
  try {
    const res = await fetch(`${GRAPH_BASE}/me/messages`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${PAGE_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        recipient: { id: psid },
        messaging_type: 'RESPONSE',
        message: { text: body },
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

/** Download a Messenger attachment (needs the page token) and transcribe it. */
async function transcribeAudio(url, mimeType) {
  const dl = await fetch(url, { headers: { Authorization: `Bearer ${PAGE_TOKEN}` } });
  if (!dl.ok) throw new Error(`attachment download failed: ${dl.status}`);
  const buf = Buffer.from(await dl.arrayBuffer());

  const stt = await fetch('https://api.deepgram.com/v1/listen?model=nova-3&language=multi&smart_format=true', {
    method: 'POST',
    headers: {
      Authorization: `Token ${DEEPGRAM_API_KEY}`,
      'Content-Type': mimeType || 'audio/mpeg',
    },
    body: buf,
  });
  if (!stt.ok) throw new Error(`deepgram ${stt.status}: ${(await stt.text()).slice(0, 200)}`);
  const data = await stt.json();
  const transcript = data?.results?.channels?.[0]?.alternatives?.[0]?.transcript?.trim();
  if (!transcript) throw new Error('no transcript');
  return transcript;
}

/**
 * Handle one Messenger message event.
 * Text goes straight to Mike; audio is transcribed first; anything else gets a
 * short acknowledgment so the patient is never left hanging.
 */
async function handleMessage(message) {
  const psid = message?.sender?.id;
  if (!psid) return null;

  let text = (message?.message?.text ?? '').trim();
  if (!text && Array.isArray(message?.message?.attachments) && message.message.attachments.length > 0) {
    const att = message.message.attachments[0];
    if (att?.type === 'audio') {
      try {
        text = await transcribeAudio(att.payload?.url, att.payload?.mime_type ?? att.type);
      } catch (err) {
        console.error('audio transcription failed:', err?.message ?? err);
        await sendText(psid, 'I could not understand that voice note. Please type your question, or call the clinic.');
        return { psid, ok: false, mode: 'audio-failed' };
      }
    } else {
      await sendText(psid, 'I can read text and voice notes. For anything else, please call the clinic.');
      return { psid, ok: true, mode: 'unsupported-attachment' };
    }
  }
  if (!text) return null;

  const cal = { apiKey: CAL_API_KEY, username: CAL_USERNAME, groqApiKey: GROQ_API_KEY };
  // Separate memory namespace from WhatsApp so a person who uses both keeps
  // two independent conversations.
  const history = await loadHistory(`wa:${psid}`);
  let botReply = 'Sorry, I am having a little trouble right now — please try again in a minute.';
  try {
    const result = await runAgent({ waId: psid, text, history, cal });
    botReply = result.reply;
    await saveHistory(`wa:${psid}`, result.messages);
  } catch (err) {
    console.error('runAgent failed:', err?.message ?? err);
  }

  const ok = await sendText(psid, botReply);
  console.log(`mike replied on Messenger to ${psid} ok=${ok} preview=${botReply.slice(0, 80).replace(/\n/g, ' | ')}`);
  return { psid, ok, mode: 'llm' };
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

  if (payload?.object !== 'page') {
    return reply(200, { ok: true, skipped: 'not a page event' });
  }

  const results = [];
  for (const entry of payload.entry ?? []) {
    for (const eventMsg of entry.messaging ?? []) {
      results.push(await handleMessage(eventMsg));
    }
  }

  return reply(200, { ok: true, handled: results.filter(Boolean).length, results });
};
