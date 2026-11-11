// Conversation history for Mike (WhatsApp chatbot), kept in Netlify Blobs.
// Key: wa:<wa_id>  Value: { history: [...], updatedAt: epoch ms }

import { getStore, getDeployStore } from '@netlify/blobs';

// SITE_ID comes from the runtime; NETLIFY_BLOBS_TOKEN is a scoped PAT set in
// Netlify env vars (manual CLI deploys don't auto-inject Blobs context).
const STORE_NAME = 'conversations';
const SESSION_TTL_MS = 30 * 60 * 1000; // 30 min of silence = fresh conversation

function blobsStore() {
  const siteID = process.env.SITE_ID;
  const token = process.env.NETLIFY_BLOBS_TOKEN;
  if (siteID && token) {
    try {
      return getStore({ name: STORE_NAME, siteID, token });
    } catch (err) {
      console.error('getStore(token) failed, falling back to default store:', err?.message ?? err);
    }
  }
  try {
    return getDeployStore({ name: STORE_NAME });
  } catch (err) {
    console.error('getDeployStore unavailable:', err?.message ?? err);
    return getStore(STORE_NAME);
  }
}

function now() {
  return Date.now();
}

export async function loadHistory(waId) {
  try {
    const store = blobsStore();
    const raw = await store.get(`wa:${waId}`, { type: 'json' });
    if (!raw) return [];
    if (now() - Number(raw.updatedAt ?? 0) > SESSION_TTL_MS) return [];
    return Array.isArray(raw.history) ? raw.history : [];
  } catch (err) {
    console.error('loadHistory failed (starting fresh):', err?.message ?? err);
    return [];
  }
}

/** Drop the system prompt from stored history; keep the recent tail only. */
export function trimHistory(messages, max = 24) {
  const filtered = messages.filter((m) => m.role !== 'system');
  return filtered.slice(-max);
}

export async function saveHistory(waId, history) {
  try {
    const store = blobsStore();
    await store.setJSON(`wa:${waId}`, { history: trimHistory(history), updatedAt: now() });
  } catch (err) {
    console.error('saveHistory failed:', err?.message ?? err);
  }
}
