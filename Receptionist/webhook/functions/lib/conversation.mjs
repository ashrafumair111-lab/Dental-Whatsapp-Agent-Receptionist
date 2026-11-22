// Conversation history for Mike (WhatsApp chatbot), kept in Netlify Blobs.
// Key: wa:<wa_id>  Value: { history: [...], updatedAt: epoch ms }

import { getStore, getDeployStore } from '@netlify/blobs';

// Conversation history, kept in Netlify Blobs.
// BLOBS_SITE_ID/SITE_ID + NETLIFY_BLOBS_TOKEN come from deploy env (manual CLI
// deploys don't auto-inject Blobs context).
const STORE_NAME = 'conversations';
const SESSION_TTL_MS = 30 * 60 * 1000; // 30 min of silence = fresh conversation
// Strong consistency: without it reads can hit a replica that hasn't seen our
// latest write yet — history then looks lost right after saveHistory().
const CONSISTENCY = 'strong';

if (!(process.env.BLOBS_SITE_ID || process.env.SITE_ID) || !process.env.NETLIFY_BLOBS_TOKEN) {
  console.warn('Blobs not configured (memory off: history resets every turn)');
}

function blobsStore() {
  // 1) Manual-token store (works for reads+writes when the token allows it).
  const siteID = process.env.BLOBS_SITE_ID || process.env.SITE_ID;
  const token = process.env.NETLIFY_BLOBS_TOKEN;
  if (siteID && token) {
    try {
      return getStore({ name: STORE_NAME, siteID, token, consistency: CONSISTENCY });
    } catch (err) {
      console.error('getStore(token) failed:', err?.message ?? err);
    }
  }
  // 2) Platform deploy store.
  try {
    return getDeployStore({ name: STORE_NAME });
  } catch (err) {
    console.error('getDeployStore unavailable:', err?.message ?? err);
  }
  // 3) Default platform store.
  return getStore(STORE_NAME);
}

function now() {
  return Date.now();
}

export async function loadHistory(waId) {
  try {
    const store = blobsStore();
    const raw = await store.get(`wa:${waId}`, { type: 'json', consistency: CONSISTENCY });
    if (!raw) {
      console.log(`loadHistory(${waId}): none yet`);
      return [];
    }
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

/**
 * Best-effort replay guard: Meta re-delivers a webhook when a run is slow.
 * Returns true if this message id was already handled recently.
 */
export async function seenRecently(id, ttlMs = 10 * 60 * 1000) {
  try {
    const store = blobsStore();
    const key = `seen:${id}`;
    const prev = await store.get(key, { type: 'json', consistency: CONSISTENCY });
    if (prev && now() - Number(prev.at ?? 0) < ttlMs) return true;
    await store.setJSON(key, { at: now() }, { consistency: CONSISTENCY });
    return false;
  } catch (err) {
    console.error('seenRecently failed (allowing message):', err?.message ?? err);
    return false;
  }
}

export async function saveHistory(waId, history) {
  try {
    const store = blobsStore();
    await store.setJSON(`wa:${waId}`, { history: trimHistory(history), updatedAt: now() }, { consistency: CONSISTENCY });
    console.log(`saveHistory(${waId}): saved ${trimHistory(history).length} msgs`);
  } catch (err) {
    console.error('saveHistory failed:', err?.message ?? err);
  }
}
