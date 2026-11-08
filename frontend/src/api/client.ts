import { API_URL, USE_MOCK } from '../config';
import { mockApi } from '../mock/mockApi';
import type { Api } from '../types';

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API_URL + path, init);
  if (!res.ok) throw new Error(`${init?.method ?? 'GET'} ${path} failed (${res.status})`);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
});

const realApi: Api = {
  postUtterance(audio, startedAtMs) {
    const form = new FormData();
    form.append('audio', audio, audio.type.includes('mp4') ? 'utterance.mp4' : 'utterance.webm');
    form.append('started_at_ms', String(startedAtMs));
    return http('/learn/utterance', { method: 'POST', body: form });
  },
  listClips: () => http('/clips'),
  getClip: (id) => http(`/clips/${id}`),
  deleteClip: (id) => http(`/clips/${id}`, { method: 'DELETE' }),
  recordRestPose: (session) => http('/record/rest-pose', json('POST', { session })),
  recordStart: (label, session) => http('/record/start', json('POST', { label, session })),
  recordStop: () => http('/record/stop', json('POST')),
  getRecordCounts: () => http('/record/counts'),
  getHistory: () => http('/history/phrases'),
  clearPhrase: () => http('/phrase/clear', json('POST')),
  setMuted: (muted) => http('/voice/mute', json('POST', { muted })),
  setPhrasingMode: (mode) => http('/phrasing/mode', json('POST', { mode })),
  getMetrics: () => http('/model/metrics'),
};

export const api: Api = USE_MOCK ? mockApi : realApi;
