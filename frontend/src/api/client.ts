import { API_URL, USE_MOCK } from '../config';
import { mockApi } from '../mock/mockApi';
import type { Api } from '../types';

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API_URL + path, init);
  if (!res.ok) {
    const text = await res.text();
    let detail: string | undefined;
    try {
      detail = JSON.parse(text)?.detail;
    } catch {
      /* body wasn't JSON */
    }
    throw new Error(detail || `${init?.method ?? 'GET'} ${path} failed (${res.status})`);
  }
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
  calibrateInfo: () => http('/calibrate/info'),
  calibrateBegin: ({ words, takes, keepSession }) => http('/calibrate/begin', json('POST', { words, takes, keep_session: keepSession })),
  calibrateTrain: (session, weights) => http('/calibrate/train', json('POST', { session, weights })),
  calibrateReset: () => http('/calibrate/reset', json('POST')),
  calibrateUndo: () => http('/calibrate/undo', json('POST')),
  tare: () => http('/record/tare', json('POST')),
  recordRestPose: (session) => http('/record/rest-pose', json('POST', { session })),
  recordStart: (label, session) => http('/record/start', json('POST', { label, session })),
  recordStop: () => http('/record/stop', json('POST')),
  getRecordCounts: () => http('/record/counts'),
  getHistory: () => http('/history/phrases'),
  clearPhrase: () => http('/phrase/clear', json('POST')),
  setMuted: (muted) => http('/voice/mute', json('POST', { muted })),
  setPhrasingMode: (mode) => http('/phrasing/mode', json('POST', { mode })),
  listVoices: () => http('/voice/list'),
  selectVoice: (voiceId) => http('/voice/select', json('POST', { voice_id: voiceId })),
  getMetrics: () => http('/model/metrics'),
  getProgressionStatus: () => http('/progression/status'),
  getProgressionWindow: () => http('/progression/window'),
  getProgressionPrediction: (force) => http(`/progression/predict${force ? '?force=true' : ''}`),
};

export const api: Api = USE_MOCK ? mockApi : realApi;
