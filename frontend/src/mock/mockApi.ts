import type { Api, Clip, ModelMetrics, UtteranceResult } from '../types';
import { mockEngine } from './engine';
import { synthWindow } from './signal';
import { jitter, pick, sleep } from './util';

const TRANSCRIPTS = ['water please', 'help me', 'yes', 'no thank you', 'call Jordan', 'I need water', 'yes please help'];

interface StoredClip extends Clip {
  source: 'record' | 'learn';
}

const clips = new Map<number, StoredClip>();
const counts: Record<string, number> = { help: 7, yes: 5, no: 6, Jordan: 4, water: 8, nothing: 3 };
let nextClipId = 100;
let pending: { label: string } | null = null;

const delay = () => sleep(jitter(120, 350));

const METRICS: ModelMetrics = {
  model: 'MiniRocket + Ridge',
  heldout_session: 3,
  accuracy: 0.87,
  labels: ['help', 'yes', 'no', 'Jordan', 'water', 'nothing'],
  matrix: [
    [36, 1, 2, 0, 1, 0],
    [1, 35, 3, 0, 0, 1],
    [2, 4, 34, 0, 0, 0],
    [0, 0, 0, 37, 2, 1],
    [1, 0, 0, 3, 33, 3],
    [0, 1, 1, 0, 4, 34],
  ],
  comparison: [
    { model: 'DTW + kNN', accuracy: 0.71 },
    { model: 'Random Forest', accuracy: 0.82 },
    { model: 'MiniRocket + Ridge', accuracy: 0.87 },
    { model: '1D CNN', accuracy: 0.85 },
  ],
};

let mockCalibrated = false;
let mockCounts: Record<string, number> = {};

export const mockApi: Api = {
  async postUtterance(_audio, startedAtMs): Promise<UtteranceResult> {
    await sleep(jitter(500, 1100));
    const transcript = pick(TRANSCRIPTS);
    let cursor = startedAtMs + 120;
    const words = transcript.split(' ').map((word) => {
      const start_ms = cursor;
      const end_ms = start_ms + Math.round(jitter(380, 620));
      cursor = end_ms + 90;
      const id = nextClipId++;
      clips.set(id, { id, word, source: 'learn', samples: synthWindow(0, end_ms - start_ms, 1) });
      return { word, start_ms, end_ms, clip_id: id };
    });
    return { transcript, words };
  },

  async listClips() {
    await delay();
    return [...clips.values()].map((c) => ({ id: c.id, label: c.word, session: 1, source: c.source, n_samples: c.samples.length, created_at_ms: Date.now() }));
  },

  async getClip(id) {
    await delay();
    const c = clips.get(id);
    if (!c) throw new Error(`clip ${id} not found`);
    return { id: c.id, word: c.word, samples: c.samples };
  },

  async deleteClip(id) {
    await delay();
    const c = clips.get(id);
    if (c?.source === 'record') counts[c.word] = Math.max(0, (counts[c.word] ?? 0) - 1);
    clips.delete(id);
  },

  async calibrateInfo() {
    return { words: ['help', 'Jordan', 'no', 'nothing'], allowed: ['help', 'yes', 'no', 'Jordan', 'water', 'nothing'], counts: { ...mockCounts }, takes_per_word: 3, seconds: 3, tared: true, model_loaded: true, calibrated: mockCalibrated, session: Object.keys(mockCounts).length ? 7 : null, can_undo: mockCalibrated };
  },

  async calibrateBegin({ words, takes, keepSession }) {
    if (!keepSession) mockCounts = {};
    const order = words.flatMap((w) => Array.from({ length: takes }, () => w)).sort(() => Math.random() - 0.5);
    for (const w of order) mockCounts[w] = (mockCounts[w] ?? 0) + 1;
    return { session: 7, order };
  },

  async calibrateTrain() {
    await sleep(2000);
    mockCalibrated = true;
    return { ok: true, new_takes: { ...mockCounts } };
  },

  async calibrateUndo() {
    mockCalibrated = false;
  },

  async calibrateReset() {
    mockCalibrated = false;
    mockCounts = {};
  },

  async tare() {
    await sleep(1500);
  },

  async recordRestPose() {
    await sleep(2000);
  },

  async recordStart(label) {
    pending = { label };
    mockEngine.burst(2000, label === 'nothing' ? 0.35 : 1);
  },

  async recordStop() {
    if (!pending) throw new Error('no recording in progress');
    const { label } = pending;
    pending = null;
    await delay();
    const id = nextClipId++;
    clips.set(id, { id, word: label, source: 'record', samples: synthWindow(0, 2000, label === 'nothing' ? 0.35 : 1) });
    counts[label] = (counts[label] ?? 0) + 1;
    return { clip_id: id, label };
  },

  async getRecordCounts() {
    await delay();
    return { ...counts };
  },

  async getHistory() {
    await delay();
    return [...mockEngine.history];
  },

  async clearPhrase() {},
  async setMuted() {},
  async setPhrasingMode() {},

  async getMetrics() {
    await sleep(jitter(250, 500));
    return METRICS;
  },
};
