import type { Api, BaselineStatus, Clip, DailyFeatureRow, DeteriorationPrediction, ModelMetrics, PredictionResult, UtteranceResult, Voice } from '../types';
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

const MOCK_VOICES: Voice[] = [
  { voice_id: 'mock-voice-1', name: 'Jordan (cloned)' },
  { voice_id: 'mock-voice-2', name: 'Rachel' },
  { voice_id: 'mock-voice-3', name: 'Adam' },
];

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

const round = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d;

/** 14 days of mildly worsening features, so the demo shows the full prediction UI. */
const MOCK_WINDOW: DailyFeatureRow[] = Array.from({ length: 14 }, (_, i) => {
  const day = new Date('2026-10-25T00:00:00Z');
  day.setUTCDate(day.getUTCDate() + i);
  return {
    day: day.toISOString().slice(0, 10),
    features: {
      n_clips: 8,
      n_detections: 12,
      n_accepted_detections: 10 - Math.floor(i / 4),
      velocity_variance: round(0.015 - 0.0004 * i, 5),
      tremor_frequency_hz: round(5.0 + 0.12 * i, 2),
      jaw_asymmetry_index: round(0.18 + 0.01 * i, 3),
      mean_clip_duration_s: round(1.2 + 0.015 * i, 3),
      speech_cadence_wpm: round(42 - 0.6 * i, 1),
      detection_confidence_mean: round(0.88 - 0.006 * i, 3),
      rejection_rate: round(0.08 + 0.004 * i, 3),
      rest_activity_rms: round(0.02 + 0.0015 * i, 4),
    },
  };
});

const MOCK_PREDICTION: DeteriorationPrediction = {
  risk_score: 'moderate',
  deterioration_index: 0.34,
  confidence: 0.71,
  markers: [
    {
      name: 'tremor_frequency_hz',
      direction: 'increased',
      severity: 'moderate',
      evidence: 'Rose from 5.0 Hz to 6.6 Hz over the 14-day window, a steady upward trend.',
    },
    {
      name: 'jaw_asymmetry_index',
      direction: 'increased',
      severity: 'low',
      evidence: 'Edged up from 0.18 to 0.31, a mild but consistent increase.',
    },
    {
      name: 'speech_cadence_wpm',
      direction: 'decreased',
      severity: 'moderate',
      evidence: 'Dropped from 42 to 34.2 words/min, suggesting slower mouthing of words.',
    },
  ],
  clinical_notes: [
    'Consider a check-in with the care team to review the tremor-frequency trend.',
    'No action needed for jaw asymmetry yet; continue monitoring daily.',
    'If cadence keeps dropping, a short rest between words may help accuracy.',
  ],
  summary:
    'Over the last 14 days, tremor frequency and jaw asymmetry both drifted upward while speech cadence slowed — a mild but consistent pattern worth a check-in rather than urgent concern.',
};

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
    return { default_weight: 5, sessions: [{ session: 4, total: 91, counts: { help: 15, no: 16 }, tared: true, current: false }, { session: 7, total: 30, counts: { help: 5, no: 5 }, tared: true, current: Object.keys(mockCounts).length > 0 }], words: ['help', 'Jordan', 'no', 'nothing'], allowed: ['help', 'yes', 'no', 'Jordan', 'water', 'nothing'], counts: { ...mockCounts }, takes_per_word: 3, seconds: 3, tared: true, model_loaded: true, calibrated: mockCalibrated, session: Object.keys(mockCounts).length ? 7 : null, can_undo: mockCalibrated };
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

  async listVoices() {
    await delay();
    return MOCK_VOICES;
  },
  async selectVoice() {},

  async getMetrics() {
    await sleep(jitter(250, 500));
    return METRICS;
  },

  async getProgressionStatus(): Promise<BaselineStatus> {
    await delay();
    return {
      ready: true,
      days_collected: 14,
      days_required: 14,
      days_remaining: 0,
      window_start: MOCK_WINDOW[0].day,
      window_end: MOCK_WINDOW[13].day,
      message: 'baseline complete; ready to predict',
    };
  },

  async getProgressionWindow() {
    await delay();
    return MOCK_WINDOW;
  },

  async getProgressionPrediction(force): Promise<PredictionResult> {
    await sleep(jitter(400, 900));
    return {
      target_day: '2026-11-07',
      window_start: MOCK_WINDOW[0].day,
      window_end: MOCK_WINDOW[13].day,
      response: MOCK_PREDICTION,
      cached: !force,
    };
  },
};
