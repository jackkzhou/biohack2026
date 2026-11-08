export interface ImuSample {
  /** Device timestamp in milliseconds (only differences matter). */
  t: number;
  /** 0 = chin, 1 = head */
  sensor: number;
  /** gyro, rad/s */
  g: [number, number, number];
  /** accelerometer, m/s^2 */
  a: [number, number, number];
}

export type ServerMessage =
  | { type: 'samples'; data: ImuSample[] }
  | { type: 'word'; word: string; confidence: number }
  | { type: 'nothing'; label?: string; confidence?: number }
  | { type: 'pause'; progress: number }
  | { type: 'phrase'; words: string[]; sentence: string }
  | { type: 'speaking'; active: boolean }
  | { type: 'audio'; url: string }
  | { type: 'status'; connected: boolean; rate_hz: number[]; dropped: number };

export interface UtteranceWord {
  word: string;
  start_ms: number;
  end_ms: number;
  clip_id: number;
}
export interface UtteranceResult {
  transcript: string;
  words: UtteranceWord[];
}

export interface Clip {
  id: number;
  word: string;
  samples: ImuSample[];
}

export interface ClipInfo {
  id: number;
  label: string;
  session: number | null;
  source: string;
  n_samples: number | null;
  created_at_ms: number;
}

export interface SessionInfo {
  session: number;
  total: number;
  counts: Record<string, number>;
  /** sessions recorded without a tare cannot be used: their chin axes do not match live data */
  tared: boolean;
  current: boolean;
}

export interface CalibrateInfo {
  default_weight: number;
  sessions: SessionInfo[];
  words: string[];
  allowed: string[];
  counts: Record<string, number>;
  takes_per_word: number;
  seconds: number;
  tared: boolean;
  model_loaded: boolean;
  calibrated: boolean;
  session: number | null;
  can_undo: boolean;
}

export interface PhraseRecord {
  id: number;
  words: string[];
  sentence: string;
  at_ms: number;
}

export interface ModelMetrics {
  model: string;
  heldout_session: number;
  accuracy: number;
  labels: string[];
  /** matrix[true][predicted] */
  matrix: number[][];
  comparison: { model: string; accuracy: number }[];
}

export type PhrasingMode = 'lookup' | 'llm';

export interface Voice {
  voice_id: string;
  name: string;
}

export interface BaselineStatus {
  ready: boolean;
  days_collected: number;
  days_required: number;
  days_remaining: number;
  window_start: string | null;
  window_end: string | null;
  message: string;
}

export type RiskScore = 'low' | 'moderate' | 'elevated' | 'high';
export type MarkerSeverity = 'low' | 'moderate' | 'high';

export interface ProgressionMarker {
  name: string;
  direction: 'increased' | 'decreased' | 'unchanged';
  severity: MarkerSeverity;
  evidence: string;
}

export interface DeteriorationPrediction {
  risk_score: RiskScore;
  deterioration_index: number;
  confidence: number;
  markers: ProgressionMarker[];
  clinical_notes: string[];
  summary: string;
}

export interface PredictionResult {
  target_day: string;
  window_start: string;
  window_end: string;
  response: DeteriorationPrediction;
  cached: boolean;
}

export interface DailyFeatures {
  n_clips: number;
  n_detections: number;
  n_accepted_detections: number;
  velocity_variance: number | null;
  tremor_frequency_hz: number | null;
  jaw_asymmetry_index: number | null;
  mean_clip_duration_s: number | null;
  speech_cadence_wpm: number | null;
  detection_confidence_mean: number | null;
  rejection_rate: number | null;
  rest_activity_rms: number | null;
}

export interface DailyFeatureRow {
  day: string;
  features: DailyFeatures;
}

export interface Api {
  postUtterance(audio: Blob, startedAtMs: number): Promise<UtteranceResult>;
  listClips(): Promise<ClipInfo[]>;
  getClip(id: number): Promise<Clip>;
  deleteClip(id: number): Promise<void>;
  tare(): Promise<void>;
  calibrateInfo(): Promise<CalibrateInfo>;
  calibrateBegin(opts: { words: string[]; takes: number; keepSession: boolean }): Promise<{ session: number; order: string[] }>;
  calibrateTrain(session: number | null, weights?: Record<number, number>): Promise<{ ok: boolean; new_takes: Record<string, number>; labels?: string[] }>;
  calibrateUndo(): Promise<void>;
  calibrateReset(): Promise<void>;
  recordRestPose(session: number): Promise<void>;
  recordStart(label: string, session: number): Promise<void>;
  recordStop(): Promise<{ clip_id: number; label: string }>;
  getRecordCounts(): Promise<Record<string, number>>;
  getHistory(): Promise<PhraseRecord[]>;
  clearPhrase(): Promise<void>;
  setMuted(muted: boolean): Promise<void>;
  setPhrasingMode(mode: PhrasingMode): Promise<void>;
  listVoices(): Promise<Voice[]>;
  selectVoice(voiceId: string): Promise<void>;
  getMetrics(): Promise<ModelMetrics>;
  getProgressionStatus(): Promise<BaselineStatus>;
  getProgressionWindow(): Promise<DailyFeatureRow[]>;
  getProgressionPrediction(force?: boolean): Promise<PredictionResult>;
}

export const VOCAB = ['help', 'yes', 'no', 'Jordan', 'water', 'nothing'] as const;
export const SAMPLES_PER_WORD = 40;
