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
  | { type: 'nothing' }
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

export interface Api {
  postUtterance(audio: Blob, startedAtMs: number): Promise<UtteranceResult>;
  listClips(): Promise<ClipInfo[]>;
  getClip(id: number): Promise<Clip>;
  deleteClip(id: number): Promise<void>;
  tare(): Promise<void>;
  recordRestPose(session: number): Promise<void>;
  recordStart(label: string, session: number): Promise<void>;
  recordStop(): Promise<{ clip_id: number; label: string }>;
  getRecordCounts(): Promise<Record<string, number>>;
  getHistory(): Promise<PhraseRecord[]>;
  clearPhrase(): Promise<void>;
  setMuted(muted: boolean): Promise<void>;
  setPhrasingMode(mode: PhrasingMode): Promise<void>;
  getMetrics(): Promise<ModelMetrics>;
}

export const VOCAB = ['help', 'yes', 'no', 'Jordan', 'water', 'nothing'] as const;
export const SAMPLES_PER_WORD = 40;
