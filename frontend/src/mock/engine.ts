import type { ImuSample, PhraseRecord, ServerMessage } from '../types';
import { imuSampleAt } from './signal';
import { jitter, pick, sleep } from './util';

type Listener = (m: ServerMessage) => void;

const PHRASES = [
  { words: ['water'], sentence: 'Can I have some water, please?' },
  { words: ['water', 'help'], sentence: 'I need water, please help.' },
  { words: ['Jordan', 'help'], sentence: 'Jordan, I need your help.' },
  { words: ['Jordan', 'water'], sentence: 'Jordan, can I have some water?' },
  { words: ['yes'], sentence: 'Yes.' },
  { words: ['no'], sentence: 'No, thank you.' },
  { words: ['help'], sentence: 'Please help me.' },
];

/**
 * Fake backend event source: streams ~25 messages/s of IMU samples, and (when
 * enabled) plays out scripted word -> pause -> phrase -> speaking sequences.
 */
class MockEngine {
  private listeners = new Set<Listener>();
  private tickTimer?: number;
  private statusTimer?: number;
  private t0 = performance.now();
  private lastT = 0;
  private activity = 0;
  private burstUntil = 0;
  private burstIntensity = 1;
  private dropped = 0;
  private scriptEnabled = false;
  private scriptRun = 0;
  private nextHistoryId = 4;

  history: PhraseRecord[] = [
    { id: 3, words: ['Jordan', 'water'], sentence: 'Jordan, can I have some water?', at_ms: Date.now() - 3 * 60_000 },
    { id: 2, words: ['yes'], sentence: 'Yes.', at_ms: Date.now() - 11 * 60_000 },
    { id: 1, words: ['help'], sentence: 'Please help me.', at_ms: Date.now() - 42 * 60_000 },
  ];

  now() {
    return Math.round(performance.now() - this.t0);
  }

  subscribe(l: Listener) {
    this.listeners.add(l);
    if (this.listeners.size === 1) this.start();
    return () => {
      this.listeners.delete(l);
      if (this.listeners.size === 0) this.stop();
    };
  }

  /** Make the fake sensors move for a while (e.g. while a take is recording). */
  burst(ms: number, intensity = 1) {
    this.burstUntil = performance.now() + ms;
    this.burstIntensity = intensity;
  }

  setScriptEnabled(on: boolean) {
    if (on === this.scriptEnabled) return;
    this.scriptEnabled = on;
    this.scriptRun++;
    if (on) void this.runScript(this.scriptRun);
  }

  private emit(m: ServerMessage) {
    this.listeners.forEach((l) => l(m));
  }

  private start() {
    this.lastT = this.now();
    this.tickTimer = window.setInterval(() => this.tick(), 40);
    this.statusTimer = window.setInterval(() => {
      if (Math.random() < 0.03) this.dropped += 1;
      this.emit({
        type: 'status',
        connected: true,
        rate_hz: [100, Math.random() < 0.1 ? 99 : 100],
        dropped: this.dropped,
      });
    }, 1000);
  }

  private stop() {
    clearInterval(this.tickTimer);
    clearInterval(this.statusTimer);
  }

  private tick() {
    const target = performance.now() < this.burstUntil ? this.burstIntensity : 0;
    const nowT = this.now();
    const n = Math.min(Math.floor((nowT - this.lastT) / 10), 20);
    if (n <= 0) return;
    const out: ImuSample[] = [];
    for (let i = 1; i <= n; i++) {
      this.activity += (target - this.activity) * 0.08;
      const t = this.lastT + i * 10;
      out.push(imuSampleAt(t, 0, this.activity), imuSampleAt(t, 1, this.activity * 0.6));
    }
    this.lastT = n === 20 ? nowT : this.lastT + n * 10;
    this.emit({ type: 'samples', data: out });
  }

  private async runScript(id: number) {
    const alive = () => this.scriptRun === id && this.scriptEnabled;
    await sleep(1200);
    while (alive()) {
      const p = pick(PHRASES);
      for (const w of p.words) {
        if (!alive()) return;
        this.burst(700, 1);
        await sleep(850);
        if (!alive()) return;
        const confidence = Math.random() < 0.15 ? jitter(0.5, 0.69) : jitter(0.8, 0.99);
        this.emit({ type: 'word', word: w, confidence: Math.round(confidence * 100) / 100 });
        await sleep(500);
      }
      for (let i = 1; i <= 15; i++) {
        await sleep(100);
        if (!alive()) return;
        this.emit({ type: 'pause', progress: i / 15 });
      }
      this.emit({ type: 'phrase', words: p.words, sentence: p.sentence });
      this.history.unshift({ id: this.nextHistoryId++, words: p.words, sentence: p.sentence, at_ms: Date.now() });
      this.emit({ type: 'speaking', active: true });
      await sleep(2600);
      this.emit({ type: 'speaking', active: false });
      await sleep(jitter(2500, 4500));
    }
  }
}

export const mockEngine = new MockEngine();
