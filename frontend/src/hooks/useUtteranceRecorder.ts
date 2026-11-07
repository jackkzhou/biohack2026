import { useCallback, useEffect, useRef, useState } from 'react';

export type MicState = 'idle' | 'requesting' | 'listening' | 'denied' | 'unavailable';

const VOICE_RMS = 0.02;
const SILENCE_MS = 1000;
const MAX_LEADING_SILENCE_MS = 8000;

interface Internals {
  stream?: MediaStream;
  ctx?: AudioContext;
  analyser?: AnalyserNode;
  rec?: MediaRecorder;
  timer?: number;
  active: boolean;
  upload: boolean;
  voiced: boolean;
  lastVoice: number;
  segStart: number;
}

/**
 * Listens on the mic and cuts the audio into utterances: a segment is uploaded
 * once speech was heard and then ~1 s of silence follows.
 * `onUtterance` gets the audio and the epoch-ms time the recording started.
 */
export function useUtteranceRecorder(onUtterance: (audio: Blob, startedAtMs: number) => void) {
  const [state, setState] = useState<MicState>('idle');
  const [level, setLevel] = useState(0);
  const [hearing, setHearing] = useState(false);
  const r = useRef<Internals>({ active: false, upload: false, voiced: false, lastVoice: 0, segStart: 0 });
  const cb = useRef(onUtterance);
  cb.current = onUtterance;

  const cleanup = useCallback(() => {
    const s = r.current;
    clearInterval(s.timer);
    s.stream?.getTracks().forEach((t) => t.stop());
    void s.ctx?.close().catch(() => {});
    s.stream = s.ctx = s.analyser = s.rec = undefined;
    setState((cur) => (cur === 'listening' ? 'idle' : cur));
    setLevel(0);
    setHearing(false);
  }, []);

  const startSegment = useCallback(() => {
    const s = r.current;
    if (!s.stream || !s.active) return;
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'].find((m) => MediaRecorder.isTypeSupported(m));
    const rec = new MediaRecorder(s.stream, mime ? { mimeType: mime } : undefined);
    const chunks: Blob[] = [];
    const startedAt = Date.now();
    rec.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    rec.onstop = () => {
      const upload = s.upload;
      s.upload = false;
      if (upload && chunks.length) cb.current(new Blob(chunks, { type: rec.mimeType || 'audio/webm' }), startedAt);
      s.rec = undefined;
      if (s.active) startSegment();
      else cleanup();
    };
    rec.start();
    s.rec = rec;
    s.segStart = startedAt;
    s.voiced = false;
    s.lastVoice = 0;
  }, [cleanup]);

  const finish = useCallback((upload: boolean) => {
    const s = r.current;
    if (!s.rec || s.rec.state !== 'recording') return;
    s.upload = upload;
    s.rec.stop();
  }, []);

  const tick = useCallback(() => {
    const s = r.current;
    if (!s.analyser) return;
    const buf = new Uint8Array(s.analyser.fftSize);
    s.analyser.getByteTimeDomainData(buf);
    let sum = 0;
    for (const v of buf) sum += ((v - 128) / 128) ** 2;
    const rms = Math.sqrt(sum / buf.length);
    setLevel(Math.min(1, rms * 6));
    const now = Date.now();
    const voice = rms > VOICE_RMS;
    if (voice) {
      s.lastVoice = now;
      s.voiced = true;
    }
    setHearing(voice);
    if (s.voiced && now - s.lastVoice > SILENCE_MS) finish(true);
    else if (!s.voiced && now - s.segStart > MAX_LEADING_SILENCE_MS) finish(false);
  }, [finish]);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setState('unavailable');
      return;
    }
    setState('requesting');
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    } catch {
      setState('denied');
      return;
    }
    const s = r.current;
    const ctx = new AudioContext();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(analyser);
    s.stream = stream;
    s.ctx = ctx;
    s.analyser = analyser;
    s.active = true;
    startSegment();
    s.timer = window.setInterval(tick, 60);
    setState('listening');
  }, [startSegment, tick]);

  const stop = useCallback(() => {
    const s = r.current;
    s.active = false;
    if (s.rec && s.rec.state === 'recording') {
      s.upload = s.voiced;
      s.rec.stop(); // onstop -> cleanup
    } else {
      cleanup();
    }
  }, [cleanup]);

  useEffect(
    () => () => {
      const s = r.current;
      s.active = false;
      s.upload = false;
      if (s.rec && s.rec.state === 'recording') s.rec.stop();
      else cleanup();
    },
    [cleanup],
  );

  return { state, level, hearing, start, stop };
}
