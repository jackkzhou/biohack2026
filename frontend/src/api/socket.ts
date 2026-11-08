import { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef } from 'react';
import { USE_MOCK, WS_URL } from '../config';
import { MockWebSocket } from '../mock/MockWebSocket';
import type { ImuSample, ServerMessage } from '../types';
import { api } from './client';

export type SamplesCb = (samples: ImuSample[]) => void;

export interface DetectedWord {
  id: number;
  word: string;
  confidence: number;
}

interface State {
  link: 'connecting' | 'open' | 'closed';
  /** true when IMU samples arrived within the last 2 s */
  streaming: boolean;
  sensors: { connected: boolean; rate_hz: number[]; dropped: number };
  currentWords: DetectedWord[];
  pauseProgress: number;
  phrase: { words: string[]; sentence: string } | null;
  /** increments each time a phrase completes (pages use it to refetch history) */
  phraseCount: number;
  /** a movement was detected but it was not a word (and no words are showing) */
  nothing: boolean;
  /** increments on every such event */
  nothingCount: number;
  /** what the model thought it heard in the last rejected detection (low confidence or the 'nothing' class) */
  guess: { label: string; confidence: number; at: number } | null;
  speaking: boolean;
  muted: boolean;
}

type Action =
  | { type: 'link'; link: State['link'] }
  | { type: 'streaming'; streaming: boolean }
  | { type: 'status'; connected: boolean; rate_hz: number[]; dropped: number }
  | { type: 'word'; word: DetectedWord }
  | { type: 'nothing'; label?: string; confidence?: number }
  | { type: 'pause'; progress: number }
  | { type: 'phrase'; words: string[]; sentence: string }
  | { type: 'speaking'; active: boolean }
  | { type: 'clear' }
  | { type: 'muted'; muted: boolean };

const initial: State = {
  link: 'connecting',
  streaming: false,
  sensors: { connected: false, rate_hz: [], dropped: 0 },
  currentWords: [],
  pauseProgress: 0,
  phrase: null,
  phraseCount: 0,
  nothing: false,
  nothingCount: 0,
  guess: null,
  speaking: false,
  muted: false,
};

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case 'link':
      return a.link === 'open'
        ? { ...s, link: 'open' }
        : { ...s, link: a.link, streaming: false, speaking: false, sensors: { ...s.sensors, connected: false } };
    case 'streaming':
      return s.streaming === a.streaming ? s : { ...s, streaming: a.streaming };
    case 'status':
      return { ...s, sensors: { connected: a.connected, rate_hz: a.rate_hz, dropped: a.dropped } };
    case 'word': {
      // A new word after a finished phrase starts a fresh phrase.
      const base = s.phrase ? [] : s.currentWords;
      return { ...s, currentWords: [...base, a.word], phrase: null, pauseProgress: 0, nothing: false };
    }
    case 'nothing':
      return {
        ...s,
        nothingCount: s.nothingCount + 1,
        nothing: s.currentWords.length === 0 && !s.phrase,
        guess: a.label !== undefined && a.confidence !== undefined ? { label: a.label, confidence: a.confidence, at: Date.now() } : s.guess,
      };
    case 'pause':
      return { ...s, pauseProgress: a.progress };
    case 'phrase':
      return { ...s, phrase: { words: a.words, sentence: a.sentence }, pauseProgress: 0, phraseCount: s.phraseCount + 1 };
    case 'speaking':
      return { ...s, speaking: a.active };
    case 'clear':
      return { ...s, currentWords: [], phrase: null, pauseProgress: 0, nothing: false };
    case 'muted':
      return { ...s, muted: a.muted };
  }
}

export interface LiveStream extends State {
  /** Subscribe to raw IMU sample batches without causing React renders. */
  subscribeSamples: (cb: SamplesCb) => () => void;
  clearPhrase: () => void;
  setMuted: (muted: boolean) => void;
}

export const LiveContext = createContext<LiveStream | null>(null);

export function useLiveStream(): LiveStream {
  const v = useContext(LiveContext);
  if (!v) throw new Error('useLiveStream must be used inside <LiveStreamProvider>');
  return v;
}

/** Owns the single WebSocket connection (real or mock). Used by <LiveStreamProvider>. */
export function useLiveSocket(): LiveStream {
  const [state, dispatch] = useReducer(reducer, initial);
  const listeners = useRef(new Set<SamplesCb>());
  const lastSampleAt = useRef(0);
  const wordId = useRef(0);
  const mutedRef = useRef(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: number | undefined;
    let closed = false;
    let attempt = 0;

    const speakMock = (sentence: string) => {
      if (mutedRef.current || !('speechSynthesis' in window)) return;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(sentence));
    };

    const handle = (m: ServerMessage) => {
      switch (m.type) {
        case 'samples':
          lastSampleAt.current = performance.now();
          listeners.current.forEach((cb) => cb(m.data));
          break;
        case 'word':
          dispatch({ type: 'word', word: { id: ++wordId.current, word: m.word, confidence: m.confidence } });
          break;
        case 'nothing':
          dispatch({ type: 'nothing', label: m.label, confidence: m.confidence });
          break;
        case 'pause':
          dispatch({ type: 'pause', progress: m.progress });
          break;
        case 'phrase':
          dispatch({ type: 'phrase', words: m.words, sentence: m.sentence });
          if (USE_MOCK) speakMock(m.sentence);
          break;
        case 'speaking':
          dispatch({ type: 'speaking', active: m.active });
          break;
        case 'audio':
          if (!mutedRef.current) {
            audioRef.current?.pause();
            audioRef.current = new Audio(m.url);
            audioRef.current.play().catch(() => {});
          }
          break;
        case 'status':
          dispatch({ type: 'status', connected: m.connected, rate_hz: m.rate_hz, dropped: m.dropped });
          break;
      }
    };

    const connect = () => {
      dispatch({ type: 'link', link: 'connecting' });
      const Socket = (USE_MOCK ? MockWebSocket : WebSocket) as unknown as typeof WebSocket;
      const sock = new Socket(WS_URL);
      ws = sock;
      sock.onopen = () => {
        attempt = 0;
        dispatch({ type: 'link', link: 'open' });
      };
      sock.onmessage = (ev) => {
        try {
          handle(JSON.parse(ev.data as string) as ServerMessage);
        } catch {
          /* ignore malformed frames */
        }
      };
      sock.onerror = () => sock.close();
      sock.onclose = () => {
        dispatch({ type: 'link', link: 'closed' });
        if (!closed) retry = window.setTimeout(connect, Math.min(1000 * 2 ** attempt++, 8000));
      };
    };

    connect();
    const watchdog = window.setInterval(() => {
      dispatch({ type: 'streaming', streaming: performance.now() - lastSampleAt.current < 2000 });
    }, 500);

    return () => {
      closed = true;
      clearTimeout(retry);
      clearInterval(watchdog);
      ws?.close();
    };
  }, []);

  const subscribeSamples = useCallback((cb: SamplesCb) => {
    listeners.current.add(cb);
    return () => {
      listeners.current.delete(cb);
    };
  }, []);

  const clearPhrase = useCallback(() => {
    dispatch({ type: 'clear' });
    api.clearPhrase().catch(() => {});
  }, []);

  const setMuted = useCallback((muted: boolean) => {
    mutedRef.current = muted;
    dispatch({ type: 'muted', muted });
    if (muted) {
      audioRef.current?.pause();
      if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    }
    api.setMuted(muted).catch(() => {});
  }, []);

  return useMemo(
    () => ({ ...state, subscribeSamples, clearPhrase, setMuted }),
    [state, subscribeSamples, clearPhrase, setMuted],
  );
}
