import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import { useLiveStream } from '../api/socket';
import LiveImuChart from '../components/LiveImuChart';
import PhraseBanner from '../components/PhraseBanner';
import WordChip from '../components/WordChip';
import { useMockScript } from '../mock/useMockScript';
import type { PhraseRecord, PhrasingMode } from '../types';

const PAUSE_SECONDS = 1.5;

function PauseRing({ progress, active }: { progress: number; active: boolean }) {
  const r = 44;
  const circ = 2 * Math.PI * r;
  return (
    <div className="flex flex-col items-center gap-2" role="progressbar" aria-label="Pause timer" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)}>
      <svg width="120" height="120" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r={r} fill="none" strokeWidth="10" style={{ stroke: 'rgb(var(--border))' }} />
        <circle
          cx="60"
          cy="60"
          r={r}
          fill="none"
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={circ * (1 - progress)}
          transform="rotate(-90 60 60)"
          style={{ stroke: 'rgb(var(--accent))', transition: 'stroke-dashoffset 0.1s linear' }}
        />
        <text x="60" y="68" textAnchor="middle" fontSize="24" fontWeight="700" style={{ fill: active ? 'rgb(var(--ink))' : 'rgb(var(--muted))' }}>
          {active ? `${(progress * PAUSE_SECONDS).toFixed(1)}s` : '–'}
        </text>
      </svg>
      <span className="text-sm font-medium text-muted">Phrase ends after {PAUSE_SECONDS} s of stillness</span>
    </div>
  );
}

function History({ phraseCount }: { phraseCount: number }) {
  const [items, setItems] = useState<PhraseRecord[] | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    let live = true;
    api.getHistory().then((h) => live && (setItems(h), setError(false))).catch(() => live && setError(true));
    return () => {
      live = false;
    };
  }, [phraseCount]);

  return (
    <section className="card" aria-labelledby="hist-h">
      <h2 id="hist-h" className="h-section mb-3">
        Recent phrases
      </h2>
      {error ? (
        <p className="text-danger">Couldn’t load history from the backend.</p>
      ) : items === null ? (
        <p className="text-muted">Loading…</p>
      ) : items.length === 0 ? (
        <p className="rounded-xl border border-dashed border-border p-6 text-center text-muted">No phrases yet. Silently mouth a word to get started.</p>
      ) : (
        <ul className="divide-y divide-border">
          {items.slice(0, 8).map((p) => (
            <li key={p.id} className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 py-3">
              <span className="text-xl font-semibold">“{p.sentence}”</span>
              <span className="text-sm text-muted">
                {p.words.join(' + ')} · {new Date(p.at_ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default function SilentSpeech() {
  useMockScript();
  const s = useLiveStream();
  const [mode, setMode] = useState<PhrasingMode>('lookup');
  const [fullscreen, setFullscreen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement === root.current);
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void root.current?.requestFullscreen?.();
  };

  const changeMode = useCallback((m: PhrasingMode) => {
    setMode(m);
    api.setPhrasingMode(m).catch(() => {});
  }, []);

  const waiting = s.currentWords.length > 0 && !s.phrase;
  const problem =
    s.link !== 'open'
      ? s.link === 'connecting'
        ? { text: 'Connecting to the backend…', tone: 'warn' }
        : { text: 'Backend disconnected. Retrying automatically…', tone: 'danger' }
      : !s.streaming
        ? { text: 'Sensors are not streaming. Check that the ESP32 is powered and on Wi-Fi.', tone: 'warn' }
        : null;

  return (
    <div ref={root} className={`space-y-5 ${fullscreen ? 'min-h-screen overflow-auto bg-bg p-6' : ''}`}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <span className="badge">Mode 2</span>
          <h1 className="h-page mt-1">Silent Speech</h1>
        </div>
        <div className="flex flex-wrap gap-3">
          <div className="flex gap-2" role="group" aria-label="Phrasing method">
            {(['lookup', 'llm'] as const).map((m) => (
              <button key={m} className={`btn ${mode === m ? 'btn-active' : ''}`} aria-pressed={mode === m} onClick={() => changeMode(m)}>
                {m === 'lookup' ? 'Lookup table' : 'LLM'}
              </button>
            ))}
          </div>
          <button className={`btn ${s.muted ? 'btn-active' : ''}`} aria-pressed={s.muted} onClick={() => s.setMuted(!s.muted)}>
            {s.muted ? '🔇 Voice muted' : '🔊 Mute voice'}
          </button>
          <button className="btn" onClick={s.clearPhrase} disabled={!s.currentWords.length && !s.phrase}>
            Clear phrase
          </button>
          <button className="btn" onClick={toggleFullscreen}>
            {fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          </button>
        </div>
      </div>

      {problem && (
        <p role="alert" className={`rounded-xl border p-3 text-lg font-semibold ${problem.tone === 'danger' ? 'border-danger bg-danger/10 text-danger' : 'border-warn bg-warn/10 text-warn'}`}>
          {problem.text}
        </p>
      )}

      <LiveImuChart height={fullscreen ? 150 : 110} />

      <section className="card" aria-label="Detected words" aria-live="polite">
        <div className="flex min-h-[15rem] flex-col items-center justify-center gap-6 sm:flex-row sm:flex-wrap">
          {s.currentWords.length === 0 ? (
            <p className="text-center text-3xl font-medium text-muted">Waiting for a silently mouthed word…</p>
          ) : (
            s.currentWords.map((w, i) => (
              <div key={w.id} className="flex items-center gap-6">
                {i > 0 && (
                  <span aria-hidden className="text-5xl font-bold text-muted">
                    +
                  </span>
                )}
                <WordChip word={w.word} confidence={w.confidence} size="lg" />
              </div>
            ))
          )}
          <div className="sm:ml-auto">
            <PauseRing progress={s.pauseProgress} active={waiting} />
          </div>
        </div>
      </section>

      <PhraseBanner words={s.currentWords.map((w) => w.word)} sentence={s.phrase?.sentence ?? null} speaking={s.speaking} muted={s.muted} large />

      <History phraseCount={s.phraseCount} />
    </div>
  );
}
