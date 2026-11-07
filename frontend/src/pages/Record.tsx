import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import LiveImuChart from '../components/LiveImuChart';
import { SAMPLES_PER_WORD, VOCAB } from '../types';

const NOTHING_HINTS = ['Stay still', 'Swallow', 'Nod your head', 'Turn your head', 'Idle fidgeting'];
const TAKE_MS = 2000;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const shuffle = <T,>(xs: T[]) => [...xs].sort(() => Math.random() - 0.5);
const pick = <T,>(xs: T[]) => xs[Math.floor(Math.random() * xs.length)];

function beep(freq: number) {
  try {
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.frequency.value = freq;
    gain.gain.value = 0.15;
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.15);
    osc.onended = () => void ctx.close();
  } catch {
    /* audio cues are optional */
  }
}

type Phase = 'idle' | 'countdown' | 'recording' | 'between';

export default function Record() {
  const [session, setSession] = useState(1);
  const [counts, setCounts] = useState<Record<string, number>>(() => Object.fromEntries(VOCAB.map((w) => [w, 0])));
  const [running, setRunning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [phase, setPhase] = useState<Phase>('idle');
  const [cue, setCue] = useState<{ word: string; hint?: string } | null>(null);
  const [countdown, setCountdown] = useState(3);
  const [error, setError] = useState<string | null>(null);
  const [rest, setRest] = useState<'idle' | 'capturing' | 'done'>('idle');
  const [last, setLast] = useState<{ id: number; label: string } | null>(null);

  const countsRef = useRef(counts);
  countsRef.current = counts;
  const sessionRef = useRef(session);
  sessionRef.current = session;
  const stopRequested = useRef(false);
  const bag = useRef<string[]>([]);
  const forcedNext = useRef<string | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    api.getRecordCounts().then((c) => mounted.current && setCounts((prev) => ({ ...prev, ...c }))).catch(() => {});
    return () => {
      mounted.current = false;
      stopRequested.current = true;
    };
  }, []);

  const nextWord = (): string | null => {
    if (forcedNext.current) {
      const w = forcedNext.current;
      forcedNext.current = null;
      return w;
    }
    const open = VOCAB.filter((w) => (countsRef.current[w] ?? 0) < SAMPLES_PER_WORD);
    if (open.length === 0) return null;
    bag.current = bag.current.filter((w) => open.includes(w as (typeof VOCAB)[number]));
    if (bag.current.length === 0) bag.current = shuffle([...open]);
    return bag.current.pop()!;
  };

  const runSession = async () => {
    setError(null);
    stopRequested.current = false;
    setRunning(true);
    try {
      while (!stopRequested.current && mounted.current) {
        const word = nextWord();
        if (!word) break;
        setCue({ word, hint: word === 'nothing' ? pick(NOTHING_HINTS) : undefined });
        setPhase('countdown');
        for (let n = 3; n >= 1; n--) {
          setCountdown(n);
          await sleep(1000);
        }
        beep(880);
        await api.recordStart(word, sessionRef.current);
        setPhase('recording');
        await sleep(TAKE_MS);
        beep(520);
        const res = await api.recordStop();
        if (!mounted.current) return;
        setCounts((c) => ({ ...c, [word]: (c[word] ?? 0) + 1 }));
        setLast({ id: res.clip_id, label: word });
        setPhase('between');
        await sleep(1500);
      }
    } catch (e) {
      setError(`${(e as Error).message}. Is the backend reachable?`);
    } finally {
      if (mounted.current) {
        setRunning(false);
        setStopping(false);
        setPhase('idle');
      }
    }
  };

  const captureRest = useCallback(async () => {
    setError(null);
    setRest('capturing');
    try {
      await api.recordRestPose(sessionRef.current);
      setRest('done');
    } catch (e) {
      setRest('idle');
      setError(`Rest pose failed: ${(e as Error).message}`);
    }
  }, []);

  const redo = async () => {
    if (!last) return;
    try {
      await api.deleteClip(last.id);
      setCounts((c) => ({ ...c, [last.label]: Math.max(0, (c[last.label] ?? 0) - 1) }));
      forcedNext.current = last.label;
      setLast(null);
    } catch (e) {
      setError(`Couldn’t delete the last clip: ${(e as Error).message}`);
    }
  };

  const total = VOCAB.length * SAMPLES_PER_WORD;
  const done = VOCAB.reduce((n, w) => n + Math.min(counts[w] ?? 0, SAMPLES_PER_WORD), 0);
  const takeBusy = phase === 'recording'; // redo is fine any other time

  return (
    <div className="space-y-6">
      <header className="max-w-3xl">
        <span className="badge">Mode 2 · data collection</span>
        <h1 className="h-page mt-2">Record</h1>
        <p className="mt-3 text-lg text-muted">
          Silently mouth each word when cued, about {SAMPLES_PER_WORD} times per word. Start each session by holding a relaxed face for the rest pose.
        </p>
      </header>

      {error && (
        <p role="alert" className="rounded-xl border border-danger bg-danger/10 p-3 font-medium text-danger">
          {error}
        </p>
      )}

      <section className="card flex flex-wrap items-end gap-4" aria-label="Session setup">
        <label className="flex flex-col gap-1 text-sm font-semibold">
          Session number
          <input
            type="number"
            min={1}
            value={session}
            disabled={running}
            onChange={(e) => setSession(Math.max(1, Number(e.target.value) || 1))}
            className="min-h-[3rem] w-28 rounded-xl border border-border bg-surface px-3 text-lg"
          />
        </label>
        <button className="btn" onClick={captureRest} disabled={running || rest === 'capturing'}>
          {rest === 'capturing' ? 'Hold still… (2 s)' : 'Capture 2 s rest pose'}
        </button>
        {rest === 'done' && <span className="pb-3 font-semibold text-ok">✓ Rest pose captured for session {session}</span>}
        <div className="ml-auto flex flex-wrap gap-3">
          <button className="btn" onClick={redo} disabled={!last || takeBusy}>
            ↶ Redo last{last ? ` (${last.label})` : ''}
          </button>
          {running ? (
            <button
              className="btn"
              onClick={() => {
                stopRequested.current = true;
                setStopping(true);
              }}
              disabled={stopping}
            >
              {stopping ? 'Stopping after this take…' : 'Stop after this take'}
            </button>
          ) : (
            <button className="btn btn-primary" onClick={runSession} disabled={done >= total}>
              {done >= total ? 'All words complete' : done > 0 ? 'Continue session' : 'Start session'}
            </button>
          )}
        </div>
      </section>

      <section className="card flex min-h-[20rem] flex-col items-center justify-center text-center" aria-live="assertive" aria-label="Cue">
        {!cue || phase === 'idle' ? (
          <p className="text-2xl font-medium text-muted">{done >= total ? 'All words recorded. Nice work.' : 'Press Start session. Words will be cued in random order.'}</p>
        ) : (
          <>
            <p className="text-sm font-semibold uppercase tracking-widest text-muted">
              {phase === 'countdown' ? 'Get ready to mouth' : phase === 'recording' ? 'Mouth it now' : 'Saved'}
            </p>
            <p className="my-2 text-7xl font-extrabold leading-none sm:text-9xl">{cue.word}</p>
            {cue.hint && <p className="text-3xl font-semibold text-accent">{cue.hint}</p>}
            <div className="mt-6 flex h-24 items-center justify-center">
              {phase === 'countdown' && <span className="text-7xl font-bold tabular-nums text-accent">{countdown}</span>}
              {phase === 'recording' && (
                <div className="w-72 max-w-full">
                  <div className="h-4 overflow-hidden rounded-full bg-border">
                    <div className="h-full origin-left rounded-full bg-danger" style={{ animation: `fill-x ${TAKE_MS}ms linear forwards` }} />
                  </div>
                  <p className="mt-2 text-lg font-bold text-danger">● Recording</p>
                </div>
              )}
              {phase === 'between' && <span className="text-3xl font-bold text-ok">✓</span>}
            </div>
          </>
        )}
      </section>

      <LiveImuChart height={110} />

      <section className="card" aria-labelledby="progress-h">
        <div className="mb-4 flex items-baseline justify-between">
          <h2 id="progress-h" className="h-section">
            Progress
          </h2>
          <span className="font-semibold tabular-nums text-muted">
            {done} / {total}
          </span>
        </div>
        <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {VOCAB.map((w) => {
            const n = counts[w] ?? 0;
            const complete = n >= SAMPLES_PER_WORD;
            return (
              <li key={w}>
                <div className="flex justify-between font-semibold">
                  <span>{w}</span>
                  <span className={`tabular-nums ${complete ? 'text-ok' : 'text-muted'}`}>
                    {n} / {SAMPLES_PER_WORD} {complete && '✓'}
                  </span>
                </div>
                <div className="mt-1 h-3 overflow-hidden rounded-full bg-border" role="progressbar" aria-valuemin={0} aria-valuemax={SAMPLES_PER_WORD} aria-valuenow={n} aria-label={`${w} samples`}>
                  <div className={`h-full rounded-full ${complete ? 'bg-ok' : 'bg-accent'}`} style={{ width: `${Math.min(100, (n / SAMPLES_PER_WORD) * 100)}%` }} />
                </div>
              </li>
            );
          })}
        </ul>
      </section>
    </div>
  );
}
