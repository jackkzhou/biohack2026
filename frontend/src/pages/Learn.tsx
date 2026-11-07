import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';
import PipelineDiagram from '../components/PipelineDiagram';
import Sparkline from '../components/Sparkline';
import { USE_MOCK } from '../config';
import { useUtteranceRecorder, type MicState } from '../hooks/useUtteranceRecorder';
import type { Clip, UtteranceResult } from '../types';

interface Item {
  key: number;
  status: 'pending' | 'done' | 'error';
  result?: UtteranceResult;
  error?: string;
}

function WordRow({ word, startMs, endMs, clipId, base }: { word: string; startMs: number; endMs: number; clipId: number; base: number }) {
  const [clip, setClip] = useState<Clip | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;
    api.getClip(clipId).then((c) => live && setClip(c)).catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [clipId]);
  return (
    <li className="flex flex-wrap items-center gap-x-4 gap-y-2 py-2">
      <span className="w-28 text-2xl font-bold">{word}</span>
      <span className="w-40 text-sm tabular-nums text-muted">
        {startMs - base} – {endMs - base} ms
      </span>
      {clip ? <Sparkline samples={clip.samples} /> : <span className="flex h-9 w-[150px] items-center justify-center rounded bg-surface-2 text-xs text-muted">{failed ? 'No preview' : 'Loading…'}</span>}
      <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-ok">
        <span aria-hidden>✓</span> Saved to Tiger Data <span className="font-normal text-muted">(clip #{clipId})</span>
      </span>
    </li>
  );
}

const micMessage: Partial<Record<MicState, string>> = {
  denied: 'Microphone permission was denied. Allow microphone access in your browser’s site settings, then try again.',
  unavailable: 'This browser can’t record audio here. Use a recent Chrome, Edge, Firefox or Safari on localhost or HTTPS.',
};

function Demo() {
  const [items, setItems] = useState<Item[]>([]);
  const key = useRef(0);

  const onUtterance = useCallback((audio: Blob, startedAt: number) => {
    const k = ++key.current;
    setItems((xs) => [{ key: k, status: 'pending' }, ...xs]);
    api
      .postUtterance(audio, startedAt)
      .then((result) => setItems((xs) => xs.map((x) => (x.key === k ? { ...x, status: 'done', result } : x))))
      .catch((e: Error) => setItems((xs) => xs.map((x) => (x.key === k ? { ...x, status: 'error', error: e.message } : x))));
  }, []);

  const { state, level, hearing, start, stop } = useUtteranceRecorder(onUtterance);
  const listening = state === 'listening';

  const { total, perWord } = useMemo(() => {
    const perWord = new Map<string, number>();
    let total = 0;
    for (const it of items) {
      for (const w of it.result?.words ?? []) {
        perWord.set(w.word.toLowerCase(), (perWord.get(w.word.toLowerCase()) ?? 0) + 1);
        total++;
      }
    }
    return { total, perWord: [...perWord.entries()].sort((a, b) => b[1] - a[1]) };
  }, [items]);

  return (
    <section className="card" aria-labelledby="demo-h">
      <div className="flex flex-wrap items-center gap-3">
        <h2 id="demo-h" className="h-section">
          Try it: live auto-labelling
        </h2>
        <span className="badge">Proof of concept</span>
      </div>
      <p className="mt-2 text-muted">
        Speak normally. After each pause the audio is sent to the backend, transcribed, and matched to the sensor data from the same moment. This panel is a live
        proof of concept: <strong className="text-ink">Mode 2’s model is not trained on this data.</strong>
        {USE_MOCK && ' (Demo mode: your audio is recorded, but the transcript below is simulated.)'}
      </p>

      <div className="mt-5 flex flex-wrap items-center gap-4">
        <button className={`btn min-h-[3.75rem] px-8 text-xl ${listening ? '' : 'btn-primary'}`} onClick={listening ? stop : start} disabled={state === 'requesting'}>
          {state === 'requesting' ? 'Waiting for microphone…' : listening ? '■ Stop listening' : '● Start listening'}
        </button>
        {listening && (
          <div className="flex items-center gap-3" aria-live="polite">
            <div className="h-3 w-40 overflow-hidden rounded-full bg-border" aria-hidden>
              <div className="h-full rounded-full bg-accent transition-[width] duration-75" style={{ width: `${level * 100}%` }} />
            </div>
            <span className={`text-sm font-semibold ${hearing ? 'text-accent' : 'text-muted'}`}>{hearing ? 'Hearing speech…' : 'Listening…'}</span>
          </div>
        )}
      </div>
      {micMessage[state] && (
        <p role="alert" className="mt-4 rounded-xl border border-danger bg-danger/10 p-3 font-medium text-danger">
          {micMessage[state]}
        </p>
      )}

      <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_16rem]">
        <div>
          {items.length === 0 ? (
            <p className="rounded-xl border border-dashed border-border p-6 text-center text-muted">
              Nothing yet. Press <strong>Start listening</strong> and say something like “water please”.
            </p>
          ) : (
            <ul className="space-y-4">
              {items.map((it) => (
                <li key={it.key} className="pop-in rounded-xl border border-border bg-surface-2 p-4">
                  {it.status === 'pending' && <p className="font-medium text-muted">Transcribing and labelling…</p>}
                  {it.status === 'error' && <p className="font-medium text-danger">Couldn’t label this utterance: {it.error}</p>}
                  {it.status === 'done' && it.result && (
                    <>
                      <p className="text-sm font-semibold uppercase tracking-wide text-muted">Transcript</p>
                      <p className="text-2xl font-bold">“{it.result.transcript}”</p>
                      <ul className="mt-2 divide-y divide-border">
                        {it.result.words.map((w) => (
                          <WordRow key={w.clip_id} word={w.word} startMs={w.start_ms} endMs={w.end_ms} clipId={w.clip_id} base={it.result!.words[0].start_ms} />
                        ))}
                      </ul>
                    </>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>

        <aside aria-label="Session counts" className="h-fit rounded-xl border border-border bg-surface-2 p-4">
          <p className="text-sm font-semibold uppercase tracking-wide text-muted">Auto-labelled clips this session</p>
          <p className="text-5xl font-bold tabular-nums text-accent">{total}</p>
          {perWord.length > 0 ? (
            <table className="mt-3 w-full text-left">
              <thead className="sr-only">
                <tr>
                  <th>Word</th>
                  <th>Clips</th>
                </tr>
              </thead>
              <tbody>
                {perWord.map(([w, n]) => (
                  <tr key={w} className="border-t border-border">
                    <td className="py-1.5 font-semibold">{w}</td>
                    <td className="py-1.5 text-right tabular-nums">{n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="mt-2 text-sm text-muted">Per-word counts appear here.</p>
          )}
        </aside>
      </div>
    </section>
  );
}

export default function Learn() {
  return (
    <div className="space-y-10">
      <header className="max-w-3xl">
        <span className="badge">Mode 1</span>
        <h1 className="h-page mt-2">Learn</h1>
        <p className="mt-4 text-xl">
          While Alex can still speak, a microphone writes down what he says and works out exactly <em>when</em> each word was spoken. The sensor data from that same
          moment is cut out and saved with the word as its label.
        </p>
        <p className="mt-3 text-lg text-muted">
          So his personal dataset builds itself during ordinary days, with no dedicated training sessions. The same recordings quietly bank his voice for the clone that
          will speak for him later.
        </p>
      </header>

      <Demo />

      <section aria-labelledby="pipeline-h">
        <h2 id="pipeline-h" className="h-section mb-4">
          How it fits together over time
        </h2>
        <PipelineDiagram />
      </section>
    </div>
  );
}
