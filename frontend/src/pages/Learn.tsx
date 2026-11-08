import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';
import Sparkline from '../components/Sparkline';
import { useUtteranceRecorder, type MicState } from '../hooks/useUtteranceRecorder';
import type { Clip, ClipInfo, UtteranceResult } from '../types';

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
        <span aria-hidden>✓</span> Saved
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
    <section className="card" aria-label="Train">
      <div className="flex flex-wrap items-center gap-4">
        <button className={`btn min-h-[3.75rem] px-8 text-xl ${listening ? '' : 'btn-primary'}`} onClick={listening ? stop : start} disabled={state === 'requesting'}>
          {state === 'requesting' ? 'Starting…' : listening ? '■ Stop listening' : '● Start listening'}
        </button>
        {listening && (
          <div className="flex items-center gap-3" aria-live="polite">
            <div className="h-3 w-40 overflow-hidden rounded-full bg-border" aria-hidden>
              <div className="h-full rounded-full bg-accent transition-[width] duration-75" style={{ width: `${level * 100}%` }} />
            </div>
            <span className={`text-sm font-semibold ${hearing ? 'text-accent' : 'text-muted'}`}>{hearing ? 'Hearing…' : 'Listening…'}</span>
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
              Press Start and say something.
            </p>
          ) : (
            <ul className="space-y-4">
              {items.map((it) => (
                <li key={it.key} className="pop-in rounded-xl border border-border bg-surface-2 p-4">
                  {it.status === 'pending' && <p className="font-medium text-muted">Working…</p>}
                  {it.status === 'error' && <p className="font-medium text-danger">Couldn’t label this utterance: {it.error}</p>}
                  {it.status === 'done' && it.result && (
                    <>
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
          <p className="text-sm font-semibold uppercase tracking-wide text-muted">Clips</p>
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
          ) : null}
        </aside>
      </div>
    </section>
  );
}

function ClipThumb({ info }: { info: ClipInfo }) {
  const [clip, setClip] = useState<Clip | null>(null);
  useEffect(() => {
    let live = true;
    api.getClip(info.id).then((c) => live && setClip(c)).catch(() => {});
    return () => {
      live = false;
    };
  }, [info.id]);
  return (
    <li className="flex items-center justify-between gap-3 py-2">
      <span className="text-sm tabular-nums text-muted">#{info.id}</span>
      {clip ? <Sparkline samples={clip.samples} /> : <span className="h-9 w-[150px] rounded bg-surface-2" />}
      <span className="w-14 text-right text-sm tabular-nums text-muted">{info.session != null ? `S${info.session}` : ''}</span>
    </li>
  );
}

function Library() {
  const [clips, setClips] = useState<ClipInfo[] | null>(null);
  const [error, setError] = useState(false);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    let timer: number | undefined;
    const load = () =>
      api
        .listClips()
        .then((c) => live && (setClips(c), setError(false)))
        .catch(() => {
          if (!live) return;
          setError(true);
          timer = window.setTimeout(load, 3000); // keep trying until the database is up
        });
    void load();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, []);

  const groups = useMemo(() => {
    const m = new Map<string, ClipInfo[]>();
    for (const c of clips ?? []) m.set(c.label, [...(m.get(c.label) ?? []), c]);
    return [...m.entries()].sort((a, b) => b[1].length - a[1].length);
  }, [clips]);

  return (
    <section className="card" aria-label="Clips">
      {error ? (
        <p className="text-danger">Can’t reach the database. Retrying…</p>
      ) : clips === null ? (
        <p className="text-muted">Loading…</p>
      ) : groups.length === 0 ? (
        <p className="text-muted">No clips.</p>
      ) : (
        <>
          <div className="mb-3 flex items-baseline gap-3">
            <span className="text-5xl font-bold tabular-nums text-accent">{clips.length}</span>
            <span className="text-sm font-semibold uppercase tracking-wide text-muted">clips</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {groups.map(([label, items]) => (
              <button key={label} className={`btn !min-h-[2.5rem] ${open === label ? 'btn-active' : ''}`} aria-pressed={open === label} onClick={() => setOpen(open === label ? null : label)}>
                {label} <span className="ml-1 tabular-nums text-muted">{items.length}</span>
              </button>
            ))}
          </div>
          {open && (
            <ul className="mt-3 divide-y divide-border">
              {(groups.find(([l]) => l === open)?.[1] ?? []).slice(0, 12).map((c) => (
                <ClipThumb key={c.id} info={c} />
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}

export default function Learn() {
  return (
    <div className="space-y-6">
      <Demo />
      <Library />
    </div>
  );
}
