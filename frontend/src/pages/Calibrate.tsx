import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import { useLiveStream } from '../api/socket';
import TareButton from '../components/TareButton';
import type { CalibrateInfo } from '../types';

type Phase = 'home' | 'running' | 'training';

const GET_READY_MS = 1500;
const TAKE_OPTIONS = [5, 10, 15];
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export default function Calibrate() {
  const s = useLiveStream();
  const [info, setInfo] = useState<CalibrateInfo | null>(null);
  const [phase, setPhase] = useState<Phase>('home');
  const [message, setMessage] = useState('');
  const [notice, setNotice] = useState(''); // success note, e.g. after retraining from the Training data panel
  const [order, setOrder] = useState<string[]>([]);
  const [index, setIndex] = useState(0);
  const [recording, setRecording] = useState(false);
  const [progress, setProgress] = useState(0);
  const [picking, setPicking] = useState(false); // choosing words (new calibration, or continuing one)
  const [keep, setKeep] = useState(false); // true = add to the current calibration session
  const [selected, setSelected] = useState<string[]>([]);
  const [takes, setTakes] = useState(10);
  const [confirmReset, setConfirmReset] = useState(false);
  const [showData, setShowData] = useState(false); // "Training data" panel: which sessions to train on, and their weights
  const [sel, setSel] = useState<Record<number, { on: boolean; w: number }>>({});
  const cancel = useRef(false);
  const retry = useRef(false); // redo the current take (discarding it if it was already recording)
  const skipTrain = useRef(false); // stop without training
  const inited = useRef(false);

  const refresh = useCallback(() => api.calibrateInfo().then(setInfo).catch(() => setInfo(null)), []);
  useEffect(() => {
    void refresh();
    const t = window.setInterval(() => (phase === 'home' ? void refresh() : undefined), 2000);
    return () => clearInterval(t);
  }, [refresh, phase]);

  // first load: default word choice and takes per word come from the backend settings
  useEffect(() => {
    if (info && !inited.current) {
      inited.current = true;
      setSelected(info.words);
      setTakes(info.takes_per_word);
    }
  }, [info]);

  // Defaults follow the current calibration: it gets the default weight (5x), every other session goes back to 1x.
  // A row you changed by hand (checkbox or weight) is left alone.
  const touched = useRef(new Set<number>());
  useEffect(() => {
    if (!info) return;
    setSel((cur) => {
      const next = { ...cur };
      for (const x of info.sessions) {
        if (touched.current.has(x.session) && next[x.session]) continue;
        next[x.session] = { on: x.tared, w: x.current ? info.default_weight : 1 };
      }
      return next;
    });
  }, [info]);
  const edit = (session: number, patch: Partial<{ on: boolean; w: number }>) => {
    touched.current.add(session);
    setSel((c) => ({ ...c, [session]: { ...(c[session] ?? { on: true, w: 1 }), ...patch } }));
  };

  const trainSelected = useCallback(async () => {
    const weights: Record<number, number> = {};
    for (const [k, v] of Object.entries(sel)) if (v.on && v.w > 0) weights[Number(k)] = v.w;
    setPhase('training');
    setNotice('');
    try {
      const out = await api.calibrateTrain(null, weights);
      const n = Object.keys(weights).length;
      setMessage('');
      setNotice(`Model updated: ${n} session${n === 1 ? '' : 's'} · ${(out.labels ?? []).join(', ')}`);
    } catch (e) {
      setMessage((e as Error).message);
    }
    setPhase('home');
    void refresh();
  }, [sel, refresh]);

  const recorded = info ? Object.values(info.counts).reduce((a, b) => a + b, 0) : 0;
  const hasSession = !!info && info.session !== null && recorded > 0;
  const showPicker = !!info && (!hasSession || picking);

  const train = useCallback(
    async (session: number) => {
      setPhase('training');
      try {
        await api.calibrateTrain(session);
        setMessage('');
      } catch (e) {
        setMessage((e as Error).message);
      }
      setPicking(false);
      setPhase('home');
      void refresh();
    },
    [refresh],
  );

  const start = async () => {
    cancel.current = false;
    retry.current = false;
    skipTrain.current = false;
    setMessage('');
    setNotice('');
    let begun: { session: number; order: string[] };
    try {
      begun = await api.calibrateBegin({ words: selected, takes, keepSession: keep });
    } catch (e) {
      setMessage((e as Error).message);
      return;
    }
    const seconds = info?.seconds ?? 3;
    setOrder(begun.order);
    setPhase('running');
    let saved = 0;
    // waits `ms`, returning early if the user pressed Stop or Retry
    const wait = async (ms: number) => {
      const t0 = performance.now();
      while (performance.now() - t0 < ms && !cancel.current && !retry.current) await sleep(40);
    };
    for (let i = 0; i < begun.order.length && !cancel.current; i++) {
      setIndex(i);
      let ok = false;
      while (!ok && !cancel.current) {
        retry.current = false;
        setProgress(0);
        setRecording(false);
        await wait(GET_READY_MS);
        if (cancel.current) break;
        if (retry.current) continue; // pressed Retry while getting ready: start the countdown again
        try {
          await api.recordStart(begun.order[i], begun.session);
          setRecording(true);
          const t0 = performance.now();
          while (performance.now() - t0 < seconds * 1000 && !cancel.current && !retry.current) {
            setProgress((performance.now() - t0) / (seconds * 1000));
            await sleep(50);
          }
          const res = await api.recordStop();
          setRecording(false);
          if (cancel.current || retry.current) void api.deleteClip(res.clip_id).catch(() => {}); // cut short or redone: not a clean sample
          else {
            ok = true;
            saved++;
          }
        } catch (e) {
          setRecording(false);
          setMessage((e as Error).message);
          setPicking(false);
          setPhase('home');
          void refresh();
          return;
        }
      }
    }
    setRecording(false);
    if (saved === 0 || skipTrain.current) {
      setPicking(false);
      setPhase('home');
      void refresh();
      return;
    }
    await train(begun.session);
  };

  const stop = () => {
    cancel.current = true;
  };
  const cancelRun = () => {
    skipTrain.current = true;
    cancel.current = true;
  };
  const retryTake = () => {
    retry.current = true;
  };

  const continueCalibration = () => {
    const missing = (info?.allowed ?? []).filter((w) => !(info?.counts[w] ?? 0));
    setSelected(missing);
    setKeep(true);
    setPicking(true);
    setMessage('');
  };

  const reset = async () => {
    if (!confirmReset) {
      setConfirmReset(true);
      window.setTimeout(() => setConfirmReset(false), 4000);
      return;
    }
    setConfirmReset(false);
    try {
      await api.calibrateReset();
      setMessage('');
    } catch (e) {
      setMessage((e as Error).message);
    }
    setKeep(false);
    setPicking(false);
    setSelected(info?.words ?? []);
    void refresh();
  };

  const toggle = (w: string) => setSelected((cur) => (cur.includes(w) ? cur.filter((x) => x !== w) : [...cur, w]));

  const total = order.length;
  const word = order[index] ?? '';
  const ready = !!info && info.tared && s.streaming && s.link === 'open';

  return (
    <div className="space-y-5">
      {phase === 'running' && (
        <section className="card flex min-h-[22rem] flex-col items-center justify-center gap-6 text-center">
          <p className="text-sm font-semibold uppercase tracking-wide text-muted">
            {index + 1} / {total}
          </p>
          <p className="text-7xl font-bold sm:text-9xl">{word === 'nothing' ? 'Relax' : word}</p>
          <p className={`text-2xl font-semibold ${recording ? 'text-accent' : 'text-muted'}`}>{recording ? 'Mouth it now' : 'Get ready'}</p>
          <div className="h-3 w-full max-w-md overflow-hidden rounded-full bg-border" aria-hidden>
            <div className="h-full rounded-full bg-accent" style={{ width: `${progress * 100}%`, transition: 'width 0.05s linear' }} />
          </div>
          <div className="flex flex-wrap justify-center gap-3">
            <button className="btn btn-primary" onClick={retryTake}>
              Retry take
            </button>
            <button className="btn" onClick={stop}>
              Stop &amp; train
            </button>
            <button className="btn" onClick={cancelRun}>
              Cancel
            </button>
          </div>
        </section>
      )}

      {phase === 'training' && (
        <section className="card flex min-h-[22rem] items-center justify-center">
          <p className="text-3xl font-semibold text-muted">Training…</p>
        </section>
      )}

      {phase === 'home' && (
        <section className="card flex min-h-[22rem] flex-col items-center justify-center gap-6 text-center">
          {!info && <p className="text-lg text-muted">Backend not reachable.</p>}

          {info && hasSession && !picking && (
            <>
              <p className={`text-4xl font-bold ${info.calibrated ? 'text-ok' : 'text-warn'}`}>{info.calibrated ? 'Calibrated ✓' : 'Not trained yet'}</p>
              <div className="flex flex-wrap justify-center gap-2">
                {Object.entries(info.counts)
                  .sort()
                  .map(([w, n]) => (
                    <span key={w} className="badge !text-sm !normal-case">
                      {w} <span className="ml-1 tabular-nums">{n}</span>
                    </span>
                  ))}
              </div>
              <div className="flex flex-wrap justify-center gap-3">
                <button className="btn btn-primary" onClick={continueCalibration} disabled={!ready}>
                  Continue calibration
                </button>
                {!info.calibrated && info.session !== null && (
                  <button className="btn" onClick={() => void train(info.session as number)}>
                    Train
                  </button>
                )}
                <button className={`btn ${confirmReset ? 'btn-active' : ''}`} onClick={() => void reset()}>
                  {confirmReset ? 'Tap again to reset' : 'Reset calibration'}
                </button>
              </div>
            </>
          )}

          {showPicker && info && (
            <>
              <div className="flex flex-wrap justify-center gap-2" role="group" aria-label="Words">
                {info.allowed.map((w) => (
                  <button key={w} className={`btn !min-h-[2.75rem] ${selected.includes(w) ? 'btn-active' : ''}`} aria-pressed={selected.includes(w)} onClick={() => toggle(w)}>
                    {w}
                    {keep && info.counts[w] ? <span className="ml-1.5 tabular-nums text-muted">{info.counts[w]}</span> : null}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2" role="group" aria-label="Takes per word">
                {TAKE_OPTIONS.map((n) => (
                  <button key={n} className={`btn !min-h-[2.25rem] !px-4 ${takes === n ? 'btn-active' : ''}`} aria-pressed={takes === n} onClick={() => setTakes(n)}>
                    {n}
                  </button>
                ))}
                <span className="text-sm text-muted">each · {selected.length * takes} takes</span>
              </div>
              {!info.tared && (
                <div className="flex flex-col items-center gap-2">
                  <p className="text-lg font-semibold text-warn">Tare first</p>
                  <TareButton />
                </div>
              )}
              {info.tared && !s.streaming && <p className="text-lg font-semibold text-warn">Sensors not streaming</p>}
              <div className="flex gap-3">
                <button className="btn btn-primary min-h-[3.75rem] px-10 text-xl" onClick={start} disabled={!ready || selected.length === 0}>
                  {keep ? 'Add takes' : 'Start'}
                </button>
                {picking && (
                  <button className="btn" onClick={() => setPicking(false)}>
                    Cancel
                  </button>
                )}
              </div>
            </>
          )}


          {info && info.sessions.length > 0 && (
            <div className="w-full max-w-xl text-left">
              <button className="btn !min-h-[2.25rem] mx-auto block" aria-expanded={showData} onClick={() => setShowData((v) => !v)}>
                {showData ? 'Hide training data' : 'Training data'}
              </button>
              {showData && (
                <div className="mt-3 rounded-xl border border-border p-3">
                  <ul className="divide-y divide-border">
                    {info.sessions.map((x) => {
                      const v = sel[x.session] ?? { on: x.tared, w: 1 };
                      return (
                        <li key={x.session} className={`flex items-center gap-3 py-2 ${x.tared ? '' : 'opacity-50'}`}>
                          <input
                            type="checkbox"
                            className="h-5 w-5"
                            checked={v.on && x.tared}
                            disabled={!x.tared}
                            aria-label={`Include session ${x.session}`}
                            onChange={(e) => edit(x.session, { on: e.target.checked })}
                          />
                          <span className="w-24 font-semibold">
                            S{x.session}
                            {x.current && <span className="ml-1 text-xs font-normal text-accent">current</span>}
                          </span>
                          <span className="flex-1 text-sm tabular-nums text-muted">{x.tared ? `${x.total} takes` : 'no tare (can’t use)'}</span>
                          <label className={`flex items-center gap-1 text-sm text-muted ${v.on ? '' : 'opacity-60'}`}>
                            ×
                            <input
                              type="number"
                              min={0}
                              max={20}
                              step={0.5}
                              className="w-16 rounded-lg border border-border bg-surface px-2 py-1 text-base text-ink"
                              value={v.w}
                              disabled={!x.tared}
                              aria-label={`Weight for session ${x.session}`}
                              onChange={(e) => edit(x.session, { w: Math.max(0, Number(e.target.value) || 0) })}
                            />
                          </label>
                        </li>
                      );
                    })}
                  </ul>
                  <div className="mt-3 flex justify-center">
                    <button className="btn btn-primary" onClick={() => void trainSelected()} disabled={!Object.values(sel).some((v) => v.on && v.w > 0)}>
                      Train with these
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
          {notice && !message && (
            <p role="status" className="rounded-xl border border-ok bg-ok/10 px-4 py-2 font-semibold text-ok">
              {notice}
            </p>
          )}
          {message && (
            <p role="alert" className="rounded-xl border border-danger bg-danger/10 px-4 py-2 font-medium text-danger">
              {message}
            </p>
          )}
        </section>
      )}
    </div>
  );
}
