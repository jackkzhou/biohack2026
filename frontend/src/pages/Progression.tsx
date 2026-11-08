import { useCallback, useEffect, useState } from 'react';
import { api } from '../api/client';
import DeteriorationMeter from '../components/DeteriorationMeter';
import RiskBadge from '../components/RiskBadge';
import TrendSparkline from '../components/TrendSparkline';
import type { BaselineStatus, DailyFeatureRow, DeteriorationPrediction, PredictionResult } from '../types';

const DIRECTION_ARROW: Record<string, string> = { increased: '↑', decreased: '↓', unchanged: '→' };
const SEVERITY_TONE: Record<string, string> = { low: 'text-ok', moderate: 'text-warn', high: 'text-danger' };

function Baseline({ status }: { status: BaselineStatus }) {
  const collected = status.days_required - status.days_remaining;
  const pct = Math.round((collected / status.days_required) * 100);
  return (
    <section className="card" aria-label="Baseline progress">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="h-section">Collecting baseline</p>
          <p className="mt-1 text-muted">{status.message}</p>
        </div>
        <p className="text-3xl font-bold tabular-nums text-accent">
          {collected}
          <span className="text-base font-medium text-muted">/{status.days_required}</span>
        </p>
      </div>
      <div className="mt-4 h-3 overflow-hidden rounded-full bg-border" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${pct}%` }} />
      </div>
    </section>
  );
}

function Markers({ markers }: { markers: DeteriorationPrediction['markers'] }) {
  if (markers.length === 0) return <p className="text-muted">No specific markers flagged.</p>;
  return (
    <ul className="divide-y divide-border">
      {markers.map((m, i) => (
        <li key={i} className="flex flex-wrap items-start gap-3 py-3">
          <span className={`text-xl ${SEVERITY_TONE[m.severity]}`} aria-hidden>
            {DIRECTION_ARROW[m.direction]}
          </span>
          <div className="flex-1">
            <p className="font-semibold">
              {m.name.replace(/_/g, ' ')}
              <span className={`ml-2 text-xs font-semibold uppercase tracking-wide ${SEVERITY_TONE[m.severity]}`}>{m.severity}</span>
            </p>
            <p className="text-sm text-muted">{m.evidence}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}

function Prediction({ result, onRefresh, loading }: { result: PredictionResult; onRefresh: () => void; loading: boolean }) {
  const r = result.response;
  return (
    <section className="card space-y-5" aria-label="Deterioration prediction">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="h-section">Prediction for {result.target_day}</p>
          <p className="text-sm text-muted">
            From {result.window_start} to {result.window_end} · {result.cached ? 'cached' : 'freshly generated'}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <RiskBadge risk={r.risk_score} />
          <button className="btn" onClick={onRefresh} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      <div>
        <p className="mb-1 flex items-baseline justify-between text-sm font-semibold uppercase tracking-wide text-muted">
          Deterioration index
          <span className="text-xs font-normal normal-case text-muted">Confidence {(r.confidence * 100).toFixed(0)}%</span>
        </p>
        <DeteriorationMeter value={r.deterioration_index} />
      </div>

      <p>{r.summary}</p>

      <div>
        <p className="h-section !text-base">Markers</p>
        <Markers markers={r.markers} />
      </div>

      <div>
        <p className="h-section !text-base">Clinical &amp; lifestyle notes</p>
        <ul className="list-disc space-y-1 pl-5">
          {r.clinical_notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}

const FEATURE_COLUMNS: { key: keyof DailyFeatureRow['features']; label: string; digits: number }[] = [
  { key: 'tremor_frequency_hz', label: 'Tremor Hz', digits: 2 },
  { key: 'jaw_asymmetry_index', label: 'Asymmetry', digits: 2 },
  { key: 'speech_cadence_wpm', label: 'Cadence wpm', digits: 1 },
  { key: 'detection_confidence_mean', label: 'Confidence', digits: 2 },
  { key: 'rejection_rate', label: 'Rejection', digits: 2 },
  { key: 'rest_activity_rms', label: 'Rest RMS', digits: 4 },
];

function Trends({ rows }: { rows: DailyFeatureRow[] }) {
  const [showTable, setShowTable] = useState(false);
  const series = (key: keyof DailyFeatureRow['features']) => rows.map((r) => ({ day: r.day, value: r.features[key] as number | null }));

  return (
    <section className="card space-y-4" aria-label="14-day trend">
      <p className="h-section">14-day trend</p>
      <div className="grid gap-3 sm:grid-cols-3">
        <TrendSparkline label="Tremor frequency" unit=" Hz" digits={2} points={series('tremor_frequency_hz')} />
        <TrendSparkline label="Jaw asymmetry" digits={2} points={series('jaw_asymmetry_index')} />
        <TrendSparkline label="Speech cadence" unit=" wpm" digits={1} points={series('speech_cadence_wpm')} />
      </div>
      <button className="text-sm font-semibold text-accent underline-offset-2 hover:underline" onClick={() => setShowTable((s) => !s)} aria-expanded={showTable}>
        {showTable ? 'Hide' : 'View'} all {rows.length} day{rows.length === 1 ? '' : 's'} as a table
      </button>
      {showTable && (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-border text-muted">
                <th className="py-1 pr-3">Day</th>
                {FEATURE_COLUMNS.map((c) => (
                  <th key={c.key} className="py-1 pr-3">
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.day} className="border-b border-border tabular-nums">
                  <td className="py-1 pr-3 font-semibold">{r.day}</td>
                  {FEATURE_COLUMNS.map((c) => {
                    const v = r.features[c.key] as number | null;
                    return (
                      <td key={c.key} className="py-1 pr-3">
                        {v != null ? v.toFixed(c.digits) : '–'}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default function Progression() {
  const [status, setStatus] = useState<BaselineStatus | null>(null);
  const [rows, setRows] = useState<DailyFeatureRow[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let live = true;
    Promise.all([api.getProgressionStatus(), api.getProgressionWindow()])
      .then(([s, w]) => {
        if (!live) return;
        setStatus(s);
        setRows(w);
      })
      .catch((e: Error) => live && setLoadError(e.message));
    return () => {
      live = false;
    };
  }, []);

  const fetchPrediction = useCallback((force: boolean) => {
    setLoading(true);
    setPredictionError(null);
    api
      .getProgressionPrediction(force)
      .then(setResult)
      .catch((e: Error) => setPredictionError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (status?.ready && !result && !loading && !predictionError) fetchPrediction(false);
  }, [status, result, loading, predictionError, fetchPrediction]);

  if (loadError) {
    return (
      <p role="alert" className="rounded-xl border border-danger bg-danger/10 p-3 font-medium text-danger">
        {loadError}
      </p>
    );
  }
  if (!status || !rows) return <p className="text-muted">Loading…</p>;

  return (
    <div className="space-y-6">
      {!status.ready && <Baseline status={status} />}

      {predictionError && (
        <section className="card space-y-3" aria-label="Prediction error">
          <p role="alert" className="font-medium text-danger">
            {predictionError}
          </p>
          <button className="btn" onClick={() => fetchPrediction(false)} disabled={loading}>
            {loading ? 'Trying…' : 'Try again'}
          </button>
        </section>
      )}

      {result && <Prediction result={result} onRefresh={() => fetchPrediction(true)} loading={loading} />}

      {status.ready && !result && !predictionError && loading && <p className="text-muted">Generating prediction…</p>}

      {rows.length > 0 && <Trends rows={rows} />}
    </div>
  );
}
