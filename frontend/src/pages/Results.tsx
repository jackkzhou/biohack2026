import { useEffect, useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api/client';
import ConfusionMatrix from '../components/ConfusionMatrix';
import { useThemeColors } from '../hooks/useThemeColors';
import type { ModelMetrics } from '../types';

const pct = (v: number) => `${Math.round(v * 100)}%`;

export default function Results() {
  const c = useThemeColors();
  const [m, setM] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api.getMetrics().then((x) => live && setM(x)).catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, []);

  const derived = useMemo(() => {
    if (!m) return null;
    const perWord = m.labels.map((label, i) => {
      const total = m.matrix[i].reduce((a, b) => a + b, 0);
      return { label, acc: total ? m.matrix[i][i] / total : 0, n: total };
    });
    let worst = { a: 0, b: 0, count: -1, ab: 0, ba: 0 };
    for (let i = 0; i < m.labels.length; i++) {
      for (let j = i + 1; j < m.labels.length; j++) {
        const count = m.matrix[i][j] + m.matrix[j][i];
        if (count > worst.count) worst = { a: i, b: j, count, ab: m.matrix[i][j], ba: m.matrix[j][i] };
      }
    }
    return { perWord, worst };
  }, [m]);

  if (error) {
    return (
      <p role="alert" className="rounded-xl border border-danger bg-danger/10 p-4 text-lg font-medium text-danger">
        Couldn’t load model metrics: {error}
      </p>
    );
  }
  if (!m || !derived) return <p className="text-xl text-muted">Loading results…</p>;

  const trained = m.heldout_session > 2 ? `1–${m.heldout_session - 1}` : '1';
  const { worst, perWord } = derived;

  return (
    <div className="space-y-8">
      <header>
        <span className="badge">Mode 2 · {m.model}</span>
        <h1 className="h-page mt-2">Results</h1>
      </header>

      <section className="card flex flex-col items-start gap-6 sm:flex-row sm:items-center">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-muted">Held-out accuracy</p>
          <p className="text-8xl font-extrabold leading-none text-accent tabular-nums">{pct(m.accuracy)}</p>
        </div>
        <p className="max-w-xl text-lg text-muted">
          Trained on sessions {trained}, tested on session {m.heldout_session} <strong className="text-ink">after the sensors were re-mounted</strong>. The model never saw
          that session, so this measures how well it works on a new day, not how well it memorised.
        </p>
      </section>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="card" aria-labelledby="cm-h">
          <h2 id="cm-h" className="h-section mb-4">
            Confusion matrix
          </h2>
          <ConfusionMatrix labels={m.labels} matrix={m.matrix} />
        </section>

        <section className="card" aria-labelledby="cmp-h">
          <h2 id="cmp-h" className="h-section mb-4">
            Model comparison
          </h2>
          <div className="h-72" role="img" aria-label={`Accuracy by model: ${m.comparison.map((x) => `${x.model} ${pct(x.accuracy)}`).join(', ')}`}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={m.comparison} layout="vertical" margin={{ left: 8, right: 44 }}>
                <CartesianGrid stroke={c.border} horizontal={false} />
                <XAxis type="number" domain={[0, 1]} tickFormatter={pct} stroke={c.muted} />
                <YAxis type="category" dataKey="model" width={130} stroke={c.muted} tick={{ fill: c.ink, fontSize: 14 }} />
                <Tooltip formatter={(v) => pct(Number(v))} contentStyle={{ background: c.surface, border: `1px solid ${c.border}`, color: c.ink }} cursor={{ fill: c.border, opacity: 0.3 }} />
                <Bar dataKey="accuracy" radius={6}>
                  {m.comparison.map((x) => (
                    <Cell key={x.model} fill={x.model === m.model ? c.accent : c.muted} fillOpacity={x.model === m.model ? 1 : 0.55} />
                  ))}
                  <LabelList dataKey="accuracy" position="right" formatter={(v) => pct(Number(v))} fill={c.ink} fontSize={14} fontWeight={700} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <section className="card" aria-labelledby="pw-h">
        <h2 id="pw-h" className="h-section mb-4">
          Accuracy per word
        </h2>
        <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {perWord.map((w) => {
            const involved = w.label === m.labels[worst.a] || w.label === m.labels[worst.b];
            return (
              <li key={w.label} className={`rounded-xl border p-3 ${involved ? 'border-warn bg-warn/10' : 'border-border'}`}>
                <div className="flex items-baseline justify-between">
                  <span className="text-lg font-semibold">{w.label}</span>
                  <span className="text-xl font-bold tabular-nums">{pct(w.acc)}</span>
                </div>
                <div className="mt-2 h-3 overflow-hidden rounded-full bg-border">
                  <div className={`h-full rounded-full ${involved ? 'bg-warn' : 'bg-accent'}`} style={{ width: pct(w.acc) }} />
                </div>
              </li>
            );
          })}
        </ul>
        {worst.count > 0 && (
          <p className="mt-5 rounded-xl bg-warn/10 p-3 text-lg">
            <strong className="text-warn">Most confused pair:</strong> “{m.labels[worst.a]}” and “{m.labels[worst.b]}”. {worst.ab} “{m.labels[worst.a]}” clips were read as “
            {m.labels[worst.b]}”, and {worst.ba} the other way.
          </p>
        )}
      </section>
    </div>
  );
}
