import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useThemeColors } from '../hooks/useThemeColors';

const months = Array.from({ length: 13 }, (_, m) => m);
const clipGrowth = months.slice(1).map((m) => ({
  month: m,
  help: Math.round(18 * m * (1 + 0.05 * m)),
  water: Math.round(12 * m),
  yes: Math.round(8 * m),
}));
const progression = months.map((m) => ({
  month: m,
  clarity: Math.round(Math.max(18, 98 - 0.45 * m * m)),
  silent: Math.round(80 + 0.8 * m),
}));

function Stage({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="card flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent text-lg font-bold text-accent-ink">{n}</span>
        <h3 className="text-lg font-semibold leading-tight">{title}</h3>
      </div>
      {children}
    </div>
  );
}

const Arrow = () => (
  <div aria-hidden className="flex items-center justify-center text-4xl text-muted">
    <span className="rotate-90 lg:rotate-0">→</span>
  </div>
);

function FlowSvg() {
  const steps = ['Alex speaks', 'Whisper: transcript + word timestamps', 'Matching IMU window', 'Labelled clip'];
  const w = 300;
  const boxH = 46;
  const gap = 30;
  const h = steps.length * boxH + (steps.length - 1) * gap;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label={`Auto-labelling steps: ${steps.join(', then ')}`}>
      {steps.map((s, i) => {
        const y = i * (boxH + gap);
        const last = i === steps.length - 1;
        return (
          <g key={s}>
            <rect x="2" y={y} width={w - 4} height={boxH} rx="12" strokeWidth="2" style={{ fill: last ? 'rgb(var(--accent) / 0.15)' : 'rgb(var(--surface-2))', stroke: last ? 'rgb(var(--accent))' : 'rgb(var(--border))' }} />
            <text x={w / 2} y={y + boxH / 2 + 5} textAnchor="middle" fontSize="14" fontWeight="600" style={{ fill: 'rgb(var(--ink))' }}>
              {s.length > 30 ? s.slice(0, s.indexOf(':') + 1) : s}
            </text>
            {s.length > 30 && (
              <text x={w / 2} y={y + boxH / 2 + 20} textAnchor="middle" fontSize="12" style={{ fill: 'rgb(var(--muted))' }}>
                {s.slice(s.indexOf(':') + 2)}
              </text>
            )}
            {!last && (
              <path d={`M ${w / 2} ${y + boxH + 4} v ${gap - 12} m -6 -6 l 6 6 l 6 -6`} fill="none" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ stroke: 'rgb(var(--muted))' }} />
            )}
          </g>
        );
      })}
    </svg>
  );
}

export default function PipelineDiagram() {
  const c = useThemeColors();
  const axis = { stroke: c.muted, fontSize: 12 };
  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_auto_1fr_auto_1fr]">
      <Stage n={1} title="Auto-labelling">
        <FlowSvg />
        <p className="text-sm text-muted">No training sessions: every sentence Alex says labels his own data.</p>
      </Stage>
      <Arrow />
      <Stage n={2} title="Dataset grows over months">
        <span className="badge self-start">Illustrative</span>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={clipGrowth} margin={{ left: -10, right: 8, top: 4 }}>
              <CartesianGrid stroke={c.border} strokeDasharray="3 3" />
              <XAxis dataKey="month" {...axis} label={{ value: 'Month', position: 'insideBottom', offset: -2, fill: c.muted, fontSize: 12 }} />
              <YAxis {...axis} />
              <Tooltip contentStyle={{ background: c.surface, border: `1px solid ${c.border}`, color: c.ink }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="help" name="help" stroke={c.accent} strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="water" name="water" stroke={c.s2} strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="yes" name="yes" stroke={c.s3} strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="text-sm text-muted">Labelled clips per word, cumulative.</p>
      </Stage>
      <Arrow />
      <Stage n={3} title="Progression tracking">
        <span className="badge self-start">Illustrative</span>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={progression} margin={{ left: -10, right: 8, top: 4 }}>
              <CartesianGrid stroke={c.border} strokeDasharray="3 3" />
              <XAxis dataKey="month" {...axis} label={{ value: 'Month', position: 'insideBottom', offset: -2, fill: c.muted, fontSize: 12 }} />
              <YAxis domain={[0, 100]} unit="%" {...axis} />
              <Tooltip contentStyle={{ background: c.surface, border: `1px solid ${c.border}`, color: c.ink }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="clarity" name="Speech clarity" stroke={c.danger} strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="silent" name="Silent-speech accuracy" stroke={c.accent} strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="text-sm text-muted">
          As his voice fades, silent speech takes over. Meanwhile his recordings keep banking his voice for the ElevenLabs clone.
        </p>
      </Stage>
    </div>
  );
}
