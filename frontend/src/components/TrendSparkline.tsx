import { useMemo, useRef, useState } from 'react';

export interface TrendPoint {
  day: string;
  value: number | null;
}

const WIDTH = 280;
const HEIGHT = 64;
const PAD = 8;

const toX = (i: number, n: number) => PAD + (i / Math.max(1, n - 1)) * (WIDTH - PAD * 2);

/** Single-hue 14-point trend line with a hover/keyboard readout. No legend: one series,
 * so the label above already names what's plotted (per the single-series rule). */
export default function TrendSparkline({ label, unit = '', digits = 1, points }: { label: string; unit?: string; digits?: number; points: TrendPoint[] }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const known = points.map((p) => p.value).filter((v): v is number => v != null);
  const min = known.length ? Math.min(...known) : 0;
  const max = known.length ? Math.max(...known) : 1;
  const span = max - min || 1;
  const toY = (v: number) => HEIGHT - PAD - ((v - min) / span) * (HEIGHT - PAD * 2);

  const segments = useMemo(() => {
    const out: string[] = [];
    let current: string[] = [];
    points.forEach((p, i) => {
      if (p.value == null) {
        if (current.length) out.push(current.join(' '));
        current = [];
        return;
      }
      current.push(`${toX(i, points.length).toFixed(1)},${toY(p.value).toFixed(1)}`);
    });
    if (current.length) out.push(current.join(' '));
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, min, span]);

  const lastKnownIndex = (() => {
    for (let i = points.length - 1; i >= 0; i--) if (points[i].value != null) return i;
    return null;
  })();

  const setHoverFromClientX = (clientX: number) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || points.length < 2) return;
    const rel = (clientX - rect.left) / rect.width;
    setHover(Math.max(0, Math.min(points.length - 1, Math.round(rel * (points.length - 1)))));
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') setHover((h) => Math.max(0, (h ?? lastKnownIndex ?? 0) - 1));
    else if (e.key === 'ArrowRight') setHover((h) => Math.min(points.length - 1, (h ?? lastKnownIndex ?? 0) + 1));
  };

  const hovered = hover != null ? points[hover] : null;

  return (
    <div className="rounded-xl border border-border bg-surface-2 p-3">
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted">{label}</p>
        {lastKnownIndex != null && (
          <p className="text-lg font-semibold tabular-nums">
            {(points[lastKnownIndex].value as number).toFixed(digits)}
            <span className="ml-0.5 text-xs font-normal text-muted">{unit}</span>
          </p>
        )}
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="mt-2 w-full touch-none focus-visible:outline-none"
        height={HEIGHT}
        role="img"
        tabIndex={0}
        aria-label={`${label} over the last ${points.length} days, current value ${lastKnownIndex != null ? points[lastKnownIndex].value : 'unknown'}${unit}`}
        onPointerMove={(e) => setHoverFromClientX(e.clientX)}
        onPointerLeave={() => setHover(null)}
        onFocus={() => setHover((h) => h ?? lastKnownIndex ?? 0)}
        onBlur={() => setHover(null)}
        onKeyDown={onKeyDown}
      >
        {segments.map((seg, i) => (
          <polyline key={i} points={seg} fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ stroke: 'rgb(var(--accent))' }} />
        ))}
        {lastKnownIndex != null && (
          <circle
            cx={toX(lastKnownIndex, points.length)}
            cy={toY(points[lastKnownIndex].value as number)}
            r="4"
            style={{ fill: 'rgb(var(--accent))', stroke: 'rgb(var(--surface-2))', strokeWidth: 2 }}
          />
        )}
        {hover != null && (
          <line x1={toX(hover, points.length)} x2={toX(hover, points.length)} y1={PAD} y2={HEIGHT - PAD} strokeWidth="1" style={{ stroke: 'rgb(var(--border))' }} />
        )}
        {hover != null && hovered?.value != null && (
          <circle
            cx={toX(hover, points.length)}
            cy={toY(hovered.value)}
            r="4"
            style={{ fill: 'rgb(var(--accent))', stroke: 'rgb(var(--surface-2))', strokeWidth: 2 }}
          />
        )}
      </svg>
      <div className="mt-1 flex justify-between text-[11px] text-muted" aria-hidden>
        <span>{points[0]?.day.slice(5)}</span>
        <span>{points[points.length - 1]?.day.slice(5)}</span>
      </div>
      <p role="status" className="mt-1 h-4 text-xs font-medium text-ink">
        {hovered ? `${hovered.day}: ${hovered.value != null ? `${hovered.value.toFixed(digits)}${unit}` : 'no data'}` : ' '}
      </p>
    </div>
  );
}
