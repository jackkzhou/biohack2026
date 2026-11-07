import type { ImuSample } from '../types';

/** Tiny SVG plot of gyro magnitude for the chin (accent) and head (amber) sensors. */
export default function Sparkline({ samples, width = 150, height = 36 }: { samples: ImuSample[]; width?: number; height?: number }) {
  const bySensor = [0, 1].map((s) => samples.filter((x) => x.sensor === s));
  const mag = (x: ImuSample) => Math.hypot(x.g[0], x.g[1], x.g[2]);
  const max = Math.max(0.5, ...samples.map(mag));
  const toPoints = (xs: ImuSample[]) =>
    xs.map((x, i) => `${((i / Math.max(1, xs.length - 1)) * width).toFixed(1)},${(height - 2 - (mag(x) / max) * (height - 4)).toFixed(1)}`).join(' ');
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Saved IMU window" className="shrink-0 rounded bg-surface-2">
      <polyline points={toPoints(bySensor[0])} fill="none" strokeWidth="1.5" style={{ stroke: 'rgb(var(--accent))' }} />
      <polyline points={toPoints(bySensor[1])} fill="none" strokeWidth="1.5" style={{ stroke: 'rgb(var(--s2))' }} />
    </svg>
  );
}
