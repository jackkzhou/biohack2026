import type { ImuSample } from '../types';

const TAU = Math.PI * 2;
const FREQ = [
  [2.1, 3.4, 1.7],
  [2.7, 1.9, 3.1],
];

const noise = (s: number) => (Math.random() - 0.5) * 2 * s;
const r3 = (v: number) => Math.round(v * 1000) / 1000;

/** One fake IMU sample: sine waves + noise. `activity` 0..1 scales the motion. */
export function imuSampleAt(tMs: number, sensor: number, activity: number): ImuSample {
  const t = tMs / 1000;
  const f = FREQ[sensor % 2];
  const g = [0, 1, 2].map(
    (i) =>
      r3(noise(0.03) + 0.04 * Math.sin(TAU * 0.3 * t + i) + activity * (1.6 + 0.4 * i) * Math.sin(TAU * f[i] * t + sensor + i)),
  ) as [number, number, number];
  const a = [0, 1, 2].map((i) =>
    r3((i === 2 ? 9.81 : 0) + noise(0.05) + activity * 1.2 * Math.sin(TAU * f[(i + 1) % 3] * t + i * 1.3 + sensor)),
  ) as [number, number, number];
  return { t: Math.round(tMs), sensor, g, a };
}

/** A window of both sensors with a smooth rise/fall envelope (like one mouthed word). */
export function synthWindow(startT: number, durationMs: number, intensity: number, hz = 100): ImuSample[] {
  const n = Math.max(2, Math.round((durationMs / 1000) * hz));
  const out: ImuSample[] = [];
  for (let k = 0; k < n; k++) {
    const t = startT + (k * 1000) / hz;
    const env = Math.pow(Math.sin((Math.PI * k) / (n - 1)), 2) * intensity;
    out.push(imuSampleAt(t, 0, env), imuSampleAt(t, 1, env * 0.6));
  }
  return out;
}
