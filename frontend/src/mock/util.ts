export const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
export const jitter = (min: number, max: number) => min + Math.random() * (max - min);
export const pick = <T,>(xs: readonly T[]): T => xs[Math.floor(Math.random() * xs.length)];
