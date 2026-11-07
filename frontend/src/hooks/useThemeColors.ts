import { useEffect, useState } from 'react';

const NAMES = ['bg', 'surface', 'border', 'ink', 'muted', 'accent', 's2', 's3', 'ok', 'warn', 'danger'] as const;
export type ThemeColors = Record<(typeof NAMES)[number], string>;

function read(): ThemeColors {
  const cs = getComputedStyle(document.documentElement);
  const out = {} as ThemeColors;
  for (const n of NAMES) {
    const channels = cs.getPropertyValue(`--${n}`).trim().split(/\s+/).join(', ');
    out[n] = `rgb(${channels})`;
  }
  return out;
}

export const withAlpha = (rgb: string, a: number) => rgb.replace('rgb(', 'rgba(').replace(')', `, ${a})`);

/** Resolved CSS colours for canvas/SVG libraries; refreshes when light/dark flips. */
export function useThemeColors(): ThemeColors {
  const [colors, setColors] = useState(read);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => setColors(read());
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return colors;
}
