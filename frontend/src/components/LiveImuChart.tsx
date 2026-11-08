import { useCallback, useEffect, useRef, useState } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { useLiveStream } from '../api/socket';
import { useThemeColors, withAlpha, type ThemeColors } from '../hooks/useThemeColors';

const WINDOW_S = 10;
type Channel = 'gyro' | 'accel';

interface Buf {
  t: number[];
  g: number[][];
  a: number[][];
  latest: number;
}
const newBuf = (): Buf => ({ t: [], g: [[], [], []], a: [[], [], []], latest: 0 });

function Trace({
  sensor,
  label,
  channel,
  height,
  colors,
}: {
  sensor: number;
  label: string;
  channel: Channel;
  height: number;
  colors: ThemeColors;
}) {
  const { subscribeSamples } = useLiveStream();
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const buf = useRef<Buf>(newBuf());
  const channelRef = useRef(channel);
  channelRef.current = channel;
  const raf = useRef(0);

  const draw = useCallback(() => {
    raf.current = 0;
    const u = plot.current;
    if (!u) return;
    const b = buf.current;
    const c = channelRef.current === 'gyro' ? b.g : b.a;
    u.setData([b.t, c[0], c[1], c[2]] as uPlot.AlignedData, true);
  }, []);
  const schedule = useCallback(() => {
    if (!raf.current) raf.current = requestAnimationFrame(draw);
  }, [draw]);

  // (Re)build the plot when theme or size changes.
  useEffect(() => {
    const el = host.current!;
    const b = buf.current;
    const grid = withAlpha(colors.muted, 0.2);
    const axis = {
      stroke: colors.muted,
      grid: { stroke: grid, width: 1 },
      ticks: { stroke: grid, width: 1 },
      font: '13px ui-sans-serif, system-ui, sans-serif',
    };
    const line = (stroke: string): uPlot.Series => ({ stroke, width: 2, points: { show: false } });
    const u = new uPlot(
      {
        width: Math.max(el.clientWidth, 200),
        height,
        legend: { show: false },
        cursor: { show: false },
        select: { show: false, left: 0, top: 0, width: 0, height: 0 },
        scales: {
          x: { time: false, range: () => [b.latest / 1000 - WINDOW_S, b.latest / 1000] },
          y: {
            range: (_u, min, max) => {
              if (!isFinite(min) || !isFinite(max)) return [-1, 1];
              const mid = (min + max) / 2;
              const half = Math.max(((max - min) / 2) * 1.15, channelRef.current === 'gyro' ? 0.3 : 0.5);
              return [mid - half, mid + half];
            },
          },
        },
        axes: [
          {
            ...axis,
            space: 70,
            incrs: [1, 2, 5],
            values: (_u, splits) => splits.map((v) => `${Math.round(v - b.latest / 1000)}s`),
          },
          { ...axis, size: 52, values: (_u, splits) => splits.map((v) => v.toFixed(1)) },
        ],
        series: [{}, line(colors.accent), line(colors.s2), line(colors.s3)],
      },
      [[], [], [], []],
      el,
    );
    plot.current = u;
    const ro = new ResizeObserver(() => u.setSize({ width: Math.max(el.clientWidth, 200), height }));
    ro.observe(el);
    schedule();
    return () => {
      ro.disconnect();
      u.destroy();
      plot.current = null;
    };
  }, [colors, height, schedule]);

  // Feed samples into this trace's ring buffer.
  useEffect(
    () =>
      subscribeSamples((samples) => {
        const b = buf.current;
        let any = false;
        for (const s of samples) {
          if (s.sensor !== sensor) continue;
          if (s.t < b.latest - 1000) {
            // device clock restarted
            b.t.length = 0;
            b.g.forEach((x) => (x.length = 0));
            b.a.forEach((x) => (x.length = 0));
          }
          b.t.push(s.t / 1000);
          for (let i = 0; i < 3; i++) {
            b.g[i].push(s.g[i]);
            b.a[i].push(s.a[i]);
          }
          b.latest = s.t;
          any = true;
        }
        if (!any) return;
        const cutoff = (b.latest - WINDOW_S * 1000 - 500) / 1000;
        let k = 0;
        while (k < b.t.length && b.t[k] < cutoff) k++;
        if (k > 0) {
          b.t.splice(0, k);
          b.g.forEach((x) => x.splice(0, k));
          b.a.forEach((x) => x.splice(0, k));
        }
        schedule();
      }),
    [sensor, subscribeSamples, schedule],
  );

  useEffect(() => {
    schedule();
  }, [channel, schedule]);

  useEffect(
    () => () => {
      if (raf.current) cancelAnimationFrame(raf.current);
      raf.current = 0;
    },
    [],
  );

  return (
    <div>
      <div className="text-xs font-semibold text-muted">{label}</div>
      <div ref={host} className="w-full" role="img" aria-label={`${label} sensor, last ${WINDOW_S} seconds`} />
    </div>
  );
}

export default function LiveImuChart({ height = 130 }: { height?: number }) {
  const colors = useThemeColors();
  const [channel, setChannel] = useState<Channel>('gyro');
  const { link, streaming } = useLiveStream();

  return (
    <section className="card !p-3" aria-label="Live IMU signals">
      <div className="mb-1 flex justify-end gap-1" role="group" aria-label="Signal">
        {(['gyro', 'accel'] as const).map((c) => (
          <button key={c} className={`btn !min-h-[1.75rem] !px-3 !py-0 !text-xs ${channel === c ? 'btn-active' : ''}`} aria-pressed={channel === c} onClick={() => setChannel(c)}>
            {c === 'gyro' ? 'Gyro' : 'Accel'}
          </button>
        ))}
      </div>
      {link === 'open' && !streaming && (
        <p className="mb-2 text-sm font-medium text-warn">No samples.</p>
      )}
      <div className="grid gap-1">
        <Trace sensor={0} label="Chin" channel={channel} height={height} colors={colors} />
        <Trace sensor={1} label="Head" channel={channel} height={height} colors={colors} />
      </div>
    </section>
  );
}
