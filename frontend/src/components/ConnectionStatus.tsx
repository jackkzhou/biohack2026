import { useLiveStream } from '../api/socket';
import { USE_MOCK } from '../config';

type Tone = 'ok' | 'warn' | 'danger' | 'idle';
const toneText: Record<Tone, string> = { ok: 'text-ok', warn: 'text-warn', danger: 'text-danger', idle: 'text-muted' };
const toneDot: Record<Tone, string> = { ok: 'bg-ok', warn: 'bg-warn', danger: 'bg-danger', idle: 'bg-muted' };

function Pill({ tone, children }: { tone: Tone; children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-3 py-1 text-sm font-medium">
      <span aria-hidden className={`h-2.5 w-2.5 rounded-full ${toneDot[tone]}`} />
      <span className={toneText[tone]}>{children}</span>
    </span>
  );
}

export default function ConnectionStatus() {
  const s = useLiveStream();
  const backend: [Tone, string] =
    s.link === 'open' ? ['ok', 'Backend connected'] : s.link === 'connecting' ? ['warn', 'Connecting…'] : ['danger', 'Backend disconnected'];

  let sensors: React.ReactNode;
  if (s.link !== 'open') {
    sensors = <Pill tone="idle">Sensors unknown</Pill>;
  } else if (!s.streaming || !s.sensors.connected) {
    sensors = <Pill tone="warn">Sensors not streaming</Pill>;
  } else {
    const [chin, head] = s.sensors.rate_hz;
    sensors = (
      <>
        <Pill tone="ok">Chin {chin ?? '–'} Hz</Pill>
        <Pill tone="ok">Head {head ?? '–'} Hz</Pill>
        <Pill tone={s.sensors.dropped > 0 ? 'warn' : 'ok'}>Dropped {s.sensors.dropped}</Pill>
      </>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2" role="status" aria-live="polite">
      <Pill tone={backend[0]}>{backend[1]}</Pill>
      {sensors}
      {USE_MOCK && <span className="badge">Demo data</span>}
    </div>
  );
}
