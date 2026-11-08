import { useLiveStream } from '../api/socket';
import { USE_MOCK } from '../config';

type Tone = 'ok' | 'warn' | 'danger';
const toneText: Record<Tone, string> = { ok: 'text-ok', warn: 'text-warn', danger: 'text-danger' };
const toneDot: Record<Tone, string> = { ok: 'bg-ok', warn: 'bg-warn', danger: 'bg-danger' };

export default function ConnectionStatus() {
  const s = useLiveStream();
  const [tone, text]: [Tone, string] =
    s.link === 'connecting'
      ? ['warn', 'Connecting…']
      : s.link !== 'open'
        ? ['danger', 'Offline']
        : !s.streaming || !s.sensors.connected
          ? ['warn', 'No sensors']
          : ['ok', 'Live'];
  return (
    <span className="inline-flex items-center gap-2 text-sm font-medium" role="status" aria-live="polite">
      <span aria-hidden className={`h-2.5 w-2.5 rounded-full ${toneDot[tone]}`} />
      <span className={toneText[tone]}>{text}</span>
      {USE_MOCK && <span className="badge">Demo</span>}
    </span>
  );
}
