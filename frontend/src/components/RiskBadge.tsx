import type { RiskScore } from '../types';

/** low/moderate reuse the ok/warn tokens; elevated/high share the danger hue but are told
 * apart by weight (outline vs filled), never by hue alone (icon + label stays colorblind-safe). */
const STYLE: Record<RiskScore, string> = {
  low: 'border-ok bg-ok/10 text-ok',
  moderate: 'border-warn bg-warn/10 text-warn',
  elevated: 'border-danger bg-transparent text-danger',
  high: 'border-danger bg-danger text-accent-ink',
};

const DOT: Record<RiskScore, string> = {
  low: 'bg-ok',
  moderate: 'bg-warn',
  elevated: 'bg-danger',
  high: 'bg-accent-ink',
};

const LABEL: Record<RiskScore, string> = {
  low: 'Low risk',
  moderate: 'Moderate risk',
  elevated: 'Elevated risk',
  high: 'High risk',
};

export default function RiskBadge({ risk }: { risk: RiskScore }) {
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm font-semibold uppercase tracking-wide ${STYLE[risk]}`}>
      <span className={`h-2 w-2 rounded-full ${DOT[risk]}`} aria-hidden />
      {LABEL[risk]}
    </span>
  );
}
