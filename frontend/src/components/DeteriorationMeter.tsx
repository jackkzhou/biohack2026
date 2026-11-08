/** Meter spec: fill carries severity (ok -> warn -> danger); unfilled track stays neutral. */
export default function DeteriorationMeter({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const tone = value < 0.33 ? 'bg-ok' : value < 0.66 ? 'bg-warn' : 'bg-danger';
  return (
    <div className="flex items-center gap-3">
      <div
        className="h-3 flex-1 overflow-hidden rounded-full bg-border"
        role="meter"
        aria-valuenow={Math.round(value * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Deterioration index"
      >
        <div className={`h-full rounded-full transition-[width] ${tone}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-12 text-right text-sm font-semibold tabular-nums">{pct.toFixed(0)}%</span>
    </div>
  );
}
