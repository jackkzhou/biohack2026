export const MIN_CONFIDENCE = 0.7;

export default function WordChip({ word, confidence, size = 'md' }: { word: string; confidence: number; size?: 'md' | 'lg' }) {
  const low = confidence < MIN_CONFIDENCE;
  const pct = Math.round(confidence * 100);
  return (
    <div
      className={`pop-in rounded-2xl border-2 px-5 py-3 ${low ? 'border-border bg-surface-2 text-muted' : 'border-accent bg-accent/10 text-ink'} ${size === 'lg' ? 'min-w-[11rem]' : 'min-w-[7rem]'}`}
    >
      <div className={`font-bold leading-tight ${size === 'lg' ? 'text-5xl sm:text-6xl' : 'text-2xl'}`}>{word}</div>
      <div className="mt-3 flex items-center gap-2">
        <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-border" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} aria-label={`Confidence ${pct}%`}>
          <div className={`h-full rounded-full ${low ? 'bg-muted' : 'bg-accent'}`} style={{ width: `${pct}%` }} />
        </div>
        <span className="w-10 text-right text-sm font-semibold tabular-nums">{pct}%</span>
      </div>
      {low && <div className="mt-1 text-xs font-medium uppercase tracking-wide">Low confidence</div>}
    </div>
  );
}
