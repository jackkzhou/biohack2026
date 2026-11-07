export default function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const total = (row: number[]) => row.reduce((a, b) => a + b, 0) || 1;
  return (
    <div className="overflow-x-auto">
      <table className="mx-auto border-separate border-spacing-1 text-center" aria-label="Confusion matrix: rows are true labels, columns are predicted labels">
        <thead>
          <tr>
            <th />
            <th colSpan={labels.length} className="pb-1 text-sm font-semibold text-muted">
              Predicted →
            </th>
          </tr>
          <tr>
            <th className="pr-2 text-left text-sm font-semibold text-muted">True ↓</th>
            {labels.map((l) => (
              <th key={l} scope="col" className="px-1 pb-1 text-sm font-semibold">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th scope="row" className="pr-2 text-left text-sm font-semibold">
                {labels[i]}
              </th>
              {row.map((v, j) => {
                const frac = v / total(row);
                const diag = i === j;
                return (
                  <td
                    key={j}
                    className={`h-14 min-w-[3.5rem] rounded-lg text-lg font-bold tabular-nums sm:h-16 sm:min-w-[4.5rem] ${diag ? 'ring-2 ring-accent' : ''}`}
                    style={{
                      background: v === 0 ? 'rgb(var(--surface-2))' : `rgb(var(${diag ? '--accent' : '--danger'}) / ${0.12 + frac * 0.78})`,
                      color: frac > 0.5 ? 'rgb(var(--accent-ink))' : 'rgb(var(--ink))',
                    }}
                    title={`True ${labels[i]}, predicted ${labels[j]}: ${v}`}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
