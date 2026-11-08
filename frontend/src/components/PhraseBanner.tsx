export default function PhraseBanner({
  words,
  sentence,
  speaking,
  muted,
  large = false,
}: {
  words: string[];
  sentence: string | null;
  speaking: boolean;
  muted: boolean;
  large?: boolean;
}) {
  if (!sentence && !words.length) return null;
  return (
    <section className="card text-center" aria-live="polite" aria-label="Spoken sentence">
      {sentence ? (
        <>
          <p className="text-base font-medium text-muted">{words.join(' + ')}</p>
          <p className={`mt-1 font-bold leading-tight ${large ? 'text-4xl sm:text-6xl' : 'text-3xl sm:text-4xl'}`}>“{sentence}”</p>
          <div className="mt-4 flex items-center justify-center gap-3 text-base font-semibold">
            {muted ? (
              <span className="text-muted">Voice muted</span>
            ) : speaking ? (
              <>
                <span aria-hidden className="flex h-6 items-end gap-1">
                  {[0, 0.15, 0.3, 0.1].map((d) => (
                    <span key={d} className="bar-bounce h-6 w-1.5 rounded bg-accent" style={{ animationDelay: `${d}s` }} />
                  ))}
                </span>
                <span className="text-accent">Speaking in your voice…</span>
              </>
            ) : (
              <span className="text-muted">Spoken</span>
            )}
          </div>
        </>
      ) : (
        <p className={`font-medium text-muted ${large ? 'text-2xl' : 'text-xl'}`}>
          Building phrase: {words.join(' + ')} …
        </p>
      )}
    </section>
  );
}
