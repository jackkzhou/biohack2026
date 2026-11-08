import { useEffect, useState } from 'react';
import { useLiveStream } from '../api/socket';
import LiveImuChart from '../components/LiveImuChart';
import { useMockScript } from '../mock/useMockScript';

const HOLD_MS = 5000; // how long a result stays on screen before the page goes back to "Nothing detected"

export default function SilentSpeech() {
  useMockScript();
  const s = useLiveStream();

  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const [wordAt, setWordAt] = useState<number | null>(null); // last time a word or phrase appeared
  const [now, setNow] = useState(Date.now());
  const wordCount = s.currentWords.length;
  useEffect(() => {
    if (wordCount > 0 || s.phraseCount > 0 || s.nothingCount > 0) setUpdatedAt(Date.now());
  }, [wordCount, s.phraseCount, s.nothingCount]);
  useEffect(() => {
    if (wordCount > 0 || s.phraseCount > 0) setWordAt(Date.now());
  }, [wordCount, s.phraseCount]);
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, []);
  const ago = updatedAt === null ? null : Math.max(0, Math.round((now - updatedAt) / 1000));
  const agoText = ago === null ? '–' : ago < 60 ? `${ago}s ago` : `${Math.floor(ago / 60)}m ago`;

  const lastWord = s.currentWords[s.currentWords.length - 1];
  const fresh = wordAt !== null && now - wordAt < HOLD_MS;
  const show = fresh || s.speaking;
  const word = show ? (s.currentWords.length ? s.currentWords.map((w) => w.word).join(' + ') : (s.phrase?.words.join(' + ') ?? '')) : '';
  const confidence = show && lastWord ? lastWord.confidence : null;
  const recentGuess = s.guess && now - s.guess.at < HOLD_MS ? s.guess : null;
  const idle = wordAt !== null || s.nothing; // something was shown before, or the model said "nothing"
  const problem =
    s.link !== 'open'
      ? s.link === 'connecting'
        ? { text: 'Connecting…', tone: 'warn' }
        : { text: 'Backend disconnected.', tone: 'danger' }
      : !s.streaming
        ? { text: 'Sensors not streaming.', tone: 'warn' }
        : null;

  return (
    <div className="space-y-5">
      <p className="text-center text-sm font-medium text-muted">Last updated: {agoText}</p>

      {problem && (
        <p role="alert" className={`rounded-xl border p-3 text-lg font-semibold ${problem.tone === 'danger' ? 'border-danger bg-danger/10 text-danger' : 'border-warn bg-warn/10 text-warn'}`}>
          {problem.text}
        </p>
      )}

      <section className="card flex min-h-[18rem] flex-col items-center justify-center gap-3 text-center" aria-live="polite" aria-label="Predicted word">
        {word ? (
          <>
            <p className="text-6xl font-bold leading-tight sm:text-8xl">{word}</p>
            {confidence !== null && <p className="text-2xl font-semibold tabular-nums text-accent">{Math.round(confidence * 100)}%</p>}
          </>
        ) : idle ? (
          <>
            <p className="text-4xl font-medium text-muted">Nothing detected</p>
            {recentGuess && (
              <p className="text-base tabular-nums text-muted">
                heard: {recentGuess.label} {Math.round(recentGuess.confidence * 100)}%
              </p>
            )}
          </>
        ) : (
          <p className="text-3xl font-medium text-muted">Mouth a word</p>
        )}
      </section>

      <LiveImuChart height={48} />
    </div>
  );
}
