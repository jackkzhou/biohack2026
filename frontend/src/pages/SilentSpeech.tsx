import { useEffect, useState } from 'react';
import { useLiveStream } from '../api/socket';
import LiveImuChart from '../components/LiveImuChart';
import { useMockScript } from '../mock/useMockScript';

export default function SilentSpeech() {
  useMockScript();
  const s = useLiveStream();

  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const wordCount = s.currentWords.length;
  useEffect(() => {
    if (wordCount > 0 || s.phraseCount > 0 || s.nothingCount > 0) setUpdatedAt(Date.now());
  }, [wordCount, s.phraseCount, s.nothingCount]);
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const ago = updatedAt === null ? null : Math.max(0, Math.round((now - updatedAt) / 1000));
  const agoText = ago === null ? '–' : ago < 60 ? `${ago}s ago` : `${Math.floor(ago / 60)}m ago`;

  const text = s.phrase?.sentence ?? s.currentWords.map((w) => w.word).join(' ');
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

      <section className="card flex min-h-[18rem] items-center justify-center text-center" aria-live="polite" aria-label="Predicted word">
        {text ? (
          <p className="text-6xl font-bold leading-tight sm:text-8xl">{text}</p>
        ) : s.nothing ? (
          <p className="text-4xl font-medium text-muted">Nothing detected</p>
        ) : (
          <p className="text-3xl font-medium text-muted">Mouth a word</p>
        )}
      </section>

      <LiveImuChart height={48} />
    </div>
  );
}
