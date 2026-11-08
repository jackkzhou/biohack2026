import { BrowserRouter, Navigate, NavLink, Route, Routes } from 'react-router-dom';
import LiveStreamProvider from './api/LiveStreamProvider';
import ConnectionStatus from './components/ConnectionStatus';
import TareButton from './components/TareButton';
import Learn from './pages/Learn';
import SilentSpeech from './pages/SilentSpeech';

const icon = 'h-6 w-6';
const MicIcon = () => (
  <svg className={icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
  </svg>
);
const WaveIcon = () => (
  <svg className={icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M4 12v0M8 8v8M12 4v16M16 8v8M20 12v0" />
  </svg>
);

const tabs = [
  ['/learn', 'Train', MicIcon],
  ['/silent-speech', 'Speech', WaveIcon],
] as const;

export default function App() {
  return (
    <BrowserRouter>
      <LiveStreamProvider>
        <div className="flex min-h-screen flex-col">
          <div className="flex items-center justify-center gap-4 px-4 pt-3">
            <ConnectionStatus />
            <TareButton />
          </div>
          <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-6 pb-28 sm:px-6">
            <Routes>
              <Route path="/learn" element={<Learn />} />
              <Route path="/silent-speech" element={<SilentSpeech />} />
              <Route path="*" element={<Navigate to="/learn" replace />} />
            </Routes>
          </main>
          <nav aria-label="Main" className="fixed inset-x-0 bottom-0 z-10 border-t border-border bg-surface pb-[env(safe-area-inset-bottom)]">
            <div className="mx-auto flex max-w-md justify-center gap-2 px-4 py-2">
              {tabs.map(([to, label, Icon]) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) =>
                    `flex flex-1 flex-col items-center gap-1 rounded-xl py-2 text-sm font-semibold transition-colors ${isActive ? 'text-accent' : 'text-muted hover:text-ink'}`
                  }
                >
                  <Icon />
                  {label}
                </NavLink>
              ))}
            </div>
          </nav>
        </div>
      </LiveStreamProvider>
    </BrowserRouter>
  );
}
