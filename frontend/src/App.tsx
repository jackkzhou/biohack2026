import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom';
import LiveStreamProvider from './api/LiveStreamProvider';
import ConnectionStatus from './components/ConnectionStatus';
import Home from './pages/Home';
import Learn from './pages/Learn';
import Record from './pages/Record';
import Results from './pages/Results';
import SilentSpeech from './pages/SilentSpeech';

const links = [
  ['/', 'Home'],
  ['/learn', 'Learn'],
  ['/record', 'Record'],
  ['/silent-speech', 'Silent Speech'],
  ['/results', 'Results'],
] as const;

export default function App() {
  return (
    <BrowserRouter>
      <LiveStreamProvider>
        <div className="flex min-h-screen flex-col">
          <header className="border-b border-border bg-surface">
            <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-3 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
              <nav aria-label="Main" className="flex flex-wrap items-center gap-1">
                <span className="mr-3 text-lg font-bold text-accent">Silent Speech</span>
                {links.map(([to, label]) => (
                  <NavLink
                    key={to}
                    to={to}
                    end={to === '/'}
                    className={({ isActive }) =>
                      `rounded-lg px-4 py-2.5 text-base font-semibold transition-colors ${isActive ? 'bg-accent text-accent-ink' : 'text-ink hover:bg-surface-2'}`
                    }
                  >
                    {label}
                  </NavLink>
                ))}
              </nav>
              <ConnectionStatus />
            </div>
          </header>
          <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/learn" element={<Learn />} />
              <Route path="/record" element={<Record />} />
              <Route path="/silent-speech" element={<SilentSpeech />} />
              <Route path="/results" element={<Results />} />
              <Route path="*" element={<Home />} />
            </Routes>
          </main>
        </div>
      </LiveStreamProvider>
    </BrowserRouter>
  );
}
