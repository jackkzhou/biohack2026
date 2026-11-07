import { useEffect } from 'react';
import { USE_MOCK } from '../config';
import { mockEngine } from './engine';

/** In mock mode, plays scripted word/phrase events while the calling page is mounted. */
export function useMockScript() {
  useEffect(() => {
    if (!USE_MOCK) return;
    mockEngine.setScriptEnabled(true);
    return () => mockEngine.setScriptEnabled(false);
  }, []);
}
