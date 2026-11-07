import type { ReactNode } from 'react';
import { LiveContext, useLiveSocket } from './socket';

export default function LiveStreamProvider({ children }: { children: ReactNode }) {
  const value = useLiveSocket();
  return <LiveContext.Provider value={value}>{children}</LiveContext.Provider>;
}
