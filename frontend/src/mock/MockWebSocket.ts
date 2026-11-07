import { mockEngine } from './engine';

/** Just enough of the WebSocket API for the app; replays mockEngine messages as JSON. */
export class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState = 0;
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;

  private unsub?: () => void;
  private openTimer: number;

  constructor(_url: string) {
    this.openTimer = window.setTimeout(() => {
      this.readyState = 1;
      this.onopen?.(new Event('open'));
      this.unsub = mockEngine.subscribe((msg) => {
        this.onmessage?.({ data: JSON.stringify(msg) } as MessageEvent);
      });
    }, 300);
  }

  send(_data: string) {}

  close() {
    clearTimeout(this.openTimer);
    this.unsub?.();
    this.unsub = undefined;
    if (this.readyState !== 3) {
      this.readyState = 3;
      this.onclose?.(new CloseEvent('close'));
    }
  }
}
