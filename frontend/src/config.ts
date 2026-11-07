const env = import.meta.env;

/** Anything except the literal string "false" means mock mode. */
export const USE_MOCK = (env.VITE_USE_MOCK ?? 'true') !== 'false';
export const API_URL = (env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');
export const WS_URL = env.VITE_WS_URL ?? 'ws://localhost:8000/ws';
