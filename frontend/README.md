# Silent Speech frontend

Vite + React + TypeScript + Tailwind. Live sensor charts use uPlot; static charts use Recharts.
Talks only to the FastAPI backend, never to the ESP32 or the database. Contains no keys or credentials.

```
npm install
cp .env.example .env     # optional: the defaults already run in mock mode
npm run dev              # http://localhost:5173
```

## Mock vs real backend

| Variable | Default | |
|---|---|---|
| `VITE_USE_MOCK` | `true` | `true` = everything comes from `src/mock/`. `false` = real backend. |
| `VITE_API_URL` | `http://localhost:8000` | REST base URL |
| `VITE_WS_URL` | `ws://localhost:8000/ws` | WebSocket URL |

Restart `npm run dev` after changing `.env`. No code changes are needed to switch.

## Backend contract

WebSocket messages (`/ws`), as specified: `samples`, `word`, `pause`, `phrase`, `speaking`, `status`, and optionally `audio {url}`.
`samples[].t` is treated as **milliseconds**; `sensor` 0 = chin, 1 = head; `g` = gyro rad/s, `a` = accel m/s².

REST, as specified:
`POST /learn/utterance` (multipart `audio`, `started_at_ms`), `GET /clips/{id}`, `DELETE /clips/{id}`,
`POST /record/rest-pose`, `POST /record/start {label, session}`, `POST /record/stop`,
`GET /history/phrases`, `GET /model/metrics`.

**Assumed shapes and endpoints the brief did not define. Please confirm or adjust in `src/api/client.ts`:**

- `GET /clips/{id}` -> `{ id, word, samples: [{t, sensor, g, a}] }`
- `POST /record/stop` -> `{ clip_id, label }` (needed so "Redo last" can `DELETE /clips/{id}`)
- `POST /record/rest-pose` body `{ session }`
- `GET /record/counts` -> `{ "help": 12, ... }` (hydrates the progress grid; failure is ignored)
- `GET /history/phrases` -> `[{ id, words, sentence, at_ms }]`, newest first
- `POST /phrase/clear`, `POST /voice/mute {muted}`, `POST /phrasing/mode {mode: "lookup" | "llm"}` (failures are ignored)

## Structure

`src/api` REST client + `useLiveStream()` (one shared WebSocket, auto-reconnect) · `src/mock` fake WebSocket/REST · `src/components` · `src/pages`

## Notes

- Raw samples bypass React state: they go into ring buffers and uPlot redraws at most once per animation frame.
- In mock mode the finished phrase is also read aloud with the browser's speech synthesis, and Learn records your real mic but returns a simulated transcript.
- Light/dark follows the OS setting.
