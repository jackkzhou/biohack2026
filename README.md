# biohack2026
# Silent Speech

**Say a sentence without making a sound, in your own voice.**

Face-mounted IMUs recognize silently mouthed words, chain them into a phrase, and speak it in the user's cloned voice. Built for people with ALS who are losing their speech.

## Modes

- **Mode 1: Learn.** While the user can still speak, Whisper transcribes their words and auto-labels the matching IMU data, building a personal dataset and banking their voice over time.
- **Mode 2: Silent Speech.** Mouth a word → the model predicts it → after a ~1.5 s pause, the phrase is spoken in the cloned voice (e.g. "help" + "Jordan" → "Jordan, I need help.").

## Architecture

```
ESP32 (C++) ──USB / Bluetooth──▶ Python backend (FastAPI) ──WebSocket──▶ React frontend
                                         │
                                         └──▶ Tiger Data (storage)
```

- **ESP32** reads two BNO085 IMUs and streams readings as text lines.
- **Backend** reads the stream, runs inference, calls Whisper and ElevenLabs, and saves to Tiger Data.
- **Frontend** shows live data, predictions and results. It only talks to the backend.

## Hardware

ESP32 DevKit + 2× Adafruit BNO085 (chin = articulation, behind ear = head-motion reference).

| | Jaw IMU (sensor 0) | Head IMU (sensor 1, still) |
|---|---|---|
| Power / GND | 3V3 / GND (via rails) | 3V3 / GND (via rails) |
| SDA / SCL | D18 / D19 | D21 / D22 |

## Repo

```
firmware/   ESP32 sketch
backend/    FastAPI app (reader, inference, TTS, database)
ml/         recording, training, evaluation
frontend/   React app
```

## Model

5 words (help, yes, no, Jordan, water) + a "nothing" class, ~40 mouthed samples each. Models compared: DTW, Random Forest, MiniRocket, 1D CNN. Evaluated on a held-out session with re-mounted sensors.

## Run

```bash
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env   # set FAKE_STREAM=true to run without hardware
uvicorn backend.main:app --reload      # run from the repo root
cd frontend && npm install && npm run dev
```

### Collect a dataset

Flash the firmware (`cd firmware && pio run -t upload`), set `SERIAL_PORT` and `TIGER_DB_URL` in `backend/.env`, stop the uvicorn server (only one program can open the serial port), then:

```bash
python -m backend.collect --labels help yes no Jordan water nothing
```

Type a label, press Enter to start, speak/mouth the word, the take stops by itself after 3 seconds. Each take (accel + gyro, both IMUs, ~200 Hz) is saved to `data/clips/*.csv` and sent to Tiger Data. If the database was unreachable, run `python -m backend.collect --upload` later; it skips clips already stored.

Set `SERIAL_PORT`, `TIGER_DB_URL`, `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` in `backend/.env`.

## Team

- **A:** hardware + firmware
- **B:** ML + backend
- **C:** frontend + voice

---

## Ingestion API & ML pipeline (teammate)

# biohack2026

## Tiger Data sensor ingestion

Create the TimescaleDB table and hypertable with `db/schema.sql`. Configure
`DATABASE_URL` with the Tiger Data PostgreSQL connection URL, then install
dependencies and start the ingestion API:

```bash
python -m pip install -r requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Send a batch to `POST /stream` with timezone-aware ISO 8601 timestamps:

```json
{
  "session_id": "session-001",
  "label": "hello",
  "samples": [
    {
      "time": "2026-11-07T18:35:00Z",
      "ax": 0.12,
      "ay": -0.03,
      "az": 9.81
    }
  ]
}
```

`ml.db_loader.load_session_data(db_url, session_id, window_size)` returns an
`(X, y)` tuple. `X` has shape `(windows, window_size, 3)` for
`fit_random_forest`, `fit_minirocket_ridge`, and the other time-series models;
windows overlap by one row and do not cross label changes.

## Time-series models

Each model can be imported and tested independently. The modules share
`ml/time_series_utils.py` for input validation and reusable feature helpers:

- `ml/dtw.py`
- `ml/random_forest.py`
- `ml/minirocket.py`
- `ml/cnn.py`

For Colab, upload the chosen model file and `time_series_utils.py` into the same
directory, then import that model (for example, `from random_forest import
fit_random_forest`). Existing imports from
`ml.time_series_classification` remain supported.
