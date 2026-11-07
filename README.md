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

| | Chin IMU | Head IMU |
|---|---|---|
| Power / GND | 3V3 / GND (via rails) | 3V3 / GND (via rails) |
| SDA / SCL | D21 / D22 | D18 / D19 |

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

Set `SERIAL_PORT`, `TIGER_DB_URL`, `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` in `backend/.env`.

## Team

- **A:** hardware + firmware
- **B:** ML + backend
- **C:** frontend + voice