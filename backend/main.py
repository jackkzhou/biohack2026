"""Silent Speech backend. Run from the repo root:  uvicorn backend.main:app --reload

REST routes and WebSocket messages mirror the frontend contract in
frontend/src/types.ts and frontend/src/api/client.ts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config import settings
from backend.db import Database
from backend.fake_stream import WORDS, FakeStream
from backend.inference import GroundTruthClassifier, Inference, ModelClassifier, load_phrases
from backend.reader import Reader
from backend.recorder import Recorder, RecorderError, Transcriber
from backend.tts import TTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("main")

VOCAB = ["help", "yes", "no", "Jordan", "water"]
SAMPLES_HZ = 25


class Hub:
    """WebSocket clients. A slow or dead client is dropped, never waited on for long."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def broadcast(self, msg: dict) -> None:
        if not self.clients:
            return
        text = json.dumps(msg, separators=(",", ":"))

        async def one(ws: WebSocket) -> None:
            try:
                await asyncio.wait_for(ws.send_text(text), 1.0)
            except Exception:  # noqa: BLE001
                self.clients.discard(ws)

        await asyncio.gather(*(one(ws) for ws in list(self.clients)))


class State:
    muted = False
    history: list[dict] = []


hub = Hub()
state = State()
reader = Reader(settings.serial_port, settings.serial_baud, settings.fake_stream)
db = Database(settings.tiger_db_url, settings.store_raw_stream)
tts = TTS(settings.elevenlabs_api_key, settings.elevenlabs_voice_id, settings.elevenlabs_model)
fake = FakeStream(reader) if settings.fake_stream else None
recorder = Recorder(reader, db, settings.data_dir, Transcriber(settings.whisper_model, settings.whisper_language), VOCAB,
                    settings.learn_pad_ms, settings.learn_audio_offset_ms)
inference: Inference  # built in lifespan (needs the running loop for its callbacks)


async def on_phrase(words: list[str], sentence: str) -> None:
    rec = {"id": int(time.time() * 1000), "words": words, "sentence": sentence, "at_ms": int(time.time() * 1000)}
    state.history.insert(0, rec)
    del state.history[200:]
    db.save_phrase(rec)
    if state.muted:
        return
    job = tts.speak(sentence)
    if not job:
        return
    await hub.broadcast({"type": "audio", "url": f"{settings.public_base_url}/tts/{job.id}.mp3"})
    await hub.broadcast({"type": "speaking", "active": True})

    async def finish() -> None:
        await job.wait_done()
        # The browser plays the stream itself, so estimate when it ends from the audio length.
        remaining = tts.playback_seconds(job) - (time.monotonic() - job.started)
        await asyncio.sleep(max(0.0, remaining) + 0.2)
        await hub.broadcast({"type": "speaking", "active": False})

    asyncio.create_task(finish())


async def sample_pump() -> None:
    """~25 Hz: newest sample of each sensor goes to the browser; the rest stay in the buffer/DB."""
    while True:
        await asyncio.sleep(1 / SAMPLES_HZ)
        items = reader.drain_outbox()
        if not items or not hub.clients:
            continue
        latest = {}
        for t_ms, sensor, g, a in items:
            latest[sensor] = {"t": round(t_ms, 1), "sensor": sensor, "g": [round(v, 4) for v in g], "a": [round(v, 4) for v in a]}
        await hub.broadcast({"type": "samples", "data": [latest[s] for s in sorted(latest)]})


def status_message() -> dict:
    return {"type": "status", "connected": reader.connected, "rate_hz": reader.rates(), "dropped": reader.dropped}


async def status_pump() -> None:
    while True:
        await asyncio.sleep(1.0)
        await hub.broadcast(status_message())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference
    await db.start()
    await tts.start()
    reader.add_sink(db.push_sample)
    reader.start()

    classifier = None
    try:
        classifier = ModelClassifier.load(settings.models_dir)
    except Exception:  # noqa: BLE001
        log.exception("could not load the model from backend/models/")
    if classifier is None and fake:
        log.warning("no trained model: using the TEST-ONLY ground-truth classifier on the fake stream")
        classifier = GroundTruthClassifier(fake)
    elif classifier is None:
        log.warning("no trained model in backend/models/: words will not be detected")

    inference = Inference(
        reader, classifier, hub.broadcast, on_phrase, load_phrases(settings.models_dir),
        conf_threshold=settings.conf_threshold, pause_s=settings.phrase_pause_s,
        activity_threshold=settings.activity_threshold, on_detection=db.save_detection,
    )
    recorder.on_rest_noise = inference.calibrate
    if fake:
        def on_start(label: str) -> None:
            nothing = label == "nothing"
            fake.burst(random.choice(WORDS) if nothing else label, delay=0.3, duration=1.0, intensity=0.35 if nothing else 1.0)
        recorder.on_record_start = on_start

    tasks = [asyncio.create_task(c) for c in (inference.run(), sample_pump(), status_pump())]
    if fake:
        tasks.append(asyncio.create_task(fake.run()))
    log.info("backend ready (fake=%s, db=%s, tts=%s, model=%s)", bool(fake), db.enabled, tts.enabled, inference.model_loaded)
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        reader.stop()
        await tts.stop()
        await db.stop()


app = FastAPI(title="Silent Speech", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(RecorderError)
async def recorder_error(_, e: RecorderError):
    from fastapi.responses import JSONResponse

    return JSONResponse({"detail": str(e)}, status_code=e.status)


# ---- WebSocket -------------------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    hub.clients.add(ws)
    try:
        await ws.send_text(json.dumps(status_message()))
        while True:
            await ws.receive_text()  # the frontend sends nothing; this just detects disconnects
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)


# ---- REST ------------------------------------------------------------------
class SessionBody(BaseModel):
    session: int


class StartBody(BaseModel):
    label: str
    session: int


class MuteBody(BaseModel):
    muted: bool


class ModeBody(BaseModel):
    mode: Literal["lookup", "llm"]


@app.get("/health")
async def health() -> dict:
    """Booleans only: no secrets, no configuration values."""
    return {"sensors": reader.connected, "fake_stream": bool(fake), "database": db.online, "speech": tts.enabled,
            "model": inference.model_loaded}


@app.post("/learn/utterance")
async def learn_utterance(audio: UploadFile = File(...), started_at_ms: float = Form(...)) -> dict:
    suffix = Path(audio.filename or "").suffix or ".webm"
    return await recorder.learn_utterance(await audio.read(), suffix, started_at_ms)


@app.get("/clips/{clip_id}")
async def get_clip(clip_id: int) -> dict:
    return await recorder.get_clip(clip_id)


@app.delete("/clips/{clip_id}", status_code=204)
async def delete_clip(clip_id: int) -> Response:
    recorder.delete_clip(clip_id)
    return Response(status_code=204)


@app.post("/record/rest-pose", status_code=204)
async def rest_pose(body: SessionBody) -> Response:
    await recorder.rest_pose(body.session)
    return Response(status_code=204)


@app.post("/record/start", status_code=204)
async def record_start(body: StartBody) -> Response:
    if not reader.connected:
        raise HTTPException(503, "sensors are not streaming")
    recorder.start(body.label, body.session)
    return Response(status_code=204)


@app.post("/record/stop")
async def record_stop() -> dict:
    return recorder.stop()


@app.get("/record/counts")
async def record_counts() -> dict:
    return await recorder.counts()


@app.get("/history/phrases")
async def history() -> list[dict]:
    rows = await db.history()
    return rows if rows is not None else state.history[:50]


@app.post("/phrase/clear", status_code=204)
async def phrase_clear() -> Response:
    inference.clear()
    return Response(status_code=204)


@app.post("/voice/mute", status_code=204)
async def voice_mute(body: MuteBody) -> Response:
    state.muted = body.muted
    return Response(status_code=204)


@app.post("/phrasing/mode", status_code=204)
async def phrasing_mode(body: ModeBody) -> Response:
    inference.mode = body.mode  # 'llm' is accepted; phrases still come from the lookup table
    return Response(status_code=204)


@app.get("/model/metrics")
async def model_metrics() -> dict:
    p = settings.models_dir / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "no metrics.json in backend/models/ yet")
    return json.loads(p.read_text())


@app.get("/tts/{job_id}.mp3")
async def tts_audio(job_id: int) -> StreamingResponse:
    job = tts.get(job_id)
    if not job:
        raise HTTPException(404, "unknown or expired audio")
    return StreamingResponse(job.stream(), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})
