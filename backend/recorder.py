"""Labelled clips: manual recording (Record page) and Learn mode (Whisper auto-labelling).

Every clip is written to a local CSV first (always works, survives a database
outage), kept in a small in-memory cache, and queued for Tiger Data.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from backend.db import Database
from backend.inference import activity_level
from backend.reader import SENSOR_CHIN, SENSOR_HEAD, Reader

log = logging.getLogger("recorder")

CSV_HEADER = "t,t_esp,gx,gy,gz,ax,ay,az,qw,qx,qy,qz,sensor"
MIN_ROWS_PER_SENSOR = 5
REST_SECONDS = 2.0
CACHE_CLIPS = 200


class RecorderError(Exception):
    """Carries an HTTP status for main.py."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", label)[:40] or "x"


class Transcriber:
    """Lazy local Whisper (faster-whisper) with word timestamps."""

    def __init__(self, model: str, language: str) -> None:
        self.model_name, self.language = model, language
        self._model: Any = None
        self._lock = asyncio.Lock()

    def _load(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise RecorderError("faster-whisper is not installed (pip install faster-whisper)", 503) from e
            log.info("loading Whisper model %s", self.model_name)
            self._model = WhisperModel(self.model_name, compute_type="int8")
        return self._model

    def _run(self, path: str) -> tuple[str, list[tuple[str, float, float]]]:
        segments, _ = self._load().transcribe(path, word_timestamps=True, language=self.language or None, vad_filter=True)
        text, words = [], []
        for seg in segments:
            text.append(seg.text.strip())
            for w in seg.words or []:
                words.append((w.word.strip(), float(w.start), float(w.end)))
        return " ".join(text).strip(), words

    async def transcribe(self, path: Path) -> tuple[str, list[tuple[str, float, float]]]:
        async with self._lock:  # one model, one job at a time
            return await asyncio.get_running_loop().run_in_executor(None, self._run, str(path))


class Recorder:
    def __init__(self, reader: Reader, db: Database, data_dir: Path, transcriber: Transcriber, vocab: list[str],
                 pad_ms: float = 80, audio_offset_ms: float = 0) -> None:
        self.reader, self.db, self.transcriber = reader, db, transcriber
        self.clip_dir = data_dir / "clips"
        self.voice_dir = data_dir / "voice"
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        self.vocab = {w.lower(): w for w in vocab}
        self.pad_s, self.audio_offset_s = pad_ms / 1000, audio_offset_ms / 1000
        self._active: dict | None = None
        self._cache: OrderedDict[int, tuple[dict, np.ndarray]] = OrderedDict()
        self._last_id = 0
        self.last: tuple[dict, np.ndarray] | None = None  # most recently saved clip
        self.on_rest_noise = None  # callback(rms) set by main
        self.on_record_start = None  # callback(label) set by main (fake stream hook)

    # ---- ids / storage ---------------------------------------------------
    def _new_id(self) -> int:
        self._last_id = max(self._last_id + 1, int(time.time() * 1000))
        return self._last_id

    def _csv_path(self, meta: dict) -> Path:
        return self.clip_dir / f"{meta['id']}_{meta['source']}_{_safe(meta['label'])}_s{meta['session']}.csv"

    def _save(self, label: str, session: int, source: str, rows: np.ndarray) -> dict:
        meta = {"id": self._new_id(), "label": label, "session": session, "source": source,
                "t_start": float(rows[0, 0]), "t_end": float(rows[-1, 0])}
        np.savetxt(self._csv_path(meta), rows, delimiter=",", header=CSV_HEADER, comments="", fmt="%.6f")
        self._cache[meta["id"]] = (meta, rows)
        self.last = (meta, rows)
        while len(self._cache) > CACHE_CLIPS:
            self._cache.popitem(last=False)
        self.db.save_clip(meta, rows)
        return meta

    def _window_rows(self, t0: float, t1: float) -> np.ndarray:
        chin, head = self.reader.window_both(t0, t1)
        if len(chin) < MIN_ROWS_PER_SENSOR or len(head) < MIN_ROWS_PER_SENSOR:
            raise RecorderError("no sensor data for that time window", 503)
        rows = np.vstack([chin, head])
        return rows[np.argsort(rows[:, 0], kind="stable")]

    # ---- manual recording ------------------------------------------------
    def start(self, label: str, session: int) -> None:
        if self._active:
            log.warning("start while recording '%s'; discarding it", self._active["label"])
        self._active = {"label": label, "session": session, "t0": time.time()}
        if self.on_record_start:
            self.on_record_start(label)

    def stop(self) -> dict:
        a, self._active = self._active, None
        if not a:
            raise RecorderError("no recording in progress", 409)
        rows = self._window_rows(a["t0"], time.time())
        meta = self._save(a["label"], a["session"], "record", rows)
        return {"clip_id": meta["id"], "label": meta["label"]}

    async def rest_pose(self, session: int) -> None:
        """Hold still for 2 s: stores the resting pose and the sensor noise floor used for motion detection."""
        t0 = time.time()
        await asyncio.sleep(REST_SECONDS)
        chin, head = self.reader.window_both(t0, time.time())
        if len(chin) < MIN_ROWS_PER_SENSOR or len(head) < MIN_ROWS_PER_SENSOR:
            raise RecorderError("no sensor data for the rest pose", 503)
        rms = activity_level(chin, head)
        self.db.save_rest_pose(session, chin[:, 2:8].mean(0).tolist(), head[:, 2:8].mean(0).tolist(), rms)
        if self.on_rest_noise:
            self.on_rest_noise(rms)

    # ---- learn mode ------------------------------------------------------
    async def learn_utterance(self, audio: bytes, suffix: str, started_at_ms: float) -> dict:
        voice_path = self.voice_dir / f"{int(started_at_ms)}{suffix}"
        voice_path.write_bytes(audio)  # banked for the voice clone
        transcript, words = await self.transcriber.transcribe(voice_path)
        base = started_at_ms / 1000 + self.audio_offset_s
        out = []
        for raw, ws, we in words:
            word = re.sub(r"^[^\w]+|[^\w]+$", "", raw)
            if not word:
                continue
            word = self.vocab.get(word.lower(), word)
            t0, t1 = base + ws, base + we
            try:
                rows = self._window_rows(t0 - self.pad_s, t1 + self.pad_s)
            except RecorderError:
                log.info("no sensor data for '%s'; skipped", word)
                continue
            meta = self._save(word, 0, "learn", rows)
            out.append({"word": word, "start_ms": int(round(t0 * 1000)), "end_ms": int(round(t1 * 1000)), "clip_id": meta["id"]})
        return {"transcript": transcript, "words": out}

    # ---- reading / deleting ----------------------------------------------
    async def get_clip(self, clip_id: int) -> dict:
        found = self._cache.get(clip_id)
        if not found:
            found = self._read_disk(clip_id) or await self.db.get_clip(clip_id)
        if not found:
            raise RecorderError(f"clip {clip_id} not found", 404)
        meta, rows = found
        t0 = rows[0, 0] if len(rows) else 0.0
        samples = [
            {"t": round((r[0] - t0) * 1000, 1), "sensor": int(r[12]), "g": [round(float(v), 4) for v in r[2:5]],
             "a": [round(float(v), 4) for v in r[5:8]]}
            for r in rows
        ]
        return {"id": clip_id, "word": meta["label"], "samples": samples}

    def _read_disk(self, clip_id: int) -> tuple[dict, np.ndarray] | None:
        for p in self.clip_dir.glob(f"{clip_id}_*.csv"):
            m = re.match(r"(\d+)_(\w+?)_(.*)_s(-?\d+)\.csv$", p.name)
            rows = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
            if m and len(rows):
                return {"id": clip_id, "source": m[2], "label": m[3], "session": int(m[4])}, rows
        return None

    def delete_clip(self, clip_id: int) -> None:
        self._cache.pop(clip_id, None)
        for p in self.clip_dir.glob(f"{clip_id}_*.csv"):
            p.unlink(missing_ok=True)
        self.db.delete_clip(clip_id)

    async def counts(self) -> dict[str, int]:
        counts = await self.db.counts()
        if counts is not None:
            return counts
        counts = {}
        for p in self.clip_dir.glob("*_record_*.csv"):
            m = re.match(r"\d+_record_(.*)_s-?\d+\.csv$", p.name)
            if m:
                counts[m[1]] = counts.get(m[1], 0) + 1
        return counts
