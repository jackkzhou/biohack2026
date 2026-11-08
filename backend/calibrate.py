"""On-site calibration: record a few fresh takes right before a demo, then retrain with extra weight on them.

A new session (new day, remounted hat) looks different from the training sessions, so a model trained only
on old sessions loses about half its accuracy. Mixing in ~10 takes per word from *today's* setup, weighted
up, recovers a large part of that. Flow (all driven from the app's Calibrate tab):

    GET  /calibrate/info    words, takes per word, whether the tare is done, current model
    POST /calibrate/begin   new session number + a shuffled take order
    POST /record/start|stop the existing recording routes save each take (tared, to disk + database)
    POST /calibrate/train   retrain (random forest) on the tared old sessions + this session, hot-swap it in
    POST /calibrate/undo    go back to the model that was running before
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from pathlib import Path

import numpy as np

from backend.inference import Inference, ModelClassifier
from ml.tare import tared_sessions

log = logging.getLogger("calibrate")

CLIP_SESSION_RE = re.compile(r"_record_.*_s(-?\d+)\.csv$")
MIN_BASE_PER_CLASS = 5


class CalibrationError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def next_session(clip_dir: Path) -> int:
    nums = [int(m[1]) for p in clip_dir.glob("*_record_*_s*.csv") if (m := CLIP_SESSION_RE.search(p.name))]
    return max(nums, default=0) + 1


def session_counts(clip_dir: Path, session: int | None) -> dict[str, int]:
    """Takes saved so far in one session, per word (read from the CSV names, which are always written first)."""
    out: dict[str, int] = {}
    if session is None:
        return out
    for p in clip_dir.glob(f"*_record_*_s{session}.csv"):
        m = re.match(r"\d+_record_(.*)_s-?\d+\.csv$", p.name)
        if m:
            out[m[1]] = out.get(m[1], 0) + 1
    return out


def _train(clip_dir: Path, tares_dir: Path, words: list[str], session: int, weight: float, out_dir: Path) -> dict:
    from ml.silent_model import SilentSpeechRF
    from ml.train import load_clips

    X, y, s = load_clips(clip_dir, ["record"], 0.12, {w.lower() for w in words}, None, "default")
    if len(y) == 0:
        raise CalibrationError("no usable clips found")
    keep = np.isin(s, sorted(tared_sessions(tares_dir) | {session}))  # old untared sessions would not match tared live data
    X, y, s = X[keep], y[keep], s[keep]
    new = s == session
    if not new.any():
        raise CalibrationError("no clips were recorded in this calibration session", 409)
    present = [w for w in words if (y[new] == w).any()]  # the classes this calibration actually covers
    if len(present) < 2:
        raise CalibrationError("record at least two different words (nothing counts as one)", 409)
    sel = np.isin(y, present)
    X, y, s, new = X[sel], y[sel], s[sel], new[sel]
    counts = {w: int((y[new] == w).sum()) for w in present}
    base_counts = {w: int(((y == w) & ~new).sum()) for w in present}
    thin = [w for w in present if base_counts[w] < MIN_BASE_PER_CLASS]
    if thin:
        log.warning("few older takes for %s; relying on the new session", thin)
    model = SilentSpeechRF(300, 42).fit(X, y, sample_weight=np.where(new, weight, 1.0))
    model.preprocess = "default"
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "labels.json").write_text(json.dumps([str(c) for c in model.classes_]))
    meta = {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "session": session, "weight": weight,
            "new_takes": counts, "base_takes": base_counts, "base_sessions": sorted(set(s[~new].tolist()))}
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    return {"model": model, "meta": meta}


class Calibration:
    def __init__(self, clip_dir: Path, tares_dir: Path, out_dir: Path, words: list[str], takes: int, weight: float,
                 allowed: list[str]) -> None:
        self.clip_dir, self.tares_dir, self.out_dir = clip_dir, tares_dir, out_dir
        self.words, self.takes, self.weight = words, takes, weight  # defaults offered by the UI
        self.allowed = allowed  # every word a calibration may include
        self.session: int | None = None
        self.calibrated = False
        self._previous: tuple[ModelClassifier | None] | None = None
        self._lock = asyncio.Lock()

    def info(self, tared: bool, inference: Inference) -> dict:
        counts = session_counts(self.clip_dir, self.session)
        return {"words": self.words, "allowed": self.allowed, "takes_per_word": self.takes, "seconds": 3.0,
                "tared": tared, "model_loaded": inference.model_loaded, "calibrated": self.calibrated,
                "session": self.session, "counts": counts, "can_undo": self._previous is not None}

    def begin(self, words: list[str] | None, takes: int | None, keep_session: bool) -> dict:
        """Start recording takes. `keep_session` adds to the current calibration session (same tare, same
        data); otherwise a new session number is used and the previous calibration is left behind."""
        words = [w for w in (words or self.words) if w in self.allowed]
        if not words:
            raise CalibrationError("pick at least one word")
        if keep_session and self.session is not None:
            session = self.session
        else:
            session = next_session(self.clip_dir)
            self.session = session
        n = max(1, min(takes or self.takes, 30))
        order = [w for w in words for _ in range(n)]
        random.shuffle(order)
        return {"session": session, "order": order}

    async def train(self, inference: Inference, session: int | None) -> dict:
        session = session if session is not None else self.session
        if session is None:
            raise CalibrationError("start a calibration first")
        if self._lock.locked():
            raise CalibrationError("already training", 409)
        async with self._lock:
            loop = asyncio.get_running_loop()
            out = await loop.run_in_executor(
                None, _train, self.clip_dir, self.tares_dir, self.allowed, session, self.weight, self.out_dir)
            model = out["model"]
            new = ModelClassifier(model, [str(c) for c in model.classes_])
            previous = inference.set_classifier(new)
            if not self.calibrated:  # keep the model from before the FIRST calibration so undo goes all the way back
                self._previous = (previous,)
            self.calibrated = True
            log.info("calibrated model live (session %s, %s)", session, out["meta"]["new_takes"])
            return {"ok": True, **out["meta"], "labels": new.labels}

    def undo(self, inference: Inference) -> None:
        if self._previous is None:
            raise CalibrationError("nothing to undo", 409)
        inference.set_classifier(self._previous[0])
        self._previous = None
        self.calibrated = False

    def reset(self, inference: Inference) -> None:
        """Forget this calibration: back to the model from before it, and the next one gets a fresh session.
        The tare is not touched, and the takes already recorded stay in the database and in data/clips."""
        if self._previous is not None:
            self.undo(inference)
        self.session = None
        self.calibrated = False
