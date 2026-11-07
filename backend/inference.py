"""Word detection and phrase building.

Loop (every 50 ms), working on the newest sensor data:
  1. Activity = RMS of the chin gyro with the head gyro subtracted (head motion cancels).
  2. Motion starts a segment; 0.3 s of stillness ends it -> one word-sized window.
  3. The window goes through the shared `ml.preprocess.preprocess_window` and the
     trained model. Low-confidence results and the "nothing" class are ignored.
  4. ~1.5 s of stillness after the last motion ends the phrase; the words are
     mapped to a sentence with a lookup table.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Protocol

import numpy as np

from backend.reader import GYRO, SENSOR_CHIN, SENSOR_HEAD, Reader

log = logging.getLogger("inference")

TICK_S = 0.05
ACT_WINDOW_S = 0.15
WORD_END_GAP_S = 0.9  # your words have pauses up to ~0.7 s inside them (9% of training words exceed 0.8 s); must stay below PHRASE_PAUSE_S
SEGMENT_PAD_S = 0.1
MAX_WORD_S = 3.0  # training takes are 3 s
MIN_WORD_S = 0.3
NOTHING = "nothing"

# Ordered key first, then order-insensitive key; unknown combinations get a generic sentence.
DEFAULT_PHRASES: dict[str, str] = {
    "help": "Please help me.",
    "yes": "Yes.",
    "no": "No, thank you.",
    "water": "Can I have some water, please?",
    "jordan": "Jordan.",
    "help+jordan": "Jordan, I need your help.",
    "jordan+water": "Jordan, can I have some water?",
    "help+water": "I need water, please help.",
    "jordan+yes": "Jordan, yes.",
    "jordan+no": "Jordan, no.",
    "no+water": "No water, thank you.",
    "water+yes": "Yes, I would like some water.",
}


def activity_level(chin: np.ndarray, head: np.ndarray) -> float:
    """RMS of |chin gyro - head gyro| over the window (0 if there is too little data)."""
    if len(chin) < 2 or len(head) < 2:
        return 0.0
    hg = np.stack([np.interp(chin[:, 0], head[:, 0], head[:, 2 + i]) for i in range(3)], axis=1)
    d = np.linalg.norm(chin[:, GYRO] - hg, axis=1)
    return float(np.sqrt(np.mean(d**2)))


def activity_level_invariant(chin: np.ndarray, head: np.ndarray) -> float:
    """Orientation-free motion level: chin rotation speed minus head rotation speed (lengths only)."""
    from ml.preprocess import activity_level_invariant as f

    return f(chin, head)


class Classifier(Protocol):
    def classify(self, chin: np.ndarray, head: np.ndarray) -> tuple[str, float]: ...


class ModelClassifier:
    """Wraps a trained model from backend/models/.

    Expects `model.joblib` with `predict_proba(X)` and `classes_`, where X has
    shape (n, channels, steps) as returned by `preprocess_window` (a flattened
    2-D input is tried as a fallback). Optional `labels.json` overrides classes_.
    """

    def __init__(self, model: Any, labels: list[str]) -> None:
        self.model, self.labels = model, labels
        # Which ml.preprocess pipeline the model was trained with (older models: raw axes).
        self.preprocess_name = getattr(model, "preprocess", "default")

    @property
    def orientation_free(self) -> bool:
        return self.preprocess_name == "orientation-free"

    def window(self, chin: np.ndarray, head: np.ndarray):
        """Raw rows of both sensors -> model input, exactly as the model was trained."""
        from ml.preprocess import get_preprocessor

        return get_preprocessor(self.preprocess_name)(chin, head)

    @classmethod
    def load(cls, models_dir) -> "ModelClassifier | None":
        path = models_dir / "model.joblib"
        if not path.exists():
            return None
        import joblib

        model = joblib.load(path)
        lp = models_dir / "labels.json"
        labels = json.loads(lp.read_text()) if lp.exists() else [str(c) for c in getattr(model, "classes_", [])]
        if not labels:
            raise RuntimeError("model has no class labels: add backend/models/labels.json")
        log.info("loaded model %s (%d classes)", path.name, len(labels))
        return cls(model, labels)

    def classify(self, chin: np.ndarray, head: np.ndarray) -> tuple[str, float]:
        x = self.window(chin, head)
        if x is None:
            return NOTHING, 0.0
        try:
            p = self.model.predict_proba(x[None])[0]
        except ValueError:
            p = self.model.predict_proba(x.reshape(1, -1))[0]
        i = int(np.argmax(p))
        return self.labels[i], float(p[i])


class GroundTruthClassifier:
    """Test-only stand-in used with FAKE_STREAM when no trained model exists.

    It "recognises" the word the fake stream generated, so the full pipeline
    (words -> pause -> phrase -> speech) can be demoed without hardware or a model.
    """

    def __init__(self, fake: Any) -> None:
        self.fake = fake

    def classify(self, chin: np.ndarray, head: np.ndarray) -> tuple[str, float]:
        w = self.fake.word_at(float(chin[0, 0]), float(chin[-1, 0]))
        return (w, 0.93) if w else (NOTHING, 0.9)


def load_phrases(models_dir) -> dict[str, str]:
    table = dict(DEFAULT_PHRASES)
    p = models_dir / "phrases.json"
    if p.exists():
        try:
            table.update({k.lower(): v for k, v in json.loads(p.read_text()).items()})
        except Exception as e:  # noqa: BLE001
            log.warning("could not read phrases.json: %s", e)
    return table


def words_to_sentence(words: list[str], table: dict[str, str]) -> str:
    low = [w.lower() for w in words]
    for key in ("+".join(low), "+".join(sorted(low))):
        if key in table:
            return table[key]
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."


Emit = Callable[[dict], Awaitable[None]]


class Inference:
    def __init__(
        self,
        reader: Reader,
        classifier: Classifier | None,
        emit: Emit,
        on_phrase: Callable[[list[str], str], Awaitable[None]],
        phrases: dict[str, str],
        *,
        conf_threshold: float,
        pause_s: float,
        activity_threshold: float,
        on_detection: Callable[[float, str, float, bool], None] | None = None,
    ) -> None:
        self.reader, self.classifier, self.emit, self.on_phrase = reader, classifier, emit, on_phrase
        self.phrases = phrases
        self.conf_threshold, self.pause_s = conf_threshold, pause_s
        self.base_threshold = self.threshold = activity_threshold
        self.on_detection = on_detection
        self.mode = "lookup"
        # a model trained without orientation also gets orientation-free motion detection
        self._activity = activity_level_invariant if getattr(classifier, "orientation_free", False) else activity_level
        self._words: list[str] = []
        self._seg_start: float | None = None
        self._last_active = 0.0
        self._sent_progress = 0.0

    @property
    def model_loaded(self) -> bool:
        return self.classifier is not None

    def calibrate(self, rest_rms: float) -> None:
        """Set the motion threshold from a rest-pose recording (never below the configured default)."""
        # capped at 3x the default so a rest pose taken while moving cannot switch detection off
        self.threshold = min(max(self.base_threshold, 4.0 * rest_rms), 3.0 * self.base_threshold)
        log.info("activity threshold set to %.3f rad/s (rest noise %.4f)", self.threshold, rest_rms)

    def clear(self) -> None:
        self._words.clear()
        self._seg_start = None
        self._sent_progress = 0.0

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(TICK_S)
            try:
                await self._tick(loop)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("inference tick failed")

    async def _tick(self, loop: asyncio.AbstractEventLoop) -> None:
        now = self.reader.latest_time()
        if now is None or not self.reader.connected:
            return  # no live data: freeze the state machine rather than invent stillness
        chin, head = self.reader.window_both(now - ACT_WINDOW_S, now)
        active = self._activity(chin, head) > self.threshold

        if active:
            self._last_active = now
            if self._seg_start is None:
                self._seg_start = now - ACT_WINDOW_S
            await self._progress(0.0)
        elif self._seg_start is not None and now - self._last_active >= WORD_END_GAP_S:
            await self._finish_segment(loop)

        if self._seg_start is not None and now - self._seg_start > MAX_WORD_S:
            await self._finish_segment(loop)

        if self._words and not active:
            await self._progress(min(1.0, (now - self._last_active) / self.pause_s))
            if now - self._last_active >= self.pause_s and self._seg_start is None:
                await self._end_phrase()

    async def _finish_segment(self, loop: asyncio.AbstractEventLoop) -> None:
        start, end = self._seg_start, self._last_active
        self._seg_start = None
        if start is None or end - start < MIN_WORD_S:
            return
        chin, head = self.reader.window_both(start - SEGMENT_PAD_S, end + SEGMENT_PAD_S)
        if self.classifier is None or len(chin) < 5 or len(head) < 5:
            return
        label, conf = await loop.run_in_executor(None, self.classifier.classify, chin, head)
        accepted = label != NOTHING and conf >= self.conf_threshold
        if self.on_detection:
            self.on_detection(end, label, conf, accepted)
        if not accepted:
            log.debug("ignored %s (%.2f)", label, conf)
            return
        self._words.append(label)
        await self.emit({"type": "word", "word": label, "confidence": round(conf, 2)})

    async def _progress(self, p: float) -> None:
        if not self._words:
            return
        if abs(p - self._sent_progress) >= 0.03 or (p in (0.0, 1.0) and p != self._sent_progress):
            self._sent_progress = p
            await self.emit({"type": "pause", "progress": round(p, 2)})

    async def _end_phrase(self) -> None:
        words, self._words = self._words, []
        self._sent_progress = 0.0
        sentence = words_to_sentence(words, self.phrases)  # 'llm' mode is accepted but falls back to the lookup table
        await self.emit({"type": "phrase", "words": words, "sentence": sentence})
        await self.on_phrase(words, sentence)
