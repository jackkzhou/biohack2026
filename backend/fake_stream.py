"""Fake ESP32: emits the same text lines as the real firmware so the whole
pipeline (parser, clock sync, buffer, inference) can be tested without hardware.

Every few seconds it "mouths" a phrase of 1-2 words. Each word has its own
motion signature on the chin gyro; the head sensor mostly just wobbles.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import time

from backend.reader import Reader

log = logging.getLogger("fake")

WORDS = ["help", "yes", "no", "Jordan", "water"]
# per word: (frequency Hz, axis weights)
_SIG = {
    "help": (3.0, (1.0, 0.2, 0.0)),
    "yes": (2.0, (0.0, 1.0, 0.2)),
    "no": (4.5, (0.2, 0.0, 1.0)),
    "Jordan": (6.0, (1.0, 1.0, 0.0)),
    "water": (1.2, (0.0, 1.0, 1.0)),
}
WORD_S = 0.7
HZ = 200


class FakeStream:
    def __init__(self, reader: Reader, autoplay: bool = True) -> None:
        self.reader = reader
        self.autoplay = autoplay
        self._bursts: list[tuple[float, float, str, float]] = []  # (t_start, t_end, word, intensity)
        # The fake ESP32 clock starts at an arbitrary value so clock sync is really exercised.
        self._esp_zero_us = random.randint(3_000_000, 60_000_000)
        self._t0 = time.time()

    def burst(self, word: str, delay: float = 0.3, duration: float = WORD_S, intensity: float = 1.0) -> None:
        """Schedule a mouthed word (used by the recorder while a take is running)."""
        start = time.time() + delay
        self._bursts.append((start, start + duration, word, intensity))
        self._bursts = [b for b in self._bursts if b[1] > time.time() - 30]

    def word_at(self, t0: float, t1: float) -> str | None:
        """Ground truth: the word whose burst overlaps [t0, t1] the most (for the stand-in classifier)."""
        best, best_overlap = None, 0.0
        for s, e, w, intensity in self._bursts:
            o = min(e, t1) - max(s, t0)
            if o > best_overlap and intensity >= 0.5:
                best, best_overlap = w, o
        return best

    def _signal(self, t: float, sensor: int) -> list[float]:
        g = [random.gauss(0, 0.01) for _ in range(3)]
        for s, e, w, intensity in self._bursts:
            if s <= t <= e:
                f, axes = _SIG[w]
                env = math.sin(math.pi * (t - s) / (e - s))
                amp = 1.5 * intensity * env * math.sin(2 * math.pi * f * (t - s))
                scale = 1.0 if sensor == 0 else 0.0
                g = [g[i] + amp * axes[i] * scale for i in range(3)]
        if sensor == 1:  # head: slow wobble, shared by nothing the model needs
            g = [g[i] + 0.02 * math.sin(0.7 * t + i) for i in range(3)]
        a = [random.gauss(0, 0.02) + off for off in (0.3, 0.1, 9.78)]
        return g + a

    async def _script(self) -> None:
        while True:
            await asyncio.sleep(random.uniform(3.5, 5.5))
            for w in random.sample(WORDS, random.choice([1, 2])):
                self.burst(w, delay=0.0)
                await asyncio.sleep(WORD_S + 0.6)

    async def run(self) -> None:
        log.info("fake sensor stream running (%d Hz per sensor)", HZ)
        script = asyncio.create_task(self._script()) if self.autoplay else None
        step = 1.0 / HZ
        nxt = time.monotonic()
        try:
            while True:
                t = time.time()
                esp_us = self._esp_zero_us + int((t - self._t0) * 1e6)
                for sensor in (0, 1):
                    vals = self._signal(t, sensor)
                    line = f"{esp_us},{sensor}," + ",".join(f"{v:.4f}" for v in vals) + ",1.0000,0.0000,0.0000,0.0000"
                    self.reader.feed_line(line, recv=t + random.uniform(0.0, 0.004))
                nxt += step
                await asyncio.sleep(max(0.0, nxt - time.monotonic()))
        finally:
            if script:
                script.cancel()
