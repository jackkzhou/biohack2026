"""Live word detection in the terminal (no frontend, DB or TTS).

    python -m backend.live                # motion-triggered (same as the web backend)
    python -m backend.live --every 3      # classify the last 3 s of data every 3 s, no motion trigger

Uses the same Reader, Inference loop and model as the web backend, so what you
see here is what the backend would do. The serial port can only be opened by
one process: stop uvicorn first.

Prints every detected segment with a confidence bar:
    ✓ green   word accepted (confidence >= CONF_THRESHOLD, not "nothing")
    · dim     rejected (idle = "nothing", or low confidence), with the model's top 3
and the final phrase in a box after a pause.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

import numpy as np

from backend.config import settings
from backend.inference import Inference, ModelClassifier, load_phrases
from backend.reader import Reader


USE_COLOR = sys.stdout.isatty()


def c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


class VerboseClassifier(ModelClassifier):
    """Same as ModelClassifier, but remembers the full probability vector for printing."""

    last: np.ndarray | None = None

    def classify(self, chin: np.ndarray, head: np.ndarray) -> tuple[str, float]:
        x = self.window(chin, head)
        if x is None:
            self.last = None
            return "nothing", 0.0
        self.last = self.model.predict_proba(x[None])[0]
        i = int(np.argmax(self.last))
        return self.labels[i], float(self.last[i])


def runners_up(clf: VerboseClassifier, n: int = 3) -> str:
    if clf.last is None:
        return ""
    order = np.argsort(clf.last)[::-1][:n]
    return "  ".join(f"{clf.labels[i]} {clf.last[i]:.2f}" for i in order)


async def run_fixed(reader: Reader, clf: VerboseClassifier, every: float, t0: float) -> None:
    """Every `every` seconds classify the newest `every` seconds of both sensors."""
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(every)
        now = reader.latest_time()
        if now is None or not reader.connected:
            continue
        chin, head = reader.window_both(now - every, now)
        if len(chin) < 5 or len(head) < 5:
            continue
        label, conf = await loop.run_in_executor(None, clf.classify, chin, head)
        bar = "█" * round(conf * 10) + "░" * (10 - round(conf * 10))
        stamp = f"{time.monotonic() - t0:6.1f}s"
        if label != "nothing" and conf >= settings.conf_threshold:
            print(c("1;32", f" ✓ {stamp}  {label:<9} {bar} {conf:4.0%}  top: {runners_up(clf)}"))
        else:
            print(c("2", f" · {stamp}  {label:<9} {bar} {conf:4.0%}  top: {runners_up(clf)}"))


async def main(every: float | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    base = ModelClassifier.load(settings.models_dir)
    classifier = VerboseClassifier(base.model, base.labels) if base else None
    if classifier is None:
        raise SystemExit("no model in backend/models/ (run: python -m ml.train --model cnn ...)")
    print(f"model: {len(classifier.labels)} classes: {', '.join(classifier.labels)}")

    reader = Reader(settings.serial_port, settings.serial_baud, settings.fake_stream)
    t0 = time.monotonic()

    def on_detection(_t: float, label: str, conf: float, accepted: bool) -> None:
        bar = "█" * round(conf * 10) + "░" * (10 - round(conf * 10))
        stamp = f"{time.monotonic() - t0:6.1f}s"
        if accepted:
            print(c("1;32", f" ✓ {stamp}  {label:<9} {bar} {conf:4.0%}"))
        else:
            why = "idle" if label == "nothing" else "low confidence"
            print(c("2", f" · {stamp}  {label:<9} {bar} {conf:4.0%}  ({why})  top: {runners_up(classifier)}"))

    async def emit(msg: dict) -> None:
        if msg["type"] == "phrase":
            words = " + ".join(msg["words"])
            line = f"  {words}  →  {msg['sentence']}  "
            print("\n" + c("1;36", "┌" + "─" * len(line) + "┐"))
            print(c("1;36", "│" + line + "│"))
            print(c("1;36", "└" + "─" * len(line) + "┘") + "\n")

    async def on_phrase(words: list[str], sentence: str) -> None:
        pass

    inf = Inference(
        reader, classifier, emit, on_phrase, load_phrases(settings.models_dir),
        conf_threshold=settings.conf_threshold, pause_s=settings.phrase_pause_s,
        activity_threshold=settings.activity_threshold, on_detection=on_detection,
    )
    reader.start()
    task = asyncio.create_task(run_fixed(reader, classifier, every, t0) if every else inf.run())
    print(f"reading {settings.serial_port or 'fake stream'} ... (Ctrl+C to quit)")
    was = None
    try:
        while True:
            await asyncio.sleep(0.5)
            if reader.connected != was:
                was = reader.connected
                print("sensors connected" if was else "waiting for sensors (check the cable / SERIAL_PORT)")
    finally:
        task.cancel()
        reader.stop()


if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--every", type=float, help="classify a fixed window every N seconds instead of waiting for motion")
        asyncio.run(main(ap.parse_args().every))
    except KeyboardInterrupt:
        pass
