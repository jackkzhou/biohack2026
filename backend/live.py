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
from ml.tare import TareCalibrator

TARE_SECONDS = 1.5


USE_COLOR = sys.stdout.isatty()
HEARTBEAT_S = 2.0  # while the jaw is still, print "(nothing detected)" this often


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


async def run_fixed(reader: Reader, clf: VerboseClassifier, every: float, t0: float, calibrator: TareCalibrator) -> None:
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
        chin, head = calibrator.apply(chin, head)
        label, conf = await loop.run_in_executor(None, clf.classify, chin, head)
        bar = "█" * round(conf * 10) + "░" * (10 - round(conf * 10))
        stamp = f"{time.monotonic() - t0:6.1f}s"
        if label != "nothing" and conf >= settings.conf_threshold:
            print(c("1;32", f" ✓ {stamp}  {label:<9} {bar} {conf:4.0%}  top: {runners_up(clf)}"))
        else:
            print(c("2", f" · {stamp}  {label:<9} {bar} {conf:4.0%}  top: {runners_up(clf)}"))


def meter_line(reader: Reader, inf: Inference) -> str:
    """One live status line: board data rate, motion level vs the trigger threshold, and state."""
    now = reader.latest_time()
    if now is None:
        return "no samples yet"
    chin, head = inf.calibrator.apply(*reader.window_both(now - 0.15, now))
    level = inf._activity(chin, head)
    thr = inf.threshold
    cells = 24
    full = min(cells, round(level / (2 * thr) * cells))  # bar spans 0 .. 2x threshold
    mark = cells // 2
    bar = "".join(("█" if i < full else "░") if i != mark else ("┃" if i >= full else "█") for i in range(cells))
    hz = reader.rates()
    state = "RECORDING A WORD" if inf._seg_start is not None else ("moving" if level > thr else "still")
    return f"data {hz[0]:.0f}/{hz[1]:.0f} Hz  motion {level:5.2f} {bar} (trigger {thr:.2f})  {state}    "


async def main(every: float | None = None, meter: bool = False, show_phrase: bool = False) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    base = ModelClassifier.load(settings.models_dir)
    classifier = VerboseClassifier(base.model, base.labels) if base else None
    if classifier is None:
        raise SystemExit("no model in backend/models/ (run: python -m ml.train --model cnn ...)")
    print(f"model: {len(classifier.labels)} classes: {', '.join(classifier.labels)}")

    reader = Reader(settings.serial_port, settings.serial_baud, settings.fake_stream)
    t0 = time.monotonic()

    last_print = [time.monotonic()]  # when anything was last printed (for the "still" heartbeat)

    def on_detection(_t: float, label: str, conf: float, accepted: bool) -> None:
        bar = "█" * round(conf * 10) + "░" * (10 - round(conf * 10))
        stamp = f"{time.monotonic() - t0:6.1f}s"
        last_print[0] = time.monotonic()
        if label == "nothing":  # the model says this movement is not a word
            print(c("2", f" · {stamp}  (nothing detected)  [moved, but classified as 'nothing' {conf:.0%}]"))
        elif accepted:
            print(c("1;32", f" ✓ {stamp}  {label:<9} {bar} {conf:4.0%}"))
        else:
            why = "idle" if label == "nothing" else "low confidence"
            print(c("2", f" · {stamp}  {label:<9} {bar} {conf:4.0%}  ({why})  top: {runners_up(classifier)}"))

    async def emit(msg: dict) -> None:
        if show_phrase and msg["type"] == "phrase":  # the boxed phrase is off by default (--phrase turns it on)
            words = " + ".join(msg["words"])
            line = f"  {words}  →  {msg['sentence']}  "
            print("\n" + c("1;36", "┌" + "─" * len(line) + "┐"))
            print(c("1;36", "│" + line + "│"))
            print(c("1;36", "└" + "─" * len(line) + "┘") + "\n")

    async def on_phrase(words: list[str], sentence: str) -> None:
        pass

    reader.start()
    calibrator = TareCalibrator()
    print("Waiting for sensors...")
    for _ in range(100):
        if reader.connected:
            break
        await asyncio.sleep(0.2)
    else:
        raise SystemExit("No data from the ESP32. Check SERIAL_PORT and that the firmware is flashed.")
    await asyncio.sleep(1.2)  # let the quaternion fusion warm up
    try:
        await asyncio.to_thread(input, "Hold the jaw and head still in a neutral pose, then press Enter to tare... ")
    except EOFError:
        pass
    tare_t0 = time.time()
    await asyncio.sleep(TARE_SECONDS)
    chin, head = reader.window_both(tare_t0, time.time())
    try:
        calibrator.tare(chin, head)
        print("Tare OK.")
    except ValueError as e:
        print(f"Tare failed ({e}); continuing without orientation calibration.")

    inf = Inference(
        reader, classifier, emit, on_phrase, load_phrases(settings.models_dir),
        conf_threshold=settings.conf_threshold, pause_s=settings.phrase_pause_s,
        activity_threshold=settings.activity_threshold, on_detection=on_detection,
        calibrator=calibrator,
    )
    task = asyncio.create_task(run_fixed(reader, classifier, every, t0, calibrator) if every else inf.run())
    print(f"reading {settings.serial_port or 'fake stream'} ... (Ctrl+C to quit)")
    was = None
    try:
        while True:
            await asyncio.sleep(0.25 if meter else 0.5)
            # A still jaw never reaches the classifier, so say so: one line every 2 s of silence.
            if (reader.connected and inf._seg_start is None and not every
                    and time.monotonic() - last_print[0] >= HEARTBEAT_S):
                last_print[0] = time.monotonic()
                print("\r\033[K" + c("2", f" · {time.monotonic() - t0:6.1f}s  (nothing detected)"))
            if meter and reader.connected:
                print("\r\033[K" + meter_line(reader, inf), end="", flush=True)
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
        ap.add_argument("--meter", action="store_true", help="show a live motion meter (data rate, motion vs trigger)")
        ap.add_argument("--phrase", action="store_true", help="also print the finished phrase in a box")
        args = ap.parse_args()
        asyncio.run(main(args.every, args.meter, args.phrase))
    except KeyboardInterrupt:
        pass
