"""Live sensor-orientation check against the training data.

    python -m backend.orient

Gravity (the mean accelerometer vector) tells you how each sensor is tilted.
This compares it, for the chin and head IMU separately, with the average over
all clips in data/clips (what the model was trained on) and shows the angle
between them, refreshed twice a second. Re-seat the sensors until both read
OK, then run live inference.

The board's onboard LED mirrors the result (needs the updated firmware flashed):
    solid = both sensors OK, fast blink = one OK, slow blink = neither.
It returns to its normal "both IMUs streaming" meaning when this script stops.

Limit: an accelerometer can't see rotation around the gravity direction
(e.g. turning your head left/right while upright), only tilt.
Stop uvicorn / backend.live first: only one process can read the board.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np

from backend.config import REPO_DIR, settings
from backend.reader import Reader

CLIP_RE = re.compile(r"(\d+)_(\w+?)_(.*)_s(-?\d+)\.csv$")
WINDOW_S = 1.0
OK_DEG, WARN_DEG, HYST_DEG = 10.0, 25.0, 3.0
USE_COLOR = sys.stdout.isatty()


def c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def reference(clip_dir: Path) -> dict[int, np.ndarray]:
    """Mean accel vector per sensor (0 chin, 1 head) over every training clip."""
    sums: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for p in sorted(clip_dir.glob("*.csv")):
        if not CLIP_RE.match(p.name):
            continue
        rows = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
        if rows.size == 0:
            continue
        for s in (0, 1):
            x = rows[rows[:, 12] == s]
            if len(x) >= 5:
                sums[s].append(x[:, 5:8].mean(axis=0))
    if not sums[0] or not sums[1]:
        raise SystemExit(f"no usable training clips in {clip_dir}")
    return {s: np.mean(v, axis=0) for s, v in sums.items()}


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def verdict(deg: float, is_ok: bool) -> str:
    if is_ok:
        return c("1;32", "OK  ")
    if deg < WARN_DEG:
        return c("1;33", "OFF ")
    return c("1;31", "BAD ")


def fmt(v: np.ndarray) -> str:
    return "[" + " ".join(f"{x:6.2f}" for x in v) + "]"


def main() -> None:
    ref = reference(REPO_DIR / "data" / "clips")
    print(f"training reference (mean accel m/s^2)  chin {fmt(ref[0])}   head {fmt(ref[1])}")
    print(f"OK < {OK_DEG:.0f} deg, OFF < {WARN_DEG:.0f} deg, BAD beyond. Hold still while checking. Ctrl+C quits.\n")
    reader = Reader(settings.serial_port, settings.serial_baud, settings.fake_stream)
    reader.start()
    ok = {0: False, 1: False}
    try:
        while True:
            time.sleep(0.5)
            now = reader.latest_time()
            if now is None or not reader.connected:
                print("\r" + "waiting for sensors ...".ljust(100), end="", flush=True)
                continue
            chin, head = reader.window_both(now - WINDOW_S, now)
            if len(chin) < 5 or len(head) < 5:
                continue
            cur = {0: chin[:, 5:8].mean(axis=0), 1: head[:, 5:8].mean(axis=0)}
            parts = []
            for s, name in ((0, "chin"), (1, "head")):
                d = angle_deg(cur[s], ref[s])
                # hysteresis: enter OK below OK_DEG, leave only above OK_DEG + HYST_DEG (no flicker at the edge)
                ok[s] = d < (OK_DEG + HYST_DEG if ok[s] else OK_DEG)
                parts.append(f"{name} {verdict(d, ok[s])}{d:5.1f}° {fmt(cur[s])}")
            n_ok = sum(ok.values())
            reader.send(f"O{1 + n_ok}\n".encode())  # 1 slow blink / 2 fast blink / 3 solid
            print("\r" + "   ".join(parts) + "  ", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        reader.send(b"O0\n")
        time.sleep(0.1)
        reader.stop()


if __name__ == "__main__":
    main()
