"""Interactive dataset collection from the terminal.

    python -m backend.collect                 # record takes
    python -m backend.collect --upload        # push saved CSVs that never reached the database

You type a label, press Enter to START, speak / mouth the word, the take stops by
itself after 3 seconds. Each take is saved to data/clips/*.csv first (always works) and queued for
Tiger Data. Stop the uvicorn server first: only one program can open the serial port.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys

import numpy as np

from backend.config import NOTHING, VOCAB, settings
from backend.db import Database
from backend.fake_stream import WORDS, FakeStream
from backend.reader import Reader
from backend.recorder import Recorder, RecorderError, Transcriber

TAKE_SECONDS = 3.0  # every take is exactly this long
MIN_HZ = 75  # warn below this per-sensor rate (target is 100)


async def ask(prompt: str) -> str:
    try:
        return (await asyncio.to_thread(input, prompt)).strip()
    except EOFError:  # input closed: behave like "quit"
        return "q"


def next_session(recorder: Recorder) -> int:
    nums = [int(m[1]) for p in recorder.clip_dir.glob("*_record_*_s*.csv") if (m := re.search(r"_s(\d+)\.csv$", p.name))]
    return max(nums, default=0) + 1


async def upload_backlog(db: Database, recorder: Recorder) -> None:
    files = sorted(recorder.clip_dir.glob("*.csv"))
    queued = 0
    for p in files:
        m = re.match(r"(\d+)_(\w+?)_(.*)_s(-?\d+)\.csv$", p.name)
        if not m:
            continue
        rows = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
        if not len(rows):
            continue
        db.save_clip({"id": int(m[1]), "source": m[2], "label": m[3], "session": int(m[4]),
                      "t_start": float(rows[0, 0]), "t_end": float(rows[-1, 0])}, rows)
        queued += 1
    print(f"queued {queued} clip file(s); clips already in the database are skipped")


async def collect(recorder: Recorder, reader: Reader, session: int, labels: list[str], seconds: float) -> None:
    print("Waiting for sensor data...")
    for _ in range(100):
        if reader.connected:
            break
        await asyncio.sleep(0.2)
    else:
        print("No data from the ESP32. Check SERIAL_PORT, close other serial monitors, and that the firmware is flashed.")
        return
    await asyncio.sleep(1.2)
    reader.rates()
    await asyncio.sleep(1.0)
    rates = reader.rates()
    print(f"Streaming: jaw {rates[0]:.0f} Hz, head {rates[1]:.0f} Hz. Session {session}.")
    await ask("Hold the jaw and head still in a neutral pose, then press Enter to tare... ")
    try:
        await recorder.tare(session=session)
        print("Tare OK.")
    except RecorderError as e:
        print(f"Tare failed ({e}); continuing without orientation calibration.")
    allowed = {w.lower(): w for w in [*VOCAB, NOTHING]}
    print("Labels:", ", ".join(allowed.values()))
    last = labels[0] if labels else ""
    while True:
        counts = await recorder.counts()
        if counts:
            print("Saved so far:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        label = await ask(f"\nLabel{f' [{last}]' if last else ''} (q = quit): ")
        if label.lower() in ("q", "quit", "exit"):
            return
        label = label or last
        if not label:
            continue
        if label.lower() not in allowed:
            print(f"'{label}' is not in the vocabulary. Use one of: {', '.join(allowed.values())}")
            continue
        label = allowed[label.lower()]
        last = label
        await ask(f"Press Enter to START '{label}'... ")
        recorder.start(label, session)
        print(f"● RECORDING '{label}' for {seconds:g} s. Speak now!", end="", flush=True)
        await asyncio.sleep(seconds)
        print("\a  stop.")
        try:
            res = recorder.stop()
        except RecorderError as e:
            print(f"Take failed: {e}")
            continue
        meta, rows = recorder.last
        dur = meta["t_end"] - meta["t_start"]
        n = [int((rows[:, 12] == s).sum()) for s in (0, 1)]
        hz = [x / dur if dur > 0 else 0 for x in n]
        print(f"Saved clip {res['clip_id']}: {dur:.2f} s, jaw {n[0]} samples ({hz[0]:.0f} Hz), head {n[1]} samples ({hz[1]:.0f} Hz)")
        if min(hz) < MIN_HZ:
            print(f"  ! a sensor is below {MIN_HZ} Hz; this take may have gaps")
        if (await ask("Enter = keep, d = discard: ")).lower() == "d":
            recorder.delete_clip(res["clip_id"])
            print("Discarded.")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=int, help="session number (default: next unused)")
    ap.add_argument("--port", help="serial port (default: SERIAL_PORT from backend/.env)")
    ap.add_argument("--labels", nargs="*", default=[], help="label names to show as a reminder")
    ap.add_argument("--seconds", type=float, default=TAKE_SECONDS, help="length of every take (default 3)")
    ap.add_argument("--fake", action="store_true", help="use the fake sensor stream (no hardware)")
    ap.add_argument("--upload", action="store_true", help="upload saved CSV clips to the database and exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    fake = args.fake or settings.fake_stream
    reader = Reader(args.port or settings.serial_port, settings.serial_baud, fake)
    db = Database(settings.tiger_db_url, store_raw=False)
    recorder = Recorder(reader, db, settings.data_dir, Transcriber(settings.whisper_model, settings.whisper_language), [])
    await db.start()
    if not db.enabled:
        print("TIGER_DB_URL is not set: takes are saved to CSV only (data/clips).")

    fake_task = None
    try:
        if args.upload:
            if not db.enabled:
                sys.exit("Set TIGER_DB_URL in backend/.env first.")
            await upload_backlog(db, recorder)
        else:
            if not fake and not (args.port or settings.serial_port):
                sys.exit("Set SERIAL_PORT in backend/.env (or pass --port).")
            reader.start()
            if fake:
                stream = FakeStream(reader, autoplay=False)
                recorder.on_record_start = lambda label: stream.burst(label if label in WORDS else WORDS[0], 0.3, 1.0)
                fake_task = asyncio.create_task(stream.run())
            await collect(recorder, reader, args.session or next_session(recorder), args.labels, args.seconds)
        if db.enabled:
            print("Sending to Tiger Data...")
            if await db.wait_flushed(20):
                print("All takes stored in Tiger Data.")
            else:
                print("Database not reachable. Takes are safe in data/clips; run `python -m backend.collect --upload` later.")
    finally:
        if fake_task:
            fake_task.cancel()
        reader.stop()
        await db.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Takes already recorded are saved in data/clips.")
