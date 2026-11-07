"""Serial reader + rolling sample buffer.

ESP32 line format (one reading per line, 100 Hz per sensor):

    t_us,sensor,gx,gy,gz,ax,ay,az,qw,qx,qy,qz        sensor 0 = chin, 1 = head

The ESP32's own `t_us` gives precise spacing between samples. It is converted to
laptop time (seconds since epoch) with a clock offset (laptop - ESP32) that is
measured when the stream starts and refined while it runs, so samples line up
with Whisper timestamps, clip start/stop times and database rows.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

# Row layout of every stored sample (one row per reading):
#   0 t (laptop s, epoch) | 1 t_esp (s) | 2-4 gx gy gz | 5-7 ax ay az | 8-11 qw qx qy qz | 12 sensor
NCOLS = 13
SENSOR_CHIN, SENSOR_HEAD = 0, 1
GYRO = slice(2, 5)

log = logging.getLogger("reader")

BUFFER_SECONDS = 120
RING_CAPACITY = BUFFER_SECONDS * 120  # 100 Hz per sensor with headroom
_U32 = 2**32


class ClockSync:
    """Estimates offset = laptop_time - esp_time.

    Every sample gives `recv - esp = offset + delay` with delay >= 0 (serial/USB/
    Bluetooth latency, OS buffering). So the true offset is never above any
    single measurement: a lower measurement is adopted immediately, while the
    estimate only drifts upward slowly, following the lowest measurement of the
    last few seconds. This tracks crystal drift between the two clocks and
    ignores latency spikes.
    """

    WINDOW_S = 2.0
    KEEP_WINDOWS = 5
    UP_RATE = 0.2

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.offset: float | None = None
        self._win_min = math.inf
        self._win_start = 0.0
        self._mins: deque[float] = deque(maxlen=self.KEEP_WINDOWS)

    def update(self, esp_s: float, recv_s: float) -> float:
        d = recv_s - esp_s
        if self.offset is None:
            self.offset = d
            self._win_start = recv_s
        if d < self.offset:
            self.offset = d
        self._win_min = min(self._win_min, d)
        if recv_s - self._win_start >= self.WINDOW_S:
            self._mins.append(self._win_min)
            self._win_min, self._win_start = math.inf, recv_s
            target = min(self._mins)
            if target > self.offset:
                self.offset += self.UP_RATE * (target - self.offset)
        return self.offset


class Ring:
    """Fixed-size ring of sample rows, time-ordered."""

    def __init__(self, capacity: int = RING_CAPACITY) -> None:
        self.a = np.zeros((capacity, NCOLS), dtype=np.float64)
        self.cap = capacity
        self.n = 0
        self.pos = 0
        self.last_t = -math.inf

    def push(self, row: np.ndarray) -> None:
        self.a[self.pos] = row
        self.pos = (self.pos + 1) % self.cap
        self.n = min(self.n + 1, self.cap)
        self.last_t = row[0]

    def ordered(self) -> np.ndarray:
        if self.n < self.cap:
            return self.a[: self.n]
        return np.concatenate([self.a[self.pos :], self.a[: self.pos]])

    def window(self, t0: float, t1: float) -> np.ndarray:
        o = self.ordered()
        if len(o) == 0:
            return o.copy()
        lo = int(np.searchsorted(o[:, 0], t0, side="left"))
        hi = int(np.searchsorted(o[:, 0], t1, side="right"))
        return o[lo:hi].copy()


class Reader:
    def __init__(self, port: str = "", baud: int = 921600, fake: bool = False) -> None:
        self.port, self.baud, self.fake = port, baud, fake
        self._lock = threading.Lock()
        self._rings = {SENSOR_CHIN: Ring(), SENSOR_HEAD: Ring()}
        self._clock = ClockSync()
        self._last_raw: int | None = None
        self._wraps = 0
        self._outbox: deque[tuple[float, int, tuple, tuple]] = deque(maxlen=4000)
        self._sinks: list[Callable[[np.ndarray], None]] = []
        self._counts = [0, 0]
        self._rate_counts = [0, 0]
        self._rate_t = time.monotonic()
        self._rates = [0.0, 0.0]
        self.dropped = 0  # CSV-looking lines that could not be parsed
        self._last_rx = 0.0
        self._port_open = False
        self._ser = None  # open serial.Serial while connected (for send())
        self._skip_first = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self.fake:
            log.info("fake sensor stream enabled; no serial port opened")
            return
        if not self.port:
            log.warning("SERIAL_PORT is not set and FAKE_STREAM is off: no sensor data will arrive")
            return
        self._thread = threading.Thread(target=self._serial_loop, name="serial-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def send(self, data: bytes) -> bool:
        """Write a command line to the board (e.g. the orientation LED). False if the port is not open."""
        ser = self._ser
        if ser is None:
            return False
        try:
            ser.write(data)
            return True
        except Exception:  # noqa: BLE001
            return False

    def add_sink(self, fn: Callable[[np.ndarray], None]) -> None:
        """fn(row) is called for every parsed sample (any thread; must not block)."""
        self._sinks.append(fn)

    # ---- serial thread ---------------------------------------------------
    def _serial_loop(self) -> None:
        import serial  # imported here so fake mode works without pyserial installed

        warned = False
        while not self._stop.is_set():
            try:
                ser = serial.Serial(self.port, self.baud, timeout=0.2)
            except Exception as e:  # noqa: BLE001
                if not warned:
                    log.warning("cannot open serial port %s (%s); retrying", self.port, e)
                    warned = True
                self._port_open = False
                self._stop.wait(2.0)
                continue
            warned = False
            log.info("serial port %s open at %d baud", self.port, self.baud)
            try:
                ser.reset_input_buffer()  # drop stale bytes so the first clock offset is not skewed by backlog
                self._reset_stream()
                self._ser = ser
                self._port_open = True
                buf = bytearray()
                while not self._stop.is_set():
                    chunk = ser.read(ser.in_waiting or 1)
                    if not chunk:
                        continue
                    recv = time.time()
                    buf += chunk
                    while (i := buf.find(b"\n")) >= 0:
                        line = bytes(buf[:i])
                        del buf[: i + 1]
                        self.feed_line(line, recv)
                    if len(buf) > 4096:  # no newline in a long time: garbage
                        buf.clear()
            except Exception as e:  # noqa: BLE001
                log.warning("serial error: %s; reconnecting", e)
            finally:
                self._port_open = False
                self._ser = None
                try:
                    ser.close()
                except Exception:  # noqa: BLE001
                    pass
            self._stop.wait(1.0)

    def _reset_stream(self) -> None:
        with self._lock:
            self._clock.reset()
            self._last_raw = None
            self._wraps = 0
            self._skip_first = True

    # ---- parsing ---------------------------------------------------------
    def feed_line(self, line: bytes | str, recv: float | None = None) -> bool:
        """Parse one line and store it. Also used by the fake stream. Returns True if stored."""
        recv = time.time() if recv is None else recv
        if isinstance(line, bytes):
            line = line.decode("ascii", "ignore")
        line = line.strip()
        if not line:
            return False
        first, self._skip_first = self._skip_first, False
        parts = line.split(",")
        if len(parts) != 12:
            if line.count(",") >= 8 and not first:
                self.dropped += 1
            return False
        try:
            t_us = int(parts[0])
            sensor = int(parts[1])
            vals = [float(p) for p in parts[2:]]
            if sensor not in (0, 1) or not all(math.isfinite(v) for v in vals) or t_us < 0:
                raise ValueError
        except ValueError:
            if not first:
                self.dropped += 1
            return False

        with self._lock:
            esp_s = self._unwrap(t_us) / 1e6
            offset = self._clock.update(esp_s, recv)
            ring = self._rings[sensor]
            t = max(esp_s + offset, ring.last_t + 1e-6)  # keep each ring strictly increasing
            row = np.array([t, esp_s, *vals, sensor], dtype=np.float64)
            ring.push(row)
            self._counts[sensor] += 1
            self._outbox.append((esp_s * 1000.0, sensor, tuple(vals[0:3]), tuple(vals[3:6])))
            self._last_rx = time.monotonic()
        for fn in self._sinks:
            try:
                fn(row)
            except Exception:  # noqa: BLE001
                log.exception("sample sink failed")
        return True

    def _unwrap(self, raw: int) -> int:
        """t_us is a 32-bit microsecond counter (wraps every ~71 min); a reboot makes it jump back."""
        if self._last_raw is not None and raw < self._last_raw:
            back = self._last_raw - raw
            if back > _U32 // 2:
                self._wraps += 1  # genuine wrap-around
            elif back > 1_000_000:  # went back more than 1 s: the ESP32 rebooted
                log.info("ESP32 clock restarted; re-measuring clock offset")
                self._clock.reset()
                self._wraps = 0
        self._last_raw = raw
        return raw + self._wraps * _U32

    # ---- access ----------------------------------------------------------
    def window(self, sensor: int, t0: float, t1: float) -> np.ndarray:
        with self._lock:
            return self._rings[sensor].window(t0, t1)

    def window_both(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._rings[SENSOR_CHIN].window(t0, t1), self._rings[SENSOR_HEAD].window(t0, t1)

    def latest_time(self) -> float | None:
        """Laptop time of the newest sample (None before any data)."""
        with self._lock:
            t = max(r.last_t for r in self._rings.values())
        return None if t == -math.inf else t

    def drain_outbox(self) -> list[tuple[float, int, tuple, tuple]]:
        with self._lock:
            items = list(self._outbox)
            self._outbox.clear()
        return items

    @property
    def clock_offset(self) -> float | None:
        return self._clock.offset

    def rates(self) -> list[float]:
        """Samples/s per sensor over the time since the last call (call about once a second)."""
        with self._lock:
            now = time.monotonic()
            dt = max(now - self._rate_t, 1e-6)
            self._rates = [round((self._counts[i] - self._rate_counts[i]) / dt) for i in (0, 1)]
            self._rate_counts, self._rate_t = list(self._counts), now
            return list(self._rates)

    @property
    def connected(self) -> bool:
        """True while valid samples arrived in the last 2 s."""
        return (self.fake or self._port_open) and time.monotonic() - self._last_rx < 2.0
