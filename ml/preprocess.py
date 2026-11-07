"""Preprocessing shared by training (ml/) and the live backend (backend/).

Both sides must call `preprocess_window` so a live word looks exactly like a
training clip. The ML side owns this file; the backend only imports it.

Sample layout (one row per IMU reading, one array per sensor):

    col  0   t        laptop time, seconds since epoch
    col  1   t_esp    ESP32 time, seconds (unwrapped, from t_us)
    cols 2-4 gx gy gz gyro, rad/s
    cols 5-7 ax ay az accel, m/s^2
    cols 8-11 qw qx qy qz
    col 12   sensor   0 = chin, 1 = head
"""
from __future__ import annotations

import numpy as np

COLUMNS = ["t", "t_esp", "gx", "gy", "gz", "ax", "ay", "az", "qw", "qx", "qy", "qz", "sensor"]
NCOLS = len(COLUMNS)
SENSOR_CHIN, SENSOR_HEAD = 0, 1
GYRO = slice(2, 5)
ACCEL = slice(5, 8)

N_STEPS = 100  # every word is resampled to this many time steps
N_CHANNELS = 15  # chin g/a (6) + head g/a (6) + chin-gyro minus head-gyro (3)
MIN_SAMPLES = 5


def _interp(src_t: np.ndarray, src: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    """Interpolate each column of `src` (n, k) from times src_t onto dst_t."""
    return np.stack([np.interp(dst_t, src_t, src[:, i]) for i in range(src.shape[1])], axis=1)


def activity_signal(chin: np.ndarray, head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chin motion with the head's motion subtracted. Returns (times, rad/s magnitude)."""
    if len(chin) < 2 or len(head) < 2:
        return np.empty(0), np.empty(0)
    head_g = _interp(head[:, 0], head[:, GYRO], chin[:, 0])
    return chin[:, 0], np.linalg.norm(chin[:, GYRO] - head_g, axis=1)


def activity_level(chin: np.ndarray, head: np.ndarray) -> float:
    """RMS of the head-referenced chin gyro magnitude (0.0 if there is too little data)."""
    _, a = activity_signal(chin, head)
    return float(np.sqrt(np.mean(a**2))) if len(a) else 0.0


def trim_to_activity(chin: np.ndarray, head: np.ndarray, threshold: float, pad_s: float = 0.1) -> tuple[float, float]:
    """(t0, t1) of the active part of a window; the whole window if nothing is active."""
    t, a = activity_signal(chin, head)
    if len(t) == 0:
        return float(chin[0, 0]), float(chin[-1, 0])
    idx = np.flatnonzero(a > threshold)
    if len(idx) == 0:
        return float(t[0]), float(t[-1])
    return max(float(t[0]), float(t[idx[0]]) - pad_s), min(float(t[-1]), float(t[idx[-1]]) + pad_s)


def preprocess_window(
    chin: np.ndarray,
    head: np.ndarray,
    *,
    trim: bool = True,
    threshold: float = 0.12,
) -> np.ndarray | None:
    """Turn raw rows of both sensors into a model input of shape (N_CHANNELS, N_STEPS).

    Steps: trim to the active part, resample both sensors onto one uniform time
    grid, add the head-referenced gyro, remove each channel's mean (cancels
    gravity and how the sensor happens to be mounted today).
    Returns None when there is too little data.
    """
    if len(chin) < MIN_SAMPLES or len(head) < MIN_SAMPLES:
        return None
    t0, t1 = trim_to_activity(chin, head, threshold) if trim else (float(chin[0, 0]), float(chin[-1, 0]))
    if t1 - t0 < 0.05:
        t0, t1 = float(chin[0, 0]), float(chin[-1, 0])
    if t1 - t0 <= 0:
        return None
    grid = np.linspace(t0, t1, N_STEPS)
    cols = np.r_[2:8]
    c = _interp(chin[:, 0], chin[:, cols], grid)
    h = _interp(head[:, 0], head[:, cols], grid)
    x = np.concatenate([c, h, c[:, :3] - h[:, :3]], axis=1)  # (N_STEPS, N_CHANNELS)
    x = x - x.mean(axis=0, keepdims=True)
    return x.T.astype(np.float32)
