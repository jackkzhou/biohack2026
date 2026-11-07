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


# ---------------------------------------------------------------------------
# Orientation-free variant
# ---------------------------------------------------------------------------
# The default pipeline above feeds the model raw per-axis gyro/accel. Those
# depend on how each sensor happens to be mounted and tilted (and the
# chin-minus-head gyro assumes both are mounted identically), so a model can
# learn "the way the sensors sat on training day" instead of the word.
#
# Everything below is built only from quantities that do not change when a
# sensor's frame is rotated: vector lengths, an angle between two vectors of the
# SAME sensor, and the projection onto the window's own dominant rotation axis.
# Each sensor is treated independently, so the two IMUs can be mounted any way.

N_CHANNELS_INVARIANT = 10
INVARIANT_CHANNELS = [
    "chin |gyro|", "chin gyro on main axis", "chin gyro off main axis", "chin |dyn accel|", "chin tilt change",
    "head |gyro|", "head |dyn accel|", "head tilt change",
    "chin gyro along gravity", "chin dyn accel along gravity",
]


def activity_signal_invariant(chin: np.ndarray, head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chin rotation speed minus head rotation speed (both lengths, so no axis alignment needed)."""
    if len(chin) < 2 or len(head) < 2:
        return np.empty(0), np.empty(0)
    head_speed = np.interp(chin[:, 0], head[:, 0], np.linalg.norm(head[:, GYRO], axis=1))
    return chin[:, 0], np.maximum(0.0, np.linalg.norm(chin[:, GYRO], axis=1) - head_speed)


def activity_level_invariant(chin: np.ndarray, head: np.ndarray) -> float:
    _, a = activity_signal_invariant(chin, head)
    return float(np.sqrt(np.mean(a**2))) if len(a) else 0.0


def _trim_invariant(chin: np.ndarray, head: np.ndarray, threshold: float, pad_s: float = 0.1) -> tuple[float, float]:
    t, a = activity_signal_invariant(chin, head)
    if len(t) == 0:
        return float(chin[0, 0]), float(chin[-1, 0])
    idx = np.flatnonzero(a > threshold)
    if len(idx) == 0:
        return float(t[0]), float(t[-1])
    return max(float(t[0]), float(t[idx[0]]) - pad_s), min(float(t[-1]), float(t[idx[-1]]) + pad_s)


def _main_axis_projection(g: np.ndarray) -> np.ndarray:
    """Gyro (T, 3) projected on its own dominant rotation axis. The sign is fixed by making the
    largest excursion positive, so the result does not depend on the sensor's axis directions."""
    if not np.any(g):
        return np.zeros(len(g))
    _, _, vt = np.linalg.svd(g, full_matrices=False)  # uncentred: principal direction of the motion
    p = g @ vt[0]
    return -p if p[np.argmax(np.abs(p))] < 0 else p


def _along_gravity(g: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Components of the gyro and of the dynamic accel along the window's mean accel (= gravity direction).
    Both vectors are from the same sensor, so mounting does not matter. Worth ~6 points of accuracy."""
    m = a.mean(axis=0)
    up = m / (np.linalg.norm(m) + 1e-9)
    return g @ up, (a - m) @ up


def _tilt_change(a: np.ndarray) -> np.ndarray:
    """Angle (rad) between each accel vector and the window's mean accel (= gravity direction)."""
    m = a.mean(axis=0)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(m)
    cos = np.divide(a @ m, denom, out=np.ones(len(a)), where=denom > 1e-9)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def preprocess_window_invariant(
    chin: np.ndarray,
    head: np.ndarray,
    *,
    trim: bool = True,
    threshold: float = 0.12,
) -> np.ndarray | None:
    """Like `preprocess_window`, but the output (N_CHANNELS_INVARIANT, N_STEPS) is unchanged by
    rotating either sensor. Returns None when there is too little data."""
    if len(chin) < MIN_SAMPLES or len(head) < MIN_SAMPLES:
        return None
    t0, t1 = _trim_invariant(chin, head, threshold) if trim else (float(chin[0, 0]), float(chin[-1, 0]))
    if t1 - t0 < 0.05:
        t0, t1 = float(chin[0, 0]), float(chin[-1, 0])
    if t1 - t0 <= 0:
        return None
    grid = np.linspace(t0, t1, N_STEPS)
    cg, ca = _interp(chin[:, 0], chin[:, GYRO], grid), _interp(chin[:, 0], chin[:, ACCEL], grid)
    hg, ha = _interp(head[:, 0], head[:, GYRO], grid), _interp(head[:, 0], head[:, ACCEL], grid)
    c_speed = np.linalg.norm(cg, axis=1)
    main = _main_axis_projection(cg)
    off = np.sqrt(np.maximum(c_speed**2 - main**2, 0.0))
    c_grav_gyro, c_grav_acc = _along_gravity(cg, ca)
    x = np.stack(
        [
            c_speed, main, off, np.linalg.norm(ca - ca.mean(axis=0), axis=1), _tilt_change(ca),
            np.linalg.norm(hg, axis=1), np.linalg.norm(ha - ha.mean(axis=0), axis=1), _tilt_change(ha),
            c_grav_gyro, c_grav_acc,
        ],
        axis=1,
    )
    return x.T.astype(np.float32)


PREPROCESSORS = {"default": preprocess_window, "orientation-free": preprocess_window_invariant}


def get_preprocessor(name: str):
    """'default' (raw axes) or 'orientation-free'. Models record which one they were trained with."""
    try:
        return PREPROCESSORS[name]
    except KeyError:
        raise ValueError(f"unknown preprocessing {name!r}; choose from {sorted(PREPROCESSORS)}") from None
