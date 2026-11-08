"""Daily IMU feature summaries for ALS progression monitoring.

Turns a day's recorded jaw/speech activity (mouthed-word clips, real-time
detections, rest-pose calibrations) into a small set of statistical features
that feed the 14-day rolling window consumed by `backend/progression.py`.

Everything here is a heuristic proxy, not a validated clinical biomarker:
tremor frequency, jaw asymmetry and cadence are useful *trend* signals for a
within-person baseline, not an absolute diagnostic measurement.

Row layout matches `ml/preprocess.py` (t, t_esp, gx,gy,gz, ax,ay,az, qw,qx,qy,qz, sensor).
"""
from __future__ import annotations

import numpy as np

from ml.preprocess import ACCEL, GYRO, SENSOR_CHIN, SENSOR_HEAD, activity_signal

TREMOR_BAND_HZ = (3.0, 12.0)  # physiological + low-frequency pathological tremor range
MIN_CLIP_SAMPLES = 10
MIN_TREMOR_SAMPLES = 20


def _by_sensor(rows: np.ndarray, sensor: int) -> np.ndarray:
    if len(rows) == 0:
        return rows
    return rows[rows[:, 12] == sensor]


def _euler_ranges(chin: np.ndarray) -> tuple[float, float] | None:
    """(pitch range, yaw range) in radians from the chin quaternion, or None if too little data."""
    if len(chin) < MIN_CLIP_SAMPLES:
        return None
    qw, qx, qy, qz = (chin[:, 8 + i] for i in range(4))
    if not np.any(qw) and not np.any(qx):  # orientation never populated (older firmware/model)
        return None
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return float(pitch.max() - pitch.min()), float(yaw.max() - yaw.min())


def jaw_asymmetry(chin: np.ndarray) -> float | None:
    """0 = jaw rotation spread evenly across pitch/yaw, near 1 = dominated by a single axis."""
    ranges = _euler_ranges(chin)
    if ranges is None:
        return None
    pitch_range, yaw_range = ranges
    total = pitch_range + yaw_range
    if total < 1e-6:
        return None
    return abs(pitch_range - yaw_range) / total


def velocity_variance(chin: np.ndarray, head: np.ndarray) -> float | None:
    """Variance of the head-referenced chin angular-velocity magnitude (rad/s)^2."""
    _, mag = activity_signal(chin, head)
    if len(mag) < MIN_CLIP_SAMPLES:
        return None
    return float(np.var(mag))


def _head_referenced_gyro(chin: np.ndarray, head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(times, 3-axis gyro) with the head's rotation subtracted. Unlike `activity_signal`'s norm,
    this keeps sign/axis information so a single-axis oscillation doesn't get full-wave rectified
    (which would double its apparent frequency under an FFT)."""
    if len(chin) < 2 or len(head) < 2:
        return np.empty(0), np.empty((0, 3))
    head_g = np.stack([np.interp(chin[:, 0], head[:, 0], head[:, 2 + i]) for i in range(3)], axis=1)
    return chin[:, 0], chin[:, GYRO] - head_g


def tremor_frequency_hz(chin: np.ndarray, head: np.ndarray) -> float | None:
    """Dominant frequency of head-referenced jaw rotation within the tremor band, or None."""
    t, gyro = _head_referenced_gyro(chin, head)
    if len(t) < MIN_TREMOR_SAMPLES:
        return None
    duration = t[-1] - t[0]
    if duration <= 0:
        return None
    fs = (len(t) - 1) / duration
    if fs < 2 * TREMOR_BAND_HZ[1]:  # cannot resolve the tremor band at this sample rate
        return None
    grid_t = np.linspace(t[0], t[-1], len(t))
    power = np.zeros(len(grid_t) // 2 + 1)
    for axis in range(3):
        grid_axis = np.interp(grid_t, t, gyro[:, axis])
        power += np.abs(np.fft.rfft(grid_axis - np.mean(grid_axis))) ** 2
    freqs = np.fft.rfftfreq(len(grid_t), d=1.0 / fs)
    band = (freqs >= TREMOR_BAND_HZ[0]) & (freqs <= TREMOR_BAND_HZ[1])
    if not np.any(band) or not np.any(power[band] > 0):
        return None
    return float(freqs[band][np.argmax(power[band])])


def clip_features(rows: np.ndarray) -> dict[str, float | None]:
    """Per-clip feature dict; any feature that could not be computed is None."""
    chin, head = _by_sensor(rows, SENSOR_CHIN), _by_sensor(rows, SENSOR_HEAD)
    return {
        "velocity_variance": velocity_variance(chin, head),
        "tremor_frequency_hz": tremor_frequency_hz(chin, head),
        "jaw_asymmetry_index": jaw_asymmetry(chin),
        "duration_s": float(chin[-1, 0] - chin[0, 0]) if len(chin) >= 2 else None,
    }


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def speech_cadence_wpm(detections: list[dict]) -> float | None:
    """Accepted detections per minute, from the first to the last accepted detection that day."""
    accepted = sorted(d["at_ms"] for d in detections if d.get("accepted"))
    if len(accepted) < 2:
        return None
    span_min = (accepted[-1] - accepted[0]) / 60_000.0
    if span_min <= 0:
        return None
    return len(accepted) / span_min


def aggregate_daily_features(
    clips: list[tuple[dict, np.ndarray]],
    detections: list[dict],
    rest_poses: list[dict],
) -> dict:
    """One day's worth of clips/detections/rest-poses -> the feature dict sent to Gemini."""
    per_clip = [clip_features(rows) for _, rows in clips if len(rows) >= MIN_CLIP_SAMPLES]
    accepted = [d for d in detections if d.get("accepted")]
    return {
        "n_clips": len(clips),
        "n_detections": len(detections),
        "n_accepted_detections": len(accepted),
        "velocity_variance": _mean([c["velocity_variance"] for c in per_clip]),
        "tremor_frequency_hz": _mean([c["tremor_frequency_hz"] for c in per_clip]),
        "jaw_asymmetry_index": _mean([c["jaw_asymmetry_index"] for c in per_clip]),
        "mean_clip_duration_s": _mean([c["duration_s"] for c in per_clip]),
        "speech_cadence_wpm": speech_cadence_wpm(detections),
        "detection_confidence_mean": _mean([d["confidence"] for d in detections]),
        "rejection_rate": (1 - len(accepted) / len(detections)) if detections else None,
        "rest_activity_rms": _mean([r["activity_rms"] for r in rest_poses]),
    }
