"""Checks that the orientation-free preprocessing really ignores sensor orientation.

    python -m ml.test_orientation_invariance [path/to/clip.csv ...]

Rotating each sensor's gyro and accel by an arbitrary 3D rotation (a different one per
sensor, as if the IMUs were mounted differently) must not change the model input.
Without clip arguments it uses synthetic data.
"""
from __future__ import annotations

import sys

import numpy as np

from ml.preprocess import N_CHANNELS_INVARIANT, N_STEPS, preprocess_window, preprocess_window_invariant


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q  # proper rotation


def rotate(rows: np.ndarray, R: np.ndarray) -> np.ndarray:
    out = rows.copy()
    out[:, 2:5] = rows[:, 2:5] @ R.T
    out[:, 5:8] = rows[:, 5:8] @ R.T
    return out


def synthetic(rng: np.random.Generator, sensor: int, moving: bool) -> np.ndarray:
    t = np.arange(0, 1.5, 0.005) + 1e9
    n = len(t)
    env = np.exp(-(((t - t[0]) - 0.75) / 0.2) ** 2) if moving else np.zeros(n)
    g = rng.normal(0, 0.01, (n, 3)) + (env[:, None] * np.array([1.2, 0.4, -0.2]) * np.sin(2 * np.pi * 3 * (t - t[0]))[:, None])
    tilt = 0.3 * env
    a = np.stack([np.sin(tilt) * 9.8, np.zeros(n), np.cos(tilt) * 9.8], axis=1) + rng.normal(0, 0.03, (n, 3))
    rows = np.zeros((n, 13))
    rows[:, 0], rows[:, 1], rows[:, 2:5], rows[:, 5:8], rows[:, 12] = t, t - t[0], g, a, sensor
    return rows


def check(chin: np.ndarray, head: np.ndarray, rng: np.random.Generator, name: str) -> bool:
    base = preprocess_window_invariant(chin, head)
    assert base is not None and base.shape == (N_CHANNELS_INVARIANT, N_STEPS), base
    worst = 0.0
    for _ in range(20):
        x = preprocess_window_invariant(rotate(chin, random_rotation(rng)), rotate(head, random_rotation(rng)))
        worst = max(worst, float(np.abs(x - base).max()))
    old = preprocess_window(chin, head)
    old_rot = preprocess_window(rotate(chin, random_rotation(rng)), rotate(head, random_rotation(rng)))
    moved = float(np.abs(old - old_rot).max())
    ok = worst < 1e-3
    print(f"{'PASS' if ok else 'FAIL'} {name}: orientation-free changes by at most {worst:.2e} under rotation; "
          f"the default pipeline changes by {moved:.2f}")
    return ok


def main() -> None:
    rng = np.random.default_rng(0)
    results = [check(synthetic(rng, 0, True), synthetic(rng, 1, False), rng, "synthetic word")]
    for path in sys.argv[1:]:
        rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        results.append(check(rows[rows[:, 12] == 0], rows[rows[:, 12] == 1], rng, path.split("/")[-1]))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
