"""Per-session jaw/head relative-orientation ("tare") calibration.

Shared by training (ml/) and the live backend (backend/), same as preprocess.py.

Each BNO085 reports its own fused orientation (qw,qx,qy,qz), but settles on a
different internal reference axis depending on exactly how the hat sits that
day -- a remount changes the *apparent* relative orientation between the jaw
and head sensor even though the real anatomy relationship hasn't changed.
`preprocess_window`'s chin-minus-head gyro subtraction assumes both sensors'
raw axes are already aligned, so that drift becomes something the model has
to either ignore or (worse) learn as if it mattered.

Tare fixes this once per session instead of trying to make every window
robust to it: hold the jaw and head still in the same neutral pose right
after the hat goes on, call `tare()` on that buffer, and every later window
has the jaw's raw gyro/accel rotated into the head's frame before anything
else sees it -- so preprocess_window keeps its original assumption, and that
assumption is now true regardless of today's mounting.

Does not help `preprocess_window_invariant` at all -- that pipeline is
already invariant to each sensor's own mounting by construction (see
ml/test_orientation_invariance.py), so there is nothing for tare to fix there.
"""
from __future__ import annotations

import numpy as np

MIN_TARE_SAMPLES = 20  # ~0.2 s at 100 Hz; callers should capture ~1-2 s


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _average_quaternion(quats: np.ndarray) -> np.ndarray:
    """Mean of near-identical unit quaternions (e.g. one still pose), sign-aligned first.

    q and -q represent the same rotation; averaging without aligning signs can
    cancel instead of average. Good enough for a short static buffer where all
    samples are close together -- not a general orientation-averaging method
    (that needs an eigen-decomposition for samples that are spread out).
    """
    ref = quats[0]
    aligned = np.where((quats @ ref)[:, None] < 0, -quats, quats)
    mean = aligned.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-9:
        raise ValueError("quaternion buffer averaged to (near) zero; samples do not agree")
    return mean / norm


class TareCalibrator:
    """Holds the jaw-to-head rotation offset for one session.

    Call `tare(chin, head)` once at the start of a session with both sensors
    held still in the same neutral pose; call `apply(chin, head)` on every
    later window before preprocessing. Before `tare()` is called, `apply` is
    a no-op, so it is always safe to call.
    """

    def __init__(self) -> None:
        self._rotation: np.ndarray | None = None  # 3x3, jaw raw frame -> head raw frame

    @property
    def enabled(self) -> bool:
        return self._rotation is not None

    def tare(self, chin: np.ndarray, head: np.ndarray) -> None:
        """Compute the jaw->head rotation offset from a static buffer of both sensors.

        `chin`/`head` are raw rows in the shared Ring-row layout (cols 8:12 =
        qw,qx,qy,qz); both must be held still in the same neutral pose while
        this buffer was captured. Raises ValueError if the buffer is too
        short or has no real orientation data (e.g. the firmware's fusion
        hasn't produced its first quaternion yet).
        """
        if len(chin) < MIN_TARE_SAMPLES or len(head) < MIN_TARE_SAMPLES:
            raise ValueError(f"tare buffer too short: need >= {MIN_TARE_SAMPLES} samples per sensor")
        q_jaw = _average_quaternion(chin[:, 8:12])
        q_head = _average_quaternion(head[:, 8:12])
        if np.allclose(q_jaw, 0) or np.allclose(q_head, 0):
            raise ValueError("tare buffer has no orientation data (quaternions are all zero)")
        r_jaw, r_head = _quat_to_matrix(q_jaw), _quat_to_matrix(q_head)
        self._rotation = r_head.T @ r_jaw  # a jaw-frame vector, expressed in the head's frame

    def clear(self) -> None:
        self._rotation = None

    def apply(self, chin: np.ndarray, head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rotate the jaw's gyro+accel into the head's frame using the tared offset.

        Returns (chin, head) unchanged if `tare()` hasn't been called yet, so
        this is always safe to call even before calibration.
        """
        if self._rotation is None or len(chin) == 0:
            return chin, head
        out = chin.copy()
        out[:, 2:5] = chin[:, 2:5] @ self._rotation.T  # gyro
        out[:, 5:8] = chin[:, 5:8] @ self._rotation.T  # accel
        return out, head
