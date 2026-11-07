"""Load training clips from Tiger Data.

Reads the schema backend/db.py actually writes (`clips` + `clip_samples`),
not the `mouth_sensor_data` table in db/schema.sql, which nothing in the
running system ever created or wrote to. Each clip's two sensors are turned
into one training window via `ml.preprocess.preprocess_window` -- the same
function backend/inference.py calls on live data, so a training example is
built exactly the way a live word is.

Caveat carried over unchanged from live inference: the head IMU's gyro is
subtracted from the jaw IMU's gyro axis-by-axis (see preprocess_window), which
assumes the two are mounted in roughly the same orientation. The firmware
doesn't stream real orientation (quaternion columns are always 0), so there is
no data available here to correct for the head and jaw IMUs actually being
mounted differently -- this loader cannot fix that, only match what inference
already does with the same raw axes.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

if __package__:
    from .preprocess import preprocess_window
else:
    from preprocess import preprocess_window

SENSOR_CHIN, SENSOR_HEAD = 0, 1

_CLIPS_QUERY = text("SELECT id, label FROM clips WHERE source = :source ORDER BY id")
_SAMPLES_QUERY = text(
    """
    SELECT time, t_us, sensor, gx, gy, gz, ax, ay, az
    FROM clip_samples
    WHERE clip_id = :clip_id
    ORDER BY time
    """
)


def _sensor_rows(records: list, sensor: int) -> np.ndarray:
    """DB rows for one sensor -> Ring-row layout [t, t_esp, g(3), a(3), q(4), sensor].

    Quaternion columns are filled with 0, matching backend/db.py's get_clip
    (the firmware never sends real orientation).
    """
    sel = [r for r in records if r.sensor == sensor]
    if not sel:
        return np.empty((0, 13), dtype=np.float64)
    return np.array(
        [[r.time.timestamp(), r.t_us / 1e6, r.gx, r.gy, r.gz, r.ax, r.ay, r.az, 0, 0, 0, 0, r.sensor] for r in sel],
        dtype=np.float64,
    )


def load_clips(db_url: str, source: str = "record") -> tuple[np.ndarray, np.ndarray]:
    """Load every stored clip as one preprocessed window each.

    `source` matches the `clips.source` column backend/collect.py writes
    ("record" for explicitly labeled takes); change it to pull other sources.
    Returns (X, y) with X.shape == (n, N_STEPS, N_CHANNELS) -- the
    (samples, time, channels) convention every model in ml/ expects via
    time_series_utils.series_array -- and y[i] that window's label.
    preprocess_window itself returns (N_CHANNELS, N_STEPS), so each window is
    transposed before stacking. Clips with too little data on either sensor
    are skipped.
    """
    url = make_url(db_url)
    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+psycopg2")
    engine = create_engine(url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            clips = connection.execute(_CLIPS_QUERY, {"source": source}).fetchall()
            windows: list[np.ndarray] = []
            labels: list[str] = []
            for clip in clips:
                records = connection.execute(_SAMPLES_QUERY, {"clip_id": clip.id}).fetchall()
                chin = _sensor_rows(records, SENSOR_CHIN)
                head = _sensor_rows(records, SENSOR_HEAD)
                x = preprocess_window(chin, head)
                if x is None:
                    continue
                windows.append(x.T)  # (N_CHANNELS, N_STEPS) -> (N_STEPS, N_CHANNELS)
                labels.append(clip.label)
    finally:
        engine.dispose()

    if not windows:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0,), dtype=object)
    return np.stack(windows), np.asarray(labels, dtype=object)
