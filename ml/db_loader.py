from collections.abc import Iterator

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

_SESSION_QUERY = text(
    """
    SELECT time, label, ax, ay, az
    FROM mouth_sensor_data
    WHERE session_id = :session_id
    ORDER BY time ASC
    """
)


def _label_segments(labels: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield half-open index ranges for contiguous runs of equal labels."""
    if len(labels) == 0:
        return

    start = 0
    for index in range(1, len(labels)):
        if labels[index] != labels[start]:
            yield start, index
            start = index
    yield start, len(labels)


def load_session_data(
    db_url: str,
    session_id: str,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one session as overlapping (windows, time, 3) data and window labels.

    A window is included only when all its rows have the same label. Adjacent
    windows advance by one row. If no complete window is available, X has shape
    (0, window_size, 3) and y has shape (0,).
    """
    if not isinstance(window_size, int) or isinstance(window_size, bool):
        raise TypeError("window_size must be an integer")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be blank")
    session_id = session_id.strip()

    url = make_url(db_url)
    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+psycopg2")
    engine = create_engine(url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                _SESSION_QUERY, {"session_id": session_id}
            ).fetchall()
    finally:
        engine.dispose()

    if not rows:
        return (
            np.empty((0, window_size, 3), dtype=np.float32),
            np.empty((0,), dtype=object),
        )

    labels = np.asarray([row.label for row in rows], dtype=object)
    values = np.asarray(
        [[row.ax, row.ay, row.az] for row in rows], dtype=np.float32
    )
    if not np.isfinite(values).all():
        raise ValueError(f"Session {session_id!r} contains non-finite sensor values")

    windows: list[np.ndarray] = []
    window_labels: list[object] = []
    for start, stop in _label_segments(labels):
        segment_length = stop - start
        if segment_length < window_size:
            continue
        for offset in range(segment_length - window_size + 1):
            window_start = start + offset
            windows.append(values[window_start : window_start + window_size])
            window_labels.append(labels[window_start])

    if not windows:
        return (
            np.empty((0, window_size, 3), dtype=np.float32),
            np.empty((0,), dtype=object),
        )

    return np.stack(windows), np.asarray(window_labels)
