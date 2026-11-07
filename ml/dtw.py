"""Dynamic time warping 1-nearest-neighbor baseline."""

from __future__ import annotations

from typing import Any

if __package__:
    from .time_series_utils import check_train_data
else:
    from time_series_utils import check_train_data


def fit_dtw_1nn(X_train: Any, y_train: Any):
    """Fit 1-nearest-neighbor using DTW distance (requires tslearn)."""
    from tslearn.neighbors import KNeighborsTimeSeriesClassifier

    X_train, y_train = check_train_data(X_train, y_train)
    model = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
    return model.fit(X_train, y_train)


# ---------------------------------------------------------------------------
# Template matching with a self-contained DTW (numpy only)
# ---------------------------------------------------------------------------


def downsample(X, factor: int = 2):
    """(n, channels, steps) -> average every `factor` steps (DTW cost grows with steps squared)."""
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    steps = X.shape[-1] // factor * factor
    return X[..., :steps].reshape(*X.shape[:-1], steps // factor, factor).mean(axis=-1)


def dtw_to_templates(query, templates, band: int):
    """DTW distance from one series to many templates at once.

    query: (channels, steps); templates: (n, channels, steps). `band` is the largest allowed
    time shift in steps (Sakoe-Chiba band): a word cannot be stretched arbitrarily far, which
    also keeps it fast. All templates advance through the DP together, so the python loop is
    only steps x band long. Returns (n,) distances (root of the summed squared differences).
    """
    import numpy as np

    n, _, steps = templates.shape
    inf = np.float32(np.inf)
    prev = np.full((n, steps + 1), inf, dtype=np.float32)
    prev[:, 0] = 0.0
    for i in range(steps):
        cur = np.full((n, steps + 1), inf, dtype=np.float32)
        local = ((templates - query[None, :, i : i + 1]) ** 2).sum(axis=1)  # (n, steps)
        lo, hi = max(0, i - band), min(steps, i + band + 1)
        for j in range(lo, hi):
            cur[:, j + 1] = local[:, j] + np.minimum(np.minimum(prev[:, j], prev[:, j + 1]), cur[:, j])
        prev = cur
    return np.sqrt(prev[:, steps])
