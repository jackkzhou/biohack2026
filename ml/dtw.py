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
