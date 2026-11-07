"""Shared validation and feature helpers for time-series models."""

from __future__ import annotations

from typing import Any


def series_array(X: Any, name: str):
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"{name} must have shape (samples, time, channels); got {X.shape}")
    if X.shape[0] == 0 or X.shape[1] == 0 or X.shape[2] == 0:
        raise ValueError(f"{name} must not have an empty sample, time, or channel dimension")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return X


def labels_array(y: Any, n_samples: int, name: str):
    import numpy as np

    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n_samples:
        raise ValueError(f"{name} must be 1D with {n_samples} labels; got shape {y.shape}")
    return y


def check_train_data(X_train: Any, y_train: Any):
    X_train = series_array(X_train, "X_train")
    y_train = labels_array(y_train, len(X_train), "y_train")
    if len(set(y_train.tolist())) < 2:
        raise ValueError("Training data must contain at least two classes")
    return X_train, y_train


def evaluate_predictions(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Return accuracy and a per-class report for predictions."""
    from sklearn.metrics import accuracy_score, classification_report

    y_true = labels_array(y_true, len(y_true), "y_true")
    y_pred = labels_array(y_pred, len(y_true), "y_pred")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def extract_handcrafted_features(X: Any, sampling_period: float = 1.0):
    """Extract seven summary features per channel from dense 3D series."""
    import numpy as np

    X = series_array(X, "X")
    if not np.isfinite(sampling_period) or sampling_period <= 0:
        raise ValueError("sampling_period must be finite and positive")

    peaks = np.sum(
        (X[:, 1:-1, :] > X[:, :-2, :]) & (X[:, 1:-1, :] >= X[:, 2:, :]),
        axis=1,
    )
    if X.shape[1] < 3:
        peaks = np.zeros((len(X), X.shape[2]), dtype=np.int64)

    features = np.stack(
        [
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            peaks,
            np.mean(np.square(X), axis=1),
            np.full((len(X), X.shape[2]), X.shape[1] * sampling_period),
        ],
        axis=2,
    )
    return features.reshape(len(X), -1)
