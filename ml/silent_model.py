"""The model object saved to backend/models/model.joblib.

The backend calls `model.predict_proba(x[None])` where x comes from
`ml.preprocess.preprocess_window`, i.e. input shape (n, N_CHANNELS, N_STEPS).
This wrapper accepts exactly that shape, so training and live inference agree.

Lives in its own module (not train.py) so the backend can unpickle it by
importing `ml.silent_model` without pulling in training-only code.
"""
from __future__ import annotations

import numpy as np

from ml.time_series_utils import extract_handcrafted_features

N_SEGMENTS = 5  # coarse temporal shape: per-channel mean/std in 5 equal parts of the word


def features(X: np.ndarray) -> np.ndarray:
    """(n, channels, steps) -> (n, features)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[None]
    summary = extract_handcrafted_features(X.transpose(0, 2, 1))  # utils expect (n, time, channels)
    parts = np.array_split(X, N_SEGMENTS, axis=2)
    shape = np.concatenate([np.stack([p.mean(axis=2), p.std(axis=2)], axis=2) for p in parts], axis=2)
    return np.concatenate([summary, shape.reshape(len(X), -1)], axis=1)


class SilentSpeechRF:
    """Random forest on summary + temporal-shape features of a preprocessed word."""

    def __init__(self, n_estimators: int = 500, random_state: int = 42) -> None:
        from sklearn.ensemble import RandomForestClassifier

        self.rf = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1
        )

    @property
    def classes_(self) -> np.ndarray:
        return self.rf.classes_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SilentSpeechRF":
        self.rf.fit(features(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict_proba(features(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict(features(X))
