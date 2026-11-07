"""Random forest baseline with handcrafted time-series features."""

from __future__ import annotations

from typing import Any

if __package__:
    from .time_series_utils import check_train_data, extract_handcrafted_features
else:
    from time_series_utils import check_train_data, extract_handcrafted_features


def fit_random_forest(
    X_train: Any,
    y_train: Any,
    *,
    sampling_period: float = 1.0,
    n_estimators: int = 500,
    random_state: int = 42,
):
    """Fit a scikit-learn random forest on per-channel summary features."""
    from sklearn.ensemble import RandomForestClassifier

    X_train, y_train = check_train_data(X_train, y_train)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(extract_handcrafted_features(X_train, sampling_period), y_train)
    return model


def predict_random_forest(model: Any, X: Any, *, sampling_period: float = 1.0):
    return model.predict(extract_handcrafted_features(X, sampling_period))
