"""MiniRocket feature extraction with a ridge classifier."""

from __future__ import annotations

from typing import Any

if __package__:
    from .time_series_utils import check_train_data, series_array
else:
    from time_series_utils import check_train_data, series_array


def fit_minirocket_ridge(
    X_train: Any,
    y_train: Any,
    *,
    num_kernels: int = 10_000,
    random_state: int = 42,
    n_jobs: int = 1,
):
    """Fit sktime MiniRocket features followed by a tuned ridge classifier."""
    import numpy as np
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline
    from sktime.transformations.panel.rocket import MiniRocket

    X_train, y_train = check_train_data(X_train, y_train)
    X_panel = np.transpose(X_train, (0, 2, 1))
    model = make_pipeline(
        MiniRocket(num_kernels=num_kernels, random_state=random_state, n_jobs=n_jobs),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 7)),
    )
    return model.fit(X_panel, y_train)


def predict_minirocket_ridge(model: Any, X: Any):
    import numpy as np

    X = series_array(X, "X")
    return model.predict(np.transpose(X, (0, 2, 1)))
