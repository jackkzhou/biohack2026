"""Compatibility imports for the individual time-series model modules.

Prefer importing each model from its own module, for example
``from ml.random_forest import fit_random_forest``.
"""

from __future__ import annotations

if __package__:
    from .cnn import fit_cnn, predict_cnn
    from .dtw import fit_dtw_1nn
    from .minirocket import fit_minirocket_ridge, predict_minirocket_ridge
    from .random_forest import fit_random_forest, predict_random_forest
    from .time_series_utils import (
        check_train_data,
        evaluate_predictions,
        extract_handcrafted_features,
        labels_array,
        series_array,
    )
else:
    from cnn import fit_cnn, predict_cnn
    from dtw import fit_dtw_1nn
    from minirocket import fit_minirocket_ridge, predict_minirocket_ridge
    from random_forest import fit_random_forest, predict_random_forest
    from time_series_utils import (
        check_train_data,
        evaluate_predictions,
        extract_handcrafted_features,
        labels_array,
        series_array,
    )

_series_array = series_array
_labels_array = labels_array
_check_train_data = check_train_data

__all__ = [
    "evaluate_predictions",
    "extract_handcrafted_features",
    "fit_cnn",
    "fit_dtw_1nn",
    "fit_minirocket_ridge",
    "fit_random_forest",
    "predict_cnn",
    "predict_minirocket_ridge",
    "predict_random_forest",
]


if __name__ == "__main__":
    # Define X_train, y_train, X_test, and y_test, then call a model as needed.
    pass
