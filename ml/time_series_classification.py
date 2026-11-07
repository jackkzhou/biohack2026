"""Modular time-series classification baselines.

Expected input shape: (n_samples, n_timestamps, n_channels), with equal-length,
finite numeric series. Install only the dependencies for the model(s) you use:
numpy scikit-learn tslearn sktime torch
"""

from __future__ import annotations

from typing import Any


def _series_array(X: Any, name: str):
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"{name} must have shape (samples, time, channels); got {X.shape}")
    if X.shape[0] == 0 or X.shape[1] == 0 or X.shape[2] == 0:
        raise ValueError(f"{name} must not have an empty sample, time, or channel dimension")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return X


def _labels_array(y: Any, n_samples: int, name: str):
    import numpy as np

    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n_samples:
        raise ValueError(f"{name} must be 1D with {n_samples} labels; got shape {y.shape}")
    return y


def _check_train_data(X_train: Any, y_train: Any):
    X_train = _series_array(X_train, "X_train")
    y_train = _labels_array(y_train, len(X_train), "y_train")
    if len(set(y_train.tolist())) < 2:
        raise ValueError("Training data must contain at least two classes")
    return X_train, y_train


def evaluate_predictions(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Return accuracy and a per-class report for predictions."""
    from sklearn.metrics import accuracy_score, classification_report

    y_true = _labels_array(y_true, len(y_true), "y_true")
    y_pred = _labels_array(y_pred, len(y_true), "y_pred")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def fit_dtw_1nn(X_train: Any, y_train: Any):
    """Fit 1-nearest-neighbor using DTW distance (tslearn)."""
    from tslearn.neighbors import KNeighborsTimeSeriesClassifier

    X_train, y_train = _check_train_data(X_train, y_train)
    model = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
    return model.fit(X_train, y_train)


def extract_handcrafted_features(X: Any, sampling_period: float = 1.0):
    """Extract seven summary features per channel from dense 3D series."""
    import numpy as np

    X = _series_array(X, "X")
    if not np.isfinite(sampling_period) or sampling_period <= 0:
        raise ValueError("sampling_period must be finite and positive")

    # Local maxima only: a point must exceed its left neighbor and be no lower
    # than its right neighbor. The fixed series length supplies the duration.
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

    X_train, y_train = _check_train_data(X_train, y_train)
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

    X_train, y_train = _check_train_data(X_train, y_train)
    # sktime panel arrays use (instances, channels, timepoints).
    X_panel = np.transpose(X_train, (0, 2, 1))
    model = make_pipeline(
        MiniRocket(num_kernels=num_kernels, random_state=random_state, n_jobs=n_jobs),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 7)),
    )
    return model.fit(X_panel, y_train)


def predict_minirocket_ridge(model: Any, X: Any):
    import numpy as np

    X = _series_array(X, "X")
    return model.predict(np.transpose(X, (0, 2, 1)))


def _make_cnn(n_channels: int, n_classes: int):
    import torch.nn as nn

    class SmallTimeSeriesCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, n_classes))

        def forward(self, x):
            return self.classifier(self.features(x).squeeze(-1))

    return SmallTimeSeriesCNN()


def fit_cnn(
    X_train: Any,
    y_train: Any,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    noise_std: float = 0.02,
    amplitude_jitter: float = 0.1,
    random_state: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Train a small 1D CNN with train-only channel scaling, jitter, and noise."""
    import numpy as np
    import torch
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader, TensorDataset

    X_train, y_train = _check_train_data(X_train, y_train)
    if (
        epochs <= 0
        or batch_size <= 0
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if noise_std < 0 or amplitude_jitter < 0:
        raise ValueError("augmentation strengths must be non-negative")

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(y_train)
    channel_mean = X_train.mean(axis=(0, 1), keepdims=True)
    channel_std = X_train.std(axis=(0, 1), keepdims=True)
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)
    X_scaled = (X_train - channel_mean) / channel_std

    # Convert (samples, time, channels) to PyTorch's (samples, channels, time).
    features = torch.as_tensor(X_scaled.transpose(0, 2, 1), dtype=torch.float32)
    targets = torch.as_tensor(encoded_y, dtype=torch.long)
    generator = torch.Generator().manual_seed(random_state)
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    model = _make_cnn(X_train.shape[2], len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            scale = 1 + amplitude_jitter * torch.randn(
                (len(batch_X), X_train.shape[2], 1), device=device
            )
            augmented = batch_X * scale + noise_std * torch.randn_like(batch_X)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(augmented), batch_y).backward()
            optimizer.step()

    return {
        "model": model,
        "device": device,
        "label_encoder": label_encoder,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "n_channels": X_train.shape[2],
    }


def predict_cnn(model_bundle: dict[str, Any], X: Any, *, batch_size: int = 256):
    import numpy as np
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    X = _series_array(X, "X")
    if X.shape[2] != model_bundle["n_channels"]:
        raise ValueError(
            f"X has {X.shape[2]} channels; model expects {model_bundle['n_channels']}"
        )
    X_scaled = (X - model_bundle["channel_mean"]) / model_bundle["channel_std"]
    features = torch.as_tensor(X_scaled.transpose(0, 2, 1), dtype=torch.float32)
    loader = torch.utils.data.DataLoader(features, batch_size=batch_size)
    model = model_bundle["model"]
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_X in loader:
            logits = model(batch_X.to(model_bundle["device"]))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return model_bundle["label_encoder"].inverse_transform(np.asarray(predictions))


if __name__ == "__main__":
    # After defining X_train, y_train, X_test, and y_test, run any subset:
    #
    # dtw = fit_dtw_1nn(X_train, y_train)
    # dtw_result = evaluate_predictions(y_test, dtw.predict(X_test))
    #
    # forest = fit_random_forest(X_train, y_train, sampling_period=0.01)
    # forest_result = evaluate_predictions(
    #     y_test, predict_random_forest(forest, X_test, sampling_period=0.01)
    # )
    #
    # rocket = fit_minirocket_ridge(X_train, y_train)
    # rocket_result = evaluate_predictions(y_test, predict_minirocket_ridge(rocket, X_test))
    #
    # cnn = fit_cnn(X_train, y_train)
    # cnn_result = evaluate_predictions(y_test, predict_cnn(cnn, X_test))
    pass
