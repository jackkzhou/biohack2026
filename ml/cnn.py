"""Small 1D convolutional neural network for time-series classification."""

from __future__ import annotations

from typing import Any

if __package__:
    from .time_series_utils import check_train_data, series_array
else:
    from time_series_utils import check_train_data, series_array


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
    """Train a 1D CNN with train-only channel scaling, jitter, and noise."""
    import numpy as np
    import torch
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader, TensorDataset

    X_train, y_train = check_train_data(X_train, y_train)
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
    X = series_array(X, "X")
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
