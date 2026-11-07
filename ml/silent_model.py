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

    preprocess = "default"  # which ml.preprocess pipeline the training input came from (set by ml/train.py)

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


class SilentSpeechCNN:
    """1D CNN (ml.cnn) behind the same interface as SilentSpeechRF.

    Takes (n, channels, steps) like the live backend produces, trains with
    `ml.cnn.fit_cnn`, and exposes `predict_proba` / `classes_`. Pickles as a
    plain state_dict (the network class is defined inside a factory function,
    so the module itself can't be pickled), and always runs on CPU once loaded.
    """

    preprocess = "default"  # which ml.preprocess pipeline the training input came from (set by ml/train.py)

    def __init__(self, epochs: int = 40, random_state: int = 42) -> None:
        self.epochs, self.random_state = epochs, random_state
        self._net = None
        self._mean = self._std = None
        self.classes_ = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SilentSpeechCNN":
        from ml.cnn import fit_cnn

        bundle = fit_cnn(np.asarray(X, dtype=np.float32).transpose(0, 2, 1), y, epochs=self.epochs,
                         random_state=self.random_state)
        self._net = bundle["model"].to("cpu").eval()
        self._mean, self._std = bundle["channel_mean"], bundle["channel_std"]
        self.classes_ = bundle["label_encoder"].classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None]
        x = (X.transpose(0, 2, 1) - self._mean) / self._std  # (n, steps, channels), train-set scaling
        with torch.no_grad():
            logits = self._net(torch.as_tensor(x.transpose(0, 2, 1), dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_net"] = {k: v.cpu() for k, v in self._net.state_dict().items()} if self._net is not None else None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if state["_net"] is not None:
            from ml.cnn import _make_cnn

            net = _make_cnn(len(self._mean.reshape(-1)), len(self.classes_))
            net.load_state_dict(state["_net"])
            self._net = net.eval()


class SilentSpeechDTW:
    """DTW template matching: compare a new movement with every saved example of each word.

    For each word the score is the mean of its k smallest DTW distances (k nearest templates),
    so one odd recording cannot decide the result. Probabilities are a softmax over the negative
    scores, scaled by the typical gap between words so the numbers are usable as confidences.
    Same interface as the other models: (n, channels, steps) in, `predict_proba` / `classes_` out.
    """

    preprocess = "default"  # which ml.preprocess pipeline the training input came from (set by ml/train.py)

    def __init__(self, k: int = 3, band: int = 10, factor: int = 2) -> None:
        self.k, self.band, self.factor = k, band, factor
        self.classes_ = np.array([])
        self._templates = None
        self._labels = None
        self._scale = None
        self._temp = 1.0

    def _prep(self, X: np.ndarray) -> np.ndarray:
        from ml.dtw import downsample

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None]
        return downsample(X, self.factor) / self._scale[None, :, None]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SilentSpeechDTW":
        X = np.asarray(X, dtype=np.float32)
        self._scale = X.std(axis=(0, 2)).astype(np.float32) + 1e-6  # every channel counts equally
        self._labels = np.asarray(y)
        self.classes_ = np.unique(self._labels)
        self._templates = self._prep(X)
        self._calibrate()
        return self

    def _calibrate(self, n: int = 150) -> None:
        """Pick the softmax temperature that makes the confidences honest: score some training
        clips against all the OTHER templates and choose the temperature with the lowest log-loss."""
        rng = np.random.default_rng(0)
        idx = rng.choice(len(self._templates), size=min(n, len(self._templates)), replace=False)
        rows, truth = [], []
        for i in idx:
            keep = np.arange(len(self._templates)) != i
            rows.append(self._scores(self._templates[i], self._templates[keep], self._labels[keep]))
            truth.append(int(np.flatnonzero(self.classes_ == self._labels[i])[0]))
        D, truth = np.array(rows), np.array(truth)
        best, self._temp = np.inf, 1.0
        for t in np.geomspace(D.std() * 0.02, D.std() * 5, 60):
            z = -D / t
            z -= z.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            nll = -logp[np.arange(len(truth)), truth].mean()
            if nll < best:
                best, self._temp = nll, float(t)

    def _scores(self, q: np.ndarray, templates: np.ndarray, labels: np.ndarray) -> np.ndarray:
        from ml.dtw import dtw_to_templates

        d = dtw_to_templates(q, templates, self.band)
        out = []
        for c in self.classes_:
            dc = np.sort(d[labels == c])
            out.append(dc[: self.k].mean() if len(dc) else np.inf)
        return np.array(out)

    def distances(self, X: np.ndarray) -> np.ndarray:
        """(n, classes) mean-of-k-nearest DTW distance per word (smaller = more similar)."""
        Q = self._prep(X)
        return np.stack([self._scores(q, self._templates, self._labels) for q in Q])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        s = -self.distances(X) / self._temp
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.distances(X).argmin(1)]
