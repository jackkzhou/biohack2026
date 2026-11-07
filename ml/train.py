"""Train the word classifier from collected clips and install it for the backend.

    python -m ml.train                       # held-out = latest session
    python -m ml.train --test-session 2      # pick the held-out session
    python -m ml.train --source record learn # also use Learn-mode clips

Reads the clips written by `backend.collect` / the Record page (data/clips/*.csv),
runs every clip through the same `preprocess_window` the live backend uses,
evaluates on a held-out session, then refits on ALL clips and writes:

    backend/models/model.joblib         loaded by backend.inference.ModelClassifier
    backend/models/labels.json          class order
    backend/models/metrics.json         served at GET /model/metrics
    backend/models/confusion_matrix.png for the slides

Restart uvicorn afterwards so the backend loads the new model.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ml.preprocess import SENSOR_CHIN, SENSOR_HEAD, preprocess_window
from ml.silent_model import SilentSpeechRF

REPO = Path(__file__).resolve().parent.parent
CLIP_RE = re.compile(r"(\d+)_(\w+?)_(.*)_s(-?\d+)\.csv$")  # same naming as backend/recorder.py
MIN_PER_CLASS = 15


def load_clips(clip_dir: Path, sources: list[str], threshold: float, vocab: set[str] | None):
    X, y, sessions, skipped = [], [], [], 0
    spelling: dict[str, str] = {}  # "jordan" -> "Jordan" (first spelling seen), so typos in case don't split a class
    for p in sorted(clip_dir.glob("*.csv")):
        m = CLIP_RE.match(p.name)
        if not m or m[2] not in sources:
            continue
        label = spelling.setdefault(m[3].lower(), m[3])
        if vocab is not None and label.lower() not in vocab:
            continue  # e.g. Learn mode saves every word Whisper heard ("the", "I", ...)
        rows = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
        if rows.size == 0:
            skipped += 1
            continue
        chin, head = rows[rows[:, 12] == SENSOR_CHIN], rows[rows[:, 12] == SENSOR_HEAD]
        x = preprocess_window(chin, head, threshold=threshold)
        if x is None:
            skipped += 1
            continue
        X.append(x)
        y.append(label)
        sessions.append(int(m[4]))
    if skipped:
        print(f"skipped {skipped} clip(s) with too little data")
    return np.stack(X) if X else np.empty((0,)), np.array(y), np.array(sessions)


def split(y: np.ndarray, sessions: np.ndarray, test_session: int | None, seed: int):
    uniq = sorted(set(sessions.tolist()))
    if test_session is None and len(uniq) >= 2:
        test_session = uniq[-1]
    if test_session is not None:
        te = sessions == test_session
        if not te.any() or te.all():
            sys.exit(f"session {test_session} can't be the held-out set (sessions present: {uniq})")
        return ~te, te, f"held-out session {test_session} (train on sessions {[s for s in uniq if s != test_session]})"
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=seed)
    mtr, mte = np.zeros(len(y), bool), np.zeros(len(y), bool)
    mtr[tr], mte[te] = True, True
    print("! only one session: using a random 25% split, which OVERSTATES accuracy. Record a second session.")
    return mtr, mte, "random 25% split (single session)"


def save_confusion_png(cm: np.ndarray, labels: list[str], acc: float, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping confusion_matrix.png")
        return
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(labels), 0.6 + 0.9 * len(labels)), dpi=160)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Held-out accuracy {acc:.0%}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=REPO / "data" / "clips")
    ap.add_argument("--out", type=Path, default=REPO / "backend" / "models")
    ap.add_argument("--test-session", type=int, help="session to hold out (default: the latest one)")
    ap.add_argument("--source", nargs="+", default=["record"], help="clip sources to use: record and/or learn")
    ap.add_argument("--vocab", nargs="*", help="only these labels (default: all; with learn clips use the 5 words + nothing)")
    ap.add_argument("--threshold", type=float, default=0.12, help="activity threshold for trimming (match ACTIVITY_THRESHOLD)")
    ap.add_argument("--conf", type=float, default=0.6, help="confidence threshold to report (match CONF_THRESHOLD)")
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    vocab = {v.lower() for v in args.vocab} if args.vocab else None
    if vocab is None and "learn" in args.source:
        sys.exit("With --source learn, pass --vocab (e.g. --vocab help yes no Jordan water nothing)")
    X, y, sessions = load_clips(args.clips, args.source, args.threshold, vocab)
    if len(y) == 0:
        sys.exit(f"no usable clips in {args.clips}")
    counts = Counter(y.tolist())
    print(f"{len(y)} clips, sessions {sorted(set(sessions.tolist()))}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if len(counts) < 2:
        sys.exit("need at least two labels")
    for k, v in counts.items():
        if v < MIN_PER_CLASS:
            print(f"! '{k}' has only {v} clips; aim for {MIN_PER_CLASS}+ per session")

    tr, te, how = split(y, sessions, args.test_session, args.seed)
    model = SilentSpeechRF(args.trees, args.seed).fit(X[tr], y[tr])
    proba = model.predict_proba(X[te])
    pred = model.classes_[proba.argmax(1)]
    conf = proba.max(1)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    labels = sorted(counts)
    acc = float(accuracy_score(y[te], pred))
    cm = confusion_matrix(y[te], pred, labels=labels)
    report = classification_report(y[te], pred, labels=labels, output_dict=True, zero_division=0)
    kept = conf >= args.conf
    acc_kept = float((pred[kept] == y[te][kept]).mean()) if kept.any() else 0.0

    print(f"\nEvaluation: {how}")
    print(classification_report(y[te], pred, labels=labels, zero_division=0))
    print("Confusion matrix (rows = actual, cols = predicted):")
    w = max(len(l) for l in labels) + 1
    print(" " * w + "".join(f"{l[:6]:>7}" for l in labels))
    for l, row in zip(labels, cm):
        print(f"{l:<{w}}" + "".join(f"{v:>7}" for v in row))
    print(f"\nHeld-out accuracy {acc:.1%} on {int(te.sum())} clips")
    print(f"At confidence >= {args.conf}: {kept.mean():.0%} of words accepted, {acc_kept:.1%} of those correct")

    final = SilentSpeechRF(args.trees, args.seed).fit(X, y)  # deploy on everything, report the held-out numbers
    args.out.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(final, args.out / "model.joblib")
    (args.out / "labels.json").write_text(json.dumps([str(c) for c in final.classes_]))
    metrics = {
        "model": "random_forest",
        "evaluation": how,
        "accuracy": round(acc, 4),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "n_total": int(len(y)),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_class": {l: {"precision": round(report[l]["precision"], 3), "recall": round(report[l]["recall"], 3),
                          "support": int(report[l]["support"])} for l in labels},
        "confidence_threshold": args.conf,
        "accepted_fraction": round(float(kept.mean()), 4),
        "accuracy_when_accepted": round(acc_kept, 4),
        "class_counts": dict(sorted(counts.items())),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_confusion_png(cm, labels, acc, args.out / "confusion_matrix.png")
    print(f"\nSaved model.joblib, labels.json, metrics.json, confusion_matrix.png to {args.out}")
    print("Restart uvicorn to load the new model.")


if __name__ == "__main__":
    main()
