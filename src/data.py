"""Load, check and split the ECG Heartbeat Categorization data.

Source: Kachuee et al. (2018), derived from MIT-BIH; Kaggle CSV copy with
87,554 training beats + 21,892 test beats = 109,446 beats, 187 samples each
(125 Hz), 5 classes.  Column 187 is the label.

Split policy (fixed, shared by every model):
  * test  = the official mitbih_test.csv, untouched until the final evaluation
  * train/val = stratified 90/10 split of mitbih_train.csv (split_seed = 42)

IMPORTANT LIMITATION: there are no patient IDs, so the split is beat-level and
beats of the same patient can sit in train, val and test.  Do not claim
new-patient generalisation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["N", "S", "V", "F", "Q"]  # normal, supraventricular, ventricular, fusion, unknown
N_SAMPLES = 187
SPLIT_SEED = 42


def _read(path):
    arr = pd.read_csv(path, header=None, dtype=np.float32).to_numpy()
    return arr[:, :N_SAMPLES], arr[:, N_SAMPLES].astype(np.int64)


def load_official(data_dir="data/raw"):
    d = Path(data_dir)
    X_tr, y_tr = _read(d / "mitbih_train.csv")
    X_te, y_te = _read(d / "mitbih_test.csv")
    return X_tr, y_tr, X_te, y_te


def make_splits(data_dir="data/raw", val_size=0.1, split_seed=SPLIT_SEED):
    """Return dict with X_train/y_train/X_val/y_val/X_test/y_test."""
    X, y, X_te, y_te = load_official(data_dir)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=split_seed
    )
    return dict(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val,
                X_test=X_te, y_test=y_te)


def class_weights(y, n_classes=5):
    """Inverse-frequency weights, normalised so that the mean weight is 1."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = len(y) / (n_classes * np.maximum(counts, 1))
    return (w / w.mean()).astype(np.float32)


def duplicate_report(X_a, X_b):
    """Number of rows of X_b that also occur (bit-identical) in X_a."""
    seen = {r.tobytes() for r in np.ascontiguousarray(X_a)}
    return sum(r.tobytes() in seen for r in np.ascontiguousarray(X_b))


def summary(splits):
    rows = []
    for k in ("train", "val", "test"):
        y = splits[f"y_{k}"]
        c = np.bincount(y, minlength=len(CLASS_NAMES))
        rows.append({"split": k, "n": len(y), **{CLASS_NAMES[i]: int(c[i]) for i in range(len(c))}})
    return pd.DataFrame(rows)
