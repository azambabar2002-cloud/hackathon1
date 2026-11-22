from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dengue_model.joblib"
_bundle = None


def _load():
    global _bundle
    if _bundle is None:                             # load once, then reuse
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_batch(X: pd.DataFrame):
    """Predict many rows at once. Returns (tiers array, probabilities DataFrame Low->Critical)."""
    b = _load()
    m = b["model"]
    X = X[b["features"]].astype(float)              # exact training column order
    tiers = m.predict(X)
    proba = pd.DataFrame(m.predict_proba(X), columns=m.classes_)[b["labels"]]
    return tiers, proba


def predict_risk(X: pd.DataFrame):
    """Single-row convenience wrapper."""
    tiers, proba = predict_batch(X)
    return tiers[0], proba.iloc[0]