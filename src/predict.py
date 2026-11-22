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


def predict_risk(X: pd.DataFrame):
    b = _load()
    m = b["model"]
    X = X[b["features"]]                            # exact training column order
    tier = m.predict(X)[0]
    proba = pd.DataFrame(m.predict_proba(X), columns=m.classes_)[b["labels"]].iloc[0]
    return tier, proba                              # proba ordered Low->Critical