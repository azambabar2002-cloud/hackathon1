"""Loads the 4 weekly models and turns model inputs into a 4-week forecast:
predicted cases, risk tier, and supplies needed, for each week separately."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "dengue_weekly_models.joblib"
LOOKUP_PATH = ROOT / "data" / "cases_to_supplies.csv"
SUPPLY_COLS = ["saline_bags_needed", "diagnostic_kits_needed", "hospital_beds_required"]

_bundle = None
_lookup = None


def load_bundle():
    global _bundle
    if _bundle is None:                             # load once, then reuse
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def load_lookup():
    global _lookup
    if _lookup is None:
        _lookup = pd.read_csv(LOOKUP_PATH).set_index("dengue_cases")
    return _lookup


def to_tier(cases):
    """Weekly cases -> tier index 0..3 (Low, Medium, High, Critical)."""
    cuts = load_bundle()["tier_cutoffs"]
    c = np.asarray(cases, dtype=float)
    return np.select([c <= cuts[0], c <= cuts[1], c <= cuts[2]], [0, 1, 2], 3)


def supplies(cases):
    """Weekly cases -> [saline, kits, beds] using the dataset's own cases-to-supplies table."""
    lk = load_lookup()
    idx = np.clip(np.rint(cases), 0, lk.index.max()).astype(int)
    return lk.loc[idx, SUPPLY_COLS].to_numpy()


def forecast_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """X: one row per ward holding the model inputs (any index).
    Returns one row per ward per forecast week (1-4)."""
    b = load_bundle()
    labels = b["labels"]
    Xf = X[b["features"]].astype(float)             # exact training column order
    parts = []
    for k, model in enumerate(b["models"], start=1):
        cases = np.clip(model.predict(Xf), 0, None)
        s = supplies(cases)
        parts.append(pd.DataFrame({
            "row": X.index, "week": k, "predicted_cases": cases,
            "risk": [labels[i] for i in to_tier(cases)],
            "saline_bags": s[:, 0], "diagnostic_kits": s[:, 1], "hospital_beds": s[:, 2]}))
    return pd.concat(parts).sort_values(["row", "week"]).reset_index(drop=True)