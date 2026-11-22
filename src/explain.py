"""Explains a single prediction: how much does each factor push the risk up or down
compared with an average week in an average ward?

Method: exact Shapley values over 10 factor groups (1,024 what-if evaluations of the
saved model, about 50 ms). Risk is measured as the probability-weighted tier
(Low = 0, Medium = 1, High = 2, Critical = 3), so the impacts always add up to the
full difference between this ward-week and an average one.
"""
import math

import numpy as np
import pandas as pd

from .predict import _load

# Dataset averages: the "typical week / typical ward" comparison point.
TYPICAL = {
    "population_density_sqkm": 43158.0, "water_logging_risk_index": 0.58, "sanitation_score": 0.55,
    "rainfall_mm": 35.6, "rainfall_mm_lag1w": 35.6, "rainfall_mm_lag2w": 35.6,
    "rainfall_mm_lag3w": 35.6, "rainfall_mm_lag4w": 35.6,
    "avg_temp_c": 26.9, "avg_temp_c_lag1w": 26.9,
    "relative_humidity_pct": 71.2, "relative_humidity_pct_lag1w": 71.2,
}

GROUPS = {
    "Rain this week": ["rainfall_mm"],
    "Rain 1 week ago": ["rainfall_mm_lag1w"],
    "Rain 2 weeks ago": ["rainfall_mm_lag2w"],
    "Rain 3 weeks ago": ["rainfall_mm_lag3w"],
    "Rain 4 weeks ago": ["rainfall_mm_lag4w"],
    "Temperature": ["avg_temp_c", "avg_temp_c_lag1w"],
    "Humidity": ["relative_humidity_pct", "relative_humidity_pct_lag1w"],
    "Water-logging risk": ["water_logging_risk_index"],
    "Sanitation": ["sanitation_score"],
    "Population density": ["population_density_sqkm"],
}


def _detail(name, x):
    v = lambda f: float(x[f].iloc[0])
    if name.startswith("Rain"):
        f = GROUPS[name][0]
        return f"{v(f):.0f} mm (average week: {TYPICAL[f]:.0f} mm)"
    if name == "Temperature":
        return f"{v('avg_temp_c'):.1f} °C now, {v('avg_temp_c_lag1w'):.1f} °C a week earlier"
    if name == "Humidity":
        return f"{v('relative_humidity_pct'):.0f}% now, {v('relative_humidity_pct_lag1w'):.0f}% a week earlier"
    if name == "Water-logging risk":
        return f"{v('water_logging_risk_index'):.2f} (average ward: 0.58)"
    if name == "Sanitation":
        return f"{v('sanitation_score'):.2f} (average ward: 0.55, higher is better)"
    return f"{v('population_density_sqkm'):,.0f} per sq km (average ward: 43,158)"


def driver_breakdown(x: pd.DataFrame):
    """x: one-row DataFrame of the 12 model features.
    Returns (DataFrame[driver, impact, detail] sorted ascending, average-week score, this-ward score)."""
    b = _load()
    m = b["model"]
    feats, labels = b["features"], b["labels"]
    x = x[feats].astype(float)

    names = list(GROUPS)
    n = len(names)
    col = {f: i for i, f in enumerate(feats)}
    states = np.arange(2 ** n)
    masks = ((states[:, None] >> np.arange(n)) & 1).astype(bool)      # which groups use the real value
    X = np.tile(np.array([TYPICAL[f] for f in feats]), (2 ** n, 1))
    for j, g in enumerate(names):
        for f in GROUPS[g]:
            X[masks[:, j], col[f]] = x[f].iloc[0]

    proba = m.predict_proba(pd.DataFrame(X, columns=feats))
    order = [list(m.classes_).index(l) for l in labels]
    v = (proba[:, order] * np.arange(len(labels))).sum(axis=1)         # expected tier 0..3

    fact = [math.factorial(k) for k in range(n + 1)]
    phi = np.zeros(n)
    for j in range(n):
        bit = 1 << j
        for s in range(2 ** n):
            if s & bit:
                continue
            k = bin(s).count("1")
            phi[j] += fact[k] * fact[n - k - 1] / fact[n] * (v[s | bit] - v[s])

    out = pd.DataFrame({"driver": names, "impact": phi,
                        "detail": [_detail(g, x) for g in names]}).sort_values("impact")
    return out.reset_index(drop=True), float(v[0]), float(v[-1])