"""What-if vector control simulator: model-driven counterfactual forecasts.

How it works (no new model is trained):
  1. Crews working for some days in a ward remove breeding sites. The dataset has no spraying data, so the size of that
     effect is an openly stated assumption: each crew-day cuts the ward's effective water-logging risk by EFFECT_PER_CREW_DAY,
     with diminishing returns:   reduction = 1 - (1 - e) ** (crews x days)
  2. The ward's water-logging risk is lowered by that reduction and the SAME 4 weekly models are run again.
     What the model does with that change is learned from 10 years of data.
     The crews work DURING week 1, so week 1 stays as forecast and the effect shows in weeks 2 to 4.
  3. Beds and saline come from the same cases-to-supplies table as the rest of the app.
  4. Cases prevented = baseline forecast minus the what-if forecast. ROI = cases prevented per crew.
"""
import numpy as np
import pandas as pd

from .predict import forecast_weeks

EFFECT_PER_CREW_DAY = 0.06          # assumption: one crew working one day cuts effective water-logging risk by 6%
MAX_CREWS_PER_WARD = 5


def reduction(crews, days, e=EFFECT_PER_CREW_DAY):
    return 1 - (1 - e) ** (np.asarray(crews, float) * days)


def run(X: pd.DataFrame, crews, days: int, e: float = EFFECT_PER_CREW_DAY) -> pd.DataFrame:
    """4-week forecast (long format; week 1 unchanged, weeks 2-4 with the crews) for every ward row when `crews[i]` crews work `days` days in ward i."""
    X2 = X.copy()
    X2["water_logging_risk_index"] = X2["water_logging_risk_index"].astype(float) * (1 - reduction(crews, days, e))
    treated, base = forecast_weeks(X2), forecast_weeks(X)
    # crews work during week 1, so week 1 cannot change; only weeks 2-4 respond
    out = pd.concat([base[base["week"] == 1], treated[treated["week"] > 1]])
    return out.sort_values(["row", "week"]).reset_index(drop=True)


def ward_curves(X: pd.DataFrame, days: int, e: float = EFFECT_PER_CREW_DAY, kmax: int = MAX_CREWS_PER_WARD):
    """Total cases (weeks 1-4) in each ward when it gets 0..kmax crews. Returns array shape (kmax+1, n_wards)."""
    n = len(X)
    out = []
    for k in range(kmax + 1):
        f = run(X, np.full(n, k), days, e)
        out.append(f.groupby("row")["predicted_cases"].sum().reindex(X.index).to_numpy())
    return np.array(out)


def auto_deploy(cases_by_k: np.ndarray, total_crews: int) -> np.ndarray:
    """Give crews one at a time to the ward where the next crew prevents the most cases (diminishing returns)."""
    kmax, n = cases_by_k.shape[0] - 1, cases_by_k.shape[1]
    got = np.zeros(n, int)
    for _ in range(int(total_crews)):
        gain = np.array([cases_by_k[got[i], i] - cases_by_k[got[i] + 1, i] if got[i] < kmax else -1.0 for i in range(n)])
        i = int(np.argmax(gain))
        if gain[i] <= 1e-9:
            break                                   # no ward benefits from another crew
        got[i] += 1
    return got