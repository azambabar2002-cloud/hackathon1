"""Constrained resource allocation for one city.

Inputs : ONE week of the city's forecast (one row per ward) and the officer's WEEKLY limits
         (stock of saline bags and test kits, total spraying crews). The dashboard runs this once per forecast week.
Outputs: what each ward receives, how much of its need is unmet, and a deficit flag.

Rules (all documented in the dashboard, so they can be defended in the Q&A):
  * The officer's stock is a weekly amount; each forecast week is allocated separately against that week's need.
  * Priority = 70% predicted cases (vs the city's busiest ward) + 30% population density (vs the densest ward).
  * Stock is handed out in ROUNDS of up to 50% of each ward's need:
      - Round 1: if there is enough for everyone's first 50%, they get it. If not, the stock is shared
        across ALL wards in proportion to priority x need, so no ward is left at zero.
      - Round 2 (the other 50%): wards are served in priority order until the stock runs out.
  * Spraying crews are whole units: crew need comes from the ward's highest weekly risk tier
    (Low 0, Medium 1, High 2, Critical 3) and crews are handed out one level at a time in priority order.
  * Every ward that gets less than it needs is flagged by the share that is unmet.
"""
import numpy as np
import pandas as pd

LABELS = ["Low", "Medium", "High", "Critical"]
RANK = {t: i for i, t in enumerate(LABELS)}

ROUND_SHARE = 0.5                                   # each round gives up to 50% of a ward's need
W_CASES, W_DENSITY = 0.7, 0.3                       # priority weights
CREWS_BY_TIER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

BANDS = [                                           # (max share unmet, label, emoji)
    (0.0, "Fully covered", "🟢"),
    (0.25, "Minor shortfall", "🟡"),
    (0.50, "Serious shortfall", "🟠"),
    (1.0, "Critical deficit", "🔴"),
]


# ----------------------------------------------------------------------------- step 1: demand + priority
def priority_scores(cases, density):
    cases, density = np.asarray(cases, float), np.asarray(density, float)
    c = cases / cases.max() if cases.max() > 0 else np.zeros_like(cases)
    d = density / density.max() if density.max() > 0 else np.zeros_like(density)
    return W_CASES * c + W_DENSITY * d


def build_ward_demand(fc: pd.DataFrame) -> pd.DataFrame:
    """fc: the city forecast for ONE week (or several; needs are summed), columns from predict.forecast_weeks plus ward info.
    Returns one row per ward with its need, risk tier, crew need and priority."""
    g = fc.groupby("ward_no")
    d = g.agg(ward_id=("ward_id", "first"), density=("population_density_sqkm", "first"),
              cases_4wk=("predicted_cases", "sum"), saline_need=("saline_bags", "sum"),
              kits_need=("diagnostic_kits", "sum"), beds_need=("hospital_beds", "max"))
    peak = fc.assign(r=fc["risk"].map(RANK)).groupby("ward_no")["r"].max()
    d["peak_risk"] = peak.map(lambda i: LABELS[i])
    d["crews_need"] = d["peak_risk"].map(CREWS_BY_TIER)
    d["priority"] = priority_scores(d["cases_4wk"], d["density"])
    for c in ("saline_need", "kits_need", "beds_need", "crews_need"):
        d[c] = d[c].astype(int)
    return d.reset_index()


# ----------------------------------------------------------------------------- step 2: share out the stock
def _proportional(target, weights, budget):
    """Share `budget` units across wards in proportion to weights x target, never above target, in whole units."""
    t = target.astype(float)
    w = weights * t
    alloc = np.zeros(len(t))
    remaining = float(budget)
    active = t > 0
    for _ in range(len(t) + 2):                      # water-filling: re-share what capped wards cannot take
        if remaining < 1e-9 or not active.any():
            break
        share = remaining * np.where(active, w, 0) / w[active].sum()
        give = np.minimum(share, t - alloc)
        alloc += give
        remaining -= give.sum()
        active = (t - alloc) > 1e-9
    base = np.floor(alloc + 1e-9).astype(int)
    left = int(round(budget - base.sum()))
    order = np.lexsort((-weights, -(alloc - base)))   # largest fraction first, ties to higher priority
    while left > 0:
        moved = False
        for i in order:
            if left > 0 and base[i] < target[i]:
                base[i] += 1
                left -= 1
                moved = True
        if not moved:
            break
    return base


def allocate_stock(demand, priority, stock, step=ROUND_SHARE):
    """Hand out `stock` units against ward `demand` (whole units). Returns units given to each ward."""
    d = np.asarray(demand, int)
    pri = np.asarray(priority, float)
    w = pri + 0.05                                   # so even a zero-priority ward gets a share
    stock = int(max(stock, 0))
    alloc = np.zeros(len(d), int)
    order = np.argsort(-pri, kind="stable")
    for _ in range(int(np.ceil(1 / step)) + 2):
        rem = d - alloc
        if stock <= 0 or rem.sum() <= 0:
            break
        target = np.minimum(rem, np.ceil(step * d).astype(int))
        if target.sum() == 0:
            break
        if target.sum() <= stock:                    # enough for the whole round: everyone gets their share
            alloc += target
            stock -= int(target.sum())
        elif alloc.sum() == 0:                       # round 1 and not enough: share across ALL wards by priority
            give = _proportional(target, w, stock)
            alloc += give
            stock -= int(give.sum())
        else:                                        # later rounds: strictly by priority order
            for i in order:
                g = int(min(target[i], stock))
                alloc[i] += g
                stock -= g
    return alloc


def allocate_crews(need, priority, total):
    """Whole crews, one level at a time: every ward that needs a crew gets its first before any gets a second."""
    need = np.asarray(need, int)
    pri = np.asarray(priority, float)
    order = np.argsort(-pri, kind="stable")
    got = np.zeros(len(need), int)
    left = int(max(total, 0))
    for level in range(1, (need.max() if len(need) else 0) + 1):
        for i in order:
            if left > 0 and need[i] >= level:
                got[i] += 1
                left -= 1
    return got


# ----------------------------------------------------------------------------- step 3: flag the shortfalls
def band(short_share: float) -> int:
    s = round(float(short_share), 6)
    for i, (limit, _, _) in enumerate(BANDS):
        if s <= limit:
            return i
    return len(BANDS) - 1


def run_allocation(dem: pd.DataFrame, saline_stock: int, kits_stock: int, crews_total: int):
    """Returns (per-ward result sorted by priority, summary dict)."""
    r = dem.sort_values(["priority", "density"], ascending=False).reset_index(drop=True)
    r.insert(0, "rank", np.arange(1, len(r) + 1))
    r["saline_given"] = allocate_stock(r["saline_need"], r["priority"], saline_stock)
    r["kits_given"] = allocate_stock(r["kits_need"], r["priority"], kits_stock)
    r["crews_given"] = allocate_crews(r["crews_need"], r["priority"], crews_total)

    items = {"saline": ("saline_need", "saline_given"), "kits": ("kits_need", "kits_given"),
             "crews": ("crews_need", "crews_given")}
    for name, (need, given) in items.items():
        r[f"{name}_short"] = r[need] - r[given]
        r[f"{name}_short_pct"] = np.where(r[need] > 0, r[f"{name}_short"] / r[need].where(r[need] > 0, 1), 0.0)
        r[f"{name}_band"] = r[f"{name}_short_pct"].map(band)
    r["worst_band"] = r[["saline_band", "kits_band", "crews_band"]].max(axis=1)
    worst_item = r[["saline_short_pct", "kits_short_pct", "crews_short_pct"]].idxmax(axis=1).str.replace("_short_pct", "")
    r["worst_item"] = np.where(r["worst_band"] == 0, "", worst_item)
    r["worst_short_pct"] = r[["saline_short_pct", "kits_short_pct", "crews_short_pct"]].max(axis=1)
    r["status"] = [f"{BANDS[b][2]} {BANDS[b][1]}" + (f": {it} {p:.0%} short" if b else "")
                   for b, it, p in zip(r["worst_band"], r["worst_item"], r["worst_short_pct"])]

    def total(need, given, stock):
        n, g = int(r[need].sum()), int(r[given].sum())
        return {"need": n, "stock": int(stock), "given": g, "short": n - g, "surplus": max(int(stock) - g, 0)}

    summary = {
        "saline": total("saline_need", "saline_given", saline_stock),
        "kits": total("kits_need", "kits_given", kits_stock),
        "crews": total("crews_need", "crews_given", crews_total),
        "wards": len(r), "with_demand": int(((r["saline_need"] + r["kits_need"] + r["crews_need"]) > 0).sum()),
        "by_band": {BANDS[i][1]: int((r["worst_band"] == i).sum()) for i in range(len(BANDS))},
    }
    return r, summary