"""Hospital surge and overflow routing for one city, for one forecast week.

Capacity is NOT in the dataset, so it is an openly stated assumption:
    dengue beds in a ward = density / 10,000 x BEDS_PER_10K x share of those beds currently free
(at least 1 bed per ward). The officer can change both numbers.

Routing rules:
  * A ward is in SURGE when the beds it needs are more than its capacity; the gap is its overflow.
  * Wards with spare beds can receive patients, but only up to their spare beds.
  * Biggest overflow is placed first, each time into the NEAREST ward that still has room (same city only).
  * Overflow that no ward in the city can take is reported as a city-wide shortage, not hidden.
"""
import numpy as np
import pandas as pd

BEDS_PER_10K = 1.0              # dengue-designated beds per 10,000 people/sq km of density (assumption)
NEAR_FULL = 0.85                # at or above this share of capacity, a ward is flagged "near full"


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def ward_capacity(density, free_share, per_10k=BEDS_PER_10K):
    cap = np.asarray(density, float) / 10_000 * per_10k * free_share
    return np.maximum(np.rint(cap), 1).astype(int)


def plan_overflow(w: pd.DataFrame, free_share: float, per_10k: float = BEDS_PER_10K):
    """w: one row per ward with ward_no, density, latitude, longitude, beds_need.
    Returns (ward table, routes table, summary dict)."""
    r = w.reset_index(drop=True).copy()
    r["capacity"] = ward_capacity(r["density"], free_share, per_10k)
    r["need"] = r["beds_need"].astype(int)
    r["net"] = r["capacity"] - r["need"]                         # + spare, - deficit
    r["overflow"] = (-r["net"]).clip(lower=0)
    spare = r["net"].clip(lower=0).to_numpy().copy()

    routes, left = [], r["overflow"].to_numpy().copy()
    for i in np.argsort(-r["overflow"].to_numpy(), kind="stable"):
        if left[i] == 0:
            continue
        dist = haversine_km(r.loc[i, "latitude"], r.loc[i, "longitude"], r["latitude"].to_numpy(), r["longitude"].to_numpy())
        for j in np.argsort(dist, kind="stable"):
            if left[i] == 0:
                break
            if j == i or spare[j] == 0:
                continue
            n = int(min(left[i], spare[j]))
            routes.append({"from_ward": int(r.loc[i, "ward_no"]), "to_ward": int(r.loc[j, "ward_no"]),
                           "beds": n, "km": round(float(dist[j]), 1)})
            left[i] -= n
            spare[j] -= n
    r["rerouted"] = r["overflow"] - left
    r["unplaced"] = left
    r["received"] = 0
    for rt in routes:
        r.loc[r["ward_no"] == rt["to_ward"], "received"] += rt["beds"]
    r["utilization"] = np.where(r["capacity"] > 0, r["need"] / r["capacity"], 0.0)
    r["status"] = np.select(
        [r["unplaced"] > 0, r["overflow"] > 0, r["utilization"] >= NEAR_FULL],
        ["🔴 Surge: no room in city", "🟠 Surge: reroute", "🟡 Near full"], "🟢 Has room")
    routes = pd.DataFrame(routes, columns=["from_ward", "to_ward", "beds", "km"])

    cap, need = int(r["capacity"].sum()), int(r["need"].sum())
    summary = {"capacity": cap, "need": need, "utilization": need / cap if cap else 0.0,
               "surge_wards": int((r["overflow"] > 0).sum()), "overflow": int(r["overflow"].sum()),
               "rerouted": int(r["rerouted"].sum()), "unplaced": int(r["unplaced"].sum())}
    return r, routes, summary
