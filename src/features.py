import numpy as np
import pandas as pd


def build_features(daily: pd.DataFrame, static: dict) -> pd.DataFrame:
    """daily: the last 35 completed days (oldest -> newest) with columns temp, rain, hum.
    static: the ward's 3 fixed features. Returns ONE row with the 14 model inputs."""
    n = len(daily)

    def week(col, k):                               # k=0: last 7 days, k=1: the 7 before...
        return daily[col].iloc[n - 7 * (k + 1): n - 7 * k]

    f = {"rainfall_mm": week("rain", 0).sum()}
    for k in range(1, 5):
        f[f"rainfall_mm_lag{k}w"] = week("rain", k).sum()
    f["avg_temp_c"] = week("temp", 0).mean()
    f["avg_temp_c_lag1w"] = week("temp", 1).mean()
    f["relative_humidity_pct"] = week("hum", 0).mean()
    f["relative_humidity_pct_lag1w"] = week("hum", 1).mean()

    # Time of year, from the first day of the latest 7-day window (same formula as training)
    iso_week = int(pd.Timestamp(daily.index[n - 7]).isocalendar()[1])
    f["woy_sin"] = float(np.sin(2 * np.pi * iso_week / 52))
    f["woy_cos"] = float(np.cos(2 * np.pi * iso_week / 52))

    f.update(static)
    return pd.DataFrame([f])