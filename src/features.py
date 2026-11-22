import pandas as pd


def build_features(daily: pd.DataFrame, static: dict) -> pd.DataFrame:
    n = len(daily)                                  # 35 rows, oldest -> newest

    def week(col, k):                               # k=0: last 7 days, k=1: the 7 before...
        return daily[col].iloc[n - 7 * (k + 1): n - 7 * k]

    f = {"rainfall_mm": week("rain", 0).sum()}
    for k in range(1, 5):
        f[f"rainfall_mm_lag{k}w"] = week("rain", k).sum()
    f["avg_temp_c"] = week("temp", 0).mean()
    f["avg_temp_c_lag1w"] = week("temp", 1).mean()
    f["relative_humidity_pct"] = week("hum", 0).mean()
    f["relative_humidity_pct_lag1w"] = week("hum", 1).mean()
    f.update(static)
    return pd.DataFrame([f])