import json, time
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _get_json(lat, lon, today):
    """1 API call per location per day; cached on disk; retries; stale fallback."""
    lat, lon = round(lat, 2), round(lon, 2)       # nearby wards share a cache file
    cache_file = CACHE_DIR / f"{lat}_{lon}_{today.isoformat()}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text()), False

    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_mean,precipitation_sum",
        "hourly": "relative_humidity_2m",
        "past_days": 35, "forecast_days": 1,
        "timezone": "Asia/Dhaka",
    }
    last_err = None
    for attempt in range(3):                       # retry with backoff
        try:
            r = requests.get(API_URL, params=params, timeout=10)
            r.raise_for_status()
            js = r.json()
            cache_file.write_text(json.dumps(js))
            return js, False
        except (requests.RequestException, ValueError) as e:
            last_err = e
            time.sleep(2 ** attempt)

    older = sorted(CACHE_DIR.glob(f"{lat}_{lon}_*.json"))   # fallback: last good data
    if older:
        return json.loads(older[-1].read_text()), True
    raise RuntimeError(f"Weather API unavailable and no cached data: {last_err}")


def fetch_daily_weather(lat, lon, today):
    """Returns (DataFrame of the last 35 COMPLETED days, is_stale)."""
    js, stale = _get_json(lat, lon, today)
    d = pd.DataFrame({
        "date": pd.to_datetime(js["daily"]["time"]),
        "temp": js["daily"]["temperature_2m_mean"],
        "rain": js["daily"]["precipitation_sum"],
    }).set_index("date")
    h = pd.DataFrame({"t": pd.to_datetime(js["hourly"]["time"]),
                      "hum": js["hourly"]["relative_humidity_2m"]})
    d["hum"] = h.groupby(h["t"].dt.normalize())["hum"].mean()   # hourly -> daily mean

    d = d[d.index < pd.Timestamp(today)].tail(35)    # exclude today (incomplete)
    if len(d) < 35 or d.isna().any().any():
        raise ValueError("Weather data incomplete (need 35 full days, no gaps)")
    return d, stale