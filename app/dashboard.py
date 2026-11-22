import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from src.features import build_features
from src.predict import predict_risk
from src.weather import fetch_daily_weather

LABELS = ["Low", "Medium", "High", "Critical"]
STATIC_COLS = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
ACTIONS = {
    "Low": "Routine surveillance. Keep drains clear and continue public awareness messaging.",
    "Medium": "Increase inspection of drains and water containers. Apply larvicide at known "
              "breeding sites and alert local clinics.",
    "High": "Deploy targeted larvicide and spraying teams. Run a community clean-up drive. "
            "Prepare hospital beds, test kits and saline.",
    "Critical": "Emergency response: immediate door-to-door source reduction, intensive "
                "spraying, stock hospitals (beds, saline, kits) and issue public warnings.",
}
SHOW = {"Low": st.success, "Medium": st.info, "High": st.warning, "Critical": st.error}


@st.cache_data
def load_wards():
    w = pd.read_csv(ROOT / "data" / "ward_static.csv")
    w["ward_no"] = w["ward_id"].str.extract(r"(\d+)$")[0].astype(int)
    return w


st.title("Dengue Outbreak Early Warning")
wards = load_wards()

city = st.selectbox("City", sorted(wards["city"].unique()))
ward_no = st.selectbox("Ward number", sorted(wards.loc[wards["city"] == city, "ward_no"]))
today = datetime.now(ZoneInfo("Asia/Dhaka")).date()
st.caption(f"Date: {today}")

if st.button("Predict"):
    row = wards[(wards["city"] == city) & (wards["ward_no"] == ward_no)].iloc[0]
    try:
        with st.spinner("Fetching weather..."):
            daily, stale = fetch_daily_weather(row["latitude"], row["longitude"], today)
    except Exception as e:
        st.error(f"Could not get weather data: {e}")
        st.stop()
    if stale:
        st.warning("Weather service unreachable. Showing the last saved data, "
                   f"which ends {daily.index[-1].date()}.")

    X = build_features(daily, {c: row[c] for c in STATIC_COLS})
    tier, proba = predict_risk(X)

    st.subheader(f"{city}, Ward {ward_no}")
    SHOW[tier](f"Risk for the next 28 days ({today} to {today + timedelta(days=28)}): **{tier}**")
    st.write("**Recommended action:**", ACTIONS[tier])

    chart = pd.DataFrame({"risk": LABELS, "probability": proba.values})
    st.altair_chart(
        alt.Chart(chart).mark_bar().encode(x=alt.X("risk", sort=LABELS), y="probability"),
        use_container_width=True)

    st.write("**Weather drivers (weekly rainfall, oldest to newest)**")
    weeks = ["4 weeks ago", "3 weeks ago", "2 weeks ago", "1 week ago", "Last 7 days"]
    rain = [X[f"rainfall_mm_lag{k}w"][0] for k in (4, 3, 2, 1)] + [X["rainfall_mm"][0]]
    rain_df = pd.DataFrame({"week": weeks, "rainfall_mm": rain})
    st.altair_chart(
        alt.Chart(rain_df).mark_bar().encode(x=alt.X("week", sort=weeks), y="rainfall_mm"),
        use_container_width=True)

    st.write("**All model inputs**")
    st.dataframe(X.T.rename(columns={0: "value"}).round(2))