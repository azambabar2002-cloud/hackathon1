import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.explain import driver_breakdown
from src.features import build_features
from src.predict import predict_batch
from src.weather import fetch_daily_weather

st.set_page_config(page_title="Dengue Early Warning", page_icon="🦟", layout="wide")

# ----------------------------------------------------------------------------- constants
LABELS = ["Low", "Medium", "High", "Critical"]
COLORS = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#EF6C00", "Critical": "#C62828"}
TEXT_ON = {"Low": "#FFFFFF", "Medium": "#1F2933", "High": "#FFFFFF", "Critical": "#FFFFFF"}
EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
EXPECTED = {"Low": "0-1 cases", "Medium": "2-19 cases", "High": "20-37 cases", "Critical": "38 or more cases"}
STATIC_COLS = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
AVG_WEEKLY_RAIN = 35.6

ACTIONS = {
    "Low": ("Routine surveillance", [
        "Keep drains, gutters and water containers clear",
        "Continue public awareness messages",
        "Keep routine fever and case reporting running"]),
    "Medium": ("Step up prevention", [
        "Inspect drains, tyres, rooftop tanks and construction sites for standing water",
        "Apply larvicide at known breeding sites",
        "Alert local clinics and health workers to watch for dengue-like fever",
        "Prepare a clean-up drive for the ward"]),
    "High": ("Targeted control now", [
        "Send larvicide and spraying teams to the most water-logged lanes first",
        "Run a community clean-up drive to empty standing water",
        "Prepare hospital beds, test kits and saline in nearby facilities",
        "Brief ward councillors and community health workers"]),
    "Critical": ("Emergency response", [
        "Start door-to-door source reduction and intensive spraying immediately",
        "Stock hospitals: beds, saline and dengue test kits",
        "Issue public warnings (SMS, mosque announcements, social media)",
        "Escalate to the city corporation health department"]),
}

# ----------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
.block-container {padding-top: 1.0rem; max-width: 1250px;}
.hero-title {font-size: 2.2rem; font-weight: 800; color: #0B3C5D; margin: 0.6rem 0 0 0; line-height: 1.15;}
.hero-sub {font-size: 1.05rem; color: #52606D; margin: 0.2rem 0 1rem 0;}
.risk-card {border-radius: 20px; padding: 24px 28px; box-shadow: 0 6px 18px rgba(11,60,93,0.18);}
.risk-label {font-size: 0.95rem; opacity: 0.92; letter-spacing: 0.04em; text-transform: uppercase;}
.risk-tier {font-size: 3rem; font-weight: 800; line-height: 1.1; margin: 4px 0 8px 0;}
.risk-sub {font-size: 1.0rem; opacity: 0.95; margin: 2px 0;}
.step-card {background: white; border: 1px solid #D5E6F0; border-radius: 16px; padding: 18px 20px; height: 100%;
            box-shadow: 0 2px 8px rgba(11,60,93,0.06);}
.step-num {display: inline-block; width: 30px; height: 30px; line-height: 30px; text-align: center; border-radius: 50%;
           background: #12698A; color: white; font-weight: 700; margin-right: 8px;}
.step-card h4 {margin: 0 0 6px 0; color: #0B3C5D;}
.note-box {background: #FFF8E1; border-left: 5px solid #F9A825; padding: 10px 14px; border-radius: 8px; color: #5D4A00;}
.driver-box {background: #E3EFF6; border-left: 5px solid #12698A; padding: 10px 14px; border-radius: 8px; color: #0B3C5D;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- data + model (cached)
@st.cache_data
def load_wards():
    w = pd.read_csv(ROOT / "data" / "ward_static.csv")
    w["ward_no"] = w["ward_id"].str.extract(r"(\d+)$")[0].astype(int)
    return w


@st.cache_data(show_spinner=False, ttl=1800)
def city_forecast(city: str, today_iso: str):
    """One weather request per city (city centre); every ward is scored with its own fixed features."""
    cw = load_wards().query("city == @city").reset_index(drop=True)
    daily, stale = fetch_daily_weather(cw["latitude"].mean(), cw["longitude"].mean(), date.fromisoformat(today_iso))
    X = pd.concat(
        [build_features(daily, {c: r[c] for c in STATIC_COLS}) for _, r in cw.iterrows()], ignore_index=True
    )
    tiers, proba = predict_batch(X)
    out = pd.concat([cw, proba.add_prefix("p_")], axis=1)
    out["tier"] = tiers
    out["score"] = (proba.values * np.arange(len(LABELS))).sum(axis=1)
    return out, X, daily, stale


@st.cache_data(show_spinner=False, ttl=1800)
def drivers_for(city: str, ward_no: int, today_iso: str):
    out, X, _, _ = city_forecast(city, today_iso)
    idx = int(out.index[out["ward_no"] == ward_no][0])
    return driver_breakdown(X.loc[[idx]])


def driver_sentence(df: pd.DataFrame) -> str:
    up = df[df.impact >= 0.05].sort_values("impact", ascending=False).head(3)
    down = df[df.impact <= -0.05].sort_values("impact").head(2)
    parts = []
    if len(up):
        parts.append("Pushing risk **up**: " + ", ".join(up.driver.str.lower()))
    if len(down):
        parts.append("Holding risk **down**: " + ", ".join(down.driver.str.lower()))
    return " | ".join(parts) or "This ward and week look close to average, so no single factor stands out."


# ----------------------------------------------------------------------------- header
wards = load_wards()
today = datetime.now(ZoneInfo("Asia/Dhaka")).date()
end_day = today + timedelta(days=28)

st.image(str(ROOT / "assets" / "banner.png"), width="stretch")
st.markdown('<p class="hero-title">Dengue Outbreak Early Warning</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Know which wards are heading for a dengue surge 2-4 weeks before hospitals fill up.</p>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- sidebar inputs
with st.sidebar:
    st.header("1. Choose an area")
    city = st.selectbox("City corporation", sorted(wards["city"].unique()), key="city")
    ward_options = sorted(wards.loc[wards["city"] == city, "ward_no"])
    ward_no = st.selectbox("Ward number", ward_options, key="ward")
    st.caption(f"📅 Today: **{today:%A, %d %B %Y}**")

    sel = wards[(wards["city"] == city) & (wards["ward_no"] == ward_no)].iloc[0]
    st.markdown("**Ward vulnerability (fixed)**")
    st.progress(float(sel["water_logging_risk_index"]), text=f"Water-logging risk: {sel['water_logging_risk_index']:.2f}")
    st.progress(float(1 - sel["sanitation_score"]), text=f"Poor sanitation: {1 - sel['sanitation_score']:.2f}")
    st.progress(float(sel["population_density_sqkm"] / wards["population_density_sqkm"].max()),
                text=f"Crowding: {sel['population_density_sqkm']:,.0f} per sq km")

    st.header("2. Check the risk")
    if st.button("Check dengue risk", type="primary", width="stretch"):
        st.session_state["checked"] = (city, int(ward_no))
    st.caption("Live weather from Open-Meteo. Results are cached for the day, so repeat checks are instant.")

checked = st.session_state.get("checked")

# ----------------------------------------------------------------------------- landing screen
if not checked:
    st.info("👈 Pick a city and ward in the sidebar, then press **Check dengue risk**.")
    st.image(str(ROOT / "assets" / "breeding_clock.png"), width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="step-card"><h4><span class="step-num">1</span>Choose</h4>'
                "Select a city corporation and a ward. The date is set to today.</div>", unsafe_allow_html=True)
    c2.markdown('<div class="step-card"><h4><span class="step-num">2</span>We read the weather</h4>'
                "The last 5 weeks of rainfall and the last 2 weeks of temperature and humidity are fetched and "
                "combined with the ward's fixed vulnerability.</div>", unsafe_allow_html=True)
    c3.markdown('<div class="step-card"><h4><span class="step-num">3</span>Get a plan</h4>'
                "You see the risk tier for the next 28 days, what drives it, and what to do about it.</div>",
                unsafe_allow_html=True)
    st.stop()

# ----------------------------------------------------------------------------- results
c_city, c_ward = checked
try:
    with st.spinner("Reading the weather and scoring wards..."):
        out, X_all, daily, stale = city_forecast(c_city, today.isoformat())
except Exception as e:
    st.error(f"Could not get weather data: {e}")
    st.stop()

if stale:
    st.warning(f"The weather service could not be reached. Showing the last saved data, which ends {daily.index[-1].date()}.")

pos = int(out.index[out["ward_no"] == c_ward][0])
row = out.loc[pos]
x = X_all.loc[[pos]]
tier = row["tier"]
conf = float(row[f"p_{tier}"])

tab_risk, tab_drivers, tab_map, tab_about = st.tabs(
    ["🎯 Risk and actions", "🌧️ Weather drivers", "🗺️ Ward map", "ℹ️ How it works"]
)

# ---- Tab 1: risk + actions
with tab_risk:
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"""<div class="risk-card" style="background:{COLORS[tier]}; color:{TEXT_ON[tier]}">
<div class="risk-label">Dengue risk for the next 28 days</div>
<div class="risk-tier">{EMOJI[tier]} {tier}</div>
<div class="risk-sub"><b>{c_city}, Ward {c_ward}</b></div>
<div class="risk-sub">{today:%d %b} to {end_day:%d %b %Y} | expected {EXPECTED[tier]} in this ward</div>
<div class="risk-sub">Model confidence: {conf:.0%}</div></div>""",
            unsafe_allow_html=True,
        )
    with right:
        pdf = pd.DataFrame({"tier": LABELS, "probability": [float(row[f"p_{t}"]) for t in LABELS]})
        fig = px.bar(pdf, x="probability", y="tier", orientation="h", color="tier", color_discrete_map=COLORS,
                     text=pdf["probability"].map(lambda v: f"{v:.0%}"), category_orders={"tier": LABELS[::-1]})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, height=210, margin=dict(l=0, r=30, t=30, b=0),
                          title=dict(text="How sure is the model?", font=dict(size=14)),
                          xaxis=dict(range=[0, 1.15], tickformat=".0%", title=None), yaxis_title=None,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    headline, steps = ACTIONS[tier]
    st.subheader(f"Recommended action: {headline}")
    st.caption("Tick items off as your team completes them.")
    for i, s in enumerate(steps):
        st.checkbox(s, key=f"act_{c_city}_{c_ward}_{tier}_{i}")
    if tier in ("High", "Critical"):
        st.markdown('<div class="note-box">⚠️ Act in the next 1-2 weeks: transmission chains are already '
                    "weeks ahead by the time hospital admissions rise.</div>", unsafe_allow_html=True)

# ---- Tab 2: weather drivers
with tab_drivers:
    st.subheader("What the weather has been doing")
    colA, colB = st.columns([3, 2])
    with colA:
        weeks = ["4 weeks ago", "3 weeks ago", "2 weeks ago", "1 week ago", "Last 7 days"]
        rain = [float(x[f"rainfall_mm_lag{k}w"].iloc[0]) for k in (4, 3, 2, 1)] + [float(x["rainfall_mm"].iloc[0])]
        fig = go.Figure(go.Bar(x=weeks, y=rain, text=[f"{v:.0f}" for v in rain], textposition="outside",
                               marker=dict(color=rain, colorscale="Blues", cmin=0, cmax=max(150, max(rain)))))
        fig.add_hline(y=AVG_WEEKLY_RAIN, line_dash="dash", line_color="#C62828",
                      annotation_text=f"Average week ({AVG_WEEKLY_RAIN} mm)", annotation_position="top left")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="Rainfall (mm)",
                          title=dict(text="Weekly rainfall (rain takes 2-4 weeks to turn into cases)", font=dict(size=14)),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with colB:
        t0, t1 = float(x["avg_temp_c"].iloc[0]), float(x["avg_temp_c_lag1w"].iloc[0])
        h0, h1 = float(x["relative_humidity_pct"].iloc[0]), float(x["relative_humidity_pct_lag1w"].iloc[0])
        st.metric("🌡️ Average temperature (last 7 days)", f"{t0:.1f} °C", f"{t0 - t1:+.1f} °C vs the week before")
        st.metric("💧 Average humidity (last 7 days)", f"{h0:.0f} %", f"{h0 - h1:+.0f} % vs the week before")
        st.metric("🌧️ Rain in the last 7 days", f"{rain[-1]:.0f} mm", f"{rain[-1] - AVG_WEEKLY_RAIN:+.0f} mm vs average week")

    st.subheader("What is pushing the risk?")
    dfd, base_score, this_score = drivers_for(c_city, c_ward, today.isoformat())
    st.markdown(f'<div class="driver-box">{driver_sentence(dfd)}</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=dfd["impact"], y=dfd["driver"], orientation="h",
        marker_color=["#C62828" if v >= 0 else "#2F7FB8" for v in dfd["impact"]],
        text=[f"{v:+.2f}" for v in dfd["impact"]], textposition="outside", cliponaxis=False,
        customdata=dfd["detail"], hovertemplate="<b>%{y}</b><br>%{customdata}<br>Effect on risk: %{x:+.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#52606D")
    lim = max(0.3, float(dfd["impact"].abs().max()) * 1.35)
    fig.update_layout(height=430, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[-lim, lim], title="Effect on risk score"),
                      yaxis_title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Red bars raise risk and blue bars lower it, compared with an average week in an average ward. "
        f"Risk score runs from 0 (Low) to 3 (Critical): an average week scores {base_score:.2f}, this ward scores "
        f"{this_score:.2f}. Hover a bar to see the actual value. Based on the model's what-if predictions (Shapley values)."
    )

# ---- Tab 3: ward map
with tab_map:
    st.subheader(f"{c_city}: risk by ward")
    m = out.copy()
    m["Ward"] = "Ward " + m["ward_no"].astype(str)
    m["marker"] = np.where(m["ward_no"] == c_ward, 30, 17)
    m["Water-logging"] = m["water_logging_risk_index"].round(2)
    m["Sanitation"] = m["sanitation_score"].round(2)
    span = max(m["latitude"].max() - m["latitude"].min(), m["longitude"].max() - m["longitude"].min(), 0.02)
    zoom = float(np.clip(9.6 - np.log2(span / 0.1), 8.5, 12.5))
    fig = px.scatter_map(
        m, lat="latitude", lon="longitude", color="tier", size="marker", size_max=30, color_discrete_map=COLORS,
        category_orders={"tier": LABELS}, hover_name="Ward", hover_data={"tier": True, "Water-logging": True,
        "Sanitation": True, "population_density_sqkm": ":,", "latitude": False, "longitude": False, "marker": False},
        zoom=zoom, center=dict(lat=m["latitude"].mean(), lon=m["longitude"].mean()), map_style="open-street-map", height=520)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, x=0, title=None))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"The larger marker is Ward {c_ward}. Colour shows each ward's predicted tier for the next 28 days.")

    st.subheader("Highest-risk wards first")
    tbl = m.sort_values(["score", "water_logging_risk_index"], ascending=False)
    tbl = pd.DataFrame({
        "Ward": tbl["Ward"], "Risk": tbl["tier"].map(lambda t: f"{EMOJI[t]} {t}"), "Risk score (0-3)": tbl["score"].round(2),
        "Water-logging": tbl["Water-logging"], "Sanitation": tbl["Sanitation"], "People per sq km": tbl["population_density_sqkm"]})
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "Risk score (0-3)": st.column_config.ProgressColumn(min_value=0, max_value=3, format="%.2f"),
        "People per sq km": st.column_config.NumberColumn(format="%d")})
    st.caption("All wards in a city share the same weather (fetched once per city); they differ by their fixed vulnerability.")

# ---- Tab 4: about
with tab_about:
    st.image(str(ROOT / "assets" / "breeding_clock.png"), width="stretch")
    st.markdown(
        """
**How the prediction works**

- **Weather lags.** Rain does not cause cases the next day. Mosquito eggs need 7-14 days to mature, then the virus needs time to
  incubate, so cases follow rain by 2-4 weeks. The model reads 5 weekly rainfall totals plus 2 weeks of temperature and humidity.
- **Ward vulnerability.** Water-logging, sanitation and crowding make the same rain more dangerous in some wards than others.
- **Prediction.** A gradient-boosting model, trained on 10 years of weekly ward data and checked on later years it never saw,
  gives one risk tier for the next 28 days: **Low** (0-1 cases), **Medium** (2-19), **High** (20-37) or **Critical** (38+).
- **Efficiency.** One weather request per city per day (cached), model loaded once, no paid or AI-token API at runtime.
"""
    )
    st.markdown('<div class="note-box">This is a decision-support tool for public health officers. It estimates risk from weather and ward '
                "conditions; it does not diagnose illness or replace local surveillance data.</div>", unsafe_allow_html=True)

st.caption("Weather data: Open-Meteo. Built for the EDU AI Engineering Hackathon 2026.")