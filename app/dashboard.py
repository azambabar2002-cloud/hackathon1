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
from src.predict import forecast_weeks
from src.weather import fetch_daily_weather

st.set_page_config(page_title="Dengue Early Warning", page_icon="🦟", layout="wide")

# ----------------------------------------------------------------------------- constants
LABELS = ["Low", "Medium", "High", "Critical"]
RANK = {t: i for i, t in enumerate(LABELS)}
COLORS = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#EF6C00", "Critical": "#C62828"}
TEXT_ON = {"Low": "#FFFFFF", "Medium": "#1F2933", "High": "#FFFFFF", "Critical": "#FFFFFF"}
EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
TIER_RANGE = {"Low": "0-1 cases a week", "Medium": "2-6 cases a week", "High": "7-11 cases a week",
              "Critical": "12 or more cases a week"}
TIER_CUTS = [1.5, 6.5, 11.5]
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
.risk-card {border-radius: 20px; padding: 22px 26px; box-shadow: 0 6px 18px rgba(11,60,93,0.18); margin-bottom: 14px;}
.risk-label {font-size: 0.9rem; opacity: 0.92; letter-spacing: 0.04em; text-transform: uppercase;}
.risk-tier {font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: 4px 0 6px 0;}
.risk-sub {font-size: 1.0rem; opacity: 0.95; margin: 2px 0;}
.wk-card {border-radius: 16px; padding: 14px 16px; box-shadow: 0 4px 12px rgba(11,60,93,0.15); height: 100%;}
.wk-title {font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase; opacity: 0.9;}
.wk-dates {font-size: 0.8rem; opacity: 0.85; margin-bottom: 4px;}
.wk-tier {font-size: 1.5rem; font-weight: 800; line-height: 1.15;}
.wk-cases {font-size: 0.95rem; margin: 2px 0 8px 0;}
.wk-sup {font-size: 0.85rem; line-height: 1.5; opacity: 0.97;}
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

# ----------------------------------------------------------------------------- helpers
def tier_of(cases: float) -> str:
    return LABELS[int(np.select([cases <= TIER_CUTS[0], cases <= TIER_CUTS[1], cases <= TIER_CUTS[2]], [0, 1, 2], 3))]


def fmt_cases(v: float) -> str:
    return "under 1 case" if v < 0.5 else f"about {v:.0f} cases"


def week_dates(k: int, today: date):
    start = today + timedelta(days=7 * (k - 1))
    return start, start + timedelta(days=6)


@st.cache_data
def load_wards():
    w = pd.read_csv(ROOT / "data" / "ward_static.csv")
    w["ward_no"] = w["ward_id"].str.extract(r"(\d+)$")[0].astype(int)
    return w


@st.cache_data(show_spinner=False, ttl=1800)
def city_forecast(city: str, today_iso: str):
    """One weather request per city (city centre); every ward is forecast with its own fixed features."""
    cw = load_wards().query("city == @city").reset_index(drop=True)
    daily, stale = fetch_daily_weather(cw["latitude"].mean(), cw["longitude"].mean(), date.fromisoformat(today_iso))
    X = pd.concat(
        [build_features(daily, {c: r[c] for c in STATIC_COLS}) for _, r in cw.iterrows()], ignore_index=True
    )
    fc = forecast_weeks(X)
    fc = fc.merge(cw[["ward_no", "ward_id", "latitude", "longitude"] + STATIC_COLS], left_on="row", right_index=True)
    return fc, X, daily, stale


@st.cache_data(show_spinner=False, ttl=1800)
def drivers_for(city: str, ward_no: int, week: int, today_iso: str):
    fc, X, _, _ = city_forecast(city, today_iso)
    idx = int(fc.loc[fc["ward_no"] == ward_no, "row"].iloc[0])
    return driver_breakdown(X.loc[[idx]], week)


def driver_sentence(df: pd.DataFrame) -> str:
    up = df[df.impact >= 0.3].sort_values("impact", ascending=False).head(3)
    down = df[df.impact <= -0.3].sort_values("impact").head(2)
    parts = []
    if len(up):
        parts.append("Pushing cases **up**: " + ", ".join(up.driver.str.lower()))
    if len(down):
        parts.append("Holding cases **down**: " + ", ".join(down.driver.str.lower()))
    return " | ".join(parts) or "This ward and week look close to average, so no single factor stands out."


# ----------------------------------------------------------------------------- header
wards = load_wards()
today = datetime.now(ZoneInfo("Asia/Dhaka")).date()
end_day = today + timedelta(days=27)

st.image(str(ROOT / "assets" / "banner.png"), width="stretch")
st.markdown('<p class="hero-title">Dengue Outbreak Early Warning</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">A week-by-week dengue forecast for the next 4 weeks, with the supplies each ward will need.</p>',
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

    st.header("2. Check the forecast")
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
    c3.markdown('<div class="step-card"><h4><span class="step-num">3</span>Get a 4-week plan</h4>'
                "You see the expected cases, risk tier and supplies for each of the next 4 weeks, what drives them, "
                "and what to do.</div>", unsafe_allow_html=True)
    st.stop()

# ----------------------------------------------------------------------------- results
c_city, c_ward = checked
try:
    with st.spinner("Reading the weather and forecasting wards..."):
        fc, X_all, daily, stale = city_forecast(c_city, today.isoformat())
except Exception as e:
    st.error(f"Could not get weather data: {e}")
    st.stop()

if stale:
    st.warning(f"The weather service could not be reached. Showing the last saved data, which ends {daily.index[-1].date()}.")

wf = fc[fc["ward_no"] == c_ward].sort_values("week").reset_index(drop=True)
pk = wf.loc[wf["predicted_cases"].idxmax()]                       # the peak week
peak_tier = max(wf["risk"], key=RANK.get)
peak_week = int(wf.loc[wf["risk"].map(RANK).idxmax(), "week"])
x = X_all.loc[[int(wf["row"].iloc[0])]]

tab_risk, tab_drivers, tab_map, tab_about = st.tabs(
    ["🎯 4-week forecast and actions", "🌧️ Weather drivers", "🗺️ Ward map", "ℹ️ How it works"]
)

# ---- Tab 1: weekly forecast + supplies + actions
with tab_risk:
    st.markdown(
        f"""<div class="risk-card" style="background:{COLORS[peak_tier]}; color:{TEXT_ON[peak_tier]}">
<div class="risk-label">Highest risk in the next 4 weeks</div>
<div class="risk-tier">{EMOJI[peak_tier]} {peak_tier}, in week {peak_week}</div>
<div class="risk-sub"><b>{c_city}, Ward {c_ward}</b> | {today:%d %b} to {end_day:%d %b %Y}</div>
<div class="risk-sub">Busiest week: week {int(pk['week'])}, {fmt_cases(float(pk['predicted_cases']))}</div></div>""",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for col, (_, r) in zip(cols, wf.iterrows()):
        t, k = r["risk"], int(r["week"])
        s, e = week_dates(k, today)
        col.markdown(
            f"""<div class="wk-card" style="background:{COLORS[t]}; color:{TEXT_ON[t]}">
<div class="wk-title">Week {k}</div><div class="wk-dates">{s:%d %b} - {e:%d %b}</div>
<div class="wk-tier">{EMOJI[t]} {t}</div><div class="wk-cases">{fmt_cases(float(r['predicted_cases']))}</div>
<div class="wk-sup">💧 {int(r['saline_bags'])} saline bags<br>🧪 {int(r['diagnostic_kits'])} test kits<br>🛏️ {int(r['hospital_beds'])} hospital beds</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    fig = go.Figure(go.Bar(
        x=[f"Week {int(k)}" for k in wf["week"]], y=wf["predicted_cases"], marker_color=[COLORS[t] for t in wf["risk"]],
        text=[f"{v:.1f}" for v in wf["predicted_cases"]], textposition="outside", cliponaxis=False))
    for cut, name in zip(TIER_CUTS, ["Medium starts", "High starts", "Critical starts"]):
        fig.add_hline(y=cut + 0.5, line_dash="dot", line_color="#52606D", annotation_text=name,
                      annotation_position="top left", annotation_font_size=10)
    fig.update_layout(height=330, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Expected dengue cases in the week",
                      title=dict(text="Expected cases, week by week", font=dict(size=14)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.subheader("Supplies to plan for")
    sup = pd.DataFrame({
        "Week": [f"Week {int(k)} ({week_dates(int(k), today)[0]:%d %b})" for k in wf["week"]],
        "Saline bags": wf["saline_bags"].astype(int), "Test kits": wf["diagnostic_kits"].astype(int),
        "Hospital beds needed": wf["hospital_beds"].astype(int)})
    total = pd.DataFrame({"Week": ["Total (4 weeks) / peak beds"], "Saline bags": [int(wf["saline_bags"].sum())],
                          "Test kits": [int(wf["diagnostic_kits"].sum())],
                          "Hospital beds needed": [int(wf["hospital_beds"].max())]})
    st.dataframe(pd.concat([sup, total], ignore_index=True), hide_index=True, width="stretch")
    st.caption("Saline and test kits are used up, so they add across weeks. Beds are occupied at the same time, "
               "so the last row shows the busiest week, not a sum.")

    headline, steps = ACTIONS[peak_tier]
    st.subheader(f"Recommended action: {headline}")
    st.caption("Based on the highest-risk week. Tick items off as your team completes them.")
    for i, s_ in enumerate(steps):
        st.checkbox(s_, key=f"act_{c_city}_{c_ward}_{peak_tier}_{i}")
    if peak_tier in ("High", "Critical"):
        st.markdown(f'<div class="note-box">⚠️ The highest risk arrives in week {peak_week}. Act now: transmission chains '
                    "start weeks before hospital admissions rise.</div>", unsafe_allow_html=True)

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

    st.subheader("What is pushing the forecast?")
    wk_pick = st.radio("Forecast week to explain", [1, 2, 3, 4], horizontal=True, index=peak_week - 1,
                       format_func=lambda k: f"Week {k}", key="driver_week")
    dfd, base_cases, this_cases = drivers_for(c_city, c_ward, int(wk_pick), today.isoformat())
    st.markdown(f'<div class="driver-box">{driver_sentence(dfd)}</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=dfd["impact"], y=dfd["driver"], orientation="h",
        marker_color=["#C62828" if v >= 0 else "#2F7FB8" for v in dfd["impact"]],
        text=[f"{v:+.1f}" for v in dfd["impact"]], textposition="outside", cliponaxis=False,
        customdata=dfd["detail"], hovertemplate="<b>%{y}</b><br>%{customdata}<br>Effect: %{x:+.1f} cases<extra></extra>"))
    fig.add_vline(x=0, line_color="#52606D")
    lim = max(1.0, float(dfd["impact"].abs().max()) * 1.35)
    fig.update_layout(height=470, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[-lim, lim], title="Extra cases from this factor"),
                      yaxis_title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Red bars add cases and blue bars remove them, compared with an average week in an average ward. "
        f"An average week gives about {base_cases:.1f} cases; this ward in week {wk_pick} is forecast at {this_cases:.1f}. "
        "Hover a bar to see the actual value. Based on the model's what-if predictions (Shapley values)."
    )

# ---- Tab 3: ward map
with tab_map:
    st.subheader(f"{c_city}: forecast by ward")
    view = st.selectbox("Show", ["Highest risk in the next 4 weeks", "Week 1", "Week 2", "Week 3", "Week 4"], key="map_view")
    wide = fc.pivot(index="ward_no", columns="week", values="predicted_cases")
    info = fc.drop_duplicates("ward_no").set_index("ward_no")[["ward_id", "latitude", "longitude"] + STATIC_COLS]
    m = info.join(wide).reset_index()
    if view.startswith("Highest"):
        m["cases"] = m[[1, 2, 3, 4]].max(axis=1)
    else:
        m["cases"] = m[int(view[-1])]
    m["tier"] = m["cases"].map(tier_of)
    m["Ward"] = "Ward " + m["ward_no"].astype(str)
    m["marker"] = np.where(m["ward_no"] == c_ward, 30, 17)
    m["Expected cases"] = m["cases"].round(1)
    span = max(m["latitude"].max() - m["latitude"].min(), m["longitude"].max() - m["longitude"].min(), 0.02)
    zoom = float(np.clip(9.6 - np.log2(span / 0.1), 8.5, 12.5))
    fig = px.scatter_map(
        m, lat="latitude", lon="longitude", color="tier", size="marker", size_max=30, color_discrete_map=COLORS,
        category_orders={"tier": LABELS}, hover_name="Ward", hover_data={"tier": True, "Expected cases": True,
        "population_density_sqkm": ":,", "latitude": False, "longitude": False, "marker": False},
        zoom=zoom, center=dict(lat=m["latitude"].mean(), lon=m["longitude"].mean()), map_style="open-street-map", height=520)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, x=0, title=None))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"The larger marker is Ward {c_ward}. Colour shows each ward's risk tier for the selected view.")

    st.subheader("Highest-risk wards first")
    sums = fc.groupby("ward_no").agg(saline=("saline_bags", "sum"), kits=("diagnostic_kits", "sum"), beds=("hospital_beds", "max"))
    t = m.set_index("ward_no").join(sums)
    t["peak"] = t[[1, 2, 3, 4]].max(axis=1)
    t = t.sort_values(["peak", "population_density_sqkm"], ascending=False)
    tbl = pd.DataFrame({
        "Ward": t["Ward"], "Peak risk": t["peak"].map(lambda v: f"{EMOJI[tier_of(v)]} {tier_of(v)}"),
        "Week 1": t[1].round(1), "Week 2": t[2].round(1), "Week 3": t[3].round(1), "Week 4": t[4].round(1),
        "Saline (4 wks)": t["saline"].astype(int), "Kits (4 wks)": t["kits"].astype(int), "Beds (peak)": t["beds"].astype(int),
        "People per sq km": t["population_density_sqkm"]})
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "People per sq km": st.column_config.NumberColumn(format="%d")})
    st.caption("Week columns show expected cases. All wards in a city share the same weather (fetched once per city); "
               "they differ by their fixed vulnerability.")

# ---- Tab 4: about
with tab_about:
    st.image(str(ROOT / "assets" / "breeding_clock.png"), width="stretch")
    st.markdown(
        """
**How the forecast works**

- **Weather lags.** Rain does not cause cases the next day. Mosquito eggs need 7-14 days to mature, then the virus needs time to
  incubate, so cases follow rain by 2-4 weeks. The model reads 5 weekly rainfall totals plus 2 weeks of temperature and humidity.
- **Ward vulnerability.** Water-logging, sanitation and crowding make the same rain more dangerous in some wards than others.
- **Time of year.** The week of the year tells the model where we are in the dengue season.
- **Four weekly forecasts.** One gradient-boosting model per forecast week, trained on 10 years of weekly ward data and checked
  on later years it never saw. Each gives the expected cases for that week.
- **Risk tiers, per week:** **Low** (0-1 cases), **Medium** (2-6), **High** (7-11), **Critical** (12 or more).
- **Supplies.** Each week's expected cases are converted to saline bags, test kits and beds using the dataset's own
  cases-to-supplies table.
- **Honest limits.** Later weeks are harder to predict because part of the rain that drives them has not fallen yet.
- **Efficiency.** One weather request per city per day (cached), models loaded once, no paid or AI-token API at runtime.
"""
    )
    st.markdown('<div class="note-box">This is a decision-support tool for public health officers. It estimates risk from weather and ward '
                "conditions; it does not diagnose illness or replace local surveillance data.</div>", unsafe_allow_html=True)

st.caption("Weather data: Open-Meteo. Built for the EDU AI Engineering Hackathon 2026.")