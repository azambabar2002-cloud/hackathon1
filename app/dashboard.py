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

from src.allocation import BANDS, CREWS_BY_TIER, ROUND_SHARE, W_CASES, W_DENSITY, build_ward_demand, run_allocation
from src.dispatch import brief_html, brief_pdf, build_notices, city_summary, phone_links, split_sms
from src.explain import driver_breakdown
from src.overflow import BEDS_PER_10K, plan_overflow
from src.simulate import EFFECT_PER_CREW_DAY, MAX_CREWS_PER_WARD, auto_deploy, run as run_whatif, ward_curves
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


@st.cache_data
def load_recorded():
    return pd.read_csv(ROOT / "data" / "recorded_2025.csv", parse_dates=["date"])


def recorded_starts():
    """Forecast start dates available from the recorded data: the Monday after each recorded weather week."""
    return sorted((d + pd.Timedelta(days=7)).date().isoformat() for d in load_recorded()["date"].unique())


@st.cache_data(show_spinner=False, ttl=1800)
def city_forecast(city: str, today_iso: str):
    """One weather request per city (city centre); every ward is forecast with its own fixed features."""
    cw = load_wards().query("city == @city").reset_index(drop=True)
    if today_iso in recorded_starts():               # recorded data: each ward's own saved weather for that week (works offline)
        dm = load_recorded()
        dm = dm[(dm["date"] == pd.Timestamp(today_iso) - pd.Timedelta(days=7)) & (dm["city"] == city)]
        X = cw[["ward_id"]].merge(dm, on="ward_id", how="left").drop(columns=["ward_id", "date", "city"])
        X = pd.concat([X, cw[STATIC_COLS]], axis=1)
        daily, stale = pd.DataFrame(), False
    else:
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
SRC_LIVE, SRC_REC = "📡 Live weather", "🗂️ Recorded data (2025)"
rec_on = st.session_state.get("src_mode") == SRC_REC
if rec_on:                                          # recorded mode: the officer picks the week; no internet needed
    today = date.fromisoformat(st.session_state.get("rec_date", "2025-08-25"))
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
    st.caption(f"📅 {'Forecast from' if rec_on else 'Today'}: **{today:%A, %d %B %Y}**")

    sel = wards[(wards["city"] == city) & (wards["ward_no"] == ward_no)].iloc[0]
    st.markdown("**Ward vulnerability (fixed)**")
    st.progress(float(sel["water_logging_risk_index"]), text=f"Water-logging risk: {sel['water_logging_risk_index']:.2f}")
    st.progress(float(1 - sel["sanitation_score"]), text=f"Poor sanitation: {1 - sel['sanitation_score']:.2f}")
    st.progress(float(sel["population_density_sqkm"] / wards["population_density_sqkm"].max()),
                text=f"Crowding: {sel['population_density_sqkm']:,.0f} per sq km")

    st.header("2. Check the forecast")
    st.radio("Weather source", [SRC_LIVE, SRC_REC], key="src_mode",
             help="Recorded data replays last year's real weather for every ward, with no internet needed.")
    if rec_on:
        opts = recorded_starts()
        st.select_slider("Forecast starting", opts, value="2025-08-25", key="rec_date",
                         format_func=lambda d: f"{date.fromisoformat(d):%d %b %Y}")
        st.caption("Scroll the date to see the forecast for any week of 2025. Works fully offline.")
    if st.button("Check dengue risk", type="primary", width="stretch"):
        st.session_state["checked"] = (city, int(ward_no))
    if not rec_on:
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

tab_risk, tab_drivers, tab_map, tab_alloc, tab_beds, tab_dispatch, tab_sim, tab_about = st.tabs(
    ["🎯 4-week forecast and actions", "🌧️ Weather drivers", "🗺️ Ward map", "📦 Supply allocation",
     "🏥 Bed overflow", "📨 Dispatch notices", "🦟 What-if: vector control", "ℹ️ How it works"]
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

# ---- Tab 4: supply allocation for the whole city, next week only
with tab_alloc:
    d1, d2 = week_dates(1, today)
    dem = build_ward_demand(fc[fc["week"] == 1])
    need_s, need_k, need_c = int(dem["saline_need"].sum()), int(dem["kits_need"].sum()), int(dem["crews_need"].sum())
    st.subheader(f"Next week's supply allocation for all {len(dem)} wards in {c_city}")
    st.caption(f"Forecast week: {d1:%d %b} to {d2:%d %b}. Enter what you have for that week; the tool shares it across wards and tells you what to order.")

    mode = st.radio("Enter stock as", ["Percent of next week's need", "Exact numbers"], horizontal=True, key="alloc_mode")
    k1, k2 = st.columns(2)
    if mode.startswith("Percent"):
        pct = k1.slider("Available regional stock (% of next week's need)", 30, 100, 60, 5, key="stock_pct")
        saline_stock, kits_stock = int(round(need_s * pct / 100)), int(round(need_k * pct / 100))
        k2.markdown(f"**Saline bags in stock:** {saline_stock:,} (need {need_s:,})  \n**Test kits in stock:** {kits_stock:,} (need {need_k:,})")
    else:
        saline_stock = int(k1.number_input("Saline bags in stock", 0, 10_000_000, int(need_s * 0.6), key=f"sal_{c_city}_{need_s}"))
        kits_stock = int(k2.number_input("Test kits in stock", 0, 10_000_000, int(need_k * 0.6), key=f"kit_{c_city}_{need_k}"))
    crews_total = int(st.number_input(f"Total spraying crews available (next week calls for {need_c})", 0, 100_000,
                                      int(round(need_c * 0.6)), key=f"crew_{c_city}_{need_c}"))

    res, sm = run_allocation(dem, saline_stock, kits_stock, crews_total)

    if sm["with_demand"] == 0:
        st.success("The forecast shows almost no demand next week, so there is nothing to allocate or order.")
    else:
        st.markdown("**📝 To order for next week**")
        m = st.columns(3)
        for col, (name, key) in zip(m, [("Saline bags", "saline"), ("Test kits", "kits"), ("Spraying crews", "crews")]):
            d = sm[key]
            col.metric(name, f"{d['short']:,} to order" if d["short"] else "Nothing to order",
                       f"have {d['stock']:,} of {d['need']:,} needed", delta_color="off")

        nb = sm["by_band"]
        crit = res[res["worst_band"] == 3]
        if len(crit):
            names = ", ".join(f"Ward {w}" for w in crit["ward_no"].head(8)) + (" ..." if len(crit) > 8 else "")
            st.error(f"🔴 CRITICAL SUPPLY DEFICIT in {len(crit)} ward(s): {names}. More than half of their need cannot be met with current stock.")
        if nb["Serious shortfall"]:
            st.warning(f"🟠 {nb['Serious shortfall']} ward(s) have a serious shortfall (26-50% unmet).")
        if nb["Minor shortfall"]:
            st.info(f"🟡 {nb['Minor shortfall']} ward(s) have a minor shortfall (up to 25% unmet).")
        if nb["Fully covered"] == sm["wards"]:
            st.success("🟢 Every ward's need is fully covered.")
        left = [f"{sm[k]['surplus']:,} {n}" for k, n in (("saline", "saline bags"), ("kits", "kits"), ("crews", "crews")) if sm[k]["surplus"]]
        if left:
            st.caption("Stock not needed next week: " + ", ".join(left))

        top = res.head(25).sort_values("rank", ascending=False)
        cov = (1 - top["worst_short_pct"]) * 100
        band_color = ["#2e9e5b", "#e2b714", "#e8833a", "#c0392b"]
        fig = go.Figure(go.Bar(
            x=cov, y=[f"#{r} Ward {w}" for r, w in zip(top["rank"], top["ward_no"])], orientation="h",
            marker_color=[band_color[b] for b in top["worst_band"]],
            text=[f"{c:.0f}%" for c in cov], textposition="outside"))
        fig.update_layout(title="Coverage of the worst-supplied item, top 25 priority wards", height=60 + 22 * len(top),
                          xaxis=dict(range=[0, 110], title="% of need met"), margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    show = pd.DataFrame({
        "Priority": res["rank"], "Ward": res["ward_no"], "Risk": res["peak_risk"],
        "Expected cases": res["cases_4wk"].round(1), "People per sq km": res["density"].round(0).astype(int),
        "Saline": res["saline_given"].astype(str) + " / " + res["saline_need"].astype(str),
        "Kits": res["kits_given"].astype(str) + " / " + res["kits_need"].astype(str),
        "Crews": res["crews_given"].astype(str) + " / " + res["crews_need"].astype(str),
        "Beds needed": res["beds_need"], "Status": res["status"]})
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Supply columns read: given / needed. Anything not given is the shortfall to order.")
    st.download_button("⬇️ Download allocation (CSV)", show.to_csv(index=False).encode("utf-8"),
                       file_name=f"allocation_{c_city}_{today}.csv", mime="text/csv")

    with st.expander("How this allocation works (assumptions)"):
        st.markdown(f"""
- **Need** per ward = next week's forecast (saline, kits, beds). Crew need follows the ward's risk tier:
  Low {CREWS_BY_TIER['Low']}, Medium {CREWS_BY_TIER['Medium']}, High {CREWS_BY_TIER['High']}, Critical {CREWS_BY_TIER['Critical']} crews (an assumption: the dataset has no crew data).
- **Priority** = {W_CASES:.0%} expected cases (relative to the city's busiest ward) + {W_DENSITY:.0%} population density (relative to the densest ward).
- **Sharing:** stock goes out in rounds of up to {ROUND_SHARE:.0%} of each ward's need. If stock cannot cover everyone's first {ROUND_SHARE:.0%},
  it is split across all wards by priority and need, so no ward gets zero. The second round goes strictly in priority order.
- **Crews** are whole units: every ward that needs one gets a first crew (by priority) before any ward gets a second.
- **Flags** use the share of need left unmet, taking each ward's worst item: 🟢 0%, 🟡 up to 25%, 🟠 26-50%, 🔴 over 50%.
- **To order** = total need minus stock, for saline, kits and crews.
- Hospital beds are not stock-limited here; bed capacity and overflow between hospitals are handled separately.""")

# ---- Tab 5: hospital surge and overflow routing (next week, same city only)
with tab_beds:
    b1, b2 = week_dates(1, today)
    geo = fc[fc["week"] == 1][["ward_no", "latitude", "longitude"]]
    bw = dem.merge(geo, on="ward_no")                       # `dem` = next week's ward demand from the allocation tab
    st.subheader(f"Hospital beds and overflow for {c_city}")
    st.caption(f"Forecast week: {b1:%d %b} to {b2:%d %b}. Wards that need more beds than they have are listed with where to send the overflow.")

    free = st.slider("Share of dengue beds currently free", 30, 100, 50, 5, key="bed_free",
                     help="Hospitals are never empty. This is the part of each ward's dengue-designated beds that is available.")
    with st.expander("Bed capacity assumption (can be changed)"):
        per10k = st.number_input("Dengue-designated beds per 10,000 people per sq km of density", 0.1, 10.0, float(BEDS_PER_10K), 0.1,
                                 key="bed_per10k")
        st.caption("The dataset has no hospital capacity, so each ward's beds = density / 10,000 x this number x the free share (at least 1).")

    bt, rt, bs = plan_overflow(bw, free / 100, per10k)

    if bs["need"] == 0:
        st.success("The forecast shows no bed demand next week, so there is no overflow to route.")
    else:
        m = st.columns(4)
        m[0].metric("Beds needed", f"{bs['need']:,}")
        m[1].metric("Beds available", f"{bs['capacity']:,}")
        m[2].metric("City bed utilization", f"{bs['utilization']:.0%}")
        m[3].metric("Wards in surge", bs["surge_wards"])
        st.progress(min(bs["utilization"], 1.0))

        if bs["surge_wards"] == 0:
            st.success("🟢 Every ward can handle its own patients next week. No rerouting needed.")
        else:
            if bs["unplaced"]:
                st.error(f"🔴 CITY-WIDE BED SHORTAGE: {bs['unplaced']} patient(s) cannot be placed anywhere in {c_city}. "
                         "Ask for beds from outside the city or open extra capacity.")
            if bs["rerouted"]:
                st.warning(f"🟠 {bs['surge_wards']} ward(s) are over capacity. {bs['rerouted']} bed(s) of overflow can be rerouted within the city.")
            if len(rt):
                st.markdown("**📞 Call these wards**")
                for _, a in rt.iterrows():
                    st.markdown(f"- **Ward {a.from_ward}** is over capacity: send **{a.beds} patient(s)** to **Ward {a.to_ward}** ({a.km} km away)")
                st.dataframe(rt.rename(columns={"from_ward": "From ward", "to_ward": "To ward", "beds": "Beds", "km": "Distance (km)"}),
                             hide_index=True, width="stretch")
                st.download_button("⬇️ Download routes (CSV)", rt.to_csv(index=False).encode("utf-8"),
                                   file_name=f"overflow_routes_{c_city}_{today}.csv", mime="text/csv")

        srt = bt.sort_values("net")
        topn = srt.head(25).sort_values("net", ascending=False)
        fig = go.Figure(go.Bar(
            x=topn["net"], y=[f"Ward {w}" for w in topn["ward_no"]], orientation="h",
            marker_color=["#c0392b" if v < 0 else "#2e9e5b" for v in topn["net"]]))
        fig.update_layout(title="Net beds per ward (spare + / deficit -), 25 tightest wards", height=60 + 22 * len(topn),
                          xaxis_title="Beds", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

        st.dataframe(pd.DataFrame({
            "Ward": srt["ward_no"], "Beds needed": srt["need"], "Beds available": srt["capacity"],
            "Net (+spare / -deficit)": srt["net"], "Utilization": (srt["utilization"] * 100).round(0).astype(int).astype(str) + "%",
            "Sends out": srt["rerouted"], "Receives": srt["received"], "Status": srt["status"]}),
            hide_index=True, width="stretch")

    with st.expander("How the routing works (assumptions)"):
        st.markdown("""
- **Beds needed** = next week's forecast for each ward. **Beds available** is an assumption (the dataset has no capacity): density-based, scaled by the free share above.
- **Surge** = a ward needs more beds than it has. The gap is its overflow.
- **Routing:** the biggest overflow is placed first, each time into the nearest ward (by map coordinates) that still has spare beds. A ward never takes more than its spare beds.
- **Same city only.** If the whole city is full, the tool reports a city-wide shortage instead of hiding it.
- The tool suggests; the officer decides and calls the hospitals. No patient data is used.""")

# ---- Tab 6: dispatch notices (SMS / WhatsApp text, HTML and PDF commissioner brief). Fully offline.
# Reuses what the earlier tabs built: res + sm (allocation), rt (overflow routes), d1 + d2 (forecast week dates).
with tab_dispatch:
    st.subheader(f"Dispatch notices for {c_city}")
    st.caption(f"Forecast week {d1:%d %b} to {d2:%d %b}. Built from the Supply allocation and Bed overflow tabs with fixed "
               "templates. No internet and no AI service are used, so this works offline.")

    notices = build_notices(res, c_city, d1, d2)
    if notices.empty:
        st.success("No ward has demand next week, so there is nothing to dispatch.")
    else:
        only = st.checkbox("Only wards with a shortfall or High/Critical risk", value=True, key="disp_only")
        view = notices[(notices["band"] > 0) | notices["risk"].isin(["High", "Critical"])] if only else notices
        if view.empty:
            view = notices
        wards_list = [int(w) for w in view["ward_no"]]
        pick = st.selectbox("Ward", wards_list, index=wards_list.index(c_ward) if c_ward in wards_list else 0,
                            format_func=lambda w: f"Ward {w}", key="disp_ward")
        n = notices[notices["ward_no"] == pick].iloc[0]

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**📱 SMS (plain text)**")
            parts = split_sms(n["sms"])
            for p in parts:
                st.code(p, language=None)                       # st.code has a built-in copy button
            st.caption(f"{len(parts)} message(s), {sum(len(p) for p in parts)} characters")
        with colB:
            st.markdown("**💬 WhatsApp**")
            st.code(n["whatsapp"], language=None)

        lk_sms, lk_wa = phone_links(n["sms"])["sms"], phone_links(n["whatsapp"])["whatsapp"]
        st.markdown(f"[📱 Open in the phone's SMS app]({lk_sms}) &nbsp;|&nbsp; [💬 Open in WhatsApp]({lk_wa})")
        st.caption("These links only open the officer's own phone app with the text filled in. Sending needs that phone to have "
                   "signal. This tool prepares the message; it does not send it.")

        st.markdown("**All wards**")
        st.dataframe(notices[["rank", "ward_no", "risk", "cases", "saline", "kits", "crews", "beds", "status"]]
                     .rename(columns={"rank": "Priority", "ward_no": "Ward", "risk": "Risk", "cases": "Cases",
                                      "saline": "Saline (given/need)", "kits": "Kits (given/need)",
                                      "crews": "Crews (assigned/need)", "beds": "Beds", "status": "Status"})
                     .assign(Cases=lambda d: d["Cases"].round(1)),
                     hide_index=True, width="stretch")

        st.divider()
        all_txt = "\n\n".join(f"===== Ward {r.ward_no} =====\nSMS:\n{r.sms}\n\nWHATSAPP:\n{r.whatsapp}"
                              for r in notices.itertuples())
        brief = brief_html(notices, city_summary(res, sm), rt, c_city, d1, d2)
        x1, x2, x3 = st.columns(3)
        x1.download_button("⬇️ All notices (TXT)", all_txt.encode("utf-8"),
                           file_name=f"dispatch_notices_{c_city}_{d1}.txt", mime="text/plain")
        x2.download_button("⬇️ Commissioner brief (HTML)", brief.encode("utf-8"),
                           file_name=f"commissioner_brief_{c_city}_{d1}.html", mime="text/html")
        try:
            pdf_bytes = brief_pdf(notices, city_summary(res, sm), rt, c_city, d1, d2)
            x3.download_button("⬇️ Commissioner brief (PDF)", pdf_bytes,
                               file_name=f"commissioner_brief_{c_city}_{d1}.pdf", mime="application/pdf")
        except ImportError:
            x3.info("PDF needs reportlab: run `pip install reportlab`. The HTML brief can be printed to PDF from any browser.")

        with st.expander("How the notices are made"):
            st.markdown("""
- Every number comes from the forecast, the allocation and the bed-overflow tabs. Nothing is typed by hand.
- The text is filled into a fixed template, so the same input always gives the same notice (deterministic, no AI model needed).
- **SMS** uses plain characters and is split into 160-character parts. **WhatsApp** uses a longer message with bold text.
- **Deficit alert** appears when a ward's need is not fully met. The band follows the allocation tab (minor, serious, critical).
- The **commissioner brief** lists city totals, deficit alerts, the ward table and bed overflow routes, and prints from any browser.""")

# ---- Tab 7: what-if vector control simulator
with tab_sim:
    st.subheader(f"What-if: send spraying crews in {c_city}")
    st.caption("Choose how many crews work, for how many days, and where. The forecast model is re-run for each ward with fewer "
               "breeding sites. Crews work during next week, so week 1 stays as forecast and you see the effect on weeks 2 to 4: cases, beds and saline with and without the crews.")

    s1, s2, s3 = st.columns(3)
    sim_total = int(s1.number_input("Crews available", 0, 1000, max(int(crews_total), 1), key=f"sim_crews_{c_city}"))
    sim_days = int(s2.slider("Days each crew works", 1, 14, 5, key="sim_days",
                             help="Larvicide and fogging rounds repeated over this many days."))
    sim_eff = s3.slider("Effect of one crew for one day (%)", 2, 10, int(EFFECT_PER_CREW_DAY * 100), key="sim_eff",
                        help="Assumption: the dataset has no spraying data, so the officer can test other values.") / 100

    base_rows = X_all.index
    cases_by_k = ward_curves(X_all, sim_days, sim_eff)                  # 4-week cases per ward for 0..5 crews
    gain1 = cases_by_k[0] - cases_by_k[1]                               # cases prevented by ONE crew in each ward
    ward_of_row = fc[fc["week"] == 1].set_index("row")["ward_no"].reindex(base_rows).to_numpy()
    dens_of_row = fc[fc["week"] == 1].set_index("row")["population_density_sqkm"].reindex(base_rows).to_numpy()

    mode_s = st.radio("Deployment", ["Auto: best wards first", "Manual: I choose the wards"], horizontal=True, key="sim_mode")
    if mode_s.startswith("Auto"):
        crews_vec = auto_deploy(cases_by_k, sim_total)
    else:
        order = np.argsort(-gain1)
        default_pick = [int(ward_of_row[i]) for i in order[:3] if gain1[i] > 0]
        pick = st.multiselect("Wards to send crews to", sorted(int(w) for w in ward_of_row), default=default_pick, key=f"sim_pick_{c_city}")
        each = int(st.number_input("Crews per selected ward", 1, MAX_CREWS_PER_WARD, 1, key="sim_each"))
        crews_vec = np.array([each if int(w) in pick else 0 for w in ward_of_row])
        if crews_vec.sum() > sim_total:
            st.error(f"You selected {int(crews_vec.sum())} crews but only {sim_total} are available. Remove wards or lower the crews per ward.")

    base = run_whatif(X_all, np.zeros(len(X_all), int), sim_days, sim_eff)
    what = run_whatif(X_all, crews_vec, sim_days, sim_eff)

    def city_week(f, col):
        return f.groupby("week")[col].sum()

    cb, cw_ = city_week(base, "predicted_cases"), city_week(what, "predicted_cases")
    prevented = float(cb.sum() - cw_.sum())
    later = cb.loc[2:].sum()                                         # weeks 2-4: the weeks the crews can still change
    b2, w2 = base[base["week"] > 1], what[what["week"] > 1]        # weeks 2-4: the weeks the crews can still change
    bed_b, bed_w = int(city_week(b2, "hospital_beds").max()), int(city_week(w2, "hospital_beds").max())
    sal_b, sal_w = int(b2["saline_bags"].sum()), int(w2["saline_bags"].sum())

    if crews_vec.sum() == 0 or later < 0.5:
        st.info("Either no crews are deployed or the forecast shows almost no cases, so there is nothing to prevent. "
                "Try a week in the dengue season.")
    else:
        m = st.columns(4)
        m[0].metric("Crews deployed", f"{int(crews_vec.sum())} of {sim_total}")
        m[1].metric("Cases prevented (weeks 2-4)", f"{prevented:,.1f}", f"-{prevented / later:.0%} of {later:,.0f}", delta_color="inverse")
        m[2].metric("Peak beds freed (weeks 2-4)", f"{bed_b - bed_w:,}", f"{bed_b} to {bed_w}", delta_color="off")
        m[3].metric("Saline bags saved (weeks 2-4)", f"{sal_b - sal_w:,}", f"{sal_b:,} to {sal_w:,}", delta_color="off")
        st.metric("Cases prevented per crew (ROI)", f"{prevented / crews_vec.sum():.2f}")

        wk = [f"Week {k} ({week_dates(k, today)[0]:%d %b})" for k in cb.index]
        fig = go.Figure([go.Bar(name="Without crews", x=wk, y=cb.round(1), marker_color="#c0392b"),
                         go.Bar(name="With crews", x=wk, y=cw_.round(1), marker_color="#2e9e5b")])
        fig.update_layout(barmode="group", title="Expected cases in the whole city, per week", yaxis_title="Cases",
                          height=330, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Crews work during week 1, so week 1 is unchanged. Weeks 2 to 4 are separate runs of the forecast model, so you can see how the effect builds.")

    # ROI ranking: what ONE crew would prevent in each ward, and what the chosen deployment prevents
    bw = base.groupby("row").agg(b_cases=("predicted_cases", "sum"))
    bw = bw.join(b2.groupby("row").agg(b_beds=("hospital_beds", "max"), b_sal=("saline_bags", "sum")))
    ww = what.groupby("row").agg(w_cases=("predicted_cases", "sum"))
    ww = ww.join(w2.groupby("row").agg(w_beds=("hospital_beds", "max"), w_sal=("saline_bags", "sum")))
    t = bw.join(ww).reindex(base_rows)
    t["ward"], t["crews"], t["density"], t["one_crew"] = ward_of_row, crews_vec, dens_of_row, gain1
    t["prevented"] = t["b_cases"] - t["w_cases"]
    t = t.sort_values(["one_crew", "density"], ascending=False).reset_index(drop=True)
    out = pd.DataFrame({
        "ROI rank": np.arange(1, len(t) + 1), "Ward": t["ward"], "Cases without crews (4 wks)": t["b_cases"].round(1),
        "Prevented by 1 crew": t["one_crew"].round(2),
        "Crews sent": t["crews"], "Cases prevented (as deployed)": t["prevented"].round(2),
        "Per crew": np.where(t["crews"] > 0, t["prevented"] / t["crews"].where(t["crews"] > 0, 1), 0.0).round(2),
        "Beds freed (wks 2-4)": (t["b_beds"] - t["w_beds"]).astype(int), "Saline saved (wks 2-4)": (t["b_sal"] - t["w_sal"]).astype(int)})
    st.markdown("**Wards ranked by ROI (cases prevented per crew)**")
    st.dataframe(out, hide_index=True, width="stretch")
    st.download_button("⬇️ Download simulation (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"vector_control_{c_city}_{today}.csv", mime="text/csv")

    with st.expander("How the simulation works (assumptions)"):
        st.markdown(f"""
- **No new model.** The same 4 weekly forecast models are run again with a change to the ward's inputs. Everything the model does with that
  change comes from 10 years of ward data.
- **What a crew changes (assumption).** The dataset has no spraying records. We assume one crew working one day cuts the ward's effective
  water-logging risk (the stagnant-water breeding sites) by {sim_eff:.0%}, with diminishing returns: reduction = 1 - (1 - {sim_eff:.2f})^(crews x days).
  So {sim_days} days with 1 crew = **{(1 - (1 - sim_eff) ** sim_days):.0%}**, with 2 crews = **{(1 - (1 - sim_eff) ** (2 * sim_days)):.0%}**. Change the slider to test other values.
- **Timing.** Crews work during next week (week 1), and we cannot change what happens before they arrive. So week 1 stays as forecast and the effect is applied to weeks 2, 3 and 4. The model decides how large it is in each of those weeks.
- **Beds and saline** come from the same cases-to-supplies table as the rest of the app, so they move in whole steps.
- **ROI** = cases prevented divided by crews sent. "Cases prevented by 1 crew" ranks every ward on equal terms; Auto mode adds crews one at a
  time where the next crew prevents the most cases.
- Results are model estimates for planning, not guarantees.""")

# ---- Tab 8: about
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
- **Dispatch notices.** Ward-level SMS and WhatsApp text and a printable commissioner brief are filled from fixed templates, offline.
- **Honest limits.** Later weeks are harder to predict because part of the rain that drives them has not fallen yet.
- **Efficiency.** One weather request per city per day (cached), models loaded once, no paid or AI-token API at runtime.
"""
    )
    st.markdown('<div class="note-box">This is a decision-support tool for public health officers. It estimates risk from weather and ward '
                "conditions; it does not diagnose illness or replace local surveillance data.</div>", unsafe_allow_html=True)

st.caption("Weather data: Open-Meteo. Built for the EDU AI Engineering Hackathon 2026.")