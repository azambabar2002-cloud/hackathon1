"""Look-and-feel helpers for the dashboard. No logic of the model lives here.

  inject_theme()   - small CSS layer on top of the existing styles (cards, tabs, sidebar, buttons)
  hero()           - landing hero with a 2D mosquito flying across (pure CSS + SVG, works offline)
  city_overview()  - one-screen city picture: key numbers + 2 charts + "where to start"
"""
import plotly.graph_objects as go
import streamlit as st

NAVY, TEAL, PALE = "#0B3C5D", "#12698A", "#D5E6F0"


# ----------------------------------------------------------------------------- step 1: theme
def inject_theme():
    st.markdown(
        f"""
<style>
#MainMenu, footer {{visibility: hidden;}}
[data-testid="stSidebar"] {{background: #F2F8FB; border-right: 1px solid {PALE};}}
[data-testid="stSidebar"] h2 {{font-size: 1.05rem; color: {NAVY};}}
.stTabs [data-baseweb="tab-list"] {{gap: 2px; overflow-x: auto;}}
.stTabs [data-baseweb="tab"] {{padding: 10px 14px; font-weight: 600; color: {NAVY};}}
[data-testid="stMetric"] {{background: #fff; border: 1px solid {PALE}; border-radius: 14px; padding: 12px 16px;
                          box-shadow: 0 2px 8px rgba(11,60,93,0.06);}}
.stButton > button[kind="primary"] {{background: {TEAL}; border: 0; border-radius: 12px; font-weight: 700; padding: .65rem 1rem;}}
.stButton > button[kind="primary"]:hover {{background: {NAVY};}}
.legend {{display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 14px 0;}}
.legend span {{border-radius: 999px; padding: 4px 12px; font-size: .82rem; font-weight: 600;}}
</style>""",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- step 2: hero with a flying mosquito
def _mosquito(cls: str = "") -> str:
    """One mosquito, facing right. Wings flap with CSS; no image file needed."""
    return f"""
<svg class="dw-mq {cls}" viewBox="0 0 120 76" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <g fill="none" stroke="{NAVY}" stroke-width="1.5" stroke-linecap="round">
    <path d="M62 48 L55 63 L49 68"/><path d="M68 49 L66 65 L61 71"/><path d="M75 48 L80 62 L78 70"/>
  </g>
  <g transform="rotate(-10 40 44)">
    <ellipse cx="40" cy="44" rx="27" ry="6.5" fill="{NAVY}"/>
    <path d="M28 39 L29 49 M38 38 L39 50 M48 38 L49 50" stroke="#E3EFF6" stroke-width="1.6"/>
  </g>
  <ellipse cx="70" cy="40" rx="9" ry="8" fill="{TEAL}"/>
  <circle cx="83" cy="39" r="5" fill="{NAVY}"/>
  <path d="M87 40 L107 46" stroke="{NAVY}" stroke-width="1.8" stroke-linecap="round"/>
  <path d="M85 35 L93 28" stroke="{NAVY}" stroke-width="1.3" stroke-linecap="round"/>
  <g class="dw-wing"><ellipse cx="58" cy="24" rx="25" ry="7" transform="rotate(-28 58 24)"
        fill="#fff" fill-opacity=".8" stroke="{TEAL}" stroke-width="1"/></g>
  <g class="dw-wing dw-wing2"><ellipse cx="64" cy="26" rx="19" ry="5.5" transform="rotate(-12 64 26)"
        fill="#fff" fill-opacity=".6" stroke="{TEAL}" stroke-width="1"/></g>
</svg>"""


_HERO_CSS = f"""
<style>
.dw-hero {{position: relative; overflow: hidden; border-radius: 20px; height: 260px; margin-bottom: 18px;
          background: linear-gradient(135deg, #E3EFF6 0%, #FFFFFF 60%); border: 1px solid {PALE};}}
.dw-hero.compact {{height: 120px;}}
.dw-text {{position: relative; z-index: 2; padding: 34px 40px; max-width: 640px;}}
.dw-hero.compact .dw-text {{padding: 22px 32px;}}
.dw-title {{font-size: 2.4rem; font-weight: 800; color: {NAVY}; line-height: 1.1; margin: 0;}}
.dw-hero.compact .dw-title {{font-size: 1.7rem;}}
.dw-sub {{font-size: 1.1rem; color: #52606D; margin: 10px 0 0 0; max-width: 520px;}}
.dw-hint {{display: inline-block; margin-top: 16px; background: {NAVY}; color: #fff; padding: 8px 16px;
          border-radius: 999px; font-size: .92rem; font-weight: 600;}}
.dw-wave {{position: absolute; left: 0; bottom: -1px; width: 100%; height: 46px; z-index: 0;}}
.dw-x {{position: absolute; top: 0; height: 100%; z-index: 1; animation: dw-across 18s linear infinite;}}
.dw-y {{position: absolute; animation: dw-wander 18s ease-in-out infinite;}}
.dw-mq {{display: block; width: 104px; animation: dw-bob .5s ease-in-out infinite alternate;}}
.dw-small .dw-mq {{width: 62px;}}
.dw-flip .dw-mq {{transform: scaleX(-1);}}
.dw-wing {{transform-box: fill-box; transform-origin: 100% 100%; animation: dw-flap .07s ease-in-out infinite alternate;}}
.dw-wing2 {{animation-delay: -.035s;}}
.dw-b {{animation-duration: 24s; animation-direction: reverse; animation-delay: -9s;}}
.dw-b .dw-y {{animation-duration: 24s; animation-delay: -9s;}}
.dw-hero.compact .dw-x {{animation-duration: 14s;}}
@keyframes dw-across {{from {{left: -12%;}} to {{left: 104%;}}}}
@keyframes dw-wander {{0% {{top: 62%;}} 14% {{top: 22%;}} 28% {{top: 52%;}} 42% {{top: 14%;}} 57% {{top: 48%;}}
                      71% {{top: 18%;}} 86% {{top: 46%;}} 100% {{top: 30%;}}}}
@keyframes dw-bob {{from {{translate: 0 -4px;}} to {{translate: 0 5px;}}}}
@keyframes dw-flap {{from {{transform: rotate(-10deg);}} to {{transform: rotate(16deg);}}}}
@media (max-width: 700px) {{.dw-title {{font-size: 1.7rem;}} .dw-text {{padding: 24px 22px;}} .dw-mq {{width: 70px;}}}}
@media (prefers-reduced-motion: reduce) {{.dw-x, .dw-y, .dw-mq, .dw-wing {{animation: none;}} .dw-x {{left: 70%;}} .dw-y {{top: 30%;}}}}
</style>"""

_WAVE = f"""<svg class="dw-wave" viewBox="0 0 1200 46" preserveAspectRatio="none" aria-hidden="true">
<path d="M0 26 C 100 6, 200 6, 300 26 S 500 46, 600 26 S 800 6, 900 26 S 1100 46, 1200 26 V46 H0 Z" fill="{PALE}" fill-opacity=".8"/></svg>"""


def hero(compact: bool = False):
    """Big animated hero on the landing screen; slim version once results are on screen."""
    if compact:
        text = ('<div class="dw-text"><p class="dw-title">Dengue Outbreak Early Warning</p>'
                '<p class="dw-sub">Next 4 weeks: expected cases, supplies and actions.</p></div>')
        flyers = f'<div class="dw-x dw-small"><div class="dw-y">{_mosquito()}</div></div>'
    else:
        text = ('<div class="dw-text"><p class="dw-title">Know where dengue will strike, weeks before it does.</p>'
                '<p class="dw-sub">A 4-week forecast for every ward, with the saline, test kits, beds and spraying crews to send.</p>'
                '<span class="dw-hint">Choose a city and ward on the left, then press "Check dengue risk"</span></div>')
        flyers = (f'<div class="dw-x"><div class="dw-y">{_mosquito()}</div></div>'
                  f'<div class="dw-x dw-b dw-small dw-flip"><div class="dw-y">{_mosquito()}</div></div>')
    st.markdown(f'{_HERO_CSS}<div class="dw-hero{" compact" if compact else ""}">{flyers}{_WAVE}{text}</div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------- step 3: city overview
def city_overview(fc, city, labels, colors, tier_range):
    """fc: the city forecast (one row per ward per week) that dashboard.py already has.
    Shows: 4 key numbers, a colour legend, 'where to start', and two simple bar charts."""
    rank = {t: i for i, t in enumerate(labels)}
    w = fc.assign(r=fc["risk"].map(rank)).groupby("ward_no").agg(cases=("predicted_cases", "sum"), r=("r", "max"))
    w["tier"] = w["r"].map(lambda i: labels[int(i)])
    n_wards = len(w)
    hot = int((w["r"] >= 2).sum())
    by_week = fc.groupby("week")["predicted_cases"].sum()
    beds = int(fc.groupby("week")["hospital_beds"].sum().max())

    st.subheader(f"{city}: the whole city at a glance")
    k = st.columns(4)
    k[0].metric("Wards at High or Critical", f"{hot} of {n_wards}")
    k[1].metric("Expected cases (4 weeks)", f"{by_week.sum():,.0f}")
    k[2].metric("Busiest week", f"Week {int(by_week.idxmax())}")
    k[3].metric("Most beds needed at once", f"{beds:,}")

    text_on = {"Low": "#fff", "Medium": "#1F2933", "High": "#fff", "Critical": "#fff"}
    chips = "".join(f'<span style="background:{colors[t]};color:{text_on[t]}">{t}: {tier_range[t]}</span>' for t in labels)
    st.markdown(f'<div class="legend">{chips}</div>', unsafe_allow_html=True)

    top = w.sort_values("cases", ascending=False).head(10)
    first3 = ", ".join(f"Ward {int(i)}" for i in top.index[:3])
    st.markdown(f'<div class="driver-box">📍 <b>Where to start:</b> {first3} have the most expected cases. '
                "Open the Supply allocation tab to see what each ward will receive.</div>", unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        t10 = top.iloc[::-1]
        fig = go.Figure(go.Bar(x=t10["cases"], y=[f"Ward {int(i)}" for i in t10.index], orientation="h",
                               marker_color=[colors[t] for t in t10["tier"]],
                               text=[f"{v:.0f}" for v in t10["cases"]], textposition="outside", cliponaxis=False))
        fig.update_layout(height=380, margin=dict(l=0, r=20, t=40, b=0), xaxis_title="Expected cases in 4 weeks",
                          title=dict(text="10 wards with the most cases", font=dict(size=14)),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        cnt = fc.groupby(["week", "risk"]).size().unstack(fill_value=0).reindex(columns=labels, fill_value=0)
        x = [f"Week {int(k_)}" for k_ in cnt.index]
        fig = go.Figure([go.Bar(name=t, x=x, y=cnt[t], marker_color=colors[t]) for t in labels])
        fig.update_layout(barmode="stack", height=380, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="Number of wards",
                          title=dict(text="How many wards are in each risk level", font=dict(size=14)),
                          legend=dict(orientation="h", y=-0.15), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})