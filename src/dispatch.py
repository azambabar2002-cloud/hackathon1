"""Dispatch notices for one city and one forecast week. Fully offline and deterministic.

No internet, no LLM, no external API: every message is filled from a fixed template using the
allocation result (`run_allocation`) and the overflow routes (`plan_overflow`).

Outputs
  * SMS text         - plain ASCII, split into 160-character parts (GSM-7 safe)
  * WhatsApp text    - longer text with *bold* and emoji, ready to paste or open in the officer's phone
  * HTML brief       - one self-contained file (inline CSS), prints cleanly from any browser
  * PDF brief        - made with reportlab (pure Python, works offline)

Sending is NOT done here. The app prepares send-ready text; delivery needs a phone or a GSM gateway.
"""
from __future__ import annotations

import html
import io
import re
from datetime import datetime
from urllib.parse import quote

import pandas as pd

BAND_LABELS = ["Fully covered", "Minor shortfall", "Serious shortfall", "Critical deficit"]
BAND_COLORS = ["#2e9e5b", "#e2b714", "#e8833a", "#c0392b"]
RISK_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
ITEMS = [("saline", "Saline bags"), ("kits", "Test kits"), ("crews", "Spraying crews")]
SMS_PART = 153                      # 160 minus the "(1/2) " prefix when a message needs several parts


# ----------------------------------------------------------------------------- small helpers
def _cases(v: float) -> str:
    return "<1" if v < 0.5 else f"~{v:.0f}"


def _ascii(s: str) -> str:
    """SMS-safe text: drop emoji and non-ASCII characters."""
    return re.sub(r"\s+", " ", s.encode("ascii", "ignore").decode()).strip()


def _short_list(r, sep=", ") -> str:
    out = []
    for key, name in ITEMS:
        short, need = int(r[f"{key}_short"]), int(r[f"{key}_need"])
        if short > 0 and need > 0:
            out.append(f"{name.lower()} {short}/{need} short ({short / need:.0%})")
    return sep.join(out)


def _short_compact(r) -> str:
    """Short form for SMS: 'saline 67%, kits 67%'."""
    return ", ".join(f"{k} {int(r[f'{k}_short']) / int(r[f'{k}_need']):.0%}" for k, _ in ITEMS
                     if int(r[f"{k}_short"]) > 0 and int(r[f"{k}_need"]) > 0)


# ----------------------------------------------------------------------------- per-ward notice
def build_notices(res: pd.DataFrame, city: str, d1, d2) -> pd.DataFrame:
    """res: output of allocation.run_allocation (one row per ward). Returns one row per ward with ready text.
    Wards with no demand at all are left out."""
    rows = []
    for _, r in res.iterrows():
        if int(r["saline_need"]) + int(r["kits_need"]) + int(r["crews_need"]) + int(r["beds_need"]) == 0:
            continue
        band = int(r["worst_band"])
        risk = str(r["peak_risk"])
        period = f"{d1:%d %b}-{d2:%d %b}"
        shorts = _short_list(r)

        sms = (f"DENGUE {city} W{int(r['ward_no'])} {d1:%d%b}-{d2:%d%b}: {risk.upper()}, "
               f"{_cases(float(r['cases_4wk']))} cases. "
               f"Saline {int(r['saline_given'])}/{int(r['saline_need'])}, Kits {int(r['kits_given'])}/{int(r['kits_need'])}, "
               f"Crews {int(r['crews_given'])}/{int(r['crews_need'])}, Beds {int(r['beds_need'])}.")
        sms += f" SHORT: {_short_compact(r)}." if shorts else " Supplies OK."

        wa = [f"{RISK_ICON[risk]} *DENGUE DISPATCH NOTICE*",
              f"*{city}, Ward {int(r['ward_no'])}*  |  {period}",
              f"Risk: *{risk}*  |  Expected cases: *{_cases(float(r['cases_4wk']))}*  |  Beds needed: *{int(r['beds_need'])}*",
              "",
              "*Supplies (given / required)*",
              f"- Saline bags: {int(r['saline_given'])} / {int(r['saline_need'])}",
              f"- Test kits: {int(r['kits_given'])} / {int(r['kits_need'])}",
              f"- Spraying crews assigned: {int(r['crews_given'])} / {int(r['crews_need'])}",
              ""]
        wa.append(f"⚠️ *DEFICIT ALERT ({BAND_LABELS[band]}):* {shorts}" if shorts else "✅ All supplies covered for this week.")
        wa += ["", f"Priority rank in city: #{int(r['rank'])}", "Prepared offline by the Dengue Early Warning tool."]

        rows.append({
            "rank": int(r["rank"]), "ward_no": int(r["ward_no"]), "risk": risk, "cases": float(r["cases_4wk"]),
            "saline": f"{int(r['saline_given'])}/{int(r['saline_need'])}", "kits": f"{int(r['kits_given'])}/{int(r['kits_need'])}",
            "crews": f"{int(r['crews_given'])}/{int(r['crews_need'])}", "beds": int(r["beds_need"]),
            "band": band, "status": BAND_LABELS[band], "shortfall": shorts,
            "sms": _ascii(sms), "whatsapp": "\n".join(wa)})
    return pd.DataFrame(rows)


def split_sms(text: str, limit: int = 160, part: int = SMS_PART) -> list[str]:
    """One SMS if it fits in 160 characters, otherwise numbered parts split at word boundaries."""
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ""
    for w in text.split(" "):
        if cur and len(cur) + 1 + len(w) > part:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        chunks.append(cur)
    return [f"({i}/{len(chunks)}) {c}" for i, c in enumerate(chunks, 1)]


def phone_links(text: str) -> dict:
    """Links that open the officer's OWN phone app with the text filled in. They only send when that phone has signal."""
    return {"sms": f"sms:?&body={quote(text)}", "whatsapp": f"https://wa.me/?text={quote(text)}"}


# ----------------------------------------------------------------------------- city summary
def city_summary(res: pd.DataFrame, summary: dict) -> dict:
    """Totals for the brief. `summary` is the second value returned by allocation.run_allocation."""
    return {"wards": summary["wards"], "with_demand": summary["with_demand"],
            "items": {k: summary[k] for k, _ in ITEMS}, "by_band": summary["by_band"]}


# ----------------------------------------------------------------------------- HTML brief
def brief_html(notices: pd.DataFrame, summ: dict, routes: pd.DataFrame, city: str, d1, d2) -> str:
    e = html.escape
    stamp = datetime.now().strftime("%d %b %Y, %H:%M")
    rows = "".join(
        f"<tr><td>{r.rank}</td><td>Ward {r.ward_no}</td><td>{RISK_ICON[r.risk]} {e(r.risk)}</td><td>{_cases(r.cases)}</td>"
        f"<td>{r.saline}</td><td>{r.kits}</td><td>{r.crews}</td><td>{r.beds}</td>"
        f"<td style='background:{BAND_COLORS[r.band]};color:#fff;font-weight:600'>{e(r.status)}</td></tr>"
        for r in notices.itertuples())
    item_rows = "".join(
        f"<tr><td>{name}</td><td>{summ['items'][k]['need']:,}</td><td>{summ['items'][k]['stock']:,}</td>"
        f"<td>{summ['items'][k]['given']:,}</td><td><b>{summ['items'][k]['short']:,}</b></td></tr>" for k, name in ITEMS)

    alerts = notices[notices.band >= 2]
    alert_html = "".join(
        f"<li><b>Ward {r.ward_no}</b> ({e(r.status)}): {e(r.shortfall)}</li>" for r in alerts.itertuples()
    ) or "<li>No ward has a serious or critical deficit.</li>"

    if routes is not None and len(routes):
        route_html = ("<table><tr><th>From ward</th><th>To ward</th><th>Beds</th><th>Distance (km)</th></tr>" +
                      "".join(f"<tr><td>{a.from_ward}</td><td>{a.to_ward}</td><td>{a.beds}</td><td>{a.km}</td></tr>"
                              for a in routes.itertuples()) + "</table>")
    else:
        route_html = "<p>No bed overflow needs rerouting.</p>"

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Commissioner brief - {e(city)}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;color:#1f2933;margin:24px;max-width:1100px}}
h1{{color:#0b3c5d;margin:0}} h2{{color:#0b3c5d;border-bottom:2px solid #d5e6f0;padding-bottom:4px;margin-top:26px}}
.sub{{color:#52606d;margin:4px 0 16px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #d5e6f0;padding:6px 8px;text-align:left}}
th{{background:#12698a;color:#fff}} tr:nth-child(even) td{{background:#f5f9fc}}
.alert{{background:#fff3f0;border-left:5px solid #c0392b;padding:8px 14px}}
.foot{{color:#7b8794;font-size:12px;margin-top:24px}}
@media print{{body{{margin:8mm}} h2{{page-break-after:avoid}} tr{{page-break-inside:avoid}}}}
</style></head><body>
<h1>Dengue Dispatch Brief: {e(city)}</h1>
<p class="sub">Forecast week {d1:%d %b %Y} to {d2:%d %b %Y} | Prepared {stamp} | {summ['with_demand']} of {summ['wards']} wards have demand</p>
<h2>City totals</h2>
<table><tr><th>Item</th><th>Required</th><th>In stock</th><th>Allocated</th><th>To order</th></tr>{item_rows}</table>
<h2>Deficit alerts</h2><div class="alert"><ul>{alert_html}</ul></div>
<h2>Ward dispatch table (priority order)</h2>
<table><tr><th>#</th><th>Ward</th><th>Risk</th><th>Cases</th><th>Saline (given/need)</th><th>Kits (given/need)</th>
<th>Crews assigned/need</th><th>Beds needed</th><th>Status</th></tr>{rows}</table>
<h2>Hospital bed overflow routes</h2>{route_html}
<p class="foot">Generated offline from the forecast model and fixed rules. Case counts, bed capacity and crew needs are planning
estimates, not patient data. The officer confirms and calls the facilities.</p>
</body></html>"""


# ----------------------------------------------------------------------------- PDF brief
def brief_pdf(notices: pd.DataFrame, summ: dict, routes: pd.DataFrame, city: str, d1, d2) -> bytes:
    """Same content as the HTML brief. Needs reportlab (pip install reportlab); raises ImportError if it is missing."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title=f"Dengue brief {city}")
    head = colors.HexColor("#12698A")

    def table(data, widths=None, band_col=None, bands=None):
        t = Table(data, colWidths=widths, repeatRows=1)
        st = [("BACKGROUND", (0, 0), (-1, 0), head), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
              ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5E6F0")),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F9FC")])]
        if band_col is not None:
            for i, b in enumerate(bands, start=1):
                st += [("BACKGROUND", (band_col, i), (band_col, i), colors.HexColor(BAND_COLORS[b])),
                       ("TEXTCOLOR", (band_col, i), (band_col, i), colors.white)]
        t.setStyle(TableStyle(st))
        return t

    el = [Paragraph(f"Dengue Dispatch Brief: {html.escape(city)}", ss["Title"]),
          Paragraph(f"Forecast week {d1:%d %b %Y} to {d2:%d %b %Y} | Prepared {datetime.now():%d %b %Y, %H:%M} | "
                    f"{summ['with_demand']} of {summ['wards']} wards have demand", ss["Normal"]),
          Spacer(1, 6 * mm), Paragraph("City totals", ss["Heading2"])]
    el.append(table([["Item", "Required", "In stock", "Allocated", "To order"]] +
                    [[n, f"{summ['items'][k]['need']:,}", f"{summ['items'][k]['stock']:,}",
                      f"{summ['items'][k]['given']:,}", f"{summ['items'][k]['short']:,}"] for k, n in ITEMS]))

    el += [Spacer(1, 4 * mm), Paragraph("Deficit alerts", ss["Heading2"])]
    alerts = notices[notices.band >= 2]
    if len(alerts):
        for r in alerts.itertuples():
            el.append(Paragraph(f"<b>Ward {r.ward_no}</b> ({html.escape(r.status)}): {html.escape(r.shortfall)}", ss["Normal"]))
    else:
        el.append(Paragraph("No ward has a serious or critical deficit.", ss["Normal"]))

    el += [Spacer(1, 4 * mm), Paragraph("Ward dispatch table (priority order)", ss["Heading2"])]
    data = [["#", "Ward", "Risk", "Cases", "Saline", "Kits", "Crews", "Beds", "Status"]]
    data += [[r.rank, f"Ward {r.ward_no}", r.risk, _cases(r.cases), r.saline, r.kits, r.crews, r.beds, r.status]
             for r in notices.itertuples()]
    el.append(table(data, band_col=8, bands=list(notices["band"])))

    el += [Spacer(1, 4 * mm), Paragraph("Hospital bed overflow routes", ss["Heading2"])]
    if routes is not None and len(routes):
        el.append(table([["From ward", "To ward", "Beds", "Distance (km)"]] +
                        [[a.from_ward, a.to_ward, a.beds, a.km] for a in routes.itertuples()]))
    else:
        el.append(Paragraph("No bed overflow needs rerouting.", ss["Normal"]))
    el += [Spacer(1, 6 * mm), Paragraph("Generated offline from the forecast model and fixed rules. Figures are planning "
                                        "estimates, not patient data. The officer confirms and calls the facilities.", ss["Italic"])]
    doc.build(el)
    return buf.getvalue()