"""Monthly owner report (upgrade 6.4): one command, real data, PDF + JSON.

  python scripts/owner_report.py --site site_demo --month 2026-10

Contents
  * site health index (criticality-weighted, 5.5) and its change vs the previous month,
    per-asset health table sorted worst first
  * inspection activity: missions, assets inspected, coverage of the site
  * defect trends: opened / resolved this month, open at month end, growth forecasts
  * cost of deferring repairs: for every open defect, repair cost now vs after deferring by
    `owner_report.deferral_months`, from per-family cost tables (repair cost grows with the
    forecast size; a defect forecast to become critical within the deferral window adds the
    failure cost)
  * solar energy loss: Σ over panels of rated kWh/month × loss fraction, where the loss
    fraction comes from soiling coverage, crack area and per-type loss factors
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from ml.config import get_nested
from ml.forecast import days_until, family
from ml.health_index import health_band, site_health
from ml.revisit_priority import parse_ts


def month_bounds(month: str) -> Tuple[datetime, datetime]:
    start = datetime.strptime(month + "-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = (start + timedelta(days=32)).replace(day=1)
    return start, end


def _in(ts: Optional[str], start: datetime, end: datetime) -> bool:
    return ts is not None and start <= parse_ts(ts) < end


def _health_at(store, asset: Dict[str, Any], when: datetime, config) -> Optional[float]:
    """Health at a past instant, re-evaluated from the observations up to then."""
    from ml.health_index import health_index
    from ml.revisit_priority import compute_revisit

    obs = [o for o in store.observations(asset["asset_id"]) if parse_ts(o["observation_ts"]) < when]
    if not obs:
        return None
    by_sess: Dict[str, float] = {}
    for o in obs:
        by_sess[o["session_id"]] = max(by_sess.get(o["session_id"], 0.0), o["severity_score"])
    comp = compute_revisit(list(by_sess.values()), obs[-1]["observation_ts"], config, when)
    open_d = [d["severity_score"] for d in store.list_defects(asset_id=asset["asset_id"])
              if parse_ts(d["first_detected_ts"]) < when and (d["resolved_ts"] is None or parse_ts(d["resolved_ts"]) >= when)]
    return health_index(comp["recent_severity"], open_d, comp["trend"], comp["days_since_inspection"], config)


def build_owner_report(store, site_id: str, month: str, config: Dict[str, Any]) -> Dict[str, Any]:
    oc = get_nested(config, "owner_report", default={}) or {}
    start, end = month_bounds(month)
    assets = store.list_assets(site_id)
    sessions = [s for s in store.list_sessions(site_id) if _in(s["started_ts"], start, end)]
    inspected = set()
    for s in sessions:
        inspected |= set(store.session_assets(s["session_id"]))
    defects = store.list_defects(site_id=site_id)
    instances = store.list_defect_instances(site_id=site_id)

    rows = []
    for a in assets:
        h_end = _health_at(store, a, end, config)
        h_prev = _health_at(store, a, start, config)
        rows.append({"asset_id": a["asset_id"], "asset_type": a["asset_type"], "health": h_end, "health_prev": h_prev,
                     "band": health_band(h_end), "criticality": a["criticality"], "condition": a["current_condition"]})
    rows.sort(key=lambda r: (r["health"] if r["health"] is not None else 101))
    site_now = site_health([{"health_index": r["health"], "criticality": r["criticality"]} for r in rows])
    site_prev = site_health([{"health_index": r["health_prev"], "criticality": r["criticality"]} for r in rows])

    opened = [d for d in defects if _in(d["first_detected_ts"], start, end)]
    resolved = [d for d in defects if _in(d.get("resolved_ts"), start, end)]
    open_at_end = [d for d in defects if parse_ts(d["first_detected_ts"]) < end and (d["resolved_ts"] is None or parse_ts(d["resolved_ts"]) >= end)]

    # cost of deferral
    costs = oc.get("repair_cost", {}) or {}
    failure = oc.get("failure_cost", {}) or {}
    growth_cost = float(oc.get("repair_cost_growth_per_month", 0.08))
    defer_m = float(oc.get("deferral_months", 3))
    currency = oc.get("currency", "EUR")
    deferral = []
    for d in open_at_end:
        fam = family(d["defect_type"])
        atype = d.get("asset_type")
        now_cost = float(get_nested(costs, fam, default=costs.get("default", 200.0)))
        inst = [i for i in instances if i["asset_id"] == d["asset_id"] and i["status"] == "active" and family(i["defect_type"]) == fam]
        dtc = [x for x in (days_until(i.get("predicted_critical_ts"), end) for i in inst) if x is not None]
        soonest = min(dtc) if dtc else None
        later = now_cost * (1 + growth_cost) ** defer_m
        risk = 0.0
        if soonest is not None and soonest <= defer_m * 30:
            risk = float(failure.get(atype, failure.get("default", 1000.0)))
        deferral.append({"defect_id": d["defect_id"], "asset_id": d["asset_id"], "defect_type": d["defect_type"],
                         "severity": d["severity_score"], "repair_now": round(now_cost, 2), "repair_deferred": round(later, 2),
                         "failure_risk_cost": round(risk, 2), "deferral_penalty": round(later - now_cost + risk, 2),
                         "days_to_critical": soonest})
    deferral.sort(key=lambda r: -r["deferral_penalty"])

    # solar energy loss
    kwh = float(oc.get("panel_kwh_per_month", 45.0))
    price = float(oc.get("energy_price_per_kwh", 0.25))
    loss_factor = oc.get("loss_per_unit_area", {"soiling": 0.9, "crack": 3.0, "corrosion": 0.5}) or {}
    energy = []
    for a in assets:
        if a["asset_type"] != "solar_panel":
            continue
        act = [i for i in instances if i["asset_id"] == a["asset_id"] and i["status"] == "active"]
        frac = min(1.0, sum(float(loss_factor.get(family(i["defect_type"]), 1.0)) * float(i["area_norm"]) for i in act))
        if not act and a["current_is_defect"]:
            frac = float(oc.get("loss_default_defective", 0.05))
        if frac > 0:
            energy.append({"asset_id": a["asset_id"], "loss_fraction": round(frac, 4), "kwh_lost": round(kwh * frac, 2),
                           "value_lost": round(kwh * frac * price, 2)})
    energy.sort(key=lambda r: -r["kwh_lost"])

    return {
        "site_id": site_id, "month": month, "generated_ts": datetime.now(timezone.utc).isoformat(), "currency": currency,
        "health": {"site": site_now, "site_prev_month": site_prev, "change": (None if site_now is None or site_prev is None else round(site_now - site_prev, 1)),
                   "assets": rows},
        "activity": {"missions": len(sessions), "assets_inspected": len(inspected), "assets_total": len(assets),
                     "coverage": (len(inspected) / len(assets)) if assets else 0.0},
        "defects": {"opened": len(opened), "resolved": len(resolved), "open_at_month_end": len(open_at_end)},
        "deferral": {"months": defer_m, "items": deferral, "total_penalty": round(sum(r["deferral_penalty"] for r in deferral), 2)},
        "energy": {"panels_affected": len(energy), "kwh_lost": round(sum(r["kwh_lost"] for r in energy), 2),
                   "value_lost": round(sum(r["value_lost"] for r in energy), 2), "items": energy},
    }


def render_owner_pdf(rep: Dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"AUDI owner report {rep['site_id']} {rep['month']}")
    st = getSampleStyleSheet()
    cur = rep["currency"]
    h = rep["health"]
    story = [Paragraph(f"Asset health report — {rep['site_id']} — {rep['month']}", st["Title"])]
    change = "" if h["change"] is None else f" ({h['change']:+.1f} vs previous month)"
    story.append(Paragraph(f"<b>Site health index:</b> {h['site'] if h['site'] is not None else 'n/a'} / 100{change}", st["Normal"]))
    a = rep["activity"]
    story.append(Paragraph(f"Missions {a['missions']} · assets inspected {a['assets_inspected']} of {a['assets_total']} "
                           f"({100 * a['coverage']:.0f}%)", st["Normal"]))
    d = rep["defects"]
    story.append(Paragraph(f"Defects opened {d['opened']} · resolved {d['resolved']} · open at month end {d['open_at_month_end']}", st["Normal"]))
    story.append(Paragraph(f"<b>Cost of deferring repairs {rep['deferral']['months']:.0f} months:</b> {cur} {rep['deferral']['total_penalty']:,.2f}", st["Normal"]))
    e = rep["energy"]
    story.append(Paragraph(f"<b>Estimated solar energy loss:</b> {e['kwh_lost']:,.1f} kWh ({cur} {e['value_lost']:,.2f}) on {e['panels_affected']} panels", st["Normal"]))

    def table(title, items, cols):
        story.append(Spacer(1, 10))
        story.append(Paragraph(title, st["Heading2"]))
        if not items:
            story.append(Paragraph("None", st["Italic"]))
            return
        t = Table([cols] + [[("" if i.get(c) is None else str(i.get(c))) for c in cols] for i in items[:40]], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f4f8")),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5dae3")), ("FONTSIZE", (0, 0), (-1, -1), 7)]))
        story.append(t)

    table("Asset health (worst first)", h["assets"], ["asset_id", "asset_type", "health", "health_prev", "band", "condition"])
    table("Deferral cost by defect", rep["deferral"]["items"], ["asset_id", "defect_type", "severity", "repair_now", "repair_deferred", "failure_risk_cost", "days_to_critical"])
    table("Energy loss by panel", e["items"], ["asset_id", "loss_fraction", "kwh_lost", "value_lost"])
    doc.build(story)
    return buf.getvalue()
