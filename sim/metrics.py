"""Scores for simulated missions: re-ID, change, defect growth and forecasting."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from ml.forecast import forecast_series


def reid_metrics(log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rank-1 on revisits, false-new-asset rate, false-match rate.

    The memory id a true asset received at its first sighting is its reference; a later
    sighting is correct when it returns that id."""
    first: Dict[str, str] = {}
    owner: Dict[str, str] = {}
    revisits = correct = false_new = false_match = firsts = first_merged = 0
    for r in log:
        ev, gt = r["event"], r["gt"]["asset_id"]
        pred = ev.get("asset_id")
        if ev["type"] in ("error", "rejected", "duplicate") and pred is None:
            continue
        if gt not in first:
            firsts += 1
            if ev["type"] != "new_asset":
                first_merged += 1          # a never-seen asset matched to another one
            if pred:
                first[gt] = pred
                owner.setdefault(pred, gt)
            continue
        revisits += 1
        if pred == first[gt]:
            correct += 1
        elif ev["type"] == "new_asset":
            false_new += 1
        else:
            false_match += 1
    return {
        "revisits": revisits,
        "rank1": correct / revisits if revisits else None,
        "false_new_rate": false_new / revisits if revisits else None,
        "false_match_rate": false_match / revisits if revisits else None,
        "first_sightings": firsts,
        "first_sightings_merged": first_merged,
    }


def change_metrics(log: List[Dict[str, Any]], min_increase: float = 0.05) -> Dict[str, Any]:
    """Change decision vs ground truth (damage extent grew since the previous sighting).

    Three detectors on the same events (7.2): label difference, patch difference (visual b),
    pixel mask (changed fraction), and the engine's final M-gated decision."""
    prev: Dict[str, Dict[str, float]] = {}
    rows = []
    for r in log:
        g = r["gt"]
        ev = r["event"]
        if ev["type"] in ("error", "rejected"):
            continue
        before = prev.get(g["asset_id"])
        prev[g["asset_id"]] = dict(g["damage"])
        if before is None or ev["type"] == "new_asset":
            continue
        grew = any(g["damage"].get(k, 0.0) - before.get(k, 0.0) >= min_increase for k in set(g["damage"]) | set(before))
        ch = ev.get("change") or {}
        hist = ev.get("history") or {}
        rows.append({
            "truth": grew,
            "engine": bool(ch.get("significant")),
            "label": ev["condition"].get("label") != hist.get("reference_condition", hist.get("previous_condition")),
            "patch": (ch.get("visual_delta") or 0.0) >= 0.3,
            "mask": ((ev.get("change_mask") or {}).get("changed_fraction") or 0.0) >= 0.01,
        })

    def score(key: str) -> Dict[str, Optional[float]]:
        tp = sum(1 for x in rows if x[key] and x["truth"])
        fp = sum(1 for x in rows if x[key] and not x["truth"])
        fn = sum(1 for x in rows if not x[key] and x["truth"])
        unchanged = sum(1 for x in rows if not x["truth"])
        p = tp / (tp + fp) if tp + fp else None
        rc = tp / (tp + fn) if tp + fn else None
        f1 = (2 * p * rc / (p + rc)) if p and rc else (0.0 if (p is not None and rc is not None) else None)
        return {"precision": p, "recall": rc, "f1": f1, "false_changes_per_100_unchanged": 100.0 * fp / unchanged if unchanged else None}

    return {"pairs": len(rows), "changed": sum(x["truth"] for x in rows),
            **{k: score(k) for k in ("engine", "label", "patch", "mask")}}


def growth_metrics(result: Dict[str, Any], critical_extent: float = 0.6) -> Dict[str, Any]:
    """Defect instances vs simulated damage, and forecast error in days (growth model vs severity slope)."""
    store, world, log = result["store"], result["world"], result["log"]
    gt_by_mem: Dict[str, Counter] = defaultdict(Counter)
    for r in log:
        if r["event"].get("asset_id"):
            gt_by_mem[r["event"]["asset_id"]][r["gt"]["asset_id"]] += 1
    mem_of = {}
    for mem, c in gt_by_mem.items():
        gt = c.most_common(1)[0][0]
        mem_of.setdefault(gt, mem)
    days = world.days_between_missions
    out = {"damaged_assets": 0, "tracked": 0, "instances_per_damaged_asset": [], "growth_errors_days": [], "slope_errors_days": []}
    for a in world.assets:
        if not a.damages:
            continue
        out["damaged_assets"] += 1
        mem = mem_of.get(a.asset_id)
        if mem is None:
            continue
        inst = [i for i in store.list_defect_instances(asset_id=mem)]
        if inst:
            out["tracked"] += 1
        out["instances_per_damaged_asset"].append(len(inst))
        d = a.damages[0]
        # truth: first mission index (continuous) where extent ≥ critical
        t_crit = None
        for k in np.arange(d.start_mission, d.start_mission + 200, 0.05):
            if d.extent(int(np.floor(k))) >= critical_extent and d.law == "linear":
                pass
            ext = (d.e0 * (1 + d.rate * (k - d.start_mission)) ** d.exponent) if d.law == "power" else d.e0 + d.rate * (k - d.start_mission)
            if ext >= critical_extent:
                t_crit = k * days
                break
        best = max(inst, key=lambda i: len(store.instance_measurements(i["instance_id"])), default=None)
        if best is None or t_crit is None:
            continue
        ms = store.instance_measurements(best["instance_id"])
        if len(ms) < 3:
            continue
        # growth model: scale the measured area to extent by the last GT ratio, fit, solve for critical extent
        last_m = ms[-1]
        gt_last = [r["gt"]["damage"] for r in log if r["event"].get("asset_id") == mem and r["event"].get("observation_id") == last_m["observation_id"]]
        if not gt_last:
            continue
        ext_last = max(gt_last[0].values() or [0.0])
        if ext_last <= 0 or last_m["area_norm"] <= 0:
            continue
        k = ext_last / last_m["area_norm"]
        series = [{"ts": m["ts"], "area_norm": m["area_norm"] * k} for m in ms]
        cfgc = {"forecast": {"critical": {"default": {"area_norm": critical_extent}, "crack": {"area_norm": critical_extent},
                                          "corrosion": {"area_norm": critical_extent}, "soiling": {"area_norm": critical_extent}},
                             "max_horizon_days": 1e6}}
        f = forecast_series(series, best["defect_type"], cfgc)
        from ml.revisit_priority import parse_ts

        t0 = parse_ts(log[0]["event"]["timestamp"]) if log else None
        if f is not None and f.days_to_critical is not None and t0 is not None:
            pred_day = (parse_ts(ms[-1]["ts"]) - t0).total_seconds() / 86400.0 + f.days_to_critical
            out["growth_errors_days"].append(pred_day - t_crit)
        # baseline: per-mission severity slope extrapolated to the critical severity level
        sev = [(r["mission"], r["event"]["severity_score"]) for r in log if r["event"].get("asset_id") == mem and r["event"]["type"] != "new_asset"]
        if len(sev) >= 3:
            x = np.array([s[0] for s in sev], float) * days
            y = np.array([s[1] for s in sev], float)
            if np.ptp(x) > 0:
                slope, icpt = np.polyfit(x, y, 1)
                if slope > 1e-9:
                    out["slope_errors_days"].append((0.8 - icpt) / slope - t_crit)
    for k in ("growth_errors_days", "slope_errors_days"):
        v = out[k]
        out[k.replace("errors_days", "mae_days")] = float(np.mean(np.abs(v))) if v else None
    return out


def site_map_hit_rate(log: List[Dict[str, Any]]) -> Optional[float]:
    """Share of detections that fell inside the correct asset footprint (1.2 verification)."""
    hits = total = 0
    for r in log:
        sm = (r["event"].get("geo") or {}).get("site_map")
        if sm is None:
            continue
        total += 1
        hits += int(sm.get("asset_id") is not None and sm["asset_id"] == r["event"].get("asset_id"))
    return hits / total if total else None
