#!/usr/bin/env python3
"""Simulated-mission evaluation and ablation studies (upgrades 4.3 and 7.2).

  python scripts/run_sim_eval.py                      # full: 3 seeds × 6 missions → reports/sim/
  python scripts/run_sim_eval.py --quick              # CI: 1 seed × 4 missions, asserts the gates
  python scripts/run_sim_eval.py --missions-dir runs/gazebo_2026-11-03   # Gazebo/PX4 recordings

Studies (one table each):
  re-ID     appearance only | location + appearance | location + appearance + SIFT verification
  change    label difference | patch difference (b) | pixel mask | engine decision (M-gate)
  forecast  severity-slope extrapolation | per-defect growth model (2.3), error in days
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ml.config import load_config, resolve_path  # noqa: E402
from sim.metrics import change_metrics, growth_metrics, reid_metrics  # noqa: E402
from sim.render import read_mission  # noqa: E402
from sim.runner import run, sim_config  # noqa: E402
from sim.world import build_world  # noqa: E402

MODES = ["appearance", "location_appearance", "location_appearance_sift"]
GATES = {"rank1_min": 0.80, "false_new_max": 0.15, "mask_auc_min": 0.75}   # CI regression gates on the quick run


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def mask_auc(log) -> Optional[float]:
    prev, pos, neg = {}, [], []
    for r in log:
        g, ev = r["gt"], r["event"]
        b = prev.get(g["asset_id"])
        prev[g["asset_id"]] = dict(g["damage"])
        if b is None or ev["type"] in ("new_asset", "rejected", "error"):
            continue
        grew = any(g["damage"].get(k, 0) - b.get(k, 0) >= 0.05 for k in set(g["damage"]) | set(b))
        f = (ev.get("change_mask") or {}).get("changed_fraction") or 0.0
        (pos if grew else neg).append(f)
    if not pos or not neg:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def evaluate(seeds: List[int], missions: int, missions_dir: Optional[Path] = None) -> Dict[str, Any]:
    base = load_config()
    out: Dict[str, Any] = {"seeds": seeds, "missions": missions, "reid": {}, "change": None, "forecast": None, "timing_s": {}}
    from ml.embedding_model import AssetEmbeddingModel

    full_runs = []
    for mode in MODES:
        rows = []
        t0 = time.time()
        for seed in seeds:
            world = build_world(seed=seed, missions=missions)
            cfg = sim_config(base, mode=mode)
            frames = None
            if missions_dir is not None:
                frames = [list(read_mission(p)) for p in sorted(missions_dir.iterdir()) if (p / "frames.jsonl").exists()]
            res = run(world, len(frames) if frames else missions, cfg, AssetEmbeddingModel.from_config(cfg), seed=seed, frames_by_mission=frames)
            rows.append(reid_metrics(res["log"]))
            if mode == MODES[-1]:
                full_runs.append(res)
        out["timing_s"][mode] = round(time.time() - t0, 1)
        out["reid"][mode] = {k: _mean([r[k] for r in rows]) for k in ("rank1", "false_new_rate", "false_match_rate")}
        out["reid"][mode]["revisits"] = int(sum(r["revisits"] for r in rows))
    ch = [change_metrics(r["log"]) for r in full_runs]
    out["change"] = {
        "pairs": sum(c["pairs"] for c in ch), "changed": sum(c["changed"] for c in ch),
        **{k: {m: _mean([c[k][m] for c in ch]) for m in ("precision", "recall", "f1", "false_changes_per_100_unchanged")}
           for k in ("label", "patch", "mask", "engine")},
        "mask_auc": _mean([mask_auc(r["log"]) for r in full_runs]),
    }
    gm = [growth_metrics(r) for r in full_runs]
    out["forecast"] = {
        "damaged_assets": sum(g["damaged_assets"] for g in gm), "tracked_assets": sum(g["tracked"] for g in gm),
        "severity_slope_mae_days": _mean([g["slope_mae_days"] for g in gm]),
        "growth_model_mae_days": _mean([g["growth_mae_days"] for g in gm]),
        "series_slope": sum(len(g["slope_errors_days"]) for g in gm), "series_growth": sum(len(g["growth_errors_days"]) for g in gm),
    }
    return out


def to_markdown(r: Dict[str, Any]) -> str:
    f = lambda v, p=3: "n/a" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
    L = ["# Simulated-mission evaluation (4.3) and ablations (7.2)", "",
         f"Seeds {r['seeds']}, {r['missions']} missions each. Synthetic site: rows of near-identical solar panels, a pole line, a wall, "
         "machinery; damage grows between missions. Backbone: weight-free projection embedder, so absolute numbers are a lower bound "
         "for DINOv2; the comparison between arms is the point.", "",
         "## Re-identification", "", "| Arm | Rank-1 | False-new rate | False-match rate | Revisits |", "|---|---|---|---|---|"]
    for m in MODES:
        x = r["reid"][m]
        L.append(f"| {m.replace('_', ' + ')} | {f(x['rank1'])} | {f(x['false_new_rate'])} | {f(x['false_match_rate'])} | {x['revisits']} |")
    c = r["change"]
    L += ["", "## Change detection", "", f"{c['pairs']} revisit pairs, {c['changed']} with real damage growth. Mask AUC {f(c['mask_auc'])}.", "",
          "| Detector | Precision | Recall | F1 | False changes / 100 unchanged |", "|---|---|---|---|---|"]
    for k, name in (("label", "label difference"), ("patch", "patch difference (b ≥ 0.3)"), ("mask", "pixel mask (≥ 1%)"), ("engine", "engine M-gate")):
        L.append(f"| {name} | {f(c[k]['precision'])} | {f(c[k]['recall'])} | {f(c[k]['f1'])} | {f(c[k]['false_changes_per_100_unchanged'], 1)} |")
    g = r["forecast"]
    L += ["", "## Forecast", "", f"{g['damaged_assets']} damaged assets, {g['tracked_assets']} with tracked defect instances.", "",
          "| Method | MAE (days) to critical | Series |", "|---|---|---|",
          f"| severity slope | {f(g['severity_slope_mae_days'], 1)} | {g['series_slope']} |",
          f"| growth model (2.3) | {f(g['growth_model_mae_days'], 1)} | {g['series_growth']} |", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--missions", type=int, default=None)
    ap.add_argument("--missions-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=resolve_path("reports/sim"))
    args = ap.parse_args(argv)
    seeds = args.seeds or ([7] if args.quick else [7, 11, 13])
    missions = args.missions or (4 if args.quick else 6)
    r = evaluate(seeds, missions, args.missions_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    name = "quick" if args.quick else "ablation"
    (args.out / f"{name}.json").write_text(json.dumps(r, indent=2))
    md = to_markdown(r)
    (args.out / f"{name}.md").write_text(md)
    print(md)
    if args.quick:
        full = r["reid"][MODES[-1]]
        fails = []
        if (full["rank1"] or 0) < GATES["rank1_min"]:
            fails.append(f"rank1 {full['rank1']} < {GATES['rank1_min']}")
        if (full["false_new_rate"] or 1) > GATES["false_new_max"]:
            fails.append(f"false_new {full['false_new_rate']} > {GATES['false_new_max']}")
        if (r["change"]["mask_auc"] or 0) < GATES["mask_auc_min"]:
            fails.append(f"mask AUC {r['change']['mask_auc']} < {GATES['mask_auc_min']}")
        if fails:
            print("GATES FAILED: " + "; ".join(fails))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
