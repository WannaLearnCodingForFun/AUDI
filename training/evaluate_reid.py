#!/usr/bin/env python3
"""Re-identification accuracy on REAL repeated views (D5) → reports/reid/.

Input: the set recorded with scripts/capture_reid_set.py (`reid.dir`/manifest.csv).
Protocol: images from the gallery session (default: the earliest) enrol one
asset per object (every gallery image becomes a prototype; detected ArUco ids
fill the marker table). Every image from the other sessions is a query.
A query whose object is not in the gallery should come back as a NEW asset.

Methods compared on the same queries:
  global   — DINOv2 cosine top-1, accepted when ≥ reid.global_threshold
             (the previous repo's nearest-neighbour identity rule);
  cascade  — the full AssetIdentifier: ArUco → index top-k → SIFT/RANSAC →
             visual_only → new.

Metrics per method, overall and per query condition (normal / damaged):
top-1 identity accuracy over queries of known objects, false-new-asset rate
(known object reported new), false-match rate (unknown object matched to an
existing asset).

Usage: python training/evaluate_reid.py [--gallery day1]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from memory.vector_memory import PrototypeIndex  # noqa: E402
from ml.asset_identity import AssetIdentifier  # noqa: E402
from ml.config import cfg, load_config, resolve_path  # noqa: E402

SITE = "reid_eval"


def load_records(root: Path) -> List[Dict[str, str]]:
    path = root / "manifest.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _rates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    known = [r for r in rows if r["known"]]
    unknown = [r for r in rows if not r["known"]]
    return {
        "queries": len(rows),
        "known_queries": len(known),
        "unknown_queries": len(unknown),
        "top1_accuracy": (sum(r["pred"] == r["object_id"] for r in known) / len(known)) if known else None,
        "false_new_rate": (sum(r["pred"] is None for r in known) / len(known)) if known else None,
        "false_match_rate": (sum(r["pred"] is not None for r in unknown) / len(unknown)) if unknown else None,
    }


def evaluate(config: Dict[str, Any], records: List[Dict[str, str]], root: Path, embedder, gallery: Optional[str] = None) -> Dict[str, Any]:
    if not records:
        raise ValueError("no re-ID records")
    sessions = sorted({r["session"] for r in records}, key=lambda s: min(x["ts"] for x in records if x["session"] == s))
    gallery = gallery or sessions[0]
    if gallery not in sessions:
        raise ValueError(f"gallery session {gallery!r} not in {sessions}")
    if len(sessions) < 2:
        raise ValueError("record at least two sessions (gallery + queries)")
    rcfg, mcfg = cfg(config, "reid"), cfg(config, "matching")
    asset_type = rcfg["asset_type"]

    def load(rel: str) -> Image.Image:
        with Image.open(root / rel) as im:
            return im.convert("RGB")

    gal = [r for r in records if r["session"] == gallery]
    qry = [r for r in records if r["session"] != gallery]
    index = PrototypeIndex(embedder.embedding_dim, cfg(config, "memory", "vector_backend"),
                           max(int(mcfg["max_prototypes_per_asset"]), len(gal)), float(mcfg["prototype_novelty"]))
    gal_imgs = [load(r["path"]) for r in gal]
    gal_emb = embedder.encode_batch(gal_imgs)
    markers: Dict[int, str] = {}
    for pid, (r, e) in enumerate(zip(gal, gal_emb)):
        index.add(pid, r["object_id"], SITE, asset_type, e)
        if r.get("marker_id"):
            markers[int(r["marker_id"])] = r["object_id"]
    ident = AssetIdentifier(config, index, lambda pid: gal_imgs[pid])
    known_objects = {r["object_id"] for r in gal}
    threshold = float(rcfg["global_threshold"])

    out: Dict[str, List[Dict[str, Any]]] = {"global": [], "cascade": []}
    stages: Dict[str, int] = defaultdict(int)
    for r in qry:
        img = load(r["path"])
        emb = embedder.encode_image(img)
        base = {"object_id": r["object_id"], "condition": r["condition"], "known": r["object_id"] in known_objects, "path": r["path"]}
        top = index.search(emb, SITE, asset_type, 1)
        g_pred = top[0].asset_id if top and top[0].similarity >= threshold else None
        out["global"].append({**base, "pred": g_pred, "similarity": top[0].similarity if top else None})
        res = ident.identify(img, None, img, emb, asset_type, SITE, None, [], markers)
        out["cascade"].append({**base, "pred": res.asset_id, "stage": res.stage})
        stages[res.stage] += 1

    metrics: Dict[str, Any] = {"gallery_session": gallery, "query_sessions": [s for s in sessions if s != gallery],
                               "objects": sorted({r["object_id"] for r in records}), "gallery_objects": sorted(known_objects),
                               "markers": len(markers), "cascade_stages": dict(stages), "methods": {}}
    for m, rows in out.items():
        metrics["methods"][m] = {"overall": _rates(rows),
                                 **{c: _rates([x for x in rows if x["condition"] == c]) for c in sorted({x["condition"] for x in rows})}}
    metrics["per_query"] = out
    return metrics


def write_report(metrics: Dict[str, Any], out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    def f(v):
        return "n/a" if v is None else f"{v:.3f}"

    lines = ["# Re-identification on real repeated views", "",
             f"Gallery session `{metrics['gallery_session']}` ({len(metrics['gallery_objects'])} objects, {metrics['markers']} ArUco-tagged); "
             f"query sessions {', '.join(metrics['query_sessions'])}.", "",
             "| Method | Subset | Queries (known/unknown) | Top-1 accuracy | False-new rate | False-match rate |", "|---|---|---|---|---|---|"]
    for m, subsets in metrics["methods"].items():
        for name, r in subsets.items():
            lines.append(f"| {m} | {name} | {r['queries']} ({r['known_queries']}/{r['unknown_queries']}) | {f(r['top1_accuracy'])} | {f(r['false_new_rate'])} | {f(r['false_match_rate'])} |")
    lines += ["", f"Cascade decisions by stage: `{json.dumps(metrics['cascade_stages'])}`", ""]
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gallery", default=None)
    args = parser.parse_args()
    config = load_config()
    root = resolve_path(cfg(config, "reid", "dir"))
    records = load_records(root)
    if not records:
        sys.exit(f"No re-ID set at {root}. Record one first, e.g.\n"
                 f"  python scripts/capture_reid_set.py --object mug_01 --session day1\n"
                 f"  python scripts/capture_reid_set.py --object mug_01 --session day2 --condition damaged")
    from ml.embedding_model import AssetEmbeddingModel

    embedder = AssetEmbeddingModel.from_config(config)
    try:
        metrics = evaluate(config, records, root, embedder, args.gallery)
    except ValueError as exc:
        sys.exit(str(exc))
    print(write_report(metrics, resolve_path(cfg(config, "reid", "report_dir"))))


if __name__ == "__main__":
    main()
