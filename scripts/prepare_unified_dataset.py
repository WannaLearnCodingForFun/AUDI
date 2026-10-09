#!/usr/bin/env python3
"""Build the unified multi-asset dataset from dataset/raw/ (Phase 1d).

Steps
1. Walk every enabled source, map each image directory through
   ml/condition_mapping.py, parse annotations (VOC / YOLO / masks) and turn each
   labelled object into a sample (whole image, box crop or mask tile).
2. Load + crop + resize each candidate, compute its perceptual hash and stage it.
3. De-duplicate near-identical samples across sources (same asset type) before
   any split, merge/drop classes below the minimum, cap classes, then split
   train/val/test stratified per class while keeping all samples from one source
   image in the same split.
4. Write dataset/processed/manifest.csv, the YOLO detection export,
   dataset/processed/STATS.md and dataset/README.md.

Usage:  python scripts/prepare_unified_dataset.py [--sources a b ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ml.condition_mapping import PARENT_CLASS, UNIFIED_CONDITIONS, Rule, match_rule  # noqa: E402
from ml.config import cfg, load_config, resolve_path  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ASSET_TYPES = list(UNIFIED_CONDITIONS)
MANIFEST_FIELDS = ["path", "asset_type", "condition", "defect_type", "is_defect", "source", "split", "phash"]
METADATA_FILE = ".kaggle_metadata.json"

Box = Tuple[int, int, int, int]


@dataclass
class Sample:
    src: str                      # absolute path of the source image
    source: str
    asset_type: str
    condition: str
    defect_type: Optional[str]
    is_defect: bool
    raw_class: str
    region: Optional[Box] = None  # crop box in source pixels (None = whole image)
    phash: Optional[int] = None
    staged: Optional[str] = None
    split: Optional[str] = None


@dataclass
class Detection:
    src: str
    boxes: List[Tuple[int, Box]]  # (class index, box)
    size: Tuple[int, int]


@dataclass
class Collected:
    samples: List[Sample] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    unmapped: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)


# ── annotation parsing ─────────────────────────────────────────────────────

def images_in(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def pad_box(box: Box, size: Tuple[int, int], padding: float) -> Box:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = w * padding, h * padding
    return (
        int(max(0, x1 - px)), int(max(0, y1 - py)),
        int(min(size[0], x2 + px)), int(min(size[1], y2 + py)),
    )


def parse_voc(xml_path: Path) -> Tuple[Tuple[int, int], List[Tuple[str, Box]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    wh = (int(size.findtext("width")), int(size.findtext("height")))
    objects = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        box = tuple(int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
        objects.append((obj.findtext("name").strip(), box))
    return wh, objects


def parse_yolo(txt_path: Path, size: Tuple[int, int]) -> List[Tuple[str, Box]]:
    """YOLO boxes (cls cx cy w h) or polygons (cls x1 y1 x2 y2 ...) → pixel boxes."""
    w, h = size
    out = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, vals = parts[0], [float(v) for v in parts[1:]]
        if len(vals) == 4:
            cx, cy, bw, bh = vals
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        else:
            xs, ys = vals[0::2], vals[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        out.append((cls, (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))))
    return out


def centre_inside(inner: Box, outer: Box) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


# ── handlers ───────────────────────────────────────────────────────────────

def handle_folder(rule: Rule, folder: Path, rel: str, source: str, out: Collected) -> None:
    defect_type = rule.defect_type
    if rule.is_defect and defect_type is None:
        defect_type = folder.name
    for img in images_in(folder):
        out.samples.append(Sample(str(img), source, rule.asset_type, rule.condition, defect_type, rule.is_defect, rel))


def handle_mask_tiles(rule: Rule, folder: Path, rel: str, source: str, out: Collected, prep: dict) -> None:
    tile = int(prep["mask_tile_size"])
    thr = float(prep["mask_crack_fraction"])
    mask_dir = folder.parent / "masks"
    for img in images_in(folder):
        mask_path = mask_dir / img.name
        if not mask_path.exists():
            out.skipped[f"{source}: image without mask"] += 1
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        h, w = mask.shape
        for y in range(0, h - tile + 1, tile):
            for x in range(0, w - tile + 1, tile):
                frac = float(mask[y:y + tile, x:x + tile].mean())
                region = (x, y, x + tile, y + tile)
                if frac == 0.0:
                    out.samples.append(Sample(str(img), source, rule.asset_type, "normal", None, False, f"{rel}:tile_clean", region))
                elif frac >= thr:
                    out.samples.append(Sample(str(img), source, rule.asset_type, "crack", "crack", True, f"{rel}:tile_crack", region))
                else:
                    out.skipped[f"{source}: tile below crack fraction"] += 1


def handle_voc_insulator(rule: Rule, folder: Path, rel: str, source: str, out: Collected, prep: dict) -> None:
    labels = folder.parent / "labels"
    det_cls = ASSET_TYPES.index(rule.asset_type)
    for img in images_in(folder):
        stem = img.stem
        xmls = [p for p in (labels / f"{stem}.xml", labels / "insulator" / f"{stem}.xml", labels / "defect" / f"{stem}.xml") if p.exists()]
        if not xmls:
            out.skipped[f"{source}: image without annotation"] += 1
            continue
        size, insulators, defects = None, [], []
        for x in xmls:
            size, objs = parse_voc(x)
            insulators += [b for n, b in objs if n == "insulator"]
            defects += [b for n, b in objs if n == "defect"]
        if insulators:
            out.detections.append(Detection(str(img), [(det_cls, b) for b in insulators], size))
        for box in insulators:
            if min(box[2] - box[0], box[3] - box[1]) < int(prep["min_crop_side"]):
                out.skipped[f"{source}: box below min_crop_side"] += 1
                continue
            key = "defect" if any(centre_inside(d, box) for d in defects) else "insulator"
            cond, dt, is_def = rule.class_map[key]
            out.samples.append(Sample(
                str(img), source, rule.asset_type, cond, dt, is_def, f"{rel}:{key}",
                pad_box(box, size, float(prep["crop_padding"])),
            ))


def handle_yolo_labels(rule: Rule, folder: Path, rel: str, source: str, out: Collected, prep: dict) -> None:
    labels = folder.parent / "labels"
    for img in images_in(folder):
        txt = labels / f"{img.stem}.txt"
        if not txt.exists() or txt.stat().st_size == 0:
            out.skipped[f"{source}: image without boxes"] += 1
            continue
        with Image.open(img) as im:
            size = im.size
        for cls, box in parse_yolo(txt, size):
            if cls not in rule.class_map:
                out.unmapped[f"{source}: annotation class {cls}"] += 1
                continue
            if min(box[2] - box[0], box[3] - box[1]) < int(prep["min_crop_side"]):
                out.skipped[f"{source}: box below min_crop_side"] += 1
                continue
            cond, dt, is_def = rule.class_map[cls]
            out.samples.append(Sample(
                str(img), source, rule.asset_type, cond, dt, is_def, f"{rel}:class{cls}",
                pad_box(box, size, float(prep["crop_padding"])),
            ))


def collect_source(name: str, raw_dir: Path, prep: dict) -> Collected:
    out = Collected()
    base = raw_dir / name
    dirs = sorted({p.parent for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS})
    for folder in dirs:
        rel = folder.relative_to(base).as_posix()
        rule = match_rule(name, rel)
        if rule is None:
            out.unmapped[f"{name}: {rel or '.'} ({len(images_in(folder))} images)"] += 1
            continue
        if rule.handler is None:
            handle_folder(rule, folder, rel, name, out)
        elif rule.handler == "mask_tiles":
            handle_mask_tiles(rule, folder, rel, name, out, prep)
        elif rule.handler == "voc_insulator":
            handle_voc_insulator(rule, folder, rel, name, out, prep)
        elif rule.handler == "yolo_labels":
            handle_yolo_labels(rule, folder, rel, name, out, prep)
        elif rule.handler == "skip":
            continue
        else:
            raise ValueError(f"unknown handler {rule.handler!r} for {name}")
    for p in base.rglob("*.csv"):
        out.unmapped[f"{name}: non-image data file {p.relative_to(base).as_posix()}"] += 1
    return out


# ── staging (parallel) ─────────────────────────────────────────────────────

def stage_one(args) -> Tuple[int, Optional[int], Optional[str], Optional[str]]:
    import imagehash

    idx, src, region, staged, max_side, quality = args
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            if region is not None:
                im = im.crop(region)
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            h = int(str(imagehash.phash(im)), 16)
            im.save(staged, "JPEG", quality=quality)
        return idx, h, staged, None
    except Exception as exc:  # noqa: BLE001
        return idx, None, None, f"{type(exc).__name__}: {exc}"


# ── selection ──────────────────────────────────────────────────────────────

def dedupe(samples: List[Sample], threshold: int, fine_grained: set) -> Tuple[List[Sample], Counter, Counter]:
    kept: List[Sample] = []
    hashes: Dict[str, List[int]] = defaultdict(list)
    owners: Dict[str, List[Sample]] = defaultdict(list)
    removed, conflicts = Counter(), Counter()
    for s in samples:
        arr = np.array(hashes[s.asset_type], dtype=np.uint64)
        if arr.size:
            dist = np.bitwise_count(arr ^ np.uint64(s.phash))
            near = np.flatnonzero(dist <= threshold)
            dup = None
            for j in near:
                other = owners[s.asset_type][int(j)]
                if other.source == s.source and other.condition != s.condition and s.source in fine_grained:
                    continue
                dup = other
                break
            if dup is not None:
                kind = "within-source" if dup.source == s.source else "cross-source"
                removed[(s.source, kind)] += 1
                if dup.condition != s.condition:
                    conflicts[(dup.source, dup.condition, s.source, s.condition)] += 1
                continue
        hashes[s.asset_type].append(s.phash)
        owners[s.asset_type].append(s)
        kept.append(s)
    return kept, removed, conflicts


def merge_small(samples: List[Sample], minimum: int) -> Tuple[List[Sample], List[str]]:
    log = []
    while True:
        counts = Counter((s.asset_type, s.condition) for s in samples)
        small = sorted(k for k, n in counts.items() if n < minimum)
        if not small:
            return samples, log
        asset_type, cond = small[0]
        parent = PARENT_CLASS.get(asset_type, {}).get(cond)
        if parent and parent != cond:
            for s in samples:
                if s.asset_type == asset_type and s.condition == cond:
                    s.condition = parent
            log.append(f"{asset_type}/{cond}: {counts[small[0]]} < {minimum} → merged into {parent}")
        else:
            samples = [s for s in samples if not (s.asset_type == asset_type and s.condition == cond)]
            log.append(f"{asset_type}/{cond}: {counts[small[0]]} < {minimum} → dropped (no parent class)")


def cap_classes(samples: List[Sample], cap: int, rng: random.Random) -> List[Sample]:
    """Round-robin over sources so a capped class keeps source diversity."""
    by_class: Dict[Tuple[str, str], Dict[str, List[Sample]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        by_class[(s.asset_type, s.condition)][s.source].append(s)
    out = []
    for key in sorted(by_class):
        pools = [list(v) for _, v in sorted(by_class[key].items())]
        for p in pools:
            rng.shuffle(p)
        chosen = []
        while len(chosen) < cap and any(pools):
            for p in pools:
                if p and len(chosen) < cap:
                    chosen.append(p.pop())
        out.extend(chosen)
    return out


def split_samples(samples: List[Sample], fractions: Dict[str, float], rng: random.Random) -> None:
    """Stratified per class; every sample from one source image lands in the same split."""
    group_split: Dict[str, str] = {}
    by_class: Dict[Tuple[str, str], List[Sample]] = defaultdict(list)
    for s in samples:
        by_class[(s.asset_type, s.condition)].append(s)
    for key in sorted(by_class):
        items = by_class[key]
        n = len(items)
        target = {k: round(n * fractions[k]) for k in ("val", "test")}
        if n >= 3:
            target = {k: max(1, v) for k, v in target.items()}
        target["train"] = n - target["val"] - target["test"]
        groups: Dict[str, List[Sample]] = defaultdict(list)
        for s in items:
            groups[s.src].append(s)
        filled = Counter()
        pending = []
        for src, members in groups.items():
            if src in group_split:
                filled[group_split[src]] += len(members)
            else:
                pending.append(src)
        rng.shuffle(pending)
        pending.sort(key=lambda g: -len(groups[g]))  # place large groups first, ties stay shuffled
        for src in pending:
            group_split[src] = max(("train", "val", "test"), key=lambda k: (target[k] - filled[k]) / max(target[k], 1))
            filled[group_split[src]] += len(groups[src])
        for s in items:
            s.split = group_split[s.src]


# ── outputs ────────────────────────────────────────────────────────────────

def export_detection(detections: List[Detection], samples: List[Sample], out_dir: Path, val_fraction: float, rng: random.Random) -> Counter:
    det_dir = out_dir / "detection"
    if det_dir.exists():
        shutil.rmtree(det_dir)
    split_of = {s.src: s.split for s in samples}
    counts = Counter()
    for d in detections:
        split = split_of.get(d.src)
        split = "val" if split in {"val", "test"} else ("train" if split == "train" else ("val" if rng.random() < val_fraction else "train"))
        img_dir, lbl_dir = det_dir / "images" / split, det_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        src = Path(d.src)
        stem = f"{src.parent.parent.name}_{src.stem}"
        shutil.copy2(src, img_dir / f"{stem}{src.suffix.lower()}")
        w, h = d.size
        lines = []
        for cls, (x1, y1, x2, y2) in d.boxes:
            lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            counts[(split, ASSET_TYPES[cls])] += 1
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    yaml_lines = [f"path: {det_dir.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {i}: {name}" for i, name in enumerate(ASSET_TYPES)]
    det_dir.mkdir(parents=True, exist_ok=True)
    (det_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return counts


def load_metadata(raw_dir: Path, name: str) -> dict:
    p = raw_dir / name / METADATA_FILE
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def write_stats(path: Path, *, samples, collected, removed, conflicts, merge_log, failures, det_counts, params, elapsed) -> None:
    L = ["# Unified dataset statistics", "", f"Generated by `scripts/prepare_unified_dataset.py` in {elapsed:.0f} s.", ""]
    L += ["## Parameters", ""] + [f"- {k}: {v}" for k, v in params.items()] + [""]

    L += ["## Final counts (asset_type / condition × split)", "", "| asset_type | condition | train | val | test | total |", "|---|---|---|---|---|---|"]
    table = defaultdict(Counter)
    for s in samples:
        table[(s.asset_type, s.condition)][s.split] += 1
    for (at, cond) in sorted(table, key=lambda k: (ASSET_TYPES.index(k[0]), k[1])):
        c = table[(at, cond)]
        L.append(f"| {at} | {cond} | {c['train']} | {c['val']} | {c['test']} | {sum(c.values())} |")
    missing = [f"{at}/{c}" for at, conds in UNIFIED_CONDITIONS.items() for c in conds if (at, c) not in table]
    L += ["", f"Unified conditions with no data: {', '.join(missing) if missing else 'none'}", ""]

    L += ["## Contribution per source (after dedupe, caps)", "", "| source | asset_type | condition | images |", "|---|---|---|---|"]
    for (src, at, cond), n in sorted(Counter((s.source, s.asset_type, s.condition) for s in samples).items()):
        L.append(f"| {src} | {at} | {cond} | {n} |")
    L.append("")

    L += ["## Candidates per source (before dedupe)", "", "| source | candidates |", "|---|---|"]
    for name, col in collected.items():
        L.append(f"| {name} | {len(col.samples)} |")
    L.append("")

    L += ["## Near-duplicates removed (perceptual hash)", "", "| source | kind | removed |", "|---|---|---|"]
    for (src, kind), n in sorted(removed.items()):
        L.append(f"| {src} | {kind} | {n} |")
    if conflicts:
        L += ["", "Duplicates whose labels disagreed (first occurrence kept):", ""]
        L += [f"- {a}/{ac} vs {b}/{bc}: {n}" for (a, ac, b, bc), n in sorted(conflicts.items())]
    L.append("")

    L += ["## Class merges / drops", ""] + ([f"- {m}" for m in merge_log] or ["- none"]) + [""]

    L += ["## Unmapped classes / folders", ""]
    unmapped = [(k, n) for col in collected.values() for k, n in col.unmapped.items()]
    L += [f"- {k}" + (f" × {n}" if n > 1 else "") for k, n in sorted(unmapped)] or ["- none"]
    L.append("")

    L += ["## Skipped items", ""]
    skipped = [(k, n) for col in collected.values() for k, n in col.skipped.items()]
    L += [f"- {k}: {n}" for k, n in sorted(skipped)] or ["- none"]
    if failures:
        L += ["", f"Unreadable images: {len(failures)}"] + [f"- {f}" for f in failures[:20]]
    L.append("")

    L += ["## Detection export (dataset/processed/detection)", "", "| split | class | boxes |", "|---|---|---|"]
    for (split, cls), n in sorted(det_counts.items()):
        L.append(f"| {split} | {cls} | {n} |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def write_readme(path: Path, sources: List[dict], raw_dir: Path, samples: List[Sample]) -> None:
    from ml.condition_mapping import rules_for

    per_source = Counter(s.source for s in samples)
    L = [
        "# AUDI unified dataset",
        "",
        "Built by `scripts/download_datasets.py` (raw Kaggle downloads → `dataset/raw/`) and",
        "`scripts/prepare_unified_dataset.py` (→ `dataset/processed/`). Neither folder is committed.",
        "",
        "- `dataset/processed/manifest.csv` — one row per image: "
        + ", ".join(f"`{f}`" for f in MANIFEST_FIELDS),
        "- `dataset/processed/classification/<asset_type>/<condition>/` — cropped / resized images",
        "- `dataset/processed/detection/` — YOLO-format asset boxes (`data.yaml`)",
        "- `dataset/processed/STATS.md` — counts, de-duplication, merges and unmapped classes",
        "",
        "## Unified conditions",
        "",
    ]
    L += [f"- **{at}**: {', '.join(conds)}" for at, conds in UNIFIED_CONDITIONS.items()]
    L += ["", "## Sources and licences", "", "| source | Kaggle dataset | licence | images used | notes |", "|---|---|---|---|---|"]
    for ds in sources:
        meta = load_metadata(raw_dir, ds["name"])
        lic = ", ".join(meta.get("licenses") or []) or "not listed (check the Kaggle page)"
        notes = "; ".join(r.note for r in rules_for(ds["name"]) if r.note) or ""
        if not ds.get("enabled", True):
            notes = ("disabled in config.yaml; " + notes).strip("; ")
        url = f"https://www.kaggle.com/datasets/{ds['slug']}"
        L.append(f"| {ds['name']} | [{ds['slug']}]({url}) | {lic} | {per_source.get(ds['name'], 0)} | {notes} |")
    L += [
        "",
        "Licences are read from Kaggle metadata at download time. Datasets marked",
        "`copyright-authors` or `unknown` are used here for research/evaluation only; check the",
        "dataset page before any redistribution or commercial use.",
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--sources", nargs="+", help="restrict to these source names")
    args = parser.parse_args()

    config = load_config(args.config)
    dcfg = cfg(config, "datasets")
    prep = cfg(config, "datasets", "prepare")
    raw_dir = resolve_path(prep["raw_dir"])
    out_dir = resolve_path(prep["output_dir"])
    cap = int(dcfg["max_images_per_class"])
    minimum = int(dcfg["min_images_per_class"])
    threshold = int(dcfg["phash_hamming_threshold"])
    split_cfg = dcfg["split"]
    seed = int(split_cfg["seed"])
    rng = random.Random(seed)
    pool_size = int(prep["candidate_pool_factor"]) * cap
    workers = int(prep["workers"]) or os.cpu_count() or 1

    sources = [s for s in dcfg["sources"] if s.get("enabled", True)]
    if args.sources:
        sources = [s for s in sources if s["name"] in args.sources]
    t0 = time.time()

    collected: Dict[str, Collected] = {}
    for ds in sources:
        if not (raw_dir / ds["name"]).exists():
            print(f"[miss] {ds['name']}: not downloaded — run scripts/download_datasets.py")
            continue
        col = collect_source(ds["name"], raw_dir, prep)
        collected[ds["name"]] = col
        print(f"[scan] {ds['name']}: {len(col.samples)} candidates, {len(col.detections)} detection images, "
              f"{sum(col.unmapped.values())} unmapped")

    # Bounded candidate pools per (source, class), in config source order (earlier source wins dedupe).
    candidates: List[Sample] = []
    for name, col in collected.items():
        groups: Dict[Tuple[str, str], List[Sample]] = defaultdict(list)
        for s in col.samples:
            groups[(s.asset_type, s.condition)].append(s)
        for key in sorted(groups):
            g = groups[key]
            rng.shuffle(g)
            candidates.extend(g[:pool_size])
    print(f"[pool] {len(candidates)} candidates to hash (pool ≤ {pool_size} per source/class)")

    staging = out_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    jobs = [
        (i, s.src, s.region, str(staging / f"{i:07d}.jpg"), int(prep["image_max_side"]), int(prep["jpeg_quality"]))
        for i, s in enumerate(candidates)
    ]
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, (i, h, staged, err) in enumerate(pool.map(stage_one, jobs, chunksize=64), 1):
            if err:
                failures.append(f"{candidates[i].src}: {err}")
            else:
                candidates[i].phash, candidates[i].staged = h, staged
            if n % 5000 == 0:
                print(f"  staged {n}/{len(jobs)}", flush=True)
    candidates = [s for s in candidates if s.phash is not None]

    kept, removed, conflicts = dedupe(candidates, threshold, set(prep["fine_grained_sources"]))
    print(f"[dedupe] kept {len(kept)} of {len(candidates)} (removed {sum(removed.values())})")
    kept, merge_log = merge_small(kept, minimum)
    for m in merge_log:
        print(f"[merge] {m}")
    final = cap_classes(kept, cap, rng)
    split_samples(final, {k: float(split_cfg[k]) for k in ("train", "val", "test")}, rng)

    cls_dir = out_dir / "classification"
    if cls_dir.exists():
        shutil.rmtree(cls_dir)
    rows = []
    for s in sorted(final, key=lambda x: (x.asset_type, x.condition, x.source, x.staged)):
        dest_dir = cls_dir / s.asset_type / s.condition
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{s.source}_{s.phash:016x}_{Path(s.staged).stem}.jpg"
        shutil.move(s.staged, dest)
        rows.append({
            "path": dest.relative_to(ROOT).as_posix(), "asset_type": s.asset_type, "condition": s.condition,
            "defect_type": s.defect_type or "", "is_defect": int(s.is_defect and s.condition != "normal"),
            "source": s.source, "split": s.split, "phash": f"{s.phash:016x}",
        })
    shutil.rmtree(staging)
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    detections = [d for col in collected.values() for d in col.detections]
    det_counts = export_detection(detections, final, out_dir, float(prep["detection_val_fraction"]), rng)

    params = {
        "max_images_per_class": cap, "min_images_per_class": minimum, "phash_hamming_threshold": threshold,
        "candidate_pool_per_source_class": pool_size, "split": dict(split_cfg), "image_max_side": prep["image_max_side"],
        "fine_grained_sources (label-distinct near-duplicates kept)": prep["fine_grained_sources"],
    }
    write_stats(out_dir / "STATS.md", samples=final, collected=collected, removed=removed, conflicts=conflicts,
                merge_log=merge_log, failures=failures, det_counts=det_counts, params=params, elapsed=time.time() - t0)
    write_readme(ROOT / "dataset" / "README.md", dcfg["sources"], raw_dir, final)

    split_counts = Counter(r["split"] for r in rows)
    print(f"[done] {len(rows)} images → {out_dir / 'manifest.csv'} ({dict(split_counts)}) in {time.time() - t0:.0f} s")
    print(f"[done] stats → {out_dir / 'STATS.md'}")


if __name__ == "__main__":
    main()
