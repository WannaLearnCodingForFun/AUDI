#!/usr/bin/env python3
"""
Prepare a balanced subset of the Kaggle PV Panel Defect Dataset.

Source: https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset

IMPORTANT: Asset IDs assigned here are DEMO ASSET identifiers for simulated
multi-mission histories. They do NOT represent real repeated observations
of the same physical solar panel.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.condition_mapping import PV_CLASSES, normalize_class_name

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_raw_dataset(base: Path) -> Path:
    """Locate downloaded Kaggle dataset root."""
    candidates = [
        base / "raw",
        base / "pv-panel-defect-dataset",
        Path.home() / ".cache" / "kaggle" / "pv-panel-defect-dataset",
    ]
    for c in candidates:
        if c.exists() and any(c.rglob("*.jpg")) or any(c.rglob("*.jpeg")) or any(c.rglob("*.png")):
            return c
    raise FileNotFoundError(
        "Kaggle PV dataset not found. Download with:\n"
        "  kaggle datasets download -d alicjalena/pv-panel-defect-dataset -p dataset/raw --unzip"
    )


def discover_class_dirs(raw_root: Path) -> Dict[str, Path]:
    """Detect class directories across train/val/test splits."""
    class_files: Dict[str, List[Path]] = {}

    for split in ["train", "val", "validation", "test"]:
        split_dir = raw_root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            canonical = normalize_class_name(class_dir.name)
            files = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
            class_files.setdefault(canonical, []).extend(files)

    if not class_files:
        # Flat layout: raw/ClassName/*.jpg
        for class_dir in sorted(raw_root.iterdir()):
            if class_dir.is_dir() and class_dir.name not in {"train", "val", "test", "validation"}:
                canonical = normalize_class_name(class_dir.name)
                files = [f for f in class_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
                if files:
                    class_files.setdefault(canonical, []).extend(files)

    return {k: v for k, v in class_files.items() if v}


def select_balanced_subset(
    class_files: Dict[str, List[Path]],
    max_per_class: int,
    seed: int,
) -> Dict[str, List[Path]]:
    rng = random.Random(seed)
    selected: Dict[str, List[Path]] = {}
    counts = [len(v) for v in class_files.values()]
    target = min(max_per_class, min(counts) if counts else 0)

    for cls, files in sorted(class_files.items()):
        unique = list({str(f): f for f in files}.values())
        rng.shuffle(unique)
        selected[cls] = sorted(unique[:target], key=lambda p: p.name)

    return selected


def assign_demo_assets(selected: Dict[str, List[Path]], seed: int) -> List[Dict]:
    """
    Build metadata rows with controlled DEMO ASSET assignments.

    Three primary demo assets with simulated multi-mission histories:
      DEMO_ASSET_A: Clean → Dusty → Physical-damage
      DEMO_ASSET_B: Clean → Clean → Bird-drop
      DEMO_ASSET_C: Clean → Electrical-damage

    Remaining images assigned to additional demo assets (one primary class each).
    """
    rng = random.Random(seed)
    rows: List[Dict] = []
    used: set[str] = set()

    def pop_image(cls: str) -> Optional[Path]:
        pool = [p for p in selected.get(cls, []) if str(p) not in used]
        if not pool:
            return None
        img = rng.choice(pool)
        used.add(str(img))
        return img

    # Primary simulated histories
    histories = [
        ("DEMO_ASSET_A", [
            ("mission_001", "Clean"),
            ("mission_002", "Dusty"),
            ("mission_003", "Physical-Damage"),
        ]),
        ("DEMO_ASSET_B", [
            ("mission_001", "Clean"),
            ("mission_002", "Clean"),
            ("mission_003", "Bird-drop"),
        ]),
        ("DEMO_ASSET_C", [
            ("mission_001", "Clean"),
            ("mission_002", "Electrical-damage"),
        ]),
    ]

    image_idx = 0
    for asset_id, missions in histories:
        for mission_id, cls in missions:
            img = pop_image(cls)
            if img is None:
                continue
            image_idx += 1
            rows.append({
                "image_id": f"img_{image_idx:04d}",
                "source_path": str(img),
                "class": cls,
                "split": "small",
                "asset_id": asset_id,
                "mission_id": mission_id,
                "demo_asset": "true",
                "simulated_history": "true",
                "note": "SIMULATED MULTI-MISSION HISTORY — not same physical panel",
            })

    # Additional demo assets — group remaining images by class
    extra_asset_num = 4
    for cls, files in sorted(selected.items()):
        for img in files:
            if str(img) in used:
                continue
            asset_id = f"DEMO_ASSET_{extra_asset_num:03d}"
            image_idx += 1
            rows.append({
                "image_id": f"img_{image_idx:04d}",
                "source_path": str(img),
                "class": cls,
                "split": "small",
                "asset_id": asset_id,
                "mission_id": "mission_001",
                "demo_asset": "true",
                "simulated_history": "false",
                "note": "DEMO ASSET — single observation from Kaggle class",
            })
            extra_asset_num += 1
            used.add(str(img))

    return rows


def copy_subset(rows: List[Dict], output_dir: Path) -> List[Dict]:
    """Copy selected images to dataset/small/{class}/ and update paths."""
    output_rows = []
    for row in rows:
        cls = row["class"]
        dest_dir = output_dir / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = Path(row["source_path"])
        dest_name = f"{row['image_id']}_{src.name}"
        dest = dest_dir / dest_name
        shutil.copy2(src, dest)
        output_rows.append({**row, "image_path": str(dest.relative_to(output_dir.parent))})
    return output_rows


def write_metadata(rows: List[Dict], path: Path) -> None:
    fieldnames = [
        "image_id", "image_path", "class", "split", "asset_id", "mission_id",
        "demo_asset", "simulated_history", "note",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_per_class: int = 60,
    seed: int = 42,
) -> Tuple[int, Path]:
    base = Path(__file__).resolve().parent.parent / "dataset"
    raw_root = raw_dir or find_raw_dataset(base)
    out = output_dir or (base / "small")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    class_files = discover_class_dirs(raw_root)
    if not class_files:
        raise RuntimeError(f"No class directories found under {raw_root}")

    selected = select_balanced_subset(class_files, max_per_class, seed)
    rows = assign_demo_assets(selected, seed)
    final_rows = copy_subset(rows, out)
    metadata_path = out / "metadata.csv"
    write_metadata(final_rows, metadata_path)

    print(f"Prepared {len(final_rows)} images across {len(selected)} classes")
    print(f"Output: {out}")
    print(f"Metadata: {metadata_path}")
    print("NOTE: asset_id values are DEMO ASSET identifiers for research demonstration.")
    return len(final_rows), metadata_path


def main():
    parser = argparse.ArgumentParser(description="Prepare PV panel defect dataset subset")
    parser.add_argument("--raw", type=str, default=None, help="Path to raw Kaggle dataset")
    parser.add_argument("--output", type=str, default=None, help="Output directory (dataset/small)")
    parser.add_argument("--max-per-class", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        raw_dir=Path(args.raw) if args.raw else None,
        output_dir=Path(args.output) if args.output else None,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
