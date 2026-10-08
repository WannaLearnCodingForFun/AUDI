#!/usr/bin/env python3
"""Prepare real dataset directory structure for future fine-tuning."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ASSET_TYPES = ["solar_panel", "utility_pole", "building", "machinery"]
SPLITS = ["train", "validation", "test"]


def prepare_dataset_structure(output_dir: Path) -> None:
    """Create empty folder hierarchy for real labeled data."""
    for split in SPLITS:
        for asset_type in ASSET_TYPES:
            (output_dir / split / asset_type).mkdir(parents=True, exist_ok=True)

    readme = output_dir / "README.md"
    if not readme.exists():
        shutil.copy(Path(__file__).resolve().parent.parent / "dataset" / "README.md", readme)

    print(f"Dataset structure ready at {output_dir}")
    print("Place REAL labeled images into train/validation/test subfolders.")
    print("Use scripts/generate_demo_dataset.py for synthetic testing data only.")


def main():
    parser = argparse.ArgumentParser(description="Prepare AUDI dataset directory structure")
    parser.add_argument("--output", type=str, default="dataset")
    args = parser.parse_args()
    prepare_dataset_structure(Path(args.output))


if __name__ == "__main__":
    main()
