#!/usr/bin/env python3
"""
Generate a small SYNTHETIC / DEMO dataset for pipeline testing.

WARNING: This is NOT real training data. Models evaluated on this data
are for integration testing only — not production accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSET_TYPES = ["solar_panel", "utility_pole", "building", "machinery"]
SPLITS = ["train", "validation", "test"]

DEMO_ASSETS = {
    "solar_panel_a": {"type": "solar_panel", "layout": "grid_blue"},
    "solar_panel_b": {"type": "solar_panel", "layout": "grid_teal"},
    "solar_panel_c": {"type": "solar_panel", "layout": "grid_navy"},
    "utility_pole_a": {"type": "utility_pole", "layout": "vertical_pole"},
    "building_a": {"type": "building", "layout": "house"},
    "machinery_a": {"type": "machinery", "layout": "gear"},
}


def _draw_layout(draw: ImageDraw.ImageDraw, layout: str, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cx, cy = size // 2, size // 2

    if layout == "grid_blue":
        for row in range(4):
            for col in range(6):
                x0, y0 = col * 36 + 4, row * 52 + 8
                draw.rectangle([x0, y0, x0 + 30, y0 + 44], fill=(30 + row * 10, 144, 255 - col * 8))
    elif layout == "grid_teal":
        for row in range(3):
            for col in range(5):
                x0, y0 = col * 42 + 10, row * 60 + 20
                draw.rectangle([x0, y0, x0 + 36, y0 + 50], fill=(0, 180 - row * 20, 160 + col * 10))
        draw.line([(0, cy), (size, cy)], fill=(255, 255, 0), width=3)
    elif layout == "grid_navy":
        for i in range(8):
            draw.rectangle([i * 28, 40, i * 28 + 22, size - 40], fill=(0, 50 + i * 15, 120))
        draw.polygon([(0, 0), (size, 0), (size, 30), (0, 50)], fill=(180, 180, 200))
    elif layout == "vertical_pole":
        draw.rectangle([cx - 12, 10, cx + 12, size - 10], fill=(139, 69, 19))
        for y in range(30, size - 30, 40):
            draw.line([(cx - 50, y), (cx + 50, y)], fill=(80, 80, 80), width=4)
        draw.ellipse([cx - 18, 8, cx + 18, 35], fill=(200, 200, 200))
    elif layout == "house":
        draw.rectangle([40, 90, size - 40, size - 20], fill=(160, 160, 160))
        draw.polygon([(30, 90), (cx, 30), (size - 30, 90)], fill=(100, 50, 50))
        draw.rectangle([cx - 25, 130, cx + 25, size - 20], fill=(80, 50, 20))
        for wx in [60, size - 80]:
            draw.rectangle([wx, 110, wx + 30, 140], fill=(135, 206, 250))
    elif layout == "gear":
        draw.ellipse([30, 30, size - 30, size - 30], fill=(255, 140, 0), outline=(0, 0, 0), width=4)
        draw.ellipse([70, 70, size - 70, size - 70], fill=(200, 100, 0))
        for angle in range(0, 360, 45):
            rad = np.deg2rad(angle)
            x = cx + int(80 * np.cos(rad))
            y = cy + int(80 * np.sin(rad))
            draw.rectangle([x - 10, y - 10, x + 10, y + 10], fill=(255, 140, 0))

    # Unique fingerprint dots per asset seed
    for _ in range(8):
        px, py = int(rng.integers(0, size)), int(rng.integers(0, size))
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(seed % 255, (seed * 5) % 255, (seed * 11) % 255))


def draw_asset(
    spec: dict,
    size: int = 224,
    damage: float = 0.0,
    seed: int = 0,
    asset_name: str = "",
) -> Image.Image:
    rng = np.random.default_rng(seed)
    bg_colors = {
        "grid_blue": (210, 225, 240),
        "grid_teal": (200, 240, 230),
        "grid_navy": (180, 200, 220),
        "vertical_pole": (180, 210, 180),
        "house": (220, 220, 200),
        "gear": (230, 210, 190),
    }
    layout = spec["layout"]
    img = Image.new("RGB", (size, size), color=bg_colors.get(layout, (200, 220, 200)))
    draw = ImageDraw.Draw(img)
    _draw_layout(draw, layout, size, seed)

    if asset_name:
        draw.rectangle([0, 0, size, 18], fill=(255, 255, 255))
        draw.text((4, 2), asset_name[:16], fill=(0, 0, 0))

    if damage > 0:
        for _ in range(int(damage * 12)):
            x, y = int(rng.integers(20, size - 20)), int(rng.integers(20, size - 20))
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 0, 0))

    return img


def generate_demo_dataset(output_dir: Path, images_per_asset: int = 3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset_type": "DEMO_SYNTHETIC", "warning": "NOT for production training", "assets": []}

    for split in SPLITS:
        for asset_name, spec in DEMO_ASSETS.items():
            asset_dir = output_dir / split / spec["type"]
            asset_dir.mkdir(parents=True, exist_ok=True)

            for i in range(images_per_asset):
                damage = 0.0 if i == 0 else 0.15 * i
                seed = hash(f"{asset_name}_{split}_{i}") % (2**31)
                img = draw_asset(spec, damage=damage, seed=seed, asset_name=asset_name)
                filename = f"{asset_name}_{split}_{i:02d}.png"
                img.save(asset_dir / filename)

                manifest["assets"].append(
                    {
                        "asset_id": asset_name,
                        "asset_type": spec["type"],
                        "split": split,
                        "path": str(asset_dir / filename),
                        "condition_damage": damage,
                    }
                )

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"DEMO dataset generated at {output_dir}")
    print(f"  Assets: {len(DEMO_ASSETS)} identities × {len(SPLITS)} splits")
    print("  WARNING: Synthetic data — not production accurate.")


def main():
    parser = argparse.ArgumentParser(description="Generate AUDI demo/synthetic dataset")
    parser.add_argument("--output", type=str, default="dataset", help="Output directory")
    parser.add_argument("--images-per-asset", type=int, default=3)
    args = parser.parse_args()
    generate_demo_dataset(Path(args.output), args.images_per_asset)


if __name__ == "__main__":
    main()
