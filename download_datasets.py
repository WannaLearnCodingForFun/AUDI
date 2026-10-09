#!/usr/bin/env python3
"""
Download AUDI training datasets from Kaggle using the repo-root kaggle.json.

Usage:
    python scripts/download_datasets.py                 # all enabled datasets
    python scripts/download_datasets.py --include-optional
    python scripts/download_datasets.py --only neu_steel crack_concrete
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py --force         # re-download

Reads the `datasets:` block of config.yaml. Every dataset is independent:
a failure is logged and the run continues. At the end it writes
dataset/raw/INVENTORY.md and dataset/raw/inventory.json (folder tree with
image counts per directory) so prepare_unified_dataset.py can map classes
from the real on-disk layout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dataset" / "raw"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ── Kaggle credentials ─────────────────────────────────────────────────────

def setup_kaggle_credentials() -> None:
    """Point the Kaggle client at kaggle.json without copying or printing it."""
    repo_cred = ROOT / "kaggle.json"
    home_cred = Path.home() / ".kaggle" / "kaggle.json"

    if repo_cred.exists():
        try:
            repo_cred.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
        except OSError:
            pass  # Windows: chmod is best-effort
        os.environ["KAGGLE_CONFIG_DIR"] = str(ROOT)
        print(f"[auth] using {repo_cred.name} from repo root")
    elif os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("[auth] using KAGGLE_USERNAME / KAGGLE_KEY env vars")
    elif home_cred.exists():
        print("[auth] using ~/.kaggle/kaggle.json")
    else:
        sys.exit(
            "No Kaggle credentials found. Put kaggle.json in the repo root "
            "(it is git-ignored) or in ~/.kaggle/."
        )

    gi = ROOT / ".gitignore"
    if not gi.exists() or "kaggle.json" not in gi.read_text(encoding="utf-8"):
        print("[warn] kaggle.json is NOT in .gitignore — add it before committing!")


def get_api():
    # Import only after KAGGLE_CONFIG_DIR is set: the client authenticates on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


# ── Config ─────────────────────────────────────────────────────────────────

def load_datasets_config() -> list[dict]:
    import yaml

    cfg_path = Path(os.environ.get("AUDI_CONFIG", ROOT / "config.yaml"))
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    items = (cfg.get("datasets") or {}).get("sources") or []
    if not items:
        sys.exit(f"No datasets.sources block in {cfg_path}")
    for d in items:
        for key in ("name", "slug", "asset_type"):
            if key not in d:
                sys.exit(f"Dataset entry missing '{key}': {d}")
        d.setdefault("enabled", True)
        d.setdefault("optional", False)
    return items


# ── Download ───────────────────────────────────────────────────────────────

def count_images(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def unzip_all(folder: Path) -> None:
    """Extract any zip archives (including nested ones) and delete them."""
    while True:
        zips = list(folder.rglob("*.zip"))
        if not zips:
            return
        for z in zips:
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(z.parent)
                z.unlink()
            except zipfile.BadZipFile:
                print(f"  [warn] bad zip skipped: {z.relative_to(RAW_DIR)}")
                z.rename(z.with_suffix(".zip.bad"))


def download_one(api, ds: dict, force: bool, retries: int = 3) -> dict:
    target = RAW_DIR / ds["name"]
    marker = target / ".download_complete"
    result = {"name": ds["name"], "slug": ds["slug"], "status": "pending", "images": 0}

    if marker.exists() and not force:
        result.update(status="cached", images=count_images(target))
        print(f"[skip] {ds['name']}: already downloaded ({result['images']} images)")
        return result

    if force and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    # Verify the slug exists before downloading
    try:
        listing = api.dataset_list_files(ds["slug"])
        files = list(getattr(listing, "files", None) or listing or [])
        size_bytes = sum(getattr(f, "totalBytes", 0) or 0 for f in files)
        print(f"[ok]   {ds['slug']} exists ({len(files)} files, ~{size_bytes / 1e9:.2f} GB)")
    except Exception as exc:  # noqa: BLE001
        result.update(status="missing", error=str(exc))
        print(f"[fail] {ds['slug']}: cannot list files ({exc}) — skipping")
        return result

    for attempt in range(1, retries + 1):
        try:
            print(f"[get]  {ds['slug']} → dataset/raw/{ds['name']} (attempt {attempt})")
            api.dataset_download_files(ds["slug"], path=str(target), unzip=False, quiet=False)
            unzip_all(target)
            n = count_images(target)
            if n == 0:
                raise RuntimeError("download finished but no images were found")
            marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
            result.update(status="downloaded", images=n)
            print(f"[done] {ds['name']}: {n} images")
            return result
        except Exception as exc:  # noqa: BLE001
            result.update(status="error", error=str(exc))
            print(f"[err]  {ds['name']}: {exc}")
            time.sleep(3 * attempt)
    return result


# ── Inventory ──────────────────────────────────────────────────────────────

def inventory(folder: Path, max_depth: int = 4) -> list[dict]:
    """Every directory (to max_depth) that directly contains images, with counts."""
    rows = []
    for d in sorted(p for p in folder.rglob("*") if p.is_dir()):
        rel = d.relative_to(folder)
        if len(rel.parts) > max_depth:
            continue
        direct = sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if direct:
            rows.append({"dir": str(rel), "images": direct})
    annotations = [
        str(p.relative_to(folder))
        for p in folder.rglob("*")
        if p.suffix.lower() in {".csv", ".json", ".xml", ".txt"} and p.stat().st_size < 50_000_000
    ][:50]
    return [{"image_dirs": rows, "annotation_files": annotations}]


def write_inventory(datasets: list[dict], results: list[dict]) -> None:
    inv = {}
    lines = ["# Raw dataset inventory", "", "Generated by scripts/download_datasets.py", ""]
    by_name = {r["name"]: r for r in results}
    for ds in datasets:
        folder = RAW_DIR / ds["name"]
        res = by_name.get(ds["name"], {"status": "not requested"})
        lines.append(f"## {ds['name']}  (`{ds['slug']}`, asset_type={ds['asset_type']})")
        lines.append(f"Status: **{res['status']}**" + (f" — {res.get('error')}" if res.get("error") else ""))
        if folder.exists():
            data = inventory(folder)[0]
            inv[ds["name"]] = {**ds, **res, **data}
            lines.append("")
            lines.append("| folder | images |")
            lines.append("|---|---|")
            for row in data["image_dirs"][:80]:
                lines.append(f"| {row['dir']} | {row['images']} |")
            if data["annotation_files"]:
                lines.append("")
                lines.append("Annotation/metadata files: " + ", ".join(f"`{a}`" for a in data["annotation_files"]))
        lines.append("")
    (RAW_DIR / "inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
    (RAW_DIR / "INVENTORY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[inv]  wrote {RAW_DIR / 'INVENTORY.md'}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", help="dataset names to download")
    parser.add_argument("--include-optional", action="store_true", help="also download optional (large) datasets")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--list", action="store_true", help="list configured datasets and exit")
    args = parser.parse_args()

    datasets = load_datasets_config()

    if args.list:
        for d in datasets:
            flag = "optional" if d["optional"] else ("enabled" if d["enabled"] else "disabled")
            print(f"{d['name']:<28} {d['asset_type']:<13} {flag:<9} {d['slug']}")
        return

    if args.only:
        unknown = set(args.only) - {d["name"] for d in datasets}
        if unknown:
            sys.exit(f"Unknown dataset names: {sorted(unknown)}")
        selected = [d for d in datasets if d["name"] in args.only]
    else:
        selected = [
            d for d in datasets
            if d["enabled"] and (not d["optional"] or args.include_optional)
        ]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    setup_kaggle_credentials()
    api = get_api()

    results = [download_one(api, d, args.force) for d in selected]
    write_inventory(datasets, results)

    print("\nSummary")
    for r in results:
        print(f"  {r['name']:<28} {r['status']:<11} {r['images']:>7} images")
    failed = [r for r in results if r["status"] in {"missing", "error"}]
    if failed:
        print(f"\n{len(failed)} dataset(s) failed; the rest are usable. Re-run to retry.")
    if all(r["status"] in {"missing", "error"} for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
