#!/usr/bin/env python3
"""Optional visualization of AUDI semantic memory state."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from memory.local_store import LocalSemanticMemoryStore


def visualize_text(store_path: Path) -> None:
    store = LocalSemanticMemoryStore(store_path=store_path)
    assets = store.list_assets()

    if not assets:
        print("No assets in semantic memory.")
        return

    print("\n" + "═" * 80)
    print("AUDI SEMANTIC MEMORY VISUALIZATION")
    print("═" * 80)

    for asset in assets:
        obs_list = asset.get("observations", [])
        last_ts = obs_list[-1]["observation_ts"] if obs_list else "N/A"
        last_obs = obs_list[-1] if obs_list else {}

        print(f"\n┌─ Asset: {asset['asset_id']}")
        print(f"│  Type:              {asset.get('asset_type', 'unknown')}")
        print(f"│  Current Condition: {asset.get('current_condition', 'unknown')}")
        print(f"│  Observations:      {len(obs_list)}")
        print(f"│  Last Inspected:    {last_ts[:19] if last_ts != 'N/A' else 'N/A'}")
        print(f"│  Revisit Priority:  {asset.get('revisit_priority_score', 0):.3f}")
        print(f"│  Change Magnitude:  {last_obs.get('change_magnitude', 0):.3f}")
        print(f"│  Severity:          {last_obs.get('severity_score', 0):.3f}")
        print(f"│  Match Confidence:  {last_obs.get('confidence', 0):.3f}")

        if obs_list:
            print("│")
            print("│  Condition Timeline:")
            for obs in obs_list:
                bar_len = int(obs.get("severity_score", 0) * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                ts = obs["observation_ts"][:10]
                print(
                    f"│    {ts} │ {obs['condition_description']:<15} │ "
                    f"S={obs.get('severity_score', 0):.2f} │ {bar}"
                )
        print("└" + "─" * 60)


def visualize_plot(store_path: Path, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; showing text visualization only.")
        visualize_text(store_path)
        return

    store = LocalSemanticMemoryStore(store_path=store_path)
    assets = store.list_assets()
    if not assets:
        print("No assets to visualize.")
        return

    fig, axes = plt.subplots(len(assets), 1, figsize=(10, 3 * len(assets)), squeeze=False)

    for idx, asset in enumerate(assets):
        ax = axes[idx][0]
        obs_list = asset.get("observations", [])
        if not obs_list:
            continue

        dates = [datetime.fromisoformat(o["observation_ts"].replace("Z", "+00:00")) for o in obs_list]
        severities = [o.get("severity_score", 0) for o in obs_list]
        changes = [o.get("change_magnitude", 0) for o in obs_list]

        ax.plot(dates, severities, "o-", label="Severity", color="red")
        ax.plot(dates, changes, "s--", label="Change Magnitude", color="blue")
        ax.set_title(f"{asset['asset_id']} ({asset.get('asset_type')}) — {asset.get('current_condition')}")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Plot saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Visualize AUDI semantic memory")
    parser.add_argument("--store", type=str, default="data/demo_semantic_memory.json")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plot")
    parser.add_argument("--output", type=str, default="memory_visualization.png")
    args = parser.parse_args()

    store_path = ROOT / args.store
    if not store_path.exists():
        print(f"Store not found: {store_path}. Run demo.py first.")
        return

    visualize_text(store_path)
    if args.plot:
        visualize_plot(store_path, ROOT / args.output)


if __name__ == "__main__":
    main()
