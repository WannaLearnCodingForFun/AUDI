#!/usr/bin/env python3
"""Write deploy/config.cloud.yaml = config.yaml + the cloud-mode overrides (docker-compose)."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def deep_merge(a, b):
    for k, v in b.items():
        a[k] = deep_merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


def main(domain: str = "localhost") -> None:
    base = yaml.safe_load((ROOT / "config.yaml").read_text())
    over = yaml.safe_load((ROOT / "deploy" / "config.cloud.yaml.example").read_text())
    over["api"]["cors_origins"] = [f"https://{domain}"]
    (ROOT / "deploy" / "config.cloud.yaml").write_text(yaml.safe_dump(deep_merge(base, over), sort_keys=False, allow_unicode=True))
    print(f"wrote deploy/config.cloud.yaml for https://{domain}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
