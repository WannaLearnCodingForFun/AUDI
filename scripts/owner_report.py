#!/usr/bin/env python3
"""Monthly owner report in one command (upgrade 6.4).

  python scripts/owner_report.py --site site_demo --month 2026-10
  → reports/owner/<site>_<month>.pdf and .json from the Local DB
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory.store import LocalStore  # noqa: E402
from ml.config import load_config, resolve_path  # noqa: E402
from ml.owner_report import build_owner_report, render_owner_pdf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--month", default=datetime.now(timezone.utc).strftime("%Y-%m"))
    ap.add_argument("--out", type=Path, default=resolve_path("reports/owner"))
    a = ap.parse_args()
    config = load_config()
    store = LocalStore(config)
    if store.get_site(a.site) is None:
        print(f"unknown site {a.site}")
        return 1
    rep = build_owner_report(store, a.site, a.month, config)
    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.out / f"{a.site}_{a.month}"
    stem.with_suffix(".json").write_text(json.dumps(rep, indent=2, default=str))
    stem.with_suffix(".pdf").write_bytes(render_owner_pdf(rep))
    h = rep["health"]
    print(f"{stem}.pdf · site health {h['site']} · {rep['activity']['missions']} missions · "
          f"deferral penalty {rep['currency']} {rep['deferral']['total_penalty']:,.2f} · energy loss {rep['energy']['kwh_lost']} kWh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
