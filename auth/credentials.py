"""Credential lookup: environment first, then the git-ignored credentials file.

scripts/seed_users.py writes generated passwords to `auth.credentials_file`
(mode 600) when the corresponding environment variables are not set.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Dict, Optional

from ml.config import resolve_path


def _parse(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def get_credential(name: str, credentials_file: str) -> Optional[str]:
    return os.environ.get(name) or _parse(resolve_path(credentials_file)).get(name)


def store_credentials(values: Dict[str, str], credentials_file: str) -> Path:
    path = resolve_path(credentials_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = _parse(path)
    merged.update(values)
    path.write_text("".join(f"{k}={v}\n" for k, v in merged.items()), encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return path
