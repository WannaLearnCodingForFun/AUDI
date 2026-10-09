#!/usr/bin/env python3
"""Create one local user per role (admin / operator / maintenance) and the cloud sync credentials.

Usernames and passwords come from the environment variables named in
`auth.seed_users` (config.yaml). When a password variable is unset, a random
password is generated and written to `auth.credentials_file` (git-ignored,
mode 600) — it is never printed. Operator and maintenance users are subscribed
to the default site so FR-19 escalations reach the maintenance user.

Usage: python scripts/seed_users.py [--reset-passwords]
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auth.credentials import get_credential, store_credentials  # noqa: E402
from auth.security import hash_password, verify_password  # noqa: E402
from memory.store import LocalStore  # noqa: E402
from ml.config import cfg, load_config, resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--reset-passwords", action="store_true", help="regenerate passwords not supplied via env vars")
    args = parser.parse_args()

    config = load_config(args.config)
    a = cfg(config, "auth")
    creds_file = a["credentials_file"]
    store = LocalStore(config)
    site = cfg(config, "sites")
    store.ensure_site(site["default_site_id"], site["default_site_name"])

    generated = {}
    rows = []
    for entry in a["seed_users"]:
        username = os.environ.get(entry["username_env"]) or get_credential(entry["username_env"], creds_file) or entry["default_username"]
        password = os.environ.get(entry["password_env"])
        source = "env"
        if not password:
            password = None if args.reset_passwords else get_credential(entry["password_env"], creds_file)
            source = "credentials file"
            if not password:
                password = secrets.token_urlsafe(int(a["generated_password_bytes"]))
                generated[entry["password_env"]] = password
                source = "generated → credentials file"
        generated.setdefault(entry["username_env"], username)

        if entry.get("cloud"):
            rows.append((username, "cloud sync", source, "used by the API to authenticate to cloud/ (seeded there at startup)"))
            continue
        user = store.get_user(username)
        if user is None:
            user = store.create_user(username, hash_password(password), entry["role"])
            action = "created"
        elif not verify_password(password, user["password_hash"]):
            store.update_user(user["user_id"], password_hash=hash_password(password), role=entry["role"], active=True)
            action = "password updated"
        else:
            action = "unchanged"
        if entry["role"] in {"operator", "maintenance"}:
            existing = next(u["sites"] for u in store.list_users() if u["user_id"] == user["user_id"])
            store.set_subscriptions(user["user_id"], sorted(set(existing) | {site["default_site_id"]}))
        rows.append((username, entry["role"], source, action))

    if generated:
        path = store_credentials(generated, creds_file)
    else:
        path = resolve_path(creds_file)
    print(f"{'username':<16} {'role':<12} {'password source':<30} result")
    for r in rows:
        print(f"{r[0]:<16} {r[1]:<12} {r[2]:<30} {r[3]}")
    print(f"\nGenerated passwords (if any) are in {path} (mode 600, git-ignored). Read them with: cat {path}")


if __name__ == "__main__":
    main()
