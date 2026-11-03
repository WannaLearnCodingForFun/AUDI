"""Encrypted image storage at rest (upgrade 6.5).

When `security.encrypt_images` is true every stored image file (crops, heatmaps, change masks)
is sealed with Fernet (AES-128-CBC + HMAC-SHA256). The key comes from the env var named by
`security.image_key_env`, else a generated mode-600 key file. Sealed files start with a magic
prefix so plaintext files written before encryption was enabled still read correctly.

Signed uploads: `sign_payload` / `verify_signature` implement HMAC-SHA256 over
"<timestamp>.<body>" for drone uploads (X-AUDI-Timestamp / X-AUDI-Signature headers), with a
replay window.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import stat
import time
from functools import lru_cache
from typing import Any, Dict, Optional

from ml.config import get_nested, resolve_path

MAGIC = b"AUDIENC1"


@lru_cache(maxsize=8)
def _fernet(key: bytes):
    from cryptography.fernet import Fernet

    return Fernet(key)


def load_image_key(config: Dict[str, Any]) -> bytes:
    sec = get_nested(config, "security", default={}) or {}
    env = os.environ.get(sec.get("image_key_env", "AUDI_IMAGE_KEY"))
    if env:
        return env.encode("ascii")
    path = resolve_path(sec.get("image_key_file", "data/.image_key"))
    if path.exists():
        return path.read_bytes().strip()
    from cryptography.fernet import Fernet

    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    path.write_bytes(key)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return key


def encryption_enabled(config: Dict[str, Any]) -> bool:
    return bool(get_nested(config, "security", "encrypt_images", default=False))


def maybe_encrypt(config: Dict[str, Any], data: bytes) -> bytes:
    if not encryption_enabled(config) or data.startswith(MAGIC):
        return data
    return MAGIC + _fernet(load_image_key(config)).encrypt(data)


def maybe_decrypt(config: Dict[str, Any], data: bytes) -> bytes:
    if not data.startswith(MAGIC):
        return data
    return _fernet(load_image_key(config)).decrypt(data[len(MAGIC):])


def sign_payload(secret: str, body: bytes, ts: Optional[int] = None) -> Dict[str, str]:
    ts = int(time.time()) if ts is None else int(ts)
    mac = hmac.new(secret.encode("utf-8"), f"{ts}.".encode("ascii") + body, hashlib.sha256).hexdigest()
    return {"X-AUDI-Timestamp": str(ts), "X-AUDI-Signature": mac}


def verify_signature(secret: str, body: bytes, ts_header: Optional[str], sig_header: Optional[str], window_s: float = 300.0) -> bool:
    if not ts_header or not sig_header:
        return False
    try:
        ts = int(ts_header)
    except ValueError:
        return False
    if abs(time.time() - ts) > window_s:
        return False
    expected = sign_payload(secret, body, ts)["X-AUDI-Signature"]
    return hmac.compare_digest(expected, sig_header)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
