"""Password hashing and JWT tokens shared by the API and the cloud service (§6.5, D14)."""

from __future__ import annotations

import os
import secrets
import stat
from datetime import timedelta
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from ml.config import resolve_path
from ml.schemas import utc_now

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenError(Exception):
    """A JWT failed to decode (malformed, expired, bad signature) or lacks its subject/role claims."""


def hash_password(password: str) -> str:
    return _pwd.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _pwd.verify(password, password_hash)
    except ValueError:
        return False


def load_secret(env_name: str, file_path: str) -> str:
    """JWT signing secret: env var if set, else a generated mode-600 file."""
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    path = resolve_path(file_path)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_urlsafe(48)
    path.write_text(value, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return value


def create_token(subject: str, role: str, secret: str, algorithm: str, minutes: int, extra: Optional[Dict] = None) -> str:
    now = utc_now()
    claims = {"sub": subject, "role": role, "iat": int(now.timestamp()), "exp": int((now + timedelta(minutes=minutes)).timestamp())}
    if extra:
        claims.update(extra)
    return jwt.encode(claims, secret, algorithm=algorithm)


def decode_token(token: str, secret: str, algorithm: str) -> Dict[str, Any]:
    try:
        claims = jwt.decode(token, secret, algorithms=[algorithm])
    except JWTError as exc:
        raise TokenError(str(exc)) from exc
    if "sub" not in claims or "role" not in claims:
        raise TokenError("Token missing subject or role")
    return claims
