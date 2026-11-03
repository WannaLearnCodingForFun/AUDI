"""Request dependencies: services, JWT authentication and role guards (§6.5, D14)."""

from __future__ import annotations

from typing import Any, Callable, Dict

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from auth.security import TokenError, decode_token
from ml.config import cfg

from api.services import Services

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

ROLE_OPERATOR = "operator"
ROLE_MAINTENANCE = "maintenance"
ROLE_ADMIN = "admin"


def get_services(request: Request) -> Services:
    return request.app.state.services


def user_from_token(services: Services, token: str) -> Dict[str, Any]:
    try:
        claims = decode_token(token, services.jwt_secret, cfg(services.config, "auth", "algorithm"))
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc
    user = services.store.get_user(claims["sub"])
    if user is None or not user["active"]:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User inactive or unknown", headers={"WWW-Authenticate": "Bearer"})
    user.pop("password_hash", None)
    return user


def current_user(token: str = Depends(oauth2_scheme), services: Services = Depends(get_services)) -> Dict[str, Any]:
    return user_from_token(services, token)


def require_roles(*roles: str) -> Callable[..., Dict[str, Any]]:
    """Allow the listed roles; admin is always allowed."""
    allowed = set(roles) | {ROLE_ADMIN}

    def guard(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
        if user["role"] not in allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires role: {', '.join(sorted(allowed))}")
        return user

    return guard


any_user = current_user
operator_only = require_roles(ROLE_OPERATOR)
maintenance_only = require_roles(ROLE_MAINTENANCE)
admin_only = require_roles(ROLE_ADMIN)
field_roles = require_roles(ROLE_OPERATOR, ROLE_MAINTENANCE)


# ── drones: scoped API keys (6.3 / 6.5) ───────────────────────────────────
def drone_from_request(request: Request, services: Services) -> Dict[str, Any]:
    key = request.headers.get("x-audi-key", "")
    if not key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-AUDI-Key")
    drone = services.store.drone_by_key(key)
    if drone is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown or revoked drone key")
    return drone


def require_drone_scope(scope: str) -> Callable[..., Dict[str, Any]]:
    """A registered drone whose key carries `scope`; the drone row is returned."""

    def guard(request: Request, services: Services = Depends(get_services)) -> Dict[str, Any]:
        drone = drone_from_request(request, services)
        if scope not in (drone.get("scopes") or []):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Drone key lacks scope {scope}")
        services.store.touch_drone(drone["drone_id"])
        return drone

    return guard


async def verify_signed_body(request: Request, services: Services, secret: str) -> bytes:
    """HMAC-SHA256 over '<ts>.<body>' (X-AUDI-Timestamp / X-AUDI-Signature) within the replay window."""
    from auth.image_crypto import verify_signature

    body = await request.body()
    window = float(cfg(services.config, "security", "signature_window_s"))
    if not verify_signature(secret, body, request.headers.get("x-audi-timestamp"), request.headers.get("x-audi-signature"), window):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Bad or expired upload signature")
    return body
