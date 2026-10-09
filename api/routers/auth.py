from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from auth.security import create_token, verify_password
from ml.config import cfg

from api.deps import any_user, get_services
from api.services import Services

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(), services: Services = Depends(get_services)) -> Dict[str, Any]:
    user = services.store.get_user(form.username)
    if user is None or not user["active"] or not verify_password(form.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password", headers={"WWW-Authenticate": "Bearer"})
    a = cfg(services.config, "auth")
    token = create_token(user["username"], user["role"], services.jwt_secret, a["algorithm"], int(a["access_token_minutes"]))
    user.pop("password_hash")
    return {"access_token": token, "token_type": "bearer", "user": user}


@router.get("/me")
def me(user: Dict[str, Any] = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    sites = next((u["sites"] for u in services.store.list_users() if u["user_id"] == user["user_id"]), [])
    return {**user, "sites": sites}
