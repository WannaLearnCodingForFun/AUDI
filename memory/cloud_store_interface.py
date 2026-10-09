"""Cloud DB client interface and HTTP implementation (FR-1, FR-2, FR-6)."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import httpx

from ml.net import ensure_ca_bundle

logger = logging.getLogger(__name__)


class CloudUnavailable(Exception):
    """The cloud service could not be reached or rejected the request."""


class CloudStoreInterface(ABC):
    @abstractmethod
    def is_available(self, timeout: float) -> bool:
        """Connectivity probe (FR-1)."""

    @abstractmethod
    def pull_site(self, site_id: str, timeout: float) -> Dict[str, List[Dict[str, Any]]]:
        """Latest memory for one site, grouped by entity (FR-2)."""

    @abstractmethod
    def push_changes(self, changes: Sequence[Dict[str, Any]], timeout: float) -> List[str]:
        """Upload outbox records; returns the acknowledged change ids (FR-6/FR-7)."""


class HttpCloudStore(CloudStoreInterface):
    def __init__(self, base_url: str, username: Optional[str], password: Optional[str], transport: Optional[httpx.BaseTransport] = None):
        ensure_ca_bundle()
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self._transport = transport
        self._token: Optional[str] = None
        self._lock = threading.Lock()

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=timeout, transport=self._transport)

    def _auth_headers(self, client: httpx.Client) -> Dict[str, str]:
        with self._lock:
            if self._token is None:
                if not self.username or not self.password:
                    raise CloudUnavailable("Cloud credentials are not configured")
                r = client.post("/cloud/auth/token", data={"username": self.username, "password": self.password})
                if r.status_code != 200:
                    raise CloudUnavailable(f"Cloud authentication failed ({r.status_code})")
                self._token = r.json()["access_token"]
            return {"Authorization": f"Bearer {self._token}"}

    def _request(self, method: str, path: str, timeout: float, **kwargs) -> httpx.Response:
        try:
            with self._client(timeout) as client:
                r = client.request(method, path, headers=self._auth_headers(client), **kwargs)
                if r.status_code == 401:
                    with self._lock:
                        self._token = None
                    r = client.request(method, path, headers=self._auth_headers(client), **kwargs)
        except httpx.HTTPError as exc:
            raise CloudUnavailable(str(exc)) from exc
        if r.status_code >= 400:
            raise CloudUnavailable(f"{method} {path} → {r.status_code}: {r.text[:200]}")
        return r

    def is_available(self, timeout: float) -> bool:
        try:
            with self._client(timeout) as client:
                return client.get("/cloud/health").status_code == 200
        except httpx.HTTPError:
            return False

    def pull_site(self, site_id: str, timeout: float) -> Dict[str, List[Dict[str, Any]]]:
        return self._request("GET", f"/cloud/sites/{site_id}/pull", timeout).json()["records"]

    def push_changes(self, changes: Sequence[Dict[str, Any]], timeout: float) -> List[str]:
        body = {
            "changes": [
                {k: c[k] for k in ("change_id", "entity", "entity_id", "site_id", "payload")} for c in changes
            ]
        }
        return list(self._request("POST", "/cloud/push", timeout, json=body).json()["acked"])
