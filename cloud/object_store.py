"""Image storage for the cloud service (upgrade 6.1).

`LocalObjectStore` keeps objects under a directory (dev / single host); `S3ObjectStore` talks
to any S3-compatible service (AWS S3, MinIO, Cloudflare R2, Ceph) through boto3 with an
optional endpoint URL. Objects are written with server-side encryption requested (SSE-S3)
and, when cloud.server.encrypt_images is set, sealed client-side with the image key too.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ml.config import get_nested, resolve_path


class ObjectStore:
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        raise NotImplementedError

    def get(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("object key escapes the store")
        return p

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def get(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        return p.read_bytes() if p.is_file() else None

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()


class S3ObjectStore(ObjectStore):
    def __init__(self, bucket: str, prefix: str = "", endpoint_url: Optional[str] = None, region: Optional[str] = None):
        import boto3  # optional dependency, only for cloud deployments

        self.s3 = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._key(key), Body=data, ContentType=content_type, ServerSideEncryption="AES256")

    def get(self, key: str) -> Optional[bytes]:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None


def build_object_store(config: Dict[str, Any]) -> ObjectStore:
    c = get_nested(config, "cloud", "server", "object_store", default={}) or {}
    if c.get("kind") == "s3":
        bucket = os.environ.get(c.get("bucket_env", "AUDI_S3_BUCKET")) or c.get("bucket")
        if not bucket:
            raise ValueError("cloud.server.object_store.kind = s3 needs a bucket")
        return S3ObjectStore(bucket, c.get("prefix", "audi"), os.environ.get(c.get("endpoint_env", "AUDI_S3_ENDPOINT")) or c.get("endpoint_url"),
                             os.environ.get(c.get("region_env", "AWS_REGION")) or c.get("region"))
    return LocalObjectStore(resolve_path(c.get("dir", "data/cloud_images")))
