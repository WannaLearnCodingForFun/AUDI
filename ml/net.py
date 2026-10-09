"""Network helpers shared by model/dataset downloaders."""

from __future__ import annotations

import os
import ssl


def ensure_ca_bundle() -> None:
    """Point OpenSSL at certifi's CA bundle when the interpreter has none.

    python.org macOS builds ship without a system CA path, which makes
    torch.hub / ultralytics / kaggle downloads fail certificate verification.
    """
    current = os.environ.get("SSL_CERT_FILE")
    if current and os.path.isfile(current):
        return
    if current:
        del os.environ["SSL_CERT_FILE"]
    if ssl.get_default_verify_paths().cafile:
        return
    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()
