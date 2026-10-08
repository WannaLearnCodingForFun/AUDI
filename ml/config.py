"""Configuration loader for AUDI ML system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_config_cache: Optional[Dict[str, Any]] = None


def get_config_path() -> Path:
    """Return path to config.yaml, allowing override via AUDI_CONFIG env var."""
    override = os.environ.get("AUDI_CONFIG")
    if override:
        return Path(override)
    return _DEFAULT_CONFIG_PATH


def load_config(config_path: Optional[Path] = None, reload: bool = False) -> Dict[str, Any]:
    """Load and cache YAML configuration."""
    global _config_cache
    path = config_path or get_config_path()
    if _config_cache is not None and not reload and path == get_config_path():
        return _config_cache

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if path == get_config_path() and not reload:
        _config_cache = config
    return config


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely retrieve nested config values."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
