"""Camera backends and device-name discovery per OS (UVC webcams via OpenCV)."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2

BACKENDS = {
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "ffmpeg": cv2.CAP_FFMPEG,
    "any": cv2.CAP_ANY,
}

DEVICE_LIST_TIMEOUT_S = 10


def default_backend() -> int:
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Windows":
        return cv2.CAP_DSHOW
    if system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def is_stream_or_file(source: Union[int, str]) -> bool:
    if isinstance(source, int):
        return False
    return bool(re.match(r"^[a-z][a-z0-9+.\-]*://", source, re.I)) or Path(source).exists()


def backend_for(name: str, source: Union[int, str]) -> int:
    if name != "auto":
        if name not in BACKENDS:
            raise ValueError(f"Unknown camera backend {name!r}; choose from auto, {', '.join(BACKENDS)}")
        return BACKENDS[name]
    if is_stream_or_file(source):
        return cv2.CAP_FFMPEG
    return default_backend()


def list_device_names() -> Dict[int, str]:
    """Index → human-readable name where the OS exposes it (best effort, no extra installs required)."""
    system = platform.system()
    if system == "Linux":
        names = {}
        for p in sorted(Path("/sys/class/video4linux").glob("video*")):
            m = re.match(r"video(\d+)$", p.name)
            name_file = p / "name"
            if m and name_file.exists():
                names[int(m.group(1))] = name_file.read_text(encoding="utf-8").strip()
        return names
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {}
    if system == "Darwin":
        cmd = [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    elif system == "Windows":
        cmd = [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    else:
        return {}
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=DEVICE_LIST_TIMEOUT_S).stderr
    except (OSError, subprocess.TimeoutExpired):
        return {}
    names: Dict[int, str] = {}
    if system == "Darwin":
        video = out.split("AVFoundation audio devices")[0]
        for m in re.finditer(r"\[(\d+)\]\s+(.+)", video):
            names[int(m.group(1))] = m.group(2).strip()
    else:
        idx = 0
        for m in re.finditer(r'"([^"]+)"\s+\(video\)', out):
            names[idx] = m.group(1)
            idx += 1
    return names


def raw_device_listing() -> Optional[str]:
    """The OS tool's own device listing (printed by camera_probe), or None if the tool is absent."""
    system = platform.system()
    if system == "Darwin" and shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    elif system == "Linux" and shutil.which("v4l2-ctl"):
        cmd = ["v4l2-ctl", "--list-devices"]
    elif system == "Windows" and shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    else:
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=DEVICE_LIST_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{' '.join(cmd)} failed: {exc}"
    return (r.stdout + r.stderr).strip()


def resolve_source(source: Union[int, str]) -> Union[int, str]:
    """int index | digit string | URL | file path | device name (matched case-insensitively)."""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    if is_stream_or_file(s):
        return s
    names = list_device_names()
    matches: List[int] = [i for i, n in names.items() if s.lower() in n.lower()]
    if len(matches) == 1:
        return matches[0]
    if not names:
        raise ValueError(f"Camera name {s!r} given but device names are unavailable on this OS; use the index from scripts/camera_probe.py")
    raise ValueError(f"Camera name {s!r} matched {len(matches)} devices: {names}")
