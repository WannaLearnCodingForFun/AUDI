"""Torch device selection: "auto" = CUDA → Apple MPS → CPU."""

from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
