"""Shared pytest configuration for the homework companion."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def add_homework_to_path(homework: str) -> Path:
    """Prepend one homework root so its original absolute imports resolve."""
    root = PROJECT_ROOT / homework
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def requested_device():
    """Use HW_DEVICE (cpu by default). Raise an error if CUDA is requested but unavailable."""
    import torch

    name = os.environ.get("HW_DEVICE", "cpu")
    if name not in {"cpu", "cuda"}:
        raise ValueError(f"HW_DEVICE must be cpu or cuda, got {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(name)
