"""
Project: Padex
File Created: 2026-03-05
Author: Xingnan Zhu
File Name: device.py
Description:
    Hardware device auto-detection for model inference.
    Tries GPU (CUDA / MPS) first, falls back to CPU.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_cached_device: str | None = None


def detect_device() -> str:
    """Auto-detect the best available device for inference.

    Priority: cuda > mps > cpu.
    Result is cached after first call.

    Returns:
        Device string compatible with ultralytics YOLO: "cuda", "mps", or "cpu".
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    # Try CUDA (NVIDIA GPU)
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("Using CUDA device: %s", device_name)
            _cached_device = "cuda"
            return _cached_device
    except ImportError:
        pass

    # Try MPS (Apple Silicon GPU)
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using Apple MPS (Metal Performance Shaders)")
            _cached_device = "mps"
            return _cached_device
    except ImportError:
        pass

    logger.info("No GPU detected, using CPU")
    _cached_device = "cpu"
    return _cached_device
