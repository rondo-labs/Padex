"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: ball.py
Description:
    Ball detection and trajectory tracking.
    Detects the ball in video frames and tracks across multiple frames.
"""

from __future__ import annotations


class BallDetector:
    """Detects the ball in video frames and tracks its trajectory."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path

    def detect(self, frame):
        """Detect ball in a single frame. Returns BallFrame."""
        raise NotImplementedError

    def track(self, frames):
        """Track ball across multiple frames with interpolation."""
        raise NotImplementedError
