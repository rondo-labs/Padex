"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: player.py
Description:
    Player detection and pose estimation using YOLO.
    Detects players in video frames and classifies teams via jersey color.
"""

from __future__ import annotations


class PlayerDetector:
    """Detects players in video frames and extracts pose keypoints."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path

    def detect(self, frame):
        """Detect players in a single frame. Returns list of PlayerFrame."""
        raise NotImplementedError

    def classify_teams(self, detections):
        """Classify detected players into teams via jersey color clustering."""
        raise NotImplementedError
