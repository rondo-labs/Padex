"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: shot.py
Description:
    Shot detection and classification from tracking data.
    Segments tracking data into individual shots and classifies shot type.
"""

from __future__ import annotations


class ShotDetector:
    """Segments tracking data into individual shots and classifies shot type."""

    def detect_shots(self, player_frames, ball_frames):
        """Identify moments of ball-player contact."""
        raise NotImplementedError

    def classify(self, shot, keypoints):
        """Classify shot type using pose keypoints."""
        raise NotImplementedError
