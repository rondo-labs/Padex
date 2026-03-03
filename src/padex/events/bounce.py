"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: bounce.py
Description:
    Bounce and wall interaction detection.
    Detects ball bounces on ground and wall surfaces.
"""

from __future__ import annotations


class BounceDetector:
    """Detects ball bounces on ground and wall surfaces."""

    def detect_bounces(self, ball_frames, court_calibration):
        """Identify bounce events from ball trajectory data."""
        raise NotImplementedError

    def classify_surface(self, bounce_position):
        """Classify which surface (ground, wall, fence) the ball hit."""
        raise NotImplementedError
