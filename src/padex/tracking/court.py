"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: court.py
Description:
    Court detection and homography calibration.
    Detects padel court lines and computes pixel-to-meter homography.
"""

from __future__ import annotations


class CourtDetector:
    """Detects padel court lines and computes pixel-to-meter homography."""

    def detect_keypoints(self, frame):
        """Detect court line intersections and glass wall edges."""
        raise NotImplementedError

    def compute_homography(self, keypoints_px, keypoints_m):
        """Compute homography matrix from matched keypoint pairs."""
        raise NotImplementedError

    def pixel_to_court(self, point_px, homography_matrix):
        """Transform pixel coordinates to court coordinates (meters)."""
        raise NotImplementedError
