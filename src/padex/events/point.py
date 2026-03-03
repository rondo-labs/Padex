"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: point.py
Description:
    Point and rally segmentation.
    Groups shots into points based on serve detection and point endings.
"""

from __future__ import annotations


class PointSegmenter:
    """Segments a match into individual points/rallies."""

    def segment(self, shots):
        """Group shots into points based on serve detection and point endings."""
        raise NotImplementedError
