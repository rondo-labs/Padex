"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: heatmap.py
Description:
    Spatial analysis and heatmap generation.
    Generates positional heatmaps from tracking data.
"""

from __future__ import annotations


class HeatmapGenerator:
    """Generates positional heatmaps from tracking data."""

    def generate(self, positions, court_dims=(10.0, 20.0)):
        """Create a heatmap from a list of positions."""
        raise NotImplementedError
