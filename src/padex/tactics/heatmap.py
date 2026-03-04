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

import numpy as np
from scipy.ndimage import gaussian_filter

from padex.schemas.events import Bounce, Shot
from padex.schemas.tracking import Position2D


class HeatmapGenerator:
    """Generates positional heatmaps from tracking data."""

    def __init__(
        self,
        court_dims: tuple[float, float] = (10.0, 20.0),
        resolution: tuple[int, int] = (50, 100),
        sigma: float = 1.5,
    ) -> None:
        self.court_width, self.court_length = court_dims
        self.res_x, self.res_y = resolution
        self.sigma = sigma

    def generate(
        self,
        positions: list[Position2D],
    ) -> np.ndarray:
        """Create a heatmap from a list of positions.

        Returns a 2D numpy array (res_y x res_x) with normalized values [0, 1].
        """
        heatmap = np.zeros((self.res_y, self.res_x), dtype=np.float64)

        for pos in positions:
            ix = int(pos.x / self.court_width * (self.res_x - 1))
            iy = int(pos.y / self.court_length * (self.res_y - 1))
            ix = max(0, min(self.res_x - 1, ix))
            iy = max(0, min(self.res_y - 1, iy))
            heatmap[iy, ix] += 1.0

        if heatmap.max() > 0:
            heatmap = gaussian_filter(heatmap, sigma=self.sigma)
            heatmap /= heatmap.max()

        return heatmap

    def generate_shot_heatmap(self, shots: list[Shot]) -> np.ndarray:
        """Generate heatmap from shot positions."""
        positions = [s.position for s in shots]
        return self.generate(positions)

    def generate_bounce_heatmap(self, bounces: list[Bounce]) -> np.ndarray:
        """Generate heatmap from bounce positions."""
        positions = [
            b.position
            for b in bounces
            if b.position is not None and isinstance(b.position, Position2D)
        ]
        return self.generate(positions)
