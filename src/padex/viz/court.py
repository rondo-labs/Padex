"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: court.py
Description:
    Court rendering utilities.
    Renders a 2D padel court diagram with position overlays.
"""

from __future__ import annotations


class CourtRenderer:
    """Renders a 2D padel court diagram."""

    COURT_WIDTH = 10.0
    COURT_LENGTH = 20.0
    NET_Y = 10.0

    def draw(self):
        """Draw an empty court. Returns a matplotlib figure."""
        raise NotImplementedError

    def plot_positions(self, positions, **kwargs):
        """Overlay positions on the court."""
        raise NotImplementedError
