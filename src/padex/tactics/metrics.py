"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: metrics.py
Description:
    Metric computation from event data.
    Computes rally-level, player-level, and team-level metrics.
"""

from __future__ import annotations


class MetricsCalculator:
    """Computes rally-level, player-level, and team-level metrics."""

    def compute_rally_metrics(self, points):
        """Compute metrics for each rally/point."""
        raise NotImplementedError

    def compute_player_metrics(self, shots, player_id: str):
        """Aggregate metrics for a single player."""
        raise NotImplementedError

    def compute_team_metrics(self, shots, team_id: str):
        """Aggregate metrics for a team."""
        raise NotImplementedError
