"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: tactics.py
Description:
    Pydantic models for Layer 3: Tactical analytics.
    Defines RallyMetrics, PlayerMetrics, TeamMetrics, MatchAnalytics.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RallyMetrics(BaseModel):
    """Metrics computed for a single rally/point."""

    point_id: str
    rally_length: int
    duration_ms: float
    net_approaches: int = 0
    wall_bounces: int = 0
    attack_defense_switches: int = 0


class PlayerMetrics(BaseModel):
    """Aggregated metrics for a single player."""

    player_id: str
    winners: int = 0
    errors: int = 0
    forced_errors: int = 0
    serve_percentage: float | None = None
    total_shots: int = 0
    shot_type_counts: dict[str, int] = Field(default_factory=dict)
    shot_type_success: dict[str, float] = Field(default_factory=dict)
    distance_covered_m: float | None = None
    avg_position_depth: float | None = None


class TeamMetrics(BaseModel):
    """Aggregated metrics for a team (pair)."""

    team_id: str
    net_control_pct: float | None = None
    formation_switches: int = 0
    avg_pair_distance: float | None = None
    wall_utilization_index: float | None = None
    transition_efficiency: float | None = None


class MatchAnalytics(BaseModel):
    """Top-level analytics for a match."""

    schema_version: str = "0.1.0"
    match_id: str
    rally_metrics: list[RallyMetrics] = Field(default_factory=list)
    player_metrics: list[PlayerMetrics] = Field(default_factory=list)
    team_metrics: list[TeamMetrics] = Field(default_factory=list)
