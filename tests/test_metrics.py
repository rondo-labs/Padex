"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: test_metrics.py
Description:
    Tests for tactical metrics computation and heatmap generation.
"""

import math

import numpy as np
import pytest

from padex.schemas.events import (
    Bounce,
    BounceType,
    Point,
    Shot,
    ShotOutcome,
    ShotType,
)
from padex.schemas.tracking import PlayerFrame, BoundingBox, Position2D
from padex.tactics.heatmap import HeatmapGenerator
from padex.tactics.metrics import MetricsCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shot(
    shot_type: ShotType = ShotType.UNKNOWN,
    outcome: ShotOutcome | None = None,
    player_id: str = "P_001",
    team_id: str = "T_1",
    x: float = 5.0,
    y: float = 5.0,
    timestamp_ms: float = 0.0,
    trajectory: list[Bounce] | None = None,
) -> Shot:
    return Shot(
        shot_id="S_001_01_01_001",
        timestamp_ms=timestamp_ms,
        player_id=player_id,
        team_id=team_id,
        position=Position2D(x=x, y=y),
        shot_type=shot_type,
        outcome=outcome,
        confidence=0.8,
        trajectory=trajectory or [],
    )


def _make_point(
    shots: list[Shot],
    point_id: str = "S_001_01_01",
) -> Point:
    duration = (
        shots[-1].timestamp_ms - shots[0].timestamp_ms if len(shots) > 1 else 0.0
    )
    return Point(
        point_id=point_id,
        shots=shots,
        winner_team_id=shots[-1].team_id if shots else None,
        duration_ms=duration,
        rally_length=len(shots),
    )


def _make_player_frame(
    frame_id: int,
    player_id: str = "P_001",
    team_id: str = "T_1",
    x: float = 5.0,
    y: float = 10.0,
) -> PlayerFrame:
    return PlayerFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33.3,
        player_id=player_id,
        team_id=team_id,
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        position=Position2D(x=x, y=y),
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# RallyMetrics tests
# ---------------------------------------------------------------------------


class TestRallyMetrics:
    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_basic_rally_metrics(self):
        shots = [
            _make_shot(timestamp_ms=0),
            _make_shot(timestamp_ms=1000),
            _make_shot(timestamp_ms=2000),
        ]
        point = _make_point(shots)
        rm = self.calc.compute_rally_metrics(point)
        assert rm.rally_length == 3
        assert rm.duration_ms == 2000.0

    def test_wall_bounces_counted(self):
        wall_bounce = Bounce(
            type=BounceType.BACK_WALL,
            position=Position2D(x=5.0, y=1.0),
            timestamp_ms=500.0,
        )
        shots = [
            _make_shot(timestamp_ms=0, trajectory=[wall_bounce]),
            _make_shot(timestamp_ms=1000),
        ]
        point = _make_point(shots)
        rm = self.calc.compute_rally_metrics(point)
        assert rm.wall_bounces == 1

    def test_net_approaches_counted(self):
        """Shots near net (y >= 7) count as net approaches."""
        shots = [
            _make_shot(y=8.0),  # near net
            _make_shot(y=2.0),  # baseline
            _make_shot(y=12.0),  # near net
        ]
        point = _make_point(shots)
        rm = self.calc.compute_rally_metrics(point)
        assert rm.net_approaches >= 2

    def test_attack_defense_switches(self):
        shots = [
            _make_shot(shot_type=ShotType.SMASH),
            _make_shot(shot_type=ShotType.LOB),
            _make_shot(shot_type=ShotType.VOLLEY),
        ]
        point = _make_point(shots)
        rm = self.calc.compute_rally_metrics(point)
        assert rm.attack_defense_switches == 2

    def test_no_switches_same_category(self):
        shots = [
            _make_shot(shot_type=ShotType.SMASH),
            _make_shot(shot_type=ShotType.VOLLEY),
        ]
        point = _make_point(shots)
        rm = self.calc.compute_rally_metrics(point)
        assert rm.attack_defense_switches == 0


# ---------------------------------------------------------------------------
# PlayerMetrics tests
# ---------------------------------------------------------------------------


class TestPlayerMetrics:
    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_basic_counts(self):
        shots = [
            _make_shot(outcome=ShotOutcome.WINNER),
            _make_shot(outcome=ShotOutcome.ERROR),
            _make_shot(outcome=ShotOutcome.NEXT_SHOT),
        ]
        pm = self.calc.compute_player_metrics(shots, [], "P_001")
        assert pm.winners == 1
        assert pm.errors == 1
        assert pm.total_shots == 3

    def test_serve_percentage(self):
        shots = [
            _make_shot(shot_type=ShotType.SERVE),
            _make_shot(shot_type=ShotType.VOLLEY),
            _make_shot(shot_type=ShotType.SERVE),
            _make_shot(shot_type=ShotType.LOB),
        ]
        pm = self.calc.compute_player_metrics(shots, [], "P_001")
        assert pm.serve_percentage == pytest.approx(0.5)

    def test_shot_type_counts(self):
        shots = [
            _make_shot(shot_type=ShotType.VOLLEY),
            _make_shot(shot_type=ShotType.VOLLEY),
            _make_shot(shot_type=ShotType.LOB),
        ]
        pm = self.calc.compute_player_metrics(shots, [], "P_001")
        assert pm.shot_type_counts["volley"] == 2
        assert pm.shot_type_counts["lob"] == 1

    def test_distance_covered(self):
        pfs = [
            _make_player_frame(0, x=0.0, y=0.0),
            _make_player_frame(1, x=3.0, y=4.0),  # distance = 5.0
        ]
        pm = self.calc.compute_player_metrics([], pfs, "P_001")
        assert pm.distance_covered_m == pytest.approx(5.0)

    def test_avg_depth(self):
        pfs = [
            _make_player_frame(0, y=4.0),
            _make_player_frame(1, y=6.0),
            _make_player_frame(2, y=8.0),
        ]
        pm = self.calc.compute_player_metrics([], pfs, "P_001")
        assert pm.avg_position_depth == pytest.approx(6.0)

    def test_no_shots_returns_defaults(self):
        pm = self.calc.compute_player_metrics([], [], "P_001")
        assert pm.total_shots == 0
        assert pm.winners == 0
        assert pm.serve_percentage is None

    def test_filters_by_player_id(self):
        shots = [
            _make_shot(player_id="P_001", outcome=ShotOutcome.WINNER),
            _make_shot(player_id="P_002", outcome=ShotOutcome.WINNER),
        ]
        pm = self.calc.compute_player_metrics(shots, [], "P_001")
        assert pm.total_shots == 1
        assert pm.winners == 1


# ---------------------------------------------------------------------------
# TeamMetrics tests
# ---------------------------------------------------------------------------


class TestTeamMetrics:
    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_net_control(self):
        pfs = [
            _make_player_frame(0, player_id="P_001", team_id="T_1", y=10.0),
            _make_player_frame(0, player_id="P_002", team_id="T_1", y=5.0),
            _make_player_frame(1, player_id="P_001", team_id="T_1", y=2.0),
            _make_player_frame(1, player_id="P_002", team_id="T_1", y=3.0),
        ]
        tm = self.calc.compute_team_metrics([], pfs, "T_1")
        # Frame 0: P_001 at y=10 (in net zone 7-13) → net frame
        # Frame 1: both below 7 → not net frame
        assert tm.net_control_pct == pytest.approx(0.5)

    def test_avg_pair_distance(self):
        pfs = [
            _make_player_frame(0, player_id="P_001", team_id="T_1", x=2.0, y=5.0),
            _make_player_frame(0, player_id="P_002", team_id="T_1", x=8.0, y=5.0),
        ]
        tm = self.calc.compute_team_metrics([], pfs, "T_1")
        assert tm.avg_pair_distance == pytest.approx(6.0)

    def test_formation_switches(self):
        pfs = [
            # Frame 0: P_001 left (x=2), P_002 right (x=8)
            _make_player_frame(0, player_id="P_001", team_id="T_1", x=2.0, y=5.0),
            _make_player_frame(0, player_id="P_002", team_id="T_1", x=8.0, y=5.0),
            # Frame 1: swapped
            _make_player_frame(1, player_id="P_001", team_id="T_1", x=8.0, y=5.0),
            _make_player_frame(1, player_id="P_002", team_id="T_1", x=2.0, y=5.0),
            # Frame 2: swap back
            _make_player_frame(2, player_id="P_001", team_id="T_1", x=2.0, y=5.0),
            _make_player_frame(2, player_id="P_002", team_id="T_1", x=8.0, y=5.0),
        ]
        tm = self.calc.compute_team_metrics([], pfs, "T_1")
        assert tm.formation_switches == 2

    def test_wall_utilization(self):
        shots = [
            _make_shot(shot_type=ShotType.WALL_RETURN, player_id="P_001"),
            _make_shot(shot_type=ShotType.VOLLEY, player_id="P_001"),
        ]
        pfs = [_make_player_frame(0, player_id="P_001", team_id="T_1")]
        tm = self.calc.compute_team_metrics(shots, pfs, "T_1")
        assert tm.wall_utilization_index == pytest.approx(0.5)

    def test_transition_efficiency(self):
        shots = [
            _make_shot(
                shot_type=ShotType.SMASH,
                outcome=ShotOutcome.WINNER,
                player_id="P_001",
            ),
            _make_shot(
                shot_type=ShotType.VOLLEY,
                outcome=ShotOutcome.NEXT_SHOT,
                player_id="P_001",
            ),
        ]
        pfs = [_make_player_frame(0, player_id="P_001", team_id="T_1")]
        tm = self.calc.compute_team_metrics(shots, pfs, "T_1")
        assert tm.transition_efficiency == pytest.approx(0.5)

    def test_empty_team(self):
        tm = self.calc.compute_team_metrics([], [], "T_1")
        assert tm.net_control_pct is None
        assert tm.avg_pair_distance is None


# ---------------------------------------------------------------------------
# MatchAnalytics tests
# ---------------------------------------------------------------------------


class TestMatchAnalytics:
    def test_compute_match(self):
        calc = MetricsCalculator()
        shots = [
            _make_shot(
                player_id="P_001",
                team_id="T_1",
                outcome=ShotOutcome.NEXT_SHOT,
                timestamp_ms=0,
            ),
            _make_shot(
                player_id="P_002",
                team_id="T_2",
                outcome=ShotOutcome.WINNER,
                timestamp_ms=1000,
            ),
        ]
        point = _make_point(shots)
        pfs = [
            _make_player_frame(0, player_id="P_001", team_id="T_1"),
            _make_player_frame(0, player_id="P_002", team_id="T_2"),
        ]
        analytics = calc.compute_match_analytics([point], pfs, "M_001")
        assert analytics.match_id == "M_001"
        assert len(analytics.rally_metrics) == 1
        assert len(analytics.player_metrics) == 2
        assert len(analytics.team_metrics) == 2


# ---------------------------------------------------------------------------
# HeatmapGenerator tests
# ---------------------------------------------------------------------------


class TestHeatmapGenerator:
    def setup_method(self):
        self.gen = HeatmapGenerator(resolution=(20, 40), sigma=1.0)

    def test_empty_positions(self):
        hm = self.gen.generate([])
        assert hm.shape == (40, 20)
        assert hm.max() == 0.0

    def test_single_position(self):
        hm = self.gen.generate([Position2D(x=5.0, y=10.0)])
        assert hm.shape == (40, 20)
        assert hm.max() == pytest.approx(1.0)

    def test_multiple_positions_normalized(self):
        positions = [
            Position2D(x=5.0, y=10.0),
            Position2D(x=5.0, y=10.0),
            Position2D(x=2.0, y=5.0),
        ]
        hm = self.gen.generate(positions)
        assert hm.max() == pytest.approx(1.0)
        assert hm.min() >= 0.0

    def test_shot_heatmap(self):
        shots = [
            _make_shot(x=5.0, y=10.0),
            _make_shot(x=3.0, y=5.0),
        ]
        hm = self.gen.generate_shot_heatmap(shots)
        assert hm.shape == (40, 20)
        assert hm.max() > 0.0

    def test_bounce_heatmap(self):
        bounces = [
            Bounce(
                type=BounceType.GROUND,
                position=Position2D(x=5.0, y=10.0),
                timestamp_ms=100.0,
            ),
        ]
        hm = self.gen.generate_bounce_heatmap(bounces)
        assert hm.shape == (40, 20)
        assert hm.max() > 0.0

    def test_positions_at_edges(self):
        positions = [
            Position2D(x=0.0, y=0.0),
            Position2D(x=10.0, y=20.0),
        ]
        hm = self.gen.generate(positions)
        assert hm.max() == pytest.approx(1.0)
