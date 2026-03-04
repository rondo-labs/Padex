"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: test_visualization.py
Description:
    Tests for Plotly visualization components.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from padex.schemas.events import Bounce, BounceType
from padex.schemas.tactics import (
    MatchAnalytics,
    PlayerMetrics,
    RallyMetrics,
    TeamMetrics,
)
from padex.schemas.tracking import (
    BallFrame,
    BallVisibility,
    BoundingBox,
    PlayerFrame,
    Position2D,
    Position3D,
)
from padex.viz.animation import RallyAnimator
from padex.viz.court import CourtRenderer
from padex.viz.dashboard import MatchDashboard


# ---------------------------------------------------------------------------
# CourtRenderer tests
# ---------------------------------------------------------------------------


class TestCourtRenderer:
    def setup_method(self):
        self.renderer = CourtRenderer()

    def test_draw_returns_figure(self):
        fig = self.renderer.draw()
        assert isinstance(fig, go.Figure)

    def test_draw_has_shapes(self):
        fig = self.renderer.draw()
        # Court rect + net + 2 service lines + 2 center lines = 6 shapes
        assert len(fig.layout.shapes) >= 5

    def test_plot_positions(self):
        positions = [Position2D(x=5.0, y=10.0), Position2D(x=3.0, y=5.0)]
        fig = self.renderer.plot_positions(positions)
        assert isinstance(fig, go.Figure)
        # Should have at least one scatter trace
        scatter_traces = [
            t for t in fig.data if isinstance(t, go.Scatter)
        ]
        assert len(scatter_traces) >= 1
        assert len(scatter_traces[-1].x) == 2

    def test_plot_positions_on_existing_fig(self):
        fig = self.renderer.draw()
        initial_traces = len(fig.data)
        fig = self.renderer.plot_positions(
            [Position2D(x=5.0, y=10.0)], fig=fig
        )
        assert len(fig.data) == initial_traces + 1

    def test_plot_heatmap(self):
        hm = np.random.rand(40, 20)
        fig = self.renderer.plot_heatmap(hm)
        assert isinstance(fig, go.Figure)
        heatmap_traces = [
            t for t in fig.data if isinstance(t, go.Heatmap)
        ]
        assert len(heatmap_traces) >= 1

    def test_plot_trajectory(self):
        bounces = [
            Bounce(
                type=BounceType.GROUND,
                position=Position2D(x=5.0, y=5.0),
                timestamp_ms=100.0,
            ),
            Bounce(
                type=BounceType.BACK_WALL,
                position=Position2D(x=5.0, y=1.0),
                timestamp_ms=200.0,
            ),
        ]
        fig = self.renderer.plot_trajectory(bounces)
        assert isinstance(fig, go.Figure)
        scatter_traces = [
            t for t in fig.data if isinstance(t, go.Scatter)
        ]
        assert len(scatter_traces) >= 1

    def test_plot_empty_trajectory(self):
        fig = self.renderer.plot_trajectory([])
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# RallyAnimator tests
# ---------------------------------------------------------------------------


def _make_pf(frame_id, player_id="P_001", team_id="T_1", x=5.0, y=5.0):
    return PlayerFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33.3,
        player_id=player_id,
        team_id=team_id,
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        position=Position2D(x=x, y=y),
        confidence=0.9,
    )


def _make_bf(frame_id, x=5.0, y=10.0):
    return BallFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33.3,
        position=Position3D(x=x, y=y, z=0.0),
        confidence=0.8,
        visibility=BallVisibility.VISIBLE,
    )


class TestRallyAnimator:
    def setup_method(self):
        self.animator = RallyAnimator()

    def test_animate_returns_figure(self):
        pfs = [_make_pf(i) for i in range(5)]
        bfs = [_make_bf(i) for i in range(5)]
        fig = self.animator.animate(pfs, bfs)
        assert isinstance(fig, go.Figure)

    def test_animate_has_frames(self):
        pfs = [_make_pf(i) for i in range(5)]
        bfs = [_make_bf(i) for i in range(5)]
        fig = self.animator.animate(pfs, bfs)
        assert len(fig.frames) == 5

    def test_animate_has_play_button(self):
        pfs = [_make_pf(0)]
        bfs = [_make_bf(0)]
        fig = self.animator.animate(pfs, bfs)
        assert fig.layout.updatemenus is not None
        assert len(fig.layout.updatemenus) >= 1

    def test_animate_empty_inputs(self):
        fig = self.animator.animate([], [])
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 0

    def test_animate_with_multiple_players(self):
        pfs = [
            _make_pf(0, player_id="P_001", team_id="T_1", x=3.0, y=5.0),
            _make_pf(0, player_id="P_002", team_id="T_1", x=7.0, y=5.0),
            _make_pf(0, player_id="P_003", team_id="T_2", x=3.0, y=15.0),
            _make_pf(0, player_id="P_004", team_id="T_2", x=7.0, y=15.0),
        ]
        bfs = [_make_bf(0)]
        fig = self.animator.animate(pfs, bfs)
        # First frame should have 4 player traces + 1 ball trace
        assert len(fig.frames[0].data) == 5


# ---------------------------------------------------------------------------
# MatchDashboard tests
# ---------------------------------------------------------------------------


class TestMatchDashboard:
    def setup_method(self):
        self.dashboard = MatchDashboard()

    def _make_analytics(self):
        return MatchAnalytics(
            match_id="M_TEST",
            rally_metrics=[
                RallyMetrics(
                    point_id="S_001_01_01",
                    rally_length=5,
                    duration_ms=3000.0,
                    wall_bounces=2,
                ),
                RallyMetrics(
                    point_id="S_001_01_02",
                    rally_length=3,
                    duration_ms=1500.0,
                ),
            ],
            player_metrics=[
                PlayerMetrics(
                    player_id="P_001",
                    winners=3,
                    errors=1,
                    total_shots=10,
                    shot_type_counts={"volley": 4, "smash": 2, "lob": 4},
                ),
                PlayerMetrics(
                    player_id="P_002",
                    winners=2,
                    errors=2,
                    total_shots=8,
                    shot_type_counts={"volley": 3, "groundstroke_fh": 5},
                ),
            ],
            team_metrics=[
                TeamMetrics(
                    team_id="T_1",
                    net_control_pct=0.6,
                ),
                TeamMetrics(
                    team_id="T_2",
                    net_control_pct=0.4,
                ),
            ],
        )

    def test_generate_returns_figure(self):
        analytics = self._make_analytics()
        fig = self.dashboard.generate(analytics)
        assert isinstance(fig, go.Figure)

    def test_generate_has_traces(self):
        analytics = self._make_analytics()
        fig = self.dashboard.generate(analytics)
        # Should have multiple traces (histogram, bars, etc.)
        assert len(fig.data) >= 4

    def test_generate_with_empty_analytics(self):
        analytics = MatchAnalytics(match_id="M_EMPTY")
        fig = self.dashboard.generate(analytics)
        assert isinstance(fig, go.Figure)

    def test_dashboard_title(self):
        analytics = self._make_analytics()
        fig = self.dashboard.generate(analytics)
        assert "M_TEST" in fig.layout.title.text
