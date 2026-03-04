"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: test_point_segmentation.py
Description:
    Tests for point/rally segmentation.
"""

import re

import pytest

from padex.events.point import (
    PauseBasedBoundaryStrategy,
    PointBoundary,
    PointSegmenter,
)
from padex.schemas.events import Shot, ShotOutcome, ShotType
from padex.schemas.tracking import BallFrame, BallVisibility, Position2D, Position3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shot(
    timestamp_ms: float,
    player_id: str = "P_001",
    team_id: str = "T_1",
    shot_type: ShotType = ShotType.UNKNOWN,
) -> Shot:
    return Shot(
        shot_id="S_001_01_01_001",  # will be renumbered
        timestamp_ms=timestamp_ms,
        player_id=player_id,
        team_id=team_id,
        position=Position2D(x=5.0, y=5.0),
        shot_type=shot_type,
        confidence=0.8,
    )


def _make_ball_frame(
    frame_id: int,
    timestamp_ms: float,
    visible: bool = True,
) -> BallFrame:
    return BallFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        position=Position3D(x=5.0, y=10.0, z=0.0) if visible else None,
        confidence=0.8 if visible else 0.0,
        visibility=BallVisibility.VISIBLE if visible else BallVisibility.OCCLUDED,
    )


# ---------------------------------------------------------------------------
# PauseBasedBoundaryStrategy tests
# ---------------------------------------------------------------------------


class TestPauseBasedBoundary:
    def setup_method(self):
        self.strategy = PauseBasedBoundaryStrategy(
            pause_threshold_ms=3000.0, min_rally_ms=500.0
        )

    def test_long_pause_creates_boundary(self):
        """Gap > 3000ms between visible frames → two points."""
        ball_frames = [
            _make_ball_frame(i, timestamp_ms=i * 33.3) for i in range(50)
        ]
        # Insert a big gap
        ball_frames += [
            _make_ball_frame(100 + i, timestamp_ms=5000.0 + i * 33.3)
            for i in range(50)
        ]
        boundaries = self.strategy.find_boundaries(ball_frames, [])
        assert len(boundaries) == 2

    def test_short_gap_no_boundary(self):
        """Gap < 3000ms → one continuous point."""
        ball_frames = [
            _make_ball_frame(i, timestamp_ms=i * 33.3) for i in range(100)
        ]
        boundaries = self.strategy.find_boundaries(ball_frames, [])
        assert len(boundaries) == 1

    def test_min_rally_length_enforced(self):
        """Very short rallies (< min_rally_ms) are filtered out."""
        ball_frames = [
            # Short segment: only 100ms
            _make_ball_frame(0, timestamp_ms=0.0),
            _make_ball_frame(1, timestamp_ms=100.0),
            # Gap
            _make_ball_frame(10, timestamp_ms=5000.0),
            # Normal segment: 1000ms
            _make_ball_frame(11, timestamp_ms=5033.3),
            _make_ball_frame(40, timestamp_ms=6000.0),
        ]
        boundaries = self.strategy.find_boundaries(ball_frames, [])
        # Only the longer segment should pass min_rally_ms filter
        assert len(boundaries) == 1
        assert boundaries[0].start_ms == 5000.0

    def test_no_visible_frames_returns_empty(self):
        ball_frames = [_make_ball_frame(0, 0.0, visible=False)]
        boundaries = self.strategy.find_boundaries(ball_frames, [])
        assert boundaries == []

    def test_empty_frames_returns_empty(self):
        assert self.strategy.find_boundaries([], []) == []


# ---------------------------------------------------------------------------
# PointSegmenter tests
# ---------------------------------------------------------------------------


class TestPointSegmenter:
    def test_single_point_from_shots(self):
        """Shots without ball_frames → one point."""
        shots = [
            _make_shot(1000.0, player_id="P_001", team_id="T_1"),
            _make_shot(2000.0, player_id="P_002", team_id="T_2"),
            _make_shot(3000.0, player_id="P_001", team_id="T_1"),
        ]
        segmenter = PointSegmenter()
        points = segmenter.segment(shots)
        assert len(points) == 1
        assert points[0].rally_length == 3

    def test_first_shot_marked_serve(self):
        """First UNKNOWN shot gets re-classified as SERVE."""
        shots = [
            _make_shot(1000.0),
            _make_shot(2000.0),
        ]
        segmenter = PointSegmenter()
        points = segmenter.segment(shots)
        assert points[0].shots[0].shot_type == ShotType.SERVE

    def test_already_serve_not_changed(self):
        """First shot already SERVE is not modified."""
        shots = [
            _make_shot(1000.0, shot_type=ShotType.SERVE),
            _make_shot(2000.0),
        ]
        segmenter = PointSegmenter()
        points = segmenter.segment(shots)
        assert points[0].shots[0].shot_type == ShotType.SERVE

    def test_outcomes_assigned(self):
        """Intermediate = NEXT_SHOT, last = WINNER."""
        shots = [_make_shot(t) for t in [1000, 2000, 3000]]
        segmenter = PointSegmenter()
        points = segmenter.segment(shots)
        p = points[0]
        assert p.shots[0].outcome == ShotOutcome.NEXT_SHOT
        assert p.shots[1].outcome == ShotOutcome.NEXT_SHOT
        assert p.shots[2].outcome == ShotOutcome.WINNER

    def test_rally_length_matches(self):
        shots = [_make_shot(t) for t in [1000, 2000, 3000, 4000]]
        segmenter = PointSegmenter()
        points = segmenter.segment(shots)
        assert points[0].rally_length == 4
        assert len(points[0].shots) == 4

    def test_duration_computed(self):
        shots = [_make_shot(1000.0), _make_shot(3000.0)]
        segmenter = PointSegmenter()
        points = segmenter.segment(shots)
        assert points[0].duration_ms == 2000.0

    def test_point_id_format(self):
        shots = [_make_shot(1000.0)]
        segmenter = PointSegmenter(set_num=2, game_num=5)
        points = segmenter.segment(shots)
        assert re.match(r"^S_\d{3}_\d{2}_\d{2}$", points[0].point_id)
        assert points[0].point_id == "S_002_05_01"

    def test_shot_ids_renumbered(self):
        shots = [_make_shot(1000.0), _make_shot(2000.0)]
        segmenter = PointSegmenter(set_num=1, game_num=1)
        points = segmenter.segment(shots)
        assert points[0].shots[0].shot_id == "S_001_01_01_001"
        assert points[0].shots[1].shot_id == "S_001_01_01_002"

    def test_winner_inferred_from_last_shot(self):
        shots = [
            _make_shot(1000.0, team_id="T_1"),
            _make_shot(2000.0, team_id="T_2"),
        ]
        segmenter = PointSegmenter()
        points = segmenter.segment(shots)
        assert points[0].winner_team_id == "T_2"

    def test_with_ball_frames_boundary(self):
        """Two points separated by a pause in ball frames."""
        ball_frames = [
            _make_ball_frame(i, i * 33.3) for i in range(30)
        ]
        ball_frames += [
            _make_ball_frame(200 + i, 5000.0 + i * 33.3)
            for i in range(30)
        ]

        shots = [
            _make_shot(100.0, team_id="T_1"),
            _make_shot(500.0, team_id="T_2"),
            _make_shot(5100.0, team_id="T_1"),
            _make_shot(5500.0, team_id="T_2"),
        ]

        segmenter = PointSegmenter()
        points = segmenter.segment(shots, ball_frames=ball_frames)
        assert len(points) == 2
        assert points[0].rally_length == 2
        assert points[1].rally_length == 2

    def test_empty_shots(self):
        segmenter = PointSegmenter()
        assert segmenter.segment([]) == []
