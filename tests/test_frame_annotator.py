"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: test_frame_annotator.py
Description:
    Tests for FrameAnnotator.
"""

import numpy as np
import pytest

from padex.schemas.events import Shot, ShotOutcome, ShotType
from padex.schemas.tracking import (
    BallFrame,
    BallVisibility,
    BoundingBox,
    CourtCalibration,
    PlayerFrame,
    Position2D,
    Position3D,
)
from padex.viz.frame import FrameAnnotator


def _frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _make_pf(player_id="P_001", team_id="T_1"):
    return PlayerFrame(
        frame_id=0,
        timestamp_ms=0.0,
        player_id=player_id,
        team_id=team_id,
        bbox=BoundingBox(x1=100, y1=200, x2=200, y2=400),
        position=Position2D(x=5.0, y=10.0),
        confidence=0.9,
    )


def _make_bf():
    return BallFrame(
        frame_id=0,
        timestamp_ms=0.0,
        bbox=BoundingBox(x1=300, y1=300, x2=320, y2=320),
        position=Position3D(x=5.0, y=10.0, z=1.0),
        confidence=0.8,
        visibility=BallVisibility.VISIBLE,
    )


def _make_shot():
    return Shot(
        shot_id="S_001_01_01_001",
        timestamp_ms=0.0,
        player_id="P_001",
        team_id="T_1",
        position=Position2D(x=5.0, y=10.0),
        shot_type=ShotType.VOLLEY,
        outcome=ShotOutcome.NEXT_SHOT,
        confidence=0.8,
    )


def _make_calibration():
    # Identity homography (no perspective transform)
    return CourtCalibration(
        frame_width=1280,
        frame_height=720,
        homography_matrix=[
            [100.0, 0.0, 100.0],
            [0.0, 30.0, 50.0],
            [0.0, 0.0, 1.0],
        ],
        court_keypoints_px=[(100, 100), (500, 100), (100, 650), (500, 650)],
        court_keypoints_m=[(0, 0), (10, 0), (0, 20), (10, 20)],
        reprojection_error=1.5,
    )


class TestFrameAnnotator:
    def setup_method(self):
        self.annotator = FrameAnnotator()

    def test_draw_player_bboxes(self):
        frame = _frame()
        self.annotator.draw_player_bboxes(frame, [_make_pf()])
        # Bbox area should have non-zero pixels
        assert frame[200:400, 100:200, :].sum() > 0

    def test_draw_ball(self):
        frame = _frame()
        self.annotator.draw_ball(frame, _make_bf())
        # Ball area should have non-zero pixels
        assert frame[300:320, 300:320, :].sum() > 0

    def test_draw_ball_none(self):
        frame = _frame()
        self.annotator.draw_ball(frame, None)
        assert frame.sum() == 0

    def test_draw_ball_no_bbox(self):
        bf = BallFrame(
            frame_id=0,
            timestamp_ms=0.0,
            position=Position3D(x=5.0, y=10.0, z=0.0),
            confidence=0.8,
        )
        frame = _frame()
        self.annotator.draw_ball(frame, bf)
        assert frame.sum() == 0

    def test_draw_court_lines(self):
        frame = _frame()
        cal = _make_calibration()
        self.annotator.draw_court_lines(frame, cal)
        assert frame.sum() > 0

    def test_draw_court_lines_none(self):
        frame = _frame()
        self.annotator.draw_court_lines(frame, None)
        assert frame.sum() == 0

    def test_draw_court_keypoints(self):
        frame = _frame()
        cal = _make_calibration()
        self.annotator.draw_court_keypoints(frame, cal)
        assert frame.sum() > 0

    def test_draw_mini_court(self):
        frame = _frame()
        self.annotator.draw_mini_court(frame, [_make_pf()])
        assert frame.sum() > 0

    def test_draw_stats_panel(self):
        frame = _frame()
        stats = {"Shots": 10, "Speed": "45 km/h"}
        self.annotator.draw_stats_panel(frame, stats)
        assert frame[:80, :260, :].sum() > 0

    def test_draw_stats_panel_empty(self):
        frame = _frame()
        self.annotator.draw_stats_panel(frame, {})
        assert frame.sum() == 0

    def test_draw_shot_label(self):
        frame = _frame()
        self.annotator.draw_shot_label(frame, _make_shot(), _make_bf())
        assert frame.sum() > 0

    def test_draw_shot_label_none(self):
        frame = _frame()
        self.annotator.draw_shot_label(frame, None, None)
        assert frame.sum() == 0

    def test_draw_frame_number(self):
        frame = _frame()
        self.annotator.draw_frame_number(frame, 42)
        assert frame[:40, :200, :].sum() > 0

    def test_annotate_frame_all(self):
        frame = _frame()
        result = self.annotator.annotate_frame(
            frame,
            frame_id=0,
            player_frames=[_make_pf()],
            ball_frame=_make_bf(),
            calibration=_make_calibration(),
            shot=_make_shot(),
            stats={"Shots": 10},
        )
        assert result is frame
        assert frame.sum() > 0

    def test_annotate_frame_minimal(self):
        frame = _frame()
        result = self.annotator.annotate_frame(
            frame,
            frame_id=0,
            player_frames=[],
        )
        assert result is frame
        # At least frame number should be drawn
        assert frame.sum() > 0

    def test_multiple_players_different_teams(self):
        frame = _frame()
        pfs = [
            _make_pf("P_001", "T_1"),
            _make_pf("P_002", "T_2"),
        ]
        self.annotator.draw_player_bboxes(frame, pfs)
        assert frame.sum() > 0
