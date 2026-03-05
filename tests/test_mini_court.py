"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: test_mini_court.py
Description:
    Tests for MiniCourt overlay.
"""

import numpy as np
import pytest

from padex.schemas.tracking import (
    BallFrame,
    BallVisibility,
    BoundingBox,
    PlayerFrame,
    Position2D,
    Position3D,
)
from padex.viz.mini_court import MiniCourt


def _make_pf(player_id="P_001", team_id="T_1", x=5.0, y=10.0):
    return PlayerFrame(
        frame_id=0,
        timestamp_ms=0.0,
        player_id=player_id,
        team_id=team_id,
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        position=Position2D(x=x, y=y),
        confidence=0.9,
    )


def _make_bf(x=5.0, y=10.0, visibility=BallVisibility.VISIBLE):
    return BallFrame(
        frame_id=0,
        timestamp_ms=0.0,
        position=Position3D(x=x, y=y, z=0.0),
        confidence=0.8,
        visibility=visibility,
    )


class TestMiniCourt:
    def setup_method(self):
        self.mc = MiniCourt(width_px=150, height_px=300, margin=10)

    def test_draw_modifies_frame(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        original = frame.copy()
        pfs = [_make_pf()]
        self.mc.draw(frame, pfs)
        assert not np.array_equal(frame, original)

    def test_draw_with_ball(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        pfs = [_make_pf()]
        bf = _make_bf()
        self.mc.draw(frame, pfs, bf)
        # Should not raise and frame should be modified
        assert frame.sum() > 0

    def test_draw_no_players(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.mc.draw(frame, [])
        # Should still draw the court
        assert frame.sum() > 0

    def test_draw_occluded_ball_not_shown(self):
        frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)
        bf_vis = _make_bf(visibility=BallVisibility.VISIBLE)
        bf_occ = _make_bf(visibility=BallVisibility.OCCLUDED)
        self.mc.draw(frame1, [], bf_vis)
        self.mc.draw(frame2, [], bf_occ)
        # Visible ball frame should have more drawn pixels
        assert frame1.sum() > frame2.sum()

    def test_draw_multiple_players(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        pfs = [
            _make_pf("P_001", "T_1", 2.0, 5.0),
            _make_pf("P_002", "T_1", 8.0, 5.0),
            _make_pf("P_003", "T_2", 2.0, 15.0),
            _make_pf("P_004", "T_2", 8.0, 15.0),
        ]
        self.mc.draw(frame, pfs)
        assert frame.sum() > 0

    def test_top_right_position(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.mc.draw(frame, [_make_pf()], position="top_right")
        # Top-right corner should have drawn pixels
        top_right = frame[:320, 1100:, :]
        assert top_right.sum() > 0

    def test_court_to_mini_center(self):
        px, py = self.mc._court_to_mini(5.0, 10.0)
        assert px == 75  # center x
        assert py == 150  # center y

    def test_court_to_mini_origin(self):
        px, py = self.mc._court_to_mini(0.0, 0.0)
        assert px == 0
        assert py == 300  # bottom (y flipped)
