"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: test_schemas.py
Description:
    Basic tests for schema validation.
    Validates Pydantic models for tracking, events, and tactics.
"""

from padex.schemas.events import BounceType, Shot, ShotType
from padex.schemas.tracking import BallFrame, BallVisibility, PlayerFrame, Position2D


def test_position2d_valid():
    pos = Position2D(x=5.0, y=10.0)
    assert pos.x == 5.0
    assert pos.y == 10.0


def test_ball_visibility_enum():
    assert BallVisibility.VISIBLE == "visible"
    assert BallVisibility.OCCLUDED == "occluded"
    assert BallVisibility.INFERRED == "inferred"


def test_shot_type_taxonomy():
    assert ShotType.BANDEJA == "bandeja"
    assert ShotType.VIBORA == "vibora"
    assert ShotType.SMASH_X3 == "smash_x3"


def test_bounce_type_enum():
    assert BounceType.BACK_WALL == "back_wall"
    assert BounceType.SIDE_FENCE == "side_fence"


def test_player_frame():
    pf = PlayerFrame(
        frame_id=0,
        timestamp_ms=0.0,
        player_id="P1",
        bbox={"x1": 0, "y1": 0, "x2": 100, "y2": 200},
        confidence=0.95,
    )
    assert pf.player_id == "P1"


def test_ball_frame_defaults():
    bf = BallFrame(frame_id=0, timestamp_ms=0.0)
    assert bf.visibility == BallVisibility.VISIBLE
    assert bf.confidence == 0.0


def test_shot_id_format():
    shot = Shot(
        shot_id="S_001_01_01_001",
        timestamp_ms=1000.0,
        player_id="P1",
        team_id="T1",
        position=Position2D(x=5.0, y=15.0),
        shot_type=ShotType.SERVE,
    )
    assert shot.shot_id == "S_001_01_01_001"
