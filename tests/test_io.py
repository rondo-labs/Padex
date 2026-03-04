"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: test_io.py
Description:
    Tests for Parquet and JSONL I/O.
"""

from pathlib import Path

import pytest

from padex.io.jsonl import read_jsonl, write_jsonl
from padex.io.parquet import (
    read_ball_parquet,
    read_player_parquet,
    write_ball_parquet,
    write_player_parquet,
)
from padex.schemas.events import Bounce, BounceType, Shot, ShotOutcome, ShotType
from padex.schemas.tracking import (
    BallFrame,
    BallVisibility,
    BoundingBox,
    PlayerFrame,
    Position2D,
    Position3D,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_player_frame(frame_id=0, with_position=True) -> PlayerFrame:
    return PlayerFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33.33,
        player_id="P_001",
        team_id="T_1",
        bbox=BoundingBox(x1=100, y1=50, x2=200, y2=300),
        position=Position2D(x=5.0, y=10.0) if with_position else None,
        confidence=0.95,
        keypoints=[],
    )


def _make_ball_frame(frame_id=0, visible=True) -> BallFrame:
    return BallFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33.33,
        bbox=BoundingBox(x1=400, y1=200, x2=420, y2=220) if visible else None,
        position=Position3D(x=5.0, y=10.0, z=1.5) if visible else None,
        confidence=0.8 if visible else 0.0,
        visibility=BallVisibility.VISIBLE if visible else BallVisibility.OCCLUDED,
    )


# ---------------------------------------------------------------------------
# Parquet tests
# ---------------------------------------------------------------------------


class TestPlayerParquet:
    def test_roundtrip(self, tmp_path: Path):
        path = tmp_path / "players.parquet"
        frames = [_make_player_frame(i) for i in range(5)]
        write_player_parquet(frames, path)
        loaded = read_player_parquet(path)
        assert len(loaded) == 5
        assert loaded[0].player_id == "P_001"
        assert loaded[0].confidence == 0.95

    def test_roundtrip_without_position(self, tmp_path: Path):
        path = tmp_path / "players.parquet"
        frames = [_make_player_frame(0, with_position=False)]
        write_player_parquet(frames, path)
        loaded = read_player_parquet(path)
        assert loaded[0].position is None

    def test_roundtrip_preserves_bbox(self, tmp_path: Path):
        path = tmp_path / "players.parquet"
        frames = [_make_player_frame()]
        write_player_parquet(frames, path)
        loaded = read_player_parquet(path)
        assert loaded[0].bbox.x1 == 100
        assert loaded[0].bbox.y2 == 300

    def test_empty_list_no_file(self, tmp_path: Path):
        path = tmp_path / "players.parquet"
        write_player_parquet([], path)
        assert not path.exists()


class TestBallParquet:
    def test_roundtrip_visible(self, tmp_path: Path):
        path = tmp_path / "ball.parquet"
        frames = [_make_ball_frame(i) for i in range(3)]
        write_ball_parquet(frames, path)
        loaded = read_ball_parquet(path)
        assert len(loaded) == 3
        assert loaded[0].visibility == BallVisibility.VISIBLE
        assert loaded[0].position.z == 1.5

    def test_roundtrip_occluded(self, tmp_path: Path):
        path = tmp_path / "ball.parquet"
        frames = [_make_ball_frame(0, visible=False)]
        write_ball_parquet(frames, path)
        loaded = read_ball_parquet(path)
        assert loaded[0].bbox is None
        assert loaded[0].position is None
        assert loaded[0].visibility == BallVisibility.OCCLUDED

    def test_roundtrip_mixed(self, tmp_path: Path):
        path = tmp_path / "ball.parquet"
        frames = [_make_ball_frame(0, visible=True), _make_ball_frame(1, visible=False)]
        write_ball_parquet(frames, path)
        loaded = read_ball_parquet(path)
        assert loaded[0].visibility == BallVisibility.VISIBLE
        assert loaded[1].visibility == BallVisibility.OCCLUDED

    def test_empty_list_no_file(self, tmp_path: Path):
        path = tmp_path / "ball.parquet"
        write_ball_parquet([], path)
        assert not path.exists()


# ---------------------------------------------------------------------------
# JSONL tests
# ---------------------------------------------------------------------------


class TestJsonl:
    def test_roundtrip_shots(self, tmp_path: Path):
        path = tmp_path / "shots.jsonl"
        shots = [
            Shot(
                shot_id="S_001_01_01_001",
                timestamp_ms=1000.0,
                player_id="P_001",
                team_id="T_1",
                position=Position2D(x=5.0, y=3.0),
                shot_type=ShotType.SERVE,
                trajectory=[
                    Bounce(
                        type=BounceType.GROUND,
                        position=Position2D(x=7.0, y=12.0),
                        timestamp_ms=1200.0,
                    )
                ],
                outcome=ShotOutcome.NEXT_SHOT,
                confidence=0.9,
            )
        ]
        write_jsonl(shots, path)
        loaded = read_jsonl(path, Shot)
        assert len(loaded) == 1
        assert loaded[0].shot_id == "S_001_01_01_001"
        assert loaded[0].shot_type == ShotType.SERVE
        assert len(loaded[0].trajectory) == 1
        assert loaded[0].trajectory[0].type == BounceType.GROUND

    def test_roundtrip_ball_frames(self, tmp_path: Path):
        path = tmp_path / "ball.jsonl"
        frames = [_make_ball_frame(0), _make_ball_frame(1, visible=False)]
        write_jsonl(frames, path)
        loaded = read_jsonl(path, BallFrame)
        assert len(loaded) == 2
        assert loaded[0].visibility == BallVisibility.VISIBLE
        assert loaded[1].visibility == BallVisibility.OCCLUDED

    def test_read_without_model_returns_dicts(self, tmp_path: Path):
        path = tmp_path / "data.jsonl"
        frames = [_make_ball_frame()]
        write_jsonl(frames, path)
        loaded = read_jsonl(path)
        assert isinstance(loaded[0], dict)

    def test_empty_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"frame_id": 0}\n\n{"frame_id": 1}\n')
        loaded = read_jsonl(path)
        assert len(loaded) == 2

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "sub" / "dir" / "data.jsonl"
        write_jsonl([_make_ball_frame()], path)
        assert path.exists()
