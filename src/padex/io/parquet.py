"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: parquet.py
Description:
    Parquet read/write for tracking data.
    Handles serialization of player/ball frames to columnar format.
    Nested Pydantic models are flattened into scalar columns for efficiency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from padex.schemas.tracking import (
    BallFrame,
    BallVisibility,
    BoundingBox,
    PlayerFrame,
    Position2D,
    Position3D,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Player frames
# ---------------------------------------------------------------------------


def write_player_parquet(frames: list[PlayerFrame], path: str | Path) -> None:
    """Write player tracking frames to Parquet."""
    if not frames:
        logger.warning("No player frames to write")
        return

    rows: list[dict] = []
    for f in frames:
        row: dict = {
            "frame_id": f.frame_id,
            "timestamp_ms": f.timestamp_ms,
            "player_id": f.player_id,
            "team_id": f.team_id,
            "bbox_x1": f.bbox.x1,
            "bbox_y1": f.bbox.y1,
            "bbox_x2": f.bbox.x2,
            "bbox_y2": f.bbox.y2,
            "position_x": f.position.x if f.position else None,
            "position_y": f.position.y if f.position else None,
            "confidence": f.confidence,
        }
        rows.append(row)

    df = pl.DataFrame(rows)
    df.write_parquet(str(path))


def read_player_parquet(path: str | Path) -> list[PlayerFrame]:
    """Read player tracking frames from Parquet."""
    df = pl.read_parquet(str(path))
    frames: list[PlayerFrame] = []
    for row in df.iter_rows(named=True):
        position = None
        if row["position_x"] is not None and row["position_y"] is not None:
            position = Position2D(x=row["position_x"], y=row["position_y"])

        frames.append(
            PlayerFrame(
                frame_id=row["frame_id"],
                timestamp_ms=row["timestamp_ms"],
                player_id=row["player_id"],
                team_id=row["team_id"],
                bbox=BoundingBox(
                    x1=row["bbox_x1"],
                    y1=row["bbox_y1"],
                    x2=row["bbox_x2"],
                    y2=row["bbox_y2"],
                ),
                position=position,
                confidence=row["confidence"],
                keypoints=[],
            )
        )
    return frames


# ---------------------------------------------------------------------------
# Ball frames
# ---------------------------------------------------------------------------


def write_ball_parquet(frames: list[BallFrame], path: str | Path) -> None:
    """Write ball tracking frames to Parquet."""
    if not frames:
        logger.warning("No ball frames to write")
        return

    rows: list[dict] = []
    for f in frames:
        row: dict = {
            "frame_id": f.frame_id,
            "timestamp_ms": f.timestamp_ms,
            "bbox_x1": f.bbox.x1 if f.bbox else None,
            "bbox_y1": f.bbox.y1 if f.bbox else None,
            "bbox_x2": f.bbox.x2 if f.bbox else None,
            "bbox_y2": f.bbox.y2 if f.bbox else None,
            "position_x": f.position.x if f.position else None,
            "position_y": f.position.y if f.position else None,
            "position_z": f.position.z if f.position else None,
            "confidence": f.confidence,
            "visibility": f.visibility.value,
        }
        rows.append(row)

    df = pl.DataFrame(rows)
    df.write_parquet(str(path))


def read_ball_parquet(path: str | Path) -> list[BallFrame]:
    """Read ball tracking frames from Parquet."""
    df = pl.read_parquet(str(path))
    frames: list[BallFrame] = []
    for row in df.iter_rows(named=True):
        bbox = None
        if row["bbox_x1"] is not None:
            bbox = BoundingBox(
                x1=row["bbox_x1"],
                y1=row["bbox_y1"],
                x2=row["bbox_x2"],
                y2=row["bbox_y2"],
            )

        position = None
        if row["position_x"] is not None:
            position = Position3D(
                x=row["position_x"],
                y=row["position_y"],
                z=row["position_z"] or 0.0,
            )

        frames.append(
            BallFrame(
                frame_id=row["frame_id"],
                timestamp_ms=row["timestamp_ms"],
                bbox=bbox,
                position=position,
                confidence=row["confidence"],
                visibility=BallVisibility(row["visibility"]),
            )
        )
    return frames
