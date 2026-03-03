"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: tracking.py
Description:
    Pydantic models for Layer 1: Tracking data.
    Defines Position2D/3D, PlayerFrame, BallFrame, CourtCalibration.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BallVisibility(str, Enum):
    """Ball visibility state in a given frame."""

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    INFERRED = "inferred"


class Position2D(BaseModel):
    """2D position on the court in meters. Origin at bottom-left corner."""

    x: float = Field(..., ge=0, le=10, description="Width axis (0–10m)")
    y: float = Field(..., ge=0, le=20, description="Length axis (0–20m)")


class Position3D(BaseModel):
    """3D position with optional height."""

    x: float = Field(..., ge=0, le=10)
    y: float = Field(..., ge=0, le=20)
    z: float = Field(0.0, ge=0, description="Height in meters")


class BoundingBox(BaseModel):
    """Pixel-space bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float


class PoseKeypoint(BaseModel):
    """Single keypoint from pose estimation."""

    name: str
    x: float
    y: float
    confidence: float = Field(..., ge=0, le=1)


class PlayerFrame(BaseModel):
    """Per-frame tracking data for a single player."""

    frame_id: int
    timestamp_ms: float
    player_id: str
    team_id: str | None = None
    bbox: BoundingBox
    position: Position2D | None = None
    confidence: float = Field(..., ge=0, le=1)
    keypoints: list[PoseKeypoint] = Field(default_factory=list)


class BallFrame(BaseModel):
    """Per-frame tracking data for the ball."""

    frame_id: int
    timestamp_ms: float
    bbox: BoundingBox | None = None
    position: Position3D | None = None
    confidence: float = Field(0.0, ge=0, le=1)
    visibility: BallVisibility = BallVisibility.VISIBLE


class CourtCalibration(BaseModel):
    """Homography calibration from pixel coordinates to court coordinates."""

    schema_version: str = "0.1.0"
    frame_width: int
    frame_height: int
    homography_matrix: list[list[float]] = Field(
        ..., description="3x3 homography matrix as nested list"
    )
    court_keypoints_px: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Detected court keypoints in pixel coordinates",
    )
    court_keypoints_m: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Corresponding court keypoints in meters",
    )
    reprojection_error: float | None = None
