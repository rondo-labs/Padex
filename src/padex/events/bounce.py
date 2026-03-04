"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: bounce.py
Description:
    Bounce and wall interaction detection.
    Detects ball bounces via velocity direction reversal on smoothed
    trajectories, then classifies surface using court geometry rules.
"""

from __future__ import annotations

import abc
import logging

import numpy as np

from padex.schemas.events import Bounce, BounceType
from padex.schemas.tracking import (
    BallFrame,
    BallVisibility,
    CourtCalibration,
    Position2D,
    Position3D,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class BounceDetectionStrategy(abc.ABC):
    """Abstract interface for bounce detection backends."""

    @abc.abstractmethod
    def detect(self, ball_frames: list[BallFrame]) -> list[int]:
        """Identify frame indices where a bounce occurred."""
        ...


class SurfaceClassifier(abc.ABC):
    """Abstract interface for bounce surface classification."""

    @abc.abstractmethod
    def classify(
        self,
        position: Position2D | Position3D,
        velocity_before: np.ndarray,
        velocity_after: np.ndarray,
    ) -> BounceType:
        """Classify bounce surface from position and velocity change."""
        ...


# ---------------------------------------------------------------------------
# Velocity-based bounce detection
# ---------------------------------------------------------------------------


class VelocityBounceDetectionStrategy(BounceDetectionStrategy):
    """Detects bounces by finding velocity direction reversals.

    Algorithm:
    1. Extract positions from VISIBLE BallFrame entries
    2. Smooth with centered moving average
    3. Compute finite-difference velocities
    4. Find frames where velocity sign changes with sufficient magnitude
    5. Merge detections within min_separation_frames
    """

    def __init__(
        self,
        smooth_window: int = 5,
        velocity_threshold: float = 0.3,
        min_separation_frames: int = 8,
        min_confidence: float = 0.1,
    ) -> None:
        self.smooth_window = smooth_window
        self.velocity_threshold = velocity_threshold
        self.min_separation_frames = min_separation_frames
        self.min_confidence = min_confidence

    def detect(self, ball_frames: list[BallFrame]) -> list[int]:
        # 1. Extract visible frames with positions
        visible = [
            (i, bf)
            for i, bf in enumerate(ball_frames)
            if bf.visibility == BallVisibility.VISIBLE
            and bf.position is not None
            and bf.confidence >= self.min_confidence
        ]

        if len(visible) < self.smooth_window + 2:
            return []

        indices = [i for i, _ in visible]
        positions = np.array(
            [[bf.position.x, bf.position.y] for _, bf in visible]
        )

        # 2. Compute raw velocities
        velocities = np.diff(positions, axis=0)

        # 3. Find direction reversals using windowed velocity estimates
        #    For each candidate reversal at index k, compute average velocity
        #    in a window before and after k to avoid noise sensitivity.
        half = max(1, self.smooth_window // 2)
        bounce_indices: list[int] = []
        for k in range(half, len(velocities) - half):
            v_before = velocities[max(0, k - half) : k].mean(axis=0)
            v_after = velocities[k : k + half].mean(axis=0)

            y_reversal = (
                np.sign(v_before[1]) != np.sign(v_after[1])
                and abs(v_before[1]) > self.velocity_threshold
                and abs(v_after[1]) > self.velocity_threshold
            )
            x_reversal = (
                np.sign(v_before[0]) != np.sign(v_after[0])
                and abs(v_before[0]) > self.velocity_threshold
                and abs(v_after[0]) > self.velocity_threshold
            )

            if y_reversal or x_reversal:
                # Map back to original frame index
                bounce_indices.append(indices[k])

        # 5. Merge nearby
        return self._merge_nearby(bounce_indices, self.min_separation_frames)

    @staticmethod
    def _smooth(positions: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply centered moving average smoothing."""
        smoothed = np.copy(positions)
        half = window // 2
        for i in range(half, len(positions) - half):
            smoothed[i] = positions[i - half : i + half + 1].mean(axis=0)
        return smoothed

    @staticmethod
    def _merge_nearby(indices: list[int], min_sep: int) -> list[int]:
        """Merge bounce detections within min_sep frames."""
        if not indices:
            return []
        merged = [indices[0]]
        for idx in indices[1:]:
            if idx - merged[-1] >= min_sep:
                merged.append(idx)
        return merged


# ---------------------------------------------------------------------------
# Court geometry surface classifier
# ---------------------------------------------------------------------------


class CourtGeometrySurfaceClassifier(SurfaceClassifier):
    """Rule-based surface classification using padel court geometry.

    Court: x=0-10m (width), y=0-20m (length), net at y=10.0.
    """

    WALL_PROXIMITY_M: float = 0.8
    CORNER_PROXIMITY_M: float = 1.0
    NET_PROXIMITY_M: float = 0.5
    FENCE_HEIGHT_THRESHOLD: float = 1.0  # meters: above = glass, below = fence

    def classify(
        self,
        position: Position2D | Position3D,
        velocity_before: np.ndarray,
        velocity_after: np.ndarray,
    ) -> BounceType:
        x, y = position.x, position.y
        z = getattr(position, "z", 0.0) or 0.0

        near_net = abs(y - 10.0) < self.NET_PROXIMITY_M
        near_back_y0 = y < self.WALL_PROXIMITY_M
        near_back_y20 = y > 20.0 - self.WALL_PROXIMITY_M
        near_side_x0 = x < self.WALL_PROXIMITY_M
        near_side_x10 = x > 10.0 - self.WALL_PROXIMITY_M

        near_back = near_back_y0 or near_back_y20
        near_side = near_side_x0 or near_side_x10

        if near_net:
            return BounceType.NET
        if near_back and near_side:
            return BounceType.CORNER
        if near_back:
            if z > self.FENCE_HEIGHT_THRESHOLD:
                return BounceType.BACK_WALL
            return BounceType.BACK_FENCE
        if near_side:
            if z > self.FENCE_HEIGHT_THRESHOLD:
                return BounceType.SIDE_WALL
            return BounceType.SIDE_FENCE

        return BounceType.GROUND


# ---------------------------------------------------------------------------
# BounceDetector facade
# ---------------------------------------------------------------------------


class BounceDetector:
    """Main bounce detection facade.

    Combines velocity-based detection with geometry-based surface
    classification.
    """

    def __init__(
        self,
        detection_strategy: BounceDetectionStrategy | None = None,
        surface_classifier: SurfaceClassifier | None = None,
    ) -> None:
        self.detection_strategy = (
            detection_strategy or VelocityBounceDetectionStrategy()
        )
        self.surface_classifier = (
            surface_classifier or CourtGeometrySurfaceClassifier()
        )

    def detect_bounces(
        self,
        ball_frames: list[BallFrame],
        court_calibration: CourtCalibration | None = None,
    ) -> list[Bounce]:
        """Identify and classify all bounce events in a ball trajectory."""
        bounce_frame_indices = self.detection_strategy.detect(ball_frames)

        bounces: list[Bounce] = []
        for idx in bounce_frame_indices:
            bf = ball_frames[idx]
            if bf.position is None:
                continue

            v_before = self._velocity_at(ball_frames, idx, direction="before")
            v_after = self._velocity_at(ball_frames, idx, direction="after")

            bounce_type = self.surface_classifier.classify(
                bf.position, v_before, v_after
            )

            bounces.append(
                Bounce(
                    type=bounce_type,
                    position=Position2D(x=bf.position.x, y=bf.position.y),
                    timestamp_ms=bf.timestamp_ms,
                )
            )

        return bounces

    def classify_surface(
        self, bounce_position: Position2D | Position3D
    ) -> BounceType:
        """Classify surface for a single position (convenience method)."""
        zero_vel = np.zeros(2)
        return self.surface_classifier.classify(
            bounce_position, zero_vel, zero_vel
        )

    def _velocity_at(
        self,
        ball_frames: list[BallFrame],
        idx: int,
        direction: str,
        window: int = 3,
    ) -> np.ndarray:
        """Estimate velocity vector from surrounding frames."""
        if direction == "before":
            start = max(0, idx - window)
            segment = ball_frames[start:idx]
        else:
            end = min(len(ball_frames), idx + window + 1)
            segment = ball_frames[idx + 1 : end]

        positions = [
            (bf.position.x, bf.position.y)
            for bf in segment
            if bf.position is not None
            and bf.visibility != BallVisibility.INFERRED
        ]

        if len(positions) < 2:
            return np.zeros(2)

        pts = np.array(positions)
        return pts[-1] - pts[0]
