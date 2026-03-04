"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: point.py
Description:
    Point and rally segmentation.
    Groups shots into points using pause-based boundary detection,
    assigns serve labels, infers outcomes.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass

from padex.schemas.events import Point, Shot, ShotOutcome, ShotType
from padex.schemas.tracking import BallFrame, BallVisibility

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PointBoundary:
    """Time boundaries for a single point."""

    start_ms: float
    end_ms: float


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class PointBoundaryStrategy(abc.ABC):
    """Abstract interface for point boundary detection."""

    @abc.abstractmethod
    def find_boundaries(
        self,
        ball_frames: list[BallFrame],
        shots: list[Shot],
    ) -> list[PointBoundary]:
        """Identify start and end timestamps for each point."""
        ...


# ---------------------------------------------------------------------------
# Pause-based boundary detection
# ---------------------------------------------------------------------------


class PauseBasedBoundaryStrategy(PointBoundaryStrategy):
    """Detect point boundaries by finding gaps in ball activity.

    A gap longer than pause_threshold_ms between VISIBLE ball frames
    indicates a point boundary.
    """

    def __init__(
        self,
        pause_threshold_ms: float = 3000.0,
        min_rally_ms: float = 500.0,
    ) -> None:
        self.pause_threshold_ms = pause_threshold_ms
        self.min_rally_ms = min_rally_ms

    def find_boundaries(
        self,
        ball_frames: list[BallFrame],
        shots: list[Shot],
    ) -> list[PointBoundary]:
        if not ball_frames:
            return []

        visible_timestamps = [
            bf.timestamp_ms
            for bf in ball_frames
            if bf.visibility == BallVisibility.VISIBLE
        ]

        if not visible_timestamps:
            return []

        boundaries: list[PointBoundary] = []
        point_start = visible_timestamps[0]
        prev_ts = visible_timestamps[0]

        for ts in visible_timestamps[1:]:
            gap = ts - prev_ts
            if gap >= self.pause_threshold_ms:
                if prev_ts - point_start >= self.min_rally_ms:
                    boundaries.append(
                        PointBoundary(start_ms=point_start, end_ms=prev_ts)
                    )
                point_start = ts
            prev_ts = ts

        # Last point
        if prev_ts - point_start >= self.min_rally_ms:
            boundaries.append(
                PointBoundary(start_ms=point_start, end_ms=prev_ts)
            )

        return boundaries


# ---------------------------------------------------------------------------
# PointSegmenter facade
# ---------------------------------------------------------------------------


class PointSegmenter:
    """Main point segmentation facade.

    Responsibilities:
    1. Segment match into points using boundary detection
    2. Group shots into points
    3. Mark first shot as SERVE
    4. Assign shot outcomes (NEXT_SHOT / WINNER)
    5. Infer point winner
    """

    def __init__(
        self,
        boundary_strategy: PointBoundaryStrategy | None = None,
        set_num: int = 1,
        game_num: int = 1,
    ) -> None:
        self.boundary_strategy = (
            boundary_strategy or PauseBasedBoundaryStrategy()
        )
        self.set_num = set_num
        self.game_num = game_num

    def segment(
        self,
        shots: list[Shot],
        ball_frames: list[BallFrame] | None = None,
    ) -> list[Point]:
        """Group shots into points.

        Args:
            shots: Ordered list of Shot from ShotDetector.
            ball_frames: Optional, for pause-based boundary detection.

        Returns:
            list[Point] with shots, outcomes, and metadata.
        """
        if not shots:
            return []

        if ball_frames is not None:
            boundaries = self.boundary_strategy.find_boundaries(
                ball_frames, shots
            )
            if boundaries:
                return self._build_from_boundaries(shots, boundaries)

        # Fallback: treat all shots as one point
        return self._build_single_point(shots)

    def _build_from_boundaries(
        self, shots: list[Shot], boundaries: list[PointBoundary]
    ) -> list[Point]:
        points: list[Point] = []

        for point_idx, boundary in enumerate(boundaries, start=1):
            point_shots = [
                s
                for s in shots
                if boundary.start_ms <= s.timestamp_ms <= boundary.end_ms
            ]
            if not point_shots:
                continue

            renumbered = self._renumber_shots(point_shots, point_idx)
            renumbered = self._mark_serve(renumbered)
            renumbered = self._assign_outcomes(renumbered)

            point_id = (
                f"S_{self.set_num:03d}_{self.game_num:02d}_{point_idx:02d}"
            )
            points.append(
                Point(
                    point_id=point_id,
                    shots=renumbered,
                    winner_team_id=self._infer_winner(renumbered),
                    duration_ms=boundary.end_ms - boundary.start_ms,
                    rally_length=len(renumbered),
                )
            )

        return points

    def _build_single_point(self, shots: list[Shot]) -> list[Point]:
        renumbered = self._renumber_shots(shots, 1)
        renumbered = self._mark_serve(renumbered)
        renumbered = self._assign_outcomes(renumbered)

        point_id = f"S_{self.set_num:03d}_{self.game_num:02d}_01"

        duration = 0.0
        if len(renumbered) > 1:
            duration = renumbered[-1].timestamp_ms - renumbered[0].timestamp_ms

        return [
            Point(
                point_id=point_id,
                shots=renumbered,
                winner_team_id=self._infer_winner(renumbered),
                duration_ms=duration,
                rally_length=len(renumbered),
            )
        ]

    @staticmethod
    def _mark_serve(shots: list[Shot]) -> list[Shot]:
        """Mark first shot as SERVE if it's UNKNOWN."""
        if not shots:
            return shots
        if shots[0].shot_type == ShotType.UNKNOWN:
            shots[0] = shots[0].model_copy(
                update={"shot_type": ShotType.SERVE}
            )
        return shots

    @staticmethod
    def _assign_outcomes(shots: list[Shot]) -> list[Shot]:
        """Assign outcomes: intermediate = NEXT_SHOT, last = WINNER."""
        result = []
        for i, shot in enumerate(shots):
            if i < len(shots) - 1:
                result.append(
                    shot.model_copy(update={"outcome": ShotOutcome.NEXT_SHOT})
                )
            else:
                result.append(
                    shot.model_copy(update={"outcome": ShotOutcome.WINNER})
                )
        return result

    @staticmethod
    def _infer_winner(shots: list[Shot]) -> str | None:
        """Infer winner from last shot attribution."""
        if not shots:
            return None
        return shots[-1].team_id

    def _renumber_shots(
        self, shots: list[Shot], point_idx: int
    ) -> list[Shot]:
        """Rebuild shot IDs with correct point context."""
        result = []
        for shot_idx, shot in enumerate(shots, start=1):
            new_id = (
                f"S_{self.set_num:03d}_{self.game_num:02d}"
                f"_{point_idx:02d}_{shot_idx:03d}"
            )
            result.append(shot.model_copy(update={"shot_id": new_id}))
        return result
