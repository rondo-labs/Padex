"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: test_bounce_detection.py
Description:
    Tests for bounce detection and surface classification.
"""

import numpy as np
import pytest

from padex.events.bounce import (
    BounceDetector,
    CourtGeometrySurfaceClassifier,
    VelocityBounceDetectionStrategy,
)
from padex.schemas.events import BounceType
from padex.schemas.tracking import BallFrame, BallVisibility, Position2D, Position3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ball_frame(
    frame_id: int,
    x: float,
    y: float,
    visibility: BallVisibility = BallVisibility.VISIBLE,
    confidence: float = 0.8,
) -> BallFrame:
    return BallFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33.3,
        position=Position3D(x=x, y=y, z=0.0),
        confidence=confidence,
        visibility=visibility,
    )


# ---------------------------------------------------------------------------
# VelocityBounceDetectionStrategy tests
# ---------------------------------------------------------------------------


class TestVelocityBounceDetection:
    def setup_method(self):
        self.strategy = VelocityBounceDetectionStrategy(
            smooth_window=3,
            velocity_threshold=0.2,
            min_separation_frames=3,
        )

    def test_y_reversal_detected(self):
        """Ball moving down then bouncing up → one bounce."""
        frames = []
        # Moving down: y decreasing
        for i in range(5):
            frames.append(_make_ball_frame(i, x=5.0, y=10.0 - i * 0.5))
        # Bounce: y starts increasing
        for i in range(5, 10):
            frames.append(_make_ball_frame(i, x=5.0, y=7.5 + (i - 5) * 0.5))

        bounces = self.strategy.detect(frames)
        assert len(bounces) >= 1

    def test_smooth_arc_no_bounce(self):
        """Monotonically increasing y → no bounce."""
        frames = [
            _make_ball_frame(i, x=5.0, y=2.0 + i * 0.5) for i in range(15)
        ]
        bounces = self.strategy.detect(frames)
        assert len(bounces) == 0

    def test_too_few_frames_returns_empty(self):
        frames = [_make_ball_frame(0, x=5.0, y=5.0)]
        bounces = self.strategy.detect(frames)
        assert bounces == []

    def test_occluded_frames_skipped(self):
        """OCCLUDED frames are not used for velocity computation."""
        frames = []
        for i in range(5):
            frames.append(_make_ball_frame(i, x=5.0, y=10.0 - i * 0.5))
        # Insert occluded frames
        for i in range(5, 8):
            frames.append(
                _make_ball_frame(
                    i, x=5.0, y=7.5, visibility=BallVisibility.OCCLUDED
                )
            )
        for i in range(8, 13):
            frames.append(_make_ball_frame(i, x=5.0, y=7.5 + (i - 8) * 0.5))

        # Should still detect the reversal from visible frames only
        bounces = self.strategy.detect(frames)
        assert len(bounces) >= 1

    def test_multiple_bounces_separated(self):
        """Two well-separated reversals → two bounces."""
        frames = []
        # First arc: down then up
        for i in range(6):
            frames.append(_make_ball_frame(i, x=5.0, y=10.0 - i * 0.5))
        for i in range(6, 12):
            frames.append(_make_ball_frame(i, x=5.0, y=7.0 + (i - 6) * 0.5))
        # Second arc: down then up
        for i in range(12, 18):
            frames.append(_make_ball_frame(i, x=5.0, y=10.0 - (i - 12) * 0.5))
        for i in range(18, 24):
            frames.append(_make_ball_frame(i, x=5.0, y=7.0 + (i - 18) * 0.5))

        bounces = self.strategy.detect(frames)
        assert len(bounces) >= 2

    def test_merge_nearby_detections(self):
        merged = VelocityBounceDetectionStrategy._merge_nearby(
            [10, 11, 12, 25, 26], min_sep=5
        )
        assert merged == [10, 25]

    def test_merge_empty(self):
        assert VelocityBounceDetectionStrategy._merge_nearby([], 5) == []


# ---------------------------------------------------------------------------
# CourtGeometrySurfaceClassifier tests
# ---------------------------------------------------------------------------


class TestSurfaceClassifier:
    def setup_method(self):
        self.classifier = CourtGeometrySurfaceClassifier()

    @pytest.mark.parametrize(
        "x,y,z,expected",
        [
            (5.0, 5.0, 0.0, BounceType.GROUND),
            (5.0, 10.0, 0.0, BounceType.NET),
            (5.0, 0.3, 0.0, BounceType.BACK_FENCE),
            (5.0, 19.8, 0.0, BounceType.BACK_FENCE),
            (5.0, 0.3, 2.0, BounceType.BACK_WALL),
            (0.3, 10.0, 0.0, BounceType.NET),  # net takes priority
            (0.3, 5.0, 0.0, BounceType.SIDE_FENCE),
            (9.8, 5.0, 2.0, BounceType.SIDE_WALL),
            (0.5, 0.5, 0.0, BounceType.CORNER),
        ],
    )
    def test_surface_classification(self, x, y, z, expected):
        pos = Position3D(x=x, y=y, z=z)
        zero_vel = np.zeros(2)
        result = self.classifier.classify(pos, zero_vel, zero_vel)
        assert result == expected

    def test_center_court_is_ground(self):
        pos = Position2D(x=5.0, y=10.5)
        zero_vel = np.zeros(2)
        result = self.classifier.classify(pos, zero_vel, zero_vel)
        assert result == BounceType.GROUND


# ---------------------------------------------------------------------------
# BounceDetector facade tests
# ---------------------------------------------------------------------------


class TestBounceDetectorFacade:
    def test_detect_bounces_with_reversal(self):
        detector = BounceDetector(
            detection_strategy=VelocityBounceDetectionStrategy(
                smooth_window=3,
                velocity_threshold=0.2,
                min_separation_frames=3,
            )
        )
        # Ball moving right then bouncing left (x reversal) in mid-court
        frames = []
        for i in range(6):
            frames.append(_make_ball_frame(i, x=3.0 + i * 0.5, y=5.0))
        for i in range(6, 12):
            frames.append(_make_ball_frame(i, x=6.0 - (i - 6) * 0.5, y=5.0))

        bounces = detector.detect_bounces(frames)
        assert len(bounces) >= 1
        assert bounces[0].type == BounceType.GROUND

    def test_classify_surface_convenience(self):
        detector = BounceDetector()
        result = detector.classify_surface(Position2D(x=5.0, y=5.0))
        assert result == BounceType.GROUND

    def test_empty_frames(self):
        detector = BounceDetector()
        bounces = detector.detect_bounces([])
        assert bounces == []
