"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: test_ball_detection.py
Description:
    Tests for ball detection, Kalman tracking, and BallDetector facade.
    Unit tests use synthetic data; integration tests use video + YOLO model.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from padex.schemas.tracking import BallVisibility, BoundingBox, Position3D
from padex.tracking.ball import (
    BallDetector,
    KalmanBallTracker,
    RawBallDetection,
)

VIDEO_PATH = Path("assets/raw/video/TapiaChingottoLebronGalanHighlights_1080p.mp4")
MODEL_PATH = Path("yolo26m.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_detection(
    cx=500.0, cy=300.0, size=20, conf=0.8, frame_id=0, timestamp_ms=0.0
) -> RawBallDetection:
    half = size / 2
    return RawBallDetection(
        bbox=BoundingBox(
            x1=cx - half, y1=cy - half, x2=cx + half, y2=cy + half
        ),
        confidence=conf,
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
    )


# ---------------------------------------------------------------------------
# Unit tests — no video or model needed
# ---------------------------------------------------------------------------


class TestRawBallDetection:
    def test_creation(self):
        det = _make_raw_detection()
        assert det.confidence == 0.8
        assert det.frame_id == 0

    def test_bbox_center(self):
        det = _make_raw_detection(cx=400, cy=200, size=20)
        assert det.bbox.x1 == 390
        assert det.bbox.x2 == 410


class TestKalmanBallTracker:
    def setup_method(self):
        # Identity homography: pixel coords == court coords
        self.H = np.eye(3)
        self.tracker = KalmanBallTracker(homography_matrix=self.H)

    def test_first_detection_initializes(self):
        det = _make_raw_detection(cx=5.0, cy=10.0, size=1)
        pos, vis = self.tracker.update(det, 0.0)
        assert vis == BallVisibility.VISIBLE
        assert pos is not None
        assert isinstance(pos, Position3D)

    def test_missing_before_init_returns_occluded(self):
        pos, vis = self.tracker.update(None, 0.0)
        assert vis == BallVisibility.OCCLUDED
        assert pos is None

    def test_occlusion_returns_inferred(self):
        # Initialize with a detection
        det = _make_raw_detection(cx=5.0, cy=10.0, size=1)
        self.tracker.update(det, 0.0)

        # Missing detection → INFERRED
        pos, vis = self.tracker.update(None, 33.3)
        assert vis == BallVisibility.INFERRED
        assert pos is not None

    def test_too_many_occlusions_resets(self):
        det = _make_raw_detection(cx=5.0, cy=10.0, size=1)
        self.tracker.update(det, 0.0)

        # Miss more than MAX_OCCLUDED_FRAMES
        for i in range(KalmanBallTracker.MAX_OCCLUDED_FRAMES + 1):
            pos, vis = self.tracker.update(None, (i + 1) * 33.3)

        assert vis == BallVisibility.OCCLUDED
        assert pos is None

    def test_ghost_gating_rejects_distant_detection(self):
        # Initialize at (5, 10)
        det1 = _make_raw_detection(cx=5.0, cy=10.0, size=1)
        self.tracker.update(det1, 0.0)

        # Detection far away (ghost from glass reflection)
        det2 = _make_raw_detection(cx=0.0, cy=0.0, size=1)
        pos, vis = self.tracker.update(det2, 33.3)
        # Should be rejected as ghost → INFERRED (using prediction)
        assert vis == BallVisibility.INFERRED

    def test_position_clamped_to_court(self):
        # Detection outside court bounds
        det = _make_raw_detection(cx=15.0, cy=25.0, size=1)
        pos, vis = self.tracker.update(det, 0.0)
        assert pos is not None
        assert 0 <= pos.x <= 10
        assert 0 <= pos.y <= 20

    def test_consistent_tracking(self):
        # Feed a sequence of detections moving linearly
        positions = []
        for i in range(10):
            x = 2.0 + i * 0.5
            y = 5.0 + i * 0.3
            det = _make_raw_detection(cx=x, cy=y, size=1)
            pos, vis = self.tracker.update(det, i * 33.3)
            assert vis == BallVisibility.VISIBLE
            positions.append((pos.x, pos.y))

        # Positions should be roughly monotonically increasing
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        assert xs[-1] > xs[0]
        assert ys[-1] > ys[0]

    def test_reset_clears_state(self):
        det = _make_raw_detection(cx=5.0, cy=10.0, size=1)
        self.tracker.update(det, 0.0)
        assert self.tracker._initialized

        self.tracker.reset()
        assert not self.tracker._initialized
        assert self.tracker._kf is None

    def test_no_homography_returns_none_position(self):
        tracker = KalmanBallTracker(homography_matrix=None)
        det = _make_raw_detection(cx=500.0, cy=300.0, size=20)
        pos, vis = tracker.update(det, 0.0)
        # Without homography, can't convert to court coords
        assert pos is None
        assert vis == BallVisibility.OCCLUDED

    def test_set_homography(self):
        tracker = KalmanBallTracker()
        assert tracker._H_matrix is None
        tracker.set_homography(np.eye(3))
        assert tracker._H_matrix is not None


class TestBallDetectorFacade:
    def test_detect_without_model(self):
        """BallDetector.detect with a mock strategy."""

        class MockStrategy:
            def detect(self, frame, frame_id, timestamp_ms):
                return _make_raw_detection(
                    cx=5.0, cy=10.0, frame_id=frame_id, timestamp_ms=timestamp_ms
                )

            def reset(self):
                pass

        detector = BallDetector(detection_strategy=MockStrategy())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame, frame_id=0, timestamp_ms=0.0)
        assert result.frame_id == 0
        assert result.bbox is not None
        assert result.visibility == BallVisibility.VISIBLE

    def test_detect_no_ball_found(self):
        class NullStrategy:
            def detect(self, frame, frame_id, timestamp_ms):
                return None

            def reset(self):
                pass

        detector = BallDetector(detection_strategy=NullStrategy())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame, frame_id=0, timestamp_ms=0.0)
        assert result.bbox is None
        assert result.visibility == BallVisibility.OCCLUDED
        assert result.confidence == 0.0

    def test_track_with_mock(self):
        """BallDetector.track with mock strategy and Kalman tracker."""
        call_count = 0

        class MovingBallStrategy:
            def detect(self, frame, frame_id, timestamp_ms):
                nonlocal call_count
                call_count += 1
                x = 2.0 + frame_id * 0.5
                y = 5.0 + frame_id * 0.3
                return _make_raw_detection(
                    cx=x, cy=y, frame_id=frame_id, timestamp_ms=timestamp_ms
                )

            def reset(self):
                pass

        H = np.eye(3)
        tracker = KalmanBallTracker(homography_matrix=H)
        detector = BallDetector(
            detection_strategy=MovingBallStrategy(), tracker=tracker
        )

        frames = [
            (i, i * 33.3, np.zeros((480, 640, 3), dtype=np.uint8))
            for i in range(10)
        ]
        results = detector.track(frames, homography_matrix=H)

        assert len(results) == 10
        assert all(r.visibility == BallVisibility.VISIBLE for r in results)
        assert all(r.position is not None for r in results)

    def test_reset(self):
        class DummyStrategy:
            reset_called = False

            def detect(self, frame, frame_id, timestamp_ms):
                return None

            def reset(self):
                self.reset_called = True

        strategy = DummyStrategy()
        tracker = KalmanBallTracker()
        detector = BallDetector(detection_strategy=strategy, tracker=tracker)
        detector.reset()
        assert strategy.reset_called


# ---------------------------------------------------------------------------
# Integration tests — require video + YOLO model + sahi
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not VIDEO_PATH.exists() or not MODEL_PATH.exists(),
    reason="Test video or YOLO model not available",
)
class TestWithVideoAndModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        from padex.io.video import VideoReader

        self.reader = VideoReader(VIDEO_PATH)
        self.detector = BallDetector(model_path=str(MODEL_PATH))
        yield
        self.reader.__exit__(None, None, None)

    def test_detect_single_frame(self):
        frame = self.reader.read_frame(3000)
        result = self.detector.detect(frame, frame_id=3000, timestamp_ms=100000.0)
        assert result.frame_id == 3000
        assert result.visibility in (BallVisibility.VISIBLE, BallVisibility.OCCLUDED)

    def test_track_produces_ball_frames(self):
        frames = []
        for fid, ts, frame in self.reader.frames(start_frame=3000, end_frame=3030):
            frames.append((fid, ts, frame))
        results = self.detector.track(frames)
        assert len(results) == len(frames)
        for bf in results:
            assert bf.visibility in (
                BallVisibility.VISIBLE,
                BallVisibility.OCCLUDED,
                BallVisibility.INFERRED,
            )
