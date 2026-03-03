"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: test_court_detection.py
Description:
    Tests for court detection and homography calibration.
    Unit tests use synthetic data; integration tests use the highlights video.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from padex.tracking.court import (
    COURT_MODEL,
    CourtDetector,
    HoughLineKeypointDetector,
    KeypointDetectionResult,
    PadelCourtModel,
)

VIDEO_PATH = Path("assets/raw/video/TapiaChingottoLebronGalanHighlights_1080p.mp4")


# ---------------------------------------------------------------------------
# Unit tests — no video needed
# ---------------------------------------------------------------------------


class TestPadelCourtModel:
    def test_keypoint_count(self):
        assert len(COURT_MODEL.KEYPOINTS) == 12

    def test_keypoints_within_bounds(self):
        for name, (x, y) in COURT_MODEL.KEYPOINTS.items():
            assert 0 <= x <= 10, f"{name} x={x} out of bounds"
            assert 0 <= y <= 20, f"{name} y={y} out of bounds"

    def test_line_count(self):
        assert len(COURT_MODEL.LINES) == 8

    def test_lines_reference_valid_keypoints(self):
        for a, b in COURT_MODEL.LINES:
            assert a in COURT_MODEL.KEYPOINTS, f"Unknown keypoint: {a}"
            assert b in COURT_MODEL.KEYPOINTS, f"Unknown keypoint: {b}"

    def test_court_dimensions(self):
        assert COURT_MODEL.COURT_WIDTH == 10.0
        assert COURT_MODEL.COURT_LENGTH == 20.0
        assert COURT_MODEL.NET_Y == 10.0


class TestHomographySynthetic:
    def setup_method(self):
        self.detector = CourtDetector()

    def test_identity_homography(self):
        """When pixel coords == meter coords, H should be ~identity."""
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)]
        H, error = self.detector.compute_homography(points, points)
        np.testing.assert_allclose(H, np.eye(3), atol=1e-6)
        assert error < 1e-6

    def test_known_scale_transform(self):
        """Pixel coords are 100x the meter coords (simple scaling)."""
        px = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 2000.0), (0.0, 2000.0)]
        m = [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)]
        H, error = self.detector.compute_homography(px, m)
        assert error < 0.01
        # Check that (500, 1000) maps to (5, 10) — center of court
        result = self.detector.pixel_to_court((500.0, 1000.0), H)
        np.testing.assert_allclose(result, (5.0, 10.0), atol=0.1)

    def test_pixel_to_court_roundtrip(self):
        """pixel -> court -> pixel should be identity."""
        px = [(100.0, 50.0), (900.0, 50.0), (800.0, 500.0), (200.0, 500.0)]
        m = [(0.0, 20.0), (10.0, 20.0), (10.0, 0.0), (0.0, 0.0)]
        H, _ = self.detector.compute_homography(px, m)

        original = (500.0, 300.0)
        court = self.detector.pixel_to_court(original, H)
        back = self.detector.court_to_pixel(court, H)
        np.testing.assert_allclose(back, original, atol=0.5)

    def test_compute_homography_fails_with_bad_points(self):
        """Collinear points should fail."""
        px = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        m = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        with pytest.raises(ValueError):
            self.detector.compute_homography(px, m)


class TestValidateHomography:
    def test_rejects_zero_matrix(self):
        H = np.zeros((3, 3))
        assert not CourtDetector._validate_homography(H, 1920, 1080)

    def test_identity_with_large_frame(self):
        # Identity maps court corners (0-10, 0-20) to pixel (0-10, 0-20)
        # which is within a 1920x1080 frame — should pass
        H = np.eye(3)
        assert CourtDetector._validate_homography(H, 1920, 1080)

    def test_accepts_valid_scaling(self):
        # Scale 1920x1080 -> 10x20 court
        H = np.array([
            [10.0 / 1920, 0, 0],
            [0, 20.0 / 1080, 0],
            [0, 0, 1],
        ])
        assert CourtDetector._validate_homography(H, 1920, 1080)


class TestHoughDetectorInternals:
    def setup_method(self):
        self.det = HoughLineKeypointDetector()

    def test_line_intersection_perpendicular(self):
        h_line = np.array([0, 100, 200, 100], dtype=np.float64)
        v_line = np.array([100, 0, 100, 200], dtype=np.float64)
        pt = HoughLineKeypointDetector._line_intersection(h_line, v_line)
        assert pt is not None
        np.testing.assert_allclose(pt, (100.0, 100.0), atol=0.1)

    def test_line_intersection_parallel_returns_none(self):
        l1 = np.array([0, 100, 200, 100], dtype=np.float64)
        l2 = np.array([0, 200, 200, 200], dtype=np.float64)
        assert HoughLineKeypointDetector._line_intersection(l1, l2) is None

    def test_merge_nearby_points(self):
        points = [(100.0, 100.0), (102.0, 101.0), (300.0, 300.0)]
        merged = self.det._merge_nearby_points(points)
        assert len(merged) == 2
        np.testing.assert_allclose(merged[0], (101.0, 100.5), atol=0.5)

    def test_preprocess_returns_same_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.det._preprocess(frame)
        assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# Integration tests — require test video
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not VIDEO_PATH.exists(), reason="Test video not available")
class TestWithVideo:
    @pytest.fixture(autouse=True)
    def setup(self):
        from padex.io.video import VideoReader

        self.reader = VideoReader(VIDEO_PATH)
        self.detector = CourtDetector()
        yield
        self.reader.__exit__(None, None, None)

    def test_video_reader_opens(self):
        assert self.reader.frame_count > 0
        assert self.reader.fps > 0

    def test_video_reader_frame_dimensions(self):
        w, h = self.reader.frame_size
        assert w >= 1280
        assert h >= 720

    def test_video_reader_read_frame(self):
        frame = self.reader.read_frame(100)
        assert frame is not None
        assert frame.shape[2] == 3  # BGR

    def test_court_visibility_samples(self):
        """Sample frames and check is_court_visible doesn't crash."""
        for frame_id, _, frame in self.reader.frames(step=500, end_frame=5000):
            result = self.detector.is_court_visible(frame)
            assert isinstance(result, bool)

    def test_calibrate_on_sampled_frames(self):
        """Try calibrating several frames, at least one should succeed."""
        calibrations = []
        for frame_id, _, frame in self.reader.frames(step=300, end_frame=9000):
            cal = self.detector.calibrate_frame(frame)
            if cal is not None:
                calibrations.append((frame_id, cal))

        # We expect at least one successful calibration from a highlights video
        assert len(calibrations) > 0, "No frames could be calibrated"

        # Check the best calibration has reasonable error
        best = min(calibrations, key=lambda c: c[1].reprojection_error or 999)
        assert best[1].reprojection_error is not None
        assert best[1].reprojection_error < 2.0
        assert best[1].homography_matrix is not None
        assert len(best[1].homography_matrix) == 3
        assert len(best[1].homography_matrix[0]) == 3
