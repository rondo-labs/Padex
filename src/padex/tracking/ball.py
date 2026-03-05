"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: ball.py
Description:
    Ball detection and trajectory tracking.
    Uses SAHI (Slicing Aided Hyper Inference) with YOLO for small-ball
    detection, and a Kalman filter for cross-frame tracking and gap-filling.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass

import cv2
import numpy as np

from padex.schemas.tracking import (
    BallFrame,
    BallVisibility,
    BoundingBox,
    Position3D,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RawBallDetection:
    """Single ball detection from a frame."""

    bbox: BoundingBox
    confidence: float
    frame_id: int
    timestamp_ms: float


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class BallDetectionStrategy(abc.ABC):
    """Abstract interface for ball detection backends."""

    @abc.abstractmethod
    def detect(
        self, frame: np.ndarray, frame_id: int, timestamp_ms: float
    ) -> RawBallDetection | None:
        """Detect ball in a single frame. Returns None if not found."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        ...


class BallTracker(abc.ABC):
    """Abstract interface for ball tracking across frames."""

    @abc.abstractmethod
    def update(
        self, detection: RawBallDetection | None, timestamp_ms: float
    ) -> tuple[Position3D | None, BallVisibility]:
        """Update tracker with (possibly absent) detection.

        Returns (position_in_meters, visibility_state).
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
        ...

    @abc.abstractmethod
    def set_homography(self, H: np.ndarray | None) -> None:
        """Set or update the pixel-to-court homography matrix."""
        ...


# ---------------------------------------------------------------------------
# SAHI + YOLO ball detection strategy
# ---------------------------------------------------------------------------


class SahiYoloBallDetectionStrategy(BallDetectionStrategy):
    """Ball detection using SAHI sliced inference with YOLO.

    SAHI slices the image into overlapping patches to detect small objects
    that would be missed by standard YOLO inference on downscaled frames.
    """

    BALL_CLASS_ID = 32  # COCO 'sports ball'

    def __init__(
        self,
        model_path: str = "assets/weights/yolo26m.pt",
        confidence_threshold: float = 0.25,
        slice_size: int = 512,
        overlap_ratio: float = 0.2,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.device = device
        self._model = None

    def _load_model(self):
        from sahi import AutoDetectionModel

        return AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            device=self.device or "",
        )

    def _ensure_model(self):
        if self._model is None:
            self._model = self._load_model()

    def detect(
        self, frame: np.ndarray, frame_id: int, timestamp_ms: float
    ) -> RawBallDetection | None:
        from sahi.predict import get_sliced_prediction

        self._ensure_model()

        # SAHI expects RGB, OpenCV provides BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = get_sliced_prediction(
            image=frame_rgb,
            detection_model=self._model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            perform_standard_pred=False,
            verbose=0,
        )

        # Filter to ball class only
        ball_preds = [
            p
            for p in result.object_prediction_list
            if p.category.id == self.BALL_CLASS_ID
        ]

        if not ball_preds:
            return None

        # Pick highest confidence detection
        best = max(ball_preds, key=lambda p: p.score.value)
        x1, y1, x2, y2 = best.bbox.to_xyxy()

        return RawBallDetection(
            bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            confidence=float(best.score.value),
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
        )

    def reset(self) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# Kalman ball tracker
# ---------------------------------------------------------------------------


class KalmanBallTracker(BallTracker):
    """4D Kalman filter tracking ball in court coordinates.

    State vector: [x, y, vx, vy] in meters.
    Observation:  [x, y] from YOLO detection projected to court coords.
    """

    MAX_OCCLUDED_FRAMES: int = 10
    GATE_DISTANCE_M: float = 3.0

    def __init__(
        self,
        process_noise_std: float = 0.5,
        measurement_noise_std: float = 0.3,
        homography_matrix: np.ndarray | None = None,
    ) -> None:
        self._H_matrix = homography_matrix
        self._process_noise_std = process_noise_std
        self._measurement_noise_std = measurement_noise_std
        self._kf: cv2.KalmanFilter | None = None
        self._initialized = False
        self._occluded_count = 0

    def _build_kalman(self) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        # Constant velocity model
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        kf.processNoiseCov = (
            np.eye(4, dtype=np.float32) * self._process_noise_std**2
        )
        kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * self._measurement_noise_std**2
        )
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        return kf

    def set_homography(self, H: np.ndarray | None) -> None:
        self._H_matrix = H

    def update(
        self, detection: RawBallDetection | None, timestamp_ms: float
    ) -> tuple[Position3D | None, BallVisibility]:
        if detection is None:
            return self._handle_missing()

        # Project bbox center to court coordinates
        court_xy = self._bbox_center_to_court(detection.bbox)
        if court_xy is None:
            return self._handle_missing()

        x_m, y_m = court_xy

        # Initialize on first detection
        if not self._initialized or self._kf is None:
            self._kf = self._build_kalman()
            self._kf.statePost = np.array(
                [x_m, y_m, 0, 0], dtype=np.float32
            ).reshape(4, 1)
            self._initialized = True
            self._occluded_count = 0
            return self._clamp_to_court(x_m, y_m), BallVisibility.VISIBLE

        # Gate: reject detections far from prediction
        prediction = self._kf.predict()
        pred_x, pred_y = float(prediction[0, 0]), float(prediction[1, 0])
        dist = np.sqrt((pred_x - x_m) ** 2 + (pred_y - y_m) ** 2)
        if dist > self.GATE_DISTANCE_M:
            # Ghost detection (glass reflection) — use prediction instead
            self._occluded_count += 1
            if self._occluded_count > self.MAX_OCCLUDED_FRAMES:
                self.reset()
                return None, BallVisibility.OCCLUDED
            return self._clamp_to_court(pred_x, pred_y), BallVisibility.INFERRED

        # Normal update: predict + correct
        measurement = np.array([[x_m], [y_m]], dtype=np.float32)
        corrected = self._kf.correct(measurement)
        self._occluded_count = 0
        cx, cy = float(corrected[0, 0]), float(corrected[1, 0])
        return self._clamp_to_court(cx, cy), BallVisibility.VISIBLE

    def reset(self) -> None:
        self._kf = None
        self._initialized = False
        self._occluded_count = 0

    def _handle_missing(self) -> tuple[Position3D | None, BallVisibility]:
        if not self._initialized or self._kf is None:
            return None, BallVisibility.OCCLUDED

        prediction = self._kf.predict()
        self._occluded_count += 1

        if self._occluded_count > self.MAX_OCCLUDED_FRAMES:
            self.reset()
            return None, BallVisibility.OCCLUDED

        x, y = float(prediction[0, 0]), float(prediction[1, 0])
        return self._clamp_to_court(x, y), BallVisibility.INFERRED

    def _bbox_center_to_court(
        self, bbox: BoundingBox
    ) -> tuple[float, float] | None:
        """Project bbox center from pixel to court coordinates."""
        if self._H_matrix is None:
            return None

        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = (bbox.y1 + bbox.y2) / 2.0
        pt = np.array([[[cx, cy]]], dtype=np.float64)
        result = cv2.perspectiveTransform(pt, self._H_matrix)
        x, y = float(result[0, 0, 0]), float(result[0, 0, 1])
        return (x, y)

    @staticmethod
    def _clamp_to_court(x: float, y: float) -> Position3D:
        """Clamp position to valid court bounds and return Position3D."""
        x_c = max(0.0, min(10.0, x))
        y_c = max(0.0, min(20.0, y))
        return Position3D(x=x_c, y=y_c, z=0.0)


# ---------------------------------------------------------------------------
# BallDetector facade
# ---------------------------------------------------------------------------


class BallDetector:
    """Main ball detection + tracking facade.

    Combines SAHI-based detection with Kalman filter tracking.
    """

    def __init__(
        self,
        detection_strategy: BallDetectionStrategy | None = None,
        tracker: BallTracker | None = None,
        model_path: str | None = None,
        confidence_threshold: float = 0.25,
    ) -> None:
        self.detection_strategy = detection_strategy or SahiYoloBallDetectionStrategy(
            model_path=model_path or "assets/weights/yolo26m.pt",
            confidence_threshold=confidence_threshold,
        )
        self.tracker = tracker or KalmanBallTracker()

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
    ) -> BallFrame:
        """Detect ball in a single frame (no cross-frame tracking)."""
        raw = self.detection_strategy.detect(frame, frame_id, timestamp_ms)
        return BallFrame(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            bbox=raw.bbox if raw else None,
            position=None,
            confidence=raw.confidence if raw else 0.0,
            visibility=BallVisibility.VISIBLE if raw else BallVisibility.OCCLUDED,
        )

    def track(
        self,
        frames: list[tuple[int, float, np.ndarray]],
        homography_matrix: np.ndarray | None = None,
    ) -> list[BallFrame]:
        """Track ball across multiple frames with Kalman gap-filling.

        Args:
            frames: list of (frame_id, timestamp_ms, frame_array).
            homography_matrix: pixel-to-court transform for Position3D.

        Returns:
            list[BallFrame] with visibility states and interpolated positions.
        """
        if homography_matrix is not None:
            self.tracker.set_homography(homography_matrix)

        ball_frames: list[BallFrame] = []
        for frame_id, timestamp_ms, frame in frames:
            raw = self.detection_strategy.detect(frame, frame_id, timestamp_ms)
            position, visibility = self.tracker.update(raw, timestamp_ms)

            ball_frames.append(
                BallFrame(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    bbox=raw.bbox if raw else None,
                    position=position,
                    confidence=raw.confidence if raw else 0.0,
                    visibility=visibility,
                )
            )

        return ball_frames

    def reset(self) -> None:
        """Reset detector and tracker state."""
        self.detection_strategy.reset()
        self.tracker.reset()
