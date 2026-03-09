"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: pipeline.py
Description:
    Orchestrates the full tracking pipeline across all detectors.
    End-to-end: video → structured tracking data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from padex.schemas.tracking import BallFrame, CourtCalibration, PlayerFrame
from padex.tracking.ball import BallDetector, SahiYoloBallDetectionStrategy, TrackNetV3BallDetectionStrategy
from padex.tracking.court import CourtDetector
from padex.tracking.device import detect_device
from padex.tracking.player import (
    PlayerDetector,
    YoloPlayerDetectionStrategy,
    YoloPoseEstimationStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Output of the tracking pipeline."""

    player_frames: list[PlayerFrame] = field(default_factory=list)
    ball_frames: list[BallFrame] = field(default_factory=list)
    calibration: CourtCalibration | None = None


class TrackingPipeline:
    """End-to-end tracking pipeline: video → structured tracking data.

    Stages:
    1. Court calibration — sample frames to find best homography
    2. Per-frame player detection + tracking
    3. Ball detection + Kalman tracking (batch)
    """

    def __init__(
        self,
        video_path: str | Path,
        court_detector: CourtDetector | None = None,
        player_detector: PlayerDetector | None = None,
        ball_detector: BallDetector | None = None,
        calibration_sample_step: int = 300,
        device: str | None = None,
        manual_calibration: CourtCalibration | None = None,
        enable_pose: bool = False,
        use_tracknet_v3: bool = False,
        ball_model_path: str | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.device = device or detect_device()
        logger.info("Inference device: %s", self.device)

        pose_strategy = (
            YoloPoseEstimationStrategy(device=self.device) if enable_pose else None
        )

        self.court_detector = court_detector or CourtDetector()
        self.player_detector = player_detector or PlayerDetector(
            detection_strategy=YoloPlayerDetectionStrategy(device=self.device),
            pose_strategy=pose_strategy,
        )
        self.ball_detector = ball_detector or BallDetector(
            use_tracknet=not use_tracknet_v3,
            use_tracknet_v3=use_tracknet_v3,
            model_path=ball_model_path,
        )
        self.calibration_sample_step = calibration_sample_step
        self.manual_calibration = manual_calibration

    def run(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
        step: int = 1,
    ) -> TrackingResult:
        """Run the full tracking pipeline.

        Args:
            start_frame: First frame to process.
            end_frame: Last frame (exclusive). None for entire video.
            step: Process every Nth frame.

        Returns:
            TrackingResult with player_frames, ball_frames, calibration.
        """
        from padex.io.video import VideoReader

        player_frames: list[PlayerFrame] = []
        ball_frames: list[BallFrame] = []

        with VideoReader(self.video_path) as reader:
            total_frames = reader.frame_count

            # Stage 1: Court calibration
            calibration: CourtCalibration | None = None
            H: np.ndarray | None = None

            if self.manual_calibration is not None:
                calibration = self.manual_calibration
                H = np.array(calibration.homography_matrix)
                logger.info("Using manual court calibration")
            else:
                calibration = self._calibrate_court(reader)
                if calibration is not None:
                    H = np.array(calibration.homography_matrix)
                    logger.info(
                        "Court calibrated (error=%.3f)",
                        calibration.reprojection_error or -1,
                    )
                else:
                    logger.warning("Court calibration failed — positions will be None")

            # Set homography for ball tracker before the loop
            if H is not None:
                self.ball_detector.tracker.set_homography(H)

            # Stage 2: Per-frame detection (streaming — no frame buffering)
            processed = 0
            for frame_id, timestamp_ms, frame in reader.frames(
                start_frame=start_frame, end_frame=end_frame, step=step
            ):
                # Player detection + tracking
                pf_list = self.player_detector.detect_and_track(
                    frame, frame_id, timestamp_ms, homography_matrix=H
                )
                player_frames.extend(pf_list)

                # Ball detection + tracking (streaming, no buffering)
                bf = self.ball_detector.detect_and_track_single(
                    frame, frame_id, timestamp_ms
                )
                ball_frames.append(bf)

                processed += 1
                if processed % 500 == 0:
                    logger.info(
                        "Processed %d / %d frames (%.1f%%)",
                        processed,
                        total_frames,
                        100.0 * processed / total_frames,
                    )

        logger.info("Tracking complete: %d frames processed", processed)

        return TrackingResult(
            player_frames=player_frames,
            ball_frames=ball_frames,
            calibration=calibration,
        )

    def _calibrate_court(self, reader) -> CourtCalibration | None:
        """Find the best court calibration from sampled frames."""
        best_cal: CourtCalibration | None = None
        best_error = float("inf")

        for frame_id, _, frame in reader.frames(
            step=self.calibration_sample_step
        ):
            cal = self.court_detector.calibrate_frame(frame)
            if cal is not None and cal.reprojection_error is not None:
                if cal.reprojection_error < best_error:
                    best_error = cal.reprojection_error
                    best_cal = cal

        return best_cal
