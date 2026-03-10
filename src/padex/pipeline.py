"""
Project: Padex
File Created: 2026-03-07
Author: Xingnan Zhu
File Name: pipeline.py
Description:
    Top-level pipeline orchestrator for Padex.

    Provides the Padex class and convenience functions for running
    the full analysis pipeline: tracking -> bounce detection -> shot
    classification, with optional annotated video export.

Usage:
    import padex

    result = padex.process("match.mp4", calibration="cal.json")
    print(f"Found {len(result.shots)} shots")

    padex.export_video(result, "match.mp4", "output.mp4")
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from padex.schemas.events import Bounce, Shot
from padex.schemas.tracking import CourtCalibration
from padex.tracking.pipeline import TrackingPipeline, TrackingResult

logger = logging.getLogger(__name__)


@dataclass
class PadexResult:
    """Output of the full Padex pipeline."""

    tracking: TrackingResult
    bounces: list[Bounce] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)
    calibration: CourtCalibration | None = None


class Padex:
    """Top-level pipeline: video -> tracking -> bounces -> shots.

    Args:
        video_path: Path to the input video file.
        calibration: Court calibration — a CourtCalibration object, a Path
            to a calibration JSON, or None for auto-detection. If None,
            looks for ``<video_stem>_calibration.json`` next to the video.
        enable_pose: Whether to run pose estimation (needed for shot
            classification). Defaults to True.
        cache_tracking: Whether to cache tracking results as pickle for
            faster re-runs. Defaults to True.
        cache_dir: Directory for tracking cache files. Defaults to
            ``output/`` under the project root.
    """

    def __init__(
        self,
        video_path: str | Path,
        calibration: CourtCalibration | str | Path | None = None,
        enable_pose: bool = True,
        cache_tracking: bool = True,
        cache_dir: str | Path | None = None,
        use_tracknet_v3: bool = False,
        ball_model_path: str | Path | None = None,
        event_model_path: str | Path | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.enable_pose = enable_pose
        self.cache_tracking = cache_tracking
        self.cache_dir = Path(cache_dir) if cache_dir else self.video_path.parent
        self.use_tracknet_v3 = use_tracknet_v3
        self.ball_model_path = str(ball_model_path) if ball_model_path else None
        self.event_model_path = Path(event_model_path) if event_model_path else None

        self._calibration = self._resolve_calibration(calibration)

    def _resolve_calibration(
        self, calibration: CourtCalibration | str | Path | None
    ) -> CourtCalibration | None:
        if isinstance(calibration, CourtCalibration):
            return calibration

        if isinstance(calibration, (str, Path)):
            cal_path = Path(calibration)
            if not cal_path.exists():
                raise FileNotFoundError(f"Calibration file not found: {cal_path}")
            with open(cal_path) as f:
                return CourtCalibration(**json.load(f))

        # Auto-discover sibling calibration file
        cal_sibling = self.video_path.with_name(
            self.video_path.stem + "_calibration.json"
        )
        if cal_sibling.exists():
            logger.info("Auto-discovered calibration: %s", cal_sibling)
            with open(cal_sibling) as f:
                return CourtCalibration(**json.load(f))

        raise ValueError(
            f"No court calibration found for '{self.video_path.name}'. "
            f"Run interactive calibration first:\n"
            f"  cal = padex.interactive_calibrate(\"{self.video_path}\")\n"
            f"  result = padex.process(\"{self.video_path}\", calibration=cal)\n"
            f"Or from CLI:\n"
            f"  padex calibrate \"{self.video_path}\""
        )

    def run(self) -> PadexResult:
        """Run the full pipeline: tracking -> bounce detection -> shot classification."""
        from padex.events.bounce import BounceDetector, MLPEventDetectionStrategy
        from padex.events.shot import PoseBasedShotTypeClassifier, ShotDetector

        # Stage 1: Tracking
        tracking = self._run_tracking()

        # Stage 2: Bounce detection
        logger.info("=== Stage 2: Bounce detection ===")
        if self.event_model_path and self.event_model_path.exists():
            logger.info("Using MLP event detector: %s", self.event_model_path)
            mlp = MLPEventDetectionStrategy(model_path=self.event_model_path)
            bounce_indices, hit_indices = mlp.detect_events(
                tracking.ball_frames, tracking.player_frames
            )
            logger.info(
                "MLP detected %d bounce candidates, %d hit candidates",
                len(bounce_indices), len(hit_indices),
            )
            # Build Bounce objects from indices using geometry classifier
            bounce_detector = BounceDetector()
            bounces = bounce_detector.detect_bounces(
                tracking.ball_frames, tracking.calibration,
                precomputed_indices=bounce_indices,
            )
        else:
            bounce_detector = BounceDetector()
            bounces = bounce_detector.detect_bounces(
                tracking.ball_frames, tracking.calibration
            )
            hit_indices = []
        logger.info("Detected %d bounces", len(bounces))

        # Stage 3: Shot detection + classification
        logger.info("=== Stage 3: Shot detection ===")
        shot_detector = ShotDetector(
            shot_type_classifier=PoseBasedShotTypeClassifier(),
        )
        shots = shot_detector.detect_shots(
            player_frames=tracking.player_frames,
            ball_frames=tracking.ball_frames,
            bounces=bounces,
        )
        logger.info("Detected %d shots", len(shots))
        for s in shots:
            logger.info(
                "  Shot %-20s | player=%-8s | type=%-18s | conf=%.2f | ts=%.0f ms",
                s.shot_id, s.player_id, s.shot_type.value, s.confidence, s.timestamp_ms,
            )

        # Filter bounces that overlap with shot contacts (within 200ms)
        # These are racket hits misdetected as bounces
        contact_times = [s.timestamp_ms for s in shots]
        contact_overlap_ms = 200.0
        filtered_bounces = [
            b for b in bounces
            if b.timestamp_ms is None
            or not any(
                abs(b.timestamp_ms - ct) < contact_overlap_ms
                for ct in contact_times
            )
        ]
        if len(filtered_bounces) < len(bounces):
            logger.info(
                "Filtered %d false bounces near shot contacts (%d → %d)",
                len(bounces) - len(filtered_bounces),
                len(bounces),
                len(filtered_bounces),
            )

        return PadexResult(
            tracking=tracking,
            bounces=filtered_bounces,
            shots=shots,
            calibration=tracking.calibration,
        )

    def _run_tracking(self) -> TrackingResult:
        cache_path = self.cache_dir / f"{self.video_path.stem}_tracking_cache.pkl"

        if self.cache_tracking and cache_path.exists():
            logger.info("=== Stage 1: Loading cached tracking results ===")
            with open(cache_path, "rb") as f:
                tracking = pickle.load(f)
            logger.info(
                "Loaded from cache: %d player frames, %d ball frames",
                len(tracking.player_frames),
                len(tracking.ball_frames),
            )
            return tracking

        logger.info("=== Stage 1: Running tracking pipeline ===")
        pipeline = TrackingPipeline(
            video_path=self.video_path,
            enable_pose=self.enable_pose,
            manual_calibration=self._calibration,
            use_tracknet_v3=self.use_tracknet_v3,
            ball_model_path=self.ball_model_path,
        )
        tracking = pipeline.run()
        logger.info(
            "Tracking done: %d player frames, %d ball frames",
            len(tracking.player_frames),
            len(tracking.ball_frames),
        )

        if self.cache_tracking:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(tracking, f)
            logger.info("Tracking results cached: %s", cache_path)

        return tracking

    def export_video(
        self,
        result: PadexResult,
        output_path: str | Path,
        shot_display_ms: float = 1500.0,
    ) -> Path:
        """Export annotated video with shot labels overlaid.

        Args:
            result: PadexResult from run().
            output_path: Path for the output video file.
            shot_display_ms: How long to display each shot label (ms).

        Returns:
            Path to the exported video file.
        """
        from padex.io.video import VideoReader, VideoWriter
        from padex.viz.frame import FrameAnnotator

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        shot_events = sorted(result.shots, key=lambda s: s.timestamp_ms)

        def get_active_shot(timestamp_ms: float):
            active = None
            for s in shot_events:
                if s.timestamp_ms <= timestamp_ms <= s.timestamp_ms + shot_display_ms:
                    active = s
            return active

        player_lookup: dict[int, list] = defaultdict(list)
        for pf in result.tracking.player_frames:
            player_lookup[pf.frame_id].append(pf)

        ball_lookup: dict[int, object] = {}
        for bf in result.tracking.ball_frames:
            ball_lookup[bf.frame_id] = bf

        shot_counts: dict[str, int] = defaultdict(int)
        for s in result.shots:
            shot_counts[s.shot_type.value] += 1

        annotator = FrameAnnotator()

        logger.info("Exporting annotated video to %s", output_path)
        with VideoReader(self.video_path) as reader:
            fps = reader.fps
            w, h = reader.frame_size
            bounce_display_frames = int(fps)  # ~1 second

            # Pre-compute bounce → frame_id mapping
            bounce_events: list[tuple[int, object]] = []
            for b in result.bounces:
                if b.timestamp_ms is not None:
                    b_frame = round(b.timestamp_ms / (1000.0 / fps))
                    bounce_events.append((b_frame, b))

            with VideoWriter(output_path, fps=fps, frame_size=(w, h)) as writer:
                for frame_id, timestamp_ms, frame in reader.frames():
                    player_frames_here = player_lookup.get(frame_id, [])
                    ball_frame_here = ball_lookup.get(frame_id)
                    active_shot = get_active_shot(timestamp_ms)

                    # Collect active bounces with fade progress
                    active_bounces = []
                    for b_frame, bounce in bounce_events:
                        elapsed = frame_id - b_frame
                        if 0 <= elapsed < bounce_display_frames:
                            progress = elapsed / bounce_display_frames
                            active_bounces.append((bounce, progress))

                    stats = {"Frame": frame_id, "Shots": len(result.shots)}
                    for shot_type, count in sorted(
                        shot_counts.items(), key=lambda x: -x[1]
                    )[:4]:
                        stats[shot_type[:12]] = count

                    annotator.annotate_frame(
                        frame=frame,
                        frame_id=frame_id,
                        player_frames=player_frames_here,
                        ball_frame=ball_frame_here,
                        calibration=result.calibration,
                        shot=active_shot,
                        stats=stats,
                        active_bounces=active_bounces,
                    )
                    writer.write(frame)

                    if frame_id % 100 == 0:
                        logger.info("  Written frame %d", frame_id)

        logger.info("Annotated video saved: %s", output_path)
        return output_path


def process(
    video_path: str | Path,
    calibration: CourtCalibration | str | Path | None = None,
    event_model_path: str | Path | None = None,
    **kwargs,
) -> PadexResult:
    """Convenience function: run the full Padex pipeline.

    Args:
        video_path: Path to the input video file.
        calibration: Court calibration (CourtCalibration, JSON path, or None).
        **kwargs: Additional arguments passed to Padex().

    Returns:
        PadexResult with tracking, bounces, and shots.
    """
    return Padex(video_path, calibration=calibration, event_model_path=event_model_path, **kwargs).run()


def export_video(
    result: PadexResult,
    video_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> Path:
    """Convenience function: export annotated video.

    Args:
        result: PadexResult from process().
        video_path: Path to the original video (for frame data).
        output_path: Path for the output video file.
        **kwargs: Additional arguments passed to Padex.export_video().

    Returns:
        Path to the exported video file.
    """
    p = Padex(video_path, calibration=result.calibration, cache_tracking=True)
    return p.export_video(result, output_path, **kwargs)
