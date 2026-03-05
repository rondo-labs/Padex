"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: run_analysis.py
Description:
    MVP end-to-end script.
    Input: padel match video → Output: annotated video + JSON/HTML report.

Usage:
    uv run python examples/run_analysis.py input.mp4 --output output.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from padex.events.bounce import BounceDetector
from padex.events.point import PointSegmenter
from padex.events.shot import ShotDetector
from padex.io.video import VideoReader, VideoWriter
from padex.tactics.metrics import MetricsCalculator
from padex.tactics.report import MatchReporter
from padex.tracking.pipeline import TrackingPipeline
from padex.viz.frame import FrameAnnotator

logger = logging.getLogger(__name__)


def _build_frame_index(tracking_result):
    """Index tracking data by frame_id for fast lookup during Pass 2."""
    from collections import defaultdict

    player_by_frame = defaultdict(list)
    for pf in tracking_result.player_frames:
        player_by_frame[pf.frame_id].append(pf)

    ball_by_frame = {}
    for bf in tracking_result.ball_frames:
        ball_by_frame[bf.frame_id] = bf

    return player_by_frame, ball_by_frame


def _build_shot_index(shots):
    """Index shots by frame_id (approximate via timestamp)."""
    shot_by_ts = {}
    for s in shots:
        shot_by_ts[s.timestamp_ms] = s
    return shot_by_ts


def _find_active_shot(timestamp_ms, shot_by_ts, display_duration_ms=500.0):
    """Find a shot to display if one was hit within display_duration_ms."""
    for ts, shot in shot_by_ts.items():
        if 0 <= (timestamp_ms - ts) < display_duration_ms:
            return shot
    return None


def _compute_stats_at_frame(shots, frame_timestamp_ms):
    """Compute cumulative stats up to the current frame timestamp."""
    shot_counts = defaultdict(int)
    total = 0
    for s in shots:
        if s.timestamp_ms <= frame_timestamp_ms:
            shot_counts[s.player_id] += 1
            total += 1

    if total == 0:
        return {}

    stats = {"Total Shots": total}
    for pid, count in sorted(shot_counts.items()):
        stats[pid] = count
    return stats


def run(
    video_path: str | Path,
    output_path: str | Path,
    report_dir: str | Path | None = None,
    match_id: str = "M_001",
    start_frame: int = 0,
    end_frame: int | None = None,
    device: str | None = None,
) -> None:
    """Run the full analysis pipeline.

    Pass 1: Tracking → Events → Analytics
    Pass 2: Video annotation
    Pass 3: Report generation

    Args:
        video_path: Input video file.
        output_path: Output annotated video file.
        report_dir: Directory for JSON/HTML reports (default: same as output).
        match_id: Match identifier.
        start_frame: First frame to process.
        end_frame: Last frame (exclusive). None = all.
        device: Inference device ("cuda", "mps", "cpu"). None = auto-detect.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    if report_dir is None:
        report_dir = output_path.parent
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pass 1: Data generation
    # ------------------------------------------------------------------
    logger.info("Pass 1: Running tracking pipeline...")
    pipeline = TrackingPipeline(video_path, device=device)
    result = pipeline.run(start_frame=start_frame, end_frame=end_frame)
    logger.info(
        "Tracking complete: %d player frames, %d ball frames",
        len(result.player_frames),
        len(result.ball_frames),
    )

    logger.info("Detecting bounces...")
    bounce_detector = BounceDetector()
    bounces = bounce_detector.detect_bounces(
        result.ball_frames, result.calibration,
    )
    logger.info("Detected %d bounces", len(bounces))

    logger.info("Detecting shots...")
    shot_detector = ShotDetector()
    shots = shot_detector.detect_shots(
        result.player_frames, result.ball_frames, bounces,
    )
    logger.info("Detected %d shots", len(shots))

    logger.info("Segmenting points...")
    segmenter = PointSegmenter()
    points = segmenter.segment(shots, result.ball_frames)
    logger.info("Segmented %d points", len(points))

    logger.info("Computing analytics...")
    calc = MetricsCalculator()
    analytics = calc.compute_match_analytics(
        points, result.player_frames, match_id,
    )

    # ------------------------------------------------------------------
    # Pass 2: Video annotation
    # ------------------------------------------------------------------
    logger.info("Pass 2: Annotating video...")
    player_by_frame, ball_by_frame = _build_frame_index(result)
    shot_by_ts = _build_shot_index(shots)
    annotator = FrameAnnotator()

    with VideoReader(video_path) as reader:
        fps = reader.fps
        frame_size = reader.frame_size

        with VideoWriter(output_path, fps, frame_size) as writer:
            for fid, ts_ms, frame in reader.frames(
                start_frame=start_frame, end_frame=end_frame,
            ):
                pfs = player_by_frame.get(fid, [])
                bf = ball_by_frame.get(fid)
                active_shot = _find_active_shot(ts_ms, shot_by_ts)
                stats = _compute_stats_at_frame(shots, ts_ms)

                annotator.annotate_frame(
                    frame,
                    frame_id=fid,
                    player_frames=pfs,
                    ball_frame=bf,
                    calibration=result.calibration,
                    shot=active_shot,
                    stats=stats,
                )

                writer.write(frame)

                if fid % 300 == 0:
                    logger.info("  Annotated frame %d", fid)

    logger.info("Output video: %s (%d frames)", output_path, writer.frame_count)

    # ------------------------------------------------------------------
    # Pass 3: Reports
    # ------------------------------------------------------------------
    logger.info("Pass 3: Generating reports...")
    reporter = MatchReporter()
    json_path = report_dir / f"{match_id}_analytics.json"
    reporter.to_json(analytics, json_path)
    logger.info("JSON report: %s", json_path)

    html_path = report_dir / f"{match_id}_report.html"
    reporter.to_html(analytics, path=html_path)
    logger.info("HTML report: %s", html_path)

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Padex MVP — padel match video analysis",
    )
    parser.add_argument("video", help="Input video file path")
    parser.add_argument(
        "--output", "-o",
        default="output.mp4",
        help="Output annotated video path (default: output.mp4)",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for JSON/HTML reports (default: same as output)",
    )
    parser.add_argument(
        "--match-id",
        default="M_001",
        help="Match identifier (default: M_001)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame to process",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame (exclusive)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Inference device (default: auto-detect GPU, fallback CPU)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run(
        video_path=args.video,
        output_path=args.output,
        report_dir=args.report_dir,
        match_id=args.match_id,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        device=args.device,
    )


if __name__ == "__main__":
    main()
