"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: run_analysis.py
Description:
    MVP end-to-end script.
    Input: padel match video → Output: annotated video + JSON/HTML report.

    Supports two court calibration modes:
    - Automatic: uses CourtDetector (default, less reliable)
    - Manual:    user clicks 12 keypoints on a reference frame (--calibrate)

Usage:
    # Automatic calibration (default)
    uv run python examples/run_analysis.py input.mp4 --output output.mp4

    # Manual calibration — opens a window to mark court keypoints first
    uv run python examples/run_analysis.py input.mp4 --output output.mp4 --calibrate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from padex.events.bounce import BounceDetector
from padex.events.point import PointSegmenter
from padex.events.shot import ShotDetector
from padex.io.video import VideoReader, VideoWriter
from padex.schemas.tracking import CourtCalibration
from padex.tactics.metrics import MetricsCalculator
from padex.tactics.report import MatchReporter
from padex.tracking.court import COURT_MODEL, CourtDetector
from padex.tracking.pipeline import TrackingPipeline
from padex.viz.frame import FrameAnnotator

logger = logging.getLogger(__name__)

# Ordered keypoints for manual annotation — logical flow on the court
KEYPOINT_ORDER = [
    "bottom_left",
    "bottom_right",
    "service_near_left",
    "service_near_center",
    "service_near_right",
    "net_left",
    "net_right",
    "service_far_left",
    "service_far_center",
    "service_far_right",
    "top_left",
    "top_right",
]

# Colors matching the keypoint groups for visual clarity
KEYPOINT_COLORS = {
    "bottom_left": (0, 0, 255),       # red
    "bottom_right": (0, 0, 255),
    "top_left": (255, 0, 0),          # blue
    "top_right": (255, 0, 0),
    "net_left": (0, 255, 255),        # yellow
    "net_right": (0, 255, 255),
    "service_near_left": (0, 255, 0), # green
    "service_near_center": (0, 255, 0),
    "service_near_right": (0, 255, 0),
    "service_far_left": (255, 0, 255), # magenta
    "service_far_center": (255, 0, 255),
    "service_far_right": (255, 0, 255),
}

KEYPOINT_DESCRIPTIONS = {
    "bottom_left": "Bottom-Left corner (near side, left sideline)",
    "bottom_right": "Bottom-Right corner (near side, right sideline)",
    "service_near_left": "Near service line x Left sideline",
    "service_near_center": "Near service line x Center line",
    "service_near_right": "Near service line x Right sideline",
    "net_left": "Net x Left sideline",
    "net_right": "Net x Right sideline",
    "service_far_left": "Far service line x Left sideline",
    "service_far_center": "Far service line x Center line",
    "service_far_right": "Far service line x Right sideline",
    "top_left": "Top-Left corner (far side, left sideline)",
    "top_right": "Top-Right corner (far side, right sideline)",
}


# ------------------------------------------------------------------
# Manual calibration UI
# ------------------------------------------------------------------

def _find_reference_frame(video_path: Path, start_frame: int = 0) -> tuple[np.ndarray, int]:
    """Find a suitable reference frame for manual calibration.

    Opens a browser window where the user can scrub through the video
    with arrow keys to pick a good frame.

    Controls:
        → / d : forward 90 frames (~3s)
        ← / a : back 90 frames
        ↑ / w : forward 900 frames (~30s)
        ↓ / s : back 900 frames
        Enter  : confirm this frame
        q      : quit without selecting
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fid = start_frame

    WINDOW = "Select Reference Frame (arrows=navigate, Enter=confirm, q=quit)"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    while True:
        fid = max(0, min(fid, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            break

        # Draw HUD
        display = frame.copy()
        ts = fid / fps if fps > 0 else 0
        hud = f"Frame {fid}/{total}  |  {ts:.1f}s  |  arrows=navigate  Enter=confirm"
        cv2.putText(display, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(0) & 0xFF

        if key == 13:  # Enter
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key in (83, ord("d")):  # right arrow
            fid += 90
        elif key in (81, ord("a")):  # left arrow
            fid -= 90
        elif key in (82, ord("w")):  # up arrow
            fid += 900
        elif key in (84, ord("s")):  # down arrow
            fid -= 900

    cap.release()
    cv2.destroyAllWindows()
    return frame, fid


def _annotate_keypoints(frame: np.ndarray) -> dict[str, tuple[float, float]]:
    """Interactive keypoint annotation on a single frame.

    Guides the user through clicking each of the 12 court keypoints.

    Controls:
        Left click  : place current keypoint
        Right click  : undo last keypoint
        s            : skip current keypoint (if occluded)
        Enter        : finish early (if >= 4 points marked)
        q            : abort
    """
    keypoints: dict[str, tuple[float, float]] = {}
    current_idx = 0
    click_pos: tuple[int, int] | None = None

    WINDOW = "Court Keypoint Annotation"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    def on_mouse(event, x, y, flags, param):
        nonlocal click_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            click_pos = (-1, -1)  # signal for undo

    cv2.setMouseCallback(WINDOW, on_mouse)

    while current_idx < len(KEYPOINT_ORDER):
        name = KEYPOINT_ORDER[current_idx]
        desc = KEYPOINT_DESCRIPTIONS[name]
        color = KEYPOINT_COLORS[name]

        # Draw frame with existing keypoints
        display = frame.copy()

        # Draw already-placed keypoints with lines
        _draw_placed_keypoints(display, keypoints)

        # HUD
        progress = f"[{current_idx + 1}/{len(KEYPOINT_ORDER)}]"
        cv2.putText(display, f"{progress} Click: {desc}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, "right-click=undo  s=skip  Enter=finish  q=quit", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"Marked: {len(keypoints)}/12 (need >= 4)", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(30) & 0xFF

        if click_pos is not None:
            if click_pos == (-1, -1):
                # Undo: go back to previous keypoint
                if current_idx > 0:
                    current_idx -= 1
                    prev_name = KEYPOINT_ORDER[current_idx]
                    keypoints.pop(prev_name, None)
                    logger.info("Undo: removed %s", prev_name)
                click_pos = None
            else:
                # Place keypoint
                x, y = click_pos
                keypoints[name] = (float(x), float(y))
                logger.info("Placed %s at (%d, %d)", name, x, y)
                current_idx += 1
                click_pos = None

        elif key == ord("s"):
            # Skip this keypoint
            logger.info("Skipped %s", name)
            current_idx += 1

        elif key == 13:  # Enter — finish early
            if len(keypoints) >= 4:
                break
            else:
                logger.warning("Need at least 4 keypoints to finish (have %d)", len(keypoints))

        elif key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()
    return keypoints


def _draw_placed_keypoints(
    frame: np.ndarray,
    keypoints: dict[str, tuple[float, float]],
) -> None:
    """Draw placed keypoints and connecting court lines on the frame."""
    # Draw points
    for name, (x, y) in keypoints.items():
        color = KEYPOINT_COLORS.get(name, (255, 255, 255))
        cx, cy = int(x), int(y)
        cv2.circle(frame, (cx, cy), 6, color, -1)
        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1)
        cv2.putText(frame, name, (cx + 10, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw lines between placed keypoints that form court lines
    for a, b in COURT_MODEL.LINES:
        if a in keypoints and b in keypoints:
            pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
            pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(frame, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)


def _verify_calibration(
    frame: np.ndarray,
    calibration: CourtCalibration,
) -> bool:
    """Show the calibration result overlaid on the frame for user verification.

    Projects all 12 court keypoints onto the frame using the computed homography
    so the user can visually check alignment.

    Returns True if the user accepts, False to redo.
    """
    display = frame.copy()
    H_inv = np.linalg.inv(np.array(calibration.homography_matrix))

    # Project all court keypoints to pixel space
    for name, (mx, my) in COURT_MODEL.KEYPOINTS.items():
        pt = np.array([[[mx, my]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(pt, H_inv)
        px, py = int(projected[0, 0, 0]), int(projected[0, 0, 1])
        color = KEYPOINT_COLORS.get(name, (255, 255, 255))
        cv2.drawMarker(display, (px, py), color, cv2.MARKER_CROSS, 20, 2)
        cv2.putText(display, name, (px + 12, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw projected court lines
    for a, b in COURT_MODEL.LINES:
        ma = COURT_MODEL.KEYPOINTS[a]
        mb = COURT_MODEL.KEYPOINTS[b]
        pt_a = cv2.perspectiveTransform(
            np.array([[[ma[0], ma[1]]]], dtype=np.float64), H_inv
        )
        pt_b = cv2.perspectiveTransform(
            np.array([[[mb[0], mb[1]]]], dtype=np.float64), H_inv
        )
        p1 = (int(pt_a[0, 0, 0]), int(pt_a[0, 0, 1]))
        p2 = (int(pt_b[0, 0, 0]), int(pt_b[0, 0, 1]))
        cv2.line(display, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)

    error = calibration.reprojection_error or 0
    cv2.putText(display, f"Reprojection error: {error:.4f}m", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, "Yellow lines = projected court model", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(display, "Enter=accept  r=redo  q=quit", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    WINDOW = "Verify Calibration"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.imshow(WINDOW, display)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter
            cv2.destroyAllWindows()
            return True
        elif key == ord("r"):
            cv2.destroyAllWindows()
            return False
        elif key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(0)


def manual_calibrate(
    video_path: Path,
    start_frame: int = 0,
    calibration_file: Path | None = None,
) -> CourtCalibration:
    """Full manual calibration workflow.

    1. Find reference frame
    2. Annotate keypoints
    3. Compute homography
    4. Verify
    5. Optionally save to JSON

    Args:
        video_path: Path to the video file.
        start_frame: Suggested start frame for browsing.
        calibration_file: If provided, save/load calibration JSON here.

    Returns:
        CourtCalibration ready for pipeline use.
    """
    # Try to load existing calibration
    if calibration_file and calibration_file.exists():
        logger.info("Loading existing calibration from %s", calibration_file)
        data = json.loads(calibration_file.read_text())
        return CourtCalibration(**data)

    while True:
        # Step 1: Find a good frame
        logger.info("Step 1: Select a reference frame with a clear court view...")
        frame, fid = _find_reference_frame(video_path, start_frame)
        h, w = frame.shape[:2]
        logger.info("Selected frame %d (%dx%d)", fid, w, h)

        # Step 2: Annotate keypoints
        logger.info("Step 2: Click on court keypoints...")
        keypoints_px = _annotate_keypoints(frame)

        if len(keypoints_px) < 4:
            logger.error("Not enough keypoints (%d). Need at least 4.", len(keypoints_px))
            continue

        # Step 3: Compute calibration
        logger.info("Computing homography from %d keypoints...", len(keypoints_px))
        try:
            calibration = CourtDetector.manual_calibration(
                keypoints_px=keypoints_px,
                frame_width=w,
                frame_height=h,
            )
        except ValueError as e:
            logger.error("Calibration failed: %s. Try again.", e)
            continue

        logger.info("Reprojection error: %.4f m", calibration.reprojection_error or 0)

        # Step 4: Verify
        logger.info("Step 3: Verify the calibration...")
        accepted = _verify_calibration(frame, calibration)
        if accepted:
            # Save calibration
            if calibration_file:
                calibration_file.parent.mkdir(parents=True, exist_ok=True)
                calibration_file.write_text(calibration.model_dump_json(indent=2))
                logger.info("Calibration saved to %s", calibration_file)
            return calibration
        else:
            logger.info("Calibration rejected. Restarting...")
            continue


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
    calibrate: bool = False,
    calibration_file: str | Path | None = None,
    enable_pose: bool = False,
) -> None:
    """Run the full analysis pipeline.

    Pass 0: Manual court calibration (optional, --calibrate)
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
        calibrate: If True, run manual calibration before tracking.
        calibration_file: Path to save/load calibration JSON.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_dir is None:
        report_dir = output_path.parent
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pass 0: Manual court calibration (optional)
    # ------------------------------------------------------------------
    manual_cal: CourtCalibration | None = None

    if calibrate:
        cal_path = Path(calibration_file) if calibration_file else (
            report_dir / f"{match_id}_calibration.json"
        )
        manual_cal = manual_calibrate(video_path, start_frame, cal_path)
        logger.info("Manual calibration ready (error=%.4f m)",
                     manual_cal.reprojection_error or 0)
    elif calibration_file:
        cal_path = Path(calibration_file)
        if cal_path.exists():
            logger.info("Loading calibration from %s", cal_path)
            data = json.loads(cal_path.read_text())
            manual_cal = CourtCalibration(**data)

    # ------------------------------------------------------------------
    # Pass 1: Data generation
    # ------------------------------------------------------------------
    logger.info("Pass 1: Running tracking pipeline...")
    pipeline = TrackingPipeline(
        video_path, device=device, manual_calibration=manual_cal,
        enable_pose=enable_pose,
    )
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
        default="output/output.mp4",
        help="Output annotated video path (default: output/output.mp4)",
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
        "--calibrate", "-c",
        action="store_true",
        help="Manually calibrate the court before running (recommended)",
    )
    parser.add_argument(
        "--pose",
        action="store_true",
        help="Enable YOLO-Pose keypoint detection and draw skeleton on output video",
    )
    parser.add_argument(
        "--calibration-file",
        default=None,
        help="Path to save/load court calibration JSON",
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
        calibrate=args.calibrate,
        calibration_file=args.calibration_file,
        enable_pose=args.pose,
    )


if __name__ == "__main__":
    main()
