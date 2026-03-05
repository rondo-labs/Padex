"""
Project: Padex
File Created: 2026-03-05
Author: Xingnan Zhu
File Name: manual_calibrate.py
Description:
    Interactive manual court calibration tool.

    Opens a video, lets the user pick a good frame, then guides them through
    clicking 12 court keypoints.  Produces a CourtCalibration JSON that can
    be loaded directly into TrackingPipeline.

Usage:
    python scripts/manual_calibrate.py assets/raw/video/match.mp4
    python scripts/manual_calibrate.py assets/raw/video/match.mp4 -o calibration.json

Controls:
    Frame selection:
        D / →       Next frame (+90 frames)
        A / ←       Previous frame (-90 frames)
        W / ↑       Jump forward (+900 frames)
        S / ↓       Jump backward (-900 frames)
        Enter       Confirm this frame and start labeling

    Keypoint labeling:
        Left-click  Place current keypoint
        Z           Undo last keypoint
        N           Skip current keypoint (if not visible)
        Enter       Finish labeling (need >= 4 points)
        Q / Esc     Quit without saving
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from padex.tracking.court import COURT_MODEL, CourtDetector

# Keypoints in labeling order: corners first, then net, then service lines.
KEYPOINT_ORDER = [
    "bottom_left",
    "bottom_right",
    "top_left",
    "top_right",
    "net_left",
    "net_right",
    "service_near_left",
    "service_near_center",
    "service_near_right",
    "service_far_left",
    "service_far_center",
    "service_far_right",
]

# Color per keypoint group (BGR)
KEYPOINT_COLORS = {
    "bottom_left": (0, 0, 255),
    "bottom_right": (0, 68, 255),
    "top_left": (255, 0, 0),
    "top_right": (255, 68, 68),
    "net_left": (0, 255, 255),
    "net_right": (0, 200, 200),
    "service_near_left": (0, 255, 0),
    "service_near_center": (0, 200, 0),
    "service_near_right": (0, 150, 0),
    "service_far_left": (255, 0, 255),
    "service_far_center": (200, 0, 200),
    "service_far_right": (150, 0, 150),
}

# Court meter coords for display
KEYPOINT_METERS = COURT_MODEL.KEYPOINTS


class FrameSelector:
    """Phase 1: Let the user browse frames and pick a good one."""

    def __init__(self, video_path: Path) -> None:
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_id = 0
        self.frame: np.ndarray | None = None

    def run(self) -> tuple[np.ndarray, int] | None:
        """Show frames, return (frame, frame_id) or None if user quits."""
        self._read_frame()

        cv2.namedWindow("Select Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Frame", 1280, 720)

        while True:
            display = self._draw_overlay()
            cv2.imshow("Select Frame", display)

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q") or key == 27:  # Esc
                cv2.destroyAllWindows()
                return None
            elif key == 13:  # Enter
                cv2.destroyAllWindows()
                return self.frame.copy(), self.frame_id
            elif key == ord("d") or key == 83:  # → right arrow
                self._seek(90)
            elif key == ord("a") or key == 81:  # ← left arrow
                self._seek(-90)
            elif key == ord("w") or key == 82:  # ↑ up arrow
                self._seek(900)
            elif key == ord("s") or key == 84:  # ↓ down arrow
                self._seek(-900)

    def _seek(self, delta: int) -> None:
        self.frame_id = max(0, min(self.total_frames - 1, self.frame_id + delta))
        self._read_frame()

    def _read_frame(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame

    def _draw_overlay(self) -> np.ndarray:
        display = self.frame.copy()
        h, w = display.shape[:2]
        timestamp = self.frame_id / self.fps if self.fps > 0 else 0

        # Info bar
        info = f"Frame {self.frame_id}/{self.total_frames}  |  {timestamp:.1f}s  |  {w}x{h}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        controls = "A/D: prev/next  W/S: jump  Enter: confirm  Q: quit"
        cv2.putText(display, controls, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return display

    def release(self) -> None:
        self.cap.release()


class KeypointLabeler:
    """Phase 2: Let the user click keypoints on the selected frame."""

    def __init__(self, frame: np.ndarray) -> None:
        self.original = frame.copy()
        self.frame_h, self.frame_w = frame.shape[:2]
        self.current_idx = 0
        self.labeled: dict[str, tuple[float, float]] = {}
        self.click_pos: tuple[int, int] | None = None

    def run(self) -> dict[str, tuple[float, float]] | None:
        """Run the labeling loop. Returns keypoint dict or None if quit."""
        cv2.namedWindow("Label Keypoints", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Label Keypoints", 1280, 720)
        cv2.setMouseCallback("Label Keypoints", self._on_mouse)

        while True:
            display = self._draw_overlay()
            cv2.imshow("Label Keypoints", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:  # Quit
                cv2.destroyAllWindows()
                return None

            elif key == ord("z"):  # Undo
                self._undo()

            elif key == ord("n"):  # Skip current keypoint
                if self.current_idx < len(KEYPOINT_ORDER):
                    name = KEYPOINT_ORDER[self.current_idx]
                    logger.info("Skipped: %s", name)
                    self.current_idx += 1

            elif key == 13:  # Enter = finish
                if len(self.labeled) >= 4:
                    cv2.destroyAllWindows()
                    return self.labeled
                else:
                    logger.warning("Need at least 4 keypoints, have %d", len(self.labeled))

            # Handle click
            if self.click_pos is not None and self.current_idx < len(KEYPOINT_ORDER):
                name = KEYPOINT_ORDER[self.current_idx]
                self.labeled[name] = (float(self.click_pos[0]), float(self.click_pos[1]))
                logger.info("Placed: %s at (%d, %d)", name, self.click_pos[0], self.click_pos[1])
                self.current_idx += 1
                self.click_pos = None

            # Auto-finish if all 12 placed
            if self.current_idx >= len(KEYPOINT_ORDER):
                cv2.destroyAllWindows()
                return self.labeled

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_pos = (x, y)

    def _undo(self) -> None:
        if self.current_idx > 0:
            self.current_idx -= 1
            name = KEYPOINT_ORDER[self.current_idx]
            # Find the last placed point to undo
            # It might have been skipped, so walk back
            while self.current_idx >= 0:
                name = KEYPOINT_ORDER[self.current_idx]
                if name in self.labeled:
                    del self.labeled[name]
                    logger.info("Undone: %s", name)
                    break
                self.current_idx -= 1
                if self.current_idx < 0:
                    self.current_idx = 0
                    break

    def _draw_overlay(self) -> np.ndarray:
        display = self.original.copy()

        # Draw already-placed keypoints
        for name, (px, py) in self.labeled.items():
            color = KEYPOINT_COLORS[name]
            ix, iy = int(px), int(py)
            cv2.circle(display, (ix, iy), 6, color, -1)
            cv2.circle(display, (ix, iy), 8, (255, 255, 255), 1)

            meters = KEYPOINT_METERS[name]
            label = f"{name} ({meters[0]},{meters[1]})"
            cv2.putText(display, label, (ix + 10, iy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Current keypoint prompt
        if self.current_idx < len(KEYPOINT_ORDER):
            name = KEYPOINT_ORDER[self.current_idx]
            meters = KEYPOINT_METERS[name]
            prompt = f"Click: {name}  ({meters[0]}m, {meters[1]}m)  [{self.current_idx + 1}/12]"
            color = KEYPOINT_COLORS[name]
        else:
            prompt = "All keypoints placed! Press Enter to confirm."
            color = (0, 255, 0)

        # Prompt bar at top
        cv2.rectangle(display, (0, 0), (self.frame_w, 40), (0, 0, 0), -1)
        cv2.putText(display, prompt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Status bar at bottom
        status = f"Labeled: {len(self.labeled)}/12  |  Z: undo  N: skip  Enter: finish  Q: quit"
        cv2.rectangle(display, (0, self.frame_h - 35), (self.frame_w, self.frame_h), (0, 0, 0), -1)
        cv2.putText(display, status, (10, self.frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return display


def verify_calibration(
    frame: np.ndarray,
    calibration_data: dict,
) -> None:
    """Show the frame with all 12 keypoints projected back for visual verification."""
    H = np.array(calibration_data["homography_matrix"])
    H_inv = np.linalg.inv(H)

    display = frame.copy()

    # Project all 12 court keypoints to pixel space
    for name, (mx, my) in KEYPOINT_METERS.items():
        pt = np.array([[[mx, my]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(pt, H_inv)
        px, py = int(projected[0, 0, 0]), int(projected[0, 0, 1])
        color = KEYPOINT_COLORS.get(name, (255, 255, 255))

        cv2.circle(display, (px, py), 5, color, -1)
        cv2.circle(display, (px, py), 7, (255, 255, 255), 1)
        cv2.putText(display, name, (px + 8, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Draw court lines
    for kp_a, kp_b in COURT_MODEL.LINES:
        ma = KEYPOINT_METERS[kp_a]
        mb = KEYPOINT_METERS[kp_b]
        pa = cv2.perspectiveTransform(np.array([[[ma[0], ma[1]]]], dtype=np.float64), H_inv)
        pb = cv2.perspectiveTransform(np.array([[[mb[0], mb[1]]]], dtype=np.float64), H_inv)
        pt_a = (int(pa[0, 0, 0]), int(pa[0, 0, 1]))
        pt_b = (int(pb[0, 0, 0]), int(pb[0, 0, 1]))
        cv2.line(display, pt_a, pt_b, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
    msg = f"Verification  |  Reprojection error: {calibration_data['reprojection_error']:.4f}m  |  Press any key to close"
    cv2.putText(display, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.namedWindow("Verify Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Verify Calibration", 1280, 720)
    cv2.imshow("Verify Calibration", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive manual court calibration tool."
    )
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output JSON path (default: <video_stem>_calibration.json in current dir)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        logger.error("Video not found: %s", args.video)
        sys.exit(1)

    output_path = args.output or Path(f"{args.video.stem}_calibration.json")

    # Phase 1: Select a good frame
    logger.info("Phase 1: Select a frame with good court visibility")
    selector = FrameSelector(args.video)
    result = selector.run()
    selector.release()

    if result is None:
        logger.info("Cancelled.")
        return
    frame, frame_id = result
    h, w = frame.shape[:2]
    logger.info("Selected frame %d (%dx%d)", frame_id, w, h)

    # Phase 2: Label keypoints
    logger.info("Phase 2: Click on court keypoints")
    labeler = KeypointLabeler(frame)
    keypoints = labeler.run()

    if keypoints is None:
        logger.info("Cancelled.")
        return

    logger.info("Labeled %d keypoints", len(keypoints))

    # Phase 3: Compute calibration
    try:
        calibration = CourtDetector.manual_calibration(
            keypoints_px=keypoints,
            frame_width=w,
            frame_height=h,
        )
    except ValueError as e:
        logger.error("Calibration failed: %s", e)
        sys.exit(1)

    logger.info("Reprojection error: %.4f meters", calibration.reprojection_error or -1)

    # Save to JSON
    cal_dict = calibration.model_dump()
    cal_dict["source_video"] = str(args.video)
    cal_dict["source_frame_id"] = frame_id
    cal_dict["labeled_keypoints"] = {k: list(v) for k, v in keypoints.items()}

    with open(output_path, "w") as f:
        json.dump(cal_dict, f, indent=2)
    logger.info("Saved calibration: %s", output_path)

    # Phase 4: Visual verification
    logger.info("Phase 3: Verify — court lines projected onto frame")
    verify_calibration(frame, cal_dict)

    # Print usage hint
    print("\n--- How to use this calibration ---")
    print(f"""
from padex.schemas.tracking import CourtCalibration
from padex.tracking.pipeline import TrackingPipeline
import json

with open("{output_path}") as f:
    cal = CourtCalibration(**json.load(f))

pipeline = TrackingPipeline(
    video_path="{args.video}",
    manual_calibration=cal,
)
result = pipeline.run()
""")


if __name__ == "__main__":
    main()
