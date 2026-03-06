"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: frame.py
Description:
    Frame annotation module for video output.
    Draws player bboxes, ball markers, court lines, mini court,
    stats panel, and shot labels on video frames using OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np

from padex.schemas.events import Shot
from padex.schemas.tracking import (
    BallFrame,
    BallVisibility,
    CourtCalibration,
    PlayerFrame,
)
from padex.tracking.court import COURT_MODEL
from padex.viz.mini_court import MiniCourt

# Team colors (BGR)
_TEAM_COLORS = {
    "T_1": (229, 136, 30),   # blue
    "T_2": (53, 57, 229),    # red
}
_DEFAULT_TEAM_COLOR = (200, 200, 200)

# Ball colors by visibility (BGR)
_BALL_COLORS = {
    BallVisibility.VISIBLE: (59, 235, 255),    # yellow
    BallVisibility.OCCLUDED: (128, 128, 128),   # gray
    BallVisibility.INFERRED: (255, 165, 0),     # orange
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

# COCO skeleton connections (pairs of keypoint indices)
_COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
]

# COCO keypoint names (index order)
_COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Colors per skeleton region (BGR)
_SKELETON_COLORS = {
    (0, 1): (255, 200, 50),   (0, 2): (255, 200, 50),   # head - cyan
    (1, 3): (255, 200, 50),   (2, 4): (255, 200, 50),
    (5, 6): (0, 255, 0),                                   # shoulders - green
    (5, 7): (0, 200, 255),    (7, 9): (0, 200, 255),       # left arm - orange
    (6, 8): (255, 0, 200),    (8, 10): (255, 0, 200),      # right arm - magenta
    (5, 11): (0, 255, 0),     (6, 12): (0, 255, 0),        # torso - green
    (11, 12): (0, 255, 0),
    (11, 13): (0, 255, 255),  (13, 15): (0, 255, 255),     # left leg - yellow
    (12, 14): (255, 255, 0),  (14, 16): (255, 255, 0),     # right leg - cyan
}


class FrameAnnotator:
    """All OpenCV frame annotation logic in one place."""

    def __init__(self) -> None:
        self._mini_court = MiniCourt()

    def draw_player_bboxes(
        self,
        frame: np.ndarray,
        player_frames: list[PlayerFrame],
    ) -> None:
        """Draw bounding boxes and player IDs."""
        for pf in player_frames:
            color = _TEAM_COLORS.get(pf.team_id, _DEFAULT_TEAM_COLOR)
            x1, y1 = int(pf.bbox.x1), int(pf.bbox.y1)
            x2, y2 = int(pf.bbox.x2), int(pf.bbox.y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = pf.player_id
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), _FONT, 0.5, (255, 255, 255), 1)

    def draw_pose_keypoints(
        self,
        frame: np.ndarray,
        player_frames: list[PlayerFrame],
    ) -> None:
        """Draw pose keypoints and skeleton connections for each player."""
        for pf in player_frames:
            if not pf.keypoints:
                continue

            # Build name→(x, y, conf) lookup
            kp_map: dict[str, tuple[int, int, float]] = {}
            for kp in pf.keypoints:
                kp_map[kp.name] = (int(kp.x), int(kp.y), kp.confidence)

            # Index-based lookup
            kp_by_idx: dict[int, tuple[int, int]] = {}
            for idx, name in enumerate(_COCO_KEYPOINT_NAMES):
                if name in kp_map:
                    x, y, conf = kp_map[name]
                    if conf > 0.3:
                        kp_by_idx[idx] = (x, y)

            # Draw skeleton lines
            for i, j in _COCO_SKELETON:
                if i in kp_by_idx and j in kp_by_idx:
                    color = _SKELETON_COLORS.get((i, j), (200, 200, 200))
                    cv2.line(frame, kp_by_idx[i], kp_by_idx[j], color, 2, cv2.LINE_AA)

            # Draw keypoint circles
            for idx, (x, y) in kp_by_idx.items():
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 4, (0, 0, 0), 1)

    def draw_ball(
        self,
        frame: np.ndarray,
        ball_frame: BallFrame | None,
    ) -> None:
        """Draw ball marker at bbox center."""
        if ball_frame is None or ball_frame.bbox is None:
            return

        cx = int((ball_frame.bbox.x1 + ball_frame.bbox.x2) / 2)
        cy = int((ball_frame.bbox.y1 + ball_frame.bbox.y2) / 2)
        color = _BALL_COLORS.get(ball_frame.visibility, (59, 235, 255))

        cv2.circle(frame, (cx, cy), 8, color, -1)
        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1)

    def draw_court_lines(
        self,
        frame: np.ndarray,
        calibration: CourtCalibration | None,
    ) -> None:
        """Draw court lines projected onto the video frame."""
        if calibration is None:
            return

        H = np.array(calibration.homography_matrix)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return

        kp = COURT_MODEL.KEYPOINTS
        for name1, name2 in COURT_MODEL.LINES:
            if name1 not in kp or name2 not in kp:
                continue
            p1_m = np.array([[kp[name1]]], dtype=np.float64)
            p2_m = np.array([[kp[name2]]], dtype=np.float64)
            p1_px = cv2.perspectiveTransform(p1_m, H_inv)[0][0]
            p2_px = cv2.perspectiveTransform(p2_m, H_inv)[0][0]
            cv2.line(
                frame,
                (int(p1_px[0]), int(p1_px[1])),
                (int(p2_px[0]), int(p2_px[1])),
                (0, 255, 0),
                2,
            )

    def draw_court_keypoints(
        self,
        frame: np.ndarray,
        calibration: CourtCalibration | None,
    ) -> None:
        """Draw detected court keypoints on the frame."""
        if calibration is None:
            return

        for px, py in calibration.court_keypoints_px:
            cv2.circle(frame, (int(px), int(py)), 5, (0, 255, 255), -1)

    def draw_mini_court(
        self,
        frame: np.ndarray,
        player_frames: list[PlayerFrame],
        ball_frame: BallFrame | None = None,
    ) -> None:
        """Draw the mini court overlay."""
        self._mini_court.draw(frame, player_frames, ball_frame)

    def draw_stats_panel(
        self,
        frame: np.ndarray,
        stats: dict[str, str | int | float],
    ) -> None:
        """Draw a semi-transparent stats panel in the top-left corner."""
        if not stats:
            return

        lines = [f"{k}: {v}" for k, v in stats.items()]
        line_h = 22
        panel_h = len(lines) * line_h + 20
        panel_w = 250

        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(lines):
            y = 25 + i * line_h
            cv2.putText(frame, line, (15, y), _FONT, 0.5, (255, 255, 255), 1)

    def draw_shot_label(
        self,
        frame: np.ndarray,
        shot: Shot | None,
        ball_frame: BallFrame | None,
    ) -> None:
        """Show shot type label near the ball when a shot is detected."""
        if shot is None or ball_frame is None or ball_frame.bbox is None:
            return

        cx = int((ball_frame.bbox.x1 + ball_frame.bbox.x2) / 2)
        cy = int((ball_frame.bbox.y1 + ball_frame.bbox.y2) / 2) - 20
        label = shot.shot_type.value.upper()

        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.6, 2)
        cv2.rectangle(frame, (cx - 2, cy - th - 4), (cx + tw + 4, cy + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (cx, cy), _FONT, 0.6, (0, 255, 255), 2)

    def draw_frame_number(self, frame: np.ndarray, frame_id: int) -> None:
        """Draw frame number in the top-left corner."""
        text = f"Frame: {frame_id}"
        cv2.putText(frame, text, (10, 30), _FONT, 0.7, (255, 255, 255), 2)

    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        player_frames: list[PlayerFrame],
        ball_frame: BallFrame | None = None,
        calibration: CourtCalibration | None = None,
        shot: Shot | None = None,
        stats: dict[str, str | int | float] | None = None,
    ) -> np.ndarray:
        """One-stop method: apply all annotations and return the frame.

        Args:
            frame: Video frame (BGR, HWC). Modified in-place and returned.
            frame_id: Current frame index.
            player_frames: Player tracking data for this frame.
            ball_frame: Ball tracking data (optional).
            calibration: Court calibration for line overlay (optional).
            shot: Active shot event to display label (optional).
            stats: Stats dict to render in panel (optional).

        Returns:
            The annotated frame.
        """
        self.draw_court_lines(frame, calibration)
        self.draw_court_keypoints(frame, calibration)
        self.draw_player_bboxes(frame, player_frames)
        self.draw_pose_keypoints(frame, player_frames)
        self.draw_ball(frame, ball_frame)
        self.draw_shot_label(frame, shot, ball_frame)
        self.draw_mini_court(frame, player_frames, ball_frame)
        if stats:
            self.draw_stats_panel(frame, stats)
        self.draw_frame_number(frame, frame_id)
        return frame
