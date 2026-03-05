"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: mini_court.py
Description:
    Mini court overlay for video frames.
    Draws a scaled-down 2D padel court in a corner of the video frame,
    showing real-time player and ball positions.
"""

from __future__ import annotations

import cv2
import numpy as np

from padex.schemas.tracking import BallFrame, BallVisibility, PlayerFrame


# Team colors (BGR for OpenCV)
_TEAM_COLORS = {
    "T_1": (229, 136, 30),   # blue (#1E88E5)
    "T_2": (53, 57, 229),    # red (#E53935)
}
_DEFAULT_COLOR = (200, 200, 200)
_BALL_COLOR = (59, 235, 255)  # yellow (#FFEB3B)

# Court dimensions in meters
_COURT_W = 10.0
_COURT_L = 20.0
_NET_Y = 10.0
_SERVICE_NEAR_Y = 3.0
_SERVICE_FAR_Y = 17.0
_CENTER_X = 5.0


class MiniCourt:
    """Draws a miniature 2D padel court on a video frame."""

    def __init__(
        self,
        width_px: int = 150,
        height_px: int = 300,
        margin: int = 10,
    ) -> None:
        self.width = width_px
        self.height = height_px
        self.margin = margin

    def _court_to_mini(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Convert court meters (0-10, 0-20) to mini court pixel coords."""
        px = int(x_m / _COURT_W * self.width)
        py = int((1.0 - y_m / _COURT_L) * self.height)  # flip y
        return px, py

    def draw(
        self,
        frame: np.ndarray,
        player_frames: list[PlayerFrame],
        ball_frame: BallFrame | None = None,
        position: str = "bottom_right",
    ) -> None:
        """Draw the mini court overlay on the frame (in-place).

        Args:
            frame: Video frame (BGR, HWC).
            player_frames: Player tracking data for this frame.
            ball_frame: Ball tracking data for this frame (optional).
            position: Corner placement — "bottom_right" or "top_right".
        """
        h, w = frame.shape[:2]
        total_w = self.width + 2 * self.margin
        total_h = self.height + 2 * self.margin

        if position == "top_right":
            x0 = w - total_w - 5
            y0 = 5
        else:  # bottom_right
            x0 = w - total_w - 5
            y0 = h - total_h - 5

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x0 + total_w, y0 + total_h),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Offset for court area inside the margin
        cx0 = x0 + self.margin
        cy0 = y0 + self.margin

        # Court background (green)
        cv2.rectangle(
            frame,
            (cx0, cy0),
            (cx0 + self.width, cy0 + self.height),
            (30, 100, 30),
            -1,
        )

        # Draw court lines
        self._draw_lines(frame, cx0, cy0)

        # Draw players
        for pf in player_frames:
            if pf.position is None:
                continue
            px, py = self._court_to_mini(pf.position.x, pf.position.y)
            color = _TEAM_COLORS.get(pf.team_id, _DEFAULT_COLOR)
            cv2.circle(frame, (cx0 + px, cy0 + py), 5, color, -1)
            cv2.circle(frame, (cx0 + px, cy0 + py), 5, (255, 255, 255), 1)

        # Draw ball
        if ball_frame and ball_frame.position and ball_frame.visibility == BallVisibility.VISIBLE:
            bx, by = self._court_to_mini(
                ball_frame.position.x, ball_frame.position.y,
            )
            cv2.circle(frame, (cx0 + bx, cy0 + by), 4, _BALL_COLOR, -1)

    def _draw_lines(self, frame: np.ndarray, cx0: int, cy0: int) -> None:
        """Draw court boundary, net, service lines, center line."""
        white = (255, 255, 255)
        gray = (180, 180, 180)

        def _line(x1m: float, y1m: float, x2m: float, y2m: float, color=white, thickness=1):
            p1x, p1y = self._court_to_mini(x1m, y1m)
            p2x, p2y = self._court_to_mini(x2m, y2m)
            cv2.line(frame, (cx0 + p1x, cy0 + p1y), (cx0 + p2x, cy0 + p2y), color, thickness)

        # Court boundary
        cv2.rectangle(
            frame, (cx0, cy0), (cx0 + self.width, cy0 + self.height), white, 1,
        )

        # Net (dashed effect — just draw thicker gray)
        _line(0, _NET_Y, _COURT_W, _NET_Y, gray, 2)

        # Service lines
        _line(0, _SERVICE_NEAR_Y, _COURT_W, _SERVICE_NEAR_Y)
        _line(0, _SERVICE_FAR_Y, _COURT_W, _SERVICE_FAR_Y)

        # Center service lines
        _line(_CENTER_X, _SERVICE_NEAR_Y, _CENTER_X, _NET_Y)
        _line(_CENTER_X, _NET_Y, _CENTER_X, _SERVICE_FAR_Y)
