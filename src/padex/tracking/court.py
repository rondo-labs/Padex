"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: court.py
Description:
    Court detection and homography calibration.
    Detects padel court lines and computes pixel-to-meter homography.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from padex.schemas.tracking import CourtCalibration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class KeypointDetectionResult:
    """Output of a keypoint detection strategy."""

    keypoints_px: list[tuple[float, float]]
    keypoints_m: list[tuple[float, float]]
    confidence: float = 0.0
    debug_frame: np.ndarray | None = None


@dataclass(frozen=True)
class PadelCourtModel:
    """Standard padel court reference geometry (20m x 10m).

    Origin: bottom-left corner.  x: 0-10m (width), y: 0-20m (length).
    Net at y = 10.0.
    """

    COURT_WIDTH: float = 10.0
    COURT_LENGTH: float = 20.0
    NET_Y: float = 10.0
    SERVICE_NEAR_Y: float = 3.0
    SERVICE_FAR_Y: float = 17.0

    KEYPOINTS: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        # Outer corners
        "bottom_left": (0.0, 0.0),
        "bottom_right": (10.0, 0.0),
        "top_left": (0.0, 20.0),
        "top_right": (10.0, 20.0),
        # Net-sideline intersections
        "net_left": (0.0, 10.0),
        "net_right": (10.0, 10.0),
        # Service line — near side (y=3.0, 7m from net)
        "service_near_left": (0.0, 3.0),
        "service_near_center": (5.0, 3.0),
        "service_near_right": (10.0, 3.0),
        # Service line — far side (y=17.0, 7m from net)
        "service_far_left": (0.0, 17.0),
        "service_far_center": (5.0, 17.0),
        "service_far_right": (10.0, 17.0),
    })

    LINES: list[tuple[str, str]] = field(default_factory=lambda: [
        # Baselines
        ("bottom_left", "bottom_right"),
        ("top_left", "top_right"),
        # Sidelines
        ("bottom_left", "top_left"),
        ("bottom_right", "top_right"),
        # Net
        ("net_left", "net_right"),
        # Service lines
        ("service_near_left", "service_near_right"),
        ("service_far_left", "service_far_right"),
        # Center service line
        ("service_near_center", "service_far_center"),
    ])


COURT_MODEL = PadelCourtModel()


# ---------------------------------------------------------------------------
# Abstract keypoint detector interface
# ---------------------------------------------------------------------------


class KeypointDetector(abc.ABC):
    """Abstract interface for court keypoint detection.

    Implementations detect court keypoints in a video frame and return
    matched pixel/meter coordinate pairs.
    """

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> KeypointDetectionResult | None:
        """Detect court keypoints in a single frame.

        Returns None if no valid court is detected.
        """
        ...

    @abc.abstractmethod
    def is_court_visible(self, frame: np.ndarray) -> bool:
        """Quick check: does this frame show a playable court view?"""
        ...


# ---------------------------------------------------------------------------
# Traditional CV implementation
# ---------------------------------------------------------------------------


class HoughLineKeypointDetector(KeypointDetector):
    """Court keypoint detector using Canny edges + Hough lines + intersections."""

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 80,
        hough_min_line_length: int = 80,
        hough_max_line_gap: int = 15,
        min_keypoints: int = 4,
        white_lower_hsv: tuple[int, int, int] = (0, 0, 180),
        white_upper_hsv: tuple[int, int, int] = (180, 60, 255),
        angle_tolerance_deg: float = 30.0,
        cluster_distance_px: float = 25.0,
        intersection_merge_px: float = 20.0,
        roi_margin_top: float = 0.12,
        roi_margin_bottom: float = 0.08,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap
        self.min_keypoints = min_keypoints
        self.white_lower = np.array(white_lower_hsv)
        self.white_upper = np.array(white_upper_hsv)
        self.angle_tolerance_deg = angle_tolerance_deg
        self.cluster_distance_px = cluster_distance_px
        self.intersection_merge_px = intersection_merge_px
        self.roi_margin_top = roi_margin_top
        self.roi_margin_bottom = roi_margin_bottom

    # -- public interface --------------------------------------------------

    def detect(self, frame: np.ndarray) -> KeypointDetectionResult | None:
        h, w = frame.shape[:2]

        # Stage 1-2: preprocess and isolate white court lines
        blurred = self._preprocess(frame)
        edge_mask = self._isolate_court_lines(blurred, h, w)

        # Stage 3: detect line segments
        lines = self._detect_lines(edge_mask)
        if lines is None:
            return None

        # Stage 4: cluster into width / length direction groups
        w_lines, l_lines = self._cluster_lines(lines)
        if len(w_lines) < 2 or len(l_lines) < 2:
            return None

        # Stage 5: find intersections
        intersections = self._find_intersections(w_lines, l_lines, w, h)
        if len(intersections) < self.min_keypoints:
            return None

        # Stage 6: match to court model
        px_coords, m_coords = self._match_keypoints_to_court(intersections, (h, w))
        if len(px_coords) < self.min_keypoints:
            return None

        confidence = min(1.0, len(px_coords) / 8.0)

        return KeypointDetectionResult(
            keypoints_px=px_coords,
            keypoints_m=m_coords,
            confidence=confidence,
        )

    def is_court_visible(self, frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]

        # Crop to central region (avoid scoreboards / ads)
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.10), int(w * 0.90)
        roi = frame[y1:y2, x1:x2]

        # Check 1: dominant court color (blue or green)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([130, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        court_color_ratio = (
            np.count_nonzero(blue_mask) + np.count_nonzero(green_mask)
        ) / (roi.shape[0] * roi.shape[1])

        if court_color_ratio < 0.15:
            return False

        # Check 2: white line pixel density
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        white_ratio = np.count_nonzero(white_mask) / (roi.shape[0] * roi.shape[1])
        if white_ratio < 0.003 or white_ratio > 0.15:
            return False

        # Check 3: enough long straight lines
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=10
        )
        if lines is None or len(lines) < 3:
            return False

        return True

    # -- pipeline stages (private) -----------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Stage 1: Gaussian blur to reduce noise."""
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def _isolate_court_lines(
        self, frame: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Stage 2: HSV white thresholding + morphology + Canny."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)

        # Morphological close to bridge small gaps in lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # Apply ROI to exclude scoreboard areas
        roi_top = int(height * self.roi_margin_top)
        roi_bottom = int(height * (1 - self.roi_margin_bottom))
        mask = np.zeros_like(white_mask)
        mask[roi_top:roi_bottom, :] = white_mask[roi_top:roi_bottom, :]

        edges = cv2.Canny(mask, self.canny_low, self.canny_high)
        return edges

    def _detect_lines(self, edge_mask: np.ndarray) -> np.ndarray | None:
        """Stage 3: Probabilistic Hough transform."""
        lines = cv2.HoughLinesP(
            edge_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )
        if lines is None or len(lines) < 4:
            return None
        return lines.reshape(-1, 4)

    def _cluster_lines(
        self, lines: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Stage 4: Separate into width-direction and length-direction groups.

        In broadcast perspective:
        - Width lines (baselines, service lines, net) appear near-horizontal.
        - Length lines (sidelines, center line) appear as diagonals converging
          to a vanishing point — NOT vertical.  Any line with angle > threshold
          is treated as a length line.
        """
        width_lines: list[np.ndarray] = []
        length_lines: list[np.ndarray] = []

        for x1, y1, x2, y2 in lines:
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if angle < self.angle_tolerance_deg:
                width_lines.append(np.array([x1, y1, x2, y2]))
            elif angle > self.angle_tolerance_deg:
                length_lines.append(np.array([x1, y1, x2, y2]))

        w_clusters = self._merge_parallel_lines(width_lines, axis="width")
        l_clusters = self._merge_parallel_lines(length_lines, axis="length")
        return w_clusters, l_clusters

    def _merge_parallel_lines(
        self, lines: list[np.ndarray], axis: str
    ) -> list[np.ndarray]:
        """Merge lines that are close and parallel into representative lines."""
        if not lines:
            return []

        if axis == "width":
            # Cluster by y midpoint
            lines.sort(key=lambda l: (l[1] + l[3]) / 2)
        else:
            # Cluster length-direction lines by x midpoint
            lines.sort(key=lambda l: (l[0] + l[2]) / 2)

        clusters: list[list[np.ndarray]] = [[lines[0]]]
        for line in lines[1:]:
            if axis == "width":
                ref = np.mean([l[1] + l[3] for l in clusters[-1]]) / 2
                val = (line[1] + line[3]) / 2
            else:
                ref = np.mean([l[0] + l[2] for l in clusters[-1]]) / 2
                val = (line[0] + line[2]) / 2

            if abs(val - ref) < self.cluster_distance_px:
                clusters[-1].append(line)
            else:
                clusters.append([line])

        # Average each cluster into one representative line
        result = []
        for cluster in clusters:
            arr = np.array(cluster)
            result.append(arr.mean(axis=0).astype(np.float64))
        return result

    def _find_intersections(
        self,
        h_lines: list[np.ndarray],
        v_lines: list[np.ndarray],
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[float, float]]:
        """Stage 5: Compute H×V line intersections within frame bounds."""
        intersections: list[tuple[float, float]] = []

        for hl in h_lines:
            for vl in v_lines:
                pt = self._line_intersection(hl, vl)
                if pt is None:
                    continue
                x, y = pt
                # Allow small margin outside frame
                margin = 10
                if -margin <= x <= frame_w + margin and -margin <= y <= frame_h + margin:
                    intersections.append((float(x), float(y)))

        # Merge nearby intersections
        merged = self._merge_nearby_points(intersections)
        return merged

    @staticmethod
    def _line_intersection(
        line1: np.ndarray, line2: np.ndarray
    ) -> tuple[float, float] | None:
        """Compute intersection of two line segments (extended to full lines)."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None

        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        t = t_num / denom

        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (float(ix), float(iy))

    def _merge_nearby_points(
        self, points: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Merge points within merge distance into their centroid."""
        if not points:
            return []

        merged: list[tuple[float, float]] = []
        used = [False] * len(points)

        for i, (x1, y1) in enumerate(points):
            if used[i]:
                continue
            group_x, group_y, count = x1, y1, 1
            for j in range(i + 1, len(points)):
                if used[j]:
                    continue
                x2, y2 = points[j]
                dist = np.hypot(x1 - x2, y1 - y2)
                if dist < self.intersection_merge_px:
                    group_x += x2
                    group_y += y2
                    count += 1
                    used[j] = True
            used[i] = True
            merged.append((group_x / count, group_y / count))

        return merged

    def _match_keypoints_to_court(
        self,
        intersections: list[tuple[float, float]],
        frame_shape: tuple[int, int],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Stage 6: Match detected intersections to court model keypoints.

        Uses geometric sorting: group by rows (y), sort within rows (x),
        then assign to court keypoints based on grid structure.
        """
        frame_h, frame_w = frame_shape
        if len(intersections) < 4:
            return [], []

        # Sort by y (top to bottom in image = far to near on court)
        pts = sorted(intersections, key=lambda p: p[1])

        # Group into rows by y-proximity
        rows: list[list[tuple[float, float]]] = [[pts[0]]]
        for pt in pts[1:]:
            if abs(pt[1] - rows[-1][-1][1]) < self.cluster_distance_px:
                rows[-1].append(pt)
            else:
                rows.append([pt])

        # Sort each row by x (left to right)
        for row in rows:
            row.sort(key=lambda p: p[0])

        # Build candidate match based on row/column structure
        # Court horizontal lines from top of image (far side) to bottom (near side):
        # Possible: top_baseline, service_far, net, service_near, bottom_baseline
        court_h_lines_y = [20.0, 17.0, 10.0, 3.0, 0.0]
        # For each row, figure out which court line it likely corresponds to

        n_rows = len(rows)
        if n_rows < 2:
            return [], []

        # Pick the best assignment of rows to court horizontal lines
        # using vertical spacing ratios
        best_px: list[tuple[float, float]] = []
        best_m: list[tuple[float, float]] = []
        best_score = -1

        # Try all combinations of selecting n_rows lines from 5 possible
        from itertools import combinations

        for combo in combinations(range(5), min(n_rows, 5)):
            if len(combo) < 2:
                continue

            px_coords: list[tuple[float, float]] = []
            m_coords: list[tuple[float, float]] = []

            for row_idx, line_idx in enumerate(combo):
                if row_idx >= len(rows):
                    break
                row = rows[row_idx]
                court_y = court_h_lines_y[line_idx]

                # Assign x positions within this row
                row_px, row_m = self._assign_row_x(row, court_y)
                px_coords.extend(row_px)
                m_coords.extend(row_m)

            if len(px_coords) < self.min_keypoints:
                continue

            # Score: check spacing consistency
            score = self._score_assignment(px_coords, m_coords)
            if score > best_score:
                best_score = score
                best_px = px_coords
                best_m = m_coords

        return best_px, best_m

    def _assign_row_x(
        self, row: list[tuple[float, float]], court_y: float
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Assign x-coordinates within a row to court keypoint x-positions."""
        # Possible x positions on court: 0.0 (left), 5.0 (center), 10.0 (right)
        # Center line only exists between service lines (y=7 and y=13)
        has_center = court_y in (3.0, 17.0)

        px_list: list[tuple[float, float]] = []
        m_list: list[tuple[float, float]] = []

        n = len(row)
        if n == 1:
            # Can't determine which sideline — skip
            return [], []
        elif n == 2:
            # Left and right sideline
            px_list = [row[0], row[1]]
            m_list = [(0.0, court_y), (10.0, court_y)]
        elif n == 3 and has_center:
            # Left, center, right
            px_list = [row[0], row[1], row[2]]
            m_list = [
                (0.0, court_y),
                (5.0, court_y),
                (10.0, court_y),
            ]
        elif n >= 3:
            # Take leftmost and rightmost as sidelines
            px_list = [row[0], row[-1]]
            m_list = [(0.0, court_y), (10.0, court_y)]

        return px_list, m_list

    @staticmethod
    def _score_assignment(
        px_coords: list[tuple[float, float]],
        m_coords: list[tuple[float, float]],
    ) -> float:
        """Score a candidate keypoint assignment by homography consistency.

        A good assignment yields a low reprojection error.
        Returns a score where higher is better.
        """
        if len(px_coords) < 4:
            return -1.0

        src = np.array(px_coords, dtype=np.float64)
        dst = np.array(m_coords, dtype=np.float64)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return -1.0

        inlier_count = int(mask.sum()) if mask is not None else 0
        if inlier_count < 4:
            return -1.0

        # Reprojection error on inliers
        inlier_mask = mask.ravel().astype(bool)
        src_in = src[inlier_mask]
        dst_in = dst[inlier_mask]
        projected = cv2.perspectiveTransform(
            src_in.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - dst_in) ** 2, axis=1))))

        # Score: more inliers and lower error is better
        if error < 1e-6:
            return float(inlier_count) * 100.0
        return float(inlier_count) / error


# ---------------------------------------------------------------------------
# CourtDetector facade
# ---------------------------------------------------------------------------


class CourtDetector:
    """Main court detection facade.

    Combines keypoint detection (pluggable strategy) with homography math.
    """

    def __init__(self, keypoint_detector: KeypointDetector | None = None) -> None:
        self.keypoint_detector = keypoint_detector or HoughLineKeypointDetector()

    def detect_keypoints(self, frame: np.ndarray) -> KeypointDetectionResult | None:
        """Delegate to the configured keypoint detector."""
        return self.keypoint_detector.detect(frame)

    def is_court_visible(self, frame: np.ndarray) -> bool:
        """Delegate to the configured keypoint detector."""
        return self.keypoint_detector.is_court_visible(frame)

    def compute_homography(
        self,
        keypoints_px: list[tuple[float, float]],
        keypoints_m: list[tuple[float, float]],
    ) -> tuple[np.ndarray, float]:
        """Compute homography matrix with RANSAC.

        Returns:
            (H, reprojection_error) where H is the 3x3 homography matrix.
        """
        src = np.array(keypoints_px, dtype=np.float64)
        dst = np.array(keypoints_m, dtype=np.float64)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            raise ValueError("Homography computation failed — not enough inliers")

        # Reprojection error on inliers
        inlier_mask = mask.ravel().astype(bool)
        src_in = src[inlier_mask]
        dst_in = dst[inlier_mask]

        projected = cv2.perspectiveTransform(
            src_in.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - dst_in) ** 2, axis=1))))

        return H, error

    def pixel_to_court(
        self, point_px: tuple[float, float], homography_matrix: np.ndarray
    ) -> tuple[float, float]:
        """Transform a single pixel point to court coordinates (meters)."""
        pt = np.array([[point_px]], dtype=np.float64)
        result = cv2.perspectiveTransform(pt, homography_matrix)
        x, y = result[0, 0]
        return (float(x), float(y))

    def court_to_pixel(
        self, point_m: tuple[float, float], homography_matrix: np.ndarray
    ) -> tuple[float, float]:
        """Transform court coordinates (meters) to pixel coordinates."""
        H_inv = np.linalg.inv(homography_matrix)
        pt = np.array([[point_m]], dtype=np.float64)
        result = cv2.perspectiveTransform(pt, H_inv)
        x, y = result[0, 0]
        return (float(x), float(y))

    def calibrate_frame(self, frame: np.ndarray) -> CourtCalibration | None:
        """Full pipeline: frame -> CourtCalibration or None."""
        result = self.detect_keypoints(frame)
        if result is None or len(result.keypoints_px) < 4:
            logger.debug("Not enough keypoints detected")
            return None

        try:
            H, error = self.compute_homography(
                result.keypoints_px, result.keypoints_m
            )
        except ValueError:
            logger.debug("Homography computation failed")
            return None

        h, w = frame.shape[:2]

        if not self._validate_homography(H, w, h):
            logger.debug("Homography validation failed")
            return None

        return CourtCalibration(
            frame_width=w,
            frame_height=h,
            homography_matrix=H.tolist(),
            court_keypoints_px=list(result.keypoints_px),
            court_keypoints_m=list(result.keypoints_m),
            reprojection_error=error,
        )

    @staticmethod
    def manual_calibration(
        keypoints_px: dict[str, tuple[float, float]],
        frame_width: int,
        frame_height: int,
    ) -> CourtCalibration:
        """Create a CourtCalibration from manually annotated keypoints.

        Use this when the broadcast camera is fixed and you prefer to mark
        court keypoints once rather than relying on automatic detection.

        Args:
            keypoints_px: Mapping of keypoint name to pixel coordinates.
                Must contain at least 4 keypoints that exist in
                PadelCourtModel.KEYPOINTS (e.g. ``{"bottom_left": (100, 900),
                "bottom_right": (1800, 900), ...}``).
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.

        Returns:
            A CourtCalibration ready to pass to TrackingPipeline.

        Raises:
            ValueError: If fewer than 4 valid keypoints are provided or
                homography computation fails.
        """
        court_kps = COURT_MODEL.KEYPOINTS
        px_coords: list[tuple[float, float]] = []
        m_coords: list[tuple[float, float]] = []

        for name, px in keypoints_px.items():
            if name not in court_kps:
                logger.warning("Unknown keypoint '%s', skipping", name)
                continue
            px_coords.append(px)
            m_coords.append(court_kps[name])

        if len(px_coords) < 4:
            raise ValueError(
                f"Need at least 4 keypoints, got {len(px_coords)}. "
                f"Valid names: {list(court_kps.keys())}"
            )

        src = np.array(px_coords, dtype=np.float64)
        dst = np.array(m_coords, dtype=np.float64)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            raise ValueError("Homography computation failed")

        # Reprojection error
        inlier_mask = mask.ravel().astype(bool)
        src_in = src[inlier_mask]
        dst_in = dst[inlier_mask]
        projected = cv2.perspectiveTransform(
            src_in.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - dst_in) ** 2, axis=1))))

        return CourtCalibration(
            frame_width=frame_width,
            frame_height=frame_height,
            homography_matrix=H.tolist(),
            court_keypoints_px=list(px_coords),
            court_keypoints_m=list(m_coords),
            reprojection_error=error,
        )

    @staticmethod
    def _validate_homography(
        H: np.ndarray,
        frame_w: int,
        frame_h: int,
        max_reproj_error: float = 2.0,
    ) -> bool:
        """Sanity-check a computed homography matrix."""
        # Determinant should be non-zero (not degenerate)
        det = abs(np.linalg.det(H))
        if det < 1e-10 or det > 1e12:
            return False

        # Instead of projecting frame corners (which go off-court in
        # broadcast views), project known court corners to pixel space
        # and verify they land in a plausible region.
        H_inv = np.linalg.inv(H)
        court_corners = np.array(
            [[[0, 0]], [[10, 0]], [[10, 20]], [[0, 20]]],
            dtype=np.float64,
        )
        px_corners = cv2.perspectiveTransform(court_corners, H_inv).reshape(-1, 2)

        # Court corners should project somewhere near the frame
        margin = max(frame_w, frame_h) * 0.5
        for x, y in px_corners:
            if x < -margin or x > frame_w + margin:
                return False
            if y < -margin or y > frame_h + margin:
                return False

        # Check that projected court corners form a convex quadrilateral
        # (no self-intersection / fold-over)
        v1 = px_corners[1] - px_corners[0]
        v2 = px_corners[2] - px_corners[1]
        v3 = px_corners[3] - px_corners[2]
        v4 = px_corners[0] - px_corners[3]
        crosses = [
            v1[0] * v2[1] - v1[1] * v2[0],
            v2[0] * v3[1] - v2[1] * v3[0],
            v3[0] * v4[1] - v3[1] * v4[0],
            v4[0] * v1[1] - v4[1] * v1[0],
        ]
        # All cross products should have the same sign
        if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
            return False

        return True
