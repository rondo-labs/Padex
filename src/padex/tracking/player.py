"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: player.py
Description:
    Player detection, tracking, and team classification.
    Uses YOLO for detection, ByteTrack for tracking, and jersey color
    clustering for team assignment.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from padex.schemas.tracking import BoundingBox, PlayerFrame, PoseKeypoint, Position2D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RawDetection:
    """Single detection from YOLO before filtering or enrichment."""

    bbox: BoundingBox
    confidence: float
    track_id: int | None = None
    crop: np.ndarray | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class PlayerDetectionStrategy(abc.ABC):
    """Abstract interface for player detection backends."""

    @abc.abstractmethod
    def detect(
        self, frame: np.ndarray, frame_id: int, timestamp_ms: float
    ) -> list[RawDetection]:
        """Detect players in a single frame."""
        ...

    @abc.abstractmethod
    def detect_with_tracking(
        self, frame: np.ndarray, frame_id: int, timestamp_ms: float
    ) -> list[RawDetection]:
        """Detect + track players across frames (stateful)."""
        ...

    @abc.abstractmethod
    def reset_tracking(self) -> None:
        """Reset tracker state for a new video/scene."""
        ...


class TeamClassifier(abc.ABC):
    """Abstract interface for team classification."""

    @abc.abstractmethod
    def classify(
        self, detections: list[RawDetection], frame: np.ndarray
    ) -> dict[int, str]:
        """Assign team IDs to detections.

        Returns mapping from detection index to team_id string.
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset classifier state."""
        ...


class PoseEstimationStrategy(abc.ABC):
    """Abstract interface for pose estimation backends."""

    @abc.abstractmethod
    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[BoundingBox],
    ) -> list[list[PoseKeypoint]]:
        """Estimate pose keypoints for each bbox.

        Returns a list of keypoint lists (one per bbox, in same order).
        Empty list for a bbox if estimation failed.
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset estimator state."""
        ...


# ---------------------------------------------------------------------------
# YOLO detection strategy
# ---------------------------------------------------------------------------


class YoloPlayerDetectionStrategy(PlayerDetectionStrategy):
    """Player detection using ultralytics YOLO with ByteTrack tracking."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolo26m.pt",
        confidence_threshold: float = 0.5,
        device: str | None = None,
        imgsz: int = 1280,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.imgsz = imgsz
        self._model = self._load_model()

    def _load_model(self):
        from ultralytics import YOLO

        return YOLO(self.model_path)

    def detect(
        self, frame: np.ndarray, frame_id: int, timestamp_ms: float
    ) -> list[RawDetection]:
        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
            device=self.device,
            imgsz=self.imgsz,
        )
        return self._parse_results(results, frame)

    def detect_with_tracking(
        self, frame: np.ndarray, frame_id: int, timestamp_ms: float
    ) -> list[RawDetection]:
        results = self._model.track(
            frame,
            conf=self.confidence_threshold,
            classes=[self.PERSON_CLASS_ID],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device,
            imgsz=self.imgsz,
        )
        return self._parse_results(results, frame, with_tracking=True)

    def reset_tracking(self) -> None:
        self._model = self._load_model()

    def _parse_results(
        self, results, frame: np.ndarray, with_tracking: bool = False
    ) -> list[RawDetection]:
        detections: list[RawDetection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        track_ids = None
        if with_tracking and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)

        h, w = frame.shape[:2]
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            # Clamp to frame boundaries
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            crop = frame[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else None

            detections.append(
                RawDetection(
                    bbox=BoundingBox(
                        x1=float(xyxy[i][0]),
                        y1=float(xyxy[i][1]),
                        x2=float(xyxy[i][2]),
                        y2=float(xyxy[i][3]),
                    ),
                    confidence=float(confs[i]),
                    track_id=int(track_ids[i]) if track_ids is not None else None,
                    crop=crop,
                )
            )

        return detections


# ---------------------------------------------------------------------------
# Jersey color team classifier
# ---------------------------------------------------------------------------


class JerseyColorTeamClassifier(TeamClassifier):
    """Team classification via jersey color clustering in HSV space."""

    def __init__(
        self,
        n_histogram_bins: int = 16,
        upper_body_ratio: float = 0.45,
        n_warmup_frames: int = 10,
        min_crop_height: int = 20,
    ) -> None:
        self.n_histogram_bins = n_histogram_bins
        self.upper_body_ratio = upper_body_ratio
        self.n_warmup_frames = n_warmup_frames
        self.min_crop_height = min_crop_height

        self._feature_buffer: list[np.ndarray] = []
        self._frame_count = 0
        self._centers: np.ndarray | None = None

    def classify(
        self, detections: list[RawDetection], frame: np.ndarray
    ) -> dict[int, str]:
        if len(detections) < 2:
            return {}

        # Extract features for each detection
        features: list[tuple[int, np.ndarray]] = []
        for i, det in enumerate(detections):
            feat = self._extract_jersey_histogram(det.crop)
            if feat is not None:
                features.append((i, feat))

        if len(features) < 2:
            return {}

        self._frame_count += 1

        # Accumulate during warmup
        for _, feat in features:
            self._feature_buffer.append(feat)

        # Fit centers after warmup
        if self._centers is None:
            if self._frame_count < self.n_warmup_frames:
                return {}
            all_feats = np.array(self._feature_buffer)
            self._centers = self._kmeans_2(all_feats)

        # Assign to nearest center
        result: dict[int, str] = {}
        for idx, feat in features:
            d0 = np.linalg.norm(feat - self._centers[0])
            d1 = np.linalg.norm(feat - self._centers[1])
            team = "T_1" if d0 < d1 else "T_2"
            result[idx] = team

        return result

    def reset(self) -> None:
        self._feature_buffer.clear()
        self._frame_count = 0
        self._centers = None

    def _extract_jersey_histogram(
        self, crop: np.ndarray | None
    ) -> np.ndarray | None:
        """Extract normalized HS histogram from the upper body region."""
        if crop is None or crop.shape[0] < self.min_crop_height:
            return None

        # Take upper portion (jersey area)
        jersey_h = int(crop.shape[0] * self.upper_body_ratio)
        jersey = crop[:jersey_h, :]

        if jersey.size == 0:
            return None

        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1],
            None,
            [self.n_histogram_bins, self.n_histogram_bins],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)

    @staticmethod
    def _kmeans_2(
        features: np.ndarray, max_iters: int = 20
    ) -> np.ndarray:
        """Simple 2-means clustering using numpy. Returns 2 cluster centers."""
        n = len(features)
        if n < 2:
            return features[:2] if n == 2 else np.zeros((2, features.shape[1]))

        # Initialize: pick two most distant points
        idx0 = 0
        dists = np.linalg.norm(features - features[idx0], axis=1)
        idx1 = int(np.argmax(dists))

        centers = np.array([features[idx0], features[idx1]], dtype=np.float64)

        for _ in range(max_iters):
            # Assign
            d0 = np.linalg.norm(features - centers[0], axis=1)
            d1 = np.linalg.norm(features - centers[1], axis=1)
            labels = (d1 < d0).astype(int)  # 0 or 1

            # Update
            new_centers = np.zeros_like(centers)
            for k in range(2):
                mask = labels == k
                if mask.any():
                    new_centers[k] = features[mask].mean(axis=0)
                else:
                    new_centers[k] = centers[k]

            if np.allclose(new_centers, centers, atol=1e-6):
                break
            centers = new_centers

        return centers


# ---------------------------------------------------------------------------
# YOLO-Pose estimation strategy
# ---------------------------------------------------------------------------


class YoloPoseEstimationStrategy(PoseEstimationStrategy):
    """Pose estimation using YOLO-Pose model (yolo26m-pose.pt).

    Extracts 17 COCO keypoints per detected person.
    """

    COCO_KEYPOINT_NAMES = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(
        self,
        model_path: str = "yolo26m-pose.pt",
        confidence_threshold: float = 0.3,
        device: str | None = None,
        imgsz: int = 640,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.imgsz = imgsz
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)

    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[BoundingBox],
    ) -> list[list[PoseKeypoint]]:
        if not bboxes:
            return []

        self._ensure_model()

        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
            imgsz=self.imgsz,
        )

        if not results or results[0].keypoints is None:
            return [[] for _ in bboxes]

        # Parse all pose detections from YOLO-Pose
        det_keypoints = results[0].keypoints
        det_boxes = results[0].boxes
        if det_boxes is None:
            return [[] for _ in bboxes]

        det_xyxy = det_boxes.xyxy.cpu().numpy()
        kpts_xy = det_keypoints.xy.cpu().numpy()  # (N, 17, 2)
        kpts_conf = det_keypoints.conf.cpu().numpy()  # (N, 17)

        # Match each input bbox to the closest YOLO-Pose detection
        result: list[list[PoseKeypoint]] = []
        for bbox in bboxes:
            bcx = (bbox.x1 + bbox.x2) / 2.0
            bcy = (bbox.y1 + bbox.y2) / 2.0

            best_idx = -1
            best_dist = float("inf")
            for j in range(len(det_xyxy)):
                dcx = (det_xyxy[j][0] + det_xyxy[j][2]) / 2.0
                dcy = (det_xyxy[j][1] + det_xyxy[j][3]) / 2.0
                dist = (bcx - dcx) ** 2 + (bcy - dcy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j

            if best_idx < 0:
                result.append([])
                continue

            kp_list: list[PoseKeypoint] = []
            for k in range(len(self.COCO_KEYPOINT_NAMES)):
                x_val = float(kpts_xy[best_idx, k, 0])
                y_val = float(kpts_xy[best_idx, k, 1])
                c_val = float(kpts_conf[best_idx, k])
                if c_val > 0.0:
                    kp_list.append(
                        PoseKeypoint(
                            name=self.COCO_KEYPOINT_NAMES[k],
                            x=x_val,
                            y=y_val,
                            confidence=c_val,
                        )
                    )
            result.append(kp_list)

        return result

    def reset(self) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# PlayerDetector facade
# ---------------------------------------------------------------------------


class PlayerDetector:
    """Main player detection facade.

    Combines detection (pluggable), team classification (pluggable),
    and court position mapping with ghost filtering.
    """

    MAX_PLAYERS: int = 4
    COURT_MARGIN_M: float = 1.0

    def __init__(
        self,
        detection_strategy: PlayerDetectionStrategy | None = None,
        team_classifier: TeamClassifier | None = None,
        pose_strategy: PoseEstimationStrategy | None = None,
        model_path: str = "yolo26m.pt",
        confidence_threshold: float = 0.5,
    ) -> None:
        self.detection_strategy = detection_strategy or YoloPlayerDetectionStrategy(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )
        self.team_classifier = team_classifier or JerseyColorTeamClassifier()
        self.pose_strategy = pose_strategy

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
        homography_matrix: np.ndarray | None = None,
    ) -> list[PlayerFrame]:
        """Detect players in a single frame (no cross-frame tracking)."""
        raw = self.detection_strategy.detect(frame, frame_id, timestamp_ms)
        return self._build_player_frames(
            raw, frame, frame_id, timestamp_ms, homography_matrix
        )

    def detect_and_track(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
        homography_matrix: np.ndarray | None = None,
    ) -> list[PlayerFrame]:
        """Detect + track players across frames with ByteTrack."""
        raw = self.detection_strategy.detect_with_tracking(
            frame, frame_id, timestamp_ms
        )
        return self._build_player_frames(
            raw, frame, frame_id, timestamp_ms, homography_matrix
        )

    def reset_tracking(self) -> None:
        """Reset tracker and classifier state for a new video/scene."""
        self.detection_strategy.reset_tracking()
        self.team_classifier.reset()
        if self.pose_strategy is not None:
            self.pose_strategy.reset()

    # -- Private helpers ---------------------------------------------------

    def _build_player_frames(
        self,
        raw_detections: list[RawDetection],
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
        homography_matrix: np.ndarray | None,
    ) -> list[PlayerFrame]:
        """Convert raw detections into filtered, enriched PlayerFrames."""
        if not raw_detections:
            return []

        # 1. Compute court positions and filter ghosts
        positioned: list[tuple[RawDetection, Position2D | None]] = []
        for det in raw_detections:
            pos = None
            if homography_matrix is not None:
                court_pt = self._pixel_to_court(det.bbox, homography_matrix)
                if court_pt is None:
                    # Ghost detection — outside court bounds
                    continue
                pos = Position2D(x=court_pt[0], y=court_pt[1])
            positioned.append((det, pos))

        # 2. Top-4 by confidence
        if len(positioned) > self.MAX_PLAYERS:
            positioned.sort(key=lambda t: t[0].confidence, reverse=True)
            positioned = positioned[: self.MAX_PLAYERS]

        # 3. Team classification
        dets_for_classify = [t[0] for t in positioned]
        team_map = self.team_classifier.classify(dets_for_classify, frame)

        # 4. Pose estimation (optional)
        pose_results: list[list[PoseKeypoint]] | None = None
        if self.pose_strategy is not None:
            bboxes = [det.bbox for det, _ in positioned]
            pose_results = self.pose_strategy.estimate(frame, bboxes)

        # 5. Build PlayerFrame objects
        player_frames: list[PlayerFrame] = []
        for i, (det, pos) in enumerate(positioned):
            if det.track_id is not None:
                player_id = f"P_{det.track_id:03d}"
            else:
                player_id = f"P_det_{i:03d}"

            team_id = team_map.get(i)
            keypoints = pose_results[i] if pose_results and i < len(pose_results) else []

            player_frames.append(
                PlayerFrame(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    player_id=player_id,
                    team_id=team_id,
                    bbox=det.bbox,
                    position=pos,
                    confidence=det.confidence,
                    keypoints=keypoints,
                )
            )

        return player_frames

    def _pixel_to_court(
        self, bbox: BoundingBox, H: np.ndarray
    ) -> tuple[float, float] | None:
        """Map bbox foot position to court coords. None if out of bounds."""
        foot = self._bbox_foot_position(bbox)
        pt = np.array([[foot]], dtype=np.float64)
        result = cv2.perspectiveTransform(pt, H)
        x, y = float(result[0, 0, 0]), float(result[0, 0, 1])

        margin = self.COURT_MARGIN_M
        if x < -margin or x > 10.0 + margin or y < -margin or y > 20.0 + margin:
            return None

        # Clamp to valid Position2D range
        x_clamped = max(0.0, min(10.0, x))
        y_clamped = max(0.0, min(20.0, y))
        return (x_clamped, y_clamped)

    @staticmethod
    def _bbox_foot_position(bbox: BoundingBox) -> tuple[float, float]:
        """Compute foot position (bottom-center of bbox)."""
        cx = (bbox.x1 + bbox.x2) / 2.0
        return (cx, bbox.y2)
