"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: shot.py
Description:
    Shot detection and classification from tracking data.
    Detects ball-player contact via velocity change + proximity,
    then classifies shot type (Phase 2: only SERVE + UNKNOWN).
"""

from __future__ import annotations

import abc
import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from padex.schemas.events import Bounce, BounceType, Shot, ShotType
from padex.schemas.tracking import (
    BallFrame,
    BallVisibility,
    PlayerFrame,
    Position2D,
    PoseKeypoint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ContactEvent:
    """A detected ball-player contact."""

    frame_id: int
    timestamp_ms: float
    player_id: str
    team_id: str | None
    ball_position: Position2D
    player_position: Position2D | None
    contact_confidence: float


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class ShotContactStrategy(abc.ABC):
    """Abstract interface for contact detection backends."""

    @abc.abstractmethod
    def detect_contacts(
        self,
        player_frames: list[PlayerFrame],
        ball_frames: list[BallFrame],
    ) -> list[ContactEvent]:
        """Identify contact events between players and ball."""
        ...


class ShotTypeClassifier(abc.ABC):
    """Abstract interface for shot type classification."""

    @abc.abstractmethod
    def classify(
        self,
        contact: ContactEvent,
        ball_frames_before: list[BallFrame],
        ball_frames_after: list[BallFrame],
        bounces_before: list[Bounce],
        keypoints: list[PoseKeypoint],
    ) -> tuple[ShotType, float]:
        """Returns (shot_type, confidence)."""
        ...


# ---------------------------------------------------------------------------
# Proximity + velocity contact detection
# ---------------------------------------------------------------------------


class ProximityVelocityContactStrategy(ShotContactStrategy):
    """Detect contacts using ball velocity change + player proximity.

    Algorithm:
    1. Build frame-indexed player lookup
    2. Compute ball velocities from visible frames
    3. Find velocity change peaks (delta-V > threshold)
    4. Attribute to nearest player within proximity radius
    5. Suppress temporally close detections
    """

    def __init__(
        self,
        max_proximity_m: float = 2.0,
        velocity_change_threshold: float = 0.5,
        min_separation_ms: float = 300.0,
        smooth_window: int = 3,
    ) -> None:
        self.max_proximity_m = max_proximity_m
        self.velocity_change_threshold = velocity_change_threshold
        self.min_separation_ms = min_separation_ms
        self.smooth_window = smooth_window

    def detect_contacts(
        self,
        player_frames: list[PlayerFrame],
        ball_frames: list[BallFrame],
    ) -> list[ContactEvent]:
        player_lookup = self._build_player_lookup(player_frames)

        # Extract visible ball frames with positions
        visible_ball = [
            bf
            for bf in ball_frames
            if bf.position is not None
            and bf.visibility != BallVisibility.INFERRED
        ]

        if len(visible_ball) < 3:
            return []

        positions = np.array(
            [[bf.position.x, bf.position.y] for bf in visible_ball]
        )
        velocities = np.diff(positions, axis=0)

        # Find velocity change candidates
        contacts: list[ContactEvent] = []
        for k in range(1, len(velocities)):
            delta_v = np.linalg.norm(velocities[k] - velocities[k - 1])
            if delta_v < self.velocity_change_threshold:
                continue

            bf = visible_ball[k]
            players_at_frame = player_lookup.get(bf.frame_id, [])
            if not players_at_frame:
                continue

            # Find nearest player within proximity radius
            ball_pos = np.array([bf.position.x, bf.position.y])
            nearest = None
            nearest_dist = float("inf")
            for pf in players_at_frame:
                if pf.position is None:
                    continue
                p_pos = np.array([pf.position.x, pf.position.y])
                dist = float(np.linalg.norm(ball_pos - p_pos))
                if dist < nearest_dist and dist <= self.max_proximity_m:
                    nearest_dist = dist
                    nearest = pf

            if nearest is None:
                continue

            contacts.append(
                ContactEvent(
                    frame_id=bf.frame_id,
                    timestamp_ms=bf.timestamp_ms,
                    player_id=nearest.player_id,
                    team_id=nearest.team_id,
                    ball_position=Position2D(
                        x=bf.position.x, y=bf.position.y
                    ),
                    player_position=nearest.position,
                    contact_confidence=min(
                        1.0,
                        delta_v / (self.velocity_change_threshold * 3),
                    ),
                )
            )

        return self._suppress_nearby(contacts, self.min_separation_ms)

    @staticmethod
    def _build_player_lookup(
        player_frames: list[PlayerFrame],
    ) -> dict[int, list[PlayerFrame]]:
        lookup: dict[int, list[PlayerFrame]] = defaultdict(list)
        for pf in player_frames:
            lookup[pf.frame_id].append(pf)
        return lookup

    @staticmethod
    def _suppress_nearby(
        contacts: list[ContactEvent], min_sep_ms: float
    ) -> list[ContactEvent]:
        if not contacts:
            return []
        result = [contacts[0]]
        for c in contacts[1:]:
            if c.timestamp_ms - result[-1].timestamp_ms >= min_sep_ms:
                result.append(c)
        return result


# ---------------------------------------------------------------------------
# Phase 2 shot type classifier: serve + unknown only
# ---------------------------------------------------------------------------


class ServeOnlyShotTypeClassifier(ShotTypeClassifier):
    """Phase 2 classifier: returns UNKNOWN for all shots.

    Serve classification is handled by PointSegmenter (first shot of point).
    """

    def classify(
        self,
        contact: ContactEvent,
        ball_frames_before: list[BallFrame],
        ball_frames_after: list[BallFrame],
        bounces_before: list[Bounce],
        keypoints: list[PoseKeypoint],
    ) -> tuple[ShotType, float]:
        return (ShotType.UNKNOWN, 0.3)


# ---------------------------------------------------------------------------
# Pose-based shot type classifier
# ---------------------------------------------------------------------------


class PoseBasedShotTypeClassifier(ShotTypeClassifier):
    """Classifies shot types using pre-contact ball state + pose + trajectory.

    Decision tree (priority order):
    1. Pre-contact ball state: no ground bounce → net play (volley family)
    2. Pre-contact ball state: wall bounce → defensive (wall_return family)
    3. Remaining: ground bounce only → baseline play (groundstroke family)

    Within each category, pose (wrist height) and post-contact trajectory
    refine the specific shot type.
    """

    NET_Y = 10.0
    BASELINE_THRESHOLD = 3.0  # meters from back wall
    MIN_KP_CONF = 0.3

    def classify(
        self,
        contact: ContactEvent,
        ball_frames_before: list[BallFrame],
        ball_frames_after: list[BallFrame],
        bounces_before: list[Bounce],
        keypoints: list[PoseKeypoint],
    ) -> tuple[ShotType, float]:
        kp_map = self._build_kp_map(keypoints)
        is_overhead, wrist_diff = self._is_overhead(kp_map)
        player_pos = contact.player_position or contact.ball_position

        had_ground = self._had_ground_bounce(bounces_before)
        had_wall, wall_type = self._had_wall_bounce(bounces_before)

        # --- Branch 1: No ground bounce before contact → net play ---
        if not had_ground:
            return self._classify_net_play(
                is_overhead, wrist_diff, kp_map,
                ball_frames_after, player_pos,
            )

        # --- Branch 2: Wall bounce before contact → defensive ---
        if had_wall:
            return self._classify_wall_play(
                is_overhead, contact, ball_frames_after, wall_type,
            )

        # --- Branch 3: Ground bounce only → baseline / transition ---
        return self._classify_baseline_play(
            is_overhead, kp_map, contact, ball_frames_after,
        )

    # ------------------------------------------------------------------
    # Branch classifiers
    # ------------------------------------------------------------------

    def _classify_net_play(
        self,
        is_overhead: bool,
        wrist_diff: float,
        kp_map: dict[str, PoseKeypoint],
        ball_frames_after: list[BallFrame],
        player_pos: Position2D,
    ) -> tuple[ShotType, float]:
        """Ball didn't bounce → volley / bandeja / vibora / smash."""
        if is_overhead:
            # High overhead with large wrist-shoulder gap → smash
            if wrist_diff > 50:
                if self._is_exit_smash(ball_frames_after):
                    return (ShotType.SMASH_X3, 0.7)
                return (ShotType.SMASH, 0.75)
            # Side spin motion → vibora
            if self._has_side_spin(kp_map):
                return (ShotType.VIBORA, 0.65)
            # Moderate overhead → bandeja
            return (ShotType.BANDEJA, 0.7)

        # Low contact, no bounce
        if self._is_short_trajectory(ball_frames_after):
            return (ShotType.DROP_SHOT, 0.65)
        return (ShotType.VOLLEY, 0.7)

    def _classify_wall_play(
        self,
        is_overhead: bool,
        contact: ContactEvent,
        ball_frames_after: list[BallFrame],
        wall_type: BounceType | None,
    ) -> tuple[ShotType, float]:
        """Ball bounced off wall before contact → defensive shots."""
        player_pos = contact.player_position or contact.ball_position
        at_baseline = (
            player_pos.y < self.BASELINE_THRESHOLD
            or player_pos.y > (20.0 - self.BASELINE_THRESHOLD)
        )

        if at_baseline:
            if self._is_lob_trajectory(ball_frames_after):
                return (ShotType.LOB, 0.65)
            return (ShotType.CONTRA_PARED, 0.6)

        return (ShotType.WALL_RETURN, 0.6)

    def _classify_baseline_play(
        self,
        is_overhead: bool,
        kp_map: dict[str, PoseKeypoint],
        contact: ContactEvent,
        ball_frames_after: list[BallFrame],
    ) -> tuple[ShotType, float]:
        """Ball bounced on ground only → baseline / transition shots."""
        if is_overhead:
            # Overhead from baseline after ground bounce → bajada or lob
            if self._is_lob_trajectory(ball_frames_after):
                return (ShotType.LOB, 0.6)
            return (ShotType.BAJADA, 0.6)

        if self._is_chiquita(contact, ball_frames_after):
            return (ShotType.CHIQUITA, 0.6)

        if self._is_forehand(kp_map):
            return (ShotType.GROUNDSTROKE_FH, 0.65)
        return (ShotType.GROUNDSTROKE_BH, 0.65)

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _build_kp_map(
        self, keypoints: list[PoseKeypoint]
    ) -> dict[str, PoseKeypoint]:
        return {
            kp.name: kp
            for kp in keypoints
            if kp.confidence >= self.MIN_KP_CONF
        }

    @staticmethod
    def _is_overhead(
        kp_map: dict[str, PoseKeypoint],
    ) -> tuple[bool, float]:
        """Check if wrist is above shoulder (pixel coords: lower y = higher).

        Returns (is_overhead, wrist_diff in pixels).
        """
        required = [
            "left_shoulder", "right_shoulder",
            "left_wrist", "right_wrist",
        ]
        if not all(k in kp_map for k in required):
            return (False, 0.0)

        wrist_y = min(kp_map["left_wrist"].y, kp_map["right_wrist"].y)
        shoulder_y = min(
            kp_map["left_shoulder"].y, kp_map["right_shoulder"].y
        )
        diff = shoulder_y - wrist_y  # positive = wrist above shoulder
        return (diff > 0, diff)

    @staticmethod
    def _had_ground_bounce(bounces_before: list[Bounce]) -> bool:
        return any(b.type == BounceType.GROUND for b in bounces_before)

    @staticmethod
    def _had_wall_bounce(
        bounces_before: list[Bounce],
    ) -> tuple[bool, BounceType | None]:
        wall_types = {
            BounceType.BACK_WALL, BounceType.SIDE_WALL,
            BounceType.BACK_FENCE, BounceType.SIDE_FENCE,
            BounceType.CORNER,
        }
        for b in reversed(bounces_before):
            if b.type in wall_types:
                return (True, b.type)
        return (False, None)

    @staticmethod
    def _is_forehand(kp_map: dict[str, PoseKeypoint]) -> bool:
        l_wrist = kp_map.get("left_wrist")
        r_wrist = kp_map.get("right_wrist")
        l_hip = kp_map.get("left_hip")
        r_hip = kp_map.get("right_hip")
        if not (l_wrist and r_wrist):
            return True
        if l_hip and r_hip:
            body_cx = (l_hip.x + r_hip.x) / 2
            return abs(r_wrist.x - body_cx) > abs(l_wrist.x - body_cx)
        return True

    @staticmethod
    def _has_side_spin(kp_map: dict[str, PoseKeypoint]) -> bool:
        l_elbow = kp_map.get("left_elbow")
        r_elbow = kp_map.get("right_elbow")
        l_wrist = kp_map.get("left_wrist")
        r_wrist = kp_map.get("right_wrist")
        if not (l_elbow and r_elbow and l_wrist and r_wrist):
            return False
        l_lateral = abs(l_wrist.x - l_elbow.x)
        r_lateral = abs(r_wrist.x - r_elbow.x)
        return max(l_lateral, r_lateral) > 40

    @staticmethod
    def _is_short_trajectory(ball_frames_after: list[BallFrame]) -> bool:
        positions = [
            bf.position
            for bf in ball_frames_after[:10]
            if bf.position is not None
        ]
        if len(positions) < 2:
            return False
        total_dist = sum(
            np.sqrt(
                (positions[i].x - positions[i - 1].x) ** 2
                + (positions[i].y - positions[i - 1].y) ** 2
            )
            for i in range(1, len(positions))
        )
        return total_dist < 2.0

    @staticmethod
    def _is_lob_trajectory(ball_frames_after: list[BallFrame]) -> bool:
        positions = [
            bf.position
            for bf in ball_frames_after[:15]
            if bf.position is not None
        ]
        if len(positions) < 3:
            return False
        return abs(positions[-1].y - positions[0].y) > 3.0

    @staticmethod
    def _is_exit_smash(ball_frames_after: list[BallFrame]) -> bool:
        for bf in ball_frames_after[:20]:
            if bf.position is None:
                continue
            if bf.position.y < 0.5 or bf.position.y > 19.5:
                return True
            if bf.position.x < 0.5 or bf.position.x > 9.5:
                return True
        return False

    @staticmethod
    def _is_chiquita(
        contact: ContactEvent, ball_frames_after: list[BallFrame]
    ) -> bool:
        positions = [
            bf.position
            for bf in ball_frames_after[:10]
            if bf.position is not None
        ]
        if len(positions) < 2:
            return False
        player_pos = contact.player_position or contact.ball_position
        ball_end = positions[-1]
        return abs(ball_end.y - 10.0) < abs(player_pos.y - 10.0)


# ---------------------------------------------------------------------------
# ShotDetector facade
# ---------------------------------------------------------------------------


class ShotDetector:
    """Main shot detection facade.

    Combines contact detection with shot type classification and
    trajectory (bounce) assembly.
    """

    def __init__(
        self,
        contact_strategy: ShotContactStrategy | None = None,
        shot_type_classifier: ShotTypeClassifier | None = None,
    ) -> None:
        self.contact_strategy = (
            contact_strategy or ProximityVelocityContactStrategy()
        )
        self.shot_type_classifier = (
            shot_type_classifier or ServeOnlyShotTypeClassifier()
        )

    def detect_shots(
        self,
        player_frames: list[PlayerFrame],
        ball_frames: list[BallFrame],
        bounces: list[Bounce] | None = None,
        set_num: int = 1,
        game_num: int = 1,
        point_num: int = 1,
    ) -> list[Shot]:
        """Identify shots and build trajectory for each.

        Args:
            player_frames: All PlayerFrames for the sequence.
            ball_frames: All BallFrames from BallDetector.
            bounces: All Bounce events from BounceDetector.
            set_num, game_num, point_num: For hierarchical ID.

        Returns:
            list[Shot] in chronological order.
        """
        contacts = self.contact_strategy.detect_contacts(
            player_frames, ball_frames
        )
        if not contacts:
            return []

        bounces = bounces or []

        shots: list[Shot] = []
        for shot_idx, contact in enumerate(contacts):
            # Timestamp of previous contact (or start of sequence)
            prev_ts = (
                contacts[shot_idx - 1].timestamp_ms
                if shot_idx > 0
                else -float("inf")
            )
            # Timestamp of next contact (or end of sequence)
            next_ts = (
                contacts[shot_idx + 1].timestamp_ms
                if shot_idx + 1 < len(contacts)
                else float("inf")
            )

            # Bounces between this contact and the next (trajectory)
            shot_bounces = [
                b
                for b in bounces
                if contact.timestamp_ms <= (b.timestamp_ms or 0) < next_ts
            ]

            # Bounces between previous contact and this one
            bounces_before = [
                b
                for b in bounces
                if prev_ts < (b.timestamp_ms or 0) < contact.timestamp_ms
            ]

            # Ball frames before and after contact
            ball_before = [
                bf
                for bf in ball_frames
                if prev_ts < bf.timestamp_ms <= contact.timestamp_ms
            ][-30:]
            ball_after = [
                bf
                for bf in ball_frames
                if bf.timestamp_ms >= contact.timestamp_ms
            ][:30]

            # Classify
            contact_kps = self._find_keypoints(
                player_frames, contact.player_id, contact.frame_id
            )
            shot_type, conf = self.shot_type_classifier.classify(
                contact, ball_before, ball_after, bounces_before, contact_kps
            )

            shot_id = (
                f"S_{set_num:03d}_{game_num:02d}_{point_num:02d}"
                f"_{shot_idx + 1:03d}"
            )

            shots.append(
                Shot(
                    shot_id=shot_id,
                    timestamp_ms=contact.timestamp_ms,
                    player_id=contact.player_id,
                    team_id=contact.team_id or "T_UNKNOWN",
                    position=contact.ball_position,
                    shot_type=shot_type,
                    trajectory=shot_bounces,
                    outcome=None,
                    confidence=contact.contact_confidence * conf,
                )
            )

        return shots

    @staticmethod
    def _find_keypoints(
        player_frames: list[PlayerFrame],
        player_id: str,
        frame_id: int,
    ) -> list[PoseKeypoint]:
        """Find keypoints for a specific player at a specific frame."""
        for pf in player_frames:
            if pf.player_id == player_id and pf.frame_id == frame_id:
                return pf.keypoints
        return []
