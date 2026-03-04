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

from padex.schemas.events import Bounce, Shot, ShotType
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
        ball_frames_after: list[BallFrame],
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
        ball_frames_after: list[BallFrame],
        keypoints: list[PoseKeypoint],
    ) -> tuple[ShotType, float]:
        return (ShotType.UNKNOWN, 0.3)


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
            # Bounces between this contact and the next
            next_ts = (
                contacts[shot_idx + 1].timestamp_ms
                if shot_idx + 1 < len(contacts)
                else float("inf")
            )
            shot_bounces = [
                b
                for b in bounces
                if contact.timestamp_ms <= (b.timestamp_ms or 0) < next_ts
            ]

            # Classify
            ball_after = [
                bf
                for bf in ball_frames
                if bf.timestamp_ms >= contact.timestamp_ms
            ][:30]
            shot_type, conf = self.shot_type_classifier.classify(
                contact, ball_after, []
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

    def classify(
        self,
        shot: Shot,
        keypoints: list[PoseKeypoint],
    ) -> ShotType:
        """Re-classify an existing shot (Phase 3 hook)."""
        raise NotImplementedError("Pose-based classification is Phase 3")
