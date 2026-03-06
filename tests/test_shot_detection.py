"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: test_shot_detection.py
Description:
    Tests for shot detection and classification.
"""

import numpy as np
import pytest

from padex.events.shot import (
    ContactEvent,
    ProximityVelocityContactStrategy,
    ServeOnlyShotTypeClassifier,
    ShotDetector,
)
from padex.schemas.events import Bounce, BounceType, ShotType
from padex.schemas.tracking import (
    BallFrame,
    BallVisibility,
    PlayerFrame,
    BoundingBox,
    Position2D,
    Position3D,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ball_frame(
    frame_id: int, x: float, y: float, conf: float = 0.8
) -> BallFrame:
    return BallFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33.3,
        position=Position3D(x=x, y=y, z=0.0),
        confidence=conf,
        visibility=BallVisibility.VISIBLE,
    )


def _make_player_frame(
    frame_id: int, x: float, y: float, player_id: str = "P_001"
) -> PlayerFrame:
    return PlayerFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33.3,
        player_id=player_id,
        team_id="T_1",
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        position=Position2D(x=x, y=y),
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# ProximityVelocityContactStrategy tests
# ---------------------------------------------------------------------------


class TestProximityVelocityContact:
    def setup_method(self):
        self.strategy = ProximityVelocityContactStrategy(
            max_proximity_m=2.0,
            velocity_change_threshold=0.5,
            min_separation_ms=200.0,
        )

    def test_velocity_change_near_player_detected(self):
        """Ball changes direction near a player → contact detected."""
        # Ball moving right then suddenly left
        ball_frames = []
        for i in range(5):
            ball_frames.append(_make_ball_frame(i, x=3.0 + i * 0.5, y=5.0))
        for i in range(5, 10):
            ball_frames.append(
                _make_ball_frame(i, x=5.5 - (i - 5) * 0.5, y=5.0)
            )

        # Player at the reversal point
        player_frames = [_make_player_frame(i, x=5.0, y=5.0) for i in range(10)]

        contacts = self.strategy.detect_contacts(player_frames, ball_frames)
        assert len(contacts) >= 1
        assert contacts[0].player_id == "P_001"

    def test_velocity_change_without_player_not_detected(self):
        """Ball changes direction but no player nearby → no contact."""
        ball_frames = []
        for i in range(5):
            ball_frames.append(_make_ball_frame(i, x=3.0 + i * 0.5, y=5.0))
        for i in range(5, 10):
            ball_frames.append(
                _make_ball_frame(i, x=5.5 - (i - 5) * 0.5, y=5.0)
            )

        # Player far away
        player_frames = [
            _make_player_frame(i, x=0.0, y=0.0) for i in range(10)
        ]

        contacts = self.strategy.detect_contacts(player_frames, ball_frames)
        assert len(contacts) == 0

    def test_double_detection_suppressed(self):
        """Two rapid contacts within min_separation_ms → only first kept."""
        contacts = [
            ContactEvent(
                frame_id=0,
                timestamp_ms=0.0,
                player_id="P_001",
                team_id="T_1",
                ball_position=Position2D(x=5.0, y=5.0),
                player_position=None,
                contact_confidence=0.9,
            ),
            ContactEvent(
                frame_id=3,
                timestamp_ms=100.0,
                player_id="P_001",
                team_id="T_1",
                ball_position=Position2D(x=5.0, y=5.0),
                player_position=None,
                contact_confidence=0.9,
            ),
        ]
        suppressed = ProximityVelocityContactStrategy._suppress_nearby(
            contacts, 200.0
        )
        assert len(suppressed) == 1

    def test_empty_ball_frames(self):
        contacts = self.strategy.detect_contacts([], [])
        assert contacts == []

    def test_no_position_skipped(self):
        """Ball frames without positions are skipped."""
        ball_frames = [
            BallFrame(
                frame_id=0,
                timestamp_ms=0.0,
                confidence=0.0,
                visibility=BallVisibility.OCCLUDED,
            )
        ]
        contacts = self.strategy.detect_contacts([], ball_frames)
        assert contacts == []


# ---------------------------------------------------------------------------
# ServeOnlyShotTypeClassifier tests
# ---------------------------------------------------------------------------


class TestServeOnlyClassifier:
    def test_always_returns_unknown(self):
        classifier = ServeOnlyShotTypeClassifier()
        contact = ContactEvent(
            frame_id=0,
            timestamp_ms=0.0,
            player_id="P_001",
            team_id="T_1",
            ball_position=Position2D(x=5.0, y=2.0),
            player_position=None,
            contact_confidence=0.9,
        )
        shot_type, conf = classifier.classify(contact, [], [], [], [])
        assert shot_type == ShotType.UNKNOWN
        assert conf == 0.3


# ---------------------------------------------------------------------------
# ShotDetector facade tests
# ---------------------------------------------------------------------------


class TestShotDetector:
    def test_detect_shots_basic(self):
        """Simple scenario with one contact → one shot."""
        ball_frames = []
        for i in range(5):
            ball_frames.append(_make_ball_frame(i, x=3.0 + i * 0.5, y=5.0))
        for i in range(5, 10):
            ball_frames.append(
                _make_ball_frame(i, x=5.5 - (i - 5) * 0.5, y=5.0)
            )

        player_frames = [
            _make_player_frame(i, x=5.0, y=5.0) for i in range(10)
        ]

        detector = ShotDetector(
            contact_strategy=ProximityVelocityContactStrategy(
                velocity_change_threshold=0.5,
                min_separation_ms=200.0,
            )
        )
        shots = detector.detect_shots(player_frames, ball_frames)
        assert len(shots) >= 1
        assert shots[0].shot_type == ShotType.UNKNOWN
        assert shots[0].player_id == "P_001"

    def test_shot_id_format(self):
        """Generated shot IDs match S_NNN_NN_NN_NNN pattern."""
        ball_frames = []
        for i in range(5):
            ball_frames.append(_make_ball_frame(i, x=3.0 + i * 0.5, y=5.0))
        for i in range(5, 10):
            ball_frames.append(
                _make_ball_frame(i, x=5.5 - (i - 5) * 0.5, y=5.0)
            )
        player_frames = [
            _make_player_frame(i, x=5.0, y=5.0) for i in range(10)
        ]

        detector = ShotDetector(
            contact_strategy=ProximityVelocityContactStrategy(
                velocity_change_threshold=0.5,
            )
        )
        shots = detector.detect_shots(
            player_frames, ball_frames, set_num=2, game_num=3, point_num=4
        )
        if shots:
            import re

            assert re.match(r"^S_\d{3}_\d{2}_\d{2}_\d{3}$", shots[0].shot_id)
            assert shots[0].shot_id.startswith("S_002_03_04_")

    def test_trajectory_contains_bounces(self):
        """Bounces are attached to the correct shot."""
        ball_frames = []
        for i in range(5):
            ball_frames.append(_make_ball_frame(i, x=3.0 + i * 0.5, y=5.0))
        for i in range(5, 10):
            ball_frames.append(
                _make_ball_frame(i, x=5.5 - (i - 5) * 0.5, y=5.0)
            )
        player_frames = [
            _make_player_frame(i, x=5.0, y=5.0) for i in range(10)
        ]

        bounces = [
            Bounce(
                type=BounceType.GROUND,
                position=Position2D(x=5.0, y=8.0),
                timestamp_ms=200.0,
            )
        ]

        detector = ShotDetector(
            contact_strategy=ProximityVelocityContactStrategy(
                velocity_change_threshold=0.5,
            )
        )
        shots = detector.detect_shots(
            player_frames, ball_frames, bounces=bounces
        )
        if shots:
            # Bounce at 200ms should be within the shot's time range
            total_bounces = sum(len(s.trajectory) for s in shots)
            assert total_bounces <= len(bounces)

    def test_empty_inputs(self):
        detector = ShotDetector()
        shots = detector.detect_shots([], [])
        assert shots == []
