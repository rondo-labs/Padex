"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: test_shot_classification.py
Description:
    Tests for pose-based shot type classification.
"""

import pytest

from padex.events.shot import (
    ContactEvent,
    PoseBasedShotTypeClassifier,
)
from padex.schemas.events import ShotType
from padex.schemas.tracking import (
    BallFrame,
    BallVisibility,
    PoseKeypoint,
    Position2D,
    Position3D,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keypoints(
    wrist_y: float = 200.0,
    shoulder_y: float = 250.0,
    hip_x_offset: float = 0.0,
    elbow_lateral: float = 20.0,
) -> list[PoseKeypoint]:
    """Create a minimal set of keypoints for testing.

    In pixel coords: lower y = higher on screen.
    wrist_y < shoulder_y means wrist is above shoulder.
    """
    return [
        PoseKeypoint(name="left_shoulder", x=400.0, y=shoulder_y, confidence=0.9),
        PoseKeypoint(name="right_shoulder", x=500.0, y=shoulder_y, confidence=0.9),
        PoseKeypoint(name="left_wrist", x=380.0, y=wrist_y, confidence=0.9),
        PoseKeypoint(name="right_wrist", x=520.0, y=wrist_y, confidence=0.9),
        PoseKeypoint(
            name="left_elbow",
            x=390.0 - elbow_lateral,
            y=(wrist_y + shoulder_y) / 2,
            confidence=0.9,
        ),
        PoseKeypoint(
            name="right_elbow",
            x=510.0 + elbow_lateral,
            y=(wrist_y + shoulder_y) / 2,
            confidence=0.9,
        ),
        PoseKeypoint(
            name="left_hip", x=420.0 + hip_x_offset, y=350.0, confidence=0.9
        ),
        PoseKeypoint(
            name="right_hip", x=480.0 + hip_x_offset, y=350.0, confidence=0.9
        ),
    ]


def _make_contact(
    player_x: float = 5.0,
    player_y: float = 5.0,
    ball_x: float = 5.0,
    ball_y: float = 5.0,
) -> ContactEvent:
    return ContactEvent(
        frame_id=0,
        timestamp_ms=0.0,
        player_id="P_001",
        team_id="T_1",
        ball_position=Position2D(x=ball_x, y=ball_y),
        player_position=Position2D(x=player_x, y=player_y),
        contact_confidence=0.9,
    )


def _make_ball_frames(
    positions: list[tuple[float, float]],
) -> list[BallFrame]:
    return [
        BallFrame(
            frame_id=i,
            timestamp_ms=i * 33.3,
            position=Position3D(x=x, y=y, z=0.0),
            confidence=0.8,
            visibility=BallVisibility.VISIBLE,
        )
        for i, (x, y) in enumerate(positions)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPoseBasedClassifier:
    def setup_method(self):
        self.clf = PoseBasedShotTypeClassifier()

    def test_no_keypoints_returns_unknown(self):
        contact = _make_contact()
        shot_type, conf = self.clf.classify(contact, [], [])
        assert shot_type == ShotType.UNKNOWN
        assert conf == 0.3

    def test_insufficient_keypoints_returns_unknown(self):
        kps = [PoseKeypoint(name="nose", x=450.0, y=200.0, confidence=0.9)]
        contact = _make_contact()
        shot_type, _ = self.clf.classify(contact, [], kps)
        assert shot_type == ShotType.UNKNOWN

    def test_smash_at_net_high_wrist(self):
        """Wrist far above shoulder at net → SMASH."""
        kps = _make_keypoints(wrist_y=150.0, shoulder_y=250.0)  # diff=100 > 50
        contact = _make_contact(player_y=10.0)  # at net
        ball_after = _make_ball_frames([(5, 10)] * 5)
        shot_type, _ = self.clf.classify(contact, ball_after, kps)
        assert shot_type == ShotType.SMASH

    def test_smash_exit_becomes_x3(self):
        """Smash with ball exiting court → SMASH_X3."""
        kps = _make_keypoints(wrist_y=150.0, shoulder_y=250.0)
        contact = _make_contact(player_y=10.0)
        # Ball goes near edge of court (simulating exit)
        ball_after = _make_ball_frames(
            [(5.0, min(19.8, 10.0 + i * 0.8)) for i in range(15)]
            + [(9.8, 19.8)] * 5  # near edge
        )
        shot_type, _ = self.clf.classify(contact, ball_after, kps)
        assert shot_type == ShotType.SMASH_X3

    def test_bandeja_moderate_height(self):
        """Moderate wrist above shoulder at net → BANDEJA."""
        kps = _make_keypoints(wrist_y=220.0, shoulder_y=250.0)  # diff=30 < 50
        contact = _make_contact(player_y=10.0)
        ball_after = _make_ball_frames([(5, 10)] * 5)
        shot_type, _ = self.clf.classify(contact, ball_after, kps)
        assert shot_type == ShotType.BANDEJA

    def test_vibora_side_spin(self):
        """Moderate height at net + side spin → VIBORA."""
        kps = _make_keypoints(
            wrist_y=220.0, shoulder_y=250.0, elbow_lateral=60.0
        )
        contact = _make_contact(player_y=10.0)
        ball_after = _make_ball_frames([(5, 10)] * 5)
        shot_type, _ = self.clf.classify(contact, ball_after, kps)
        assert shot_type == ShotType.VIBORA

    def test_volley_low_at_net(self):
        """Wrist below shoulder at net → VOLLEY."""
        kps = _make_keypoints(wrist_y=280.0, shoulder_y=250.0)
        contact = _make_contact(player_y=10.0)
        # Ball travels far enough to not be drop shot
        ball_after = _make_ball_frames(
            [(5.0, 10.0 + i * 0.5) for i in range(10)]
        )
        shot_type, _ = self.clf.classify(contact, ball_after, kps)
        assert shot_type == ShotType.VOLLEY

    def test_drop_shot_short_trajectory(self):
        """Low at net + short trajectory → DROP_SHOT."""
        kps = _make_keypoints(wrist_y=280.0, shoulder_y=250.0)
        contact = _make_contact(player_y=10.0)
        # Ball barely moves
        ball_after = _make_ball_frames(
            [(5.0, 10.0 + i * 0.1) for i in range(10)]
        )
        shot_type, _ = self.clf.classify(contact, ball_after, kps)
        assert shot_type == ShotType.DROP_SHOT

    def test_lob_from_baseline(self):
        """High wrist at baseline + long arc → LOB."""
        kps = _make_keypoints(wrist_y=200.0, shoulder_y=250.0)
        contact = _make_contact(player_y=2.0)  # baseline
        # Ball travels far in y
        ball_after = _make_ball_frames(
            [(5.0, 2.0 + i * 1.0) for i in range(15)]
        )
        shot_type, _ = self.clf.classify(contact, ball_after, kps)
        assert shot_type == ShotType.LOB

    def test_bajada_high_baseline_no_lob(self):
        """High wrist at baseline but short y travel → BAJADA."""
        kps = _make_keypoints(wrist_y=200.0, shoulder_y=250.0)
        contact = _make_contact(player_y=2.0)
        ball_after = _make_ball_frames(
            [(5.0, 2.0 + i * 0.1) for i in range(5)]
        )
        shot_type, _ = self.clf.classify(contact, ball_after, kps)
        assert shot_type == ShotType.BAJADA

    def test_wall_return_with_wall_bounce(self):
        """Low baseline + ball near wall → WALL_RETURN."""
        kps = _make_keypoints(wrist_y=280.0, shoulder_y=250.0)
        contact = _make_contact(player_y=3.0)
        # Ball goes near wall
        ball_after = _make_ball_frames(
            [(0.5, 3.0 + i * 0.3) for i in range(10)]
        )
        shot_type, _ = self.clf.classify(contact, ball_after, kps)
        assert shot_type == ShotType.WALL_RETURN

    def test_contra_pared_near_back_wall(self):
        """Near back wall + wall bounce → CONTRA_PARED."""
        kps = _make_keypoints(wrist_y=280.0, shoulder_y=250.0)
        contact = _make_contact(player_y=1.5)  # near back wall
        ball_after = _make_ball_frames(
            [(0.5, 1.5 + i * 0.3) for i in range(10)]
        )
        shot_type, _ = self.clf.classify(contact, ball_after, kps)
        assert shot_type == ShotType.CONTRA_PARED

    def test_chiquita_toward_net(self):
        """Low baseline + ball moves toward net → CHIQUITA."""
        kps = _make_keypoints(wrist_y=280.0, shoulder_y=250.0)
        contact = _make_contact(player_y=4.0)
        # Ball goes toward net (y=10) and doesn't touch walls
        ball_after = _make_ball_frames(
            [(5.0, 4.0 + i * 0.8) for i in range(10)]
        )
        shot_type, _ = self.clf.classify(contact, ball_after, kps)
        assert shot_type == ShotType.CHIQUITA

    def test_groundstroke_fh(self):
        """Low baseline, no wall, not toward net → GROUNDSTROKE_FH."""
        kps = _make_keypoints(wrist_y=280.0, shoulder_y=250.0)
        contact = _make_contact(player_y=4.0)
        # Ball goes slightly sideways (not toward net)
        ball_after = _make_ball_frames(
            [(5.0 + i * 0.3, 4.0) for i in range(10)]
        )
        shot_type, _ = self.clf.classify(contact, ball_after, kps)
        assert shot_type in (ShotType.GROUNDSTROKE_FH, ShotType.GROUNDSTROKE_BH)

    def test_low_confidence_keypoints_ignored(self):
        """Keypoints with confidence < MIN_KP_CONF are excluded."""
        kps = [
            PoseKeypoint(name="left_shoulder", x=400.0, y=250.0, confidence=0.1),
            PoseKeypoint(name="right_shoulder", x=500.0, y=250.0, confidence=0.9),
            PoseKeypoint(name="left_wrist", x=380.0, y=200.0, confidence=0.1),
            PoseKeypoint(name="right_wrist", x=520.0, y=200.0, confidence=0.9),
        ]
        contact = _make_contact()
        shot_type, _ = self.clf.classify(contact, [], kps)
        # left_shoulder and left_wrist filtered out → insufficient → UNKNOWN
        assert shot_type == ShotType.UNKNOWN


class TestPoseClassifierHelpers:
    def test_is_forehand_right_dominant(self):
        kp_map = {
            "left_wrist": PoseKeypoint(
                name="left_wrist", x=400.0, y=200.0, confidence=0.9
            ),
            "right_wrist": PoseKeypoint(
                name="right_wrist", x=600.0, y=200.0, confidence=0.9
            ),
            "left_hip": PoseKeypoint(
                name="left_hip", x=440.0, y=350.0, confidence=0.9
            ),
            "right_hip": PoseKeypoint(
                name="right_hip", x=460.0, y=350.0, confidence=0.9
            ),
        }
        assert PoseBasedShotTypeClassifier._is_forehand(kp_map) is True

    def test_is_forehand_left_dominant(self):
        kp_map = {
            "left_wrist": PoseKeypoint(
                name="left_wrist", x=200.0, y=200.0, confidence=0.9
            ),
            "right_wrist": PoseKeypoint(
                name="right_wrist", x=460.0, y=200.0, confidence=0.9
            ),
            "left_hip": PoseKeypoint(
                name="left_hip", x=440.0, y=350.0, confidence=0.9
            ),
            "right_hip": PoseKeypoint(
                name="right_hip", x=460.0, y=350.0, confidence=0.9
            ),
        }
        assert PoseBasedShotTypeClassifier._is_forehand(kp_map) is False

    def test_has_wall_bounce(self):
        frames = _make_ball_frames([(0.5, 5.0)])
        assert PoseBasedShotTypeClassifier._has_wall_bounce(frames) is True

    def test_no_wall_bounce(self):
        frames = _make_ball_frames([(5.0, 10.0)])
        assert PoseBasedShotTypeClassifier._has_wall_bounce(frames) is False

    def test_short_trajectory(self):
        frames = _make_ball_frames([(5.0, 5.0), (5.1, 5.1)])
        assert PoseBasedShotTypeClassifier._is_short_trajectory(frames)

    def test_long_trajectory(self):
        frames = _make_ball_frames([(5.0, 5.0), (5.0, 8.0)])
        assert not PoseBasedShotTypeClassifier._is_short_trajectory(frames)
