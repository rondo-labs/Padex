"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: test_report.py
Description:
    Tests for match report generation (JSON + HTML).
"""

import json
from pathlib import Path

import pytest

from padex.schemas.events import Point, Shot, ShotOutcome, ShotType
from padex.schemas.tactics import MatchAnalytics, PlayerMetrics, RallyMetrics, TeamMetrics
from padex.schemas.tracking import BoundingBox, PlayerFrame, Position2D
from padex.tactics.report import MatchReporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shot(
    player_id="P_001",
    team_id="T_1",
    outcome=ShotOutcome.NEXT_SHOT,
    shot_type=ShotType.UNKNOWN,
    timestamp_ms=0.0,
) -> Shot:
    return Shot(
        shot_id="S_001_01_01_001",
        timestamp_ms=timestamp_ms,
        player_id=player_id,
        team_id=team_id,
        position=Position2D(x=5.0, y=5.0),
        shot_type=shot_type,
        outcome=outcome,
        confidence=0.8,
    )


def _make_point(shots, point_id="S_001_01_01") -> Point:
    duration = shots[-1].timestamp_ms - shots[0].timestamp_ms if len(shots) > 1 else 0.0
    return Point(
        point_id=point_id,
        shots=shots,
        winner_team_id=shots[-1].team_id if shots else None,
        duration_ms=duration,
        rally_length=len(shots),
    )


def _make_player_frame(frame_id, player_id="P_001", team_id="T_1"):
    return PlayerFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33.3,
        player_id=player_id,
        team_id=team_id,
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        position=Position2D(x=5.0, y=10.0),
        confidence=0.9,
    )


def _sample_analytics():
    return MatchAnalytics(
        match_id="M_TEST",
        rally_metrics=[
            RallyMetrics(
                point_id="S_001_01_01",
                rally_length=4,
                duration_ms=2000.0,
                wall_bounces=1,
            ),
        ],
        player_metrics=[
            PlayerMetrics(
                player_id="P_001",
                winners=2,
                errors=1,
                total_shots=5,
                serve_percentage=0.4,
                shot_type_counts={"volley": 3, "smash": 2},
            ),
        ],
        team_metrics=[
            TeamMetrics(
                team_id="T_1",
                net_control_pct=0.65,
                avg_pair_distance=4.5,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMatchReporter:
    def test_compute_returns_analytics(self):
        reporter = MatchReporter()
        shots = [
            _make_shot(player_id="P_001", outcome=ShotOutcome.NEXT_SHOT, timestamp_ms=0),
            _make_shot(player_id="P_002", team_id="T_2", outcome=ShotOutcome.WINNER, timestamp_ms=1000),
        ]
        point = _make_point(shots)
        pfs = [
            _make_player_frame(0, player_id="P_001", team_id="T_1"),
            _make_player_frame(0, player_id="P_002", team_id="T_2"),
        ]
        analytics = reporter.compute([point], pfs, "M_001")
        assert isinstance(analytics, MatchAnalytics)
        assert analytics.match_id == "M_001"
        assert len(analytics.rally_metrics) == 1
        assert len(analytics.player_metrics) == 2

    def test_to_json(self, tmp_path):
        reporter = MatchReporter()
        analytics = _sample_analytics()
        out_path = tmp_path / "analytics.json"
        reporter.to_json(analytics, out_path)

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["match_id"] == "M_TEST"
        assert len(data["rally_metrics"]) == 1
        assert data["player_metrics"][0]["winners"] == 2

    def test_to_json_creates_dirs(self, tmp_path):
        reporter = MatchReporter()
        analytics = _sample_analytics()
        out_path = tmp_path / "nested" / "dir" / "analytics.json"
        reporter.to_json(analytics, out_path)
        assert out_path.exists()

    def test_to_html_returns_string(self):
        reporter = MatchReporter()
        analytics = _sample_analytics()
        html = reporter.to_html(analytics)
        assert isinstance(html, str)
        assert "M_TEST" in html
        assert "<!DOCTYPE html>" in html

    def test_to_html_contains_tables(self):
        reporter = MatchReporter()
        analytics = _sample_analytics()
        html = reporter.to_html(analytics)
        assert "P_001" in html
        assert "T_1" in html
        assert "Winners" in html

    def test_to_html_writes_file(self, tmp_path):
        reporter = MatchReporter()
        analytics = _sample_analytics()
        out_path = tmp_path / "report.html"
        reporter.to_html(analytics, path=out_path)
        assert out_path.exists()
        content = out_path.read_text()
        assert "M_TEST" in content

    def test_to_html_empty_analytics(self):
        reporter = MatchReporter()
        analytics = MatchAnalytics(match_id="M_EMPTY")
        html = reporter.to_html(analytics)
        assert "M_EMPTY" in html
        assert "No rally data" in html

    def test_to_html_contains_plotly(self):
        reporter = MatchReporter()
        analytics = _sample_analytics()
        html = reporter.to_html(analytics)
        assert "plotly" in html.lower()

    def test_roundtrip_json(self, tmp_path):
        reporter = MatchReporter()
        analytics = _sample_analytics()
        out_path = tmp_path / "rt.json"
        reporter.to_json(analytics, out_path)
        loaded = MatchAnalytics.model_validate_json(out_path.read_text())
        assert loaded.match_id == analytics.match_id
        assert loaded.rally_metrics[0].rally_length == 4
