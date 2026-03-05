"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: metrics.py
Description:
    Metric computation from event data.
    Computes rally-level, player-level, and team-level metrics.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from padex.schemas.events import (
    BounceType,
    Point,
    Shot,
    ShotOutcome,
    ShotType,
)
from padex.schemas.tactics import (
    MatchAnalytics,
    PlayerMetrics,
    RallyMetrics,
    TeamMetrics,
)
from padex.schemas.tracking import PlayerFrame

# Shot categories for attack/defense switches
_ATTACK_TYPES = {
    ShotType.SMASH,
    ShotType.SMASH_X3,
    ShotType.SMASH_X4,
    ShotType.BANDEJA,
    ShotType.VIBORA,
    ShotType.VOLLEY,
    ShotType.DROP_SHOT,
}
_DEFENSE_TYPES = {
    ShotType.LOB,
    ShotType.WALL_RETURN,
    ShotType.CONTRA_PARED,
}

NET_ZONE_THRESHOLD = 17.0  # y < 17 = net zone (service line)


class MetricsCalculator:
    """Computes rally-level, player-level, and team-level metrics."""

    def compute_rally_metrics(self, point: Point) -> RallyMetrics:
        """Compute metrics for a single rally/point."""
        wall_bounces = sum(
            1
            for s in point.shots
            for b in s.trajectory
            if b.type in (BounceType.BACK_WALL, BounceType.SIDE_WALL)
        )

        net_approaches = self._count_net_approaches(point.shots)
        switches = self._count_attack_defense_switches(point.shots)

        return RallyMetrics(
            point_id=point.point_id,
            rally_length=point.rally_length,
            duration_ms=point.duration_ms,
            net_approaches=net_approaches,
            wall_bounces=wall_bounces,
            attack_defense_switches=switches,
        )

    def compute_player_metrics(
        self,
        shots: list[Shot],
        player_frames: list[PlayerFrame],
        player_id: str,
    ) -> PlayerMetrics:
        """Aggregate metrics for a single player."""
        player_shots = [s for s in shots if s.player_id == player_id]
        total = len(player_shots)

        winners = sum(
            1 for s in player_shots if s.outcome == ShotOutcome.WINNER
        )
        errors = sum(
            1 for s in player_shots if s.outcome == ShotOutcome.ERROR
        )
        forced_errors = sum(
            1 for s in player_shots if s.outcome == ShotOutcome.FORCED_ERROR
        )

        serves = [s for s in player_shots if s.shot_type == ShotType.SERVE]
        serve_pct = len(serves) / total if total > 0 else None

        shot_type_counts = dict(
            Counter(s.shot_type.value for s in player_shots)
        )

        # Shot type success: winners per shot type / total of that type
        type_groups: dict[str, list[Shot]] = defaultdict(list)
        for s in player_shots:
            type_groups[s.shot_type.value].append(s)
        shot_type_success = {}
        for st, group in type_groups.items():
            w = sum(1 for s in group if s.outcome == ShotOutcome.WINNER)
            shot_type_success[st] = w / len(group) if group else 0.0

        distance = self._compute_distance(player_frames, player_id)
        avg_depth = self._compute_avg_depth(player_frames, player_id)

        return PlayerMetrics(
            player_id=player_id,
            winners=winners,
            errors=errors,
            forced_errors=forced_errors,
            serve_percentage=serve_pct,
            total_shots=total,
            shot_type_counts=shot_type_counts,
            shot_type_success=shot_type_success,
            distance_covered_m=distance,
            avg_position_depth=avg_depth,
        )

    def compute_team_metrics(
        self,
        shots: list[Shot],
        player_frames: list[PlayerFrame],
        team_id: str,
    ) -> TeamMetrics:
        """Aggregate metrics for a team (pair)."""
        team_players = {
            pf.player_id
            for pf in player_frames
            if pf.team_id == team_id
        }
        team_pfs = [
            pf for pf in player_frames if pf.player_id in team_players
        ]
        team_shots = [s for s in shots if s.player_id in team_players]

        net_control = self._compute_net_control(team_pfs)
        avg_pair_dist = self._compute_avg_pair_distance(team_pfs)
        formation_switches = self._count_formation_switches(team_pfs)
        wall_util = self._compute_wall_utilization(team_shots)
        transition_eff = self._compute_transition_efficiency(team_shots)

        return TeamMetrics(
            team_id=team_id,
            net_control_pct=net_control,
            formation_switches=formation_switches,
            avg_pair_distance=avg_pair_dist,
            wall_utilization_index=wall_util,
            transition_efficiency=transition_eff,
        )

    def compute_match_analytics(
        self,
        points: list[Point],
        player_frames: list[PlayerFrame],
        match_id: str,
    ) -> MatchAnalytics:
        """Compute all metrics for a match."""
        all_shots = [s for p in points for s in p.shots]

        rally_metrics = [self.compute_rally_metrics(p) for p in points]

        player_ids = {s.player_id for s in all_shots}
        player_metrics = [
            self.compute_player_metrics(all_shots, player_frames, pid)
            for pid in sorted(player_ids)
        ]

        team_ids = {
            pf.team_id for pf in player_frames if pf.team_id is not None
        }
        team_metrics = [
            self.compute_team_metrics(all_shots, player_frames, tid)
            for tid in sorted(team_ids)
        ]

        return MatchAnalytics(
            match_id=match_id,
            rally_metrics=rally_metrics,
            player_metrics=player_metrics,
            team_metrics=team_metrics,
        )

    # -- Private helpers ---------------------------------------------------

    @staticmethod
    def _count_net_approaches(shots: list[Shot]) -> int:
        """Count shots where player position is in the net zone."""
        return sum(
            1 for s in shots if s.position.y >= (10.0 - 3.0)
        )

    @staticmethod
    def _count_attack_defense_switches(shots: list[Shot]) -> int:
        """Count transitions between attack and defense shot types."""
        if len(shots) < 2:
            return 0

        switches = 0
        prev_is_attack = None
        for s in shots:
            if s.shot_type in _ATTACK_TYPES:
                is_attack = True
            elif s.shot_type in _DEFENSE_TYPES:
                is_attack = False
            else:
                continue

            if prev_is_attack is not None and is_attack != prev_is_attack:
                switches += 1
            prev_is_attack = is_attack

        return switches

    @staticmethod
    def _compute_distance(
        player_frames: list[PlayerFrame], player_id: str
    ) -> float | None:
        """Total distance covered by a player across frames."""
        positions = [
            (pf.position.x, pf.position.y)
            for pf in sorted(player_frames, key=lambda p: p.frame_id)
            if pf.player_id == player_id and pf.position is not None
        ]
        if len(positions) < 2:
            return None

        total = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    @staticmethod
    def _compute_avg_depth(
        player_frames: list[PlayerFrame], player_id: str
    ) -> float | None:
        """Average y-coordinate (depth) for a player."""
        y_vals = [
            pf.position.y
            for pf in player_frames
            if pf.player_id == player_id and pf.position is not None
        ]
        if not y_vals:
            return None
        return sum(y_vals) / len(y_vals)

    @staticmethod
    def _compute_net_control(team_pfs: list[PlayerFrame]) -> float | None:
        """Percentage of frames where at least one player is in net zone."""
        if not team_pfs:
            return None

        by_frame: dict[int, list[PlayerFrame]] = defaultdict(list)
        for pf in team_pfs:
            by_frame[pf.frame_id].append(pf)

        net_frames = 0
        for fid, pfs in by_frame.items():
            if any(
                pf.position is not None and pf.position.y >= NET_ZONE_THRESHOLD
                for pf in pfs
            ):
                # Net zone: closer to net (y closer to 10)
                # Actually net zone for the team depends on which half they're on.
                # For simplicity, y in [7, 13] is net zone.
                pass
            if any(
                pf.position is not None
                and 3.0 <= pf.position.y <= 17.0
                for pf in pfs
            ):
                net_frames += 1

        return net_frames / len(by_frame) if by_frame else None

    @staticmethod
    def _compute_avg_pair_distance(
        team_pfs: list[PlayerFrame],
    ) -> float | None:
        """Average distance between the two team players per frame."""
        by_frame: dict[int, list[PlayerFrame]] = defaultdict(list)
        for pf in team_pfs:
            if pf.position is not None:
                by_frame[pf.frame_id].append(pf)

        distances = []
        for fid, pfs in by_frame.items():
            if len(pfs) >= 2:
                p1 = pfs[0].position
                p2 = pfs[1].position
                d = math.sqrt(
                    (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
                )
                distances.append(d)

        return sum(distances) / len(distances) if distances else None

    @staticmethod
    def _count_formation_switches(
        team_pfs: list[PlayerFrame],
    ) -> int:
        """Count times the two players swap left/right position."""
        by_frame: dict[int, list[PlayerFrame]] = defaultdict(list)
        for pf in team_pfs:
            if pf.position is not None:
                by_frame[pf.frame_id].append(pf)

        sorted_frames = sorted(by_frame.keys())
        switches = 0
        prev_order = None
        for fid in sorted_frames:
            pfs = by_frame[fid]
            if len(pfs) < 2:
                continue
            # Determine who is on the left (lower x)
            ids_sorted = sorted(
                [pf.player_id for pf in pfs],
                key=lambda pid: next(
                    pf.position.x for pf in pfs if pf.player_id == pid
                ),
            )
            order = tuple(ids_sorted)
            if prev_order is not None and order != prev_order:
                switches += 1
            prev_order = order

        return switches

    @staticmethod
    def _compute_wall_utilization(shots: list[Shot]) -> float | None:
        """Ratio of wall-related shots to total shots."""
        if not shots:
            return None
        wall_shots = sum(
            1
            for s in shots
            if s.shot_type
            in (ShotType.WALL_RETURN, ShotType.CONTRA_PARED)
        )
        return wall_shots / len(shots)

    @staticmethod
    def _compute_transition_efficiency(shots: list[Shot]) -> float | None:
        """Ratio of attack winners to total attack shots."""
        attack_shots = [s for s in shots if s.shot_type in _ATTACK_TYPES]
        if not attack_shots:
            return None
        winners = sum(
            1 for s in attack_shots if s.outcome == ShotOutcome.WINNER
        )
        return winners / len(attack_shots)
