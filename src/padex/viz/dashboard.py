"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: dashboard.py
Description:
    Match overview dashboard using Plotly.
    Generates a match overview dashboard from analytics data.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from padex.schemas.tactics import MatchAnalytics


class MatchDashboard:
    """Generates a match overview dashboard using Plotly subplots."""

    def generate(
        self,
        analytics: MatchAnalytics,
    ) -> go.Figure:
        """Create a dashboard from MatchAnalytics.

        Includes:
        1. Rally length distribution
        2. Player shot type distribution
        3. Winners/Errors comparison
        4. Net control comparison (teams)

        Returns:
            Plotly Figure with subplots.
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Rally Length Distribution",
                "Shot Type Distribution",
                "Winners / Errors",
                "Team Net Control",
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        self._add_rally_length_chart(fig, analytics, row=1, col=1)
        self._add_shot_type_chart(fig, analytics, row=1, col=2)
        self._add_winners_errors_chart(fig, analytics, row=2, col=1)
        self._add_net_control_chart(fig, analytics, row=2, col=2)

        fig.update_layout(
            title=f"Match Dashboard — {analytics.match_id}",
            height=800,
            width=1200,
            showlegend=True,
        )

        return fig

    def _add_rally_length_chart(
        self,
        fig: go.Figure,
        analytics: MatchAnalytics,
        row: int,
        col: int,
    ) -> None:
        lengths = [rm.rally_length for rm in analytics.rally_metrics]
        if not lengths:
            return
        fig.add_trace(
            go.Histogram(
                x=lengths,
                nbinsx=max(lengths) if lengths else 10,
                marker_color="#42A5F5",
                name="Rally Length",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Shots", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)

    def _add_shot_type_chart(
        self,
        fig: go.Figure,
        analytics: MatchAnalytics,
        row: int,
        col: int,
    ) -> None:
        for pm in analytics.player_metrics:
            if not pm.shot_type_counts:
                continue
            types = list(pm.shot_type_counts.keys())
            counts = list(pm.shot_type_counts.values())
            fig.add_trace(
                go.Bar(
                    x=types,
                    y=counts,
                    name=pm.player_id,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Shot Type", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)

    def _add_winners_errors_chart(
        self,
        fig: go.Figure,
        analytics: MatchAnalytics,
        row: int,
        col: int,
    ) -> None:
        player_ids = [pm.player_id for pm in analytics.player_metrics]
        winners = [pm.winners for pm in analytics.player_metrics]
        errors = [pm.errors for pm in analytics.player_metrics]

        fig.add_trace(
            go.Bar(
                x=player_ids,
                y=winners,
                name="Winners",
                marker_color="#66BB6A",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Bar(
                x=player_ids,
                y=errors,
                name="Errors",
                marker_color="#EF5350",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Player", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)

    def _add_net_control_chart(
        self,
        fig: go.Figure,
        analytics: MatchAnalytics,
        row: int,
        col: int,
    ) -> None:
        team_ids = []
        net_pcts = []
        for tm in analytics.team_metrics:
            if tm.net_control_pct is not None:
                team_ids.append(tm.team_id)
                net_pcts.append(tm.net_control_pct * 100)

        if not team_ids:
            return

        fig.add_trace(
            go.Bar(
                x=team_ids,
                y=net_pcts,
                marker_color=["#1E88E5", "#E53935"][: len(team_ids)],
                name="Net Control %",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Team", row=row, col=col)
        fig.update_yaxes(title_text="%", row=row, col=col)
