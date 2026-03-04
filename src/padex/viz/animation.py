"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: animation.py
Description:
    Rally replay animation using Plotly.
    Animates a rally showing player and ball movement.
"""

from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from padex.schemas.events import Shot
from padex.schemas.tracking import BallFrame, BallVisibility, PlayerFrame
from padex.viz.court import CourtRenderer


class RallyAnimator:
    """Animates a rally showing player and ball movement using Plotly."""

    TEAM_COLORS = {"T_1": "#1E88E5", "T_2": "#E53935"}
    BALL_COLOR = "#FFEB3B"
    DEFAULT_COLOR = "#9E9E9E"

    def animate(
        self,
        player_frames: list[PlayerFrame],
        ball_frames: list[BallFrame],
        shots: list[Shot] | None = None,
        fps: int = 10,
    ) -> go.Figure:
        """Create an animated Plotly figure of a rally.

        Args:
            player_frames: Player tracking data.
            ball_frames: Ball tracking data.
            shots: Optional shot events to annotate.
            fps: Frames per second for animation.

        Returns:
            Plotly Figure with animation frames.
        """
        renderer = CourtRenderer()
        fig = renderer.draw(title="Rally Animation")

        # Group by frame_id
        players_by_frame: dict[int, list[PlayerFrame]] = defaultdict(list)
        for pf in player_frames:
            players_by_frame[pf.frame_id].append(pf)

        ball_by_frame: dict[int, BallFrame] = {}
        for bf in ball_frames:
            ball_by_frame[bf.frame_id] = bf

        shot_frames = set()
        if shots:
            # Map shot timestamps to nearest frame
            for s in shots:
                nearest = min(
                    ball_by_frame.keys(),
                    key=lambda fid: abs(
                        ball_by_frame[fid].timestamp_ms - s.timestamp_ms
                    ),
                    default=None,
                )
                if nearest is not None:
                    shot_frames.add(nearest)

        all_frame_ids = sorted(
            set(players_by_frame.keys()) | set(ball_by_frame.keys())
        )

        if not all_frame_ids:
            return fig

        # Build animation frames
        frames = []
        for fid in all_frame_ids:
            data = []

            # Players
            pfs = players_by_frame.get(fid, [])
            for pf in pfs:
                if pf.position is None:
                    continue
                color = self.TEAM_COLORS.get(
                    pf.team_id or "", self.DEFAULT_COLOR
                )
                data.append(
                    go.Scatter(
                        x=[pf.position.x],
                        y=[pf.position.y],
                        mode="markers+text",
                        marker=dict(color=color, size=14),
                        text=[pf.player_id[-3:]],
                        textposition="top center",
                        showlegend=False,
                    )
                )

            # Ball
            bf = ball_by_frame.get(fid)
            if bf and bf.position is not None:
                ball_symbol = "star" if fid in shot_frames else "circle"
                ball_size = 14 if fid in shot_frames else 10
                data.append(
                    go.Scatter(
                        x=[bf.position.x],
                        y=[bf.position.y],
                        mode="markers",
                        marker=dict(
                            color=self.BALL_COLOR,
                            size=ball_size,
                            symbol=ball_symbol,
                        ),
                        showlegend=False,
                    )
                )

            frames.append(
                go.Frame(data=data, name=str(fid))
            )

        fig.frames = frames

        # Set initial data from first frame
        if frames:
            for trace in frames[0].data:
                fig.add_trace(trace)

        # Animation controls
        duration_ms = 1000 // fps
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.5,
                    xanchor="center",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(
                                        duration=duration_ms, redraw=True
                                    ),
                                    fromcurrent=True,
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            args=[
                                [str(fid)],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                ),
                            ],
                            label=str(fid),
                            method="animate",
                        )
                        for fid in all_frame_ids
                    ],
                    x=0.1,
                    len=0.8,
                    y=-0.05,
                    currentvalue=dict(
                        prefix="Frame: ", visible=True
                    ),
                )
            ],
        )

        return fig
