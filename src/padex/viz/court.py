"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: court.py
Description:
    Court rendering utilities using Plotly.
    Renders a 2D padel court diagram with position overlays.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from padex.schemas.events import Bounce
from padex.schemas.tracking import Position2D


class CourtRenderer:
    """Renders a 2D padel court diagram using Plotly."""

    COURT_WIDTH = 10.0
    COURT_LENGTH = 20.0
    NET_Y = 10.0
    SERVICE_LINE_OFFSET = 7.0  # 7m from net

    # Court line coordinates
    _SERVICE_Y_NEAR = 3.0
    _SERVICE_Y_FAR = 17.0
    _CENTER_X = 5.0

    LINE_COLOR = "white"
    COURT_COLOR = "#2E7D32"
    NET_COLOR = "#BDBDBD"

    def draw(self, title: str = "Padel Court") -> go.Figure:
        """Draw an empty padel court. Returns a Plotly figure."""
        fig = go.Figure()

        # Court background
        fig.add_shape(
            type="rect",
            x0=0,
            y0=0,
            x1=self.COURT_WIDTH,
            y1=self.COURT_LENGTH,
            fillcolor=self.COURT_COLOR,
            line=dict(color=self.LINE_COLOR, width=2),
        )

        # Net
        fig.add_shape(
            type="line",
            x0=0,
            y0=self.NET_Y,
            x1=self.COURT_WIDTH,
            y1=self.NET_Y,
            line=dict(color=self.NET_COLOR, width=3, dash="dash"),
        )

        # Service lines
        for sy in (self._SERVICE_Y_NEAR, self._SERVICE_Y_FAR):
            fig.add_shape(
                type="line",
                x0=0,
                y0=sy,
                x1=self.COURT_WIDTH,
                y1=sy,
                line=dict(color=self.LINE_COLOR, width=1),
            )

        # Center service lines
        fig.add_shape(
            type="line",
            x0=self._CENTER_X,
            y0=self._SERVICE_Y_NEAR,
            x1=self._CENTER_X,
            y1=self.NET_Y,
            line=dict(color=self.LINE_COLOR, width=1),
        )
        fig.add_shape(
            type="line",
            x0=self._CENTER_X,
            y0=self.NET_Y,
            x1=self._CENTER_X,
            y1=self._SERVICE_Y_FAR,
            line=dict(color=self.LINE_COLOR, width=1),
        )

        fig.update_layout(
            title=title,
            xaxis=dict(
                range=[-0.5, self.COURT_WIDTH + 0.5],
                constrain="domain",
                showgrid=False,
                zeroline=False,
                title="Width (m)",
            ),
            yaxis=dict(
                range=[-0.5, self.COURT_LENGTH + 0.5],
                scaleanchor="x",
                showgrid=False,
                zeroline=False,
                title="Length (m)",
            ),
            plot_bgcolor="#1B5E20",
            width=500,
            height=900,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return fig

    def plot_positions(
        self,
        positions: list[Position2D],
        fig: go.Figure | None = None,
        name: str = "Positions",
        color: str = "yellow",
        size: int = 8,
    ) -> go.Figure:
        """Overlay positions on the court."""
        if fig is None:
            fig = self.draw()

        xs = [p.x for p in positions]
        ys = [p.y for p in positions]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=name,
                marker=dict(color=color, size=size),
            )
        )

        return fig

    def plot_heatmap(
        self,
        heatmap: np.ndarray,
        fig: go.Figure | None = None,
        colorscale: str = "Hot",
        opacity: float = 0.6,
    ) -> go.Figure:
        """Overlay a heatmap on the court."""
        if fig is None:
            fig = self.draw()

        fig.add_trace(
            go.Heatmap(
                z=heatmap,
                x0=0,
                dx=self.COURT_WIDTH / heatmap.shape[1],
                y0=0,
                dy=self.COURT_LENGTH / heatmap.shape[0],
                colorscale=colorscale,
                opacity=opacity,
                showscale=True,
                name="Heatmap",
            )
        )

        return fig

    def plot_trajectory(
        self,
        bounces: list[Bounce],
        fig: go.Figure | None = None,
        color: str = "cyan",
    ) -> go.Figure:
        """Plot a bounce trajectory on the court."""
        if fig is None:
            fig = self.draw()

        if not bounces:
            return fig

        xs = []
        ys = []
        texts = []
        for b in bounces:
            if b.position is not None:
                xs.append(b.position.x)
                ys.append(b.position.y)
                texts.append(b.type.value)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers+text",
                name="Trajectory",
                line=dict(color=color, width=2),
                marker=dict(color=color, size=10, symbol="diamond"),
                text=texts,
                textposition="top center",
                textfont=dict(size=9),
            )
        )

        return fig
