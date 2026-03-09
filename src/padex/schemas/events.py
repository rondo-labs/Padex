"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: events.py
Description:
    Pydantic models for Layer 2: Event data.
    Defines ShotType, BounceType, Shot, Point, Game, Set, MatchStructure.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from padex.schemas.tracking import Position2D, Position3D


class ShotType(str, Enum):
    """Padel-specific shot type taxonomy."""

    # Serve
    SERVE = "serve"
    # Net play
    VOLLEY = "volley"
    BANDEJA = "bandeja"
    VIBORA = "vibora"
    SMASH = "smash"
    SMASH_X3 = "smash_x3"
    SMASH_X4 = "smash_x4"
    DROP_SHOT = "drop_shot"
    # Baseline
    BAJADA = "bajada"
    GROUNDSTROKE_FH = "groundstroke_fh"
    GROUNDSTROKE_BH = "groundstroke_bh"
    # Transition
    CHIQUITA = "chiquita"
    # Defensive
    LOB = "lob"
    WALL_RETURN = "wall_return"
    CONTRA_PARED = "contra_pared"
    # Fallback
    UNKNOWN = "unknown"


class BounceType(str, Enum):
    """Surface where the ball bounced."""

    GROUND = "ground"
    BACK_WALL = "back_wall"
    SIDE_WALL = "side_wall"
    BACK_FENCE = "back_fence"
    SIDE_FENCE = "side_fence"
    CORNER = "corner"
    NET = "net"


class BallEventType(str, Enum):
    """Per-frame ball event state for ML-based detection."""

    FLYING = "flying"
    BOUNCE = "bounce"
    HIT = "hit"
    OCCLUDED = "occluded"


class ShotOutcome(str, Enum):
    """How the shot resolved."""

    WINNER = "winner"
    ERROR = "error"
    FORCED_ERROR = "forced_error"
    NEXT_SHOT = "next_shot"
    LET = "let"


class Bounce(BaseModel):
    """A single bounce in the ball trajectory."""

    type: BounceType
    position: Position2D | Position3D
    timestamp_ms: float | None = None


class Shot(BaseModel):
    """Atomic unit of padel data — a single shot with its full trajectory."""

    shot_id: str = Field(
        ...,
        pattern=r"^S_\d{3}_\d{2}_\d{2}_\d{3}$",
        description="Hierarchical ID: S_{set}_{game}_{point}_{shot}",
    )
    timestamp_ms: float
    player_id: str
    team_id: str
    position: Position2D
    shot_type: ShotType = ShotType.UNKNOWN
    trajectory: list[Bounce] = Field(default_factory=list)
    outcome: ShotOutcome | None = None
    confidence: float = Field(0.0, ge=0, le=1)


class Point(BaseModel):
    """A single point (rally) in the match."""

    point_id: str = Field(
        ...,
        pattern=r"^S_\d{3}_\d{2}_\d{2}$",
        description="Hierarchical ID: S_{set}_{game}_{point}",
    )
    shots: list[Shot] = Field(default_factory=list)
    winner_team_id: str | None = None
    duration_ms: float | None = None
    rally_length: int = Field(0, description="Number of shots in the rally")


class Game(BaseModel):
    """A single game within a set."""

    game_id: str
    points: list[Point] = Field(default_factory=list)
    server_player_id: str | None = None
    winner_team_id: str | None = None


class Set(BaseModel):
    """A single set within a match."""

    set_id: str
    games: list[Game] = Field(default_factory=list)
    score: tuple[int, int] | None = None
    winner_team_id: str | None = None


class MatchStructure(BaseModel):
    """Top-level match structure."""

    schema_version: str = "0.1.0"
    match_id: str
    date: str | None = None
    venue: str | None = None
    teams: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of teams, each with player_ids and team_id",
    )
    sets: list[Set] = Field(default_factory=list)
    winner_team_id: str | None = None
