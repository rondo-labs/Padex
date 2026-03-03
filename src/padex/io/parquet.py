"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: parquet.py
Description:
    Parquet read/write for tracking data.
    Handles serialization of player/ball frames to columnar format.
"""

from __future__ import annotations

from pathlib import Path


def write_tracking_parquet(data, path: str | Path) -> None:
    """Write tracking data (player/ball frames) to Parquet."""
    raise NotImplementedError


def read_tracking_parquet(path: str | Path):
    """Read tracking data from Parquet. Returns a Polars DataFrame."""
    raise NotImplementedError
