"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: video.py
Description:
    Video frame extraction utilities.
    Reads video files and yields frames with timestamps.
"""

from __future__ import annotations

from pathlib import Path


class VideoReader:
    """Reads video files and yields frames."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)

    def frames(self):
        """Yield (frame_id, timestamp_ms, frame) tuples."""
        raise NotImplementedError

    @property
    def fps(self) -> float:
        """Video frames per second."""
        raise NotImplementedError

    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        raise NotImplementedError
