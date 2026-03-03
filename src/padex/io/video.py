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

from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Reads video files and yields frames with metadata."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

    def frames(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
        step: int = 1,
    ) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """Yield (frame_id, timestamp_ms, frame) tuples.

        Args:
            start_frame: First frame to read.
            end_frame: Last frame (exclusive). None = all frames.
            step: Read every Nth frame (for sampling).
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_id = start_frame
        end = end_frame or self.frame_count

        while frame_id < end:
            ret, frame = self._cap.read()
            if not ret:
                break

            if (frame_id - start_frame) % step == 0:
                timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                yield frame_id, timestamp_ms, frame

            frame_id += 1

    def read_frame(self, frame_id: int) -> np.ndarray | None:
        """Read a single specific frame by index."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self._cap.read()
        return frame if ret else None

    @property
    def fps(self) -> float:
        """Video frames per second."""
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame_size(self) -> tuple[int, int]:
        """Returns (width, height)."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *args) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def __del__(self) -> None:
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()
