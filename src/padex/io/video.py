"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: video.py
Description:
    Video I/O utilities.
    VideoReader reads video files and yields frames with timestamps.
    VideoWriter writes annotated frames to output video files.
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


class VideoWriter:
    """Writes frames to a video file using OpenCV."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        """Initialize video writer.

        Args:
            path: Output video file path.
            fps: Frames per second.
            frame_size: (width, height) of output frames.
            codec: FourCC codec string (default: mp4v).
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(
            str(self.path), fourcc, fps, frame_size,
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.path}")

        self._frame_count = 0

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame."""
        self._writer.write(frame)
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count

    def release(self) -> None:
        """Release the writer."""
        if self._writer.isOpened():
            self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *args) -> None:
        self.release()

    def __del__(self) -> None:
        if hasattr(self, "_writer") and self._writer.isOpened():
            self._writer.release()
