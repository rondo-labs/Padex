"""
Project: Padex
File Created: 2026-03-04
Author: Xingnan Zhu
File Name: test_video_writer.py
Description:
    Tests for VideoWriter.
"""

import numpy as np
import pytest

from padex.io.video import VideoWriter


class TestVideoWriter:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "out.mp4"
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with VideoWriter(out, fps=30.0, frame_size=(640, 480)) as writer:
            writer.write(frame)
        assert out.exists()

    def test_frame_count(self, tmp_path):
        out = tmp_path / "out.mp4"
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with VideoWriter(out, fps=30.0, frame_size=(640, 480)) as writer:
            for _ in range(5):
                writer.write(frame)
            assert writer.frame_count == 5

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "out.mp4"
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with VideoWriter(out, fps=30.0, frame_size=(640, 480)) as writer:
            writer.write(frame)
        assert out.exists()

    def test_context_manager_releases(self, tmp_path):
        out = tmp_path / "out.mp4"
        writer = VideoWriter(out, fps=30.0, frame_size=(640, 480))
        writer.__enter__()
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        writer.__exit__(None, None, None)
        # After exit, writer should be released
        assert not writer._writer.isOpened()

    def test_release_idempotent(self, tmp_path):
        out = tmp_path / "out.mp4"
        writer = VideoWriter(out, fps=30.0, frame_size=(640, 480))
        writer.release()
        writer.release()  # should not raise
