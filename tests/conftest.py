"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: conftest.py
Description:
    Shared test fixtures for Padex test suite.
"""

from pathlib import Path

import pytest

VIDEO_PATH = Path("assets/raw/video/TapiaChingottoLebronGalanHighlights_1080p.mp4")


@pytest.fixture
def video_path():
    """Path to the test highlights video."""
    return VIDEO_PATH


@pytest.fixture
def has_video():
    """Whether the test video is available."""
    return VIDEO_PATH.exists()
