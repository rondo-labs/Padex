"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: pipeline.py
Description:
    Orchestrates the full tracking pipeline across all detectors.
    End-to-end: video → structured tracking data.
"""

from __future__ import annotations


class TrackingPipeline:
    """End-to-end tracking pipeline: video → structured tracking data."""

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path

    def run(self):
        """Run the full tracking pipeline on the video."""
        raise NotImplementedError
