"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: report.py
Description:
    Match report generation.
    Generates a structured match report from analytics data.
"""

from __future__ import annotations


class MatchReporter:
    """Generates a structured match report from analytics data."""

    def generate(self, match_analytics):
        """Generate a match report from MatchAnalytics."""
        raise NotImplementedError
