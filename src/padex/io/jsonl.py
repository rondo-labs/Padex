"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: jsonl.py
Description:
    JSONL read/write for event data.
    Handles streaming serialization of Pydantic models.
"""

from __future__ import annotations

from pathlib import Path


def write_jsonl(records, path: str | Path) -> None:
    """Write a list of Pydantic models to JSONL."""
    raise NotImplementedError


def read_jsonl(path: str | Path, model_class=None):
    """Read JSONL file, optionally validating against a Pydantic model."""
    raise NotImplementedError
