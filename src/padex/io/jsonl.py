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

import json
import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def write_jsonl(records: list[BaseModel], path: str | Path) -> None:
    """Write a list of Pydantic models to JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")

    logger.debug("Wrote %d records to %s", len(records), path)


def read_jsonl(path: str | Path, model_class: type[T] | None = None) -> list[T] | list[dict]:
    """Read JSONL file, optionally validating against a Pydantic model.

    Args:
        path: Path to the JSONL file.
        model_class: If provided, each line is validated against this model.

    Returns:
        List of model instances (if model_class given) or list of dicts.
    """
    path = Path(path)
    results: list = []

    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON at line %d in %s", line_num, path)
                continue

            if model_class is not None:
                results.append(model_class.model_validate(obj))
            else:
                results.append(obj)

    logger.debug("Read %d records from %s", len(results), path)
    return results
