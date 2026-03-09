"""
Project: Padex
File Created: 2026-03-07
Author: Xingnan Zhu
File Name: weights.py
Description:
    Model weight management with automatic downloading from GitHub Releases.

    Weights are cached locally in ~/.padex/weights/ and downloaded on
    first use. Users can also specify custom paths to skip downloading.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path.home() / ".padex" / "weights"

GITHUB_RELEASE_BASE = (
    "https://github.com/rondo-labs/Padex/releases/download/v0.1.0"
)

WEIGHT_REGISTRY: dict[str, str] = {
    "yolo26m.pt": f"{GITHUB_RELEASE_BASE}/yolo26m.pt",
    "yolo26m-pose.pt": f"{GITHUB_RELEASE_BASE}/yolo26m-pose.pt",
    "ball_detection_TrackNet.pt": f"{GITHUB_RELEASE_BASE}/ball_detection_TrackNet.pt",
    "ball_detection_TrackNetV3.pt": f"{GITHUB_RELEASE_BASE}/ball_detection_TrackNetV3.pt",
}


def get_weight_path(name: str) -> Path:
    """Return local path to a weight file, downloading from GitHub if needed.

    Args:
        name: Weight filename (e.g. "yolo26m.pt").

    Returns:
        Path to the local weight file.

    Raises:
        ValueError: If the weight name is not in the registry.
        RuntimeError: If download fails.
    """
    if name not in WEIGHT_REGISTRY:
        raise ValueError(
            f"Unknown weight '{name}'. Available: {list(WEIGHT_REGISTRY.keys())}"
        )

    local_path = WEIGHTS_DIR / name
    if local_path.exists():
        return local_path

    url = WEIGHT_REGISTRY[name]
    logger.info("Downloading %s from %s", name, url)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(".tmp")

    try:
        _download_with_progress(url, tmp_path)
        shutil.move(str(tmp_path), str(local_path))
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {name}: {exc}") from exc

    logger.info("Saved to %s", local_path)
    return local_path


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with progress logging."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    logger.info(
                        "  %.1f / %.1f MB (%.0f%%)", mb, total_mb, pct
                    )
