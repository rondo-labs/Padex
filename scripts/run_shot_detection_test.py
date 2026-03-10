"""
Project: Padex
File Created: 2026-03-06
Author: Xingnan Zhu
File Name: run_shot_detection_test.py
Description:
    Example script: runs the full Padex pipeline on a video and exports
    an annotated video with shot type labels.

    This is a thin wrapper around the padex library API.

Usage:
    uv run python scripts/run_shot_detection_test.py [VIDEO_PATH]
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shot_detection_test")

PROJECT_ROOT = Path(__file__).parent.parent


def main() -> None:
    from padex import Padex
    from padex.calibration import interactive_calibrate

    # Parse video path
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    else:
        video = PROJECT_ROOT / "assets" / "processed" / "video" / "ThreeTest.mp4"

    if not video.exists():
        logger.error("Video not found: %s", video)
        sys.exit(1)

    # Resolve calibration: CLI arg → sibling JSON → interactive
    calibration: Path | None = None
    if len(sys.argv) > 2:
        calibration = Path(sys.argv[2])
    else:
        cal_sibling = video.with_name(video.stem + "_calibration.json")
        if cal_sibling.exists():
            calibration = cal_sibling
            logger.info("Using existing calibration: %s", calibration)
        else:
            logger.info("No calibration file found — launching interactive calibration")
            cal = interactive_calibrate(video)
            if cal is None:
                logger.error("Calibration required. Exiting.")
                sys.exit(1)
            calibration = video.with_name(video.stem + "_calibration.json")

    # Resolve V3 weights
    v3_weights = PROJECT_ROOT / "assets" / "weights" / "TrackNetV3_best.pt"
    use_v3 = v3_weights.exists()
    if use_v3:
        logger.info("Using TrackNet V3 weights: %s", v3_weights)

    # Run pipeline
    padex = Padex(
        video_path=video,
        calibration=calibration,
        cache_dir=PROJECT_ROOT / "output",
        use_tracknet_v3=use_v3,
        ball_model_path=v3_weights if use_v3 else None,
        use_physics_events=True,
    )
    result = padex.run()

    # Export annotated video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "output" / timestamp
    out_video = out_dir / "shot_detection.mp4"
    padex.export_video(result, out_video)

    # Summary
    logger.info("=== Summary ===")
    logger.info("Total shots detected: %d", len(result.shots))
    logger.info("Shot type breakdown:")
    from collections import Counter
    for shot_type, count in Counter(s.shot_type.value for s in result.shots).most_common():
        logger.info("  %-20s: %d", shot_type, count)
    logger.info("Bounces: %d", len(result.bounces))
    logger.info("Output: %s", out_dir)


if __name__ == "__main__":
    main()
