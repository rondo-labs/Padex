"""
Project: Padex
File Created: 2026-03-08
Author: Xingnan Zhu
File Name: run_tracking_only.py
Description:
    Run tracking pipeline only (no pose, no shot detection, no export).
    Generates a tracking cache pickle for weak label generation.

Usage:
    uv run python scripts/run_tracking_only.py <VIDEO_PATH> [CALIBRATION_JSON]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tracking_only")

PROJECT_ROOT = Path(__file__).parent.parent


def main() -> None:
    from padex.calibration import interactive_calibrate
    from padex.pipeline import Padex

    if len(sys.argv) < 2:
        logger.error("Usage: run_tracking_only.py <VIDEO_PATH> [CALIBRATION_JSON]")
        sys.exit(1)

    video = Path(sys.argv[1])
    if not video.exists():
        logger.error("Video not found: %s", video)
        sys.exit(1)

    # Resolve calibration
    calibration: Path | None = None
    if len(sys.argv) > 2:
        calibration = Path(sys.argv[2])
    else:
        cal_sibling = video.with_name(video.stem + "_calibration.json")
        if cal_sibling.exists():
            calibration = cal_sibling
            logger.info("Using existing calibration: %s", calibration)
        else:
            logger.info("No calibration — launching interactive calibration")
            cal = interactive_calibrate(video)
            if cal is None:
                logger.error("Calibration required. Exiting.")
                sys.exit(1)
            calibration = video.with_name(video.stem + "_calibration.json")

    # Resolve V3 weights (local assets/weights/ takes priority over ~/.padex/)
    v3_weights = PROJECT_ROOT / "assets" / "weights" / "TrackNetV3_best.pt"
    use_v3 = v3_weights.exists()
    if use_v3:
        logger.info("Using TrackNet V3 weights: %s", v3_weights)
    else:
        logger.info("TrackNet V3 weights not found, falling back to V2")

    # Run tracking only (no pose → much faster)
    try:
        padex = Padex(
            video_path=video,
            calibration=calibration,
            enable_pose=False,
            cache_dir=PROJECT_ROOT / "output",
            use_tracknet_v3=use_v3,
            ball_model_path=v3_weights if use_v3 else None,
        )
        logger.info("Padex initialized. Starting tracking...")

        # Quick video info
        from padex.io.video import VideoReader
        with VideoReader(video) as vr:
            logger.info("Video: %d frames, %.1f fps", vr.frame_count, vr.fps)

        # Just trigger tracking (cached automatically)
        result = padex._run_tracking()
        logger.info(
            "Done. %d player frames, %d ball frames. Cache saved to output/",
            len(result.player_frames),
            len(result.ball_frames),
        )
    except Exception:
        logger.exception("Tracking failed")


if __name__ == "__main__":
    main()
