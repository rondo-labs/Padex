"""
Padex Quickstart
================

This example shows how to analyze a padel match video using Padex.

Prerequisites:
    pip install padex
    # or: uv add padex

Steps:
    1. First run: interactive calibration UI opens (click court keypoints)
    2. Calibration is saved next to the video for future runs
    3. Pipeline runs: tracking → bounce detection → shot classification
    4. Annotated video is exported with shot type labels
"""

import padex

# ── Path to your padel match video ──────────────────────────────────────
VIDEO = "path/to/your/match.mp4"

# ── Step 1: Court calibration (only needed once per camera angle) ───────
#
# If no calibration file exists next to the video, run interactive
# calibration first. A window will open where you click court keypoints.
#
# The result is saved as <video_stem>_calibration.json and reused
# automatically on future runs.

cal = padex.interactive_calibrate(VIDEO)
# Or skip this step if you already have a calibration file:
# cal = "path/to/match_calibration.json"

# ── Step 2: Run the full pipeline ───────────────────────────────────────

result = padex.process(VIDEO, calibration=cal)

print(f"Detected {len(result.shots)} shots and {len(result.bounces)} bounces")
for shot in result.shots:
    print(f"  {shot.shot_type.value:20s} | player={shot.player_id} | t={shot.timestamp_ms:.0f}ms")

# ── Step 3: Export annotated video ──────────────────────────────────────

padex.export_video(result, VIDEO, "output/annotated.mp4")
print("Annotated video saved to output/annotated.mp4")


# ── Alternative: CLI usage ──────────────────────────────────────────────
#
# From the command line, you can do everything in one command:
#
#   padex process path/to/match.mp4
#
# This will auto-launch calibration if needed, then run the pipeline
# and export an annotated video to output/<timestamp>/shot_detection.mp4
#
# To calibrate separately:
#
#   padex calibrate path/to/match.mp4 -o my_calibration.json
#   padex process path/to/match.mp4 --calibration my_calibration.json
