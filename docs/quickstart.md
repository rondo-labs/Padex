# Quick Start

This guide walks you through analyzing your first padel match video with Padex.

## Prerequisites

- Python 3.11 or later
- A padel match video file (broadcast footage)

## Step 1: Install Padex

=== "pip"

    ```bash
    pip install padex
    ```

=== "uv"

    ```bash
    uv add padex
    ```

## Step 2: Calibrate the Court

Padex needs to know where the court is in your video. The first time you process a video, you'll calibrate by clicking on court keypoints.

=== "Python"

    ```python
    import padex

    cal = padex.interactive_calibrate("match.mp4")
    ```

=== "CLI"

    ```bash
    padex calibrate match.mp4
    ```

This opens two windows:

1. **Frame selector** — Browse the video to find a frame with clear court lines. Use `A/D` to navigate, `Enter` to confirm.
2. **Keypoint labeler** — Click on 12 court keypoints (corners, net, service lines). Use `N` to skip invisible points, `Z` to undo.

The calibration is saved as `match_calibration.json` next to your video and reused automatically on future runs.

!!! tip
    You only need to calibrate once per camera angle. If all your videos use the same camera position, you can reuse the same calibration file.

## Step 3: Run the Pipeline

=== "Python"

    ```python
    import padex

    result = padex.process("match.mp4", calibration="match_calibration.json")

    print(f"Detected {len(result.shots)} shots")
    print(f"Detected {len(result.bounces)} bounces")

    for shot in result.shots:
        print(f"  {shot.shot_type.value:20s} | player={shot.player_id}")
    ```

=== "CLI"

    ```bash
    padex process match.mp4
    ```

The pipeline runs three stages:

1. **Tracking** — Detects players (YOLO + ByteTrack), ball (TrackNet), and maps positions to court coordinates
2. **Bounce detection** — Identifies ball bounces and classifies surfaces (ground, wall, fence)
3. **Shot classification** — Detects contact events and classifies shot types (volley, bandeja, chiquita, etc.)

!!! note
    The first run downloads model weights (~130MB) to `~/.padex/weights/`. Tracking takes ~5 FPS on Apple Silicon — a 2-minute video processes in about 12 minutes.

## Step 4: Export Annotated Video

=== "Python"

    ```python
    padex.export_video(result, "match.mp4", "output/annotated.mp4")
    ```

=== "CLI"

    The CLI exports automatically to `output/<timestamp>/shot_detection.mp4`.

The annotated video includes:

- Player bounding boxes with team colors
- Ball position marker
- Court overlay lines
- Shot type labels displayed on each contact
- Stats overlay (shot counts by type)

## Step 5: Iterate

Tracking results are cached automatically. On subsequent runs, only bounce detection, shot classification, and video export are re-run — taking seconds instead of minutes.

To force a fresh tracking run:

=== "Python"

    ```python
    result = padex.process("match.mp4", cache_tracking=False)
    ```

=== "CLI"

    ```bash
    padex process match.mp4 --no-cache
    ```

## Next Steps

- [Court Calibration Guide](guides/calibration.md) — Detailed calibration instructions
- [Pipeline Guide](guides/pipeline.md) — Advanced pipeline configuration
- [Shot Classification](guides/shot-classification.md) — How shot types are determined
- [CLI Reference](guides/cli.md) — All command-line options
