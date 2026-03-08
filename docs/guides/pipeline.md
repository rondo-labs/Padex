# Pipeline Guide

The Padex pipeline processes a padel match video through three stages: tracking, bounce detection, and shot classification.

## Basic Usage

```python
import padex

result = padex.process("match.mp4", calibration="cal.json")
```

This is equivalent to:

```python
p = padex.Padex("match.mp4", calibration="cal.json")
result = p.run()
```

## The `Padex` Class

```python
padex.Padex(
    video_path="match.mp4",
    calibration=None,           # CourtCalibration, Path, or None
    enable_pose=True,           # Run pose estimation (needed for shot classification)
    cache_tracking=True,        # Cache tracking results as pickle
    cache_dir=None,             # Directory for cache files (default: next to video)
)
```

### Calibration Resolution

When `calibration` is:

- **`CourtCalibration` object** — Used directly
- **`Path` or `str`** — Loaded from JSON file
- **`None`** — Looks for `<video_stem>_calibration.json` next to the video. If not found, raises `ValueError` with instructions.

## Pipeline Stages

### Stage 1: Tracking

The slowest stage (~5 FPS on Apple Silicon). Processes every frame:

- **Player detection**: YOLO v26m detects up to 4 players, filtered by court boundaries
- **Player tracking**: ByteTrack assigns persistent IDs across frames
- **Team classification**: K-means clustering on jersey color histograms
- **Pose estimation**: YOLO-Pose extracts 17 keypoints per player (if `enable_pose=True`)
- **Ball detection**: TrackNet processes 3 consecutive frames as 9-channel input
- **Ball tracking**: Kalman filter fills detection gaps

### Stage 2: Bounce Detection

Detects ball bounces by finding velocity direction reversals on smoothed trajectories. Each bounce is classified by surface:

- `ground` — Ball hits the playing surface
- `back_wall` — Ball hits the back glass wall
- `side_wall` — Ball hits the side glass/fence
- `net` — Ball hits the net

### Stage 3: Shot Classification

Detects player-ball contact events and classifies each shot. See [Shot Classification](shot-classification.md) for details.

## The `PadexResult` Object

```python
result = padex.process("match.mp4", calibration="cal.json")

result.tracking       # TrackingResult (player_frames, ball_frames)
result.bounces        # list[Bounce]
result.shots          # list[Shot]
result.calibration    # CourtCalibration or None
```

Each shot contains:

```python
shot.shot_id          # e.g. "shot_001"
shot.shot_type        # ShotType enum (e.g. ShotType.VOLLEY)
shot.player_id        # e.g. "P_001"
shot.timestamp_ms     # Contact time in milliseconds
shot.confidence       # Classification confidence (0-1)
```

## Tracking Cache

Tracking results are cached as pickle files to avoid re-running the slow tracking stage:

```
<cache_dir>/<video_stem>_tracking_cache.pkl
```

On subsequent runs, only bounce detection, shot classification, and video export run — taking seconds instead of minutes.

Disable caching:

```python
result = padex.process("match.mp4", cache_tracking=False)
```

## Exporting Annotated Video

```python
padex.export_video(result, "match.mp4", "output/annotated.mp4")
```

Or using the `Padex` instance:

```python
p = padex.Padex("match.mp4", calibration="cal.json")
result = p.run()
p.export_video(result, "output/annotated.mp4")
```

The annotated video includes player bounding boxes, ball markers, court overlay lines, shot type labels (displayed for 1.5 seconds after each contact), and a stats overlay.
