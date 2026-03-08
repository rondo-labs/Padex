# Court Calibration

Padex needs a homography matrix to map pixel coordinates to real-world court coordinates (meters). This is done by clicking on known court keypoints in the video.

## Why Calibration is Needed

Without calibration, Padex can only give you pixel positions. With calibration, you get:

- Player positions in meters on a standard 20m x 10m court
- Ball positions mapped to real-world coordinates
- Accurate bounce surface classification (ground vs wall)
- Tactical metrics (distance covered, net control, etc.)

## The 12 Court Keypoints

The calibration uses up to 12 keypoints on the standard padel court:

| # | Keypoint | Court Position (x, y) |
|---|----------|----------------------|
| 1 | bottom_left | (0.0, 0.0) |
| 2 | bottom_right | (10.0, 0.0) |
| 3 | top_left | (0.0, 20.0) |
| 4 | top_right | (10.0, 20.0) |
| 5 | net_left | (0.0, 10.0) |
| 6 | net_right | (10.0, 10.0) |
| 7 | service_near_left | (0.0, 3.0) |
| 8 | service_near_center | (5.0, 3.0) |
| 9 | service_near_right | (10.0, 3.0) |
| 10 | service_far_left | (0.0, 17.0) |
| 11 | service_far_center | (5.0, 17.0) |
| 12 | service_far_right | (10.0, 17.0) |

!!! tip
    You need at least 4 keypoints for calibration. More points = better accuracy. The 4 corners are the minimum; adding net and service line points significantly improves precision.

## Running Calibration

=== "Python"

    ```python
    import padex

    cal = padex.interactive_calibrate("match.mp4")
    # Saved automatically as match_calibration.json
    ```

=== "CLI"

    ```bash
    padex calibrate match.mp4
    padex calibrate match.mp4 -o custom_cal.json
    ```

## Phase 1: Frame Selection

A window opens showing the video. Navigate to find a frame with clear, unobstructed court lines.

| Key | Action |
|-----|--------|
| `D` / `→` | Forward 90 frames (~3 sec) |
| `A` / `←` | Back 90 frames |
| `W` / `↑` | Forward 900 frames (~30 sec) |
| `S` / `↓` | Back 900 frames |
| `Enter` | Confirm this frame |
| `Q` | Quit |

!!! tip
    Choose a frame where you can see as many court lines as possible. Avoid frames with players standing on the lines.

## Phase 2: Keypoint Labeling

Click on each keypoint in order. The current keypoint to place is shown at the top of the screen.

| Key | Action |
|-----|--------|
| Left click | Place current keypoint |
| `N` | Skip current keypoint (if not visible) |
| `Z` | Undo last keypoint |
| `Enter` | Finish (need >= 4 points) |
| `Q` | Quit without saving |

Each keypoint is color-coded by group:

- **Red**: Bottom corners
- **Blue**: Top corners
- **Yellow**: Net intersections
- **Green**: Near service line
- **Magenta**: Far service line

## Calibration Output

The calibration is saved as a JSON file:

```json
{
  "homography_matrix": [[...], [...], [...]],
  "frame_width": 1920,
  "frame_height": 1080,
  "reprojection_error": 0.135,
  "source_video": "match.mp4",
  "source_frame_id": 450,
  "labeled_keypoints": {
    "bottom_left": [234, 890],
    "bottom_right": [1650, 885],
    ...
  }
}
```

The `reprojection_error` tells you how accurate the calibration is, in meters. Values under 0.5m are good; under 0.2m is excellent.

## Reusing Calibration

Padex automatically discovers calibration files. If you have `match_calibration.json` next to `match.mp4`, it will be loaded automatically:

```python
# This finds match_calibration.json automatically
result = padex.process("match.mp4")
```

You can also pass the path explicitly:

```python
result = padex.process("match.mp4", calibration="path/to/cal.json")
```

For videos from the same camera angle, you can reuse the same calibration file:

```python
result = padex.process("match2.mp4", calibration="match_calibration.json")
```
