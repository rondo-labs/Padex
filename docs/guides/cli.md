# CLI Reference

Padex provides a command-line interface with two commands: `process` and `calibrate`.

## `padex process`

Run the full analysis pipeline on a video.

```bash
padex process VIDEO [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `VIDEO` | Path to the input video file (required) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--calibration`, `-c` | Auto-detect | Path to court calibration JSON file |
| `--output`, `-o` | `output/` | Output directory for annotated video |
| `--no-cache` | `false` | Disable tracking cache (re-run tracking from scratch) |
| `--no-export` | `false` | Skip annotated video export |

### Calibration Behavior

When `--calibration` is not provided:

1. Looks for `<video_stem>_calibration.json` next to the video
2. If not found, launches the **interactive calibration UI**
3. After calibration, saves the file and continues with the pipeline

### Examples

```bash
# First run on a new video (will prompt for calibration)
padex process match.mp4

# Use existing calibration
padex process match.mp4 -c match_calibration.json

# Custom output directory, no cache
padex process match.mp4 -o results/ --no-cache

# Run tracking + analysis only, skip video export
padex process match.mp4 --no-export
```

### Output

The annotated video is saved to:

```
<output_dir>/<timestamp>/shot_detection.mp4
```

A summary is printed to the console with detected shots, player attributions, and confidence scores.

---

## `padex calibrate`

Run interactive court calibration without processing.

```bash
padex calibrate VIDEO [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `VIDEO` | Path to the input video file (required) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | `<video_stem>_calibration.json` | Output path for calibration JSON |

### Examples

```bash
# Calibrate and save next to video
padex calibrate match.mp4

# Save to custom path
padex calibrate match.mp4 -o calibrations/court1.json
```

### Interactive UI

The calibration runs in two phases:

**Phase 1: Frame Selection** — Browse the video to find a frame with clear court lines.

| Key | Action |
|-----|--------|
| `D` / Right arrow | Forward 30 frames |
| `A` / Left arrow | Back 30 frames |
| `W` / Up arrow | Forward 300 frames |
| `S` / Down arrow | Back 300 frames |
| `Enter` | Confirm frame |
| `Q` / `Esc` | Cancel |

**Phase 2: Keypoint Labeling** — Click the 12 court reference points in order.

| Key | Action |
|-----|--------|
| Left click | Place keypoint |
| `Z` | Undo last point |
| `Enter` | Confirm (after all 12 points) |
| `Q` / `Esc` | Cancel |

See the [Calibration Guide](calibration.md) for detailed instructions on which points to click.
