# Padex

[![PyPI version](https://img.shields.io/pypi/v/padex.svg)](https://pypi.org/project/padex/)
[![Python](https://img.shields.io/pypi/pyversions/padex.svg)](https://pypi.org/project/padex/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://readthedocs.org/projects/padex/badge/?version=latest)](https://padex.readthedocs.io)

Open-source Python toolkit that transforms padel match broadcast footage into structured tracking data, event streams, and tactical analytics.

> **padex** = padel + x (analytics/extraction)

## Installation

```bash
pip install padex
```

## Quick Start

### Python API

```python
import padex

result = padex.process("match.mp4", calibration="match_calibration.json")
print(f"Detected {len(result.shots)} shots, {len(result.bounces)} bounces")

padex.export_video(result, "match.mp4", "output/annotated.mp4")
```

### CLI

```bash
# Run full pipeline (auto-launches calibration if needed)
padex process match.mp4

# Calibrate court separately
padex calibrate match.mp4
```

## Features

- **Player Detection & Tracking** -- YOLO + ByteTrack with team classification
- **Ball Tracking** -- TrackNet with Kalman filtering
- **Court Calibration** -- Interactive 12-point homography
- **Bounce Detection** -- Ground and wall bounce classification
- **Shot Classification** -- Pose-based three-signal decision tree (15 padel shot types)
- **Video Annotation** -- Annotated output with overlays, court lines, and shot labels

## Documentation

Full documentation: [padex.readthedocs.io](https://padex.readthedocs.io)

- [Quick Start](https://padex.readthedocs.io/quickstart/)
- [Pipeline Guide](https://padex.readthedocs.io/guides/pipeline/)
- [Shot Classification](https://padex.readthedocs.io/guides/shot-classification/)
- [API Reference](https://padex.readthedocs.io/api/)

## Development

```bash
git clone https://github.com/rondo-labs/padex.git
cd padex
uv sync --group dev
uv run pytest tests/ -v
```

## License

See [LICENSE](LICENSE) for details.
