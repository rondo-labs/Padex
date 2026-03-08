# Padex

**Open-source Python toolkit that transforms padel match broadcast footage into structured tracking data, event streams, and tactical analytics.**

> **padex** = padel + x (analytics/extraction)

Padex is the first comprehensive CV-based analytics pipeline purpose-built for padel. It processes raw broadcast video and outputs player tracking, ball trajectories, shot classifications, and match statistics.

## Installation

```bash
pip install padex
```

Model weights (~130MB) are downloaded automatically on first run.

## 30-Second Example

```python
import padex

cal = padex.interactive_calibrate("match.mp4")
result = padex.process("match.mp4", calibration=cal)
padex.export_video(result, "match.mp4", "output.mp4")
```

Or from the command line:

```bash
padex process match.mp4
```

## What's Next

- [Quick Start](quickstart.md) — Full walkthrough of your first analysis
- [Guides](guides/calibration.md) — In-depth guides on calibration, pipeline, and shot classification
- [Architecture](architecture.md) — How Padex works under the hood
- [API Reference](api/index.md) — Complete API documentation
