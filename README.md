# Padex

Open-source Python toolkit that transforms padel match broadcast footage into structured tracking data, event streams, and tactical analytics. The first comprehensive CV-based analytics pipeline purpose-built for padel.

> **padex** = padel + x (analytics/extraction)

## Why Padex

Padel is one of the fastest-growing sports in the world, yet it has virtually no open-source analytics tooling. Unlike football — which has mature ecosystems like StatsBomb, Kloppy, and socceraction — padel lacks standardized data formats, open tracking pipelines, and community-driven analytics frameworks.

Padel's enclosed court (glass walls + metal fences), fixed dimensions (20m × 10m), and 4-player format make it uniquely tractable for CV-based tracking. Padex leverages this to deliver a complete end-to-end pipeline: from raw broadcast video to interactive match reports.

## Features

- **Player Detection & Tracking** — YOLO-based detection with ByteTrack, team classification via jersey color clustering
- **Ball Tracking** — SAHI + YOLO with Kalman filtering and visibility states (visible / occluded / inferred)
- **Court Calibration** — Automatic court line detection and homography for pixel-to-real-world coordinate transformation
- **Bounce Detection** — Ground and wall bounce classification with surface identification
- **Shot Detection** — Contact-based shot segmentation with type classification
- **Point Segmentation** — Automatic rally/point boundary detection
- **Tactical Metrics** — Rally-level, player-level, and team-level analytics (winners, errors, net control, wall utilization, etc.)
- **Video Annotation** — Annotated output video with player bboxes, ball markers, court overlay, and mini court
- **Match Reports** — Interactive HTML reports with embedded Plotly dashboards

## Architecture

Padex is organized into three layers, each building on the previous:

```
Layer 1: Tracking          Layer 2: Events           Layer 3: Tactics
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Court Calibration│      │  Bounce Detection │      │  Rally Metrics   │
│  Player Detection │ ───▶ │  Shot Detection   │ ───▶ │  Player Metrics  │
│  Ball Tracking    │      │  Point Segmentation│     │  Team Metrics    │
└──────────────────┘      └──────────────────┘      │  Reports         │
     Parquet                    JSONL                └──────────────────┘
                                                          JSON / HTML
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rondo-labs/padex.git
cd padex

# Install with uv
uv sync
```

### Run End-to-End Analysis

```bash
uv run python examples/run_analysis.py input.mp4 --output output.mp4 --device auto
```

This runs the full pipeline in three passes:
1. **Tracking** — Court calibration, player detection, ball tracking, bounce/shot/point detection, analytics
2. **Annotation** — Generates annotated video with overlays (players, ball, court lines, mini court, stats)
3. **Reporting** — Produces JSON analytics and an interactive HTML match report

Device auto-detection supports CUDA, MPS (Apple Silicon), and CPU.

### Output Structure

```
match_20260301_teamA_vs_teamB/
├── metadata.json
├── tracking/
│   ├── players.parquet
│   ├── ball.parquet
│   └── court_calibration.json
├── events/
│   ├── shots.jsonl
│   ├── points.jsonl
│   └── structure.json
└── analytics/
    ├── rally_metrics.jsonl
    ├── player_metrics.json
    └── team_metrics.json
```

## Package Structure

```
src/padex/
├── tracking/          # Layer 1 — CV pipeline
│   ├── pipeline.py    # End-to-end tracking orchestration
│   ├── court.py       # Court detection + homography
│   ├── player.py      # Player detection + tracking + team classification
│   ├── ball.py        # Ball detection + Kalman filtering
│   └── device.py      # Hardware auto-detection (CUDA/MPS/CPU)
├── events/            # Layer 2 — Event detection
│   ├── shot.py        # Shot detection + classification
│   ├── bounce.py      # Bounce detection + surface classification
│   ├── point.py       # Point/rally segmentation
│   └── taxonomy.py    # Shot type definitions
├── tactics/           # Layer 3 — Analytics
│   ├── metrics.py     # Metric computation
│   ├── heatmap.py     # Spatial analysis
│   └── report.py      # JSON + HTML report generation
├── schemas/           # Data contracts (Pydantic models)
│   ├── tracking.py
│   ├── events.py
│   └── tactics.py
├── io/                # Data I/O
│   ├── video.py       # Video reader/writer
│   ├── parquet.py     # Parquet read/write
│   └── jsonl.py       # JSONL read/write
└── viz/               # Visualization
    ├── frame.py       # Video frame annotation
    ├── mini_court.py  # Real-time 2D court overlay
    ├── court.py       # Plotly court rendering
    ├── dashboard.py   # Interactive Plotly dashboard
    └── animation.py   # Rally replay
```

## Tech Stack

- **Python** ≥ 3.11
- **ultralytics** — YOLO v8/v11 for detection and pose estimation
- **opencv-python** — Video I/O and frame processing
- **polars** + **pyarrow** — Efficient columnar data handling
- **pydantic** — Schema validation and data contracts
- **sahi** — Sliced inference for small object detection (ball)
- **plotly** — Interactive charts and dashboards
- **scipy** — Signal processing for bounce/shot detection
- **lap** — Linear assignment for multi-object tracking

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/

# Type check
uv run mypy src/padex/
```

## Data Schema

Padex uses a schema-first design. All data flowing between layers validates against Pydantic models defined in `src/padex/schemas/`.

Key conventions:
- **Coordinate system**: Origin at bottom-left, x = 0–10m (width), y = 0–20m (length), net at y = 10.0
- **ID format**: Hierarchical — `S_002_05_03_007` = set 2, game 5, point 3, shot 7
- **Shot trajectories**: Ordered bounce sequences embedded in shot events, preserving the full causal chain
- **Ball visibility**: Three states — `visible`, `occluded`, `inferred`

Full specification in `padel_data_schema_v0.1.0.docx`.

## Roadmap

- [x] Project setup and schema definitions
- [x] Court detection and homography calibration
- [x] Player detection, tracking, and team classification
- [x] Ball detection with Kalman filtering
- [x] Bounce and shot detection
- [x] Point/rally segmentation
- [x] Tactical metrics computation
- [x] Video annotation with overlays and mini court
- [x] Interactive HTML match reports
- [ ] Advanced shot type classification (bandeja vs víbora vs smash from pose data)
- [ ] 3D ball trajectory reconstruction
- [ ] Multi-camera support
- [ ] Real-time streaming pipeline
- [ ] Public model weights and example datasets

## License

See [LICENSE](LICENSE) for details.
