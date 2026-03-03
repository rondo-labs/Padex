# CLAUDE.md — Padex Project Guide

## What is Padex

Padex is an open-source Python toolkit that transforms padel match broadcast footage into structured tracking data, event streams, and tactical analytics. It is the first comprehensive CV-based analytics pipeline purpose-built for padel.

The name: padel + x (analytics/extraction).

## Why Padex Exists

Padel is a fast-growing sport with virtually no open-source analytics tooling. Unlike football, which has mature ecosystems (StatsBomb, Kloppy, socceraction, floodlight), padel has no standardized data format, no open tracking pipeline, and no community-driven analytics framework. Padex fills this gap.

Padel's enclosed court (glass walls + metal fences), fixed dimensions (20m × 10m), and 4-player format make it uniquely tractable for CV-based tracking compared to football's 22 players on a large open pitch. However, padel introduces its own challenges: glass reflections causing ghost detections, small fast-moving balls, and wall bounces creating complex 3D trajectories.

## Architecture — Three Layers

The system is organized into three layers, each building on the previous. This maps directly to the package structure.

### Layer 1: `padex.tracking` — Raw Tracking Engine

CV pipeline that processes broadcast video frame-by-frame.

Core models:
- **YOLO (v8/v11)**: Player and ball detection with bounding boxes
- **YOLO-Pose**: Skeleton keypoint extraction for shot type classification
- **SAM3**: Player segmentation to handle glass reflection artifacts and occlusion

Outputs per-frame:
- Player positions (up to 4), bounding boxes, confidence, pose keypoints
- Ball position with visibility state (visible / occluded / inferred)
- Court calibration via homography (pixel coords → real-world meters)

Key challenges to solve:
- Glass wall reflections producing ghost player detections → apply NMS with court boundary constraints
- Ball tracking at high speed (< 20px in broadcast) → consider TrackNet-style architecture or optical flow assist
- Team classification → jersey color clustering (simplified by 2v2 format)
- Single-camera homography → padel court line detection + glass wall edges as calibration points

Storage: **Parquet** (columnar, efficient for millions of rows at 30fps)

### Layer 2: `padex.events` — Event Detection

Derives discrete match events from tracking data. This is where padel domain knowledge lives.

Core event: **Shot** — the atomic unit of padel data. Each shot captures:
- The moment of contact (timestamp, player, position, shot type)
- The ball's **full trajectory as an ordered bounce sequence** until next contact or point end
- Outcome (winner, error, next_shot, etc.)

The trajectory array is the key design decision: ground bounces and wall bounces live together inside a single shot event as an ordered list, preserving the causal chain of the entire ball journey.

```
trajectory: [
  { type: "ground", position: {x, y} },
  { type: "back_wall", position: {x, y, z} },
  { type: "ground", position: {x, y} }
]
```

Bounce types: ground, back_wall, side_wall, back_fence, side_fence, corner, net

Shot type taxonomy (padel-specific):
- Serve: serve
- Net play: volley, bandeja, víbora, smash, smash_x3 (por tres), smash_x4 (por cuatro), drop_shot
- Baseline: bajada, groundstroke_fh, groundstroke_bh
- Transition: chiquita
- Defensive: lob, wall_return, contra_pared
- Fallback: unknown

Match hierarchy: match → set → game → point → shot

Storage: **JSONL** for shots/points (streaming, nested structures), **JSON** for match structure

### Layer 3: `padex.tactics` — Analytics & Visualization

Computes tactical metrics from layers 1 and 2.

Rally-level: rally length, duration, net approaches, wall bounces, attack/defense switches
Player-level: winners, errors, serve %, shot type usage and success rates, distance covered, avg position depth
Team-level: net control %, formation switches, pair distance, wall utilization index, transition efficiency, heatmaps, shot direction matrices

Storage: **JSON**

## Data Schema

The full data schema specification is defined in `padel_data_schema_v0.1.0.docx` (versioned alongside this document). This is the contract for all data flowing between layers.

Key design decisions:
- **Coordinate system**: Origin at bottom-left corner, x = 0–10m (width), y = 0–20m (length), z = height (optional). Net at y = 10.0.
- **ID convention**: Hierarchical encoding — `S_002_05_03_007` means set 2, game 5, point 3, shot 7. Context extractable from ID alone.
- **Trajectory as bounce sequence inside shot events**: Not separate wall/ground events. One shot = one complete ball journey.
- **Ball visibility states**: visible / occluded / inferred — critical for trajectory interpolation.
- **Schema versioning**: semver (currently 0.1.0, pre-release draft). All data files carry schema_version.

## File Structure Per Match

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

## Tech Stack

- **Language**: Python
- **Project management**: uv
- **CV models**: YOLO v8/v11, YOLO-Pose, SAM3, potentially TrackNet for ball tracking
- **Data formats**: Parquet (tracking), JSONL (events), JSON (structure/metrics)
- **Key libraries** (expected): ultralytics, opencv-python, numpy, polars/pandas, pyarrow

## Package Structure

```
padex/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── src/
│   └── padex/
│       ├── __init__.py
│       ├── tracking/          # Layer 1
│       │   ├── __init__.py
│       │   ├── player.py      # Player detection + pose
│       │   ├── ball.py        # Ball detection + trajectory
│       │   ├── court.py       # Court detection + homography
│       │   └── pipeline.py    # Orchestrates full tracking
│       ├── events/            # Layer 2
│       │   ├── __init__.py
│       │   ├── shot.py        # Shot detection + classification
│       │   ├── bounce.py      # Bounce/wall interaction detection
│       │   ├── point.py       # Point/rally segmentation
│       │   └── taxonomy.py    # Shot type definitions + enums
│       ├── tactics/           # Layer 3
│       │   ├── __init__.py
│       │   ├── metrics.py     # Metric computation
│       │   ├── heatmap.py     # Spatial analysis
│       │   └── report.py      # Match report generation
│       ├── schemas/           # Data contracts
│       │   ├── __init__.py
│       │   ├── tracking.py    # Pydantic models for Layer 1
│       │   ├── events.py      # Pydantic models for Layer 2
│       │   └── tactics.py     # Pydantic models for Layer 3
│       ├── io/                # Data I/O
│       │   ├── __init__.py
│       │   ├── parquet.py     # Parquet read/write
│       │   ├── jsonl.py       # JSONL read/write
│       │   └── video.py       # Video frame extraction
│       └── viz/               # Visualization
│           ├── __init__.py
│           ├── court.py       # Court rendering
│           ├── animation.py   # Rally replay
│           └── dashboard.py   # Match overview
├── tests/
├── docs/
└── examples/
```

## Development Priorities

Phase 1 — Foundation:
1. Set up project with uv, define pyproject.toml
2. Implement schemas (Pydantic models for all three layers)
3. Build court detection + homography (fixed court makes this the easiest starting point)
4. Player detection + tracking with team classification

Phase 2 — Core Pipeline:
5. Ball detection + trajectory tracking
6. Bounce detection (ground and wall)
7. Shot segmentation + basic shot type classification
8. Point/rally structure extraction

Phase 3 — Intelligence:
9. Advanced shot type classification (bandeja vs víbora vs smash from pose data)
10. Tactical metrics computation
11. Visualization layer
12. Match report generation

## Design Principles

- **Schema-first**: Data contracts (Pydantic models) are defined before implementation. All data flowing between layers must validate against the schema.
- **Modular**: Each layer is independently usable. Someone with their own tracking data can skip Layer 1 and feed directly into Layer 2.
- **Padel-native**: Every design decision should reflect padel's unique characteristics. Don't force football conventions where they don't fit.
- **Open-source friendly**: Clear documentation, typed interfaces, example data, low barrier to contribution.
- **Iterative**: Ship working increments. A pipeline that tracks 4 players reliably is more valuable than one that attempts everything and works poorly.
