# Architecture

Padex is organized into three layers, each building on the previous.

## Three-Layer Pipeline

```
Layer 1: Tracking          Layer 2: Events           Layer 3: Tactics
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Court Calibration│      │  Bounce Detection │      │  Rally Metrics   │
│  Player Detection │ ───▶ │  Shot Detection   │ ───▶ │  Player Metrics  │
│  Ball Tracking    │      │  Shot Classification│    │  Team Metrics    │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

### Layer 1: Tracking (`padex.tracking`)

CV pipeline that processes video frame-by-frame.

| Component | Model | Purpose |
|-----------|-------|---------|
| Player detection | YOLO v26m | Bounding box detection, confidence > 0.5, max 4 players |
| Player tracking | ByteTrack | Cross-frame ID assignment |
| Team classification | K-means on HSV histograms | Jersey color clustering into T_1 / T_2 |
| Pose estimation | YOLO-Pose v26m | 17 COCO keypoints per player |
| Ball detection | TrackNet | 3-frame CNN, 640x360 heatmap output |
| Ball tracking | Kalman filter | 4D state [x, y, vx, vy], gap-filling |
| Court calibration | Homography | 12-point pixel-to-meter mapping |

### Layer 2: Events (`padex.events`)

Derives discrete match events from tracking data.

- **Bounce detection** — Velocity direction reversal on smoothed trajectories, classified by court geometry (ground, back_wall, side_wall, etc.)
- **Shot detection** — Proximity + velocity change between ball and player
- **Shot classification** — Three-signal decision tree: pre-contact ball state + player pose + post-contact trajectory

### Layer 3: Tactics (`padex.tactics`)

Computes tactical metrics from events. Rally-level, player-level, and team-level analytics.

## Coordinate System

```
         y = 20.0 (far baseline)
    ┌─────────────────────┐
    │                     │
    │    Far service box   │  y = 17.0
    │─────────────────────│
    │                     │
    │        NET          │  y = 10.0
    │                     │
    │─────────────────────│
    │   Near service box   │  y = 3.0
    │                     │
    └─────────────────────┘
  x=0                   x=10.0
         y = 0.0 (near baseline)
```

- **Origin**: Bottom-left corner of the court
- **x-axis**: 0 to 10.0 meters (court width)
- **y-axis**: 0 to 20.0 meters (court length)
- **z-axis**: Height in meters (optional, used for 3D ball position)
- **Net**: y = 10.0

## Data Flow

```
raw video
    │
    ├─► Court Calibration ──► homography matrix (3x3)
    │
    ├─► Player Detection ──► PlayerFrame (bbox, position, pose, team)
    │
    └─► Ball Detection ────► BallFrame (position, visibility)
            │
            ├─► Bounce Detection ──► Bounce (type, position, timestamp)
            │
            └─► Shot Detection ────► Shot (type, player, timestamp, confidence)
```

## Data Formats

| Layer | Format | Reason |
|-------|--------|--------|
| Tracking | Parquet | Columnar, efficient for millions of rows at 30fps |
| Events | JSONL | Streaming, nested structures (trajectory arrays) |
| Structure | JSON | Match/set/game/point hierarchy |
| Calibration | JSON | Small, human-readable |

## Package Structure

```
src/padex/
├── __init__.py          # Top-level API: Padex, process, export_video
├── pipeline.py          # Pipeline orchestrator
├── calibration.py       # Interactive court calibration
├── weights.py           # Model weight management + auto-download
├── cli.py               # Command-line interface
├── tracking/            # Layer 1
│   ├── pipeline.py      # TrackingPipeline
│   ├── court.py         # Court detection + homography
│   ├── player.py        # Player detection + tracking + pose
│   ├── ball.py          # Ball detection (TrackNet / SAHI+YOLO)
│   └── device.py        # Hardware auto-detection
├── events/              # Layer 2
│   ├── shot.py          # Shot detection + classification
│   ├── bounce.py        # Bounce detection
│   ├── point.py         # Point/rally segmentation
│   └── taxonomy.py      # Shot type definitions
├── tactics/             # Layer 3
│   ├── metrics.py       # Metric computation
│   ├── heatmap.py       # Spatial analysis
│   └── report.py        # Report generation
├── schemas/             # Pydantic data models
├── io/                  # Video, Parquet, JSONL I/O
└── viz/                 # Visualization + annotation
```
