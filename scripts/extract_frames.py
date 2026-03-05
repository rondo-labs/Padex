"""
Project: Padex
File Created: 2026-03-05
Author: Xingnan Zhu
File Name: extract_frames.py
Description:
    Extract frames from padel match videos for court keypoint annotation.
    Filters out near-duplicate frames, non-court frames, and frames without
    enough players detected (requires YOLO).

Usage:
    python scripts/extract_frames.py assets/raw/video/match.mp4 -o data/frames/match
    python scripts/extract_frames.py assets/raw/video/match.mp4 --interval 5 --min-players 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path so we can import padex
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def compute_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute a normalized HSV histogram for frame similarity comparison."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def is_similar(hist_a: np.ndarray, hist_b: np.ndarray, threshold: float) -> bool:
    """Check if two frames are visually similar using histogram correlation."""
    score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return score > threshold


def is_court_frame(frame: np.ndarray) -> bool:
    """Heuristic check: does this frame show a wide court view (not a closeup/replay)?

    Checks:
    1. Sufficient court-colored area (blue or green) in central region
    2. Not a black/transition frame
    """
    h, w = frame.shape[:2]

    # Check for black/transition frames
    mean_brightness = np.mean(frame)
    if mean_brightness < 30:
        return False

    # Crop to central region (avoid scoreboards, banners)
    y1, y2 = int(h * 0.15), int(h * 0.85)
    x1, x2 = int(w * 0.10), int(w * 0.90)
    roi = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Blue court surfaces (common in padel)
    blue_mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([130, 255, 255]))
    # Green court surfaces
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

    court_pixels = np.count_nonzero(blue_mask) + np.count_nonzero(green_mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    court_ratio = court_pixels / total_pixels

    # Need at least 15% court-colored pixels for a wide court view
    return court_ratio > 0.15


def load_player_detector(device: str | None = None):
    """Load YOLO model for player detection."""
    from ultralytics import YOLO

    model_path = PROJECT_ROOT / "assets" / "weights" / "yolo26m.pt"
    if not model_path.exists():
        logger.error("YOLO weights not found: %s", model_path)
        sys.exit(1)

    model = YOLO(str(model_path))
    return model


def count_players(model, frame: np.ndarray, device: str | None = None) -> int:
    """Count the number of persons detected by YOLO in a frame."""
    PERSON_CLASS_ID = 0
    results = model.predict(
        frame,
        conf=0.5,
        classes=[PERSON_CLASS_ID],
        verbose=False,
        device=device,
    )
    if results and results[0].boxes is not None:
        return len(results[0].boxes)
    return 0


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval_sec: float = 3.0,
    min_similarity: float = 0.93,
    max_frames: int = 500,
    min_players: int = 4,
    device: str | None = None,
) -> int:
    """Extract diverse court-view frames from a video.

    Args:
        video_path: Path to the input video.
        output_dir: Directory to save extracted frames.
        interval_sec: Minimum seconds between sampled frames.
        min_similarity: Histogram correlation threshold for dedup (higher = stricter).
        max_frames: Maximum number of frames to extract.
        min_players: Minimum number of players detected to keep a frame.
        device: Inference device for YOLO (None = auto-detect).

    Returns:
        Number of frames extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    frame_step = max(1, int(fps * interval_sec))

    logger.info("Video: %s", video_path.name)
    logger.info("FPS: %.1f | Total frames: %d | Duration: %.1fs", fps, total_frames, duration_sec)
    logger.info(
        "Sampling every %.1fs (%d frames) | Dedup threshold: %.2f | Min players: %d",
        interval_sec, frame_step, min_similarity, min_players,
    )

    # Load YOLO for player detection
    logger.info("Loading YOLO model for player detection...")
    yolo_model = load_player_detector()

    saved_count = 0
    skipped_similar = 0
    skipped_no_court = 0
    skipped_few_players = 0
    saved_histograms: list[np.ndarray] = []
    video_stem = video_path.stem

    frame_id = 0
    while frame_id < total_frames and saved_count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        # Filter 1: Is this a court-view frame?
        if not is_court_frame(frame):
            skipped_no_court += 1
            frame_id += frame_step
            continue

        # Filter 2: Are there enough players visible?
        n_players = count_players(yolo_model, frame, device=device)
        if n_players < min_players:
            skipped_few_players += 1
            frame_id += frame_step
            continue

        # Filter 3: Is this frame different enough from already saved frames?
        hist = compute_histogram(frame)
        is_dup = any(is_similar(hist, saved_hist, min_similarity) for saved_hist in saved_histograms)

        if is_dup:
            skipped_similar += 1
            frame_id += frame_step
            continue

        # Save frame
        timestamp_sec = frame_id / fps
        filename = f"{video_stem}_f{frame_id:06d}_t{timestamp_sec:.1f}s.jpg"
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        saved_histograms.append(hist)
        saved_count += 1

        if saved_count % 20 == 0:
            logger.info("  Saved %d frames so far...", saved_count)

        frame_id += frame_step

    cap.release()

    logger.info("--- Results ---")
    logger.info("Saved: %d frames to %s", saved_count, output_dir)
    logger.info(
        "Skipped — no court: %d | few players: %d | similar: %d",
        skipped_no_court, skipped_few_players, skipped_similar,
    )

    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames from padel videos for court keypoint annotation."
    )
    parser.add_argument("video", type=Path, help="Path to input video file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output directory (default: data/frames/<video_stem>)",
    )
    parser.add_argument(
        "--interval", type=float, default=3.0,
        help="Minimum seconds between sampled frames (default: 3.0)",
    )
    parser.add_argument(
        "--min-similarity", type=float, default=0.93,
        help="Histogram correlation threshold for dedup, 0-1 (default: 0.93)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=500,
        help="Maximum number of frames to extract (default: 500)",
    )
    parser.add_argument(
        "--min-players", type=int, default=4,
        help="Minimum number of players detected to keep a frame (default: 4)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Inference device for YOLO, e.g. 'mps', 'cuda', 'cpu' (default: auto)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        logger.error("Video file not found: %s", args.video)
        sys.exit(1)

    output_dir = args.output or (PROJECT_ROOT / "data" / "frames" / args.video.stem)

    extract_frames(
        video_path=args.video,
        output_dir=output_dir,
        interval_sec=args.interval,
        min_similarity=args.min_similarity,
        max_frames=args.max_frames,
        min_players=args.min_players,
        device=args.device,
    )


if __name__ == "__main__":
    main()
