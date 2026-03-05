"""
Project: Padex
File Created: 2026-03-05
Author: Xingnan Zhu
File Name: batch_collect_frames.py
Description:
    Batch-download padel match videos from YouTube and extract 1-3 diverse
    court-view frames per video for court keypoint annotation training data.

    Downloads each video temporarily, extracts qualifying frames, then deletes
    the video to save disk space.

Prerequisites:
    brew install yt-dlp

Usage:
    python scripts/batch_collect_frames.py scripts/video_urls.txt
    python scripts/batch_collect_frames.py scripts/video_urls.txt --max-per-video 2
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUT_DIR = PROJECT_ROOT / "data" / "frames" / "multi_source"


def check_yt_dlp() -> bool:
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_video(url: str, output_path: Path) -> bool:
    """Download a video from YouTube using yt-dlp.

    Downloads at 720p max to save bandwidth and disk space.
    """
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        "--no-playlist",
        "--quiet",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300)
        return output_path.exists()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("Download failed for %s: %s", url, e)
        return False


def is_court_frame(frame: np.ndarray) -> bool:
    """Check if frame shows a wide court view."""
    h, w = frame.shape[:2]

    mean_brightness = np.mean(frame)
    if mean_brightness < 30:
        return False

    y1, y2 = int(h * 0.15), int(h * 0.85)
    x1, x2 = int(w * 0.10), int(w * 0.90)
    roi = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([130, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))

    court_pixels = np.count_nonzero(blue_mask) + np.count_nonzero(green_mask)
    total_pixels = roi.shape[0] * roi.shape[1]

    return (court_pixels / total_pixels) > 0.15


def compute_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute normalized HSV histogram for similarity comparison."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def is_similar_to_any(
    hist: np.ndarray,
    existing_hists: list[np.ndarray],
    threshold: float = 0.90,
) -> bool:
    """Check if frame is too similar to any existing frame (across ALL videos)."""
    for h in existing_hists:
        score = cv2.compareHist(hist, h, cv2.HISTCMP_CORREL)
        if score > threshold:
            return True
    return False


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    yolo_model,
    existing_hists: list[np.ndarray],
    max_frames: int = 2,
    min_players: int = 4,
    device: str | None = None,
    video_label: str = "",
) -> list[np.ndarray]:
    """Extract up to max_frames diverse court-view frames from one video.

    Returns list of histograms of saved frames (for cross-video dedup).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0 or fps == 0:
        cap.release()
        return []

    # Sample at wider intervals since we only want 1-3 frames
    duration_sec = total_frames / fps
    interval = max(30, int(duration_sec / 20))  # ~20 sample points across video
    frame_step = max(1, int(fps * interval))

    saved = 0
    new_hists = []
    local_hists: list[np.ndarray] = []

    # Start from 30% into video to skip intros/logos
    start_frame = int(total_frames * 0.3)
    frame_id = start_frame

    while frame_id < total_frames and saved < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        if not is_court_frame(frame):
            frame_id += frame_step
            continue

        # Player count check
        from scripts.extract_frames import count_players
        n_players = count_players(yolo_model, frame, device=device)
        if n_players < min_players:
            frame_id += frame_step
            continue

        # Dedup: against all previously saved frames (cross-video)
        hist = compute_histogram(frame)
        if is_similar_to_any(hist, existing_hists + local_hists, threshold=0.90):
            frame_id += frame_step
            continue

        # Save
        timestamp_sec = frame_id / fps
        filename = f"{video_label}_f{frame_id:06d}_t{timestamp_sec:.0f}s.jpg"
        out_path = output_dir / filename
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        local_hists.append(hist)
        saved += 1
        frame_id += frame_step

    cap.release()
    return local_hists


def load_urls(url_file: Path) -> list[str]:
    """Load YouTube URLs from a text file (one per line, # for comments)."""
    urls = []
    with open(url_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-download padel videos and extract court frames."
    )
    parser.add_argument("url_file", type=Path, help="Text file with YouTube URLs (one per line)")
    parser.add_argument(
        "-o", "--output", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-per-video", type=int, default=2,
        help="Max frames to extract per video (default: 2)",
    )
    parser.add_argument(
        "--min-players", type=int, default=4,
        help="Minimum players detected per frame (default: 4)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="YOLO inference device (default: auto)",
    )
    parser.add_argument(
        "--keep-videos", action="store_true",
        help="Keep downloaded videos instead of deleting after extraction",
    )
    args = parser.parse_args()

    if not check_yt_dlp():
        logger.error("yt-dlp not found. Install it: brew install yt-dlp")
        sys.exit(1)

    urls = load_urls(args.url_file)
    if not urls:
        logger.error("No URLs found in %s", args.url_file)
        sys.exit(1)

    logger.info("Loaded %d URLs", len(urls))

    args.output.mkdir(parents=True, exist_ok=True)

    # Load YOLO once
    logger.info("Loading YOLO model...")
    from scripts.extract_frames import load_player_detector
    yolo_model = load_player_detector()

    all_hists: list[np.ndarray] = []
    total_saved = 0

    for i, url in enumerate(urls, 1):
        logger.info("[%d/%d] Processing: %s", i, len(urls), url)

        # Generate a short label from the URL
        video_label = f"v{i:03d}"

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / f"{video_label}.mp4"

            # Download
            if not download_video(url, video_path):
                logger.warning("  Skipped (download failed)")
                continue

            size_mb = video_path.stat().st_size / (1024 * 1024)
            logger.info("  Downloaded: %.1f MB", size_mb)

            # Extract frames
            new_hists = extract_frames_from_video(
                video_path=video_path,
                output_dir=args.output,
                yolo_model=yolo_model,
                existing_hists=all_hists,
                max_frames=args.max_per_video,
                min_players=args.min_players,
                device=args.device,
                video_label=video_label,
            )

            n_saved = len(new_hists)
            all_hists.extend(new_hists)
            total_saved += n_saved
            logger.info("  Extracted: %d frames (total: %d)", n_saved, total_saved)

            # Video is auto-deleted when tmpdir is cleaned up
            # unless --keep-videos was specified
            if args.keep_videos:
                keep_path = PROJECT_ROOT / "assets" / "raw" / "video" / f"{video_label}.mp4"
                keep_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.move(str(video_path), str(keep_path))
                logger.info("  Kept video: %s", keep_path)

    logger.info("=== Done ===")
    logger.info("Total frames saved: %d from %d videos", total_saved, len(urls))
    logger.info("Output: %s", args.output)


if __name__ == "__main__":
    main()
