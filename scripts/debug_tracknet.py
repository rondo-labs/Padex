"""
Project: Padex
File Created: 2026-03-09
Author: Xingnan Zhu
File Name: debug_tracknet.py
Description:
    Diagnostic script for TrackNet V2 ball detection.
    Visualizes raw heatmap output and tests different thresholds and frame orders
    to determine if low detection rate is caused by model or post-processing.

Usage:
    uv run python scripts/debug_tracknet.py <VIDEO_PATH> [--frames 30] [--save-dir output/debug]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("debug_tracknet")

PROJECT_ROOT = Path(__file__).parent.parent
THRESHOLDS = [10, 30, 50, 80, 127]
INFER_W, INFER_H = 640, 360


def build_and_load_model(model_path: str, device: str):
    """Load TrackNet V2 model from checkpoint."""
    import torch

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from padex.tracking.ball import _build_tracknet

    model = _build_tracknet()
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    model.to(torch.device(device))
    logger.info("Model loaded on %s", device)
    return model, torch.device(device)


def run_inference(model, frames_rgb: list[np.ndarray], device, order: str = "newest_first"):
    """Run TrackNet inference on 3 frames. Returns raw heatmap (H, W) uint8."""
    import torch

    f0, f1, f2 = frames_rgb  # f0=oldest, f2=newest

    if order == "newest_first":
        stacked = np.concatenate([f2, f1, f0], axis=2)
    else:  # oldest_first
        stacked = np.concatenate([f0, f1, f2], axis=2)

    tensor = torch.from_numpy(stacked.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)  # (1, 256, H*W)

    heatmap = out.argmax(dim=1).squeeze().cpu().numpy()
    heatmap = heatmap.reshape(INFER_H, INFER_W).astype(np.uint8)
    return heatmap


def analyze_heatmap(heatmap: np.ndarray) -> dict:
    """Compute statistics about heatmap response."""
    nonzero = heatmap[heatmap > 0]
    stats = {
        "max": int(heatmap.max()),
        "mean_nonzero": float(nonzero.mean()) if len(nonzero) > 0 else 0.0,
        "nonzero_pixels": int(len(nonzero)),
        "p50": int(np.percentile(nonzero, 50)) if len(nonzero) > 0 else 0,
        "p90": int(np.percentile(nonzero, 90)) if len(nonzero) > 0 else 0,
        "p99": int(np.percentile(nonzero, 99)) if len(nonzero) > 0 else 0,
    }
    # Count detections at different thresholds
    for t in THRESHOLDS:
        feature = (heatmap.astype(np.float32) / max(heatmap.max(), 1) * 255).astype(np.uint8)
        _, binary = cv2.threshold(feature, t, 255, cv2.THRESH_BINARY)
        num_labels, _, stat_arr, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        fg = num_labels - 1
        # Count valid blobs (area <= 200)
        valid_blobs = 0
        if fg > 0:
            for i in range(1, num_labels):
                if stat_arr[i, cv2.CC_STAT_AREA] <= 200:
                    valid_blobs += 1
        stats[f"blobs_t{t}"] = valid_blobs
    return stats


def save_heatmap_vis(heatmap: np.ndarray, frame: np.ndarray, frame_id: int, save_dir: Path):
    """Save side-by-side: original frame + heatmap overlay."""
    frame_small = cv2.resize(frame, (INFER_W, INFER_H))

    # Normalize heatmap to 0-255 for visualization
    if heatmap.max() > 0:
        hm_vis = (heatmap.astype(np.float32) / heatmap.max() * 255).astype(np.uint8)
    else:
        hm_vis = heatmap.copy()

    hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)

    # Overlay heatmap on frame
    overlay = cv2.addWeighted(frame_small, 0.6, hm_color, 0.4, 0)

    # Threshold visualization at t=50 and t=127
    for t, color in [(50, (0, 255, 0)), (127, (0, 0, 255))]:
        feature = (heatmap.astype(np.float32) / max(heatmap.max(), 1) * 255).astype(np.uint8)
        _, binary = cv2.threshold(feature, t, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

    # Stack: original | heatmap | overlay
    combined = np.hstack([frame_small, hm_color, overlay])
    out_path = save_dir / f"frame_{frame_id:05d}.jpg"
    cv2.imwrite(str(out_path), combined)


def main():
    parser = argparse.ArgumentParser(description="Debug TrackNet V2 heatmap output")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to analyze")
    parser.add_argument("--save-dir", type=str, default="output/debug_tracknet", help="Directory to save visualizations")
    parser.add_argument("--skip", type=int, default=30, help="Sample every Nth frame")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from padex.weights import get_weight_path

    model_path = str(get_weight_path("ball_detection_TrackNet.pt"))

    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = "cpu"

    model, torch_device = build_and_load_model(model_path, device)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info("Video: %d frames @ %.1f fps", total_frames, fps)

    frame_buffer: list[np.ndarray] = []
    frame_ids: list[int] = []

    results_newest = []
    results_oldest = []

    frames_analyzed = 0
    frame_id = 0

    while frames_analyzed < args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (INFER_W, INFER_H))
        frame_buffer.append(resized)
        frame_ids.append(frame_id)

        if len(frame_buffer) > 3:
            frame_buffer.pop(0)
            frame_ids.pop(0)

        if len(frame_buffer) == 3:
            # Test both frame orders
            hm_newest = run_inference(model, frame_buffer, torch_device, order="newest_first")
            hm_oldest = run_inference(model, frame_buffer, torch_device, order="oldest_first")

            stats_n = analyze_heatmap(hm_newest)
            stats_o = analyze_heatmap(hm_oldest)
            stats_n["frame_id"] = frame_id
            stats_o["frame_id"] = frame_id
            results_newest.append(stats_n)
            results_oldest.append(stats_o)

            # Save visualization for newest_first (current convention)
            save_heatmap_vis(hm_newest, frame, frame_id, save_dir)

            frames_analyzed += 1
            if frames_analyzed % 5 == 0:
                logger.info("Analyzed %d / %d frames", frames_analyzed, args.frames)

        frame_id += args.skip

    cap.release()

    # Summary report
    logger.info("\n" + "=" * 60)
    logger.info("TRACKNET V2 DIAGNOSTIC REPORT")
    logger.info("=" * 60)

    def summarize(results: list[dict], label: str):
        if not results:
            return
        logger.info("\n[%s]", label)
        max_vals = [r["max"] for r in results]
        logger.info("  Heatmap max value — mean: %.1f, min: %d, max: %d",
                    np.mean(max_vals), np.min(max_vals), np.max(max_vals))
        logger.info("  Frames with max > 0: %d / %d", sum(v > 0 for v in max_vals), len(max_vals))
        for t in THRESHOLDS:
            key = f"blobs_t{t}"
            blob_counts = [r[key] for r in results]
            detected = sum(c > 0 for c in blob_counts)
            logger.info("  Threshold=%3d → detected in %d/%d frames (%.0f%%)",
                        t, detected, len(results), 100.0 * detected / len(results))

    summarize(results_newest, "Frame order: newest_first [f2, f1, f0] (current)")
    summarize(results_oldest, "Frame order: oldest_first [f0, f1, f2] (alternative)")

    # Per-frame comparison
    logger.info("\n[Per-frame max heatmap value]")
    logger.info("  frame_id | newest_first | oldest_first")
    for rn, ro in zip(results_newest[:10], results_oldest[:10]):
        logger.info("  %7d | %12d | %12d", rn["frame_id"], rn["max"], ro["max"])

    logger.info("\nHeatmap visualizations saved to: %s", save_dir)
    logger.info("Green contours = threshold 50, Red contours = threshold 127")

    # Recommendation
    any_detected_n50 = sum(r["blobs_t50"] > 0 for r in results_newest)
    any_detected_n127 = sum(r["blobs_t127"] > 0 for r in results_newest)
    any_detected_o50 = sum(r["blobs_t50"] > 0 for r in results_oldest)

    logger.info("\n[RECOMMENDATION]")
    if any_detected_n127 > len(results_newest) * 0.1:
        logger.info("  V2 works well with current threshold=127. No changes needed.")
    elif any_detected_n50 > len(results_newest) * 0.1:
        logger.info("  Lower HEATMAP_THRESHOLD to 50 — model detects ball but threshold too strict.")
    elif any_detected_o50 > any_detected_n50:
        logger.info("  Try oldest_first frame order — may improve detection rate.")
    elif any_detected_n50 > 0:
        logger.info("  Weak detections at threshold=50. Consider lowering further or upgrading to V3.")
    else:
        logger.info("  No detections at any threshold. Model may not generalize to padel. Upgrade to V3 recommended.")


if __name__ == "__main__":
    main()
