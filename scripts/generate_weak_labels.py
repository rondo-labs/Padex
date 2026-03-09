"""
Project: Padex
File Created: 2026-03-08
Author: Xingnan Zhu
File Name: generate_weak_labels.py
Description:
    Generate weak training labels for the MLP event detector using
    existing rule-based detectors + ball-player distance correction.

Usage:
    uv run python scripts/generate_weak_labels.py <tracking_cache.pkl> [-o labels.npz]
"""

from __future__ import annotations

import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_weak_labels")


def main() -> None:
    from padex.events.bounce import (
        VelocityBounceDetectionStrategy,
        _nearest_player_distance,
        extract_all_features,
    )
    from padex.events.shot import ProximityVelocityContactStrategy
    from padex.schemas.tracking import BallVisibility

    if len(sys.argv) < 2:
        logger.error("Usage: generate_weak_labels.py <tracking_cache.pkl> [-o output.npz]")
        sys.exit(1)

    cache_path = Path(sys.argv[1])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "-o" else cache_path.with_suffix(".npz")

    # Load tracking cache
    logger.info("Loading tracking cache: %s", cache_path)
    with open(cache_path, "rb") as f:
        tracking = pickle.load(f)

    ball_frames = tracking.ball_frames
    player_frames = tracking.player_frames
    n_frames = len(ball_frames)
    logger.info("Loaded %d ball frames, %d player frames", n_frames, len(player_frames))

    # Build player lookup by frame_id
    player_lookup: dict[int, list] = defaultdict(list)
    for pf in player_frames:
        player_lookup[pf.frame_id].append(pf)

    # --- Diagnostic: visibility distribution ---
    from collections import Counter
    vis_counts = Counter(bf.visibility for bf in ball_frames)
    pos_counts = sum(1 for bf in ball_frames if bf.position is not None)
    logger.info("Visibility distribution: %s", dict(vis_counts))
    logger.info("Frames with position: %d / %d", pos_counts, n_frames)

    # --- Step 1: Rule-based bounce detection ---
    bounce_detector = VelocityBounceDetectionStrategy()
    bounce_indices = set(bounce_detector.detect(ball_frames))
    logger.info("Rule-based bounce candidates: %d", len(bounce_indices))

    # --- Step 2: Rule-based hit (contact) detection ---
    contact_detector = ProximityVelocityContactStrategy()
    contacts = contact_detector.detect_contacts(player_frames, ball_frames)
    # Map contact timestamp_ms → nearest ball frame index
    hit_indices: set[int] = set()
    for contact in contacts:
        best_idx = min(
            range(n_frames),
            key=lambda i: abs(ball_frames[i].timestamp_ms - contact.timestamp_ms),
        )
        hit_indices.add(best_idx)
    logger.info("Rule-based hit candidates: %d", len(hit_indices))

    # --- Step 3: Distance-based correction ---
    # Bounce candidates near a player (< 1.5m) → reclassify as hit
    corrected = 0
    for idx in list(bounce_indices):
        bf = ball_frames[idx]
        if bf.position is None:
            continue
        dist = _nearest_player_distance(bf.position, player_lookup.get(bf.frame_id, []))
        if dist < 1.5:
            bounce_indices.discard(idx)
            hit_indices.add(idx)
            corrected += 1
    logger.info("Distance correction: %d bounces reclassified as hits", corrected)

    # --- Step 4: Assign labels ---
    # 0=flying, 1=bounce, 2=hit, 3=occluded
    labels = np.zeros(n_frames, dtype=np.int64)
    for i, bf in enumerate(ball_frames):
        if bf.visibility != BallVisibility.VISIBLE or bf.position is None:
            labels[i] = 3  # occluded
        elif i in bounce_indices:
            labels[i] = 1  # bounce
        elif i in hit_indices:
            labels[i] = 2  # hit
        else:
            labels[i] = 0  # flying

    # --- Step 5: Extract features ---
    logger.info("Extracting features...")
    features = extract_all_features(ball_frames, player_frames)

    # --- Step 6: Save ---
    np.savez(
        output_path,
        features=features,
        labels=labels,
    )

    # Summary
    label_names = ["flying", "bounce", "hit", "occluded"]
    for cls_id, name in enumerate(label_names):
        count = int((labels == cls_id).sum())
        logger.info("  %-10s: %d (%.1f%%)", name, count, 100.0 * count / n_frames)
    logger.info("Saved to %s (features shape: %s)", output_path, features.shape)


if __name__ == "__main__":
    main()
