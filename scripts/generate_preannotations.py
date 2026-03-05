"""
Project: Padex
File Created: 2026-03-05
Author: Xingnan Zhu
File Name: generate_preannotations.py
Description:
    Generate full 12-keypoint pre-annotations from 4 manually labeled corner points.
    Computes homography from 4 points, then projects all 12 court keypoints
    back to pixel space. Outputs Label Studio JSON for re-import.

Usage:
    1. In Label Studio, label ONLY 4 corner points per image (first pass).
    2. Export annotations as JSON.
    3. Run:
       python scripts/generate_preannotations.py annotations/exports/4pt_export.json -o annotations/exports/12pt_preannotations.json
    4. Import the output JSON back into Label Studio as predictions for correction.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from padex.tracking.court import COURT_MODEL

# All 12 keypoints: name -> (x_meters, y_meters)
ALL_KEYPOINTS = COURT_MODEL.KEYPOINTS


def extract_labeled_points(
    annotation: dict,
) -> list[tuple[str, float, float]]:
    """Extract keypoint labels and their percentage coordinates from a Label Studio annotation.

    Returns list of (label_name, x_pct, y_pct) where x_pct/y_pct are 0-100.
    """
    points = []
    results = []

    # Handle both "annotations" and "completions" formats
    if "annotations" in annotation and annotation["annotations"]:
        results = annotation["annotations"][0].get("result", [])
    elif "completions" in annotation and annotation["completions"]:
        results = annotation["completions"][0].get("result", [])

    for r in results:
        if r.get("type") != "keypointlabels":
            continue
        value = r["value"]
        labels = value.get("keypointlabels", [])
        if not labels:
            continue
        label = labels[0]
        x_pct = value["x"]
        y_pct = value["y"]
        points.append((label, x_pct, y_pct))

    return points


def compute_all_keypoints(
    labeled_points: list[tuple[str, float, float]],
) -> dict[str, tuple[float, float]] | None:
    """From >= 4 labeled keypoints, compute homography and project all 12 keypoints.

    Args:
        labeled_points: List of (label_name, x_pct, y_pct).

    Returns:
        Dict of keypoint_name -> (x_pct, y_pct) for all 12 points, or None on failure.
    """
    if len(labeled_points) < 4:
        return None

    # Build source (pixel-pct) and destination (court-meters) arrays
    src_pts = []  # pixel percentages
    dst_pts = []  # court meters
    for label, x_pct, y_pct in labeled_points:
        if label not in ALL_KEYPOINTS:
            logger.warning("Unknown label '%s', skipping", label)
            continue
        src_pts.append([x_pct, y_pct])
        dst_pts.append(list(ALL_KEYPOINTS[label]))

    if len(src_pts) < 4:
        return None

    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)

    # Compute homography: pixel-pct -> court-meters
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None

    # Invert: court-meters -> pixel-pct
    H_inv = np.linalg.inv(H)

    # Project all 12 keypoints
    result = {}
    for name, (mx, my) in ALL_KEYPOINTS.items():
        pt = np.array([[[mx, my]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(pt, H_inv)
        px, py = projected[0, 0]
        result[name] = (float(px), float(py))

    return result


def build_prediction_result(
    keypoints: dict[str, tuple[float, float]],
    labeled_set: set[str],
) -> list[dict]:
    """Build Label Studio prediction result entries for all 12 keypoints.

    Points that were manually labeled get visibility=visible,
    projected points get visibility=occluded (so annotator knows to verify).
    """
    results = []
    for name, (x_pct, y_pct) in keypoints.items():
        # Skip points that project way outside the image
        if x_pct < -5 or x_pct > 105 or y_pct < -5 or y_pct > 105:
            continue

        # Clamp to valid range
        x_pct = max(0.0, min(100.0, x_pct))
        y_pct = max(0.0, min(100.0, y_pct))

        region_id = str(uuid.uuid4())[:8]

        # Keypoint entry
        results.append({
            "id": region_id,
            "from_name": "keypoints",
            "to_name": "image",
            "type": "keypointlabels",
            "value": {
                "x": round(x_pct, 2),
                "y": round(y_pct, 2),
                "width": 0.4,
                "keypointlabels": [name],
            },
        })

        # Visibility choice (per-region)
        visibility = "visible" if name in labeled_set else "occluded"
        results.append({
            "id": str(uuid.uuid4())[:8],
            "from_name": "visibility",
            "to_name": "image",
            "type": "choices",
            "value": {
                "choices": [visibility],
            },
            "parentID": region_id,
        })

    return results


def generate_preannotations(input_path: Path, output_path: Path) -> int:
    """Process exported Label Studio JSON and generate 12-point pre-annotations.

    Returns number of successfully processed tasks.
    """
    with open(input_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    output_tasks = []
    success = 0
    failed = 0

    for task in data:
        labeled_points = extract_labeled_points(task)
        if len(labeled_points) < 4:
            logger.warning(
                "Task %s: only %d points labeled, need >= 4. Skipping.",
                task.get("id", "?"),
                len(labeled_points),
            )
            failed += 1
            continue

        all_kps = compute_all_keypoints(labeled_points)
        if all_kps is None:
            logger.warning("Task %s: homography failed. Skipping.", task.get("id", "?"))
            failed += 1
            continue

        labeled_set = {name for name, _, _ in labeled_points}
        prediction_result = build_prediction_result(all_kps, labeled_set)

        # Build output task with predictions
        out_task = {
            "data": task.get("data", {}),
            "predictions": [
                {
                    "result": prediction_result,
                    "score": 1.0,
                }
            ],
        }
        output_tasks.append(out_task)
        success += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_tasks, f, indent=2)

    logger.info("Processed: %d success, %d failed", success, failed)
    logger.info("Output: %s", output_path)
    return success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 12-point pre-annotations from 4-point Label Studio export."
    )
    parser.add_argument("input", type=Path, help="Label Studio JSON export (4-point annotations)")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output JSON path (default: annotations/exports/12pt_preannotations.json)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    output = args.output or (PROJECT_ROOT / "annotations" / "exports" / "12pt_preannotations.json")
    generate_preannotations(args.input, output)


if __name__ == "__main__":
    main()
