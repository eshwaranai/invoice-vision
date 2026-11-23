"""Page Detection and Perspective Rectification Benchmark.

Evaluates OpenCV document boundary detection on 72 synthetic phone photos across:
  - 3 severities: mild, medium, severe
  - 2 background types: dark_wood (high contrast) vs light_surface (low contrast)

Measures:
  - Page detection success rate (%)
  - Corner error (mean Euclidean distance in px across the 4 corners)
  - Polygon IoU between ground-truth quadrilateral and detected quadrilateral
  - Failure diagnosis for missed or inaccurate detections
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from invoice_vision.preprocess.page_detector import (
    detect_page_corners,
    order_corners,
    page_detect_and_warp,
)

MANIFEST_PATH = Path("data/processed/synth_photos/manifest.json")
RESULTS_DIR = Path("results")


def compute_polygon_iou(quad_a: np.ndarray, quad_b: np.ndarray, img_shape: Tuple[int, int]) -> float:
    """Compute exact polygon Intersection over Union using raster masks."""
    h, w = img_shape[:2]
    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)

    pts_a = np.int32([quad_a])
    pts_b = np.int32([quad_b])

    cv2.fillPoly(mask_a, pts_a, 1)
    cv2.fillPoly(mask_b, pts_b, 1)

    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def diagnose_failure(img: np.ndarray, gt_corners: np.ndarray) -> str:
    """Diagnose why page boundary detection failed or was inaccurate."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Check contrast at borders
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(gt_corners)], 255)
    doc_mean = float(gray[mask == 255].mean()) if (mask == 255).any() else 0.0
    bg_mean = float(gray[mask == 0].mean()) if (mask == 0).any() else 0.0
    contrast_diff = abs(doc_mean - bg_mean)

    if contrast_diff < 25.0:
        return f"Low edge contrast (|doc - bg| = {contrast_diff:.1f} < 25.0, camouflaged paper boundary)"

    # Check edge continuity
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 120)
    edge_density_inside = float((edges[mask == 255] > 0).mean())
    if edge_density_inside > 0.15:
        return f"High internal text edge density ({edge_density_inside*100:.1f}%) fragmenting outer contour"

    return f"Contour approximation failed to produce convex quad (contrast: {contrast_diff:.1f})"


def run_benchmark() -> pd.DataFrame:
    print("=" * 80)
    print("PAGE DETECTION & RECTIFICATION BENCHMARK (n=72 images)")
    print("=" * 80)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records: List[Dict[str, Any]] = manifest["records"]
    results = []

    for idx, r in enumerate(records, 1):
        fn = r["filename"]
        img_p = Path(r["relative_path"])
        sev = r["severity"]
        bg = r["background"]
        gt_corners = np.array(r["ground_truth_corners"], dtype=np.float32)
        gt_ordered = order_corners(gt_corners)

        img = cv2.imread(str(img_p))
        if img is None:
            continue

        h, w = img.shape[:2]
        detected_corners, success = detect_page_corners(img, min_area_ratio=0.20)

        corner_err = None
        per_corner_err = None
        iou = None
        failure_reason = ""

        if success and detected_corners is not None:
            det_ordered = order_corners(detected_corners)
            errs = np.linalg.norm(det_ordered - gt_ordered, axis=1)
            corner_err = float(np.mean(errs))
            per_corner_err = [round(float(e), 2) for e in errs]
            iou = compute_polygon_iou(det_ordered, gt_ordered, (h, w))
            # If corner error is excessively large, treat as false positive detection
            if corner_err > 150.0:
                is_accurate = False
                failure_reason = f"False boundary detected (corner error: {corner_err:.1f} px, IoU: {iou:.3f})"
            else:
                is_accurate = True
        else:
            is_accurate = False
            failure_reason = diagnose_failure(img, gt_ordered)

        results.append({
            "filename": fn,
            "source_image": r["source_image"],
            "vendor_group": r["vendor_group"],
            "severity": sev,
            "background": bg,
            "detected": success,
            "accurate": is_accurate,
            "corner_error_px": corner_err,
            "iou": iou,
            "failure_reason": failure_reason,
        })

    df = pd.DataFrame(results)

    # 1. Summary by Severity and Background
    print("\n" + "=" * 90)
    print("PAGE DETECTION RATE & CORNER ERROR MATRIX")
    print("=" * 90)
    print(f"{'Condition':<25} | {'Detected':<12} | {'Accurate (<=150px)':<20} | {'Mean Corner Err':<18} | {'Mean IoU'}")
    print("-" * 90)

    for sev in ["mild", "medium", "severe"]:
        for bg in ["dark_wood", "light_surface"]:
            sub = df[(df["severity"] == sev) & (df["background"] == bg)]
            n = len(sub)
            det_n = sub["detected"].sum()
            acc_n = sub["accurate"].sum()
            det_pct = (det_n / n) * 100.0
            acc_pct = (acc_n / n) * 100.0

            acc_sub = sub[sub["accurate"] & sub["corner_error_px"].notna()]
            mean_err = acc_sub["corner_error_px"].mean() if len(acc_sub) > 0 else float("nan")
            mean_iou = acc_sub["iou"].mean() if len(acc_sub) > 0 else float("nan")

            cond_str = f"{sev}_{bg}"
            err_str = f"{mean_err:.2f} px" if not np.isnan(mean_err) else "N/A"
            iou_str = f"{mean_iou:.4f}" if not np.isnan(mean_iou) else "N/A"
            print(f"{cond_str:<25} | {det_n}/{n} ({det_pct:4.1f}%) | {acc_n}/{n} ({acc_pct:4.1f}%)         | {err_str:<18} | {iou_str}")

    # 2. Overall by Background
    print("-" * 90)
    for bg in ["dark_wood", "light_surface"]:
        sub = df[df["background"] == bg]
        n = len(sub)
        det_n = sub["detected"].sum()
        acc_n = sub["accurate"].sum()
        acc_sub = sub[sub["accurate"] & sub["corner_error_px"].notna()]
        mean_err = acc_sub["corner_error_px"].mean() if len(acc_sub) > 0 else float("nan")
        mean_iou = acc_sub["iou"].mean() if len(acc_sub) > 0 else float("nan")
        print(f"{f'ALL {bg}':<25} | {det_n}/{n} ({det_n/n*100:4.1f}%) | {acc_n}/{n} ({acc_n/n*100:4.1f}%)         | {mean_err:.2f} px            | {mean_iou:.4f}")

    # 3. Overall by Severity
    print("-" * 90)
    for sev in ["mild", "medium", "severe"]:
        sub = df[df["severity"] == sev]
        n = len(sub)
        det_n = sub["detected"].sum()
        acc_n = sub["accurate"].sum()
        acc_sub = sub[sub["accurate"] & sub["corner_error_px"].notna()]
        mean_err = acc_sub["corner_error_px"].mean() if len(acc_sub) > 0 else float("nan")
        mean_iou = acc_sub["iou"].mean() if len(acc_sub) > 0 else float("nan")
        print(f"{f'ALL {sev}':<25} | {det_n}/{n} ({det_n/n*100:4.1f}%) | {acc_n}/{n} ({acc_n/n*100:4.1f}%)         | {mean_err:.2f} px            | {mean_iou:.4f}")

    # 4. Failures list
    failures = df[~df["accurate"]]
    print("\n" + "=" * 90)
    print(f"FAILED / INACCURATE DETECTIONS (Total: {len(failures)} / 72)")
    print("=" * 90)
    for _, row in failures.iterrows():
        print(f"  {row['filename']:<36} [{row['severity']}|{row['background']}] -> {row['failure_reason']}")

    return df


if __name__ == "__main__":
    run_benchmark()
