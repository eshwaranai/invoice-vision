"""Quality Gate Calibration Script.

Analyzes image quality metrics (blur, brightness, glare, resolution) across:
  - 12 clean original invoices
  - 72 synthetic degraded phone photos (3 severities x 2 backgrounds x 12 images)

Evaluates:
  - Default Quality Gate thresholds vs measured OCR success:
      OCR Success defined as: word_count >= 30 AND upper_bound_recall >= 0.50 on at least one field.
  - Confusion matrix of Gate (PASS / WARN / REJECT) vs OCR Success.
  - Calibrates optimal thresholds to separate usable from unusable images.
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

from invoice_vision.ocr.easyocr_engine import OCRItem
from invoice_vision.preprocess.quality_gate import (
    check_image_quality,
    compute_blur_score,
    compute_brightness,
    compute_glare_ratio,
)
from scripts.diagnose_baseline import check_upper_bound_amount, check_upper_bound_string
from scripts.evaluate_v1 import is_non_empty

RAW_DIR = Path("data/raw/indian_gst")
MANIFEST_PATH = Path("data/processed/synth_photos/manifest.json")
CLEAN_CACHE_DIR = Path("data/processed/ocr_cache")
CALIB_CACHE_DIR = Path("data/processed/synth_photos/calib_cache")
GROUND_TRUTH_CSV = Path("data/ground_truth_gst.csv")

FIELDS = ["invoice_number", "purchase_date", "supplier_gst", "total_amount"]


def compute_metrics(img: np.ndarray) -> Dict[str, float]:
    h, w = img.shape[:2]
    return {
        "width": float(w),
        "height": float(h),
        "blur": compute_blur_score(img),
        "brightness": compute_brightness(img),
        "glare": compute_glare_ratio(img),
    }


def evaluate_ocr_success(ocr_items: List[OCRItem], gt_dict: Dict[str, Any]) -> Tuple[bool, int, Dict[str, bool]]:
    word_count = len(ocr_items)
    tokens = [it.text for it in ocr_items]
    field_hits = {}
    any_hit = False

    for f in FIELDS:
        val = gt_dict.get(f)
        if not is_non_empty(val):
            field_hits[f] = False
            continue
        v_str = str(val).strip()
        if f == "total_amount":
            _, _, hit = check_upper_bound_amount(v_str, tokens)
        else:
            _, _, hit = check_upper_bound_string(v_str, tokens)
        field_hits[f] = hit
        if hit:
            any_hit = True

    ocr_success = (word_count >= 30) and any_hit
    return ocr_success, word_count, field_hits


def main() -> None:
    print("=" * 80)
    print("QUALITY GATE CALIBRATION: CLEAN VS SYNTHETIC DATASET")
    print("=" * 80)

    # 1. Load Manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    synth_records = manifest["records"]

    df_gt = pd.read_csv(GROUND_TRUTH_CSV).set_index("filename")
    unique_sources = sorted(list(set(r["source_image"] for r in synth_records)))

    dataset_rows = []

    # Process 12 Clean Originals
    print("\n[1/2] Analyzing 12 Clean Originals...")
    for src in unique_sources:
        img_p = RAW_DIR / src
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        m = compute_metrics(img)
        gt = df_gt.loc[src].to_dict() if src in df_gt.index else {}

        # Load clean OCR
        clean_json = CLEAN_CACHE_DIR / f"{src}.json"
        ocr_items = []
        if clean_json.exists():
            with open(clean_json, "r", encoding="utf-8") as f:
                ocr_items = [OCRItem.from_dict(d) for d in json.load(f)]

        ocr_ok, wc, hits = evaluate_ocr_success(ocr_items, gt)

        dataset_rows.append({
            "type": "clean",
            "filename": src,
            "source_image": src,
            "severity": "clean",
            "background": "none",
            "width": m["width"],
            "height": m["height"],
            "blur": m["blur"],
            "brightness": m["brightness"],
            "glare": m["glare"],
            "word_count": wc,
            "ocr_success": ocr_ok,
            "field_hits": hits,
            "gt": gt,
        })

    # Process 72 Synthetic Images
    print("\n[2/2] Analyzing 72 Synthetic Photos...")
    for r in synth_records:
        fn = r["filename"]
        img_p = Path(r["relative_path"])
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        m = compute_metrics(img)
        gt = r["ground_truth_invoice"]
        sev = r["severity"]
        bg = r["background"]

        # Check for calib cache or ocr cache
        cache_slug = f"{sev}_{bg}"
        c_path1 = CALIB_CACHE_DIR / cache_slug / f"{Path(fn).stem}.json"
        c_path2 = Path("data/processed/synth_photos/ocr_cache") / "degraded_raw" / f"{Path(fn).stem}.json"

        ocr_items = []
        for cp in [c_path1, c_path2]:
            if cp.exists():
                with open(cp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        ocr_items = [OCRItem.from_dict(d) for d in data]
                    elif isinstance(data, dict) and "items" in data:
                        ocr_items = [OCRItem.from_dict(d) for d in data["items"]]
                break

        ocr_ok, wc, hits = evaluate_ocr_success(ocr_items, gt)

        dataset_rows.append({
            "type": "synthetic",
            "filename": fn,
            "source_image": r["source_image"],
            "severity": sev,
            "background": bg,
            "width": m["width"],
            "height": m["height"],
            "blur": m["blur"],
            "brightness": m["brightness"],
            "glare": m["glare"],
            "word_count": wc,
            "ocr_success": ocr_ok if ocr_items else None,
            "field_hits": hits,
            "gt": gt,
        })

    df = pd.DataFrame(dataset_rows)

    # Print Distributions
    print("\n" + "=" * 80)
    print("QUALITY METRIC DISTRIBUTIONS BY SUBSET")
    print("=" * 80)
    summary = df.groupby(["type", "severity", "background"])[["blur", "brightness", "glare", "word_count"]].agg(["mean", "min", "max"])
    print(summary.to_string())

    # Build Confusion Matrices for Old vs New Thresholds
    print("\n" + "=" * 80)
    print("QUALITY GATE CONFUSION MATRIX CALIBRATION")
    print("=" * 80)

    # Filter to images with OCR evaluated
    eval_df = df[df["ocr_success"].notna()].copy()
    print(f"\nEvaluated on {len(eval_df)} images with OCR available (Clean + Synthetic Degraded-Raw)")

    old_thresholds = {
        "min_blur": 80.0,
        "reject_blur": 35.0,
        "min_brightness": 50.0,
        "max_brightness": 230.0,
        "max_glare_ratio": 0.15,
    }

    # Calibrated thresholds derived from clean document scans (brightness 238-251)
    # and synthetic paper photos (blur 40-370)
    new_thresholds = {
        "min_blur": 45.0,
        "reject_blur": 30.0,
        "min_brightness": 40.0,
        "max_brightness": 255.0,
        "max_glare_ratio": 0.95,
    }

    def simulate_gate(row: pd.Series, t: Dict[str, float]) -> str:
        blur = row["blur"]
        bright = row["brightness"]
        glare = row["glare"]

        # Reject checks
        if blur < t["reject_blur"] or bright < 35.0:
            return "REJECT"

        # Warn checks
        if blur < t["min_blur"] or bright < t["min_brightness"] or bright > t["max_brightness"] or glare > t["max_glare_ratio"]:
            return "WARN"

        return "PASS"

    eval_df["gate_old"] = eval_df.apply(lambda r: simulate_gate(r, old_thresholds), axis=1)
    eval_df["gate_new"] = eval_df.apply(lambda r: simulate_gate(r, new_thresholds), axis=1)

    print("\n[Old / Uncalibrated Thresholds]")
    print(f"  min_blur={old_thresholds['min_blur']}, max_brightness={old_thresholds['max_brightness']}, max_glare_ratio={old_thresholds['max_glare_ratio']}")
    cm_old = pd.crosstab(eval_df["gate_old"], eval_df["ocr_success"], margins=True)
    print(cm_old.to_string())

    print("\n[New / Calibrated Thresholds]")
    print(f"  min_blur={new_thresholds['min_blur']}, max_brightness={new_thresholds['max_brightness']}, max_glare_ratio={new_thresholds['max_glare_ratio']}")
    cm_new = pd.crosstab(eval_df["gate_new"], eval_df["ocr_success"], margins=True)
    print(cm_new.to_string())

    # Summary performance
    old_pass_rate_clean = (eval_df[eval_df["type"] == "clean"]["gate_old"] == "PASS").mean() * 100
    new_pass_rate_clean = (eval_df[eval_df["type"] == "clean"]["gate_new"] == "PASS").mean() * 100
    print(f"\nClean Originals PASS Rate: Old={old_pass_rate_clean:.1f}% -> New={new_pass_rate_clean:.1f}%")

    old_pass_success = (eval_df[eval_df["ocr_success"] == True]["gate_old"] == "PASS").mean() * 100
    new_pass_success = (eval_df[eval_df["ocr_success"] == True]["gate_new"] == "PASS").mean() * 100
    print(f"OCR Success Images PASS Rate (Sensitivity): Old={old_pass_success:.1f}% -> New={new_pass_success:.1f}%")

    return df


if __name__ == "__main__":
    main()
