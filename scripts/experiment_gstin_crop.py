"""Experiment: Crop-level OCR for GSTIN extraction across preprocessing variants.

Compares:
  (a) no scaling
  (b) 3x cubic upscaling
  (c) 3x + CLAHE on grayscale
  (d) 3x + Otsu threshold

Measures:
  - Upper-Bound Recall on supplier_gst (Exact, Lenient, Fuzzy)
  - Number of extracted GSTINs passing Mod-36 Luhn checksum after position-aware correction
  - Average inference time per crop
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd

# Add src/ to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from invoice_vision.ocr.easyocr_engine import OCRItem
from invoice_vision.parse.spatial_rules import (
    cluster_into_rows,
    get_box_bounds,
    position_aware_gstin_correct,
    validate_gstin_checksum,
)
from scripts.diagnose_baseline import check_upper_bound_string, normalize_str, to_lenient


RAW_DIR = Path("data/raw/indian_gst")
OCR_CACHE_DIR = Path("data/processed/ocr_cache")
CROP_CACHE_DIR = Path("data/processed/gstin_crop_cache")
GROUND_TRUTH_CSV = Path("data/ground_truth_gst.csv")
MODEL_STORAGE_DIR = Path("data/models/easyocr")
USER_NETWORK_DIR = MODEL_STORAGE_DIR / "user_network"

VARIANTS = ["no_scaling", "3x_cubic", "3x_clahe", "3x_otsu"]
GST_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def box_to_padded_rect(
    box: List[List[float]], pad: int = 10, img_w: int = 9999, img_h: int = 9999
) -> Tuple[int, int, int, int]:
    """Convert a 4-point polygon box to an axis-aligned bounding rect with padding."""
    min_x = max(0, int(np.floor(min(p[0] for p in box))) - pad)
    max_x = min(img_w, int(np.ceil(max(p[0] for p in box))) + pad)
    min_y = max(0, int(np.floor(min(p[1] for p in box))) - pad)
    max_y = min(img_h, int(np.ceil(max(p[1] for p in box))) + pad)
    return min_x, min_y, max_x, max_y


def compute_iou(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two rectangles."""
    x_left = max(r1[0], r2[0])
    y_top = max(r1[1], r2[1])
    x_right = min(r1[2], r2[2])
    y_bottom = min(r1[3], r2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    a1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
    a2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
    union = float(a1 + a2 - inter)
    return inter / union if union > 0 else 0.0


def find_gstin_candidate_rects(
    items: List[OCRItem], img_w: int, img_h: int
) -> List[Tuple[int, int, int, int]]:
    """Identify candidate bounding rectangles for GSTIN tokens in an image."""
    rows = cluster_into_rows(items)
    raw_boxes: List[List[List[float]]] = []

    # 1. Label rows: rows containing GSTIN indicator keywords
    for r in rows:
        r_txt = r.text.lower()
        # Skip tax rate breakdown rows (CGST / SGST rate columns)
        if any(hdr in r_txt for hdr in ["taxable value", "rate of tax", "cgst %", "sgst %", "igst %"]):
            continue

        has_label = any(
            kw in r_txt
            for kw in ["gstin", "gst no", "gstno", "gst.no", "gst reg", "gst num", "uin", "gst:", "gsl"]
        )
        if has_label:
            for it_idx, it in enumerate(r.items):
                clean = re.sub(r"[\s\-_:/\.]", "", it.text).upper()
                clean = re.sub(r"^(?:GSTIN|GSTNO|GSTNUMBER|GSTREG|GST|UIN)", "", clean)
                if len(clean) >= 10:
                    raw_boxes.append(it.box)
                elif any(kw in it.text.lower() for kw in ["gstin", "gst no", "gst:", "uin", "gsl"]):
                    if it_idx + 1 < len(r.items):
                        raw_boxes.append(r.items[it_idx + 1].box)
                    else:
                        # Synthetic right region of label if OCR missed the value token
                        min_x, max_x, min_y, max_y, _, _ = get_box_bounds(it.box)
                        right_box = [
                            [max_x, min_y],
                            [min(float(img_w), max_x + 300.0), min_y],
                            [min(float(img_w), max_x + 300.0), max_y],
                            [max_x, max_y],
                        ]
                        raw_boxes.append(right_box)

    # 2. Standalone alphanumeric candidate tokens (13-17 chars, must have both digits and letters)
    for it in items:
        clean = re.sub(r"[\s\-_:/\.]", "", it.text).upper()
        clean = re.sub(r"^(?:GSTIN|GSTNO|GSTNUMBER|GSTREG|GST|UIN)", "", clean)
        if 13 <= len(clean) <= 17:
            has_digit = any(c.isdigit() for c in clean)
            has_alpha = any(c.isalpha() for c in clean)
            if has_digit and has_alpha:
                raw_boxes.append(it.box)

    # Convert to padded rectangles and deduplicate using IoU > 0.5
    rects = [box_to_padded_rect(b, pad=10, img_w=img_w, img_h=img_h) for b in raw_boxes]
    unique_rects: List[Tuple[int, int, int, int]] = []
    for r in rects:
        # Avoid degenerate rects
        if (r[2] - r[0] < 15) or (r[3] - r[1] < 8):
            continue
        if not any(compute_iou(r, ur) > 0.5 for ur in unique_rects):
            unique_rects.append(r)

    return unique_rects


def preprocess_crop(crop: np.ndarray, variant: str) -> np.ndarray:
    """Apply preprocessing pipeline to a cropped candidate image region."""
    if variant == "no_scaling":
        return crop

    elif variant == "3x_cubic":
        return cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    elif variant == "3x_clahe":
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.resize(enhanced, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    elif variant == "3x_otsu":
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.resize(thresh, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_crop_ocr_for_variant(
    variant: str,
    all_crops: Dict[str, List[Tuple[Tuple[int, int, int, int], np.ndarray]]],
    force: bool = False,
) -> Dict[str, Any]:
    """Execute or load crop-level OCR for a given preprocessing variant."""
    CROP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CROP_CACHE_DIR / f"{variant}.json"

    if not force and cache_file.exists():
        print(f"Loading cached crop OCR results for variant '{variant}' from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"\n--- Running Crop-Level OCR for Variant: '{variant}' ---")
    import easyocr

    USER_NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(MODEL_STORAGE_DIR),
        user_network_directory=str(USER_NETWORK_DIR),
        download_enabled=False,
    )

    results_by_image: Dict[str, List[Dict[str, Any]]] = {}
    total_crop_count = 0
    start_time = time.perf_counter()

    for fn, crops in all_crops.items():
        image_results: List[Dict[str, Any]] = []
        for rect, crop_bgr in crops:
            total_crop_count += 1
            proc_img = preprocess_crop(crop_bgr, variant)

            t_c0 = time.perf_counter()
            raw_res = reader.readtext(proc_img, allowlist=GST_ALLOWLIST)
            t_c1 = time.perf_counter()

            # Parse detected text tokens
            tokens = []
            confidences = []
            for _, t, c in raw_res:
                t_str = str(t).strip()
                if t_str != "":
                    tokens.append(t_str)
                    confidences.append(float(c))

            image_results.append({
                "rect": list(rect),
                "tokens": tokens,
                "confidences": confidences,
                "inference_time_s": round(t_c1 - t_c0, 4),
            })

        results_by_image[fn] = image_results
        print(f"  Processed {fn}: {len(crops)} crops")

    total_time = time.perf_counter() - start_time
    avg_crop_time_ms = (total_time / total_crop_count * 1000) if total_crop_count > 0 else 0.0

    output_data = {
        "variant": variant,
        "total_crops": total_crop_count,
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_crop_ms": round(avg_crop_time_ms, 2),
        "results_by_image": results_by_image,
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Finished '{variant}' in {total_time:.2f}s ({avg_crop_time_ms:.1f} ms/crop). Saved to {cache_file}")
    return output_data


def analyze_variant_results(
    variant_data: Dict[str, Any], df_gt: pd.DataFrame
) -> Dict[str, Any]:
    """Evaluate Upper-bound recall and checksum validation statistics for a variant."""
    results_by_image = variant_data["results_by_image"]
    gst_regex = re.compile(r"\b[0-9A-Z]{15}\b")

    total_gt_count = 0
    exact_hits = 0
    lenient_hits = 0
    fuzzy_hits = 0

    total_valid_checksums_found = 0
    images_with_valid_checksum = 0
    unique_valid_gstins: Set[str] = set()

    detailed_image_reports: List[Dict[str, Any]] = []

    for _, row in df_gt.iterrows():
        fn = row["filename"]
        gt_supplier = row.get("supplier_gst")
        has_gt = gt_supplier is not None and not pd.isna(gt_supplier) and str(gt_supplier).strip() != "" and str(gt_supplier).strip().lower() != "nan"

        crop_res_list = results_by_image.get(fn, [])

        # Collect all recognized tokens in this image
        all_tokens: List[str] = []
        for cr in crop_res_list:
            all_tokens.extend(cr["tokens"])
            if len(cr["tokens"]) > 1:
                all_tokens.append("".join(cr["tokens"]))

        # Identify all 15-char candidates and test checksum
        image_valid_checksums: List[str] = []
        for tok in all_tokens:
            clean = re.sub(r"[\s\-_:/\.]", "", tok).upper()
            clean_noprefix = re.sub(r"^(?:GSTIN|GSTNO|GSTNUMBER|GSTREG|GST|UIN)", "", clean)

            candidates: List[str] = []
            if len(clean_noprefix) == 15:
                candidates.append(clean_noprefix)
            elif len(clean) == 15:
                candidates.append(clean)
            elif len(clean_noprefix) > 15:
                for i in range(len(clean_noprefix) - 14):
                    candidates.append(clean_noprefix[i : i + 15])

            for cand in candidates:
                corrected = position_aware_gstin_correct(cand)
                if validate_gstin_checksum(corrected):
                    image_valid_checksums.append(corrected)
                    unique_valid_gstins.add(corrected)

        if image_valid_checksums:
            images_with_valid_checksum += 1
            total_valid_checksums_found += len(image_valid_checksums)

        # Upper-Bound Recall calculation against supplier_gst
        exact_match, len_match, fuz_match = False, False, False
        if has_gt:
            total_gt_count += 1
            gt_str = str(gt_supplier).strip()
            exact_match, len_match, fuz_match = check_upper_bound_string(gt_str, all_tokens)
            if exact_match:
                exact_hits += 1
            if len_match:
                lenient_hits += 1
            if fuz_match:
                fuzzy_hits += 1

        detailed_image_reports.append({
            "filename": fn,
            "has_gt": has_gt,
            "gt_supplier": str(gt_supplier) if has_gt else "",
            "exact_match": exact_match,
            "lenient_match": len_match,
            "fuzzy_match": fuz_match,
            "valid_checksums": list(set(image_valid_checksums)),
        })

    return {
        "variant": variant_data["variant"],
        "total_crops": variant_data["total_crops"],
        "avg_time_ms": variant_data["avg_time_per_crop_ms"],
        "total_gt": total_gt_count,
        "exact_recall_pct": round(exact_hits / total_gt_count * 100, 1) if total_gt_count > 0 else 0.0,
        "exact_recall_count": exact_hits,
        "lenient_recall_pct": round(lenient_hits / total_gt_count * 100, 1) if total_gt_count > 0 else 0.0,
        "lenient_recall_count": lenient_hits,
        "fuzzy_recall_pct": round(fuzzy_hits / total_gt_count * 100, 1) if total_gt_count > 0 else 0.0,
        "fuzzy_recall_count": fuzzy_hits,
        "images_with_valid_checksum": images_with_valid_checksum,
        "total_valid_checksums_found": total_valid_checksums_found,
        "unique_valid_gstins_count": len(unique_valid_gstins),
        "unique_valid_gstins": sorted(list(unique_valid_gstins)),
        "detailed_reports": detailed_image_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop-level OCR experiment for GSTIN.")
    parser.add_argument("--force", action="store_true", help="Force recomputation of OCR caches.")
    args = parser.parse_args()

    df_gt = pd.read_csv(GROUND_TRUTH_CSV)

    # 1. Extract crop candidates from raw images using cached full-page OCR boxes
    print("Extracting candidate crop bounding boxes from raw invoice images...")
    all_crops: Dict[str, List[Tuple[Tuple[int, int, int, int], np.ndarray]]] = {}
    total_crops_collected = 0

    for _, row in df_gt.iterrows():
        fn = row["filename"]
        img_path = RAW_DIR / fn
        cache_path = OCR_CACHE_DIR / f"{fn}.json"

        if not img_path.exists() or not cache_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(cache_path, "r", encoding="utf-8") as f:
            items_data = json.load(f)
        items = [OCRItem.from_dict(d) for d in items_data]

        candidate_rects = find_gstin_candidate_rects(items, img_w=w, img_h=h)

        crop_list: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
        for rect in candidate_rects:
            min_x, min_y, max_x, max_y = rect
            crop = img[min_y:max_y, min_x:max_x]
            if crop.size > 0:
                crop_list.append((rect, crop))

        all_crops[fn] = crop_list
        total_crops_collected += len(crop_list)

    print(f"Collected {total_crops_collected} candidate crops across {len(all_crops)} images.")

    # 2. Run OCR on crops for each of the 4 variants
    comparison_summary: List[Dict[str, Any]] = []

    for var in VARIANTS:
        var_data = run_crop_ocr_for_variant(var, all_crops, force=args.force)
        stats = analyze_variant_results(var_data, df_gt)
        comparison_summary.append(stats)

    # 3. Print Comparison Table
    print("\n" + "=" * 105)
    print("CROP-LEVEL OCR EXPERIMENT: GSTIN EXTRACTION COMPARISON")
    print("=" * 105)
    header = f"{'Variant':<16} | {'Avg Time':<10} | {'Lenient Recall':<16} | {'Fuzzy Recall':<16} | {'Valid Checksums':<18} | {'Unique Valid GSTINs'}"
    print(header)
    print("-" * 105)

    for st in comparison_summary:
        time_str = f"{st['avg_time_ms']} ms"
        len_str = f"{st['lenient_recall_pct']}% ({st['lenient_recall_count']}/{st['total_gt']})"
        fuz_str = f"{st['fuzzy_recall_pct']}% ({st['fuzzy_recall_count']}/{st['total_gt']})"
        chk_str = f"{st['images_with_valid_checksum']}/30 images ({st['total_valid_checksums_found']} total)"
        uniq_str = f"{st['unique_valid_gstins_count']}"
        print(f"{st['variant']:<16} | {time_str:<10} | {len_str:<16} | {fuz_str:<16} | {chk_str:<18} | {uniq_str}")
    print("-" * 105)


if __name__ == "__main__":
    main()
