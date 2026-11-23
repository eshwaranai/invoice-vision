"""Generate YOLO Pre-Labels from Cached OCR and Ground Truth.

For the 30 clean Indian GST invoices:
  - Aligns cached EasyOCR word bounding boxes with ground truth values.
  - Matches classes:
      * 1: invoice_no
      * 2: date
      * 3: gstin (supplier and customer)
      * 5: total
  - Writes normalized YOLO txt files to data/prelabels/<stem>.txt
  - Writes data/prelabels/README.md warning that pre-labels are unreviewed.
  - Writes data/classes.txt.
  - Reports coverage per class.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
from dateutil import parser as date_parser
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
OCR_CACHE_DIR = ROOT_DIR / "data" / "processed" / "ocr_cache"
RAW_DIR = ROOT_DIR / "data" / "raw" / "indian_gst"
GT_CSV_PATH = ROOT_DIR / "data" / "ground_truth_gst.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "prelabels"
CLASSES_PATH = ROOT_DIR / "data" / "classes.txt"

CLASSES = [
    "vendor",           # 0
    "invoice_no",       # 1
    "date",             # 2
    "gstin",            # 3
    "table",            # 4
    "total",            # 5
    "qr",               # 6
    "stamp_signature",  # 7
]


def normalize_str(s: Any) -> str:
    """Uppercase and strip punctuation/whitespace."""
    return re.sub(r"[\s\-_:/\,\.]", "", str(s or "")).upper()


def to_lenient(s: Any) -> str:
    """Map visually ambiguous OCR characters to canonical forms."""
    trans = str.maketrans({
        "O": "0", "0": "0",
        "I": "1", "1": "1", "L": "1",
        "S": "5", "5": "5",
        "Z": "7", "7": "7",
        "B": "8", "8": "8",
    })
    return normalize_str(s).translate(trans)


def levenshtein_dist(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def box_to_yolo(
    box_pts: List[List[float]], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert OCR 4-point polygon box to normalized YOLO [xc, yc, w, h]."""
    xs = [p[0] for p in box_pts]
    ys = [p[1] for p in box_pts]
    xmin = max(0.0, min(xs))
    xmax = min(float(img_w), max(xs))
    ymin = max(0.0, min(ys))
    ymax = min(float(img_h), max(ys))

    bw = max(1.0, xmax - xmin)
    bh = max(1.0, ymax - ymin)
    xc = xmin + bw / 2.0
    yc = ymin + bh / 2.0

    return (xc / img_w, yc / img_h, bw / img_w, bh / img_h)


def merge_boxes(box1: List[List[float]], box2: List[List[float]]) -> List[List[float]]:
    """Enclose two boxes into a single bounding quad."""
    xs = [p[0] for p in box1] + [p[0] for p in box2]
    ys = [p[1] for p in box1] + [p[1] for p in box2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def match_invoice_no(gt_val: str, words: List[dict]) -> Optional[List[List[float]]]:
    """Find OCR box matching ground-truth invoice number."""
    if not gt_val or str(gt_val).strip() == "" or str(gt_val).lower() == "nan":
        return None

    gt_norm = normalize_str(gt_val)
    gt_len = to_lenient(gt_val)

    # 1. Single word exact or lenient or fuzzy
    for w in words:
        wt_norm = normalize_str(w["text"])
        wt_len = to_lenient(w["text"])
        if wt_norm == gt_norm or wt_len == gt_len:
            return w["box"]
        if len(gt_norm) >= 4 and (gt_norm in wt_norm or wt_norm in gt_norm):
            return w["box"]
        if len(gt_norm) >= 4 and abs(len(wt_norm) - len(gt_norm)) <= 2:
            if levenshtein_dist(gt_norm, wt_norm) <= 2:
                return w["box"]

    # 2. Bigrams (two adjacent words)
    for i in range(len(words) - 1):
        comb = normalize_str(words[i]["text"] + words[i + 1]["text"])
        if comb == gt_norm or (len(gt_norm) >= 4 and gt_norm in comb):
            return merge_boxes(words[i]["box"], words[i + 1]["box"])
        if len(gt_norm) >= 4 and abs(len(comb) - len(gt_norm)) <= 2:
            if levenshtein_dist(gt_norm, comb) <= 2:
                return merge_boxes(words[i]["box"], words[i + 1]["box"])

    return None


def match_date(gt_val: str, words: List[dict]) -> Optional[List[List[float]]]:
    """Find OCR box matching ground-truth invoice date."""
    if not gt_val or str(gt_val).strip() == "" or str(gt_val).lower() == "nan":
        return None

    try:
        gt_date_obj = date_parser.parse(str(gt_val).strip(), dayfirst=True).date()
    except Exception:
        gt_date_obj = None

    gt_norm = normalize_str(gt_val)

    # 1. Single word match
    for w in words:
        wt = w["text"].strip()
        if normalize_str(wt) == gt_norm:
            return w["box"]
        if gt_date_obj:
            try:
                w_date = date_parser.parse(wt, dayfirst=True, fuzzy=False).date()
                if w_date == gt_date_obj:
                    return w["box"]
            except Exception:
                pass

    # 2. Bigram match
    for i in range(len(words) - 1):
        comb = words[i]["text"] + " " + words[i + 1]["text"]
        if normalize_str(comb) == gt_norm:
            return merge_boxes(words[i]["box"], words[i + 1]["box"])
        if gt_date_obj:
            try:
                w_date = date_parser.parse(comb, dayfirst=True, fuzzy=False).date()
                if w_date == gt_date_obj:
                    return merge_boxes(words[i]["box"], words[i + 1]["box"])
            except Exception:
                pass

    return None


def match_gstin(gt_val: str, words: List[dict]) -> Optional[List[List[float]]]:
    """Find OCR box matching ground-truth GSTIN."""
    if not gt_val or str(gt_val).strip() == "" or str(gt_val).lower() == "nan":
        return None

    gt_norm = normalize_str(gt_val)
    gt_len = to_lenient(gt_val)

    # 1. Exact or lenient match
    for w in words:
        wt_norm = normalize_str(w["text"])
        wt_len = to_lenient(w["text"])
        if wt_norm == gt_norm or wt_len == gt_len:
            return w["box"]
        if len(gt_norm) == 15 and 13 <= len(wt_norm) <= 17:
            if (wt_norm[:2] == gt_norm[:2] and levenshtein_dist(wt_norm, gt_norm) <= 4) or levenshtein_dist(wt_len, gt_len) <= 3:
                return w["box"]

    return None


def match_total(gt_val: str, words: List[dict]) -> Optional[List[List[float]]]:
    """Find OCR box matching ground-truth total amount."""
    if not gt_val or str(gt_val).strip() == "" or str(gt_val).lower() == "nan":
        return None

    clean_gt = re.sub(r"[^\d\.]", "", str(gt_val).replace(",", "").strip())
    try:
        gt_float = float(clean_gt)
    except Exception:
        gt_float = None
    int_gt = clean_gt.split(".")[0]

    for w in words:
        wt = w["text"].strip()
        wt_clean = re.sub(r"[^\d\.]", "", wt.replace(",", ""))
        if int_gt and len(int_gt) >= 3 and int_gt == wt_clean.split(".")[0]:
            return w["box"]
        if gt_float is not None and wt_clean:
            try:
                wf = float(wt_clean)
                if abs(wf - gt_float) <= 0.5:
                    return w["box"]
            except Exception:
                pass

    return None


def write_classes_txt() -> None:
    """Write YOLO class definition file."""
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        for c in CLASSES:
            f.write(f"{c}\n")
    print(f"Wrote classes to: {CLASSES_PATH}")


def write_prelabels_readme() -> None:
    """Write quality disclaimer in data/prelabels/README.md."""
    readme_path = OUTPUT_DIR / "README.md"
    content = """# Automated Pre-Labels (YOLO Object Detection Format)

> [!WARNING]
> **UNREVIEWED HEURISTIC LABELS - MANDATORY HUMAN VERIFICATION REQUIRED**:
> The label files in this directory (`*.txt`) were generated automatically by aligning cached EasyOCR word bounding boxes with ground truth annotations. 
> - **Incomplete Field Coverage**: Only 4 classes (`invoice_no` [1], `date` [2], `gstin` [3], and `total` [5]) are generated.
> - **Missing Classes**: Classes `vendor` (0), `table` (4), `qr` (6), and `stamp_signature` (7) are **NOT** pre-labeled and must be drawn manually.
> - **Potential Errors**: OCR text errors, bounding box offsets, or multi-token splits may cause slightly loose, misaligned, or omitted boxes.
> **DO NOT TRAIN MODELS DIRECTLY ON THESE PRE-LABELS** without completing a manual review per `docs/annotation_guideline.md`.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote README to: {readme_path}")


def main() -> None:
    print("=" * 75)
    print("GENERATING YOLO PRE-LABELS FOR 30 CLEAN INVOICES")
    print("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_classes_txt()
    write_prelabels_readme()

    if not GT_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {GT_CSV_PATH}")

    gt_df = pd.read_csv(GT_CSV_PATH).set_index("filename")

    coverage = {
        "invoice_no": 0,
        "date": 0,
        "gstin": 0,
        "total": 0,
    }
    total_boxes = 0
    total_images = len(gt_df)

    for fn, row in gt_df.iterrows():
        stem = Path(str(fn)).stem
        out_txt = OUTPUT_DIR / f"{stem}.txt"
        cache_f = OCR_CACHE_DIR / f"{fn}.json"
        img_f = RAW_DIR / str(fn)

        if not cache_f.exists() or not img_f.exists():
            print(f"Skipping {fn} (missing cache or raw image)")
            continue

        # Get exact image dimensions
        img = cv2.imread(str(img_f))
        h, w = img.shape[:2]

        with open(cache_f, "r", encoding="utf-8") as f:
            words = json.load(f)

        yolo_lines: List[str] = []

        # 1. invoice_no (Class 1)
        inv_box = match_invoice_no(str(row.get("invoice_number", "")), words)
        if inv_box is not None:
            xc, yc, bw, bh = box_to_yolo(inv_box, w, h)
            yolo_lines.append(f"1 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            coverage["invoice_no"] += 1
            total_boxes += 1

        # 2. date (Class 2)
        date_box = match_date(str(row.get("purchase_date", "")), words)
        if date_box is not None:
            xc, yc, bw, bh = box_to_yolo(date_box, w, h)
            yolo_lines.append(f"2 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            coverage["date"] += 1
            total_boxes += 1

        # 3. gstin (Class 3) - check supplier and customer
        gst_boxes_found = 0
        for col in ["supplier_gst", "customer_gst"]:
            val = str(row.get(col, "")).strip()
            g_box = match_gstin(val, words)
            if g_box is not None:
                xc, yc, bw, bh = box_to_yolo(g_box, w, h)
                yolo_lines.append(f"3 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                gst_boxes_found += 1
                total_boxes += 1
        if gst_boxes_found > 0:
            coverage["gstin"] += 1

        # 4. total (Class 5)
        tot_box = match_total(str(row.get("total_amount", "")), words)
        if tot_box is not None:
            xc, yc, bw, bh = box_to_yolo(tot_box, w, h)
            yolo_lines.append(f"5 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            coverage["total"] += 1
            total_boxes += 1

        # Write YOLO label file
        with open(out_txt, "w", encoding="utf-8") as f:
            for l in yolo_lines:
                f.write(f"{l}\n")

    print("\nPre-Label Coverage Report Across 30 Clean Images:")
    print("-" * 75)
    for cls_name, count in coverage.items():
        pct = (count / total_images) * 100.0
        print(f"  * Class {cls_name:<12}: {count:2d}/{total_images} images with pre-label ({pct:5.1f}%)")
    print(f"\nTotal Pre-Label Bounding Boxes Generated: {total_boxes}")
    print(f"Output Directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
