"""Generate High-Precision YOLO Pre-Labels for 24 Synthetic GST Invoices.

Steps:
  1. Run EasyOCR on CPU for each image, caching word boxes in data/processed/ocr_cache/.
  2. Ground-truth guided alignment for invoice_number, invoice_date, supplier/buyer gstin,
     grand_total, and supplier name block for vendor.
  3. Heuristic suggestions for table and vendor from src/invoice_vision/prelabel/suggest.py.
  4. Preserves 8 YOLO classes (vendor, invoice_no, date, gstin, table, total, qr, stamp_signature).
  5. Writes pre-labels to data/prelabels_gst_synth/<stem>.txt.
  6. Reports coverage per class and per layout (counts only).
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
from dateutil import parser as date_parser
import easyocr
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from invoice_vision.prelabel.suggest import (
    CLASS_NAMES,
    SuggestedBox,
    WordBox,
    load_words_from_source,
    suggest_date_box,
    suggest_gstin_boxes,
    suggest_invoice_no_box,
    suggest_qr_box,
    suggest_table_box,
    suggest_total_box,
    suggest_vendor_box,
)

RAW_SYNTH_DIR = ROOT_DIR / "data" / "raw" / "gst_synth"
LABELS_SYNTH_DIR = ROOT_DIR / "data" / "processed" / "gst_synth_labels"
OCR_CACHE_DIR = ROOT_DIR / "data" / "processed" / "ocr_cache"
PRELABELS_OUT_DIR = ROOT_DIR / "data" / "prelabels_gst_synth"
MANIFEST_PATH = ROOT_DIR / "data" / "gst_synth_manifest.csv"


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


def box_pts_to_rect(box_pts: List[List[float]]) -> Tuple[float, float, float, float]:
    """Convert 4-point polygon to (xmin, ymin, xmax, ymax)."""
    xs = [p[0] for p in box_pts]
    ys = [p[1] for p in box_pts]
    return min(xs), min(ys), max(xs), max(ys)


def rect_to_yolo(rect: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert (xmin, ymin, xmax, ymax) to normalized YOLO (xc, yc, w, h)."""
    xmin, ymin, xmax, ymax = rect
    xmin = max(0.0, min(float(img_w), xmin))
    ymin = max(0.0, min(float(img_h), ymin))
    xmax = max(0.0, min(float(img_w), xmax))
    ymax = max(0.0, min(float(img_h), ymax))

    bw = max(1.0, xmax - xmin)
    bh = max(1.0, ymax - ymin)
    xc = xmin + bw / 2.0
    yc = ymin + bh / 2.0
    return (xc / img_w, yc / img_h, bw / img_w, bh / img_h)


def merge_rects(r1: Tuple[float, float, float, float], r2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Enclose two rects into a single bounding rect."""
    return (min(r1[0], r2[0]), min(r1[1], r2[1]), max(r1[2], r2[2]), max(r1[3], r2[3]))


def compute_iou(r1: Tuple[float, float, float, float], r2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union between two (xmin, ymin, xmax, ymax) rects."""
    xi1 = max(r1[0], r2[0])
    yi1 = max(r1[1], r2[1])
    xi2 = min(r1[2], r2[2])
    yi2 = min(r1[3], r2[3])
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area1 = max(1.0, (r1[2] - r1[0]) * (r1[3] - r1[1]))
    area2 = max(1.0, (r2[2] - r2[0]) * (r2[3] - r2[1]))
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


# ==============================================================================
# GROUND TRUTH ALIGNMENT LOGIC
# ==============================================================================

def match_gt_invoice_no(gt_val: str, words: List[dict]) -> Optional[Tuple[float, float, float, float]]:
    """Match ground-truth invoice number against OCR words."""
    if not gt_val or str(gt_val).strip() in ("", "nan"):
        return None
    gt_norm = normalize_str(gt_val)
    gt_len = to_lenient(gt_val)

    # 1. Exact or substring match in single word
    for w in words:
        wt_norm = normalize_str(w["text"])
        wt_len = to_lenient(w["text"])
        if wt_norm == gt_norm or wt_len == gt_len:
            return box_pts_to_rect(w["box"])
        if len(gt_norm) >= 4 and (gt_norm in wt_norm or wt_norm in gt_norm):
            return box_pts_to_rect(w["box"])
        if len(gt_norm) >= 4 and abs(len(wt_norm) - len(gt_norm)) <= 2:
            if levenshtein_dist(gt_norm, wt_norm) <= 2:
                return box_pts_to_rect(w["box"])

    # 2. Bigrams
    for i in range(len(words) - 1):
        comb = normalize_str(words[i]["text"] + words[i + 1]["text"])
        if comb == gt_norm or (len(gt_norm) >= 4 and gt_norm in comb):
            return merge_rects(box_pts_to_rect(words[i]["box"]), box_pts_to_rect(words[i + 1]["box"]))
        if len(gt_norm) >= 4 and abs(len(comb) - len(gt_norm)) <= 2:
            if levenshtein_dist(gt_norm, comb) <= 2:
                return merge_rects(box_pts_to_rect(words[i]["box"]), box_pts_to_rect(words[i + 1]["box"]))

    return None


def match_gt_date(gt_val: str, words: List[dict]) -> Optional[Tuple[float, float, float, float]]:
    """Match ground-truth invoice date against OCR words."""
    if not gt_val or str(gt_val).strip() in ("", "nan"):
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
            return box_pts_to_rect(w["box"])
        if gt_date_obj:
            try:
                w_date = date_parser.parse(wt, dayfirst=True, fuzzy=False).date()
                if w_date == gt_date_obj:
                    return box_pts_to_rect(w["box"])
            except Exception:
                pass

    # 2. Bigrams
    for i in range(len(words) - 1):
        comb = words[i]["text"] + " " + words[i + 1]["text"]
        if normalize_str(comb) == gt_norm:
            return merge_rects(box_pts_to_rect(words[i]["box"]), box_pts_to_rect(words[i + 1]["box"]))
        if gt_date_obj:
            try:
                w_date = date_parser.parse(comb, dayfirst=True, fuzzy=False).date()
                if w_date == gt_date_obj:
                    return merge_rects(box_pts_to_rect(words[i]["box"]), box_pts_to_rect(words[i + 1]["box"]))
            except Exception:
                pass

    # 3. Trigrams (e.g. 08 Apr 2025)
    for i in range(len(words) - 2):
        comb = words[i]["text"] + " " + words[i + 1]["text"] + " " + words[i + 2]["text"]
        if gt_date_obj:
            try:
                w_date = date_parser.parse(comb, dayfirst=True, fuzzy=False).date()
                if w_date == gt_date_obj:
                    r1 = merge_rects(box_pts_to_rect(words[i]["box"]), box_pts_to_rect(words[i + 1]["box"]))
                    return merge_rects(r1, box_pts_to_rect(words[i + 2]["box"]))
            except Exception:
                pass

    return None


def match_gt_gstin(gt_val: str, words: List[dict]) -> Optional[Tuple[float, float, float, float]]:
    """Match ground-truth GSTIN against OCR words."""
    if not gt_val or str(gt_val).strip() in ("", "nan"):
        return None
    gt_norm = normalize_str(gt_val)
    gt_len = to_lenient(gt_val)

    for w in words:
        wt_norm = normalize_str(w["text"])
        wt_len = to_lenient(w["text"])
        if wt_norm == gt_norm or wt_len == gt_len:
            return box_pts_to_rect(w["box"])
        if len(gt_norm) == 15 and 13 <= len(wt_norm) <= 17:
            if (wt_norm[:2] == gt_norm[:2] and levenshtein_dist(wt_norm, gt_norm) <= 4) or levenshtein_dist(wt_len, gt_len) <= 3:
                return box_pts_to_rect(w["box"])
        if gt_norm in wt_norm or gt_len in wt_len:
            return box_pts_to_rect(w["box"])

    # Bigrams
    for i in range(len(words) - 1):
        comb_norm = normalize_str(words[i]["text"] + words[i + 1]["text"])
        comb_len = to_lenient(words[i]["text"] + words[i + 1]["text"])
        if comb_norm == gt_norm or comb_len == gt_len:
            return merge_rects(box_pts_to_rect(words[i]["box"]), box_pts_to_rect(words[i + 1]["box"]))
        if gt_norm in comb_norm or gt_len in comb_len:
            return merge_rects(box_pts_to_rect(words[i]["box"]), box_pts_to_rect(words[i + 1]["box"]))

    return None


def match_gt_total(gt_val: Any, words: List[dict]) -> Optional[Tuple[float, float, float, float]]:
    """Match ground-truth grand total against OCR words."""
    if gt_val is None or str(gt_val).strip() in ("", "nan"):
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
            return box_pts_to_rect(w["box"])
        if gt_float is not None and wt_clean:
            try:
                wf = float(wt_clean)
                if abs(wf - gt_float) <= 0.5:
                    return box_pts_to_rect(w["box"])
            except Exception:
                pass

    return None


def match_gt_vendor_block(
    supplier_name: str, words: List[dict], img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Match supplier name tokens in top 40% of invoice to form vendor block."""
    if not supplier_name or str(supplier_name).strip() in ("", "nan"):
        return None

    tokens = [t.lower() for t in re.findall(r"\b[A-Za-z0-9]{3,}\b", supplier_name)]
    if not tokens:
        return None

    matched_boxes = []
    top_limit = img_h * 0.40

    for w in words:
        rect = box_pts_to_rect(w["box"])
        if rect[1] > top_limit:
            continue
        wt_clean = re.sub(r"[^a-zA-Z0-9]", "", w["text"].lower())
        for tok in tokens:
            if tok in wt_clean or wt_clean in tok or (len(tok) >= 4 and levenshtein_dist(tok, wt_clean) <= 1):
                matched_boxes.append(rect)
                break

    if not matched_boxes:
        return None

    merged = matched_boxes[0]
    for b in matched_boxes[1:]:
        merged = merge_rects(merged, b)

    # Slightly pad supplier name rect
    pad_x = 10.0
    pad_y = 6.0
    return (
        max(0.0, merged[0] - pad_x),
        max(0.0, merged[1] - pad_y),
        min(float(img_w), merged[2] + pad_x),
        min(float(img_h), merged[3] + pad_y),
    )


# ==============================================================================
# PIPELINE RUNNER
# ==============================================================================

def run_prelabel_pipeline() -> Tuple[
    Dict[str, Dict[str, int]], Dict[str, int], Dict[str, Dict[str, int]], Dict[str, int]
]:
    """Run EasyOCR, align GT and heuristics, and write YOLO pre-labels."""
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PRELABELS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing {MANIFEST_PATH}")

    manifest_df = pd.read_csv(MANIFEST_PATH)
    total_manifest = len(manifest_df)
    print(f"Loaded manifest with {total_manifest} records.")

    print("\nInitializing EasyOCR Reader (CPU mode, en)...")
    reader = easyocr.Reader(["en"], gpu=False)
    print("EasyOCR Reader ready.\n")

    layout_class_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_class_counts: Dict[str, int] = defaultdict(int)

    new_layout_class_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    new_total_class_counts: Dict[str, int] = defaultdict(int)

    for idx, row in manifest_df.iterrows():
        fn = str(row["filename"])
        stem = Path(fn).stem
        layout = str(row["layout"])
        img_path = RAW_SYNTH_DIR / fn
        label_json_path = LABELS_SYNTH_DIR / f"{stem}.json"
        cache_json_path = OCR_CACHE_DIR / f"{fn}.json"

        is_new = int(re.sub(r"\D", "", stem) or 0) >= 24

        print(f"[{idx+1:02d}/{total_manifest:02d}] Processing {fn} ({layout}, {row['level']})...")

        if not img_path.exists():
            print(f"  Warning: image {img_path} not found, skipping.")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: failed to read image {img_path}, skipping.")
            continue
        h, w = img.shape[:2]

        # 1. EasyOCR (cached)
        if cache_json_path.exists():
            with open(cache_json_path, "r", encoding="utf-8") as f:
                ocr_words = json.load(f)
        else:
            ocr_results = reader.readtext(img)
            ocr_words = [
                {
                    "box": [[float(p[0]), float(p[1])] for p in box],
                    "text": str(text),
                    "confidence": float(conf),
                }
                for box, text, conf in ocr_results
            ]
            with open(cache_json_path, "w", encoding="utf-8") as f:
                json.dump(ocr_words, f, indent=2, ensure_ascii=False)
            # Also cache by stem for interoperability
            stem_cache_path = OCR_CACHE_DIR / f"{stem}.json"
            if not stem_cache_path.exists():
                with open(stem_cache_path, "w", encoding="utf-8") as f:
                    json.dump(ocr_words, f, indent=2, ensure_ascii=False)

        # Load ground truth JSON
        lbl_data: Dict[str, Any] = {}
        if label_json_path.exists():
            with open(label_json_path, "r", encoding="utf-8") as f:
                lbl_data = json.load(f)

        # Standardize words for suggest.py
        word_boxes, _, _ = load_words_from_source(ocr_words, img_shape=(h, w))

        proposed_boxes: List[Tuple[int, Tuple[float, float, float, float], str]] = []

        # ----------------------------------------------------------------------
        # CLASS 0: VENDOR
        # ----------------------------------------------------------------------
        supplier_name = str(lbl_data.get("supplier", {}).get("name", ""))
        v_rect = match_gt_vendor_block(supplier_name, ocr_words, w, h)
        if v_rect is None:
            v_sug = suggest_vendor_box(word_boxes, w, h)
            if v_sug:
                v_rect = (v_sug.xmin, v_sug.ymin, v_sug.xmax, v_sug.ymax)
        if v_rect:
            proposed_boxes.append((0, v_rect, "vendor"))

        # ----------------------------------------------------------------------
        # CLASS 1: INVOICE_NO
        # ----------------------------------------------------------------------
        inv_no_val = str(lbl_data.get("invoice_number", ""))
        inv_rect = match_gt_invoice_no(inv_no_val, ocr_words)
        if inv_rect is None:
            inv_sug = suggest_invoice_no_box(word_boxes, w, h)
            if inv_sug:
                inv_rect = (inv_sug.xmin, inv_sug.ymin, inv_sug.xmax, inv_sug.ymax)
        if inv_rect:
            proposed_boxes.append((1, inv_rect, "invoice_no"))

        # ----------------------------------------------------------------------
        # CLASS 2: DATE
        # ----------------------------------------------------------------------
        date_val = str(lbl_data.get("invoice_date", ""))
        date_rect = match_gt_date(date_val, ocr_words)
        if date_rect is None:
            dt_sug = suggest_date_box(word_boxes, w, h)
            if dt_sug:
                date_rect = (dt_sug.xmin, dt_sug.ymin, dt_sug.xmax, dt_sug.ymax)
        if date_rect:
            proposed_boxes.append((2, date_rect, "date"))

        # ----------------------------------------------------------------------
        # CLASS 3: GSTIN (supplier and buyer)
        # ----------------------------------------------------------------------
        sup_gstin = str(lbl_data.get("supplier", {}).get("gstin", ""))
        buy_gstin = str(lbl_data.get("buyer", {}).get("gstin", ""))
        gstin_found: List[Tuple[float, float, float, float]] = []

        if sup_gstin:
            r = match_gt_gstin(sup_gstin, ocr_words)
            if r:
                gstin_found.append(r)
        if buy_gstin:
            r = match_gt_gstin(buy_gstin, ocr_words)
            if r:
                gstin_found.append(r)

        # Fallback / heuristic gstin boxes
        for gst_sug in suggest_gstin_boxes(word_boxes, w, h):
            sug_r = (gst_sug.xmin, gst_sug.ymin, gst_sug.xmax, gst_sug.ymax)
            if not any(compute_iou(sug_r, existing) > 0.4 for existing in gstin_found):
                gstin_found.append(sug_r)

        for gr in gstin_found:
            proposed_boxes.append((3, gr, "gstin"))

        # ----------------------------------------------------------------------
        # CLASS 4: TABLE
        # ----------------------------------------------------------------------
        tbl_sug = suggest_table_box(word_boxes, w, h)
        if tbl_sug:
            proposed_boxes.append((4, (tbl_sug.xmin, tbl_sug.ymin, tbl_sug.xmax, tbl_sug.ymax), "table"))

        # ----------------------------------------------------------------------
        # CLASS 5: TOTAL
        # ----------------------------------------------------------------------
        tot_val = lbl_data.get("totals", {}).get("grand_total")
        tot_rect = match_gt_total(tot_val, ocr_words)
        if tot_rect is None:
            tot_sug = suggest_total_box(word_boxes, w, h)
            if tot_sug:
                tot_rect = (tot_sug.xmin, tot_sug.ymin, tot_sug.xmax, tot_sug.ymax)
        if tot_rect:
            proposed_boxes.append((5, tot_rect, "total"))

        # ----------------------------------------------------------------------
        # CLASS 6: QR CODE
        # ----------------------------------------------------------------------
        qr_sug = suggest_qr_box(img=img, width_px=w, height_px=h)
        if qr_sug:
            proposed_boxes.append((6, (qr_sug.xmin, qr_sug.ymin, qr_sug.xmax, qr_sug.ymax), "qr"))

        # Deduplicate same-class overlapping boxes (IoU > 0.5)
        final_boxes: List[Tuple[int, Tuple[float, float, float, float], str]] = []
        for cid, rect, cname in proposed_boxes:
            overlap = False
            for prev_cid, prev_rect, _ in final_boxes:
                if prev_cid == cid and compute_iou(rect, prev_rect) > 0.5:
                    overlap = True
                    break
            if not overlap:
                final_boxes.append((cid, rect, cname))

        # Write YOLO format
        yolo_lines: List[str] = []
        for cid, rect, cname in final_boxes:
            xc, yc, bw, bh = rect_to_yolo(rect, w, h)
            yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            layout_class_counts[layout][cname] += 1
            total_class_counts[cname] += 1
            if is_new:
                new_layout_class_counts[layout][cname] += 1
                new_total_class_counts[cname] += 1

        out_txt = PRELABELS_OUT_DIR / f"{stem}.txt"
        out_txt.write_text("\n".join(yolo_lines) + "\n" if yolo_lines else "", encoding="utf-8")

    return layout_class_counts, total_class_counts, new_layout_class_counts, new_total_class_counts


def print_coverage_report(
    layout_class_counts: Dict[str, Dict[str, int]],
    total_class_counts: Dict[str, int],
    title: str = "PRE-LABEL COVERAGE SUMMARY",
) -> None:
    """Print clean coverage table per class and per layout (counts only)."""
    layouts = ["classic", "handwritten", "minimal", "thermal"]
    classes = CLASS_NAMES

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    header = f"{'Class':<16} | " + " | ".join(f"{ly:>12}" for ly in layouts) + " | " + f"{'Total':>8}"
    print(header)
    print("-" * 80)

    for c in classes:
        row_str = f"{c:<16} | "
        for ly in layouts:
            cnt = layout_class_counts[ly].get(c, 0)
            row_str += f"{cnt:>12} | "
        tot = total_class_counts.get(c, 0)
        row_str += f"{tot:>8}"
        print(row_str)

    print("-" * 80)
    tot_row = f"{'ALL CLASSES':<16} | "
    grand_total = 0
    for ly in layouts:
        ly_sum = sum(layout_class_counts[ly].values())
        grand_total += ly_sum
        tot_row += f"{ly_sum:>12} | "
    tot_row += f"{grand_total:>8}"
    print(tot_row)
    print("=" * 80 + "\n")


def main() -> None:
    all_ly, all_tot, new_ly, new_tot = run_prelabel_pipeline()
    print_coverage_report(
        new_ly,
        new_tot,
        title="PRE-LABEL COVERAGE FOR 60 NEW SYNTHETIC INVOICES (gstsyn_024 - gstsyn_083)",
    )
    print_coverage_report(
        all_ly,
        all_tot,
        title="PRE-LABEL COVERAGE FOR ALL 84 SYNTHETIC INVOICES (gstsyn_000 - gstsyn_083)",
    )


if __name__ == "__main__":
    main()

