"""Evaluation script for baseline OCR and rule-based extraction pipeline.

Evaluates EasyOCR + Baseline Heuristics against Indian GST Ground Truth.
Generates:
  - results/baseline_per_image.csv
  - results/baseline_summary.md
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psutil
from dateutil import parser as date_parser

# Ensure src/ is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from invoice_vision.ocr.easyocr_engine import EasyOCREngine
from invoice_vision.parse.baseline_rules import extract_baseline_fields


RAW_IMAGES_DIR = Path("data/raw/indian_gst")
GROUND_TRUTH_CSV = Path("data/ground_truth_gst.csv")
RESULTS_DIR = Path("results")
PER_IMAGE_CSV = RESULTS_DIR / "baseline_per_image.csv"
SUMMARY_MD = RESULTS_DIR / "baseline_summary.md"

EVAL_FIELDS = ["invoice_number", "purchase_date", "supplier_gst", "total_amount"]


def is_non_empty(val: Any) -> bool:
    """Check if value is present and non-empty."""
    if val is None or pd.isna(val):
        return False
    s = str(val).strip()
    return s != "" and s.lower() != "nan"


def normalize_date(val: Optional[str], dayfirst: bool = True) -> Optional[str]:
    """Parse date string into ISO format (YYYY-MM-DD)."""
    if not is_non_empty(val):
        return None
    try:
        dt = date_parser.parse(str(val).strip(), dayfirst=dayfirst)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def normalize_amount(val: Optional[str]) -> Optional[float]:
    """Parse monetary string to float value."""
    if not is_non_empty(val):
        return None
    clean = re.sub(r"[^\d\.]", "", str(val).replace(",", "").strip())
    try:
        return float(clean)
    except Exception:
        return None


def normalize_text(val: Optional[str]) -> Optional[str]:
    """Uppercase and strip spaces for text codes."""
    if not is_non_empty(val):
        return None
    return str(val).upper().replace(" ", "").strip()


def lenient_text(val: Optional[str]) -> Optional[str]:
    """Canonicalize OCR character confusions: O->0, I/L->1, S->5."""
    norm = normalize_text(val)
    if norm is None:
        return None
    mapping = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5"})
    return norm.translate(mapping)


def evaluate_field(
    field: str, pred: Optional[str], truth: Optional[str]
) -> Tuple[bool, bool, Optional[bool]]:
    """Evaluate exact, normalized, and lenient match flags for a given field."""
    # 1. Exact match
    exact_match = str(pred or "").strip() == str(truth or "").strip()

    # 2. Normalized match
    norm_match = False
    lenient_match: Optional[bool] = None

    if field == "purchase_date":
        t_norm = normalize_date(truth, dayfirst=True)
        # Handle both DD/MM/YYYY and MM/DD/YYYY conventions for ambiguous dates
        p_norm_d1 = normalize_date(pred, dayfirst=True)
        p_norm_d0 = normalize_date(pred, dayfirst=False)
        norm_match = (
            t_norm is not None
            and (p_norm_d1 == t_norm or p_norm_d0 == t_norm)
        )

    elif field == "total_amount":
        p_val = normalize_amount(pred)
        t_val = normalize_amount(truth)
        norm_match = (
            p_val is not None and t_val is not None and abs(p_val - t_val) <= 0.5
        )

    elif field in ("invoice_number", "supplier_gst"):
        p_norm = normalize_text(pred)
        t_norm = normalize_text(truth)
        norm_match = p_norm is not None and t_norm is not None and p_norm == t_norm

        p_len = lenient_text(pred)
        t_len = lenient_text(truth)
        lenient_match = p_len is not None and t_len is not None and p_len == t_len

    return exact_match, norm_match, lenient_match


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    process = psutil.Process(os.getpid())

    # Read ground truth
    if not GROUND_TRUTH_CSV.exists():
        print(f"Error: {GROUND_TRUTH_CSV} not found!")
        sys.exit(1)

    df_gt = pd.read_csv(GROUND_TRUTH_CSV)
    gt_map = {row["filename"]: row for _, row in df_gt.iterrows()}

    # Initialize EasyOCR engine (CPU, low-memory mode)
    engine = EasyOCREngine(languages=["en"], gpu=False)

    image_files = sorted(RAW_IMAGES_DIR.glob("*.jpg"))
    total_images = len(image_files)
    print(f"Starting baseline evaluation on {total_images} images (using OCR cache)...")

    results_rows: List[Dict[str, Any]] = []
    ocr_timings: List[float] = []
    peak_rss_mb: float = process.memory_info().rss / (1024 * 1024)

    for idx, img_path in enumerate(image_files, 1):
        filename = img_path.name
        gt_row = gt_map.get(filename)
        if gt_row is None:
            continue

        # Run or load cached OCR
        t0 = time.perf_counter()
        ocr_items = engine.recognize_image(img_path)
        t1 = time.perf_counter()
        duration = t1 - t0
        ocr_timings.append(duration)

        # Update memory peak
        current_rss = process.memory_info().rss / (1024 * 1024)
        if current_rss > peak_rss_mb:
            peak_rss_mb = current_rss

        # Extract fields via baseline rules
        preds = extract_baseline_fields(ocr_items)

        row_record: Dict[str, Any] = {
            "filename": filename,
            "ocr_time_seconds": round(duration, 3),
            "total_failures": 0,
        }

        # Evaluate each field
        for field in EVAL_FIELDS:
            truth_val = gt_row.get(field)
            pred_val = preds.get(field)
            has_gt = is_non_empty(truth_val)

            row_record[f"truth_{field}"] = truth_val if has_gt else ""
            row_record[f"pred_{field}"] = pred_val or ""

            if has_gt:
                exact, norm, lenient = evaluate_field(field, pred_val, str(truth_val))
                row_record[f"exact_{field}"] = exact
                row_record[f"norm_{field}"] = norm
                if lenient is not None:
                    row_record[f"lenient_{field}"] = lenient
                if not norm:
                    row_record["total_failures"] += 1
            else:
                row_record[f"exact_{field}"] = "N/A"
                row_record[f"norm_{field}"] = "N/A"
                if field in ("invoice_number", "supplier_gst"):
                    row_record[f"lenient_{field}"] = "N/A"

        results_rows.append(row_record)
        gc.collect()

    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(PER_IMAGE_CSV, index=False, encoding="utf-8")
    print(f"Saved per-image evaluation table to: {PER_IMAGE_CSV}")

    # Compute Summary Statistics
    summary_stats: Dict[str, Dict[str, Any]] = {}
    for field in EVAL_FIELDS:
        scored_mask = df_results[f"exact_{field}"] != "N/A"
        n_gt = scored_mask.sum()
        if n_gt > 0:
            exact_count = (df_results.loc[scored_mask, f"exact_{field}"] == True).sum()
            norm_count = (df_results.loc[scored_mask, f"norm_{field}"] == True).sum()
            stat: Dict[str, Any] = {
                "ground_truth_count": int(n_gt),
                "exact_count": int(exact_count),
                "exact_acc": round(exact_count / n_gt * 100, 1),
                "norm_count": int(norm_count),
                "norm_acc": round(norm_count / n_gt * 100, 1),
            }
            if field in ("invoice_number", "supplier_gst"):
                lenient_count = (df_results.loc[scored_mask, f"lenient_{field}"] == True).sum()
                stat["lenient_count"] = int(lenient_count)
                stat["lenient_acc"] = round(lenient_count / n_gt * 100, 1)
            summary_stats[field] = stat

    # Compute Performance Metrics
    avg_ocr_time = sum(ocr_timings) / len(ocr_timings) if ocr_timings else 0.0
    total_ocr_time = sum(ocr_timings)

    # Format Markdown Summary
    md_lines: List[str] = [
        "# Baseline Extraction & Evaluation Summary (Corrected Scorer)",
        "",
        "## Performance & Resource Utilization",
        f"- **Hardware**: CPU Only (8 GB System RAM constraint)",
        f"- **Engine**: EasyOCR (English)",
        f"- **Total Invoices Evaluated**: {len(results_rows)}",
        f"- **Total OCR Processing Time (cached load)**: {total_ocr_time:.2f} seconds",
        f"- **Peak Process Memory (RSS)**: {peak_rss_mb:.2f} MB",
        "",
        "## Extraction Accuracy Metrics",
        "",
        "| Field | Non-Empty GT | Exact Match Acc | Normalized Match Acc | Lenient Match Acc (O/0, I/1, S/5) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for field, s in summary_stats.items():
        lenient_str = f"{s['lenient_acc']}% ({s['lenient_count']}/{s['ground_truth_count']})" if "lenient_acc" in s else "N/A"
        md_lines.append(
            f"| `{field}` | {s['ground_truth_count']} | {s['exact_acc']}% ({s['exact_count']}/{s['ground_truth_count']}) | {s['norm_acc']}% ({s['norm_count']}/{s['ground_truth_count']}) | {lenient_str} |"
        )

    # Identify 10 worst failure cases
    df_failures = df_results.sort_values(by="total_failures", ascending=False).head(10)

    md_lines.extend([
        "",
        "## Top 10 Worst Failures (Sorted by Failed Normalized Fields)",
        "",
        "| Filename | Failed Fields Count | Field | Ground Truth | Predicted |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for _, row in df_failures.iterrows():
        fn = row["filename"]
        fail_cnt = row["total_failures"]
        for field in EVAL_FIELDS:
            if row[f"norm_{field}"] is False:
                gt_val = row[f"truth_{field}"]
                pred_val = row[f"pred_{field}"] or "[EMPTY]"
                md_lines.append(f"| `{fn}` | {fail_cnt} | `{field}` | `{gt_val}` | `{pred_val}` |")

    summary_text = "\n".join(md_lines)
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Saved evaluation summary report to: {SUMMARY_MD}")

    # Print Summary Table to Console
    print("\n" + "=" * 80)
    print("CORRECTED BASELINE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Field':<18} | {'GT Count':<8} | {'Exact Match':<16} | {'Normalized Match':<18} | {'Lenient Match'}")
    print("-" * 80)
    for field, s in summary_stats.items():
        exact_str = f"{s['exact_acc']}% ({s['exact_count']}/{s['ground_truth_count']})"
        norm_str = f"{s['norm_acc']}% ({s['norm_count']}/{s['ground_truth_count']})"
        lenient_str = f"{s['lenient_acc']}% ({s['lenient_count']}/{s['ground_truth_count']})" if "lenient_acc" in s else "N/A"
        print(f"{field:<18} | {s['ground_truth_count']:<8} | {exact_str:<16} | {norm_str:<18} | {lenient_str}")


if __name__ == "__main__":
    main()
