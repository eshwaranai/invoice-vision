"""Calibration check: Compare clean_original vs degraded_raw across severities.

Checks:
  - Word count per image (clean vs degraded)
  - Word count ratio (must be >= 50% for medium severity)
  - Upper-bound recall per field
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import cv2
import pandas as pd

# Add workspace root and src/
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from invoice_vision.ocr.easyocr_engine import OCRItem
from scripts.diagnose_baseline import check_upper_bound_amount, check_upper_bound_string
from scripts.evaluate_v1 import is_non_empty

MANIFEST_PATH = Path("data/processed/synth_photos/manifest.json")
CLEAN_CACHE_DIR = Path("data/processed/ocr_cache")
CALIB_CACHE_DIR = Path("data/processed/synth_photos/calib_cache")
MODEL_STORAGE_DIR = Path("data/models/easyocr")
USER_NETWORK_DIR = MODEL_STORAGE_DIR / "user_network"

FIELDS = ["invoice_number", "purchase_date", "supplier_gst", "total_amount"]


def get_ocr_reader() -> Any:
    import easyocr
    USER_NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(MODEL_STORAGE_DIR),
        user_network_directory=str(USER_NETWORK_DIR),
        download_enabled=False,
    )


def run_or_load_ocr(img_path: Path, cache_path: Path, reader: Any) -> List[OCRItem]:
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [OCRItem.from_dict(d) for d in data]

    img = cv2.imread(str(img_path))
    raw_res = reader.readtext(img, canvas_size=1600, mag_ratio=1.0)
    items = []
    for bbox, text, prob in raw_res:
        t_clean = str(text).strip()
        if t_clean:
            items.append(OCRItem(box=[[float(p[0]), float(p[1])] for p in bbox], text=t_clean, confidence=float(prob)))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([it.to_dict() for it in items], f, indent=2)

    return items


def main() -> None:
    print("=" * 80)
    print("CALIBRATION CHECK: CLEAN ORIGINAL VS DEGRADED RAW (v2)")
    print("=" * 80)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records: List[Dict[str, Any]] = manifest["records"]
    reader = get_ocr_reader()

    # Get distinct source images (12 images)
    source_images = sorted(list(set(r["source_image"] for r in records)))

    # 1. Clean Original baseline stats
    clean_stats = {}
    for src in source_images:
        cache_file = CLEAN_CACHE_DIR / f"{src}.json"
        with open(cache_file, "r", encoding="utf-8") as f:
            items = [OCRItem.from_dict(d) for d in json.load(f)]
        # get ground truth from manifest
        rec = next(r for r in records if r["source_image"] == src)
        clean_stats[src] = {
            "items": items,
            "word_count": len(items),
            "tokens": [it.text for it in items],
            "gt": rec["ground_truth_invoice"],
        }

    total_clean_words = sum(s["word_count"] for s in clean_stats.values())
    avg_clean_words = total_clean_words / len(clean_stats)
    print(f"\n[Clean Original Baseline] (n=12 images)")
    print(f"  Total Words Detected: {total_clean_words} (Avg: {avg_clean_words:.1f} words/image)")

    # 2. Check each condition: (severity, background)
    conditions = [
        ("mild", "dark_wood"),
        ("mild", "light_surface"),
        ("medium", "dark_wood"),
        ("medium", "light_surface"),
        ("severe", "dark_wood"),
        ("severe", "light_surface"),
    ]

    results_table = []

    for sev, bg in conditions:
        cond_slug = f"{sev}_{bg}"
        cond_records = [r for r in records if r["severity"] == sev and r["background"] == bg]

        total_words = 0
        per_image_words = []
        upper_bound_hits = {f: 0 for f in FIELDS}
        gt_counts = {f: 0 for f in FIELDS}

        print(f"\nEvaluating Condition: {cond_slug} ...")
        for r in cond_records:
            fn = r["filename"]
            img_p = Path(r["relative_path"])
            cache_p = CALIB_CACHE_DIR / cond_slug / f"{Path(fn).stem}.json"

            t0 = time.perf_counter()
            items = run_or_load_ocr(img_p, cache_p, reader)
            t1 = time.perf_counter()

            wc = len(items)
            total_words += wc
            per_image_words.append(wc)

            tokens = [it.text for it in items]
            gt = r["ground_truth_invoice"]

            for f in FIELDS:
                truth = gt.get(f)
                if not is_non_empty(truth):
                    continue
                gt_counts[f] += 1
                t_str = str(truth).strip()
                if f == "total_amount":
                    _, _, fuz = check_upper_bound_amount(t_str, tokens)
                else:
                    _, _, fuz = check_upper_bound_string(t_str, tokens)
                if fuz:
                    upper_bound_hits[f] += 1

            print(f"  {fn:<32} -> {wc:>3} words ({t1-t0:.2f}s)")

        avg_words = total_words / len(cond_records)
        ratio_pct = (avg_words / avg_clean_words) * 100.0

        results_table.append({
            "condition": cond_slug,
            "severity": sev,
            "background": bg,
            "total_words": total_words,
            "avg_words": round(avg_words, 1),
            "ratio_pct": round(ratio_pct, 1),
            "upper_bound_inv": f"{upper_bound_hits['invoice_number']}/{gt_counts['invoice_number']}",
            "upper_bound_date": f"{upper_bound_hits['purchase_date']}/{gt_counts['purchase_date']}",
            "upper_bound_gst": f"{upper_bound_hits['supplier_gst']}/{gt_counts['supplier_gst']}",
            "upper_bound_total": f"{upper_bound_hits['total_amount']}/{gt_counts['total_amount']}",
        })

    # Summary Display
    print("\n" + "=" * 95)
    print("CALIBRATION RESULTS SUMMARY (n=12 images per condition)")
    print("=" * 95)
    print(f"{'Condition':<24} | {'Avg Words':<10} | {'Clean Ratio':<12} | {'Invoice No':<12} | {'Date':<10} | {'GSTIN':<10} | {'Total'}")
    print("-" * 95)
    for row in results_table:
        print(f"{row['condition']:<24} | {row['avg_words']:<10} | {row['ratio_pct']:>5.1f}%      | {row['upper_bound_inv']:<12} | {row['upper_bound_date']:<10} | {row['upper_bound_gst']:<10} | {row['upper_bound_total']}")
    print("-" * 95)

    # Check 50% criteria for medium severity
    med_ratios = [r["ratio_pct"] for r in results_table if r["severity"] == "medium"]
    avg_med_ratio = sum(med_ratios) / len(med_ratios)
    print(f"\nAverage Medium Degraded-Raw vs Clean Ratio: {avg_med_ratio:.1f}%")
    if avg_med_ratio >= 50.0:
        print(">> CALIBRATION PASSED: Medium degraded word count is >= 50% of clean_original! Ready for full ablation.")
    else:
        print(f">> CALIBRATION WARNING: Medium degraded word count ({avg_med_ratio:.1f}%) is under 50%. Adjustments required.")


if __name__ == "__main__":
    main()
