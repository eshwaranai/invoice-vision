"""Diagnose Control and In-Page Word Analysis.

Implements User Requirements 1 & 2:
1. Report upper-bound recall per field (invoice_number, purchase_date, supplier_gst, total_amount)
   for:
     - clean_original (1.0x native)
     - control_upscaled_2.5x (2.5x bicubic clean, isolated resolution effect)
     - degraded_raw across 4 severities (mild, medium, severe, hard) and 2 backgrounds (dark_wood, light_surface).
   Compares degraded vs upscaled control.
2. Explains why degraded word counts exceed clean:
     - Share of words with confidence < 0.30.
     - Share of words located outside the true page quad.
     - Counts only words inside the true page quad for upper-bound recall comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from invoice_vision.ocr.easyocr_engine import OCRItem
from scripts.diagnose_baseline import check_upper_bound_amount, check_upper_bound_string
from scripts.evaluate_v1 import is_non_empty

RAW_DIR = Path("data/raw/indian_gst")
MANIFEST_PATH = Path("data/processed/synth_photos/manifest.json")
CLEAN_CACHE_DIR = Path("data/processed/ocr_cache")
CONTROL_DIR = Path("data/processed/clean_upscaled")
CONTROL_CACHE_DIR = Path("data/processed/ocr_cache_control")
CALIB_CACHE_DIR = Path("data/processed/synth_photos/calib_cache")
MODEL_STORAGE_DIR = Path("data/models/easyocr")
USER_NETWORK_DIR = MODEL_STORAGE_DIR / "user_network"

FIELDS = ["invoice_number", "purchase_date", "supplier_gst", "total_amount"]

TARGET_IMAGES = [
    ("gst_000.jpg", "Senate Space"),
    ("gst_001.jpg", "Linkedin Singapore Pte. Ltd"),
    ("gst_004.jpg", "Microsoft / Microsoft India"),
    ("gst_005.jpg", "Microsoft / Microsoft India"),
    ("gst_012.jpg", "Net2Source LLP"),
    ("gst_014.jpg", "Bansal Electrovision Pvt. Ltd"),
    ("gst_015.jpg", "Sai Computer Solution"),
    ("gst_020.jpg", "SBI General Insurance Company Limited"),
    ("gst_021.jpg", "SIMEQ Technologies LLP"),
    ("gst_024.jpg", "Amazon - Retailez Private Limited"),
    ("gst_026.jpg", "Amazon - Whole9Yards Online LLP"),
    ("gst_028.jpg", "Amazon - Altaf H Vali Electronics Pvt Ltd"),
]


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


def run_or_load_ocr(img: np.ndarray, cache_p: Path, reader: Any) -> List[OCRItem]:
    if cache_p.exists():
        with open(cache_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [OCRItem.from_dict(d) for d in data]

    cache_p.parent.mkdir(parents=True, exist_ok=True)
    raw_res = reader.readtext(img, canvas_size=1600, mag_ratio=1.0)
    items = []
    for bbox, text, prob in raw_res:
        t_clean = str(text).strip()
        if t_clean:
            items.append(
                OCRItem(
                    box=[[float(p[0]), float(p[1])] for p in bbox],
                    text=t_clean,
                    confidence=float(prob),
                )
            )

    with open(cache_p, "w", encoding="utf-8") as f:
        json.dump([it.to_dict() for it in items], f, indent=2)

    return items


def is_box_inside_quad(box: List[List[float]], quad_pts: np.ndarray) -> bool:
    """Check if center of bounding box is inside quadrilateral."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    # cv2.pointPolygonTest returns positive if inside, 0 on edge, negative outside
    dist = cv2.pointPolygonTest(quad_pts.astype(np.float32), (cx, cy), False)
    return dist >= 0.0


def analyze_ocr_items(
    ocr_items: List[OCRItem],
    gt_quad: np.ndarray,
    gt_invoice: Dict[str, Any],
) -> Dict[str, Any]:
    total_words = len(ocr_items)
    low_conf_count = sum(1 for it in ocr_items if it.confidence < 0.30)
    low_conf_share = (low_conf_count / total_words) if total_words > 0 else 0.0

    in_page_items = [it for it in ocr_items if is_box_inside_quad(it.box, gt_quad)]
    in_page_words = len(in_page_items)
    outside_quad_count = total_words - in_page_words
    outside_quad_share = (outside_quad_count / total_words) if total_words > 0 else 0.0

    # In-page tokens ONLY for upper-bound recall
    in_page_tokens = [it.text for it in in_page_items]
    all_tokens = [it.text for it in ocr_items]

    hits_in_page = {}
    hits_all = {}
    for f in FIELDS:
        val = gt_invoice.get(f)
        if not is_non_empty(val):
            hits_in_page[f] = None
            hits_all[f] = None
            continue
        v_str = str(val).strip()
        if f == "total_amount":
            _, _, hit_in = check_upper_bound_amount(v_str, in_page_tokens)
            _, _, hit_all = check_upper_bound_amount(v_str, all_tokens)
        else:
            _, _, hit_in = check_upper_bound_string(v_str, in_page_tokens)
            _, _, hit_all = check_upper_bound_string(v_str, all_tokens)
        hits_in_page[f] = hit_in
        hits_all[f] = hit_all

    return {
        "total_words": total_words,
        "in_page_words": in_page_words,
        "outside_quad_words": outside_quad_count,
        "outside_quad_share": round(outside_quad_share * 100.0, 1),
        "low_conf_words": low_conf_count,
        "low_conf_share": round(low_conf_share * 100.0, 1),
        "hits_in_page": hits_in_page,
        "hits_all": hits_all,
    }


def main() -> None:
    print("=" * 85)
    print("UPPER-BOUND RECALL & IN-PAGE WORD VALIDITY DIAGNOSTIC")
    print("=" * 85)

    reader = get_ocr_reader()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    records: List[Dict[str, Any]] = manifest["records"]

    # 1. Clean Original Baseline & 2.5x Bicubic Control
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    clean_results = []
    control_results = []

    print("\n[Step 1/3] Processing Clean Original & 2.5x Upscaled Control (n=12)...")
    for fn, vendor in TARGET_IMAGES:
        src_path = RAW_DIR / fn
        clean_img = cv2.imread(str(src_path))
        h, w = clean_img.shape[:2]

        rec = next(r for r in records if r["source_image"] == fn)
        gt_inv = rec["ground_truth_invoice"]

        # Clean Original (1.0x native)
        clean_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        clean_cache_p = CLEAN_CACHE_DIR / f"{fn}.json"
        clean_items = run_or_load_ocr(clean_img, clean_cache_p, reader)
        clean_metrics = analyze_ocr_items(clean_items, clean_quad, gt_inv)
        clean_results.append({
            "filename": fn,
            "condition": "clean_original",
            **clean_metrics,
            "gt_invoice": gt_inv,
        })

        # Control 2.5x Upscaled
        up_w = int(w * 2.5)
        up_h = int(h * 2.5)
        control_p = CONTROL_DIR / f"clean_2.5x_{fn}"
        if not control_p.exists():
            up_img = cv2.resize(clean_img, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(control_p), up_img)
        else:
            up_img = cv2.imread(str(control_p))

        control_quad = np.array([[0, 0], [up_w - 1, 0], [up_w - 1, up_h - 1], [0, up_h - 1]], dtype=np.float32)
        control_cache_p = CONTROL_CACHE_DIR / f"{Path(fn).stem}.json"
        control_items = run_or_load_ocr(up_img, control_cache_p, reader)
        control_metrics = analyze_ocr_items(control_items, control_quad, gt_inv)
        control_results.append({
            "filename": fn,
            "condition": "control_upscaled_2.5x",
            **control_metrics,
            "gt_invoice": gt_inv,
        })

        print(f"  {fn:<14} -> Clean:{clean_metrics['total_words']:>3} words | Control 2.5x:{control_metrics['total_words']:>3} words")

    # 2. Degraded Raw across Severities & Backgrounds
    print("\n[Step 2/3] Processing Degraded Raw Conditions (n=96 images across 4 severities x 2 backgrounds)...")
    degraded_results = []
    severities = ["mild", "medium", "severe", "hard"]
    backgrounds = ["dark_wood", "light_surface"]

    for sev in severities:
        for bg in backgrounds:
            cond_slug = f"{sev}_{bg}"
            cond_recs = [r for r in records if r["severity"] == sev and r["background"] == bg]

            for r in cond_recs:
                img_p = Path(r["relative_path"])
                fn = r["filename"]
                gt_quad = np.array(r["ground_truth_corners"], dtype=np.float32)
                gt_inv = r["ground_truth_invoice"]

                cache_p = CALIB_CACHE_DIR / cond_slug / f"{Path(fn).stem}.json"
                img = cv2.imread(str(img_p))
                items = run_or_load_ocr(img, cache_p, reader)

                metrics = analyze_ocr_items(items, gt_quad, gt_inv)
                degraded_results.append({
                    "filename": fn,
                    "source_image": r["source_image"],
                    "condition": cond_slug,
                    "severity": sev,
                    "background": bg,
                    **metrics,
                    "gt_invoice": gt_inv,
                })
            print(f"  Processed condition: {cond_slug:<22} ({len(cond_recs)} images)")

    # 3. Summaries & Tables
    print("\n" + "=" * 95)
    print("PART 1: WORD COUNT & IN-PAGE / LOW-CONFIDENCE BREAKDOWN (Explaining Inflation)")
    print("=" * 95)
    print(f"{'Condition':<25} | {'Total Words':<12} | {'In-Page Words':<14} | {'Outside Quad %':<16} | {'Conf < 0.30 %'}")
    print("-" * 95)

    all_cond_groups = [
        ("clean_original", clean_results),
        ("control_upscaled_2.5x", control_results),
    ]
    for sev in severities:
        for bg in backgrounds:
            cond_slug = f"{sev}_{bg}"
            sub = [r for r in degraded_results if r["condition"] == cond_slug]
            all_cond_groups.append((cond_slug, sub))

    breakdown_rows = []
    for cond_name, rows in all_cond_groups:
        tot_w = np.mean([r["total_words"] for r in rows])
        inp_w = np.mean([r["in_page_words"] for r in rows])
        out_pct = np.mean([r["outside_quad_share"] for r in rows])
        low_pct = np.mean([r["low_conf_share"] for r in rows])
        print(f"{cond_name:<25} | {tot_w:>8.1f}     | {inp_w:>9.1f}      | {out_pct:>10.1f}%      | {low_pct:>9.1f}%")
        breakdown_rows.append({
            "condition": cond_name,
            "mean_total_words": round(tot_w, 1),
            "mean_in_page_words": round(inp_w, 1),
            "mean_outside_quad_pct": round(out_pct, 1),
            "mean_low_conf_pct": round(low_pct, 1),
        })

    print("-" * 95)

    # 4. Upper-Bound Recall Table (In-Page ONLY vs All Words)
    print("\n" + "=" * 95)
    print("PART 2: UPPER-BOUND RECALL PER FIELD (In-Page Words Only)")
    print("=" * 95)
    print(f"{'Condition':<25} | {'Invoice No':<12} | {'Purchase Date':<14} | {'Supplier GST':<14} | {'Total Amount'}")
    print("-" * 95)

    recall_rows = []
    for cond_name, rows in all_cond_groups:
        field_stats = {}
        for f in FIELDS:
            valid = [r for r in rows if r["hits_in_page"][f] is not None]
            hits = sum(1 for r in valid if r["hits_in_page"][f] is True)
            tot = len(valid)
            pct = (hits / tot * 100.0) if tot > 0 else 0.0
            field_stats[f] = f"{hits}/{tot} ({pct:4.1f}%)"

        print(f"{cond_name:<25} | {field_stats['invoice_number']:<12} | {field_stats['purchase_date']:<14} | {field_stats['supplier_gst']:<14} | {field_stats['total_amount']}")
        recall_rows.append({
            "condition": cond_name,
            **field_stats,
        })
    print("-" * 95)

    # Save data to json
    out_json = Path("results/diagnose_control_inpage.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "breakdown": breakdown_rows,
            "recall": recall_rows,
            "clean_raw": clean_results,
            "control_raw": control_results,
            "degraded_raw": degraded_results,
        }, f, indent=2)
    print(f"\nSaved detailed diagnostic data to: {out_json}")


if __name__ == "__main__":
    main()
