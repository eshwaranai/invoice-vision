"""Diagnostic analysis script for baseline OCR and rule-based extraction pipeline.

Performs:
  1. Upper-Bound Recall per field (Exact normalized substring, Lenient, Fuzzy/Numeric/Multi-format).
  2. Scorer sanity-check on total_amount and purchase_date.
  3. Ground-truth OCR token inspection with bounding boxes for 6 key failure cases.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser as date_parser

# Path constants
OCR_CACHE_DIR = Path("data/processed/ocr_cache")
GROUND_TRUTH_CSV = Path("data/ground_truth_gst.csv")
PER_IMAGE_CSV = Path("results/baseline_per_image.csv")


def is_non_empty(val: Any) -> bool:
    """Check if value is present and non-empty."""
    if val is None or pd.isna(val):
        return False
    s = str(val).strip()
    return s != "" and s.lower() != "nan"


def normalize_str(s: Any) -> str:
    """Uppercase and strip punctuation/whitespace."""
    return re.sub(r"[\s\-_:/\,\.]", "", str(s or "")).upper()


LENIENT_MAP = str.maketrans({
    "O": "0", "0": "0",
    "I": "1", "1": "1", "L": "1",
    "S": "5", "5": "5",
    "Z": "7", "7": "7",
    "B": "8", "8": "8",
})


def to_lenient(s: Any) -> str:
    """Map ambiguous OCR characters to canonical forms."""
    return normalize_str(s).translate(LENIENT_MAP)


def levenshtein_dist(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
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


# ------------------------------------------------------------------------------
# Upper-Bound Recall Checkers
# ------------------------------------------------------------------------------

def check_upper_bound_string(
    gt_val: str, ocr_tokens: List[str]
) -> Tuple[bool, bool, bool]:
    """Check string field (invoice_number, supplier_gst) at Exact, Lenient, Fuzzy levels."""
    gt_norm = normalize_str(gt_val)
    gt_len = to_lenient(gt_val)
    full_text_norm = normalize_str(" ".join(ocr_tokens))
    full_text_len = to_lenient(" ".join(ocr_tokens))

    # Level 1: Exact normalized substring
    exact_found = gt_norm in full_text_norm or any(gt_norm in normalize_str(t) for t in ocr_tokens)

    # Level 2: Lenient substring
    lenient_found = exact_found or (gt_len in full_text_len) or any(gt_len in to_lenient(t) for t in ocr_tokens)

    # Level 3: Fuzzy match (edit distance <= 2 against any token or sliding window)
    fuzzy_found = lenient_found
    if not fuzzy_found and len(gt_norm) >= 3:
        target_len = len(gt_norm)
        for token in ocr_tokens:
            t_norm = normalize_str(token)
            if abs(len(t_norm) - target_len) <= 2:
                if levenshtein_dist(gt_norm, t_norm) <= 2:
                    fuzzy_found = True
                    break
        # Also check sliding window on normalized text if token boundaries split it
        if not fuzzy_found and len(full_text_norm) >= target_len:
            for i in range(max(0, len(full_text_norm) - target_len - 2)):
                window = full_text_norm[i : i + target_len]
                if levenshtein_dist(gt_norm, window) <= 2:
                    fuzzy_found = True
                    break

    return exact_found, lenient_found, fuzzy_found


def check_upper_bound_amount(
    gt_val: str, ocr_tokens: List[str]
) -> Tuple[bool, bool, bool]:
    """Check total_amount at Exact, Lenient, and Numeric levels."""
    clean_gt_str = re.sub(r"[^\d\.]", "", str(gt_val).replace(",", "").strip())
    try:
        gt_num = float(clean_gt_str)
    except Exception:
        return False, False, False

    int_gt_part = clean_gt_str.split(".")[0]
    full_text = " ".join(ocr_tokens)
    full_text_norm = normalize_str(full_text)
    full_text_len = to_lenient(full_text)

    # Level 1: Exact substring of the main numeric token
    exact_found = int_gt_part in full_text_norm or any(int_gt_part in normalize_str(t) for t in ocr_tokens)

    # Level 2: Lenient substring (treating O/0, I/1, S/5, Z/7, B/8 as equal)
    gt_len = to_lenient(int_gt_part)
    lenient_found = exact_found or (gt_len in full_text_len) or any(gt_len in to_lenient(t) for t in ocr_tokens)

    # Level 3: Numeric float match within tolerance 0.5 across all extracted numeric tokens
    numeric_found = False
    # Find all numeric patterns in all tokens
    number_pattern = re.compile(r"\b\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?\b|\b\d+(?:\.\d{1,2})?\b")
    for token in ocr_tokens:
        matches = number_pattern.findall(token)
        for m in matches:
            clean_m = m.replace(",", "")
            try:
                cand_float = float(clean_m)
                if abs(cand_float - gt_num) <= 0.5:
                    numeric_found = True
                    break
            except Exception:
                continue
        if numeric_found:
            break

    return exact_found, lenient_found, (lenient_found or numeric_found)


def check_upper_bound_date(
    gt_val: str, ocr_tokens: List[str]
) -> Tuple[bool, bool, bool]:
    """Check purchase_date at Exact, Lenient, and Multi-format parsed levels."""
    try:
        gt_iso = date_parser.parse(str(gt_val).strip(), dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return False, False, False

    full_text = " ".join(ocr_tokens)
    full_text_len = to_lenient(full_text)
    raw_gt_norm = normalize_str(gt_val)
    raw_gt_len = to_lenient(gt_val)

    # Level 1: Exact substring
    exact_found = raw_gt_norm in normalize_str(full_text) or any(raw_gt_norm in normalize_str(t) for t in ocr_tokens)

    # Level 2: Lenient substring
    lenient_found = exact_found or (raw_gt_len in full_text_len) or any(raw_gt_len in to_lenient(t) for t in ocr_tokens)

    # Level 3: Date match after parsing multiple candidate date patterns
    date_matched = False
    date_pattern = re.compile(r"\b\d{1,4}[\/\-\.\s](?:\d{1,2}|[A-Za-z]{3,9})[\/\-\.\s]\d{2,4}\b")
    for token in ocr_tokens:
        matches = date_pattern.findall(token)
        for m in matches:
            for df_flag in (True, False):
                try:
                    cand_iso = date_parser.parse(m.strip(), dayfirst=df_flag).strftime("%Y-%m-%d")
                    if cand_iso == gt_iso:
                        date_matched = True
                        break
                except Exception:
                    continue
            if date_matched:
                break
        if date_matched:
            break

    return exact_found, lenient_found, (lenient_found or date_matched)


def run_upper_bound_recall() -> Dict[str, Dict[str, Any]]:
    """Compute upper-bound upper-bound recall per field."""
    df_gt = pd.read_csv(GROUND_TRUTH_CSV)
    fields = ["invoice_number", "purchase_date", "supplier_gst", "total_amount"]
    stats: Dict[str, Dict[str, Any]] = {
        f: {"total_gt": 0, "exact": 0, "lenient": 0, "fuzzy": 0} for f in fields
    }

    for _, row in df_gt.iterrows():
        fn = row["filename"]
        cache_path = OCR_CACHE_DIR / f"{fn}.json"
        if not cache_path.exists():
            continue
        with open(cache_path, "r", encoding="utf-8") as f:
            ocr_items = json.load(f)
        ocr_tokens = [item["text"].strip() for item in ocr_items if item["text"].strip()]

        for f in fields:
            gt_val = row.get(f)
            if not is_non_empty(gt_val):
                continue
            stats[f]["total_gt"] += 1

            if f in ("invoice_number", "supplier_gst"):
                e, l, fz = check_upper_bound_string(str(gt_val), ocr_tokens)
            elif f == "total_amount":
                e, l, fz = check_upper_bound_amount(str(gt_val), ocr_tokens)
            elif f == "purchase_date":
                e, l, fz = check_upper_bound_date(str(gt_val), ocr_tokens)

            if e:
                stats[f]["exact"] += 1
            if l:
                stats[f]["lenient"] += 1
            if fz:
                stats[f]["fuzzy"] += 1

    return stats


def sanity_check_scorer() -> None:
    """Print 10 examples each for total_amount and purchase_date to verify scorer."""
    df_res = pd.read_csv(PER_IMAGE_CSV)

    print("\n" + "=" * 90)
    print("SCORER SANITY CHECK: purchase_date (10 Examples)")
    print("=" * 90)
    print(f"{'Filename':<12} | {'Raw GT':<12} | {'Norm GT (d1)':<12} | {'Raw Pred':<14} | {'Norm Pred':<12} | {'Match':<6} | Diagnostic Notes")
    print("-" * 90)

    count = 0
    for _, r in df_res.iterrows():
        gt_raw = r.get("truth_purchase_date")
        if not is_non_empty(gt_raw):
            continue
        pred_raw = r.get("pred_purchase_date")
        t_d1 = None
        try:
            t_d1 = date_parser.parse(str(gt_raw).strip(), dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            pass

        p_norm = None
        p_d0 = None
        if is_non_empty(pred_raw):
            try:
                p_norm = date_parser.parse(str(pred_raw).strip(), dayfirst=True).strftime("%Y-%m-%d")
                p_d0 = date_parser.parse(str(pred_raw).strip(), dayfirst=False).strftime("%Y-%m-%d")
            except Exception:
                pass

        match = (p_norm == t_d1)
        notes = "OK"
        if not match and p_d0 == t_d1:
            notes = "SCORER BUG: Format is MM/DD/YYYY, matches under dayfirst=False!"
        elif not is_non_empty(pred_raw):
            notes = "Extractor returned EMPTY"
        elif not match:
            notes = "Different date detected"

        print(f"{r['filename']:<12} | {str(gt_raw):<12} | {str(t_d1):<12} | {str(pred_raw):<14} | {str(p_norm):<12} | {str(match):<6} | {notes}")
        count += 1
        if count >= 10:
            break

    print("\n" + "=" * 90)
    print("SCORER SANITY CHECK: total_amount (10 Examples)")
    print("=" * 90)
    print(f"{'Filename':<12} | {'Raw GT':<14} | {'Norm GT Float':<14} | {'Raw Pred':<12} | {'Norm Pred':<12} | {'Match':<6} | Diagnostic Notes")
    print("-" * 90)

    count = 0
    for _, r in df_res.iterrows():
        gt_raw = r.get("truth_total_amount")
        if not is_non_empty(gt_raw):
            continue
        pred_raw = r.get("pred_total_amount")

        t_float = None
        try:
            t_float = float(re.sub(r"[^\d\.]", "", str(gt_raw).replace(",", "").strip()))
        except Exception:
            pass

        p_float = None
        if is_non_empty(pred_raw):
            try:
                p_float = float(re.sub(r"[^\d\.]", "", str(pred_raw).replace(",", "").strip()))
            except Exception:
                pass

        match = (t_float is not None and p_float is not None and abs(t_float - p_float) <= 0.5)
        notes = "Extractor returned EMPTY (1D regex didn't span across columns)" if not is_non_empty(pred_raw) else ("MATCH" if match else "Value mismatch")

        print(f"{r['filename']:<12} | {str(gt_raw):<14} | {str(t_float):<14} | {str(pred_raw):<12} | {str(p_float):<12} | {str(match):<6} | {notes}")
        count += 1
        if count >= 10:
            break


def print_token_evidence() -> None:
    """Print OCR token evidence and bounding boxes for 6 key failures."""
    cases = [
        # (field, filename, truth_val, search_keywords, description)
        ("total_amount", "gst_000.jpg", "56,640", ["56,", "640", "total"], "Case 1: Total Amount"),
        ("total_amount", "gst_004.jpg", "12,382.92", ["12.382", "charges", "total"], "Case 2: Total Amount"),
        ("supplier_gst", "gst_004.jpg", "O6AAACMSS86C1ZL", ["oeaaac", "osaag", "gstin"], "Case 3: Supplier GST"),
        ("supplier_gst", "gst_021.jpg", "O7ADLFSS128F1ZY", ["ozadl", "o7adl", "gstin"], "Case 4: Supplier GST"),
        ("invoice_number", "gst_000.jpg", "S325", ["5325", "ilvoice", "s325"], "Case 5: Invoice Number"),
        ("invoice_number", "gst_021.jpg", "SIMEQ-035-21-22", ["simeq 035", "irvcice"], "Case 6: Invoice Number"),
    ]

    print("\n" + "=" * 90)
    print("DETAILED OCR TOKEN EVIDENCE FOR 6 KEY FAILURES")
    print("=" * 90)

    for field, fn, truth, keywords, header in cases:
        cache_path = OCR_CACHE_DIR / f"{fn}.json"
        with open(cache_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        print(f"\n--- [{header}] File: {fn} | Field: {field} | Truth: '{truth}' ---")
        found_any = False
        for it in items:
            t = it["text"]
            if any(k.lower() in t.lower() for k in keywords):
                box = it["box"]
                conf = it["confidence"]
                print(f"   * Bounding Box: {box} | Conf: {conf:.2f} | Text: '{t}'")
                found_any = True
        if not found_any:
            print("   * No keyword-matching token found.")


def main() -> None:
    print("=" * 70)
    print("UPPER-BOUND RECALL ANALYSIS (UPPER BOUND CEILING)")
    print("=" * 70)
    upper_bound_stats = run_upper_bound_recall()

    print(f"{'Field':<18} | {'GT Count':<8} | {'Exact Norm Substr':<18} | {'Lenient Substr':<16} | {'Fuzzy/Numeric/Date'}")
    print("-" * 75)
    for f, s in upper_bound_stats.items():
        n = s["total_gt"]
        e_str = f"{s['exact']/n*100:5.1f}% ({s['exact']}/{n})"
        l_str = f"{s['lenient']/n*100:5.1f}% ({s['lenient']}/{n})"
        f_str = f"{s['fuzzy']/n*100:5.1f}% ({s['fuzzy']}/{n})"
        print(f"{f:<18} | {n:<8} | {e_str:<18} | {l_str:<16} | {f_str}")

    sanity_check_scorer()
    print_token_evidence()


if __name__ == "__main__":
    main()
