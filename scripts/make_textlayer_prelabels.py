"""Generate YOLO Pre-Labels for Personal Bills from PDF Text Layer.

Reads data/processed/pdf_text/*.json:
  - vendor (0): Bounding box of top-of-page words forming the suggested vendor name.
  - invoice_no (1): Value next to Invoice No / Number / # / Bill No.
  - date (2): Value next to Invoice Date / Date / Dated (skipping Due / Period).
  - gstin (3): Every 15-character token matching the GSTIN pattern that passes the checksum.
  - total (5): Value next to Grand Total / Total Amount / Net Payable / Total (bottom-most strongest label).
  - qr (6): Detected using cv2.QRCodeDetector on the rendered PNG image.
  - Skips table (4) and stamp_signature (7) for manual annotation.

Boxes cover the VALUE only (excluding label) with a 2 px margin.
Writes YOLO format txt files to data/prelabels_own/<image_basename>.txt.
Writes data/prelabels_own/README.md marking labels as UNREVIEWED.
Reports coverage counts only (zero private values printed).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from invoice_vision.parse.spatial_rules import validate_gstin_checksum

PDF_TEXT_DIR = ROOT_DIR / "data" / "processed" / "pdf_text"
RAW_OWN_DIR = ROOT_DIR / "data" / "raw" / "own"
MANIFEST_PATH = ROOT_DIR / "data" / "own_manifest.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "prelabels_own"
CLASSES_PATH = ROOT_DIR / "data" / "classes.txt"

MARGIN_PX = 2.0


def box_to_yolo(
    min_x: float, min_y: float, max_x: float, max_y: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Apply 2 px margin and convert to normalized YOLO [xc, yc, w, h]."""
    x0 = max(0.0, min_x - MARGIN_PX)
    y0 = max(0.0, min_y - MARGIN_PX)
    x1 = min(float(img_w), max_x + MARGIN_PX)
    y1 = min(float(img_h), max_y + MARGIN_PX)

    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    xc = x0 + bw / 2.0
    yc = y0 + bh / 2.0

    return (xc / img_w, yc / img_h, bw / img_w, bh / img_h)


def write_prelabels_readme() -> None:
    """Write unreviewed disclaimer in data/prelabels_own/README.md."""
    readme_path = OUTPUT_DIR / "README.md"
    content = """# Automated Text-Layer Pre-Labels (Personal Bills)

> [!WARNING]
> **UNREVIEWED HEURISTIC LABELS - MANDATORY HUMAN VERIFICATION REQUIRED**:
> The label files in this directory (`*.txt`) were generated automatically from embedded PDF text layers
> using spatial keyword proximity rules and OpenCV QRCodeDetector.
> - **Classes Included**: `vendor` (0), `invoice_no` (1), `date` (2), `gstin` (3), `total` (5), and `qr` (6).
> - **Classes Excluded**: `table` (4) and `stamp_signature` (7) require manual polygon/box drawing.
> - **Disclaimer**: Alignment heuristics may misalign adjacent columns or pick secondary dates/totals.
> **DO NOT TRAIN MODELS DIRECTLY ON THESE PRE-LABELS** without completing a manual review per `docs/annotation_guideline.md`.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


def extract_vendor_box(
    words: List[Dict[str, Any]], vendor_name: str, h: int, w: int
) -> Optional[Tuple[float, float, float, float]]:
    """Locate words matching suggested vendor name in upper 40% of page."""
    if not vendor_name or vendor_name.lower() in ("unknown", "unknown vendor"):
        return None

    tokens = [re.sub(r"[^\w]", "", tok).lower() for tok in vendor_name.split() if len(re.sub(r"[^\w]", "", tok)) >= 3]
    if not tokens:
        return None

    top_words = [wd for wd in words if wd["bbox"][1] <= h * 0.40]
    matched_boxes = []
    for wd in top_words:
        wt = re.sub(r"[^\w]", "", wd["text"]).lower()
        if wt and any(tok in wt or wt in tok for tok in tokens):
            matched_boxes.append(wd["bbox"])

    if not matched_boxes:
        return None

    min_x = min(b[0] for b in matched_boxes)
    min_y = min(b[1] for b in matched_boxes)
    max_x = max(b[2] for b in matched_boxes)
    max_y = max(b[3] for b in matched_boxes)

    return box_to_yolo(min_x, min_y, max_x, max_y, w, h)


def extract_invoice_no_box(
    words: List[Dict[str, Any]], h: int, w: int
) -> Optional[Tuple[float, float, float, float]]:
    """Extract value next to invoice number labels."""
    inv_label_re = re.compile(
        r"\b(invoice\s*(?:no|number|#)?|bill\s*(?:no|number|#)?|inv\s*(?:no|#))\b", re.IGNORECASE
    )
    sorted_words = sorted(words, key=lambda it: (it["bbox"][1], it["bbox"][0]))

    for i, wd in enumerate(sorted_words):
        t = wd["text"].strip()
        comb = t
        lbl_box = wd["bbox"]
        if not inv_label_re.search(t) and i < len(sorted_words) - 1:
            comb = t + " " + sorted_words[i + 1]["text"]
            lbl_box = [wd["bbox"][0], wd["bbox"][1], sorted_words[i + 1]["bbox"][2], sorted_words[i + 1]["bbox"][3]]

        if inv_label_re.search(comb):
            # Exclude skip contexts like 'Reference Invoice Number:' for instructions
            for cand in sorted_words:
                if cand["bbox"] == wd["bbox"]:
                    continue
                # Search immediately to the right or directly below
                in_h_range = (lbl_box[1] - 8 <= cand["bbox"][1] <= lbl_box[3] + 35)
                to_right = (cand["bbox"][0] >= lbl_box[0] - 5)
                if in_h_range and to_right:
                    val_t = cand["text"].strip(":# ")
                    if len(val_t) >= 2 and not any(k in val_t.lower() for k in ["invoice", "date", "bill", "tax", "original", "number"]):
                        return box_to_yolo(cand["bbox"][0], cand["bbox"][1], cand["bbox"][2], cand["bbox"][3], w, h)

    return None


def extract_date_box(
    words: List[Dict[str, Any]], h: int, w: int
) -> Optional[Tuple[float, float, float, float]]:
    """Extract value next to primary invoice date labels."""
    date_label_re = re.compile(r"\b(invoice\s*date|dated|billing\s*date|date)\b", re.IGNORECASE)
    date_val_re = re.compile(
        r"(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}[\s,]+\d{2,4})",
        re.IGNORECASE,
    )
    sorted_words = sorted(words, key=lambda it: (it["bbox"][1], it["bbox"][0]))

    for i, wd in enumerate(sorted_words):
        t = wd["text"].strip()
        lbl_box = wd["bbox"]
        comb = t
        if not date_label_re.search(t) and i < len(sorted_words) - 1:
            comb = t + " " + sorted_words[i + 1]["text"]
            lbl_box = [wd["bbox"][0], wd["bbox"][1], sorted_words[i + 1]["bbox"][2], sorted_words[i + 1]["bbox"][3]]

        if date_label_re.search(comb):
            # Skip non-invoice dates
            comb_low = comb.lower()
            if any(skip in comb_low for skip in ["due", "period", "cycle", "supply", "delivery", "transaction", "order"]):
                continue

            for cand in sorted_words:
                in_range = (lbl_box[1] - 8 <= cand["bbox"][1] <= lbl_box[3] + 35) and (cand["bbox"][0] >= lbl_box[0] - 10)
                if in_range:
                    if date_val_re.search(cand["text"]):
                        return box_to_yolo(cand["bbox"][0], cand["bbox"][1], cand["bbox"][2], cand["bbox"][3], w, h)

    return None


def extract_gstin_boxes(
    words: List[Dict[str, Any]], h: int, w: int
) -> List[Tuple[float, float, float, float]]:
    """Extract all 15-character GSTIN tokens that pass checksum."""
    boxes: List[Tuple[float, float, float, float]] = []
    for wd in words:
        clean_t = re.sub(r"[^A-Za-z0-9]", "", wd["text"]).upper()
        if len(clean_t) == 15 and validate_gstin_checksum(clean_t):
            boxes.append(box_to_yolo(wd["bbox"][0], wd["bbox"][1], wd["bbox"][2], wd["bbox"][3], w, h))
    return boxes


def extract_total_box(
    words: List[Dict[str, Any]], h: int, w: int
) -> Optional[Tuple[float, float, float, float]]:
    """Extract value next to the bottom-most strongest total label."""
    num_re = re.compile(
        r"^(?:[₹`\$\s]|rs\.?)*(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)$", re.IGNORECASE
    )
    candidates = []

    for i, wd in enumerate(words):
        t = wd["text"].strip().lower()
        priority = None
        l_box = wd["bbox"]

        if "grand" in t and i < len(words) - 1 and "total" in words[i + 1]["text"].lower():
            priority = 10
            l_box = [wd["bbox"][0], wd["bbox"][1], words[i + 1]["bbox"][2], words[i + 1]["bbox"][3]]
        elif "total" in t and i < len(words) - 1 and "amount" in words[i + 1]["text"].lower():
            priority = 9
            l_box = [wd["bbox"][0], wd["bbox"][1], words[i + 1]["bbox"][2], words[i + 1]["bbox"][3]]
        elif "net" in t and i < len(words) - 1 and "payable" in words[i + 1]["text"].lower():
            priority = 8
            l_box = [wd["bbox"][0], wd["bbox"][1], words[i + 1]["bbox"][2], words[i + 1]["bbox"][3]]
        elif "amount" in t and i < len(words) - 1 and "payable" in words[i + 1]["text"].lower():
            priority = 8
            l_box = [wd["bbox"][0], wd["bbox"][1], words[i + 1]["bbox"][2], words[i + 1]["bbox"][3]]
        elif t in ("total", "total:"):
            priority = 5

        if priority is not None:
            # Search candidate values to the right or below
            for cand in words:
                if cand["bbox"] == wd["bbox"]:
                    continue
                if l_box[1] - 10 <= cand["bbox"][1] <= l_box[3] + 45 and cand["bbox"][0] >= l_box[0] - 10:
                    cand_text = cand["text"].strip()
                    m = num_re.search(cand_text)
                    if m:
                        try:
                            val_f = float(m.group(1).replace(",", ""))
                            candidates.append((priority, l_box[1], cand["bbox"]))
                        except Exception:
                            pass

    if not candidates:
        return None

    # Pick highest priority, then bottom-most (largest y)
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_box = candidates[0][2]
    return box_to_yolo(best_box[0], best_box[1], best_box[2], best_box[3], w, h)


def extract_qr_box(
    img_path: Path, h: int, w: int
) -> Optional[Tuple[float, float, float, float]]:
    """Detect QR code bounding box using OpenCV QRCodeDetector."""
    if not img_path.exists():
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    detector = cv2.QRCodeDetector()
    ret, pts = detector.detect(img)
    if ret and pts is not None and len(pts) > 0:
        pts = pts.reshape(-1, 2)
        min_x = float(np.min(pts[:, 0]))
        max_x = float(np.max(pts[:, 0]))
        min_y = float(np.min(pts[:, 1]))
        max_y = float(np.max(pts[:, 1]))
        return box_to_yolo(min_x, min_y, max_x, max_y, w, h)

    return None


def main() -> None:
    print("=" * 80)
    print("GENERATING TEXT-LAYER PRE-LABELS FOR OWN BILLS (YOLO Format)")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_prelabels_readme()

    manifest_df = pd.read_csv(MANIFEST_PATH).set_index("filename")
    json_files = sorted(list(PDF_TEXT_DIR.glob("*.json")), key=lambda p: p.name.lower())

    class_names = {
        0: "vendor",
        1: "invoice_no",
        2: "date",
        3: "gstin",
        5: "total",
        6: "qr",
    }

    coverage_per_class: Dict[int, int] = {c: 0 for c in class_names}
    family_coverage: Dict[str, Dict[str, int]] = {}

    total_pages = len(json_files)
    total_boxes = 0

    for jf in json_files:
        stem = jf.stem
        png_fn = f"{stem}.png"
        img_p = RAW_OWN_DIR / png_fn
        out_txt = OUTPUT_DIR / f"{stem}.txt"

        with open(jf, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        words = doc_data["words"]
        h = doc_data["height_px"]
        w = doc_data["width_px"]

        v_suggested = str(manifest_df.loc[png_fn, "vendor_suggested"]) if png_fn in manifest_df.index else ""
        fam_id = str(manifest_df.loc[png_fn, "template_family_suggested"]) if png_fn in manifest_df.index else "T00"

        if fam_id not in family_coverage:
            family_coverage[fam_id] = {c_name: 0 for c_name in class_names.values()}
            family_coverage[fam_id]["total_pages"] = 0

        family_coverage[fam_id]["total_pages"] += 1

        yolo_lines: List[str] = []
        page_classes_present: Set[int] = set()

        # 0: vendor
        v_box = extract_vendor_box(words, v_suggested, h, w)
        if v_box:
            yolo_lines.append(f"0 {v_box[0]:.6f} {v_box[1]:.6f} {v_box[2]:.6f} {v_box[3]:.6f}")
            page_classes_present.add(0)
            family_coverage[fam_id]["vendor"] += 1
            total_boxes += 1

        # 1: invoice_no
        inv_box = extract_invoice_no_box(words, h, w)
        if inv_box:
            yolo_lines.append(f"1 {inv_box[0]:.6f} {inv_box[1]:.6f} {inv_box[2]:.6f} {inv_box[3]:.6f}")
            page_classes_present.add(1)
            family_coverage[fam_id]["invoice_no"] += 1
            total_boxes += 1

        # 2: date
        dt_box = extract_date_box(words, h, w)
        if dt_box:
            yolo_lines.append(f"2 {dt_box[0]:.6f} {dt_box[1]:.6f} {dt_box[2]:.6f} {dt_box[3]:.6f}")
            page_classes_present.add(2)
            family_coverage[fam_id]["date"] += 1
            total_boxes += 1

        # 3: gstin
        gst_boxes = extract_gstin_boxes(words, h, w)
        for gb in gst_boxes:
            yolo_lines.append(f"3 {gb[0]:.6f} {gb[1]:.6f} {gb[2]:.6f} {gb[3]:.6f}")
            page_classes_present.add(3)
            total_boxes += 1
        if gst_boxes:
            family_coverage[fam_id]["gstin"] += 1

        # 5: total
        tot_box = extract_total_box(words, h, w)
        if tot_box:
            yolo_lines.append(f"5 {tot_box[0]:.6f} {tot_box[1]:.6f} {tot_box[2]:.6f} {tot_box[3]:.6f}")
            page_classes_present.add(5)
            family_coverage[fam_id]["total"] += 1
            total_boxes += 1

        # 6: qr
        qr_box = extract_qr_box(img_p, h, w)
        if qr_box:
            yolo_lines.append(f"6 {qr_box[0]:.6f} {qr_box[1]:.6f} {qr_box[2]:.6f} {qr_box[3]:.6f}")
            page_classes_present.add(6)
            family_coverage[fam_id]["qr"] += 1
            total_boxes += 1

        for c in page_classes_present:
            coverage_per_class[c] += 1

        with open(out_txt, "w", encoding="utf-8") as out_f:
            for l in yolo_lines:
                out_f.write(f"{l}\n")

    print(f"Generated {total_boxes} pre-label boxes across {total_pages} pages in {OUTPUT_DIR}.")

    print("\n" + "=" * 80)
    print("PRE-LABEL COVERAGE PER CLASS (Page Counts Only)")
    print("=" * 80)
    for cid, cname in class_names.items():
        cnt = coverage_per_class[cid]
        pct = (cnt / total_pages) * 100.0
        print(f"  * Class {cid} ({cname:<12}): {cnt:2d} / {total_pages} pages ({pct:5.1f}%)")

    print("\n" + "=" * 105)
    print("PRE-LABEL COVERAGE PER TEMPLATE FAMILY (Pages with Class Present)")
    print("=" * 105)
    f_hdr = "{:<6} {:<6} {:<10} {:<12} {:<10} {:<10} {:<10} {:<6}"
    print(f_hdr.format("Family", "Pages", "0:vendor", "1:inv_no", "2:date", "3:gstin", "5:total", "6:qr"))
    print("-" * 105)
    for fam_id in sorted(family_coverage.keys()):
        fc = family_coverage[fam_id]
        print(f_hdr.format(
            fam_id,
            fc["total_pages"],
            f"{fc['vendor']}/{fc['total_pages']}",
            f"{fc['invoice_no']}/{fc['total_pages']}",
            f"{fc['date']}/{fc['total_pages']}",
            f"{fc['gstin']}/{fc['total_pages']}",
            f"{fc['total']}/{fc['total_pages']}",
            f"{fc['qr']}/{fc['total_pages']}",
        ))
    print("-" * 105)


if __name__ == "__main__":
    main()
