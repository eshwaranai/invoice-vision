"""Automated Pre-Label Suggester for Indian Invoices and Bills.

Generates high-precision bounding box proposals from either:
  1. Extracted PDF text layers (data/processed/pdf_text/*.json)
  2. Cached EasyOCR word boxes (data/processed/ocr_cache/*.json)

Implements class-specific spatial and lexical heuristics:
  - vendor (0): Bounding box of the top-of-page company block (logo area plus company
    name and address lines, stopping at the first 'Bill to / Invoice / Date / GSTIN' row).
  - invoice_no (1): Values adjacent to labels containing 'invoice', 'bill no', or 'inv no'.
    Excludes non-invoice labels (Receipt, Transaction, Order, UTR, Ref, Payment ID).
  - date (2): Primary invoice issuance date token.
  - gstin (3): 15-character GSTINs passing Mod-36 checksum (high confidence), with
    candidates passing lenient position correction tagged as low/medium confidence.
  - table (4): From header row (>= 3 distinct header keywords) down to the row before the
    first totals row (Subtotal, Total, Grand Total, Tax summary) or page bottom.
  - total (5): Amount token only (plus currency symbol), max 3 words, preferring rows
    labeled Grand Total / Invoice Total / Net Payable.
  - qr (6): Embedded square images in PDFs / multi-scale OpenCV QRCodeDetector (1x, 2x, 3x).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from invoice_vision.parse.spatial_rules import (
    position_aware_gstin_correct,
    validate_gstin_checksum,
)

CLASS_NAMES = [
    "vendor",           # 0
    "invoice_no",       # 1
    "date",             # 2
    "gstin",            # 3
    "table",            # 4
    "total",            # 5
    "qr",               # 6
    "stamp_signature",  # 7
]

# Regex patterns
GSTIN_REGEX = re.compile(
    r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b", re.IGNORECASE
)
GSTIN_LIKE_PATTERN = re.compile(
    r"^[0-9A-Z]{2}[A-Z]{5}[0-9A-Z]{4}[A-Z]{1}[0-9A-Z]{1}[Z7][0-9A-Z]{1}$"
)
DATE_REGEX = re.compile(
    r"\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}[-/\.][A-Za-z]{3}[-/\.]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b"
)
AMOUNT_REGEX = re.compile(
    r"(?:₹|Rs\.?|INR|[\{<€$₹\(\[])?\s*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:[\.,][0-9]{1,3})?|[0-9]+(?:[\.,][0-9]{1,3})?)"
)


@dataclass
class WordBox:
    """Standardized representation of a single word token across OCR & PDF sources."""

    text: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float = 1.0

    @property
    def xc(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def yc(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    @property
    def width(self) -> float:
        return max(1.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(1.0, self.ymax - self.ymin)


@dataclass
class SuggestedBox:
    """Bounding box proposal with class ID and confidence tier."""

    class_id: int
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: str  # 'high', 'medium', 'low'
    score: float = 1.0
    source_field: str = ""
    matched_text: str = ""

    def to_yolo(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert pixel bounding box to normalized YOLO [xc, yc, w, h]."""
        x0 = max(0.0, min(float(img_w), self.xmin))
        y0 = max(0.0, min(float(img_h), self.ymin))
        x1 = max(0.0, min(float(img_w), self.xmax))
        y1 = max(0.0, min(float(img_h), self.ymax))

        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        xc = x0 + bw / 2.0
        yc = y0 + bh / 2.0
        return (xc / img_w, yc / img_h, bw / img_w, bh / img_h)


def load_words_from_source(
    data_source: Union[Path, Dict[str, Any], List[Dict[str, Any]]],
    img_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[List[WordBox], int, int]:
    """Load words and image dimensions from OCR JSON or PDF text layer JSON."""
    raw_data = data_source
    if isinstance(data_source, (str, Path)):
        p = Path(data_source)
        with open(p, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

    words: List[WordBox] = []
    width_px = img_shape[1] if img_shape else 800
    height_px = img_shape[0] if img_shape else 1000

    if isinstance(raw_data, dict) and "words" in raw_data:
        # PDF text layer format
        width_px = int(raw_data.get("width_px", width_px))
        height_px = int(raw_data.get("height_px", height_px))
        for w in raw_data["words"]:
            txt = str(w.get("text", "")).strip()
            if not txt:
                continue
            bx = w.get("bbox") or []
            if len(bx) == 4:
                words.append(
                    WordBox(
                        text=txt,
                        xmin=float(bx[0]),
                        ymin=float(bx[1]),
                        xmax=float(bx[2]),
                        ymax=float(bx[3]),
                        confidence=1.0,
                    )
                )
    elif isinstance(raw_data, list):
        if raw_data and isinstance(raw_data[0], WordBox):
            words = list(raw_data)
        else:
            # Cached EasyOCR format
            for item in raw_data:
                txt = str(item.get("text", "")).strip()
                if not txt:
                    continue
                box = item.get("box", [])
                conf = float(item.get("confidence", 1.0))
                if box and len(box) == 4:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    words.append(
                        WordBox(
                            text=txt,
                            xmin=float(min(xs)),
                            ymin=float(min(ys)),
                            xmax=float(max(xs)),
                            ymax=float(max(ys)),
                            confidence=conf,
                        )
                    )
        if words and not img_shape:
            width_px = max(width_px, int(max(w.xmax for w in words) + 20))
            height_px = max(height_px, int(max(w.ymax for w in words) + 20))

    return words, width_px, height_px


def group_words_into_lines(
    words: List[WordBox], line_tol_ratio: float = 0.55
) -> List[List[WordBox]]:
    """Group words into horizontal text lines based on vertical proximity."""
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w.yc, w.xmin))
    lines: List[List[WordBox]] = []

    for wd in sorted_words:
        if not lines:
            lines.append([wd])
            continue

        prev_line = lines[-1]
        line_yc = sum(w.yc for w in prev_line) / len(prev_line)
        line_h = max(w.height for w in prev_line)
        tol = max(6.0, line_h * line_tol_ratio)

        if abs(wd.yc - line_yc) <= tol:
            prev_line.append(wd)
        else:
            lines.append([wd])

    for line in lines:
        line.sort(key=lambda w: w.xmin)

    return lines


# ==============================================================================
# CLASS 0: VENDOR
# ==============================================================================
def suggest_vendor_box(
    words: List[WordBox], width_px: int, height_px: int
) -> Optional[SuggestedBox]:
    """Suggest top-of-page vendor / issuer block.

    Stops at the first row introducing customer details ('Bill to', 'Buyer', 'Consignee')
    or right-column metadata ('Invoice No', 'Invoice Date', 'Commercial Invoice').
    """
    if not words:
        return None

    # Consider words in upper 40% of page with reasonable confidence
    top_words = [w for w in words if w.ymin < height_px * 0.40 and w.confidence >= 0.15]
    if not top_words:
        return None

    stop_y = height_px * 0.40
    right_barrier = float(width_px)

    lines = group_words_into_lines(top_words)

    # Detect horizontal metadata split (right column)
    for line in lines:
        line_text = " ".join(w.text.lower() for w in line)
        for w in line:
            wt = w.text.lower()
            if w.xmin > width_px * 0.35:
                if any(
                    k in wt
                    for k in [
                        "invoice no",
                        "inv no",
                        "commercial invoice",
                        "statement date",
                        "payment ref",
                        "pan no",
                        "sold-to",
                        "sold to",
                        "tax invoice",
                        "dated:",
                        "date:",
                    ]
                ):
                    right_barrier = min(right_barrier, w.xmin - 5.0)

        # Detect bottom stopping rows
        if any(
            k in line_text
            for k in [
                "billed to",
                "bill to",
                "sold to",
                "buyer",
                "consignee",
                "customer details",
                "billing details",
                "billing summary",
                "gstin/uin",
                "gstin'uin",
                "gstin no",
            ]
        ):
            if "bill from" not in line_text and "ship from" not in line_text:
                stop_y = min(stop_y, min(w.ymin for w in line))

    # Collect vendor words on the left / upper quadrant
    vendor_words: List[WordBox] = []
    for w in top_words:
        wt = w.text.lower()
        if w.ymax <= stop_y and w.xmax <= right_barrier:
            # Skip standalone 'Tax Invoice' centered header line
            if w.ymin < height_px * 0.08 and "tax invoice" in wt:
                continue
            vendor_words.append(w)

    if not vendor_words:
        first_line = lines[0] if lines else []
        vendor_words = [w for w in first_line if w.xmax <= width_px * 0.60]

    if not vendor_words:
        return None

    vendor_words.sort(key=lambda w: w.ymin)
    cluster = [vendor_words[0]]
    for i in range(1, len(vendor_words)):
        if vendor_words[i].ymin - cluster[-1].ymax > 45.0:
            if not any(
                "bill from" in x.text.lower() or "ship from" in x.text.lower()
                for x in vendor_words[i:]
            ):
                break
        cluster.append(vendor_words[i])

    x0 = max(0.0, min(w.xmin for w in cluster) - 3.0)
    y0 = max(0.0, min(w.ymin for w in cluster) - 3.0)
    x1 = min(float(width_px), max(w.xmax for w in cluster) + 3.0)
    y1 = min(float(height_px), max(w.ymax for w in cluster) + 3.0)

    return SuggestedBox(
        class_id=0,
        class_name="vendor",
        xmin=x0,
        ymin=y0,
        xmax=x1,
        ymax=y1,
        confidence="high",
        score=0.90,
        source_field="header_block",
    )


# ==============================================================================
# CLASS 4: TABLE
# ==============================================================================
TABLE_HEADER_KEYWORDS = {
    "item",
    "description",
    "particulars",
    "particular",
    "qty",
    "quantity",
    "rate",
    "price",
    "hsn",
    "sac",
    "amount",
    "total",
    "code",
    "uom",
    "unit",
    "charges",
    "taxable",
    "cgst",
    "sgst",
    "igst",
    "discount",
    "descriptlon",
    "quantty",
    "eecnoon",
    "coona",
    "quanlity}",
}

TABLE_TOTALS_KEYWORDS = [
    "subtotal",
    "sub-total",
    "sub total",
    "grand total",
    "total amount",
    "net payable",
    "amount payable",
    "total charges",
    "total in words",
    "rupees",
    "tax summary",
    "gst summary",
    "total tax",
    "grand totai",
    "grand tota",
    "sub-tota",
    "tota",
]


def suggest_table_box(
    words: List[WordBox], width_px: int, height_px: int
) -> Optional[SuggestedBox]:
    """Suggest line-items table box.

    Finds header row (>= 3 header keywords or 'billing summary' / 'billing details')
    down to the last row before summary totals or bottom of page.
    """
    if not words:
        return None

    lines = group_words_into_lines(words)
    header_idx: Optional[int] = None

    for idx, line in enumerate(lines):
        line_text = " ".join(w.text.lower() for w in line)
        if "billing summary" in line_text or "billing details" in line_text:
            header_idx = idx
            break

        matched_kws: Set[str] = set()
        for w in line:
            clean_token = re.sub(r"[^\w]", "", w.text.lower())
            for kw in TABLE_HEADER_KEYWORDS:
                if kw in clean_token:
                    matched_kws.add(kw)
        if len(matched_kws) >= 3:
            header_idx = idx
            break

    if header_idx is None:
        return None

    # Scan forward to find summary/totals cutoff
    end_idx = len(lines) - 1
    for idx in range(header_idx + 1, len(lines)):
        line_text = " ".join(w.text.lower() for w in lines[idx])
        if any(tk in line_text for tk in TABLE_TOTALS_KEYWORDS):
            end_idx = idx
            break

    table_words: List[WordBox] = []
    for line in lines[header_idx : end_idx + 1]:
        table_words.extend(line)

    if not table_words:
        return None

    # 4 px padding per specification
    x0 = max(0.0, min(w.xmin for w in table_words) - 4.0)
    y0 = max(0.0, min(w.ymin for w in table_words) - 4.0)
    x1 = min(float(width_px), max(w.xmax for w in table_words) + 4.0)
    y1 = min(float(height_px), max(w.ymax for w in table_words) + 4.0)

    return SuggestedBox(
        class_id=4,
        class_name="table",
        xmin=x0,
        ymin=y0,
        xmax=x1,
        ymax=y1,
        confidence="high",
        score=0.92,
        source_field="item_table",
    )


# ==============================================================================
# CLASS 1: INVOICE_NO
# ==============================================================================
ALLOWED_INV_LABELS = [
    "invoice no",
    "invoice number",
    "invoice #",
    "tax invoice no",
    "bill no",
    "bill number",
    "bill #",
    "inv no",
    "inv number",
    "inv #",
    "invoice:",
]

FORBIDDEN_INV_LABELS = [
    "receipt",
    "transaction",
    "order",
    "utr",
    "ref",
    "reference",
    "payment id",
    "payment",
    "challan",
    "po no",
    "customer",
    "gstin",
]


def suggest_invoice_no_box(
    words: List[WordBox], width_px: int, height_px: int
) -> Optional[SuggestedBox]:
    """Suggest invoice number box.

    Strict rule: Accepts ONLY values next to labels containing 'invoice', 'bill no',
    or 'inv no'. Strictly excludes Receipt, Transaction, Order, UTR, Ref, Payment ID.
    If no valid label, produces no box.
    """
    if not words:
        return None

    lines = group_words_into_lines(words)
    best_candidate: Optional[WordBox] = None
    best_conf = "low"

    for line_idx, line in enumerate(lines):
        line_str = " ".join(w.text.lower() for w in line)
        if any(fl in line_str for fl in FORBIDDEN_INV_LABELS):
            continue

        # Skip document title 'tax invoice' if it is alone at the top without number indicator
        if "tax invoice" in line_str and not any(k in line_str for k in ["no", "number", "#", ":"]):
            continue

        for w_idx, w in enumerate(line):
            wt = w.text.lower().strip()

            # Check for merged prefix in single token: e.g. "Invoice#:ABC1234"
            merged_m = re.match(
                r"^(?:tax\s+)?(?:invoice|inv|bill)(?:\s*(?:no|num|number|#))?[\s#:\.\-_]+([\w\-\/]{3,})$",
                wt,
            )
            if merged_m:
                val_lower = merged_m.group(1)
                prefix_len = len(wt) - len(val_lower)
                val_orig = w.text.strip()[prefix_len:]
                sub_xmin = w.xmin + (prefix_len / max(1, len(w.text))) * w.width
                return SuggestedBox(
                    class_id=1,
                    class_name="invoice_no",
                    xmin=max(0.0, sub_xmin - 2.0),
                    ymin=max(0.0, w.ymin - 2.0),
                    xmax=min(float(width_px), w.xmax + 2.0),
                    ymax=min(float(height_px), w.ymax + 2.0),
                    confidence="high",
                    score=0.90,
                    source_field="invoice_no_rule",
                    matched_text=val_orig,
                )

            # Match allowed invoice number labels
            is_allowed = any(lbl in wt for lbl in ALLOWED_INV_LABELS)
            if not is_allowed and w_idx + 1 < len(line):
                phrase = f"{wt} {line[w_idx + 1].text.lower().strip()}"
                is_allowed = any(lbl in phrase for lbl in ALLOWED_INV_LABELS)

            if not is_allowed:
                continue

            # Search for value token: immediately to the right on same line
            for next_w in line[w_idx + 1 :]:
                nt = next_w.text.strip().strip(":#,- ")
                if not nt or nt.lower() in [
                    "no",
                    "number",
                    "#",
                    "date",
                    "dated",
                    "details",
                    "commercial",
                    "invoice",
                ]:
                    continue
                if re.search(r"[A-Za-z0-9]", nt) and len(nt) >= 3:
                    best_candidate = next_w
                    best_conf = "high"
                    break

            if best_candidate:
                break

            # Or search in line directly below (within 40 px)
            if line_idx + 1 < len(lines):
                below_line = lines[line_idx + 1]
                for below_w in below_line:
                    if abs(below_w.xmin - w.xmin) < 80.0:
                        nt = below_w.text.strip().strip(":#,- ")
                        if len(nt) >= 3 and re.search(r"[A-Za-z0-9]", nt):
                            best_candidate = below_w
                            best_conf = "medium"
                            break
            if best_candidate:
                break
        if best_candidate and best_conf == "high":
            break

    if not best_candidate:
        return None

    return SuggestedBox(
        class_id=1,
        class_name="invoice_no",
        xmin=max(0.0, best_candidate.xmin - 2.0),
        ymin=max(0.0, best_candidate.ymin - 2.0),
        xmax=min(float(width_px), best_candidate.xmax + 2.0),
        ymax=min(float(height_px), best_candidate.ymax + 2.0),
        confidence=best_conf,
        score=0.90 if best_conf == "high" else 0.70,
        source_field="invoice_no_rule",
        matched_text=best_candidate.text.strip().strip(":#,- "),
    )


# ==============================================================================
# CLASS 2: DATE
# ==============================================================================
DATE_LABELS = ["invoice date", "date of issue", "dated", "date", "bill date"]
FORBIDDEN_DATE_LABELS = ["due date", "period", "expiry", "challan"]


def suggest_date_box(
    words: List[WordBox], width_px: int, height_px: int
) -> Optional[SuggestedBox]:
    """Suggest primary invoice issuance date box."""
    if not words:
        return None

    lines = group_words_into_lines(words)
    best_candidate: Optional[WordBox] = None
    best_conf = "low"

    # Prefer lines in the upper 45% of the page with an explicit date label
    top_lines = [l for l in lines if l[0].ymin < height_px * 0.45]

    for line in top_lines:
        line_str = " ".join(w.text.lower() for w in line)
        if any(fd in line_str for fd in FORBIDDEN_DATE_LABELS):
            continue

        has_date_label = any(dl in line_str for dl in DATE_LABELS)

        for w in line:
            clean_t = w.text.strip("(),:; ")
            if DATE_REGEX.search(clean_t):
                best_candidate = w
                best_conf = "high" if has_date_label else "medium"
                if has_date_label:
                    break
        if best_candidate and best_conf == "high":
            break

    if not best_candidate:
        for line in lines:
            line_str = " ".join(w.text.lower() for w in line)
            if any(dl in line_str for dl in DATE_LABELS):
                for w in line:
                    if DATE_REGEX.search(w.text.strip("(),:; ")):
                        best_candidate = w
                        best_conf = "medium"
                        break
            if best_candidate:
                break

    if not best_candidate:
        return None

    return SuggestedBox(
        class_id=2,
        class_name="date",
        xmin=max(0.0, best_candidate.xmin - 2.0),
        ymin=max(0.0, best_candidate.ymin - 2.0),
        xmax=min(float(width_px), best_candidate.xmax + 2.0),
        ymax=min(float(height_px), best_candidate.ymax + 2.0),
        confidence=best_conf,
        score=0.88 if best_conf == "high" else 0.65,
        source_field="date_rule",
    )


# ==============================================================================
# CLASS 3: GSTIN
# ==============================================================================
GSTIN_LABELS = ["gstin", "gst", "uin", "gstin/uin", "gstin'uin", "gst no", "gst reg"]


def suggest_gstin_boxes(
    words: List[WordBox], width_px: int, height_px: int
) -> List[SuggestedBox]:
    """Suggest GSTIN boxes, validating checksum and returning high & low confidence lists."""
    results: List[SuggestedBox] = []
    seen_boxes: Set[Tuple[int, int]] = set()

    for idx, w in enumerate(words):
        t = w.text.strip()
        has_internal_lbl = bool(re.match(r"^(?:GSTIN|GST|UIN)[\s:/,\.\-]+", t, re.I))
        val_str = re.sub(r"^(?:GSTIN|GST|UIN)[\s:/,\.\-]+", "", t, flags=re.I).strip()
        clean = re.sub(r"[^\w]", "", val_str).upper()

        if len(val_str.split()) > 1:
            continue

        is_adj = has_internal_lbl
        if idx > 0 and not is_adj:
            pt = words[idx - 1].text.lower()
            if any(k in pt for k in ["gstin", "gst", "uin"]):
                is_adj = True

        if len(clean) != 15:
            continue

        box_key = (int(w.xc), int(w.yc))
        if box_key in seen_boxes:
            continue

        is_strict = validate_gstin_checksum(clean)
        corrected = position_aware_gstin_correct(clean)
        is_lenient = validate_gstin_checksum(corrected)
        is_like = bool(GSTIN_LIKE_PATTERN.match(corrected)) or (
            is_adj and bool(re.search(r"\d", clean)) and bool(re.search(r"[A-Z]", clean))
        )

        conf = None
        if is_strict:
            conf = "high"
        elif is_lenient:
            conf = "medium"
        elif is_like:
            conf = "low"

        if conf is not None:
            seen_boxes.add(box_key)
            xmin_val = w.xmin
            if val_str != t and len(t) > len(val_str):
                ratio = len(val_str) / float(len(t))
                xmin_val = w.xmax - (w.width * ratio)

            results.append(
                SuggestedBox(
                    class_id=3,
                    class_name="gstin",
                    xmin=max(0.0, xmin_val - 2.0),
                    ymin=max(0.0, w.ymin - 2.0),
                    xmax=min(float(width_px), w.xmax + 2.0),
                    ymax=min(float(height_px), w.ymax + 2.0),
                    confidence=conf,
                    score=0.95 if conf == "high" else (0.75 if conf == "medium" else 0.50),
                    source_field="gstin_rule",
                    matched_text=clean,
                )
            )

    return results


# ==============================================================================
# CLASS 5: TOTAL
# ==============================================================================
def suggest_total_box(
    words: List[WordBox], width_px: int, height_px: int
) -> Optional[SuggestedBox]:
    """Suggest final payable amount box.

    Strict rules:
      - Amount token only (with currency symbol if attached).
      - Never a sentence or any box with > 3 words.
      - Prefers rows labeled Grand Total / Invoice Total / Net Payable / Total Charges.
    """
    if not words:
        return None

    lines = group_words_into_lines(words)
    line_candidates: List[Tuple[WordBox, str, float, float]] = []

    for line_idx, line in enumerate(lines):
        line_str = " ".join(w.text.lower() for w in line)
        if "subtotal" in line_str or "sub-total" in line_str:
            continue

        priority = 0.0
        if any(
            k in line_str
            for k in [
                "grand total",
                "grand totai",
                "total charges",
                "total amount",
                "net payable",
                "amount payable",
                "net amount",
                "total payable",
            ]
        ):
            priority = 10.0
        elif any(k in line_str for k in ["total:", "invoice:", "total", "tota"]):
            priority = 5.0
        elif line_idx + 1 < len(lines):
            next_str = " ".join(w.text.lower() for w in lines[line_idx + 1])
            if any(k in next_str for k in ["only", "rupees", "thousand", "amount chargeable"]):
                priority = 4.0

        if priority > 0.0:
            amounts_in_line: List[WordBox] = []
            for item in line:
                it_str = item.text.strip()
                m = AMOUNT_REGEX.search(it_str)
                if m and len(m.group(1)) >= 3:
                    clean = re.sub(r"[^\w]", "", it_str).lower()
                    if not any(
                        sw in clean
                        for sw in ["rupee", "thousand", "hundred", "charge", "total", "invoice", "date"]
                    ):
                        amounts_in_line.append(item)
            if amounts_in_line:
                # The final payable total is the right-most amount token on the summary row
                right_most = max(amounts_in_line, key=lambda a: a.xmin)
                conf = "high" if priority >= 10.0 else ("medium" if priority >= 5.0 else "low")
                line_candidates.append((right_most, conf, priority, right_most.ymin))

    if not line_candidates:
        # Fallback: scan bottom 35% of page for right-most standalone amount
        bottom_words = [w for w in words if w.ymin > height_px * 0.60 and w.xmin > width_px * 0.40]
        for w in bottom_words:
            m = AMOUNT_REGEX.search(w.text.strip())
            if m and len(m.group(1)) >= 3 and len(w.text.split()) <= 3:
                line_candidates.append((w, "low", 0.1, w.ymin))

    if not line_candidates:
        return None

    # Sort by priority descending, then vertical position descending (bottom-most)
    line_candidates.sort(key=lambda c: (c[2], c[3]), reverse=True)
    best_w, best_conf, _, _ = line_candidates[0]

    return SuggestedBox(
        class_id=5,
        class_name="total",
        xmin=max(0.0, best_w.xmin - 2.0),
        ymin=max(0.0, best_w.ymin - 2.0),
        xmax=min(float(width_px), best_w.xmax + 2.0),
        ymax=min(float(height_px), best_w.ymax + 2.0),
        confidence=best_conf,
        score=0.92 if best_conf == "high" else 0.70,
        source_field="total_rule",
        matched_text=best_w.text.strip(),
    )


# ==============================================================================
# CLASS 6: QR CODE
# ==============================================================================
def detect_qr_in_image(
    img: np.ndarray, scales: Tuple[float, ...] = (1.0, 2.0, 3.0)
) -> Optional[Tuple[float, float, float, float]]:
    """Detect QR code in raster image using multi-scale OpenCV QRCodeDetector."""
    detector = cv2.QRCodeDetector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    orig_h, orig_w = gray.shape[:2]

    for scale in scales:
        if scale == 1.0:
            scaled = gray
        else:
            scaled = cv2.resize(
                gray, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_CUBIC
            )

        retval, points = detector.detect(scaled)
        if retval and points is not None and len(points) > 0:
            pts = points[0] / scale  # Map back to 1x coordinates
            x0 = float(np.min(pts[:, 0]))
            y0 = float(np.min(pts[:, 1]))
            x1 = float(np.max(pts[:, 0]))
            y1 = float(np.max(pts[:, 1]))
            if (x1 - x0) >= 20 and (y1 - y0) >= 20:
                return (x0, y0, x1, y1)

    return None


def suggest_qr_box(
    img: Optional[np.ndarray] = None,
    pdf_page: Optional[Any] = None,
    width_px: int = 0,
    height_px: int = 0,
) -> Optional[SuggestedBox]:
    """Suggest QR code box using PyMuPDF embedded images and multi-scale OpenCV detector."""
    # 1. If PDF page object is provided, check embedded images
    if pdf_page is not None:
        try:
            image_list = pdf_page.get_images()
            for img_info in image_list:
                xref = img_info[0]
                rects = pdf_page.get_image_rects(xref)
                for r in rects:
                    rw = r.x1 - r.x0
                    rh = r.y1 - r.y0
                    aspect = rw / rh if rh > 0 else 0
                    if 0.75 <= aspect <= 1.30 and rw >= 30 and rh >= 30:
                        scale_x = width_px / pdf_page.rect.width if pdf_page.rect.width > 0 else 1.0
                        scale_y = height_px / pdf_page.rect.height if pdf_page.rect.height > 0 else 1.0
                        return SuggestedBox(
                            class_id=6,
                            class_name="qr",
                            xmin=max(0.0, r.x0 * scale_x - 2.0),
                            ymin=max(0.0, r.y0 * scale_y - 2.0),
                            xmax=min(float(width_px), r.x1 * scale_x + 2.0),
                            ymax=min(float(height_px), r.y1 * scale_y + 2.0),
                            confidence="high",
                            score=0.95,
                            source_field="pdf_embedded_qr",
                        )
        except Exception:
            pass

    # 2. Multi-scale detection on raster image
    if img is not None:
        rect = detect_qr_in_image(img, scales=(1.0, 2.0, 3.0))
        if rect:
            return SuggestedBox(
                class_id=6,
                class_name="qr",
                xmin=rect[0],
                ymin=rect[1],
                xmax=rect[2],
                ymax=rect[3],
                confidence="high",
                score=0.92,
                source_field="opencv_qr",
            )

    return None


# ==============================================================================
# UNIFIED DOCUMENT SUGGESTER
# ==============================================================================
def suggest_document_boxes(
    data_source: Union[Path, Dict[str, Any], List[Dict[str, Any]]],
    img: Optional[np.ndarray] = None,
    pdf_page: Optional[Any] = None,
    img_shape: Optional[Tuple[int, int]] = None,
) -> List[SuggestedBox]:
    """Generate all bounding box proposals for a document."""
    words, width_px, height_px = load_words_from_source(data_source, img_shape=img_shape)

    proposals: List[SuggestedBox] = []

    # 0. Vendor
    v_box = suggest_vendor_box(words, width_px, height_px)
    if v_box:
        proposals.append(v_box)

    # 1. Invoice Number
    inv_box = suggest_invoice_no_box(words, width_px, height_px)
    if inv_box:
        proposals.append(inv_box)

    # 2. Date
    dt_box = suggest_date_box(words, width_px, height_px)
    if dt_box:
        proposals.append(dt_box)

    # 3. GSTIN (supplier and customer)
    gst_boxes = suggest_gstin_boxes(words, width_px, height_px)
    proposals.extend(gst_boxes)

    # 4. Table
    tbl_box = suggest_table_box(words, width_px, height_px)
    if tbl_box:
        proposals.append(tbl_box)

    # 5. Total
    tot_box = suggest_total_box(words, width_px, height_px)
    if tot_box:
        proposals.append(tot_box)

    # 6. QR Code
    qr_box = suggest_qr_box(img=img, pdf_page=pdf_page, width_px=width_px, height_px=height_px)
    if qr_box:
        proposals.append(qr_box)

    return proposals
