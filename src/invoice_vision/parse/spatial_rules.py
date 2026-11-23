"""Spatial 2D layout parser (v1) for Indian commercial invoices and tax bills.

Features:
  - Y-center bounding box clustering into reading rows.
  - Fuzzy keyword anchoring (SequenceMatcher >= 0.7) to withstand OCR typos (e.g. 'ILVOICE KO', 'Irvcice No').
  - 2D key-value pairing (same-row rightward search, then vertical downward fallback).
  - Robust monetary normalization handling messy OCR numbers with uncertainty flags.
  - Negative context filtering for dates (skipping Due, Period, Delivery, Billing cycle).
  - Position-aware GSTIN correction and official Mod-36 Luhn checksum validation.
  - Automated quality gate: assigns confidence score and `needs_review` flag for every extracted field.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

from invoice_vision.ocr.easyocr_engine import OCRItem


# ------------------------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------------------------

@dataclass
class ExtractedField:
    """Represents an extracted semantic entity with spatial provenance and review status."""

    value: Optional[str]
    confidence: float = 0.0
    source_box: Optional[List[List[float]]] = None
    needs_review: bool = False
    is_uncertain: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "source_box": self.source_box,
            "needs_review": self.needs_review,
            "is_uncertain": self.is_uncertain,
        }


@dataclass
class TextRow:
    """A horizontal row of spatially grouped OCR tokens."""

    items: List[OCRItem]
    y_center: float
    height: float

    @property
    def text(self) -> str:
        return " ".join(item.text for item in self.items)


# ------------------------------------------------------------------------------
# Geometry & Spatial Clustering
# ------------------------------------------------------------------------------

def get_box_bounds(box: List[List[float]]) -> Tuple[float, float, float, float, float, float]:
    """Calculate min_x, max_x, min_y, max_y, center_x, center_y for a 4-point polygon."""
    xs = [pt[0] for pt in box]
    ys = [pt[1] for pt in box]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return min_x, max_x, min_y, max_y, (min_x + max_x) / 2.0, (min_y + max_y) / 2.0


def cluster_into_rows(ocr_items: List[OCRItem], y_tolerance_ratio: float = 0.5) -> List[TextRow]:
    """Group OCR bounding boxes into horizontal reading rows by y-center clustering."""
    if not ocr_items:
        return []

    valid_items = [it for it in ocr_items if it.text and it.text.strip()]
    if not valid_items:
        return []

    sorted_items = sorted(valid_items, key=lambda it: get_box_bounds(it.box)[5])

    clusters: List[List[OCRItem]] = []
    cluster_centers: List[float] = []
    cluster_heights: List[float] = []

    for item in sorted_items:
        _, _, min_y, max_y, _, c_y = get_box_bounds(item.box)
        height = max(1.0, max_y - min_y)

        best_cluster_idx = -1
        min_dist = float("inf")

        for idx, center in enumerate(cluster_centers):
            dist = abs(c_y - center)
            tolerance = max(cluster_heights[idx], height) * y_tolerance_ratio
            if dist <= tolerance and dist < min_dist:
                min_dist = dist
                best_cluster_idx = idx

        if best_cluster_idx >= 0:
            clusters[best_cluster_idx].append(item)
            n = len(clusters[best_cluster_idx])
            cluster_centers[best_cluster_idx] = (
                cluster_centers[best_cluster_idx] * (n - 1) + c_y
            ) / n
            cluster_heights[best_cluster_idx] = max(cluster_heights[best_cluster_idx], height)
        else:
            clusters.append([item])
            cluster_centers.append(c_y)
            cluster_heights.append(height)

    rows: List[TextRow] = []
    for c_idx, items in enumerate(clusters):
        items_sorted = sorted(items, key=lambda it: get_box_bounds(it.box)[0])
        rows.append(
            TextRow(
                items=items_sorted,
                y_center=cluster_centers[c_idx],
                height=cluster_heights[c_idx],
            )
        )

    rows.sort(key=lambda r: r.y_center)
    return rows


# ------------------------------------------------------------------------------
# Fuzzy String Matching
# ------------------------------------------------------------------------------

def fuzzy_match(token: str, target: str) -> float:
    """Compute string similarity ratio between two tokens."""
    return difflib.SequenceMatcher(None, token.lower().strip(), target.lower().strip()).ratio()


def fuzzy_contains(text: str, target: str, threshold: float = 0.70) -> bool:
    """Check if any word or n-gram in text fuzzy-matches target with similarity >= threshold."""
    text_clean = text.lower().strip()
    target_clean = target.lower().strip()
    if target_clean in text_clean:
        return True

    words = text_clean.split()
    target_words = target_clean.split()
    target_len = len(target_words)

    if target_len == 1:
        for w in words:
            clean_w = re.sub(r"[^\w]", "", w)
            if fuzzy_match(clean_w, target_clean) >= threshold:
                return True
        return False

    for i in range(len(words) - target_len + 1):
        window = " ".join(words[i : i + target_len])
        clean_window = re.sub(r"[^\w\s]", "", window)
        if fuzzy_match(clean_window, target_clean) >= threshold:
            return True

    return False


# ------------------------------------------------------------------------------
# Indian GSTIN: Position-Aware Correction & Luhn Mod-36 Checksum
# ------------------------------------------------------------------------------

GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def validate_gstin_checksum(gstin: str) -> bool:
    """Validate 15-character Indian GSTIN via the official Modulus 36 Luhn-variant checksum."""
    if len(gstin) != 15:
        return False
    gstin = gstin.upper()
    total = 0
    for i in range(14):
        c = gstin[i]
        if c not in GSTIN_CHARS:
            return False
        val = GSTIN_CHARS.index(c)
        factor = 1 if (i % 2 == 0) else 2
        prod = val * factor
        total += (prod // 36) + (prod % 36)

    check_val = (36 - (total % 36)) % 36
    return GSTIN_CHARS[check_val] == gstin[14]


def position_aware_gstin_correct(candidate: str) -> str:
    """Correct character confusions based on the 15-char Indian GSTIN specification.

    Positions:
      - 0, 1: State code digits
      - 2..6: PAN alphabetic characters
      - 7..10: PAN numeric serial digits
      - 11: PAN alphabetic check
      - 12: Entity alphanumeric
      - 13: Default 'Z'
      - 14: Check character
    """
    to_digit = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "Z": "7", "B": "8"})
    to_alpha = str.maketrans({"0": "O", "1": "I", "5": "S", "7": "Z", "8": "B"})

    cleaned = list(re.sub(r"[\s\-_:/\.]", "", candidate).upper())
    if len(cleaned) != 15:
        return "".join(cleaned)

    # For foreign OIDAR registrations (state code 99), do not alter country/entity code
    if "".join(cleaned[:2]) == "99" or "SGP" in "".join(cleaned):
        if cleaned[0] in to_digit:
            cleaned[0] = cleaned[0].translate(to_digit)
        if cleaned[1] in to_digit:
            cleaned[1] = cleaned[1].translate(to_digit)
        return "".join(cleaned)

    digit_positions = {0, 1, 7, 8, 9, 10}
    alpha_positions = {2, 3, 4, 5, 6, 11}

    for i in range(15):
        if i in digit_positions:
            cleaned[i] = cleaned[i].translate(to_digit)
        elif i in alpha_positions:
            cleaned[i] = cleaned[i].translate(to_alpha)
        elif i == 13:
            if cleaned[i] in ("2", "7"):
                cleaned[i] = "Z"

    return "".join(cleaned)


# ------------------------------------------------------------------------------
# Amount Parser & Normalizer (Honesty Implementation)
# ------------------------------------------------------------------------------

def parse_amount(text: str) -> Tuple[Optional[float], bool]:
    """Extract and parse monetary value from OCR text.

    Returns:
      (amount_float, is_uncertain)
    When an amount has ambiguous separators or trailing dots (e.g. 'INR 12.382.'),
    we do NOT fill in guessed decimals; we parse as-is and flag is_uncertain=True.
    """
    if not text:
        return None, False

    s = re.sub(r"(?:inr|rs\.?|rs/-|₹|<|>|\$)", " ", text, flags=re.IGNORECASE).strip()
    s = re.sub(r"(?<=\d)[,\s]+(?=\d)", lambda m: "," if "," in m.group() else "", s)

    is_uncertain = False

    # Detect truncated/ambiguous separator pattern like "12.382."
    if re.search(r"\d+\.\d{3}\.", s) or s.endswith("."):
        is_uncertain = True

    matches = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?|\d+(?:\.\d+)?", s)
    if not matches:
        return None, False

    cand = matches[-1].replace(",", "").strip()

    # If multiple dots exist (e.g. "12.382.92"), flag uncertain
    if cand.count(".") > 1:
        is_uncertain = True
        parts = cand.split(".")
        cand = "".join(parts[:-1]) + "." + parts[-1]
    elif cand.endswith("."):
        is_uncertain = True
        cand = cand[:-1]

    try:
        val = float(cand)
        if val > 0.0:
            return val, is_uncertain
    except ValueError:
        return None, False

    return None, False


# ------------------------------------------------------------------------------
# Spatial Extractor (v1)
# ------------------------------------------------------------------------------

INVOICE_NO_LABELS = [
    "invoice no", "invoice number", "invoice #", "inv no", "inv. no.",
    "bill no", "bill number", "bill #", "tax invoice no", "irvcice no", "ilvoice ko",
    "invoice serial no"
]

INVOICE_NO_EXCLUDE = [
    "bill to", "billed to", "ship to", "bill from", "ship from", "bill of supply", "cash memo"
]

DATE_LABELS = [
    "invoice date", "bill date", "date of issue", "dated", "date", "statement date"
]

DATE_NEGATIVE_KEYWORDS = [
    "due", "due date", "period", "delivery", "delivery date",
    "billing cycle", "valid till", "dispatch"
]

TOTAL_LABELS = [
    ("grand total", 5),
    ("amount payable", 5),
    ("net payable", 5),
    ("total payable", 5),
    ("total charges", 5),
    ("you'll pay", 5),
    ("youll pay", 5),
    ("total amount", 4),
    ("invoice total", 4),
    ("total due", 4),
    ("balance due", 4),
    ("subtotal", 2),
    ("sub total", 2),
    ("total", 1),
]

CUSTOMER_INDICATORS = [
    "bill to", "buyer", "customer", "ship to", "consignee", "receiver", "sold-to", "sold to"
]

SUPPLIER_INDICATORS = [
    "supplier", "sold by", "seller", "from", "bill from", "ship from", "registered office"
]


def extract_spatial_invoice_number(rows: List[TextRow]) -> ExtractedField:
    """Extract invoice number using fuzzy keyword matching and horizontal token merging."""
    for row_idx, row in enumerate(rows):
        for item_idx, item in enumerate(row.items):
            item_text = item.text.strip().lower()

            if any(exc in item_text for exc in INVOICE_NO_EXCLUDE):
                continue

            matched_label = any(
                fuzzy_contains(item_text, lbl, threshold=0.74) for lbl in INVOICE_NO_LABELS
            )
            if not matched_label:
                continue

            # 1. Search to the right on the same row
            if item_idx + 1 < len(row.items):
                r_item = row.items[item_idx + 1]
                val = r_item.text.strip(":-# ")
                tokens = [val]
                confidences = [r_item.confidence]
                box_pts = list(r_item.box)
                prev_x = get_box_bounds(r_item.box)[1]

                for nxt in row.items[item_idx + 2 :]:
                    nb = get_box_bounds(nxt.box)
                    if nb[0] - prev_x > 35 or any(kw in nxt.text.lower() for kw in ["date", "no", "due", "lr"]):
                        break
                    tokens.append(nxt.text.strip(":-# "))
                    confidences.append(nxt.confidence)
                    box_pts.extend(nxt.box)
                    prev_x = nb[1]

                merged_val = " ".join(tokens)
                avg_conf = sum(confidences) / len(confidences)
                needs_rev = avg_conf < 0.60
                return ExtractedField(
                    value=merged_val,
                    confidence=avg_conf,
                    source_box=box_pts[:4] if len(box_pts) >= 4 else r_item.box,
                    needs_review=needs_rev,
                    is_uncertain=False,
                )

            # 2. Search row immediately below
            if row_idx + 1 < len(rows):
                for b_item in rows[row_idx + 1].items:
                    clean_token = b_item.text.strip(":-# ")
                    if len(clean_token) >= 2 and not any(kw in clean_token.lower() for kw in ["date", "due", "total", "original"]):
                        needs_rev = b_item.confidence < 0.60
                        return ExtractedField(
                            value=clean_token,
                            confidence=b_item.confidence,
                            source_box=b_item.box,
                            needs_review=needs_rev,
                            is_uncertain=False,
                        )

    return ExtractedField(value=None, confidence=0.0, source_box=None, needs_review=True, is_uncertain=False)


def extract_spatial_purchase_date(rows: List[TextRow]) -> ExtractedField:
    """Extract purchase date prioritizing positive label anchors while skipping due/period dates."""
    date_regex = re.compile(
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b|"
        r"\b(\d{1,2}[\s\-\/\.](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-\/\.]\d{2,4})\b|"
        r"\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b",
        re.IGNORECASE,
    )

    candidates: List[Tuple[int, str, float, List[List[float]]]] = []

    for row_idx, row in enumerate(rows):
        row_text_lower = row.text.lower()
        is_negative = any(neg in row_text_lower for neg in DATE_NEGATIVE_KEYWORDS)
        is_positive = any(fuzzy_contains(row_text_lower, pos, threshold=0.75) for pos in DATE_LABELS)

        for item_idx, item in enumerate(row.items):
            m = date_regex.search(item.text)
            if not m:
                continue
            date_str = m.group(0).strip(".,- ")

            if is_negative:
                continue

            priority = 1
            if is_positive:
                priority = 3
                if item_idx > 0 and any(fuzzy_contains(row.items[item_idx - 1].text, pos) for pos in DATE_LABELS):
                    priority = 4
            elif row.y_center < 350:
                priority = 2

            candidates.append((priority, date_str, item.confidence, item.box))

    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        best = candidates[0]
        needs_rev = best[2] < 0.60
        return ExtractedField(
            value=best[1],
            confidence=best[2],
            source_box=best[3],
            needs_review=needs_rev,
            is_uncertain=False,
        )

    return ExtractedField(value=None, confidence=0.0, source_box=None, needs_review=True, is_uncertain=False)


def extract_spatial_supplier_gst(rows: List[TextRow]) -> ExtractedField:
    """Extract supplier GSTIN using spatial layout, position-aware correction, and Mod-36 checksum."""
    candidates: List[Tuple[int, str, float, List[List[float]], bool]] = []
    gst_regex = re.compile(r"[0-9A-Za-z]{15}")

    for row_idx, row in enumerate(rows):
        row_text_lower = row.text.lower()

        is_customer = any(c in row_text_lower for c in CUSTOMER_INDICATORS)
        is_supplier = any(s in row_text_lower for s in SUPPLIER_INDICATORS)
        is_gst_label = any(kw in row_text_lower for kw in ["gstin", "gst no", "gst", "gst reg"])

        prev_row_text = rows[row_idx - 1].text.lower() if row_idx > 0 else ""
        if any(c in prev_row_text for c in CUSTOMER_INDICATORS):
            is_customer = True
        if any(s in prev_row_text for s in SUPPLIER_INDICATORS):
            is_supplier = True

        for item_idx, item in enumerate(row.items):
            clean_item_text = re.sub(r"[\s\-_:/\.]", "", item.text).upper()

            clean_without_prefix = re.sub(r"^(?:GSTIN|GSTNO|GSTNUMBER|GSTREG|GST|UIN)", "", clean_item_text)
            matches = gst_regex.findall(clean_without_prefix)
            if not matches:
                matches = gst_regex.findall(clean_item_text)

            target_item = item
            if not matches and is_gst_label:
                if item_idx + 1 < len(row.items):
                    cand_next = re.sub(r"[\s\-_:/\.]", "", row.items[item_idx + 1].text).upper()
                    m_next = gst_regex.findall(cand_next)
                    if m_next:
                        matches = m_next
                        target_item = row.items[item_idx + 1]

            for raw_cand in matches:
                corrected = position_aware_gstin_correct(raw_cand)
                is_valid_checksum = validate_gstin_checksum(corrected)

                priority = 1
                if is_customer:
                    priority = -5
                else:
                    if is_supplier:
                        priority += 4
                    if is_gst_label:
                        priority += 3
                    if is_valid_checksum:
                        priority += 5
                    if row.y_center < 300:
                        priority += 2

                if not (corrected[:2].isdigit() and (corrected[13] == "Z" or "SGP" in corrected)):
                    priority -= 4

                candidates.append((priority, corrected, target_item.confidence, target_item.box, is_valid_checksum))

    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        best = candidates[0]
        # Needs review if confidence is low OR checksum failed
        needs_rev = (best[2] < 0.60) or (not best[4])
        return ExtractedField(
            value=best[1],
            confidence=best[2],
            source_box=best[3],
            needs_review=needs_rev,
            is_uncertain=not best[4],
        )

    return ExtractedField(value=None, confidence=0.0, source_box=None, needs_review=True, is_uncertain=False)


def extract_spatial_total_amount(rows: List[TextRow]) -> ExtractedField:
    """Extract grand total using spatial key-value pairing with uncertainty flag."""
    candidates: List[Tuple[int, float, float, List[List[float]], bool]] = []

    for row_idx, row in enumerate(rows):
        row_text_lower = row.text.lower()

        matched_label = None
        label_priority = 0

        for lbl, prio in TOTAL_LABELS:
            if fuzzy_contains(row_text_lower, lbl, threshold=0.72):
                matched_label = lbl
                label_priority = prio
                break

        if not matched_label:
            continue

        # Strategy A: Same item
        for item in row.items:
            amt, uncertain = parse_amount(item.text)
            if amt is not None:
                candidates.append((label_priority, amt, item.confidence, item.box, uncertain))

        # Strategy B: Rightward item on same row
        for item in reversed(row.items):
            amt, uncertain = parse_amount(item.text)
            if amt is not None:
                candidates.append((label_priority, amt, item.confidence, item.box, uncertain))
                break

        # Strategy C: Item below
        if row_idx + 1 < len(rows):
            below_row = rows[row_idx + 1]
            for item in reversed(below_row.items):
                amt, uncertain = parse_amount(item.text)
                if amt is not None:
                    candidates.append((label_priority - 1, amt, item.confidence, item.box, uncertain))
                    break

    if candidates:
        candidates.sort(key=lambda c: (c[0], c[3][0][1] if c[3] else 0), reverse=True)
        best = candidates[0]
        val_amt, raw_conf, box, is_unc = best[1], best[2], best[3], best[4]

        # Lower confidence and flag review if uncertain
        final_conf = min(raw_conf, 0.45) if is_unc else raw_conf
        needs_rev = is_unc or (final_conf < 0.60)
        formatted_val = f"{val_amt:,.2f}"

        return ExtractedField(
            value=formatted_val,
            confidence=final_conf,
            source_box=box,
            needs_review=needs_rev,
            is_uncertain=is_unc,
        )

    return ExtractedField(value=None, confidence=0.0, source_box=None, needs_review=True, is_uncertain=False)


def extract_spatial_fields(ocr_items: List[OCRItem]) -> Dict[str, ExtractedField]:
    """Execute v1 2D spatial extraction pipeline across clustered text rows."""
    rows = cluster_into_rows(ocr_items)

    return {
        "invoice_number": extract_spatial_invoice_number(rows),
        "purchase_date": extract_spatial_purchase_date(rows),
        "supplier_gst": extract_spatial_supplier_gst(rows),
        "total_amount": extract_spatial_total_amount(rows),
    }
