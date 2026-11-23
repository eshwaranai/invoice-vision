"""Baseline rule-based extraction using domain keywords and regular expressions.

Authored using standard domain knowledge of Indian commercial invoices and tax receipts:
- GSTIN structure: 15-character alphanumeric token (State code + PAN + entity + Z + check)
- Indian date formats: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, DD-Mon-YYYY
- Invoice numbers: Keyword anchors (Invoice No, Inv No, Bill No) + alphanumeric tokens
- Total amounts: Keyword anchors (Grand Total, Amount Payable, Total Amount, Net Payable)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from invoice_vision.ocr.easyocr_engine import OCRItem


# ------------------------------------------------------------------------------
# Keyword Patterns (Indian Invoices)
# ------------------------------------------------------------------------------

INV_NO_KEYWORD = re.compile(
    r"(?:tax\s+)?(?:invoice|inv|bill)\s*(?:no\.?|num(?:ber)?\.?|#)\s*[:\-\.]?\s*(.*)",
    re.IGNORECASE,
)

DATE_KEYWORD = re.compile(
    r"(?:invoice\s*date|bill\s*date|purchase\s*date|date\s*of\s*issue|dated|date)\s*[:\-\.]?\s*(.*)",
    re.IGNORECASE,
)

GSTIN_KEYWORD = re.compile(
    r"(?:gstin\s*\/?\s*uin|gstin|gst\s*no\.?|gst\s*number|gst\s*reg\.?)\s*[:\-\.]?\s*(.*)",
    re.IGNORECASE,
)

TOTAL_KEYWORDS = [
    (re.compile(r"\b(?:grand\s*total|net\s*payable|amount\s*payable|total\s*payable)\b", re.IGNORECASE), 3),
    (re.compile(r"\b(?:total\s*amount|invoice\s*total|total\s*due|balance\s*due)\b", re.IGNORECASE), 2),
    (re.compile(r"\b(?:total)\b", re.IGNORECASE), 1),
]

# ------------------------------------------------------------------------------
# Value Extraction Regexes
# ------------------------------------------------------------------------------

# Indian standard GSTIN (15 chars): 2 digits + 5 letters + 4 digits + 1 letter + 1 char + Z + 1 char
GSTIN_STRICT_REGEX = re.compile(
    r"\b([0-9]{2}[A-Za-z]{5}[0-9]{4}[A-Za-z]{1}[1-9A-Za-z]{1}[Zz][0-9A-Za-z]{1})\b"
)
# Relaxed GSTIN pattern to accommodate OCR character confusion
GSTIN_RELAXED_REGEX = re.compile(
    r"\b([0-9A-Za-z]{2}[A-Za-z]{5}[0-9A-Za-z]{4}[A-Za-z]{1}[0-9A-Za-z]{3})\b"
)

# Date patterns (Day-Month-Year and ISO formats)
DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b"),
    re.compile(
        r"\b(\d{1,2}[\s\-\/\.](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-\/\.]\d{2,4})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b"),
]

# Amount pattern (e.g., 12,382.92 or 2832.00 or 56,640)
AMOUNT_REGEX = re.compile(r"(?:₹|rs\.?|inr)?\s*(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", re.IGNORECASE)


def extract_invoice_number(lines: List[str]) -> Optional[str]:
    """Extract invoice number using keyword anchoring."""
    for idx, line in enumerate(lines):
        match = INV_NO_KEYWORD.search(line)
        if match:
            candidate = match.group(1).strip()
            # If the value is on the same line after keyword
            tokens = candidate.split()
            if tokens:
                val = tokens[0].strip(":-# \t")
                if len(val) >= 2 and not any(kw in val.lower() for kw in ["date", "due", "total", "original"]):
                    return val
            # Check immediately next line if empty on the current line
            if idx + 1 < len(lines):
                next_line = lines[idx + 1].strip()
                tokens_next = next_line.split()
                if tokens_next:
                    val = tokens_next[0].strip(":-# \t")
                    if len(val) >= 2 and not any(kw in val.lower() for kw in ["date", "due", "total"]):
                        return val
    return None


def extract_purchase_date(lines: List[str]) -> Optional[str]:
    """Extract invoice date prioritizing keyword-anchored dates, then top-of-page dates."""
    # Pass 1: Look for date anchored by keyword
    for idx, line in enumerate(lines):
        if DATE_KEYWORD.search(line):
            for pat in DATE_PATTERNS:
                m = pat.search(line)
                if m:
                    return m.group(1).strip()
            # Check next line
            if idx + 1 < len(lines):
                for pat in DATE_PATTERNS:
                    m = pat.search(lines[idx + 1])
                    if m:
                        return m.group(1).strip()

    # Pass 2: Fallback to the first date token appearing in the top 15 lines
    for line in lines[:15]:
        for pat in DATE_PATTERNS:
            m = pat.search(line)
            if m:
                return m.group(1).strip()

    return None


def extract_supplier_gst(lines: List[str]) -> Optional[str]:
    """Extract supplier GSTIN prioritizing document header tokens."""
    # Exclude customer-related lines when identifying supplier GST
    customer_indicators = ("bill to", "ship to", "buyer", "customer", "consignee")

    # Pass 1: Keyword-anchored lookup in top half of document
    for line in lines:
        lower_line = line.lower()
        if any(ind in lower_line for ind in customer_indicators):
            continue
        if GSTIN_KEYWORD.search(line):
            m = GSTIN_STRICT_REGEX.search(line)
            if m:
                return m.group(1).upper()
            m_rel = GSTIN_RELAXED_REGEX.search(line)
            if m_rel:
                return m_rel.group(1).upper()

    # Pass 2: Regex search across full text (first valid token found)
    for line in lines:
        lower_line = line.lower()
        if any(ind in lower_line for ind in customer_indicators):
            continue
        m = GSTIN_STRICT_REGEX.search(line)
        if m:
            return m.group(1).upper()

    # Pass 3: Relaxed regex across full document
    for line in lines:
        m = GSTIN_RELAXED_REGEX.search(line)
        if m:
            return m.group(1).upper()

    return None


def extract_total_amount(lines: List[str]) -> Optional[str]:
    """Extract grand total amount by searching backwards from the bottom."""
    candidates: List[tuple[int, str]] = []  # (priority, amount_str)

    # Search lines from bottom up
    for line in reversed(lines):
        for pattern, priority in TOTAL_KEYWORDS:
            if pattern.search(line):
                # Search for numeric amounts in this line
                matches = AMOUNT_REGEX.findall(line)
                if matches:
                    candidates.append((priority, matches[-1]))
                break

    if candidates:
        # Pick candidate with highest priority keyword
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1].strip()

    return None


def extract_baseline_fields(ocr_items: List[OCRItem]) -> Dict[str, Optional[str]]:
    """Extract baseline invoice fields from raw OCR items using heuristic rules.

    Fields extracted:
      - invoice_number
      - purchase_date
      - supplier_gst
      - total_amount
    """
    # Sort items spatially top-to-bottom
    sorted_items = sorted(
        ocr_items,
        key=lambda item: (item.box[0][1] if item.box else 0, item.box[0][0] if item.box else 0),
    )
    lines = [item.text.strip() for item in sorted_items if item.text and item.text.strip()]

    return {
        "invoice_number": extract_invoice_number(lines),
        "purchase_date": extract_purchase_date(lines),
        "supplier_gst": extract_supplier_gst(lines),
        "total_amount": extract_total_amount(lines),
    }
