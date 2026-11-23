"""Unit tests for the pre-label suggester heuristics (src/invoice_vision/prelabel/suggest.py).

Tests table header detection, invoice_no label filtering, amount-only total box
extraction, and GSTIN checksum validation using small synthetic word boxes.
"""

from __future__ import annotations

import pytest

from src.invoice_vision.prelabel.suggest import (
    WordBox,
    suggest_date_box,
    suggest_document_boxes,
    suggest_gstin_boxes,
    suggest_invoice_no_box,
    suggest_table_box,
    suggest_total_box,
    suggest_vendor_box,
)


def _wb(text: str, xmin: float, ymin: float, xmax: float, ymax: float, conf: float = 0.95) -> WordBox:
    """Helper to create WordBox with precalculated properties."""
    return WordBox(
        text=text,
        xmin=float(xmin),
        ymin=float(ymin),
        xmax=float(xmax),
        ymax=float(ymax),
        confidence=conf,
    )


# =========================================================================
# 1. Table Header Detection & Boundary Tests
# =========================================================================

def test_table_header_detected_with_three_keywords():
    """Header row with >= 3 keywords (Description, Qty, Rate, Amount) triggers table extraction."""
    img_w, img_h = 1000, 1500
    words = [
        # Non-table header words above
        _wb("Acme Corp", 50, 50, 200, 80),
        _wb("Tax Invoice", 400, 50, 600, 80),
        # Table Header (y=300)
        _wb("Sl No", 50, 300, 100, 320),
        _wb("Description", 120, 300, 350, 320),
        _wb("Qty", 380, 300, 420, 320),
        _wb("Rate", 450, 300, 520, 320),
        _wb("Amount", 550, 300, 650, 320),
        # Table Rows
        _wb("1", 50, 340, 70, 360),
        _wb("Widget Pro", 120, 340, 250, 360),
        _wb("2", 380, 340, 400, 360),
        _wb("500.00", 450, 340, 510, 360),
        _wb("1000.00", 550, 340, 630, 360),
        # Summary row at bottom (stopping row)
        _wb("Grand", 350, 450, 420, 470),
        _wb("Total", 430, 450, 490, 470),
        _wb("1000.00", 550, 450, 640, 470),
    ]
    table_sugg = suggest_table_box(words, img_w, img_h)

    assert table_sugg is not None
    assert table_sugg.class_id == 4  # table
    # Table must start around header row (y ~ 300) and stop before/at summary (y ~ 450)
    assert table_sugg.ymin <= 300
    assert table_sugg.ymax >= 440
    # Normalized coordinates must be within [0, 1]
    xc, yc, w, h = table_sugg.to_yolo(img_w, img_h)
    assert 0.0 <= xc <= 1.0
    assert 0.0 <= yc <= 1.0


def test_table_header_detected_with_billing_details():
    """Header row with 'Billing Details' triggers table extraction."""
    img_w, img_h = 1000, 1500
    words = [
        _wb("Billing", 100, 400, 200, 430),
        _wb("Details", 210, 400, 320, 430),
        _wb("Consulting Service", 100, 450, 350, 480),
        _wb("50000.00", 700, 450, 800, 480),
        _wb("Total Charges", 100, 550, 280, 580),
    ]
    table_sugg = suggest_table_box(words, img_w, img_h)

    assert table_sugg is not None
    assert table_sugg.class_id == 4
    assert table_sugg.ymin <= 400


def test_no_table_when_keywords_absent():
    """When document lacks table header keywords, no table is proposed."""
    img_w, img_h = 1000, 1500
    words = [
        _wb("Receipt", 100, 100, 200, 120),
        _wb("Thank you for your business", 100, 200, 400, 220),
    ]
    table_sugg = suggest_table_box(words, img_w, img_h)
    assert table_sugg is None


# =========================================================================
# 2. Invoice Number Extraction & Label Filtering Tests
# =========================================================================

def test_invoice_number_accepts_valid_label():
    """Value adjacent to 'Invoice No:' is correctly extracted without the label."""
    img_w, img_h = 1000, 1500
    words = [
        _wb("Invoice", 50, 200, 130, 220),
        _wb("No:", 140, 200, 180, 220),
        _wb("INV-2024-8899", 200, 200, 360, 220),
    ]
    inv_sugg = suggest_invoice_no_box(words, img_w, img_h)

    assert inv_sugg is not None
    assert inv_sugg.class_id == 1  # invoice_no
    assert "INV-2024-8899" in inv_sugg.matched_text
    # Ensure the box only covers the value, not the 'Invoice No:' label
    assert inv_sugg.xmin >= 190


def test_invoice_number_rejects_tax_invoice_header():
    """Header 'Tax Invoice' alone is NOT extracted as invoice number."""
    img_w, img_h = 1000, 1500
    words = [
        _wb("TAX", 400, 50, 460, 80),
        _wb("INVOICE", 470, 50, 580, 80),
    ]
    inv_sugg = suggest_invoice_no_box(words, img_w, img_h)
    assert inv_sugg is None


def test_invoice_number_rejects_forbidden_prefixes():
    """Prefixes like 'Receipt No:', 'Payment ID:', 'Order No:' are rejected."""
    img_w, img_h = 1000, 1500
    words = [
        _wb("Receipt", 50, 200, 140, 220),
        _wb("No:", 150, 200, 190, 220),
        _wb("REC-98765", 210, 200, 320, 220),
        _wb("Order", 50, 250, 120, 270),
        _wb("No:", 130, 250, 170, 270),
        _wb("ORD-112233", 190, 250, 310, 270),
    ]
    inv_sugg = suggest_invoice_no_box(words, img_w, img_h)
    # Neither REC-98765 nor ORD-112233 should be extracted
    assert inv_sugg is None


def test_invoice_number_merged_token_subbox():
    """OCR merged token 'Invoice#:ABC1234' extracts sub-box for value."""
    img_w, img_h = 1000, 1500
    # Single merged token spanning x=100..300
    words = [
        _wb("Invoice#:ABC1234", 100, 200, 300, 220),
    ]
    inv_sugg = suggest_invoice_no_box(words, img_w, img_h)

    assert inv_sugg is not None
    assert "ABC1234" in inv_sugg.matched_text
    # Sub-box must be shifted right past 'Invoice#:'
    assert inv_sugg.xmin > 150


# =========================================================================
# 3. Amount-Only Total Box Extraction Tests
# =========================================================================

def test_total_amount_extracts_value_only():
    """On summary line, extracts amount token only, excluding 'Grand Total:' text."""
    img_w, img_h = 1000, 1500
    words = [
        _wb("Grand", 300, 800, 370, 825),
        _wb("Total:", 380, 800, 460, 825),
        _wb("15,450.00", 700, 800, 820, 825),
    ]
    total_sugg = suggest_total_box(words, img_w, img_h)

    assert total_sugg is not None
    assert total_sugg.class_id == 5  # total
    assert "15,450.00" in total_sugg.matched_text
    # Bounding box must cover only the amount (x ~ 700..820), not 'Grand Total:'
    assert total_sugg.xmin >= 690
    assert total_sugg.xmax <= 830


def test_total_amount_selects_rightmost_on_multivalue_row():
    """On row with Subtotal and Total, selects right-most amount token."""
    img_w, img_h = 1000, 1500
    words = [
        _wb("Total", 100, 900, 160, 925),
        _wb("Rs. 10,000.00", 300, 900, 410, 925),     # Taxable / Subtotal
        _wb("Rs. 1,800.00", 550, 900, 640, 925),      # Tax
        _wb("Rs. 11,800.00", 750, 900, 870, 925),     # Grand Total
    ]
    total_sugg = suggest_total_box(words, img_w, img_h)

    assert total_sugg is not None
    assert "11,800.00" in total_sugg.matched_text
    assert total_sugg.xmin >= 740


def test_total_amount_filters_out_rupees_words():
    """Rejects words like 'Rupees', 'Eleven Thousand', extracting only numeric amount."""
    img_w, img_h = 1000, 1500
    words = [
        _wb("Total", 100, 900, 160, 925),
        _wb("Amount", 170, 900, 240, 925),
        _wb("Charges:", 250, 900, 330, 925),
        _wb("8,500.50", 700, 900, 810, 925),
        # Following row in words
        _wb("Rupees", 100, 940, 180, 960),
        _wb("Eight", 190, 940, 250, 960),
        _wb("Thousand", 260, 940, 350, 960),
        _wb("Five", 360, 940, 410, 960),
        _wb("Hundred", 420, 940, 500, 960),
        _wb("Only", 510, 940, 560, 960),
    ]
    total_sugg = suggest_total_box(words, img_w, img_h)

    assert total_sugg is not None
    assert "8,500.50" in total_sugg.matched_text
    assert "Rupees" not in total_sugg.matched_text


def test_total_amount_handles_ocr_rupee_artifacts():
    """Correctly parses corrupt currency prefixes like '{' or '<' recognized by EasyOCR."""
    img_w, img_h = 1000, 1500
    words = [
        _wb("Grand", 100, 850, 170, 875),
        _wb("Total", 180, 850, 240, 875),
        _wb("{12,450.75", 700, 850, 830, 875),  # '{' is OCR misread for ₹
    ]
    total_sugg = suggest_total_box(words, img_w, img_h)

    assert total_sugg is not None
    assert "12,450.75" in total_sugg.matched_text


# =========================================================================
# 4. GSTIN Checksum & Validation Tests
# =========================================================================

def test_gstin_valid_mod36():
    """Valid GSTIN passing Mod-36 checksum is extracted with high confidence."""
    img_w, img_h = 1000, 1500
    words = [
        _wb("GSTIN:", 100, 150, 180, 175),
        _wb("09GHIJK1234L1ZE", 190, 150, 390, 175),  # Canonical valid Indian GSTIN
    ]
    gstin_suggs = suggest_gstin_boxes(words, img_w, img_h)
    assert len(gstin_suggs) >= 1
    assert gstin_suggs[0].class_id == 3  # gstin
    assert gstin_suggs[0].confidence == "high"
    assert "09GHIJK1234L1ZE" in gstin_suggs[0].matched_text


def test_gstin_merged_prefix_stripped():
    """Merged OCR token 'GSTIN:09GHIJK1234L1ZE' strips prefix and extracts value."""
    img_w, img_h = 1000, 1500
    words = [
        _wb("GSTIN:09GHIJK1234L1ZE", 100, 150, 420, 175),
    ]
    gstin_suggs = suggest_gstin_boxes(words, img_w, img_h)
    assert len(gstin_suggs) == 1
    assert "09GHIJK1234L1ZE" in gstin_suggs[0].matched_text
    # Must be shifted right past 'GSTIN:'
    assert gstin_suggs[0].xmin > 150


# =========================================================================
# 5. Full Pipeline Integration Test
# =========================================================================

def test_full_pipeline_mock_invoice():
    """End-to-end suggestion generates valid YOLO boxes for all major invoice fields."""
    img_w, img_h = 1000, 1200
    words = [
        # Vendor Block (top-left)
        _wb("Acme Solutions Pvt Ltd", 50, 40, 300, 70),
        _wb("123 Tech Park, Whitefield", 50, 75, 280, 100),
        _wb("Bangalore, Karnataka 560066", 50, 105, 290, 130),
        # Metadata Block (top-right)
        _wb("Invoice No: INV-2024-551", 600, 50, 850, 75),
        _wb("Date: 15/08/2024", 600, 85, 800, 110),
        _wb("GSTIN: 09GHIJK1234L1ZE", 600, 120, 870, 145),
        # Table Header & Body
        _wb("Description", 50, 250, 200, 280),
        _wb("Qty", 250, 250, 320, 280),
        _wb("Rate", 350, 250, 420, 280),
        _wb("Amount", 450, 250, 550, 280),
        _wb("Cloud Infrastructure Hosting", 50, 320, 300, 345),
        _wb("1", 320, 320, 340, 345),
        _wb("50000.00", 450, 320, 550, 345),
        # Grand Total
        _wb("Grand Total:", 400, 500, 550, 530),
        _wb("Rs. 50,000.00", 600, 500, 750, 530),
    ]

    suggestions = suggest_document_boxes(words, img_shape=(img_h, img_w))
    classes_present = {s.class_id for s in suggestions}

    # Must find vendor (0), invoice_no (1), date (2), gstin (3), table (4), total (5)
    assert 0 in classes_present  # vendor
    assert 1 in classes_present  # invoice_no
    assert 2 in classes_present  # date
    assert 3 in classes_present  # gstin
    assert 4 in classes_present  # table
    assert 5 in classes_present  # total

    # Every suggestion must satisfy YOLO coordinate bounds [0, 1]
    for s in suggestions:
        xc, yc, w, h = s.to_yolo(img_w, img_h)
        assert 0.0 <= xc <= 1.0
        assert 0.0 <= yc <= 1.0
        assert 0.0 < w <= 1.0
        assert 0.0 < h <= 1.0
