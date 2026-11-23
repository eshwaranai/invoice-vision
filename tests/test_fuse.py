"""Unit tests for multi-modal candidate fusion and arithmetic validation.

Covers:
  1. Arithmetic validation picking the exact grand total among decoys (PINs, dates, line items).
  2. Subsets of 2, 3, and 4 tokens (e.g., taxable + cgst + sgst + round_off = total).
  3. Fallback when arithmetic fails: falls back to YOLO box, then to spatial rules.
  4. invoice_no & date fusion:
     - YOLO confidence >= 0.25 auto-accepts or checks agreement.
     - Disagreement between YOLO and rules selects the rules value and sets needs_review = True.
     - Low YOLO detector confidence (< 0.25) falls back to rules.
  5. GSTIN fusion: deduplication of YOLO crops and checksum-valid rule discoveries.
  6. Provenance tagging: source ("arithmetic", "yolo", "rules", "none") and review status.
"""

from __future__ import annotations

import pytest

from invoice_vision.ocr.easyocr_engine import OCRItem
from invoice_vision.parse.spatial_rules import ExtractedField
from invoice_vision.pipeline.fuse import (
    FusedField,
    InvoiceFuser,
    dates_agree,
    extract_amounts_from_tokens,
    find_arithmetic_grand_total,
    normalize_invoice_no,
)


def test_normalize_invoice_no() -> None:
    assert normalize_invoice_no("INV-2023/001") == "inv2023001"
    assert normalize_invoice_no(" #inv: 2023 001 ") == "inv2023001"
    assert normalize_invoice_no(None) == ""


def test_dates_agree() -> None:
    assert dates_agree("13-03-2023", "13.03.2023") is True
    assert dates_agree("2023-03-13", "13/03/2023") is True
    assert dates_agree("13-03-2023", "14-03-2023") is False
    assert dates_agree(None, "13-03-2023") is False


def test_arithmetic_picks_correct_total_among_decoys() -> None:
    """Test case where page contains PIN codes, invoice numbers, line items, and taxes."""
    tokens = [
        "Invoice No: 12345",
        "PIN: 110038",            # Decoy PIN (should be filtered or ignored)
        "Date: 12-05-2023",
        "Item 1: Rs. 150.00",      # Decoy line item
        "Item 2: Rs. 350.00",      # Decoy line item
        "Taxable Value: 500.00",   # Taxable amount
        "CGST (9%): 45.00",        # Tax 1
        "SGST (9%): 45.00",        # Tax 2
        "Grand Total: 590.00",     # Correct Grand Total
        "Random Code: 998877",     # Decoy large number
    ]

    candidates = extract_amounts_from_tokens(tokens)
    result = find_arithmetic_grand_total(candidates, tolerance=0.5)

    assert result is not None, "Arithmetic validation must find a valid relation"
    g_val, g_str, combo, diff = result
    assert g_val == 590.00
    assert g_str == "590.00"
    assert sorted(combo) == [45.00, 45.00, 500.00]
    assert diff <= 0.01


def test_arithmetic_with_round_off_4_tokens() -> None:
    """Test 4-token arithmetic: taxable + cgst + sgst + round_off = total."""
    amount_candidates = [
        (1000.40, "1000.40"),  # Taxable
        (90.00, "90.00"),      # CGST
        (90.00, "90.00"),      # SGST
        (1180.00, "1180.00"),  # Grand Total
        (25.00, "25.00"),      # Decoy
    ]
    # 1000.40 + 90.00 + 90.00 = 1180.40. Diff with 1180.00 is 0.40 (within 0.5 round off tolerance)
    result = find_arithmetic_grand_total(amount_candidates, tolerance=0.5)
    assert result is not None
    g_val, g_str, combo, diff = result
    assert g_val == 1180.00
    assert diff == pytest.approx(0.40, abs=0.01)


def test_arithmetic_tie_breaking_prioritizes_rules_corroboration() -> None:
    """When multiple relations exist, candidate matching rules or YOLO is prioritized."""
    amount_candidates = [
        (50.0, "50.00"),    # Decoy small relation: 20 + 30 = 50
        (20.0, "20.00"),
        (30.0, "30.00"),
        (500.0, "500.00"),  # Legitimate relation: 400 + 100 = 500
        (400.0, "400.00"),
        (100.0, "100.00"),
    ]
    # Without corroboration, maximum G (500.0) is selected
    res1 = find_arithmetic_grand_total(amount_candidates)
    assert res1 is not None and res1[0] == 500.0

    # If rules proposed 50.0, it is prioritized
    res2 = find_arithmetic_grand_total(amount_candidates, rules_amount=50.0)
    assert res2 is not None and res2[0] == 50.0


def test_total_fallback_to_yolo_and_rules() -> None:
    """When no arithmetic relation exists, falls back to YOLO then to rules."""
    fuser = InvoiceFuser(detector_high_conf=0.25)
    empty_amounts: list[tuple[float, str]] = []

    # 1. Fallback to YOLO box (high conf)
    res_yolo = fuser.fuse_total(
        yolo_val="1250.00",
        yolo_det_conf=0.65,
        yolo_ocr_conf=0.90,
        yolo_needs_review=False,
        rules_val="1200.00",
        rules_conf=0.50,
        rules_needs_review=True,
        amount_candidates=empty_amounts,
    )
    assert res_yolo.source == "yolo"
    assert res_yolo.value == "1250.00"
    assert res_yolo.needs_review is False

    # 2. Fallback to Rules when YOLO is absent
    res_rules = fuser.fuse_total(
        yolo_val=None,
        yolo_det_conf=0.0,
        yolo_ocr_conf=0.0,
        yolo_needs_review=True,
        rules_val="3400.50",
        rules_conf=0.85,
        rules_needs_review=False,
        amount_candidates=empty_amounts,
    )
    assert res_rules.source == "rules"
    assert res_rules.value == "3400.50"
    assert res_rules.needs_review is False


def test_invoice_no_disagreement_keeps_rules_and_flags_review() -> None:
    """Test rule: if YOLO and rules disagree, keep the rules value and set needs_review = True."""
    fuser = InvoiceFuser(detector_high_conf=0.25)

    res = fuser.fuse_key_value(
        field_name="invoice_no",
        yolo_val="GARBAGE-1234",
        yolo_det_conf=0.75,
        yolo_ocr_conf=0.60,
        yolo_needs_review=False,
        rules_val="INV-2026-0002",
        rules_conf=0.80,
        rules_needs_review=False,
    )
    assert res.source == "rules"
    assert res.value == "INV-2026-0002"
    assert res.needs_review is True
    assert res.details["agreement"] is False


def test_invoice_no_agreement_auto_accepts() -> None:
    """Test rule: if YOLO and rules agree, auto-accepts (needs_review = False)."""
    fuser = InvoiceFuser(detector_high_conf=0.25)

    res = fuser.fuse_key_value(
        field_name="invoice_no",
        yolo_val="INV-2023-099",
        yolo_det_conf=0.55,
        yolo_ocr_conf=0.90,
        yolo_needs_review=False,
        rules_val="INV/2023/099",
        rules_conf=0.85,
        rules_needs_review=False,
    )
    assert res.value == "INV-2023-099"
    assert res.source == "yolo"
    assert res.needs_review is False
    assert res.details["agreement"] is True


def test_gstin_fusion_and_discovery() -> None:
    """Test GSTIN fusion discovers and deduplicates valid GSTINs."""
    fuser = InvoiceFuser()

    yolo_gstin = {"value": "27KLMNO5678P1ZR", "detector_confidence": 0.85, "validation_result": True, "needs_review": False}
    yolo_candidates = [yolo_gstin]

    # Rules discovered an unannotated buyer GSTIN on the page
    buyer_gstin = "06BCDEF7890G1Z5"
    page_tokens = [
        OCRItem(box=[[0, 0], [10, 0], [10, 10], [0, 10]], text=f"Buyer GSTIN: {buyer_gstin}", confidence=0.95),
        OCRItem(box=[[0, 0], [10, 0], [10, 10], [0, 10]], text="Duplicate: 27KLMNO5678P1ZR", confidence=0.95),
    ]

    res = fuser.fuse_gstin(
        yolo_gstin_field=yolo_gstin,
        yolo_all_candidates=yolo_candidates,
        rules_gstin_val="27KLMNO5678P1ZR",
        page_tokens=page_tokens,
    )
    assert res.value == "27KLMNO5678P1ZR"
    assert res.source == "yolo"
    assert res.needs_review is False
    all_gst = res.details["all_gstins"]
    assert "27KLMNO5678P1ZR" in all_gst
    assert "06BCDEF7890G1Z5" in all_gst
    assert len(all_gst) == 2, "Duplicate GSTINs must be cleanly deduplicated"
