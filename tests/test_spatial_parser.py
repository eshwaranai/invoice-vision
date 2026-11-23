"""Unit test suite for v1 spatial layout rules, amount parsing, and GSTIN checksum."""

from __future__ import annotations

import pytest

from invoice_vision.ocr.easyocr_engine import OCRItem
from invoice_vision.parse.spatial_rules import (
    cluster_into_rows,
    fuzzy_contains,
    fuzzy_match,
    parse_amount,
    position_aware_gstin_correct,
    validate_gstin_checksum,
    extract_spatial_fields,
)


# ------------------------------------------------------------------------------
# Amount Parser Unit Tests
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw_text, expected_amount, expected_uncertain",
    [
        ("56, 640.00", 56640.0, False),
        ("INR 12.382.", 12.382, True),
        ("You'll pay <3,750.00 each month", 3750.0, False),
        ("₹ 1,25,500.00", 125500.0, False),
        ("Total: Rs. 35,120.45", 35120.45, False),
        ("15,420", 15420.0, False),
        ("2,15,400.00", 215400.0, False),
        ("Grand Total: ₹ 92000", 92000.0, False),
        ("No amount here", None, False),
        ("", None, False),
    ],
)
def test_parse_amount_messy_cases(raw_text: str, expected_amount: float | None, expected_uncertain: bool) -> None:
    val, is_unc = parse_amount(raw_text)
    if expected_amount is None:
        assert val is None
    else:
        assert val is not None
        assert abs(val - expected_amount) < 0.01
        assert is_unc == expected_uncertain


def test_parse_amount_honesty_uncertainty() -> None:
    # Explicit test for the honesty fix: truncated token flagged as uncertain without inventing decimals
    val, is_unc = parse_amount("INR 12.382.")
    assert val == 12.382
    assert is_unc is True


# ------------------------------------------------------------------------------
# GSTIN Correction & Checksum Tests
# ------------------------------------------------------------------------------

def test_gstin_checksum_valid_canonical() -> None:
    assert validate_gstin_checksum("09GHIJK1234L1ZE") is True
    assert validate_gstin_checksum("27KLMNO5678P1ZR") is True
    assert validate_gstin_checksum("19UVWXY3456Z1ZI") is True
    assert validate_gstin_checksum("06BCDEF7890G1Z5") is True


def test_gstin_checksum_invalid() -> None:
    assert validate_gstin_checksum("09GHIJK1234L1Z9") is False
    assert validate_gstin_checksum("27KLMNO5678P1Z0") is False
    assert validate_gstin_checksum("SHORT") is False


def test_gstin_position_aware_correction() -> None:
    # Case 1: State 06 GSTIN with letter 'O' in state code and 'S' in PAN serial
    raw_ocr_1 = "O6BCDEF5S86C1ZG"
    corrected_1 = position_aware_gstin_correct(raw_ocr_1)
    assert corrected_1 == "06BCDEF5586C1ZG"
    assert validate_gstin_checksum(corrected_1) is True

    # Case 2: State 07 GSTIN with letter 'O' in state code and 'S' in PAN serial
    raw_ocr_2 = "O7PQRST5S28F1ZQ"
    corrected_2 = position_aware_gstin_correct(raw_ocr_2)
    assert corrected_2 == "07PQRST5528F1ZQ"
    assert validate_gstin_checksum(corrected_2) is True


# ------------------------------------------------------------------------------
# Fuzzy Matching Tests
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ocr_typo, target_label",
    [
        ("Irvcice No", "invoice no"),
        ("ILVOICE KO", "invoice no"),
        ("Total Charges", "total charges"),
        ("Statement Date", "statement date"),
        ("Inv. No.", "inv no"),
    ],
)
def test_fuzzy_keyword_matching(ocr_typo: str, target_label: str) -> None:
    assert fuzzy_contains(ocr_typo, target_label, threshold=0.70) is True


# ------------------------------------------------------------------------------
# Spatial Row Clustering Tests
# ------------------------------------------------------------------------------

def test_cluster_into_rows() -> None:
    item1 = OCRItem(box=[[10.0, 95.0], [50.0, 95.0], [50.0, 115.0], [10.0, 115.0]], text="Invoice No", confidence=0.9)
    item2 = OCRItem(box=[[60.0, 98.0], [120.0, 98.0], [120.0, 118.0], [60.0, 118.0]], text="INV-2024-001", confidence=0.85)
    item3 = OCRItem(box=[[10.0, 200.0], [80.0, 200.0], [80.0, 220.0], [10.0, 220.0]], text="Total", confidence=0.95)

    rows = cluster_into_rows([item1, item3, item2])
    assert len(rows) == 2
    assert rows[0].items[0].text == "Invoice No"
    assert rows[0].items[1].text == "INV-2024-001"
    assert rows[1].items[0].text == "Total"


def test_spatial_extraction_end_to_end() -> None:
    items = [
        OCRItem(box=[[10.0, 20.0], [100.0, 20.0], [100.0, 35.0], [10.0, 35.0]], text="GSTIN: 09GHIJK1234L1ZE", confidence=0.9),
        OCRItem(box=[[10.0, 50.0], [80.0, 50.0], [80.0, 65.0], [10.0, 65.0]], text="Invoice No:", confidence=0.9),
        OCRItem(box=[[90.0, 50.0], [160.0, 50.0], [160.0, 65.0], [90.0, 65.0]], text="DEL-9912", confidence=0.88),
        OCRItem(box=[[10.0, 80.0], [80.0, 80.0], [80.0, 95.0], [10.0, 95.0]], text="Invoice Date:", confidence=0.9),
        OCRItem(box=[[90.0, 80.0], [170.0, 80.0], [170.0, 95.0], [90.0, 95.0]], text="15/08/2023", confidence=0.92),
        OCRItem(box=[[10.0, 250.0], [90.0, 250.0], [90.0, 265.0], [10.0, 265.0]], text="Grand Total", confidence=0.95),
        OCRItem(box=[[200.0, 250.0], [280.0, 250.0], [280.0, 265.0], [200.0, 265.0]], text="₹ 45,200.50", confidence=0.91),
    ]

    fields = extract_spatial_fields(items)
    assert fields["invoice_number"].value == "DEL-9912"
    assert fields["invoice_number"].needs_review is False

    assert fields["purchase_date"].value == "15/08/2023"
    assert fields["purchase_date"].needs_review is False

    assert fields["supplier_gst"].value == "09GHIJK1234L1ZE"
    assert fields["supplier_gst"].needs_review is False

    assert fields["total_amount"].value == "45,200.50"
    assert fields["total_amount"].needs_review is False
    assert fields["total_amount"].is_uncertain is False
