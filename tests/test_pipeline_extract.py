"""Unit tests for the end-to-end invoice extraction pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from invoice_vision.pipeline.extract import (
    FieldExtraction,
    InvoiceExtractor,
    parse_date_field,
    parse_field_value,
    parse_gstin_field,
    parse_invoice_no,
    parse_total_amount,
    parse_vendor_field,
)


def test_parse_invoice_no() -> None:
    # Standard keyword prefix
    val, ok = parse_invoice_no("Invoice No: INV-2026-0001")
    assert ok is True
    assert val == "INV-2026-0001"

    # Leading hash and colon
    val, ok = parse_invoice_no("#: INV/2023/990")
    assert ok is True
    assert val == "INV/2023/990"

    # Bill number variation
    val, ok = parse_invoice_no("Bill No. 45012.")
    assert ok is True
    assert val == "45012"

    # Too short / empty
    val, ok = parse_invoice_no(":-")
    assert ok is False


def test_parse_date_field() -> None:
    # Dotted format
    val, ok = parse_date_field("Date: 13.03.2023")
    assert ok is True
    assert val == "13-03-2023"

    # Slash format with dayfirst
    val, ok = parse_date_field("Invoice Date: 05/11/2022")
    assert ok is True
    assert val == "05-11-2022"

    # Word month format
    val, ok = parse_date_field("Dated: 15-Aug-2021")
    assert ok is True
    assert val == "15-08-2021"

    # Invalid string
    val, ok = parse_date_field("invalid date string")
    assert ok is False


def test_parse_total_amount_preserves_decimals() -> None:
    # Decimal amount
    val, ok = parse_total_amount("Total Amount: ₹ 5,420.50")
    assert ok is True
    assert val == "5420.50"

    # Integer amount: MUST NOT invent decimals like .00
    val, ok = parse_total_amount("Grand Total: INR 698")
    assert ok is True
    assert val == "698"

    # Zero or invalid
    val, ok = parse_total_amount("No Amount")
    assert ok is False


def test_parse_gstin_with_position_aware_checksum() -> None:
    # Clean GSTIN
    val, ok = parse_gstin_field("GSTIN: 06BCDEF7890G1Z5")
    assert ok is True
    assert val == "06BCDEF7890G1Z5"

    # OCR confusion: Pos 12 has 'I' instead of '1'
    val, ok = parse_gstin_field("PP 19UVWXY3456ZIZI")
    assert ok is True
    assert val == "19UVWXY3456Z1ZI"

    # Completely invalid GSTIN
    val, ok = parse_gstin_field("NOT_A_GSTIN_TOKEN")
    assert ok is False


def test_parse_vendor_field() -> None:
    val, ok = parse_vendor_field("Sold By: ACME COMMERCIAL PRIVATE LIMITED\nBuilding A1")
    assert ok is True
    assert "ACME COMMERCIAL PRIVATE LIMITED" in val
    assert "\n" not in val


class MockBox:
    def __init__(self, cls_id: int, conf: float, xyxy: list[float]) -> None:
        import torch

        self.cls = torch.tensor([cls_id])
        self.conf = torch.tensor([conf])
        self.xyxy = torch.tensor([xyxy])


class MockDetectionResult:
    def __init__(self, boxes: list[MockBox]) -> None:
        self.boxes = boxes


class MockYOLO:
    def __init__(self) -> None:
        self.names = {0: "vendor", 1: "invoice_no", 2: "date", 3: "gstin", 4: "table", 5: "total"}

    def predict(self, img: np.ndarray, **kwargs: object) -> list[MockDetectionResult]:
        boxes = [
            MockBox(cls_id=0, conf=0.85, xyxy=[10.0, 10.0, 100.0, 40.0]),
            MockBox(cls_id=1, conf=0.75, xyxy=[10.0, 50.0, 80.0, 70.0]),
            MockBox(cls_id=3, conf=0.90, xyxy=[10.0, 80.0, 120.0, 100.0]),
            MockBox(cls_id=4, conf=0.95, xyxy=[10.0, 120.0, 200.0, 300.0]),
        ]
        return [MockDetectionResult(boxes)]


class MockEasyOCRReader:
    def readtext(self, crop: np.ndarray) -> list[tuple[object, str, float]]:
        # Return canned response based on crop shape or dimensions
        h, w = crop.shape[:2]
        if w > 300:  # GSTIN was upscaled 3x
            return [([], "19UVWXY3456ZIZI", 0.92)]
        elif h <= 35:  # invoice_no
            return [([], "SMPL-43583", 0.88)]
        else:  # vendor
            return [([], "APEX LOGISTICS LLP", 0.95)]


def test_invoice_extractor_mock_execution() -> None:
    mock_img = np.zeros((400, 300, 3), dtype=np.uint8)
    extractor = InvoiceExtractor(
        weights_path="dummy.pt",
        conf_thresh=0.25,
        ocr_reader=MockEasyOCRReader(),
        yolo_model=MockYOLO(),
    )

    result = extractor.extract(mock_img)
    assert "fields" in result
    fields = result["fields"]

    # Check vendor
    assert fields["vendor"]["value"] == "APEX LOGISTICS LLP"
    assert fields["vendor"]["detector_confidence"] == 0.85
    assert fields["vendor"]["ocr_confidence"] == 0.95
    assert fields["vendor"]["needs_review"] is False

    # Check invoice_no
    assert fields["invoice_no"]["value"] == "SMPL-43583"
    assert fields["invoice_no"]["detector_confidence"] == 0.75

    # Check gstin
    assert fields["gstin"]["value"] == "19UVWXY3456Z1ZI"
    assert fields["gstin"]["validation_result"] is True
    assert len(fields["gstin"]["candidates"]) == 1

    # Check table was skipped for OCR but box is retained
    assert fields["table"]["value"] is None
    assert fields["table"]["source_box"] == [10, 120, 200, 300]
    assert fields["table"]["detector_confidence"] == 0.95

    # Check missing fields (date, total) were populated with None and needs_review=True
    assert fields["date"]["value"] is None
    assert fields["date"]["needs_review"] is True
    assert fields["total"]["value"] is None
    assert fields["total"]["needs_review"] is True


def test_confidence_threshold_and_review_flag() -> None:
    """Boxes between [0.05, 0.25) must be extracted but flagged with needs_review = True."""
    mock_img = np.zeros((400, 300, 3), dtype=np.uint8)

    class LowConfMockYOLO:
        def __init__(self) -> None:
            self.names = {0: "vendor", 1: "invoice_no", 2: "date", 3: "gstin", 4: "table", 5: "total"}

        def predict(self, img: np.ndarray, **kwargs: object) -> list[MockDetectionResult]:
            # Detection with conf = 0.15 (above operating 0.05 threshold, below 0.25 bar)
            boxes = [
                MockBox(cls_id=1, conf=0.15, xyxy=[10.0, 50.0, 80.0, 70.0]),
            ]
            return [MockDetectionResult(boxes)]

    extractor = InvoiceExtractor(
        weights_path="dummy.pt",
        conf_thresh=0.05,
        high_conf_bar=0.25,
        ocr_reader=MockEasyOCRReader(),
        yolo_model=LowConfMockYOLO(),
    )

    result = extractor.extract(mock_img)
    inv_field = result["fields"]["invoice_no"]
    assert inv_field["value"] == "SMPL-43583"
    assert inv_field["detector_confidence"] == 0.15
    # Must be flagged for review because detector conf 0.15 < 0.25
    assert inv_field["needs_review"] is True


def test_compute_token_overlap() -> None:
    from scripts.eval_pipeline import compute_token_overlap

    # Exact overlap
    score = compute_token_overlap("ACME COMMERCIAL PRIVATE LIMITED", "ACME COMMERCIAL PRIVATE LIMITED")
    assert score == 1.0

    # Partial overlap with noise
    score = compute_token_overlap("Sold By: ACME COMMERCIAL PRIVATE LIMITED (Building A1)", "ACME COMMERCIAL PRIVATE LIMITED")
    assert score == 1.0

    # Overlap with missing word
    score = compute_token_overlap("own_family_A Corporation", "own_family_A Corporation of India Limited")
    assert round(score, 2) == 0.40

    # Disjoint
    score = compute_token_overlap("Amazon Seller Services", "ACME COMMERCIAL PRIVATE LIMITED")
    assert score == 0.0

