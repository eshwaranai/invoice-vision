"""Unit tests for YOLO label validation script."""

from __future__ import annotations

from pathlib import Path
import pytest
import sys

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "scripts"))

from scripts.validate_labels import (
    BBox,
    compute_iou,
    validate_label_content,
    validate_labels_dir,
)


def test_compute_iou_identical() -> None:
    b1 = BBox(class_id=1, xc=0.5, yc=0.5, w=0.2, h=0.2, line_no=1)
    b2 = BBox(class_id=1, xc=0.5, yc=0.5, w=0.2, h=0.2, line_no=2)
    assert pytest.approx(compute_iou(b1, b2), abs=1e-5) == 1.0


def test_compute_iou_disjoint() -> None:
    b1 = BBox(class_id=1, xc=0.2, yc=0.2, w=0.1, h=0.1, line_no=1)
    b2 = BBox(class_id=1, xc=0.8, yc=0.8, w=0.1, h=0.1, line_no=2)
    assert compute_iou(b1, b2) == 0.0


def test_valid_label_content() -> None:
    lines = [
        "1 0.500000 0.300000 0.200000 0.050000",
        "2 0.800000 0.150000 0.100000 0.030000",
        "3 0.250000 0.400000 0.300000 0.040000",
    ]
    res = validate_label_content(lines, num_classes=8)
    assert res.has_errors is False
    assert res.has_warnings is False
    assert res.is_empty is False
    assert len(res.boxes) == 3


def test_out_of_range_coordinates() -> None:
    # Center > 1.0, width > 1.0, and negative center
    lines = [
        "1 1.250000 0.300000 0.200000 0.050000",
        "2 0.500000 -0.100000 0.100000 0.030000",
        "3 0.500000 0.500000 1.500000 0.040000",
    ]
    res = validate_label_content(lines, num_classes=8)
    assert res.has_errors is True
    assert len(res.out_of_range_coords) >= 3


def test_unknown_class_id() -> None:
    lines = [
        "8 0.500000 0.300000 0.200000 0.050000",
        "-1 0.200000 0.200000 0.100000 0.050000",
    ]
    res = validate_label_content(lines, num_classes=8)
    assert res.has_errors is True
    assert len(res.unknown_classes) == 2


def test_empty_file_warning() -> None:
    lines = [
        "# Just a comment line",
        "   ",
    ]
    res = validate_label_content(lines, num_classes=8)
    assert res.is_empty is True
    assert res.has_warnings is True
    assert len(res.boxes) == 0

    # When allow_empty is enabled (e.g., confirmed negative background images), it is not a warning
    res_allowed = validate_label_content(lines, num_classes=8, allow_empty=True)
    assert res_allowed.is_empty is True
    assert res_allowed.has_warnings is False


def test_tiny_box_warning() -> None:
    # Box with width = 0.002 (< 0.005 default threshold)
    lines = [
        "1 0.500000 0.500000 0.002000 0.020000",
    ]
    res = validate_label_content(lines, num_classes=8, min_dim=0.005)
    assert res.has_warnings is True
    assert len(res.tiny_boxes) == 1


def test_duplicate_box_detection() -> None:
    # Same class (1), nearly identical coordinates (IoU > 0.90)
    lines = [
        "1 0.500000 0.500000 0.200000 0.100000",
        "1 0.501000 0.501000 0.200000 0.100000",
    ]
    res = validate_label_content(lines, num_classes=8, iou_dup_thresh=0.90)
    assert res.has_warnings is True
    assert len(res.duplicate_boxes) == 1

    # Different class IDs should NOT be flagged as duplicate
    diff_class_lines = [
        "1 0.500000 0.500000 0.200000 0.100000",
        "2 0.500000 0.500000 0.200000 0.100000",
    ]
    res2 = validate_label_content(diff_class_lines, num_classes=8)
    assert len(res2.duplicate_boxes) == 0


def test_validate_prelabels_directory_real() -> None:
    labels_dir = root_dir / "data" / "prelabels"
    splits_path = root_dir / "data" / "splits.json"
    if not labels_dir.exists():
        pytest.skip("Prelabels directory does not exist.")

    report = validate_labels_dir(labels_dir, splits_path=splits_path, num_classes=8)
    assert report["total_files"] == 30
    assert report["total_boxes"] == 63
    assert report["error_files"] == 0
    assert "train" in report["split_counts"]
    assert report["split_counts"]["train"]["total_images"] == 18
    assert report["split_counts"]["val"]["total_images"] == 6
    assert report["split_counts"]["test"]["total_images"] == 6


def test_validate_prelabels_own_real() -> None:
    labels_dir = root_dir / "data" / "prelabels_own"
    splits_path = root_dir / "data" / "splits_v2.json"
    if not labels_dir.exists() or not splits_path.exists():
        pytest.skip("prelabels_own or splits_v2.json missing.")

    report = validate_labels_dir(labels_dir, splits_path=splits_path, num_classes=8)
    assert report["total_files"] == 29
    assert report["total_boxes"] == 162
    assert report["error_files"] == 0
    assert report["empty_files"] == 0
    assert report["warning_files"] == 0
    assert report["split_counts"]["train"]["images_with_boxes"] == 15
    assert report["split_counts"]["val"]["images_with_boxes"] == 6
    assert report["split_counts"]["test"]["images_with_boxes"] == 8


def test_splits_v2_zero_leakage() -> None:
    import json
    splits_path = root_dir / "data" / "splits_v2.json"
    if not splits_path.exists():
        pytest.skip("splits_v2.json missing.")

    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = data["split_group_allocations"]
    tr = set(groups["train"])
    va = set(groups["val"])
    te = set(groups["test"])

    assert len(tr & va) == 0, f"Overlap train-val: {tr & va}"
    assert len(tr & te) == 0, f"Overlap train-test: {tr & te}"
    assert len(va & te) == 0, f"Overlap val-test: {va & te}"
