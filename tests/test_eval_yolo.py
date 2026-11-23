"""Unit tests for YOLO evaluation metrics and privacy protection."""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from scripts.eval_yolo import (
    MIN_STAT_THRESHOLD,
    compute_box_iou,
    is_public_image,
    parse_ground_truth_boxes,
    render_prediction_overlay,
)


def test_is_public_image_privacy_filter() -> None:
    """Public filter must return True ONLY for public gst_ dataset images."""
    assert is_public_image(Path("data/raw/indian_gst/gst_001.jpg")) is True
    assert is_public_image(Path("data/yolo_v1/images/val/gst_024.jpg")) is True
    assert is_public_image(Path("gst_015.png")) is True

    # Own bills must NEVER be identified as public
    assert is_public_image(Path("data/raw/own/own_001_p1.png")) is False
    assert is_public_image(Path("data/yolo_v1/images/test/own_010_p1.png")) is False
    assert is_public_image(Path("own_022_p1.jpg")) is False


def test_compute_box_iou() -> None:
    """Verify IoU calculation on identical, overlapping, and disjoint boxes."""
    # Identical
    b1 = [10.0, 10.0, 50.0, 50.0]
    assert pytest.approx(compute_box_iou(b1, b1), abs=1e-5) == 1.0

    # Disjoint
    b2 = [60.0, 60.0, 100.0, 100.0]
    assert compute_box_iou(b1, b2) == 0.0

    # Half overlap: [0, 0, 10, 10] vs [5, 0, 15, 10] -> inter=50, union=150 -> 1/3
    b3 = [0.0, 0.0, 10.0, 10.0]
    b4 = [5.0, 0.0, 15.0, 10.0]
    assert pytest.approx(compute_box_iou(b3, b4), abs=1e-4) == 1.0 / 3.0


def test_parse_ground_truth_boxes(tmp_path: Path) -> None:
    """Verify conversion from YOLO normalized coordinates to canvas pixels."""
    lbl_file = tmp_path / "sample.txt"
    # xc=0.5, yc=0.5, w=0.4, h=0.2 on 1000x2000 canvas
    # x1 = 0.3*1000 = 300, x2 = 0.7*1000 = 700
    # y1 = 0.4*2000 = 800, y2 = 0.6*2000 = 1200
    lbl_file.write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")

    boxes = parse_ground_truth_boxes(lbl_file, img_w=1000, img_h=2000)
    assert len(boxes) == 1
    assert boxes[0]["class_id"] == 0
    assert pytest.approx(boxes[0]["bbox"]) == [300.0, 800.0, 700.0, 1200.0]


def test_privacy_guard_blocks_own_bill_overlay(tmp_path: Path) -> None:
    """render_prediction_overlay must refuse to write any file for an own bill."""
    own_img = tmp_path / "own_001_p1.png"
    own_img.write_bytes(b"dummy")

    out_overlay = tmp_path / "out" / "own_001_p1.png"
    render_prediction_overlay(
        img_path=own_img,
        gt_boxes=[{"class_id": 0, "bbox": [10, 10, 50, 50]}],
        pred_boxes=[],
        class_names={0: "vendor"},
        out_path=out_overlay,
    )
    # Output file must NOT exist
    assert not out_overlay.exists()


def test_min_stat_threshold_constant() -> None:
    """MIN_STAT_THRESHOLD must be 15 boxes for reliable metric defense."""
    assert MIN_STAT_THRESHOLD == 15
