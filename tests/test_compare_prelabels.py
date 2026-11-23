"""Unit and integration tests for YOLO pre-label vs reviewed label comparison."""

from __future__ import annotations

from pathlib import Path
import pytest
import shutil
import tempfile

from scripts.validate_labels import BBox
from scripts.compare_prelabels_vs_reviewed import (
    CLASS_NAMES,
    ClassMatchStats,
    compare_batch,
    compare_image_boxes,
    match_boxes_for_class,
)


def test_bbox_iou_exact_and_disjoint():
    # Identical boxes -> IoU = 1.0
    b1 = BBox(class_id=1, xc=0.5, yc=0.5, w=0.2, h=0.2, line_no=1)
    b2 = BBox(class_id=1, xc=0.5, yc=0.5, w=0.2, h=0.2, line_no=1)
    from scripts.validate_labels import compute_iou
    assert pytest.approx(compute_iou(b1, b2), rel=1e-4) == 1.0

    # Non-overlapping boxes -> IoU = 0.0
    b3 = BBox(class_id=1, xc=0.1, yc=0.1, w=0.1, h=0.1, line_no=1)
    b4 = BBox(class_id=1, xc=0.9, yc=0.9, w=0.1, h=0.1, line_no=2)
    assert compute_iou(b3, b4) == 0.0


def test_match_boxes_kept_adjusted_deleted_added():
    # 1. Kept: IoU >= 0.5 (e.g. slight jitter)
    p_kept = BBox(class_id=1, xc=0.50, yc=0.50, w=0.20, h=0.10, line_no=1)
    r_kept = BBox(class_id=1, xc=0.51, yc=0.50, w=0.20, h=0.10, line_no=1)
    kept, adj, d, a = match_boxes_for_class([p_kept], [r_kept])
    assert (kept, adj, d, a) == (1, 0, 0, 0)

    # 2. Adjusted: 0.3 <= IoU < 0.5
    # Shift center horizontally so overlap area is ~40%
    p_adj = BBox(class_id=2, xc=0.40, yc=0.50, w=0.20, h=0.10, line_no=1)
    r_adj = BBox(class_id=2, xc=0.50, yc=0.50, w=0.20, h=0.10, line_no=1)
    kept, adj, d, a = match_boxes_for_class([p_adj], [r_adj])
    assert (kept, adj, d, a) == (0, 1, 0, 0)

    # 3. Deleted: pre-label has box, reviewed has none
    p_del = BBox(class_id=3, xc=0.20, yc=0.20, w=0.15, h=0.05, line_no=1)
    kept, adj, d, a = match_boxes_for_class([p_del], [])
    assert (kept, adj, d, a) == (0, 0, 1, 0)

    # 4. Added: reviewed has box, pre-label has none
    r_add = BBox(class_id=4, xc=0.30, yc=0.40, w=0.50, h=0.30, line_no=1)
    kept, adj, d, a = match_boxes_for_class([], [r_add])
    assert (kept, adj, d, a) == (0, 0, 0, 1)

    # 5. Weak overlap IoU < 0.3 -> treated as deleted + added
    p_far = BBox(class_id=5, xc=0.20, yc=0.20, w=0.10, h=0.10, line_no=1)
    r_far = BBox(class_id=5, xc=0.35, yc=0.20, w=0.10, h=0.10, line_no=1)
    kept, adj, d, a = match_boxes_for_class([p_far], [r_far])
    assert (kept, adj, d, a) == (0, 0, 1, 1)


def test_class_match_stats_precision_recall():
    stats = ClassMatchStats(class_id=1, class_name="invoice_no", kept=8, adjusted=2, deleted=2, added=0)
    # Total prelabels = 8 + 2 + 2 = 12
    # Total reviewed = 8 + 2 + 0 = 10
    # Strict Precision = 8 / 12 = 0.6667
    # Strict Recall = 8 / 10 = 0.8000
    # Lenient Precision = (8 + 2) / 12 = 0.8333
    # Lenient Recall = (8 + 2) / 10 = 1.0000
    assert pytest.approx(stats.precision_strict, rel=1e-3) == 8.0 / 12.0
    assert pytest.approx(stats.recall_strict, rel=1e-3) == 8.0 / 10.0
    assert pytest.approx(stats.precision_lenient, rel=1e-3) == 10.0 / 12.0
    assert pytest.approx(stats.recall_lenient, rel=1e-3) == 10.0 / 10.0

    # Test empty stats (no divide by zero)
    empty_stats = ClassMatchStats(class_id=0, class_name="vendor")
    assert empty_stats.precision_strict == 0.0
    assert empty_stats.recall_strict == 0.0
    assert empty_stats.precision_lenient == 0.0
    assert empty_stats.recall_lenient == 0.0


def test_compare_batch_synthetic_integration(tmp_path: Path):
    pre_dir = tmp_path / "prelabels"
    rev_dir = tmp_path / "reviewed"
    pre_dir.mkdir()
    rev_dir.mkdir()

    # Create synthetic file: sample_01.txt
    # Class 0: vendor (kept, exact overlap)
    # Class 1: invoice_no (adjusted, shift)
    # Class 3: gstin (deleted, false positive)
    # Class 5: total (added in reviewed)
    pre_content = (
        "0 0.500 0.100 0.400 0.080\n"  # vendor (kept)
        "1 0.400 0.250 0.200 0.040\n"  # invoice_no (adjusted)
        "3 0.300 0.350 0.250 0.030\n"  # gstin (deleted)
    )
    (pre_dir / "sample_01.txt").write_text(pre_content, encoding="utf-8")

    rev_content = (
        "0 0.500 0.100 0.400 0.080\n"  # vendor (kept, identical)
        "1 0.500 0.250 0.200 0.040\n"  # invoice_no (adjusted, IoU ~0.4)
        "5 0.800 0.850 0.150 0.040\n"  # total (added)
    )
    (rev_dir / "sample_01.txt").write_text(rev_content, encoding="utf-8")

    report = compare_batch(pre_dir, rev_dir)

    assert report["total_files"] == 1
    assert report["matching_files"] == 1

    stats_vendor = report["class_stats"][0]
    assert stats_vendor.kept == 1
    assert stats_vendor.adjusted == 0
    assert stats_vendor.deleted == 0
    assert stats_vendor.added == 0

    stats_inv = report["class_stats"][1]
    assert stats_inv.kept == 0
    assert stats_inv.adjusted == 1
    assert stats_inv.deleted == 0
    assert stats_inv.added == 0

    stats_gstin = report["class_stats"][3]
    assert stats_gstin.kept == 0
    assert stats_gstin.adjusted == 0
    assert stats_gstin.deleted == 1
    assert stats_gstin.added == 0

    stats_total = report["class_stats"][5]
    assert stats_total.kept == 0
    assert stats_total.adjusted == 0
    assert stats_total.deleted == 0
    assert stats_total.added == 1

    # Validation check
    val = report["validation_report"]
    assert val["total_files"] == 1
    assert val["total_boxes"] == 3
    assert val["error_files"] == 0
