"""Unit and integration tests for the YOLO dataset assembler (scripts/build_yolo_dataset.py).

Tests synthetic mini-batch assembly, class remapping, dropping of qr and stamp_signature,
missing batch handling, and leakage detection.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from scripts.build_yolo_dataset import (
    TARGET_CLASSES,
    TARGET_CLASS_TO_ID,
    assemble_yolo_dataset,
    compute_file_sha256,
    generate_stats_markdown,
    resolve_batch_labels_order,
)


@pytest.fixture
def mock_dataset_env(tmp_path: Path):
    """Create a temporary synthetic annotation and export directory structure."""
    ann_dir = tmp_path / "annotation"
    exp_dir = tmp_path / "exported"
    splits_file = tmp_path / "splits.json"
    out_dir = tmp_path / "yolo_out"
    stats_file = tmp_path / "dataset_stats.md"

    ann_dir.mkdir(parents=True)
    exp_dir.mkdir(parents=True)

    # 1. Mock splits_v2.json
    splits_data = {
        "combined_clean_splits": {
            "train": ["mock_01.jpg"],
            "val": ["mock_02.jpg"],
            "test": ["mock_03.jpg"],
        }
    }
    splits_file.write_text(json.dumps(splits_data), encoding="utf-8")

    # 2. Batch 01 (Exported and ready)
    b1_ann = ann_dir / "batch_01"
    b1_ann.mkdir()
    b1_exp = exp_dir / "batch_01"
    b1_exp.mkdir()

    # Create dummy images (distinct contents)
    (b1_ann / "mock_01.jpg").write_bytes(b"image_content_01")
    (b1_ann / "mock_02.jpg").write_bytes(b"image_content_02")
    (b1_ann / "mock_03.jpg").write_bytes(b"image_content_03")

    # Classes in makesense order: 8 classes
    class_order = ["vendor", "invoice_no", "date", "gstin", "table", "total", "qr", "stamp_signature"]
    (b1_ann / "classes.txt").write_text("\n".join(class_order), encoding="utf-8")

    # Mock reviewed labels:
    # mock_01: vendor (0), total (5), qr (6) -> 2 kept, 1 dropped
    (b1_exp / "mock_01.txt").write_text(
        "0 0.5 0.1 0.4 0.1\n"
        "5 0.8 0.8 0.2 0.05\n"
        "6 0.2 0.2 0.1 0.1\n",
        encoding="utf-8",
    )
    # mock_02: date (2), stamp_signature (7) -> 1 kept, 1 dropped
    (b1_exp / "mock_02.txt").write_text(
        "2 0.3 0.2 0.2 0.03\n"
        "7 0.7 0.7 0.15 0.08\n",
        encoding="utf-8",
    )
    # mock_03: gstin (3), table (4) -> 2 kept, 0 dropped
    (b1_exp / "mock_03.txt").write_text(
        "3 0.4 0.3 0.3 0.04\n"
        "4 0.5 0.5 0.8 0.3\n",
        encoding="utf-8",
    )

    # 3. Batch 02 (Missing / unexported)
    b2_ann = ann_dir / "batch_02"
    b2_ann.mkdir()
    (b2_ann / "unexported_01.jpg").write_bytes(b"unexported")

    return {
        "ann_dir": ann_dir,
        "exp_dir": exp_dir,
        "splits_file": splits_file,
        "out_dir": out_dir,
        "stats_file": stats_file,
    }


def test_tiny_dataset_assembly_and_class_remapping(mock_dataset_env):
    """Test full assembly workflow on tiny synthetic dataset."""
    env = mock_dataset_env
    mock_groups = {"mock_01.jpg": "GroupA", "mock_02.jpg": "GroupB", "mock_03.jpg": "GroupC"}
    stats = assemble_yolo_dataset(
        annotation_dir=env["ann_dir"],
        exported_dir=env["exp_dir"],
        splits_path=env["splits_file"],
        output_dir=env["out_dir"],
        with_synth=False,
        file_to_group=mock_groups,
    )

    # 1. Batch status check
    assert stats["processed_batches"] == ["batch_01"]
    assert stats["missing_batches"] == ["batch_02"]

    # 2. Image assignment to splits
    assert stats["splits"]["train"]["total_images"] == 1
    assert stats["splits"]["val"]["total_images"] == 1
    assert stats["splits"]["test"]["total_images"] == 1

    # Verify physical file copies
    assert (env["out_dir"] / "images" / "train" / "mock_01.jpg").exists()
    assert (env["out_dir"] / "images" / "val" / "mock_02.jpg").exists()
    assert (env["out_dir"] / "images" / "test" / "mock_03.jpg").exists()

    # 3. Class remapping & Dropped classes
    # QR (class 6) and stamp_signature (class 7) dropped per split
    assert stats["dropped_boxes_per_split"]["train"]["qr"] == 1
    assert stats["dropped_boxes_per_split"]["val"]["stamp_signature"] == 1
    assert stats["total_dropped_per_split"]["train"] == 1
    assert stats["total_dropped_per_split"]["val"] == 1
    assert stats["total_dropped_per_split"]["test"] == 0
    assert sum(stats["total_dropped_per_split"].values()) == 2

    # Check train label lines: vendor (0) and total (5) remain
    train_lines = (env["out_dir"] / "labels" / "train" / "mock_01.txt").read_text().splitlines()
    assert len(train_lines) == 2
    assert train_lines[0].startswith("0 ")
    assert train_lines[1].startswith("5 ")

    # Check val label lines: date (2) remains
    val_lines = (env["out_dir"] / "labels" / "val" / "mock_02.txt").read_text().splitlines()
    assert len(val_lines) == 1
    assert val_lines[0].startswith("2 ")

    # Check test label lines: gstin (3), table (4)
    test_lines = (env["out_dir"] / "labels" / "test" / "mock_03.txt").read_text().splitlines()
    assert len(test_lines) == 2
    assert test_lines[0].startswith("3 ")
    assert test_lines[1].startswith("4 ")

    # 4. data.yaml check
    yaml_file = env["out_dir"] / "data.yaml"
    assert yaml_file.exists()
    content = yaml_file.read_text()
    assert "nc: 6" in content
    assert "vendor" in content
    assert "total" in content


def test_unassigned_image_is_refused(tmp_path: Path):
    """Images not present in splits.json must be refused and reported."""
    ann_dir = tmp_path / "ann"
    exp_dir = tmp_path / "exp"
    out_dir = tmp_path / "out"
    splits_file = tmp_path / "splits.json"

    ann_dir.mkdir()
    exp_dir.mkdir()
    b1_a = ann_dir / "batch_01"
    b1_a.mkdir()
    b1_e = exp_dir / "batch_01"
    b1_e.mkdir()

    (b1_a / "orphan.jpg").write_bytes(b"orphan")
    (b1_e / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # Empty split file
    splits_file.write_text(json.dumps({"combined_clean_splits": {}}), encoding="utf-8")

    stats = assemble_yolo_dataset(ann_dir, exp_dir, splits_file, out_dir)
    assert len(stats["refused_images"]) == 1
    assert "orphan.jpg" in stats["refused_images"][0]


def test_hash_leakage_detection(mock_dataset_env):
    """Duplicate file content across splits must trigger hash leakage violation."""
    env = mock_dataset_env
    # Make mock_02 identical to mock_01
    (env["ann_dir"] / "batch_01" / "mock_02.jpg").write_bytes(b"image_content_01")

    stats = assemble_yolo_dataset(
        annotation_dir=env["ann_dir"],
        exported_dir=env["exp_dir"],
        splits_path=env["splits_file"],
        output_dir=env["out_dir"],
    )
    assert len(stats["leakage"]["hash_leakage_violations"]) >= 1


def test_stats_markdown_generation(mock_dataset_env):
    """Statistics markdown must flag classes with < 15 instances in val/test."""
    env = mock_dataset_env
    mock_groups = {"mock_01.jpg": "GroupA", "mock_02.jpg": "GroupB", "mock_03.jpg": "GroupC"}
    stats = assemble_yolo_dataset(
        annotation_dir=env["ann_dir"],
        exported_dir=env["exp_dir"],
        splits_path=env["splits_file"],
        output_dir=env["out_dir"],
        file_to_group=mock_groups,
    )
    md = generate_stats_markdown(stats, env["stats_file"])
    assert env["stats_file"].exists()
    assert "TOO FEW TO MEASURE" in md
    assert "[PASS] Zero split_group overlap" in md


def test_split_group_leakage_detection(mock_dataset_env):
    """If one split_group appears in multiple splits, it must report a leakage violation."""
    env = mock_dataset_env
    # Assign same group to images in train (mock_01) and val (mock_02)
    leaked_groups = {"mock_01.jpg": "LeakedGroup", "mock_02.jpg": "LeakedGroup", "mock_03.jpg": "GroupC"}
    stats = assemble_yolo_dataset(
        annotation_dir=env["ann_dir"],
        exported_dir=env["exp_dir"],
        splits_path=env["splits_file"],
        output_dir=env["out_dir"],
        file_to_group=leaked_groups,
    )
    assert len(stats["leakage"]["group_leakage_violations"]) == 1
    leaked_name, splits = stats["leakage"]["group_leakage_violations"][0]
    assert leaked_name == "LeakedGroup"
    assert "train" in splits and "val" in splits


def test_missing_labels_and_empty_images(tmp_path: Path):
    """Distinguish missing label files from confirmed empty label files."""
    ann_dir = tmp_path / "ann"
    exp_dir = tmp_path / "exp"
    out_dir = tmp_path / "out"
    splits_file = tmp_path / "splits.json"

    b1_a = ann_dir / "batch_01"
    b1_e = exp_dir / "batch_01"
    b1_a.mkdir(parents=True)
    b1_e.mkdir(parents=True)

    # 1. Image with missing label file
    (b1_a / "missing_lbl.jpg").write_bytes(b"missing")

    # 2. Image with confirmed empty label file (0 objects)
    (b1_a / "empty_confirmed.jpg").write_bytes(b"empty")
    (b1_e / "empty_confirmed.txt").write_text("", encoding="utf-8")

    # 3. Image with valid box
    (b1_a / "valid.jpg").write_bytes(b"valid")
    (b1_e / "valid.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # Classes
    (b1_a / "classes.txt").write_text("vendor\ninvoice_no\ndate\ngstin\ntable\ntotal\n", encoding="utf-8")

    splits_file.write_text(
        json.dumps({
            "combined_clean_splits": {
                "train": ["valid.jpg"],
                "val": ["empty_confirmed.jpg"],
                "test": ["missing_lbl.jpg"],
            }
        }),
        encoding="utf-8",
    )

    stats = assemble_yolo_dataset(ann_dir, exp_dir, splits_file, out_dir, file_to_group={})

    # Missing label image must be listed and excluded
    assert len(stats["missing_labels_images"]) == 1
    assert "missing_lbl.jpg" in stats["missing_labels_images"][0]
    assert not (out_dir / "images" / "test" / "missing_lbl.jpg").exists()

    # Confirmed empty image must be copied and have an empty .txt label
    assert (out_dir / "images" / "val" / "empty_confirmed.jpg").exists()
    empty_txt = out_dir / "labels" / "val" / "empty_confirmed.txt"
    assert empty_txt.exists()
    assert empty_txt.read_text().strip() == ""
    assert stats["splits"]["val"]["zero_box_images"] == 1


def test_validation_warnings_tiny_and_duplicate(tmp_path: Path):
    """Validation warnings detect tiny boxes and duplicate boxes."""
    ann_dir = tmp_path / "ann"
    exp_dir = tmp_path / "exp"
    out_dir = tmp_path / "out"
    splits_file = tmp_path / "splits.json"

    b1_a = ann_dir / "batch_01"
    b1_e = exp_dir / "batch_01"
    b1_a.mkdir(parents=True)
    b1_e.mkdir(parents=True)

    (b1_a / "warn_img.jpg").write_bytes(b"warn_content")
    # Tiny box: w = 0.002 (< 0.005)
    # Duplicate boxes: two boxes of class 0 with identical coords (IoU = 1.0)
    (b1_e / "warn_img.txt").write_text(
        "0 0.5 0.5 0.002 0.1\n"
        "1 0.4 0.4 0.2 0.2\n"
        "1 0.4 0.4 0.2 0.2\n",
        encoding="utf-8",
    )
    (b1_a / "classes.txt").write_text("vendor\ninvoice_no\ndate\ngstin\ntable\ntotal\n", encoding="utf-8")

    splits_file.write_text(
        json.dumps({"combined_clean_splits": {"train": ["warn_img.jpg"], "val": [], "test": []}}),
        encoding="utf-8",
    )

    stats = assemble_yolo_dataset(ann_dir, exp_dir, splits_file, out_dir, file_to_group={})

    warnings = stats["validation_warnings"]
    assert len(warnings) >= 2
    reasons = [w["reason"] for w in warnings]
    assert any("tiny box" in r for r in reasons)
    assert any("duplicate box" in r for r in reasons)


def test_verify_val_and_test_parity_success(tmp_path: Path):
    """Test that parity verification passes when val and test are identical."""
    from scripts.build_yolo_dataset import verify_val_and_test_parity

    ref_dir = tmp_path / "ref"
    tgt_dir = tmp_path / "tgt"

    for sp in ["val", "test"]:
        (ref_dir / "images" / sp).mkdir(parents=True)
        (tgt_dir / "images" / sp).mkdir(parents=True)
        (ref_dir / "images" / sp / "img1.jpg").write_bytes(b"content_same")
        (tgt_dir / "images" / sp / "img1.jpg").write_bytes(b"content_same")

    verify_val_and_test_parity(tgt_dir, ref_dir=ref_dir)
