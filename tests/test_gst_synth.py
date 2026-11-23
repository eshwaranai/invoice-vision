"""Unit tests for Synthetic GST dataset sampling, pre-labels, batches, and splits_v3."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "data" / "gst_synth_manifest.csv"
README_PATH = ROOT_DIR / "data" / "gst_synth_README.md"
SPLITS_V2_PATH = ROOT_DIR / "data" / "splits_v2.json"
SPLITS_V3_PATH = ROOT_DIR / "data" / "splits_v3.json"
BATCH_08_DIR = ROOT_DIR / "data" / "annotation" / "batch_08"
BATCH_09_DIR = ROOT_DIR / "data" / "annotation" / "batch_09"
RAW_SYNTH_DIR = ROOT_DIR / "data" / "raw" / "gst_synth"


def test_manifest_and_readme_exist():
    if not MANIFEST_PATH.exists():
        pytest.skip("Local synthetic manifest not included in public export")
    assert MANIFEST_PATH.exists(), "Missing data/gst_synth_manifest.csv"
    assert README_PATH.exists(), "Missing data/gst_synth_README.md"

    readme_text = README_PATH.read_text(encoding="utf-8")
    assert "Divya52/indian-gst-invoices" in readme_text
    assert "CC BY 4.0" in readme_text


def test_manifest_structure_and_sampling():
    if not MANIFEST_PATH.exists():
        pytest.skip("Local synthetic manifest not included in public export")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) in (24, 84), f"Expected 24 or 84 rows in manifest, found {len(reader)}"

    expected_cols = {
        "filename", "original_id", "layout", "level",
        "invoice_number", "invoice_date", "supplier_name",
        "supplier_gstin", "buyer_gstin", "grand_total"
    }
    assert expected_cols.issubset(set(reader[0].keys())), "Missing required manifest columns"

    # Verify balanced sampling per (layout, level)
    counts = {}
    for r in reader:
        key = (r["layout"], r["level"])
        counts[key] = counts.get(key, 0) + 1

    expected_layouts = {"classic", "handwritten", "minimal", "thermal"}
    expected_levels = {"clean", "photo", "scan"}

    assert len(counts) == 12, f"Expected 12 layout-level pairs, found {len(counts)}"
    for ly in expected_layouts:
        for lv in expected_levels:
            expected_n = 7 if len(reader) == 84 else 2
            assert counts.get((ly, lv)) == expected_n, f"Expected {expected_n} samples for ({ly}, {lv}), got {counts.get((ly, lv))}"


def test_splits_v3_integrity_and_zero_leakage():
    if not SPLITS_V3_PATH.exists():
        pytest.skip("Local splits_v3.json not included in public export")
    assert SPLITS_V3_PATH.exists(), "Missing data/splits_v3.json"

    with open(SPLITS_V2_PATH, "r", encoding="utf-8") as f:
        v2 = json.load(f)
    with open(SPLITS_V3_PATH, "r", encoding="utf-8") as f:
        v3 = json.load(f)

    # 1. Val and Test splits strictly untouched
    assert v3["combined_clean_splits"]["val"] == v2["combined_clean_splits"]["val"]
    assert v3["combined_clean_splits"]["test"] == v2["combined_clean_splits"]["test"]
    assert v3["split_group_allocations"]["val"] == v2["split_group_allocations"]["val"]
    assert v3["split_group_allocations"]["test"] == v2["split_group_allocations"]["test"]

    # 2. 24 new images in Train only
    new_in_train = set(v3["combined_clean_splits"]["train"]) - set(v2["combined_clean_splits"]["train"])
    assert len(new_in_train) == 24, f"Expected 24 new train images, found {len(new_in_train)}"

    # 3. 4 new layout split groups in Train only
    new_groups = set(v3["split_group_allocations"]["train"]) - set(v2["split_group_allocations"]["train"])
    expected_new_groups = {"gstsyn_classic", "gstsyn_handwritten", "gstsyn_minimal", "gstsyn_thermal"}
    assert new_groups == expected_new_groups

    # 4. Zero Group Overlap
    train_g = set(v3["split_group_allocations"]["train"])
    val_g = set(v3["split_group_allocations"]["val"])
    test_g = set(v3["split_group_allocations"]["test"])
    assert len(train_g & val_g) == 0
    assert len(train_g & test_g) == 0
    assert len(val_g & test_g) == 0


def test_batches_08_and_09():
    if not BATCH_08_DIR.exists():
        pytest.skip("Local annotation batches not included in public export")
    assert BATCH_08_DIR.exists()
    assert BATCH_09_DIR.exists()

    for b_dir, expected_count in [(BATCH_08_DIR, 15), (BATCH_09_DIR, 9)]:
        manifest_f = b_dir / "batch_manifest.json"
        assert manifest_f.exists()
        with open(manifest_f, "r", encoding="utf-8") as f:
            bm = json.load(f)
        assert bm["image_count"] == expected_count

        images = [f for f in b_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        assert len(images) == expected_count

        txts = [f for f in b_dir.glob("gstsyn_*.txt")]
        assert len(txts) == expected_count

        assert (b_dir / "classes.txt").exists()
        assert (b_dir / "labels.txt").exists()
        assert (b_dir / "classes.txt").read_text() == (b_dir / "labels.txt").read_text()

        yolo_import = b_dir / "yolo_import"
        assert yolo_import.exists()
        assert (yolo_import / "labels.txt").exists()
        import_txts = [f for f in yolo_import.glob("gstsyn_*.txt")]
        assert len(import_txts) == expected_count


def test_batches_10_to_13():
    annotation_dir = ROOT_DIR / "data" / "annotation"
    batch_dirs = [annotation_dir / f"batch_{i:02d}" for i in range(10, 14)]

    for b_dir in batch_dirs:
        if not b_dir.exists():
            pytest.skip(f"{b_dir.name} not yet assembled")

        manifest_f = b_dir / "batch_manifest.json"
        assert manifest_f.exists()
        with open(manifest_f, "r", encoding="utf-8") as f:
            bm = json.load(f)
        assert bm["image_count"] == 15
        assert bm["split"] == "train"
        assert bm["is_test"] is False

        # Check root images and txts
        images = [f for f in b_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        assert len(images) == 15
        txts = [f for f in b_dir.glob("gstsyn_*.txt")]
        assert len(txts) == 15

        # Check classes.txt and labels.txt
        assert (b_dir / "classes.txt").exists()
        assert (b_dir / "labels.txt").exists()
        assert (b_dir / "classes.txt").read_text() == (b_dir / "labels.txt").read_text()

        # Check images_only
        img_only = b_dir / "images_only"
        assert img_only.exists()
        only_images = [f for f in img_only.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        assert len(only_images) == 15

        # Check yolo_import
        yolo_import = b_dir / "yolo_import"
        assert yolo_import.exists()
        assert (yolo_import / "labels.txt").exists()
        import_txts = [f for f in yolo_import.glob("gstsyn_*.txt")]
        assert len(import_txts) == 15


def test_splits_v4_integrity_and_zero_leakage():
    splits_v4_path = ROOT_DIR / "data" / "splits_v4.json"
    if not splits_v4_path.exists():
        pytest.skip("data/splits_v4.json not yet generated")

    with open(SPLITS_V3_PATH, "r", encoding="utf-8") as f:
        v3 = json.load(f)
    with open(splits_v4_path, "r", encoding="utf-8") as f:
        v4 = json.load(f)

    # 1. Val and Test splits strictly untouched
    assert v4["combined_clean_splits"]["val"] == v3["combined_clean_splits"]["val"]
    assert v4["combined_clean_splits"]["test"] == v3["combined_clean_splits"]["test"]
    assert v4["split_group_allocations"]["val"] == v3["split_group_allocations"]["val"]
    assert v4["split_group_allocations"]["test"] == v3["split_group_allocations"]["test"]

    # 2. Exactly 60 new images in Train
    new_in_train = set(v4["combined_clean_splits"]["train"]) - set(v3["combined_clean_splits"]["train"])
    assert len(new_in_train) == 60, f"Expected 60 new train images, found {len(new_in_train)}"
    assert len(v4["combined_clean_splits"]["train"]) == 117

    # 3. Exactly 84 synthetic images in Train
    assert len(v4["gst_synth_splits"]["train"]) == 84
    assert v4["gst_synth_splits"]["val"] == []
    assert v4["gst_synth_splits"]["test"] == []

    # 4. Zero Group Overlap
    train_g = set(v4["split_group_allocations"]["train"])
    val_g = set(v4["split_group_allocations"]["val"])
    test_g = set(v4["split_group_allocations"]["test"])
    assert len(train_g & val_g) == 0
    assert len(train_g & test_g) == 0
    assert len(val_g & test_g) == 0

