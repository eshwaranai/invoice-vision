"""Unit tests for dataset packaging archive tool."""

from __future__ import annotations

from pathlib import Path
import zipfile
import pytest

from scripts.zip_dataset import create_dataset_zip


def test_create_dataset_zip(tmp_path: Path) -> None:
    """Test zip archive creation, file preservation, and size calculation."""
    src_dir = tmp_path / "mock_yolo_v1"
    src_dir.mkdir()
    (src_dir / "data.yaml").write_text("names: [a, b]\n", encoding="utf-8")

    (src_dir / "images" / "train").mkdir(parents=True)
    (src_dir / "labels" / "train").mkdir(parents=True)
    (src_dir / "images" / "train" / "img1.jpg").write_bytes(b"content" * 100)
    (src_dir / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    out_zip = tmp_path / "test_out.zip"
    count, comp_mb, uncomp_mb = create_dataset_zip(src_dir, out_zip)

    assert out_zip.exists()
    assert count == 3
    assert comp_mb > 0
    assert uncomp_mb > 0

    with zipfile.ZipFile(out_zip, "r") as zf:
        names = zf.namelist()
        assert "data.yaml" in names
        assert "images/train/img1.jpg" in names
        assert "labels/train/img1.txt" in names


def test_create_dataset_zip_custom_src_and_out(tmp_path: Path) -> None:
    """Test create_dataset_zip with custom source and output paths."""
    src_v2 = tmp_path / "mock_yolo_v2"
    src_v2.mkdir()
    (src_v2 / "data.yaml").write_text("names: [a, b, c]\n", encoding="utf-8")
    out_v2_zip = tmp_path / "custom_v2.zip"

    count, comp_mb, uncomp_mb = create_dataset_zip(source_dir=src_v2, output_zip=out_v2_zip)
    assert out_v2_zip.exists()
    assert count == 1
    assert comp_mb > 0
