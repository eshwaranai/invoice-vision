"""Unit tests for code packaging tool (zip_code.py)."""

from __future__ import annotations

from pathlib import Path
import zipfile
import pytest

from scripts.zip_code import create_code_zip, is_file_allowed


def test_is_file_allowed_rules() -> None:
    """Security and privacy checks on file filtering."""
    # Allowed source code
    assert is_file_allowed(Path("src/invoice_vision/pipeline.py")) is True
    assert is_file_allowed(Path("configs/yolo_v1.yaml")) is True
    assert is_file_allowed(Path("requirements.txt")) is True

    # Excluded images / sensitive docs
    assert is_file_allowed(Path("data/raw/own/own_001_p1.png")) is False
    assert is_file_allowed(Path("gst_001.jpg")) is False
    assert is_file_allowed(Path("invoice.pdf")) is False

    # Excluded model weights & runs
    assert is_file_allowed(Path("yolov8n.pt")) is False
    assert is_file_allowed(Path("runs/detect/train/weights/best.pt")) is False

    # Excluded notebooks & cache
    assert is_file_allowed(Path("notebooks/colab_train.ipynb")) is False
    assert is_file_allowed(Path("src/__pycache__/pipeline.cpython-311.pyc")) is False


def test_create_code_zip_in_mock_env(tmp_path: Path) -> None:
    """Verify archive packaging in isolated directory layout."""
    mock_root = tmp_path / "mock_project"
    (mock_root / "configs").mkdir(parents=True)
    (mock_root / "scripts").mkdir(parents=True)
    (mock_root / "src" / "pkg").mkdir(parents=True)
    (mock_root / "data" / "raw").mkdir(parents=True)
    (mock_root / "notebooks").mkdir(parents=True)

    # Allowed files
    (mock_root / "configs" / "conf.yaml").write_text("key: value\n")
    (mock_root / "scripts" / "tool.py").write_text("print(1)\n")
    (mock_root / "src" / "pkg" / "mod.py").write_text("x = 1\n")
    (mock_root / "requirements.txt").write_text("ultralytics==8.2.70\n")

    # Forbidden files
    (mock_root / "data" / "raw" / "bill.png").write_bytes(b"image")
    (mock_root / "notebooks" / "train.ipynb").write_text("{}")
    (mock_root / "yolov8n.pt").write_bytes(b"weights")

    out_zip = tmp_path / "code_test.zip"
    count, size_kb, extensions = create_code_zip(mock_root, out_zip)

    assert out_zip.exists()
    assert count == 4
    assert extensions == {".py", ".txt", ".yaml"}

    with zipfile.ZipFile(out_zip, "r") as zf:
        names = zf.namelist()
        assert "configs/conf.yaml" in names
        assert "scripts/tool.py" in names
        assert "src/pkg/mod.py" in names
        assert "requirements.txt" in names
        # Verify no forbidden files leaked
        assert not any("bill.png" in n for n in names)
        assert not any(".ipynb" in n for n in names)
        assert not any(".pt" in n for n in names)
