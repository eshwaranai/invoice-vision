"""Unit tests for YOLO training pipeline and configuration."""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest
import yaml

from scripts.train_yolo import (
    DEFAULT_CONFIG_PATH,
    generate_runtime_data_yaml,
    load_train_config,
    prepare_smoke_dataset,
)

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_yolo_v1_config_exists_and_valid() -> None:
    """configs/yolo_v1.yaml must exist and specify document-specific hyperparameters."""
    assert DEFAULT_CONFIG_PATH.exists()
    cfg = load_train_config(DEFAULT_CONFIG_PATH)

    # Core architecture & schedule
    assert cfg.get("model") == "yolov8n.pt"
    assert cfg.get("imgsz") == 1024
    assert cfg.get("epochs") == 100
    assert cfg.get("patience") == 30
    assert cfg.get("batch") == 8
    assert cfg.get("seed") == 0
    assert cfg.get("deterministic") is True

    # Document-appropriate augmentations
    assert cfg.get("fliplr") == 0.0
    assert cfg.get("flipud") == 0.0
    assert cfg.get("degrees") == 5.0
    assert cfg.get("perspective") == 0.0005
    assert cfg.get("scale") == 0.3
    assert cfg.get("translate") == 0.1
    assert cfg.get("mosaic") == 1.0
    assert cfg.get("close_mosaic") == 10


def test_prepare_smoke_dataset(tmp_path: Path) -> None:
    """prepare_smoke_dataset must select 4 train and 2 val images into isolated directory."""
    # Create mock dataset layout
    mock_ds = tmp_path / "mock_yolo"
    for s in ["train", "val"]:
        (mock_ds / "images" / s).mkdir(parents=True)
        (mock_ds / "labels" / s).mkdir(parents=True)

    for i in range(10):
        (mock_ds / "images" / "train" / f"gst_{i:03d}.jpg").write_bytes(b"dummy")
        (mock_ds / "labels" / "train" / f"gst_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    for i in range(5):
        (mock_ds / "images" / "val" / f"gst_{i:03d}.jpg").write_bytes(b"dummy")
        (mock_ds / "labels" / "val" / f"gst_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    mock_yaml = mock_ds / "data.yaml"
    with open(mock_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(mock_ds),
                "train": "images/train",
                "val": "images/val",
                "nc": 6,
                "names": {0: "vendor"},
            },
            f,
        )

    smoke_temp = tmp_path / "smoke_out"
    smoke_yaml_p = prepare_smoke_dataset(mock_ds, smoke_temp)

    assert smoke_yaml_p.exists()
    smoke_train_imgs = list((smoke_temp / "smoke_dataset" / "images" / "train").glob("*.jpg"))
    smoke_val_imgs = list((smoke_temp / "smoke_dataset" / "images" / "val").glob("*.jpg"))

    assert len(smoke_train_imgs) == 4
    assert len(smoke_val_imgs) == 2


def test_generate_runtime_data_yaml(tmp_path: Path) -> None:
    """generate_runtime_data_yaml must write runtime_data.yaml with resolved data_root."""
    data_root = tmp_path / "custom_yolo"
    data_root.mkdir()
    (data_root / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnc: 6\nnames:\n  0: vendor\n",
        encoding="utf-8",
    )

    runtime_yaml = generate_runtime_data_yaml(data_root)
    assert runtime_yaml.exists()
    assert runtime_yaml.name == "runtime_data.yaml"

    with open(runtime_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["path"] == str(data_root.resolve()).replace("\\", "/")
    assert cfg["train"] == "images/train"
    assert cfg["val"] == "images/val"
    assert cfg["nc"] == 6
