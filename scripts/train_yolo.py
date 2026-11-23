"""YOLOv8 Document Field Detector Training Pipeline.

Trains a pretrained YOLOv8n detector on Indian invoice documents using
document-specific augmentations configured in configs/yolo_v1.yaml.
Supports a lightweight --smoke test mode for local CPU verification.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, Optional
import yaml

import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "yolo_v1.yaml"
DEFAULT_DATA_ROOT = ROOT_DIR / "data" / "yolo_v1"
DATA_YAML_PATH = ROOT_DIR / "data" / "yolo_v1" / "data.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_yolo")


def generate_runtime_data_yaml(
    data_root: Path,
    base_yaml_path: Optional[Path] = None,
    output_yaml_path: Optional[Path] = None,
) -> Path:
    """Generate a self-contained runtime data.yaml with absolute/portable data_root path."""
    data_root = Path(data_root).resolve()
    if base_yaml_path is None or not base_yaml_path.exists():
        base_yaml_path = data_root / "data.yaml"
        if not base_yaml_path.exists():
            base_yaml_path = DATA_YAML_PATH

    if base_yaml_path.exists():
        with open(base_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 6,
            "names": {0: "vendor", 1: "invoice_no", 2: "date", 3: "gstin", 4: "table", 5: "total"},
        }

    cfg["path"] = str(data_root).replace("\\", "/")
    cfg.setdefault("train", "images/train")
    cfg.setdefault("val", "images/val")
    cfg.setdefault("test", "images/test")

    if output_yaml_path is None:
        output_yaml_path = data_root / "runtime_data.yaml"

    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return output_yaml_path


def load_train_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate training configuration from YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def prepare_smoke_dataset(data_root: Path, temp_dir: Path) -> Path:
    """Create a minimal 4-image training dataset for local smoke verification."""
    data_root = Path(data_root).resolve()
    train_img_dir = (data_root / "images" / "train").resolve()
    val_img_dir = (data_root / "images" / "val").resolve()

    smoke_root = temp_dir / "smoke_dataset"
    for split in ["train", "val"]:
        (smoke_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (smoke_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Pick exactly 4 train images (2 public, 2 own if available)
    all_train = sorted(list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png")))
    pub_train = [p for p in all_train if p.stem.startswith("gst_") and not any(k in p.stem for k in ("hard", "medium", "gentle"))]
    own_train = [p for p in all_train if not p.stem.startswith("gst_")]
    smoke_train = (pub_train[:2] + own_train[:2]) if (len(pub_train) >= 2 and len(own_train) >= 2) else all_train[:4]

    for img_p in smoke_train:
        shutil.copy2(img_p, smoke_root / "images" / "train" / img_p.name)
        lbl_p = data_root / "labels" / "train" / f"{img_p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, smoke_root / "labels" / "train" / lbl_p.name)
        else:
            (smoke_root / "labels" / "train" / f"{img_p.stem}.txt").write_text("")

    # Pick 2 val images
    all_val = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")))
    smoke_val = all_val[:2] if all_val else smoke_train[:2]
    for img_p in smoke_val:
        shutil.copy2(img_p, smoke_root / "images" / "val" / img_p.name)
        lbl_p = data_root / "labels" / "val" / f"{img_p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, smoke_root / "labels" / "val" / lbl_p.name)
        else:
            (smoke_root / "labels" / "val" / f"{img_p.stem}.txt").write_text("")

    smoke_data_yaml = smoke_root / "smoke_data.yaml"
    smoke_cfg = {
        "path": str(smoke_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 6,
        "names": {0: "vendor", 1: "invoice_no", 2: "date", 3: "gstin", 4: "table", 5: "total"},
    }
    with open(smoke_data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(smoke_cfg, f, sort_keys=False)

    logger.info(
        f"Smoke dataset prepared at {smoke_root}: {len(smoke_train)} train images, {len(smoke_val)} val images"
    )
    return smoke_data_yaml


def run_training(
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_root: Optional[Path] = None,
    smoke: bool = False,
    seed: Optional[int] = None,
    run_name: Optional[str] = None,
    override_args: Optional[Dict[str, Any]] = None,
) -> Any:
    """Execute YOLO training with full config or smoke parameters."""
    cfg = load_train_config(config_path)

    model_name = cfg.get("model", "yolov8n.pt")
    resolved_data_root = Path(data_root).resolve() if data_root else (
        DEFAULT_DATA_ROOT.resolve() if DEFAULT_DATA_ROOT.exists() else Path("data/yolo_v1").resolve()
    )

    # Resolve local pretrained weights if available
    local_weights = ROOT_DIR / model_name
    if not local_weights.exists():
        local_weights = Path(model_name)
    model_source = str(local_weights) if local_weights.exists() else model_name
    logger.info(f"Initializing YOLO model from: {model_source}")
    model = YOLO(model_source)

    temp_smoke_dir = None
    train_args: Dict[str, Any] = dict(cfg)

    # Remove non-ultralytics keys if any
    train_args.pop("model", None)

    # Write runtime data.yaml with portable path
    runtime_yaml = generate_runtime_data_yaml(resolved_data_root)
    train_args["data"] = str(runtime_yaml)

    # Automatic GPU fallback if CUDA is unavailable
    if not torch.cuda.is_available() and train_args.get("device") not in ("cpu", None):
        logger.warning(f"CUDA device {train_args.get('device')} requested but unavailable. Falling back to CPU.")
        train_args["device"] = "cpu"

    if seed is not None:
        train_args["seed"] = seed

    if smoke:
        logger.info("Executing SMOKE TEST run (1 epoch, batch=2, imgsz=640, workers=0, device=cpu, 4 images)")
        temp_smoke_dir = Path(tempfile.mkdtemp(prefix="yolo_smoke_"))
        smoke_yaml = prepare_smoke_dataset(resolved_data_root, temp_smoke_dir)
        train_args.update({
            "data": str(smoke_yaml),
            "epochs": 1,
            "batch": 2,
            "imgsz": 640,
            "workers": 0,
            "device": "cpu",
            "patience": 10,
            "project": "runs/train_smoke",
            "name": run_name or "smoke_run",
            "exist_ok": True,
            "verbose": True,
        })
    else:
        # Full training run
        train_args.setdefault("project", "runs/detect")
        train_args["name"] = run_name or train_args.get("name", "train_v1")

    # Apply manual overrides if supplied
    if override_args:
        train_args.update(override_args)

    logger.info(f"Starting training with arguments:\n{yaml.dump(train_args, sort_keys=False)}")

    try:
        results = model.train(**train_args)
        logger.info("Training completed successfully.")
        return results
    finally:
        if temp_smoke_dir and temp_smoke_dir.exists():
            shutil.rmtree(temp_smoke_dir, ignore_errors=True)
            logger.info("Smoke temporary directory cleaned up.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Indian Invoices")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Path to dataset root folder (default: data/yolo_v1)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Run smoke test: 1 epoch, batch 2, imgsz 640, workers 0, device cpu, 4 images",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--run-name", type=str, default=None, help="Run name setting the output folder")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu, 0, etc.)")
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.device is not None:
        overrides["device"] = args.device

    run_training(
        config_path=Path(args.config),
        data_root=Path(args.data_root) if args.data_root else None,
        smoke=args.smoke,
        seed=args.seed,
        run_name=args.run_name,
        override_args=overrides if overrides else None,
    )


if __name__ == "__main__":
    main()
