"""YOLOv8 Document Field Detector Evaluation Pipeline.

Evaluates trained YOLO weights on validation and test splits:
  1. Computes Precision, Recall, AP50, AP50-95, and ground-truth box counts per class.
  2. Flags any class with < 15 ground-truth boxes as 'too few to measure'.
  3. Computes breakdowns by source (public vs own) and by template family.
  4. Identifies the 10 worst performing images by file name.
  5. Exports confusion matrix.
  6. Generates prediction overlays for PUBLIC images ONLY (Strict Privacy: zero own bill overlays).
  7. Formats and writes detailed Markdown report to results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = ROOT_DIR / "data" / "yolo_v1"
DEFAULT_DATA_YAML = ROOT_DIR / "data" / "yolo_v1" / "data.yaml"
DEFAULT_FAMILY_MAP = ROOT_DIR / "data" / "family_map.csv"
DEFAULT_VENDOR_GROUPS = ROOT_DIR / "data" / "vendor_groups.yaml"
DEFAULT_GROUPS = ROOT_DIR / "data" / "groups.json"
DEFAULT_OUT_MD = ROOT_DIR / "results" / "yolo_eval_v1.md"
DEFAULT_OVERLAYS_DIR = ROOT_DIR / "results" / "overlays"

MIN_STAT_THRESHOLD = 15  # Minimum boxes required for statistically stable metric reporting

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_yolo")


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
            base_yaml_path = DEFAULT_DATA_YAML

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


def load_family_and_group_mappings() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load filename -> template_family and filename -> split_group maps."""
    file_to_family: Dict[str, str] = {}
    file_to_group: Dict[str, str] = {}

    # 1. Own bills from family_map.csv
    if DEFAULT_FAMILY_MAP.exists():
        with open(DEFAULT_FAMILY_MAP, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fn = r["filename"]
                stem = Path(fn).stem
                file_to_family[fn] = r["family"]
                file_to_family[stem] = r["family"]
                file_to_family[f"{stem}.jpg"] = r["family"]
                file_to_family[f"{stem}.png"] = r["family"]
                file_to_group[fn] = r["split_group"]
                file_to_group[stem] = r["split_group"]
                file_to_group[f"{stem}.jpg"] = r["split_group"]
                file_to_group[f"{stem}.png"] = r["split_group"]

    # 2. Public invoices from groups.json + vendor_groups.yaml
    vg_map: Dict[str, Tuple[str, str]] = {}
    if DEFAULT_VENDOR_GROUPS.exists():
        with open(DEFAULT_VENDOR_GROUPS, "r", encoding="utf-8") as f:
            vg_data = yaml.safe_load(f).get("groups", {})
            for gname, ginfo in vg_data.items():
                s_group = ginfo.get("split_group", gname)
                fams = ginfo.get("template_families", [gname])
                primary_fam = fams[0] if fams else gname
                for tf in fams:
                    vg_map[tf.lower()] = (primary_fam, s_group)

    if DEFAULT_GROUPS.exists():
        with open(DEFAULT_GROUPS, "r", encoding="utf-8") as f:
            gj_data = json.load(f).get("groups", {})
            for vendor_name, filenames in gj_data.items():
                fam = vendor_name
                grp = "public_misc"
                for tf_lower, (primary_fam, s_group) in vg_map.items():
                    if tf_lower in vendor_name.lower():
                        fam = primary_fam
                        grp = s_group
                        break
                for fn in filenames:
                    stem = Path(fn).stem
                    file_to_family[fn] = fam
                    file_to_family[stem] = fam
                    file_to_family[f"{stem}.jpg"] = fam
                    file_to_family[f"{stem}.png"] = fam
                    file_to_group[fn] = grp
                    file_to_group[stem] = grp
                    file_to_group[f"{stem}.jpg"] = grp
                    file_to_group[f"{stem}.png"] = grp

    return file_to_family, file_to_group


def is_public_image(img_path: Path) -> bool:
    """Strict Privacy Guard: returns True ONLY for public Hugging Face dataset images."""
    stem = img_path.stem.lower()
    return stem.startswith("gst_")


def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def parse_ground_truth_boxes(
    label_path: Path, img_w: int, img_h: int
) -> List[Dict[str, Any]]:
    """Parse YOLO normalized format label file into pixel coordinates."""
    boxes: List[Dict[str, Any]] = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                cid = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                x1 = (xc - w / 2.0) * img_w
                y1 = (yc - h / 2.0) * img_h
                x2 = (xc + w / 2.0) * img_w
                y2 = (yc + h / 2.0) * img_h
                boxes.append({
                    "class_id": cid,
                    "bbox": [x1, y1, x2, y2],
                    "norm_bbox": [xc, yc, w, h],
                })
    return boxes


def render_prediction_overlay(
    img_path: Path,
    gt_boxes: List[Dict[str, Any]],
    pred_boxes: List[Dict[str, Any]],
    class_names: Dict[int, str],
    out_path: Path,
) -> None:
    """Render prediction overlay for public images only (GT in green, Pred in blue/cyan)."""
    # Strict privacy safeguard
    if not is_public_image(img_path):
        logger.warning(f"Privacy guard blocked rendering overlay for private image: {img_path.name}")
        return

    im = cv2.imread(str(img_path))
    if im is None:
        return

    # Draw Ground Truth (Green: (0, 200, 0))
    for b in gt_boxes:
        cid = b["class_id"]
        cname = class_names.get(cid, str(cid))
        x1, y1, x2, y2 = map(int, b["bbox"])
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            im,
            f"GT: {cname}",
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )

    # Draw Predictions (Blue/Cyan: (255, 120, 0))
    for p in pred_boxes:
        cid = p["class_id"]
        conf = p.get("conf", 0.0)
        cname = class_names.get(cid, str(cid))
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(
            im,
            f"{cname} {conf:.2f}",
            (x1, min(im.shape[0] - 5, y2 + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 140, 0),
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), im)


def evaluate_split(
    model: YOLO,
    data_yaml_path: Path,
    split: str,
    class_names: Dict[int, str],
    file_to_family: Dict[str, str],
    file_to_group: Dict[str, str],
    overlays_dir: Optional[Path] = None,
    imgsz: int = 1024,
    device: str = "cpu",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Run standard validation and detailed per-image breakdown on a split."""
    logger.info(f"Running model.val() on split: {split} (imgsz={imgsz}, device={device})")

    # 1. Official Ultralytics validation run
    val_res = model.val(
        data=str(data_yaml_path),
        split=split,
        imgsz=imgsz,
        device=device,
        batch=4,
        verbose=False,
        plots=True,
    )

    nc = len(class_names)
    ap50_arr = val_res.box.ap50 if hasattr(val_res.box, "ap50") and len(val_res.box.ap50) == nc else [0.0] * nc
    ap_arr = val_res.box.ap if hasattr(val_res.box, "ap") and len(val_res.box.ap) == nc else [0.0] * nc
    p_arr = val_res.box.p if hasattr(val_res.box, "p") and len(val_res.box.p) == nc else [0.0] * nc
    r_arr = val_res.box.r if hasattr(val_res.box, "r") and len(val_res.box.r) == nc else [0.0] * nc

    # Confusion matrix extraction
    conf_matrix = None
    if hasattr(val_res, "confusion_matrix") and hasattr(val_res.confusion_matrix, "matrix"):
        conf_matrix = val_res.confusion_matrix.matrix.tolist()

    # 2. Detailed per-image evaluation and breakdowns
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    raw_path = Path(data_cfg.get("path", ""))
    if raw_path.is_absolute():
        base_ds_dir = raw_path
    elif (ROOT_DIR / raw_path).exists():
        base_ds_dir = (ROOT_DIR / raw_path).resolve()
    else:
        base_ds_dir = (data_yaml_path.parent / raw_path).resolve()

    split_img_dir = (base_ds_dir / "images" / split).resolve()
    split_lbl_dir = (base_ds_dir / "labels" / split).resolve()

    image_files = sorted(list(split_img_dir.glob("*.jpg")) + list(split_img_dir.glob("*.png")))

    gt_counts_per_class = {cid: 0 for cid in range(nc)}
    source_stats = {
        "public": {"gt_boxes": {cid: 0 for cid in range(nc)}, "matched_boxes": {cid: 0 for cid in range(nc)}},
        "own": {"gt_boxes": {cid: 0 for cid in range(nc)}, "matched_boxes": {cid: 0 for cid in range(nc)}},
    }
    family_stats: Dict[str, Dict[str, Any]] = {}
    image_eval_records: List[Dict[str, Any]] = []

    for img_p in image_files:
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        h_img, w_img = im.shape[:2]

        lbl_p = split_lbl_dir / f"{img_p.stem}.txt"
        gt_boxes = parse_ground_truth_boxes(lbl_p, w_img, h_img)

        for b in gt_boxes:
            cid = b["class_id"]
            if cid in gt_counts_per_class:
                gt_counts_per_class[cid] += 1

        # Run inference
        preds = model.predict(source=str(img_p), imgsz=imgsz, conf=conf_thresh, device=device, verbose=False)
        pred_boxes: List[Dict[str, Any]] = []
        if preds and len(preds) > 0:
            for b in preds[0].boxes:
                cid = int(b.cls.item())
                conf = float(b.conf.item())
                xyxy = b.xyxy.cpu().numpy()[0].tolist()
                pred_boxes.append({"class_id": cid, "conf": conf, "bbox": xyxy})

        # Match GT to Preds using greedy IoU matching
        src_cat = "public" if is_public_image(img_p) else "own"
        fam_name = file_to_family.get(img_p.name, file_to_family.get(img_p.stem, "Unknown"))

        if fam_name not in family_stats:
            family_stats[fam_name] = {
                "source": src_cat,
                "total_images": 0,
                "gt_boxes": {cid: 0 for cid in range(nc)},
                "matched_boxes": {cid: 0 for cid in range(nc)},
            }
        family_stats[fam_name]["total_images"] += 1

        matched_gt: Set[int] = set()
        matched_pred: Set[int] = set()

        for g_idx, g in enumerate(gt_boxes):
            cid = g["class_id"]
            source_stats[src_cat]["gt_boxes"][cid] += 1
            family_stats[fam_name]["gt_boxes"][cid] += 1

            best_iou = 0.0
            best_p_idx = -1
            for p_idx, p in enumerate(pred_boxes):
                if p_idx in matched_pred or p["class_id"] != cid:
                    continue
                iou = compute_box_iou(g["bbox"], p["bbox"])
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_p_idx = p_idx

            if best_p_idx >= 0:
                matched_gt.add(g_idx)
                matched_pred.add(best_p_idx)
                source_stats[src_cat]["matched_boxes"][cid] += 1
                family_stats[fam_name]["matched_boxes"][cid] += 1

        fn_count = len(gt_boxes) - len(matched_gt)
        fp_count = len(pred_boxes) - len(matched_pred)
        recall = len(matched_gt) / len(gt_boxes) if gt_boxes else (1.0 if not pred_boxes else 0.0)

        image_eval_records.append({
            "filename": img_p.name,
            "source": src_cat,
            "family": fam_name,
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
            "fn_count": fn_count,
            "fp_count": fp_count,
            "recall": recall,
            "error_score": fn_count * 1.5 + fp_count,
        })

        # Save prediction overlays for PUBLIC images ONLY
        if overlays_dir and is_public_image(img_p):
            out_overlay = overlays_dir / split / img_p.name
            render_prediction_overlay(img_p, gt_boxes, pred_boxes, class_names, out_overlay)

    # Sort to find 10 worst images (highest error_score, lowest recall)
    worst_images = sorted(image_eval_records, key=lambda x: (x["error_score"], -x["recall"]), reverse=True)[:10]

    return {
        "split": split,
        "total_images": len(image_files),
        "overall_map50": float(val_res.box.map50) if hasattr(val_res.box, "map50") else 0.0,
        "overall_map": float(val_res.box.map) if hasattr(val_res.box, "map") else 0.0,
        "class_metrics": [
            {
                "class_id": cid,
                "class_name": class_names.get(cid, str(cid)),
                "gt_count": gt_counts_per_class.get(cid, 0),
                "precision": float(p_arr[cid]),
                "recall": float(r_arr[cid]),
                "ap50": float(ap50_arr[cid]),
                "ap50_95": float(ap_arr[cid]),
                "status": "VALID" if gt_counts_per_class.get(cid, 0) >= MIN_STAT_THRESHOLD else f"[TOO FEW TO MEASURE (< {MIN_STAT_THRESHOLD} boxes)]",
            }
            for cid in range(nc)
        ],
        "source_breakdown": source_stats,
        "family_breakdown": family_stats,
        "worst_images": worst_images,
        "confusion_matrix": conf_matrix,
        "val_save_dir": str(val_res.save_dir) if hasattr(val_res, "save_dir") else None,
    }


def generate_evaluation_markdown(
    results_by_split: Dict[str, Dict[str, Any]],
    weights_path: Path,
    class_names: Dict[int, str],
    output_md_path: Path,
) -> str:
    """Generate comprehensive evaluation report formatted in GitHub Flavored Markdown."""
    lines: List[str] = [
        "# YOLOv8 Field Detector Evaluation Report",
        "",
        f"- **Model Weights**: `{weights_path}`",
        f"- **Date**: `2026-11-21`",
        f"- **Statistical Threshold**: $\\ge {MIN_STAT_THRESHOLD}$ ground-truth instances per class required for reliable metric defense.",
        "- **Privacy Enforcement**: [PASS] Prediction overlays saved for PUBLIC images only; personal bills excluded from visual export.",
        "",
        "---",
    ]

    for split_name, s_res in results_by_split.items():
        lines.extend([
            f"## Split: `{split_name.upper()}` ({s_res['total_images']} Images)",
            "",
            f"- **Overall mAP@0.50**: `{s_res['overall_map50']:.4f}`",
            f"- **Overall mAP@0.50:0.95**: `{s_res['overall_map']:.4f}`",
            "",
            "### 1. Per-Class Metrics & Sample Size Viability",
            "",
            "| Class ID | Class Name | Ground Truth ($N_{\\text{gt}}$) | Precision | Recall | AP50 | AP50-95 | Measurement Status |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for cm in s_res["class_metrics"]:
            p_str = f"{cm['precision']:.4f}"
            r_str = f"{cm['recall']:.4f}"
            ap50_str = f"{cm['ap50']:.4f}"
            ap_str = f"{cm['ap50_95']:.4f}"
            status_str = f"**{cm['status']}**" if "TOO FEW" in cm["status"] else cm["status"]
            lines.append(
                f"| {cm['class_id']} | `{cm['class_name']}` | {cm['gt_count']} | {p_str} | {r_str} | {ap50_str} | {ap_str} | {status_str} |"
            )

        lines.extend([
            "",
            "### 2. Breakdown by Data Source",
            "",
            "| Source Category | Class Name | Ground Truth | Greedy Detections (IoU >= 0.50) | Empirical Recall |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for src_name in ["public", "own"]:
            s_data = s_res["source_breakdown"].get(src_name, {})
            for cid in range(len(class_names)):
                cname = class_names[cid]
                gt = s_data.get("gt_boxes", {}).get(cid, 0)
                det = s_data.get("matched_boxes", {}).get(cid, 0)
                rec = (det / gt * 100.0) if gt > 0 else 0.0
                lines.append(f"| `{src_name}` | `{cname}` | {gt} | {det} | {rec:.1f}% |")

        lines.extend([
            "",
            "### 3. Breakdown by Template Family",
            "",
            "| Template Family | Source | Total Images | Total GT Boxes | Matched Detections | Mean Box Recall |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for fam_name, fdata in sorted(s_res["family_breakdown"].items()):
            tot_gt = sum(fdata["gt_boxes"].values())
            tot_det = sum(fdata["matched_boxes"].values())
            m_rec = (tot_det / tot_gt * 100.0) if tot_gt > 0 else 0.0
            lines.append(
                f"| `{fam_name}` | {fdata['source']} | {fdata['total_images']} | {tot_gt} | {tot_det} | {m_rec:.1f}% |"
            )

        lines.extend([
            "",
            f"### 4. Ten Worst Performing Images in `{split_name.upper()}` (by Detection Errors)",
            "",
            "| Rank | File Name | Source | Template Family | GT Boxes | Predictions | False Negatives | False Positives | Recall |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for rank, w in enumerate(s_res["worst_images"], start=1):
            lines.append(
                f"| {rank} | `{w['filename']}` | {w['source']} | `{w['family']}` | {w['gt_count']} | {w['pred_count']} | {w['fn_count']} | {w['fp_count']} | {w['recall']*100.0:.1f}% |"
            )

        # Confusion matrix section
        if s_res.get("confusion_matrix"):
            cm_mat = s_res["confusion_matrix"]
            col_headers = [class_names.get(i, f"c{i}") for i in range(len(class_names))] + ["background"]
            lines.extend([
                "",
                f"### 5. Confusion Matrix ({split_name.upper()})",
                "",
                "Rows represent True Classes, Columns represent Predicted Classes:",
                "",
                "| True \\ Pred | " + " | ".join(col_headers) + " |",
                "| :--- | " + " | ".join([":---:"] * len(col_headers)) + " |",
            ])
            for r_idx, row in enumerate(cm_mat):
                row_label = col_headers[r_idx] if r_idx < len(col_headers) else f"Row {r_idx}"
                lines.append(f"| **{row_label}** | " + " | ".join(str(int(val)) for val in row) + " |")

        lines.extend(["", "---", ""])

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    full_md = "\n".join(lines)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(full_md)

    logger.info(f"Evaluation report generated at: {output_md_path}")
    return full_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO weights on val and test splits")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(ROOT_DIR / "yolov8n.pt"),
        help="Path to YOLO weights .pt file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_YAML),
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Path to dataset root folder (default: data/yolo_v1)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Splits to evaluate (default: val test)",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu, 0)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT_MD),
        help="Output markdown file path",
    )
    parser.add_argument(
        "--overlays-dir",
        type=str,
        default=str(DEFAULT_OVERLAYS_DIR),
        help="Directory to save public prediction overlays",
    )
    args = parser.parse_args()

    weights_p = Path(args.weights)
    if not weights_p.exists():
        # Fallback to smoke weights if available
        smoke_weights = ROOT_DIR / "runs" / "train_smoke" / "smoke_run" / "weights" / "best.pt"
        if smoke_weights.exists():
            weights_p = smoke_weights
            logger.info(f"Specified weights not found; falling back to smoke weights: {weights_p}")
        else:
            weights_p = ROOT_DIR / "yolov8n.pt"
            logger.info(f"Using base pretrained weights: {weights_p}")

    # Generate runtime data.yaml with portable path
    data_root_p = Path(args.data_root).resolve() if args.data_root else (
        DEFAULT_DATA_ROOT.resolve() if DEFAULT_DATA_ROOT.exists() else Path("data/yolo_v1").resolve()
    )
    base_data_yaml_p = Path(args.data) if args.data and Path(args.data).exists() else (data_root_p / "data.yaml")
    data_yaml_p = generate_runtime_data_yaml(data_root_p, base_yaml_path=base_data_yaml_p)

    with open(data_yaml_p, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Resolve class names
    raw_names = data_cfg.get("names", {})
    if isinstance(raw_names, list):
        class_names = {i: name for i, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        class_names = {int(k): str(v) for k, v in raw_names.items()}
    else:
        class_names = {0: "vendor", 1: "invoice_no", 2: "date", 3: "gstin", 4: "table", 5: "total"}

    file_to_family, file_to_group = load_family_and_group_mappings()

    logger.info(f"Loading YOLO model weights: {weights_p}")
    model = YOLO(str(weights_p))

    overlays_dir = Path(args.overlays_dir)

    results_by_split = {}
    for s in args.splits:
        results_by_split[s] = evaluate_split(
            model=model,
            data_yaml_path=data_yaml_p,
            split=s,
            class_names=class_names,
            file_to_family=file_to_family,
            file_to_group=file_to_group,
            overlays_dir=overlays_dir,
            imgsz=args.imgsz,
            device=args.device,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
        )

    out_md = Path(args.out)
    generate_evaluation_markdown(
        results_by_split=results_by_split,
        weights_path=weights_p,
        class_names=class_names,
        output_md_path=out_md,
    )


if __name__ == "__main__":
    main()
