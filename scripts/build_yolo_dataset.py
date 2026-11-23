"""Assemble YOLOv8/YOLOv5 formatted dataset and compute label statistics.

Reads reviewed labels from data/annotation/exported/batch_*/ and matching images,
assigns them to train/val/test using data/splits_v2.json or data/splits_v3.json,
remaps class IDs to 6 classes (vendor, invoice_no, date, gstin, table, total) while
dropping qr and stamp_signature (per split), identifies missing label files vs confirmed
empty images, writes data/yolo_v1 or data/yolo_v2, validates labels for warnings
(tiny or duplicate boxes), generates dataset_stats markdown reports, and runs
strict cross-split leakage checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import zipfile

import cv2
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.validate_labels import (
    BBox,
    IOU_DUPLICATE_THRESH,
    MIN_BOX_DIM,
    compute_iou,
)

TARGET_CLASSES = [
    "vendor",       # 0
    "invoice_no",   # 1
    "date",         # 2
    "gstin",        # 3
    "table",        # 4
    "total",        # 5
]
TARGET_CLASS_TO_ID = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
DROPPED_CLASSES = ["qr", "stamp_signature"]

DEFAULT_SPLITS_PATH = ROOT_DIR / "data" / "splits_v2.json"
DEFAULT_VENDOR_GROUPS_PATH = ROOT_DIR / "data" / "vendor_groups.yaml"
DEFAULT_FAMILY_MAP_PATH = ROOT_DIR / "data" / "family_map.csv"
DEFAULT_GROUPS_PATH = ROOT_DIR / "data" / "groups.json"
DEFAULT_SYNTH_MANIFEST_PATH = ROOT_DIR / "data" / "gst_synth_manifest.csv"
DEFAULT_ANNOTATION_DIR = ROOT_DIR / "data" / "annotation"
DEFAULT_EXPORTED_DIR = ROOT_DIR / "data" / "annotation" / "exported"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "yolo_v1"
DEFAULT_STATS_PATH = ROOT_DIR / "results" / "dataset_stats_v1.md"


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_split_mapping(splits_path: Path) -> Dict[str, str]:
    """Load filename -> split ('train', 'val', 'test') mapping from splits JSON."""
    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_to_split: Dict[str, str] = {}
    combined = data.get("combined_clean_splits", {})
    for split_name in ["train", "val", "test"]:
        for fn in combined.get(split_name, []):
            file_to_split[fn] = split_name
            stem = Path(fn).stem
            file_to_split[f"{stem}.jpg"] = split_name
            file_to_split[f"{stem}.png"] = split_name

    # Also map synthetic splits if present
    synth = data.get("public_synthetic_splits", {})
    for split_name in ["train", "val", "test"]:
        for fn in synth.get(split_name, []):
            file_to_split[fn] = split_name
            stem = Path(fn).stem
            file_to_split[f"{stem}.jpg"] = split_name

    gst_synth = data.get("gst_synth_splits", {})
    for split_name in ["train", "val", "test"]:
        for fn in gst_synth.get(split_name, []):
            file_to_split[fn] = split_name
            stem = Path(fn).stem
            file_to_split[f"{stem}.jpg"] = split_name
            file_to_split[f"{stem}.png"] = split_name

    return file_to_split


def build_filename_to_split_group_map() -> Dict[str, str]:
    """Map every known filename (public, own, synth) to its split_group for leakage checking."""
    file_to_group: Dict[str, str] = {}

    # 1. Own bills from family_map.csv
    if DEFAULT_FAMILY_MAP_PATH.exists():
        with open(DEFAULT_FAMILY_MAP_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                file_to_group[r["filename"]] = r["split_group"]
                file_to_group[Path(r["filename"]).stem] = r["split_group"]

    # 2. Public invoices from groups.json + vendor_groups.yaml
    vg_map: Dict[str, str] = {}
    if DEFAULT_VENDOR_GROUPS_PATH.exists():
        with open(DEFAULT_VENDOR_GROUPS_PATH, "r", encoding="utf-8") as f:
            vg_data = yaml.safe_load(f).get("groups", {})
            for gname, ginfo in vg_data.items():
                s_group = ginfo.get("split_group", gname)
                for tf in ginfo.get("template_families", []):
                    vg_map[tf.lower()] = s_group

    if DEFAULT_GROUPS_PATH.exists():
        with open(DEFAULT_GROUPS_PATH, "r", encoding="utf-8") as f:
            gj_data = json.load(f).get("groups", {})
            for vendor_name, filenames in gj_data.items():
                matched_sgroup = "public_misc"
                for tf_lower, s_group in vg_map.items():
                    if tf_lower in vendor_name.lower():
                        matched_sgroup = s_group
                        break
                for fn in filenames:
                    file_to_group[fn] = matched_sgroup
                    file_to_group[Path(fn).stem] = matched_sgroup

    # 3. Synthetic layout invoices from gst_synth_manifest.csv
    if DEFAULT_SYNTH_MANIFEST_PATH.exists():
        with open(DEFAULT_SYNTH_MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fn = r["filename"]
                sgroup = f"gstsyn_{r['layout']}"
                file_to_group[fn] = sgroup
                file_to_group[Path(fn).stem] = sgroup

    return file_to_group


def load_synth_layout_map() -> Dict[str, str]:
    """Map gstsyn filename/stem to layout name (classic, handwritten, minimal, thermal)."""
    layout_map: Dict[str, str] = {}
    if DEFAULT_SYNTH_MANIFEST_PATH.exists():
        with open(DEFAULT_SYNTH_MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fn = r["filename"]
                ly = r["layout"]
                layout_map[fn] = ly
                layout_map[Path(fn).stem] = ly
    return layout_map


def resolve_batch_labels_order(export_dir: Path, batch_dir: Path) -> List[str]:
    """Resolve makesense export label order by reading label files inside export/batch folder."""
    # 1. Check direct labels.txt or classes.txt in export_dir
    for cand in ["labels.txt", "classes.txt"]:
        p = export_dir / cand
        if p.exists():
            lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                return lines

    # 2. Check inside any .zip in export_dir
    for zf in export_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zf) as z:
                for cand in ["labels.txt", "classes.txt"]:
                    if cand in z.namelist():
                        content = z.read(cand).decode("utf-8", errors="replace")
                        lines = [line.strip() for line in content.splitlines() if line.strip()]
                        if lines:
                            return lines
        except Exception:
            pass

    # 3. Check batch_dir / yolo_import / labels.txt
    cand_import = batch_dir / "yolo_import" / "labels.txt"
    if cand_import.exists():
        lines = [line.strip() for line in cand_import.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            return lines

    # 4. Check batch_dir / classes.txt
    cand_batch = batch_dir / "classes.txt"
    if cand_batch.exists():
        lines = [line.strip() for line in cand_batch.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            return lines

    # 5. Global fallback
    global_classes = ROOT_DIR / "data" / "classes.txt"
    if global_classes.exists():
        return [line.strip() for line in global_classes.read_text(encoding="utf-8").splitlines() if line.strip()]

    return ["vendor", "invoice_no", "date", "gstin", "table", "total", "qr", "stamp_signature"]


def find_matching_image(stem: str, batch_dir: Path) -> Optional[Path]:
    """Find image file corresponding to stem."""
    for ext in [".jpg", ".png", ".jpeg"]:
        cand = batch_dir / f"{stem}{ext}"
        if cand.exists():
            return cand

    # Fallbacks across public, own, and synth raw storage
    for parent in [
        ROOT_DIR / "data" / "raw" / "indian_gst",
        ROOT_DIR / "data" / "raw" / "own",
        ROOT_DIR / "data" / "raw" / "gst_synth",
        ROOT_DIR / "data" / "processed" / "clean_upscaled",
    ]:
        for ext in [".jpg", ".png", ".jpeg"]:
            cand = parent / f"{stem}{ext}"
            if cand.exists():
                return cand

    return None


def get_image_dimensions(img_path: Path) -> Tuple[int, int]:
    """Get (width, height) of an image file."""
    im = cv2.imread(str(img_path))
    if im is not None:
        h, w = im.shape[:2]
        return w, h
    return 1000, 1000


def check_label_file_warnings(
    lines: List[str],
    filename: str,
    export_classes: List[str],
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    """Check parsed bounding boxes for tiny boxes or duplicate boxes."""
    warnings: List[Dict[str, Any]] = []
    boxes: List[Tuple[BBox, str]] = []

    for line_idx, line in enumerate(lines, 1):
        parts = line.strip().split()
        if len(parts) >= 5:
            cid = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            cname = export_classes[cid] if cid < len(export_classes) else str(cid)
            b = BBox(class_id=cid, xc=xc, yc=yc, w=w, h=h, line_no=line_idx)
            boxes.append((b, cname))

            # 1. Tiny box warning
            if w < MIN_BOX_DIM or h < MIN_BOX_DIM:
                w_px = w * img_w
                h_px = h * img_h
                warnings.append({
                    "filename": filename,
                    "class": cname,
                    "line": line_idx,
                    "box_px": f"{w_px:.1f} x {h_px:.1f} px",
                    "reason": f"tiny box (norm_w={w:.4f}, norm_h={h:.4f} < {MIN_BOX_DIM})",
                })

    # 2. Duplicate box check (same class, IoU >= 0.90)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1, cname1 = boxes[i]
            b2, cname2 = boxes[j]
            if b1.class_id == b2.class_id:
                iou = compute_iou(b1, b2)
                if iou >= IOU_DUPLICATE_THRESH:
                    w_px = b1.w * img_w
                    h_px = b1.h * img_h
                    warnings.append({
                        "filename": filename,
                        "class": cname1,
                        "line": f"{b1.line_no} & {b2.line_no}",
                        "box_px": f"{w_px:.1f} x {h_px:.1f} px",
                        "reason": f"duplicate box with line {b2.line_no} (IoU={iou:.3f} >= {IOU_DUPLICATE_THRESH})",
                    })

    return warnings


def assemble_yolo_dataset(
    annotation_dir: Path,
    exported_dir: Path,
    splits_path: Path,
    output_dir: Path,
    with_synth: bool = False,
    file_to_group: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Assemble YOLO dataset into output_dir and gather detailed statistics."""
    file_to_split = load_split_mapping(splits_path)
    if file_to_group is None:
        file_to_group = build_filename_to_split_group_map()
    synth_layout_map = load_synth_layout_map()

    # Create target directories
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Track missing batches & missing labels
    all_batch_dirs = sorted([d for d in annotation_dir.glob("batch_*") if d.is_dir()])
    missing_batches: List[str] = []
    missing_labels_images: List[str] = []
    processed_batches: List[str] = []

    # Layout categories for synthetic multi-layout bills
    layouts = ["classic", "handwritten", "minimal", "thermal"]

    # Statistics accumulators
    stats: Dict[str, Any] = {
        "dataset_name": output_dir.name,
        "splits_path": str(splits_path.relative_to(ROOT_DIR) if splits_path.is_relative_to(ROOT_DIR) else splits_path),
        "splits": {s: {"total_images": 0, "zero_box_images": 0, "class_counts": {c: 0 for c in TARGET_CLASSES}} for s in ["train", "val", "test"]},
        "sources": {
            "public": {s: {"total_images": 0, "zero_box_images": 0, "class_counts": {c: 0 for c in TARGET_CLASSES}} for s in ["train", "val", "test"]},
            "own": {s: {"total_images": 0, "zero_box_images": 0, "class_counts": {c: 0 for c in TARGET_CLASSES}} for s in ["train", "val", "test"]},
            "synthetic_layout": {s: {"total_images": 0, "zero_box_images": 0, "class_counts": {c: 0 for c in TARGET_CLASSES}} for s in ["train", "val", "test"]},
            "synthetic": {s: {"total_images": 0, "zero_box_images": 0, "class_counts": {c: 0 for c in TARGET_CLASSES}} for s in ["train", "val", "test"]},
        },
        "synthetic_layouts": {ly: {"total_images": 0, "class_counts": {c: 0 for c in TARGET_CLASSES}} for ly in layouts},
        "dropped_boxes_per_split": {s: {c: 0 for c in DROPPED_CLASSES} for s in ["train", "val", "test"]},
        "total_dropped_per_split": {s: 0 for s in ["train", "val", "test"]},
        "refused_images": [],
        "missing_batches": [],
        "missing_labels_images": [],
        "validation_warnings": [],
        "leakage": {
            "group_leakage_violations": [],
            "hash_leakage_violations": [],
            "groups_checked": 0,
            "hashes_checked": 0,
        },
    }

    # Leakage check trackers
    group_to_splits: Dict[str, Set[str]] = {}
    hash_to_split: Dict[str, Set[str]] = {}

    synth_photo_keywords = ["_mild_", "_medium_", "_severe_", "_hard_"]

    for batch_dir in all_batch_dirs:
        b_name = batch_dir.name
        exp_dir = exported_dir / b_name
        if not exp_dir.exists():
            missing_batches.append(b_name)
            continue

        # Check for images in batch folder that lack an exported label file (MISSING LABELS)
        batch_images = [f for f in batch_dir.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg")]
        valid_label_files: List[Tuple[Path, Path]] = []

        for b_img in batch_images:
            # Strictly exclude synthetic photo copies
            if any(k in b_img.name for k in synth_photo_keywords):
                continue

            txt_cand = exp_dir / f"{b_img.stem}.txt"
            if not txt_cand.exists():
                missing_labels_images.append(f"{b_img.name} (in {b_name})")
            else:
                valid_label_files.append((txt_cand, b_img))

        if not valid_label_files:
            missing_batches.append(b_name)
            continue

        processed_batches.append(b_name)
        export_classes = resolve_batch_labels_order(exp_dir, batch_dir)

        for lf, img_path in valid_label_files:
            stem = lf.stem

            # Determine split
            target_split = file_to_split.get(img_path.name) or file_to_split.get(f"{stem}.jpg") or file_to_split.get(f"{stem}.png")
            if not target_split:
                stats["refused_images"].append(f"{img_path.name} (no split in {splits_path.name})")
                continue

            # Determine source category
            if stem.startswith("gst_"):
                src_cat = "public"
            elif stem.startswith("gstsyn_"):
                src_cat = "synthetic_layout"
            else:
                src_cat = "own"

            # Parse lines
            with open(lf, "r", encoding="utf-8") as f_in:
                raw_lines = [l.strip() for l in f_in if l.strip() and not l.startswith("#")]

            # Run validation warnings check on original exported boxes
            img_w, img_h = get_image_dimensions(img_path)
            file_warnings = check_label_file_warnings(raw_lines, lf.name, export_classes, img_w, img_h)
            if file_warnings:
                stats["validation_warnings"].extend(file_warnings)

            # Remap boxes and drop non-target classes (qr, stamp_signature)
            remapped_lines: List[str] = []
            for line in raw_lines:
                parts = line.split()
                if len(parts) >= 5:
                    orig_id = int(parts[0])
                    orig_name = export_classes[orig_id] if orig_id < len(export_classes) else "unknown"
                    if orig_name in TARGET_CLASS_TO_ID:
                        new_id = TARGET_CLASS_TO_ID[orig_name]
                        remapped_lines.append(f"{new_id} {' '.join(parts[1:5])}\n")
                        stats["splits"][target_split]["class_counts"][orig_name] += 1
                        stats["sources"][src_cat][target_split]["class_counts"][orig_name] += 1
                        if src_cat == "synthetic_layout":
                            ly = synth_layout_map.get(img_path.name, synth_layout_map.get(stem, "classic"))
                            stats["synthetic_layouts"][ly]["class_counts"][orig_name] += 1
                    else:
                        # Dropped non-target box counted per split
                        if orig_name in stats["dropped_boxes_per_split"][target_split]:
                            stats["dropped_boxes_per_split"][target_split][orig_name] += 1
                        else:
                            stats["dropped_boxes_per_split"][target_split][orig_name] = 1
                        stats["total_dropped_per_split"][target_split] += 1

            # Copy image
            dst_img = output_dir / "images" / target_split / img_path.name
            shutil.copy2(img_path, dst_img)

            # Write remapped label file (empty .txt if 0 objects confirmed)
            dst_lbl = output_dir / "labels" / target_split / f"{stem}.txt"
            with open(dst_lbl, "w", encoding="utf-8") as f_out:
                f_out.writelines(remapped_lines)

            # Update image counts
            stats["splits"][target_split]["total_images"] += 1
            stats["sources"][src_cat][target_split]["total_images"] += 1
            if src_cat == "synthetic_layout":
                ly = synth_layout_map.get(img_path.name, synth_layout_map.get(stem, "classic"))
                stats["synthetic_layouts"][ly]["total_images"] += 1

            if len(remapped_lines) == 0:
                stats["splits"][target_split]["zero_box_images"] += 1
                stats["sources"][src_cat][target_split]["zero_box_images"] += 1

            # Track for leakage
            sgroup = file_to_group.get(img_path.name, file_to_group.get(stem, "unknown_group"))
            group_to_splits.setdefault(sgroup, set()).add(target_split)

            f_hash = compute_file_sha256(img_path)
            hash_to_split.setdefault(f_hash, set()).add(target_split)

    # Legacy synthetic photos inclusion (if requested)
    if with_synth:
        synth_manifest_path = ROOT_DIR / "data" / "processed" / "synth_photos" / "manifest.json"
        if synth_manifest_path.exists():
            with open(synth_manifest_path, "r", encoding="utf-8") as f_m:
                manifest = json.load(f_m)
            for rec in manifest.get("records", []):
                rel_p = rec.get("relative_path")
                if not rel_p:
                    continue
                s_img_p = ROOT_DIR / rel_p
                s_txt_p = s_img_p.with_suffix(".txt")
                if s_img_p.exists() and s_txt_p.exists():
                    target_split = "train"
                    remapped_lines = []
                    with open(s_txt_p, "r", encoding="utf-8") as f_in:
                        for line in f_in:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            parts = line.split()
                            if len(parts) >= 5:
                                orig_id = int(parts[0])
                                if 0 <= orig_id < len(TARGET_CLASSES):
                                    cls_name = TARGET_CLASSES[orig_id]
                                    remapped_lines.append(f"{orig_id} {' '.join(parts[1:5])}\n")
                                    stats["splits"][target_split]["class_counts"][cls_name] += 1
                                    stats["sources"]["synthetic"][target_split]["class_counts"][cls_name] += 1
                                else:
                                    stats["total_dropped_per_split"][target_split] += 1

                    dst_img = output_dir / "images" / target_split / s_img_p.name
                    shutil.copy2(s_img_p, dst_img)
                    dst_lbl = output_dir / "labels" / target_split / f"{s_img_p.stem}.txt"
                    with open(dst_lbl, "w", encoding="utf-8") as f_out:
                        f_out.writelines(remapped_lines)

                    stats["splits"][target_split]["total_images"] += 1
                    stats["sources"]["synthetic"][target_split]["total_images"] += 1
                    if len(remapped_lines) == 0:
                        stats["splits"][target_split]["zero_box_images"] += 1
                        stats["sources"]["synthetic"][target_split]["zero_box_images"] += 1

    # Write data.yaml with output-specific path
    rel_path_str = f"data/{output_dir.name}"
    yaml_content = {
        "path": rel_path_str,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(TARGET_CLASSES),
        "names": {i: name for i, name in enumerate(TARGET_CLASSES)},
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f_yaml:
        yaml.dump(yaml_content, f_yaml, sort_keys=False)

    # Perform leakage checks
    stats["leakage"]["groups_checked"] = len(group_to_splits)
    for g, sps in group_to_splits.items():
        if len(sps) > 1:
            stats["leakage"]["group_leakage_violations"].append((g, sorted(list(sps))))

    stats["leakage"]["hashes_checked"] = len(hash_to_split)
    for h, sps in hash_to_split.items():
        if len(sps) > 1:
            stats["leakage"]["hash_leakage_violations"].append(sorted(list(sps)))

    stats["missing_batches"] = missing_batches
    stats["missing_labels_images"] = missing_labels_images
    stats["processed_batches"] = processed_batches
    return stats


def generate_stats_markdown(
    stats: Dict[str, Any], output_path: Path, with_synth: bool = False
) -> str:
    """Generate markdown dataset statistics report with small-class warnings."""
    ds_name = stats.get("dataset_name", "yolo_v1")
    lines: List[str] = [
        f"# YOLO Dataset Statistics: `data/{ds_name}`",
        "",
        f"- **Dataset Version**: `{ds_name}`",
        f"- **Splits Configuration**: `{stats.get('splits_path', 'data/splits_v2.json')}`",
        f"- **Includes Synthetic Photos in Train**: `{with_synth}`",
        f"- **Target Classes**: `{', '.join(TARGET_CLASSES)}` (nc=6)",
        f"- **Processed Batches**: `{', '.join(stats['processed_batches'])}`",
    ]

    if stats["missing_batches"]:
        lines.append(f"- **Missing / Unexported Batches**: `{', '.join(stats['missing_batches'])}`")
    else:
        lines.append("- **Missing / Unexported Batches**: None (all batches exported)")

    if stats["missing_labels_images"]:
        lines.append(f"- **MISSING LABELS Images (Excluded)**: {len(stats['missing_labels_images'])}")
        for ml in stats["missing_labels_images"]:
            lines.append(f"  - `{ml}`")
    else:
        lines.append("- **MISSING LABELS Images**: None (all images in processed batches have exported labels)")

    lines.extend([
        "",
        "---",
        "",
        "## 1. Overall Image Counts by Split",
        "",
        "| Split | Public Images | Own Bills | Synthetic Layout | Total Images | Images with Zero Boxes |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    tot_pub = 0
    tot_own = 0
    tot_syn_ly = 0
    tot_all = 0
    tot_zeros = 0

    for s in ["train", "val", "test"]:
        p_c = stats["sources"]["public"][s]["total_images"]
        o_c = stats["sources"]["own"][s]["total_images"]
        sl_c = stats["sources"]["synthetic_layout"][s]["total_images"]
        t_c = stats["splits"][s]["total_images"]
        z_c = stats["splits"][s]["zero_box_images"]

        tot_pub += p_c
        tot_own += o_c
        tot_syn_ly += sl_c
        tot_all += t_c
        tot_zeros += z_c

        lines.append(f"| **{s}** | {p_c} | {o_c} | {sl_c} | **{t_c}** | {z_c} |")

    lines.append(f"| **TOTAL** | **{tot_pub}** | **{tot_own}** | **{tot_syn_ly}** | **{tot_all}** | **{tot_zeros}** |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Bounding Box Counts per Class across Splits")
    lines.append("")
    lines.append("| Class ID | Class Name | Train Count | Val Count | Test Count | Total Count | Measurement Status |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    tot_train_boxes = 0
    tot_val_boxes = 0
    tot_test_boxes = 0

    for idx, cname in enumerate(TARGET_CLASSES):
        tr = stats["splits"]["train"]["class_counts"][cname]
        va = stats["splits"]["val"]["class_counts"][cname]
        te = stats["splits"]["test"]["class_counts"][cname]
        tot = tr + va + te

        tot_train_boxes += tr
        tot_val_boxes += va
        tot_test_boxes += te

        status_flags: List[str] = []
        if va < 15:
            status_flags.append(f"Val too few to measure ({va} < 15)")
        if te < 15:
            status_flags.append(f"Test too few to measure ({te} < 15)")

        status_str = f"**[TOO FEW TO MEASURE: {', '.join(status_flags)}]**" if status_flags else "PASS (>= 15)"
        lines.append(f"| {idx} | `{cname}` | {tr} | {va} | {te} | **{tot}** | {status_str} |")

    lines.append(f"| - | **TOTAL BOXES** | **{tot_train_boxes}** | **{tot_val_boxes}** | **{tot_test_boxes}** | **{tot_train_boxes + tot_val_boxes + tot_test_boxes}** | - |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Class Breakdown by Source")
    lines.append("")
    lines.append("### 3.1 Public Clean Images (`data/raw/indian_gst`)")
    lines.append("| Class Name | Train | Val | Test | Total |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for cname in TARGET_CLASSES:
        p_tr = stats["sources"]["public"]["train"]["class_counts"][cname]
        p_va = stats["sources"]["public"]["val"]["class_counts"][cname]
        p_te = stats["sources"]["public"]["test"]["class_counts"][cname]
        lines.append(f"| `{cname}` | {p_tr} | {p_va} | {p_te} | **{p_tr + p_va + p_te}** |")

    lines.append("")
    lines.append("### 3.2 Own Personal Bills (`data/raw/own` - Counts Only)")
    lines.append("| Class Name | Train | Val | Test | Total |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for cname in TARGET_CLASSES:
        o_tr = stats["sources"]["own"]["train"]["class_counts"][cname]
        o_va = stats["sources"]["own"]["val"]["class_counts"][cname]
        o_te = stats["sources"]["own"]["test"]["class_counts"][cname]
        lines.append(f"| `{cname}` | {o_tr} | {o_va} | {o_te} | **{o_tr + o_va + o_te}** |")

    if tot_syn_ly > 0:
        lines.append("")
        lines.append(f"### 3.3 Synthetic Multi-Layout Invoices (`data/raw/gst_synth` - {tot_syn_ly} Images in Train Only)")
        lines.append("| Class Name | Train | Val | Test | Total |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for cname in TARGET_CLASSES:
            s_tr = stats["sources"]["synthetic_layout"]["train"]["class_counts"][cname]
            lines.append(f"| `{cname}` | {s_tr} | 0 | 0 | **{s_tr}** |")

        lines.append("")
        lines.append("### 3.4 Synthetic Multi-Layout Breakdown by Layout (Train Split Only)")
        layouts = ["classic", "handwritten", "minimal", "thermal"]
        header = "| Class Name | " + " | ".join(f"{ly.capitalize()}" for ly in layouts) + " | Total |"
        sep = "| :--- | " + " | ".join(":---" for _ in layouts) + " | :--- |"
        lines.append(header)
        lines.append(sep)
        for cname in TARGET_CLASSES:
            row_items = [f"`{cname}`"]
            row_tot = 0
            for ly in layouts:
                cnt = stats["synthetic_layouts"][ly]["class_counts"][cname]
                row_tot += cnt
                row_items.append(str(cnt))
            row_items.append(f"**{row_tot}**")
            lines.append("| " + " | ".join(row_items) + " |")

        img_counts_row = ["**Total Images**"]
        for ly in layouts:
            img_counts_row.append(str(stats["synthetic_layouts"][ly]["total_images"]))
        img_counts_row.append(f"**{tot_syn_ly}**")
        lines.append("| " + " | ".join(img_counts_row) + " |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Dropped Classes (per Split)")
    lines.append("")
    lines.append("| Dropped Class | Train Dropped | Val Dropped | Test Dropped | Total Dropped |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for dcls in DROPPED_CLASSES:
        d_tr = stats["dropped_boxes_per_split"]["train"].get(dcls, 0)
        d_va = stats["dropped_boxes_per_split"]["val"].get(dcls, 0)
        d_te = stats["dropped_boxes_per_split"]["test"].get(dcls, 0)
        lines.append(f"| `{dcls}` | {d_tr} | {d_va} | {d_te} | **{d_tr + d_va + d_te}** |")

    tot_d_tr = stats["total_dropped_per_split"]["train"]
    tot_d_va = stats["total_dropped_per_split"]["val"]
    tot_d_te = stats["total_dropped_per_split"]["test"]
    lines.append(f"| **TOTAL DROPPED** | **{tot_d_tr}** | **{tot_d_va}** | **{tot_d_te}** | **{tot_d_tr + tot_d_va + tot_d_te}** |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Validation Warnings (Tiny / Duplicate Boxes)")
    lines.append("")
    warns = stats["validation_warnings"]
    if warns:
        lines.append(f"- **Total Warnings Flagged**: {len(warns)}")
        lines.append("")
        lines.append("| File Name | Class | Box Size (pixels) | Reason |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for w in warns:
            lines.append(f"| `{w['filename']}` | `{w['class']}` | {w['box_px']} | {w['reason']} |")
    else:
        lines.append("- **Verification Status**: [PASS] Zero validation warnings (zero tiny boxes < 0.005, zero duplicate boxes IoU >= 0.90) across all label files.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Dataset Leakage & Integrity Verification")
    lines.append("")
    leak = stats["leakage"]
    lines.append(f"- **Split Groups Verified**: {leak['groups_checked']}")
    lines.append(f"- **Cross-Split Group Leakage Violations**: {len(leak['group_leakage_violations'])}")
    if leak["group_leakage_violations"]:
        for g, sps in leak["group_leakage_violations"]:
            lines.append(f"  - [VIOLATION] split_group `{g}` appears in multiple splits: {sps}")
    else:
        lines.append("  - [PASS] Zero split_group overlap across train, val, and test.")

    lines.append(f"- **Image File Hashes Verified**: {leak['hashes_checked']}")
    lines.append(f"- **Cross-Split Hash Duplicate Violations**: {len(leak['hash_leakage_violations'])}")
    if leak["hash_leakage_violations"]:
        for sps in leak["hash_leakage_violations"]:
            lines.append(f"  - [VIOLATION] identical image content appears in splits: {sps}")
    else:
        lines.append("  - [PASS] Zero duplicate image content across train, val, and test.")

    content = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(content)

    return content


def verify_val_and_test_parity(yolo_dir: Path, ref_dir: Path = ROOT_DIR / "data" / "yolo_v1") -> None:
    """Verify that val and test image sets, label sets, and hashes are 100% identical to reference."""
    if not ref_dir.exists():
        print(f"Reference dataset {ref_dir} not found; skipping parity verification.")
        return

    ref_desc = str(ref_dir.relative_to(ROOT_DIR)) if ref_dir.is_relative_to(ROOT_DIR) else ref_dir.name
    print("\n" + "=" * 70)
    print(f"VERIFYING VAL AND TEST PARITY AGAINST REFERENCE ({ref_desc}):")
    print("=" * 70)

    for split in ["val", "test"]:
        ref_split_imgs = sorted(list((ref_dir / "images" / split).glob("*")))
        tgt_split_imgs = sorted(list((yolo_dir / "images" / split).glob("*")))

        ref_names = [p.name for p in ref_split_imgs]
        tgt_names = [p.name for p in tgt_split_imgs]

        assert set(ref_names) == set(tgt_names), (
            f"Parity mismatch in {split} images: {set(ref_names) ^ set(tgt_names)}"
        )

        for p_tgt in tgt_split_imgs:
            p_ref = ref_dir / "images" / split / p_tgt.name
            ref_hash = compute_file_sha256(p_ref)
            tgt_hash = compute_file_sha256(p_tgt)
            assert ref_hash == tgt_hash, f"Hash mismatch for {split}/{p_tgt.name}: {ref_hash} vs {tgt_hash}"

        print(f"  [PASS] {split:<4} images: exactly {len(tgt_names)} images with identical file names and SHA-256 hashes.")

        # Also verify label files and label hashes for val split if labels exist
        if split == "val" and (ref_dir / "labels" / split).exists() and (yolo_dir / "labels" / split).exists():
            ref_split_lbls = sorted(list((ref_dir / "labels" / split).glob("*.txt")))
            tgt_split_lbls = sorted(list((yolo_dir / "labels" / split).glob("*.txt")))

            ref_lbl_names = [p.name for p in ref_split_lbls]
            tgt_lbl_names = [p.name for p in tgt_split_lbls]

            assert set(ref_lbl_names) == set(tgt_lbl_names), (
                f"Parity mismatch in {split} labels: {set(ref_lbl_names) ^ set(tgt_lbl_names)}"
            )

            for p_tgt in tgt_split_lbls:
                p_ref = ref_dir / "labels" / split / p_tgt.name
                ref_hash = compute_file_sha256(p_ref)
                tgt_hash = compute_file_sha256(p_tgt)
                assert ref_hash == tgt_hash, f"Label hash mismatch for {split}/{p_tgt.name}: {ref_hash} vs {tgt_hash}"

            print(f"  [PASS] {split:<4} labels: exactly {len(tgt_lbl_names)} label files with identical SHA-256 hashes.")

    # Verify no empty label files in train and val, and every train image has a matching label file
    if (yolo_dir / "labels" / "train").exists() and (yolo_dir / "labels" / "val").exists():
        print("-" * 70)
        for split in ["train", "val"]:
            img_stems = {p.stem for p in (yolo_dir / "images" / split).glob("*")}
            lbl_dir = yolo_dir / "labels" / split
            lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}

            missing_lbls = img_stems - lbl_stems
            assert len(missing_lbls) == 0, f"Missing label files in {split}: {missing_lbls}"

            empty_lbls = [p.name for p in lbl_dir.glob("*.txt") if p.stat().st_size == 0 or not p.read_text(encoding="utf-8").strip()]
            assert len(empty_lbls) == 0, f"Empty label files found in {split}: {empty_lbls}"

            print(f"  [PASS] {split:<5}: all {len(img_stems)} images have non-empty matching label files (0 empty labels).")


    print("=" * 70 + "\n")



def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble YOLOv8/YOLOv5 dataset from reviewed annotation batches.")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR, help="Path to data/annotation")
    parser.add_argument("--exported-dir", type=Path, default=DEFAULT_EXPORTED_DIR, help="Path to data/annotation/exported")
    parser.add_argument("--splits", "--splits-path", dest="splits_path", type=Path, default=DEFAULT_SPLITS_PATH, help="Path to splits JSON (default: data/splits_v2.json)")
    parser.add_argument("--out", "--output-dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for dataset (default: data/yolo_v1)")
    parser.add_argument("--stats-path", type=Path, default=None, help="Output markdown path for dataset stats")
    parser.add_argument("--with-synth", action="store_true", default=False, help="Include synthetic photos in train split only")
    args = parser.parse_args()

    # Determine default stats path based on output dir name
    stats_path = args.stats_path
    if stats_path is None:
        if args.output_dir.name == "yolo_v2":
            stats_path = ROOT_DIR / "results" / "dataset_stats_v2.md"
        elif args.output_dir.name == "yolo_v3":
            stats_path = ROOT_DIR / "results" / "dataset_stats_v3.md"
        else:
            stats_path = DEFAULT_STATS_PATH

    print("=" * 80)
    print("YOLO DATASET ASSEMBLER & LABEL STATISTICS")
    print("=" * 80)
    print(f"Annotation Dir : {args.annotation_dir}")
    print(f"Exported Dir   : {args.exported_dir}")
    print(f"Splits Path    : {args.splits_path}")
    print(f"Output Dir     : {args.output_dir}")
    print(f"Stats Path     : {stats_path}")
    print(f"Include Synth  : {args.with_synth} (Train split only)")
    print("-" * 80)

    stats = assemble_yolo_dataset(
        annotation_dir=args.annotation_dir,
        exported_dir=args.exported_dir,
        splits_path=args.splits_path,
        output_dir=args.output_dir,
        with_synth=args.with_synth,
    )

    print("\nBatch Status:")
    print(f"  Processed batches ({len(stats['processed_batches'])}): {', '.join(stats['processed_batches'])}")
    if stats["missing_batches"]:
        print(f"  Missing / Unexported batches ({len(stats['missing_batches'])}): {', '.join(stats['missing_batches'])}")

    if stats["missing_labels_images"]:
        print(f"\nMISSING LABELS Images ({len(stats['missing_labels_images'])}):")
        for r in stats["missing_labels_images"]:
            print(f"  - {r}")
    else:
        print("\nAll batch images have corresponding label files (0 missing labels).")

    if stats["refused_images"]:
        print(f"\nRefused Images ({len(stats['refused_images'])}):")
        for r in stats["refused_images"]:
            print(f"  - {r}")
    else:
        print("All exported images successfully assigned to splits (0 refused).")

    print("\nDropped Non-Target Boxes (per split):")
    for s in ["train", "val", "test"]:
        d_items = [f"{c}: {cnt}" for c, cnt in stats["dropped_boxes_per_split"][s].items() if cnt > 0]
        d_str = ", ".join(d_items) if d_items else "none"
        print(f"  {s:<5}: {d_str} (total: {stats['total_dropped_per_split'][s]})")

    print("\nDataset Image Counts:")
    for s in ["train", "val", "test"]:
        tc = stats["splits"][s]["total_images"]
        zc = stats["splits"][s]["zero_box_images"]
        print(f"  {s:<5}: {tc} images ({zc} zero-box images)")

    # Verify parity against reference if assembling yolo_v2 or yolo_v3
    if args.output_dir.name != "yolo_v1":
        ref_dir = ROOT_DIR / "data" / "yolo_v2" if args.output_dir.name == "yolo_v3" else ROOT_DIR / "data" / "yolo_v1"
        verify_val_and_test_parity(args.output_dir, ref_dir=ref_dir)

    print("\nValidation Warnings (Tiny / Duplicate Boxes):")
    warns = stats["validation_warnings"]
    if warns:
        print(f"  Flagged warnings: {len(warns)}")
        for w in warns:
            print(f"  - {w['filename']} [{w['class']}]: {w['box_px']} -> {w['reason']}")
    else:
        print("  0 validation warnings (zero tiny boxes < 0.005, zero duplicate boxes IoU >= 0.90).")

    print("\nLeakage Checks (Counts Only):")
    leak = stats["leakage"]
    print(f"  Split groups checked            : {leak['groups_checked']}")
    print(f"  Split group leakage violations  : {len(leak['group_leakage_violations'])}")
    print(f"  Image file hashes checked       : {leak['hashes_checked']}")
    print(f"  Cross-split duplicate hashes    : {len(leak['hash_leakage_violations'])}")

    generate_stats_markdown(stats, stats_path, with_synth=args.with_synth)
    print(f"\nSaved dataset statistics to: {stats_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
