"""YOLO Label Validation and Split Distribution Inspector.

Performs automated quality checks on YOLO-format label text files:
  1. Out-of-range coordinates (< 0.0 or > 1.0, xmax > 1.0, ymax > 1.0).
  2. Unknown class IDs (outside [0, num_classes - 1]).
  3. Empty files (zero bounding boxes).
  4. Tiny boxes (width or height < min_dim, e.g. 0.005).
  5. Duplicate boxes (same class with IoU >= 0.90).
  6. Reports class frequency and image coverage per split (train, val, test).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT_DIR / "data" / "prelabels"
SPLITS_PATH = ROOT_DIR / "data" / "splits.json"
SPLITS_V2_PATH = ROOT_DIR / "data" / "splits_v2.json"
CLASSES_PATH = ROOT_DIR / "data" / "classes.txt"

NUM_CLASSES = 8
MIN_BOX_DIM = 0.005  # Box width or height below 0.5% of dimension is considered tiny
IOU_DUPLICATE_THRESH = 0.90


@dataclass
class BBox:
    """Represents a single parsed YOLO bounding box."""

    class_id: int
    xc: float
    yc: float
    w: float
    h: float
    line_no: int

    @property
    def xmin(self) -> float:
        return self.xc - self.w / 2.0

    @property
    def xmax(self) -> float:
        return self.xc + self.w / 2.0

    @property
    def ymin(self) -> float:
        return self.yc - self.h / 2.0

    @property
    def ymax(self) -> float:
        return self.yc + self.h / 2.0


def compute_iou(b1: BBox, b2: BBox) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    ixmin = max(b1.xmin, b2.xmin)
    iymin = max(b1.ymin, b2.ymin)
    ixmax = min(b1.xmax, b2.xmax)
    iymax = min(b1.ymax, b2.ymax)

    iw = max(0.0, ixmax - ixmin)
    ih = max(0.0, iymax - iymin)
    inter_area = iw * ih

    area1 = b1.w * b1.h
    area2 = b2.w * b2.h
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@dataclass
class FileValidationResult:
    """Validation report for a single label file."""

    filename: str
    is_empty: bool = False
    allow_empty: bool = False
    out_of_range_coords: List[str] = field(default_factory=list)
    unknown_classes: List[str] = field(default_factory=list)
    tiny_boxes: List[str] = field(default_factory=list)
    duplicate_boxes: List[str] = field(default_factory=list)
    boxes: List[BBox] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.out_of_range_coords or self.unknown_classes)

    @property
    def has_warnings(self) -> bool:
        empty_warn = self.is_empty and not self.allow_empty
        return bool(empty_warn or self.tiny_boxes or self.duplicate_boxes)

    @property
    def warnings_list(self) -> List[str]:
        w = list(self.tiny_boxes + self.duplicate_boxes)
        if self.is_empty and not self.allow_empty:
            w.append("Empty label file (zero bounding boxes). Note: In YOLO, empty files are valid background images.")
        return w


def validate_label_content(
    lines: List[str],
    filename: str = "label.txt",
    num_classes: int = NUM_CLASSES,
    min_dim: float = MIN_BOX_DIM,
    iou_dup_thresh: float = IOU_DUPLICATE_THRESH,
    allow_empty: bool = False,
) -> FileValidationResult:
    """Validate parsed lines of a YOLO text label."""
    result = FileValidationResult(filename=filename, allow_empty=allow_empty)

    valid_boxes: List[BBox] = []

    for line_idx, line in enumerate(lines, start=1):
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("#"):
            continue

        parts = line_clean.split()
        if len(parts) != 5:
            result.out_of_range_coords.append(
                f"Line {line_idx}: expected 5 fields (class xc yc w h), got {len(parts)}."
            )
            continue

        try:
            cid = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError as e:
            result.out_of_range_coords.append(f"Line {line_idx}: numeric parse error: {e}")
            continue

        bbox = BBox(class_id=cid, xc=xc, yc=yc, w=w, h=h, line_no=line_idx)

        # 1. Unknown class check
        if cid < 0 or cid >= num_classes:
            result.unknown_classes.append(
                f"Line {line_idx}: class ID {cid} is outside valid range [0, {num_classes - 1}]."
            )

        # 2. Out-of-range coordinate check
        tol = 1e-4
        if not (0.0 - tol <= xc <= 1.0 + tol and 0.0 - tol <= yc <= 1.0 + tol):
            result.out_of_range_coords.append(
                f"Line {line_idx}: center ({xc:.4f}, {yc:.4f}) outside [0, 1]."
            )
        if not (0.0 < w <= 1.0 + tol and 0.0 < h <= 1.0 + tol):
            result.out_of_range_coords.append(
                f"Line {line_idx}: dimension w={w:.4f}, h={h:.4f} outside (0, 1]."
            )
        if bbox.xmin < 0.0 - tol or bbox.ymin < 0.0 - tol or bbox.xmax > 1.0 + tol or bbox.ymax > 1.0 + tol:
            result.out_of_range_coords.append(
                f"Line {line_idx}: box boundary [{bbox.xmin:.4f}, {bbox.ymin:.4f}, {bbox.xmax:.4f}, {bbox.ymax:.4f}] overflows canvas."
            )

        # 3. Tiny box warning
        if w < min_dim or h < min_dim:
            result.tiny_boxes.append(
                f"Line {line_idx}: tiny box (w={w:.4f}, h={h:.4f} < {min_dim})."
            )

        valid_boxes.append(bbox)

    # 4. Empty file check
    if not valid_boxes:
        result.is_empty = True

    # 5. Duplicate box check (same class, IoU >= threshold)
    for i in range(len(valid_boxes)):
        for j in range(i + 1, len(valid_boxes)):
            b1 = valid_boxes[i]
            b2 = valid_boxes[j]
            if b1.class_id == b2.class_id:
                iou = compute_iou(b1, b2)
                if iou >= iou_dup_thresh:
                    result.duplicate_boxes.append(
                        f"Lines {b1.line_no} & {b2.line_no}: class {b1.class_id} duplicate box (IoU={iou:.2f})."
                    )

    result.boxes = valid_boxes
    return result


def validate_single_file(
    file_path: Path,
    num_classes: int = NUM_CLASSES,
    min_dim: float = MIN_BOX_DIM,
    iou_dup_thresh: float = IOU_DUPLICATE_THRESH,
    allow_empty: bool = False,
) -> FileValidationResult:
    """Validate a YOLO label file from disk."""
    if not file_path.exists():
        r = FileValidationResult(filename=file_path.name, is_empty=True, allow_empty=allow_empty)
        r.out_of_range_coords.append(f"File {file_path} not found.")
        return r

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    return validate_label_content(
        lines=lines,
        filename=file_path.name,
        num_classes=num_classes,
        min_dim=min_dim,
        iou_dup_thresh=iou_dup_thresh,
        allow_empty=allow_empty,
    )


def validate_labels_dir(
    labels_dir: Path,
    splits_path: Optional[Path] = None,
    num_classes: int = NUM_CLASSES,
    allow_empty: bool = False,
) -> Dict[str, Any]:
    """Validate all YOLO text files in a directory and summarize per split."""
    label_files = sorted(list(labels_dir.glob("*.txt")))
    # Filter out classes.txt if present
    label_files = [f for f in label_files if f.name != "classes.txt"]

    results: List[FileValidationResult] = []
    total_boxes = 0
    empty_files_count = 0
    error_files_count = 0
    warning_files_count = 0

    for lf in label_files:
        res = validate_single_file(lf, num_classes=num_classes, allow_empty=allow_empty)
        results.append(res)
        total_boxes += len(res.boxes)
        if res.is_empty:
            empty_files_count += 1
        if res.has_errors:
            error_files_count += 1
        if res.has_warnings:
            warning_files_count += 1

    # Load splits if available
    splits_info = None
    split_counts: Dict[str, Dict[str, int]] = {}
    if splits_path and splits_path.exists():
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
            if "prelabels_own" in str(labels_dir):
                splits_info = splits_data.get("own_splits") or splits_data.get("clean_splits", {})
            else:
                splits_info = splits_data.get("clean_splits") or splits_data.get("combined_clean_splits", {})

    if splits_info:
        for split_name in ["train", "val", "test"]:
            split_counts[split_name] = {f"class_{c}": 0 for c in range(num_classes)}
            split_counts[split_name]["total_boxes"] = 0
            split_counts[split_name]["images_with_boxes"] = 0
            split_counts[split_name]["total_images"] = len(splits_info.get(split_name, []))

        # Map each file to its split
        stem_to_res = {Path(r.filename).stem: r for r in results}
        for split_name, img_names in splits_info.items():
            for img_name in img_names:
                stem = Path(img_name).stem
                if stem in stem_to_res:
                    res = stem_to_res[stem]
                    if len(res.boxes) > 0:
                        split_counts[split_name]["images_with_boxes"] += 1
                    for b in res.boxes:
                        split_counts[split_name][f"class_{b.class_id}"] += 1
                        split_counts[split_name]["total_boxes"] += 1

    return {
        "total_files": len(label_files),
        "total_boxes": total_boxes,
        "empty_files": empty_files_count,
        "error_files": error_files_count,
        "warning_files": warning_files_count,
        "split_counts": split_counts,
        "file_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO label files")
    parser.add_argument("--dir", "-d", type=str, default=str(LABELS_DIR), help="Path to labels directory")
    parser.add_argument("--splits", "-s", type=str, default=None, help="Path to splits JSON")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        default=False,
        help="Allow empty label files as valid background images without flagging warnings",
    )
    args = parser.parse_args()

    target_dir = Path(args.dir)
    target_splits = Path(args.splits) if args.splits else (SPLITS_V2_PATH if "prelabels_own" in str(target_dir) else SPLITS_PATH)
    is_exported_or_dataset = "exported" in str(target_dir).lower() or "yolo" in str(target_dir).lower()
    allow_empty = args.allow_empty or is_exported_or_dataset

    print("=" * 75)
    print("YOLO LABEL VALIDATION & SPLIT SUMMARY")
    print(f"Target Directory: {target_dir}")
    print(f"Splits Source   : {target_splits}")
    print(f"Allow Empty     : {allow_empty} (confirmed background images allowed)")
    print("=" * 75)

    if not target_dir.exists():
        raise FileNotFoundError(f"Missing {target_dir}")

    report = validate_labels_dir(
        target_dir,
        splits_path=target_splits,
        num_classes=NUM_CLASSES,
        allow_empty=allow_empty,
    )

    print(f"\nFiles Scanned: {report['total_files']}")
    print(f"Total Bounding Boxes: {report['total_boxes']}")
    print(f"Empty Files: {report['empty_files']}")
    print(f"Files with Errors: {report['error_files']}")
    print(f"Files with Warnings: {report['warning_files']}")

    # Print error details if any
    errors_found = False
    for r in report["file_results"]:
        if r.has_errors:
            errors_found = True
            print(f"\n[ERROR] in {r.filename}:")
            for err in r.out_of_range_coords + r.unknown_classes:
                print(f"  - {err}")
    if not errors_found:
        print("\n[Verification Passed]: Zero out-of-range coordinates, zero unknown class IDs.")

    # Print warnings
    warnings_found = False
    for r in report["file_results"]:
        if r.warnings_list:
            warnings_found = True
            print(f"\n[WARNING] in {r.filename}:")
            for w in r.warnings_list:
                print(f"  - {w}")
    if not warnings_found:
        print("[Verification Passed]: Zero tiny boxes (<0.005), zero duplicate boxes (IoU >= 0.90), no unexpected empty files.")

    # Print Split Breakdown
    if report["split_counts"]:
        print("\n" + "=" * 75)
        print("CLASS DISTRIBUTION PER SPLIT")
        print("=" * 75)
        headers = ["Split", "Images", "Annotated", "0:vendor", "1:inv_no", "2:date", "3:gstin", "4:table", "5:total", "6:qr", "7:stamp", "Total"]
        row_fmt = "{:<7} {:>6} {:>10} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8} {:>6} {:>8} {:>7}"
        print(row_fmt.format(*headers))
        print("-" * 105)

        for s_name in ["train", "val", "test"]:
            sc = report["split_counts"][s_name]
            print(
                row_fmt.format(
                    s_name.upper(),
                    sc["total_images"],
                    f"{sc['images_with_boxes']}/{sc['total_images']}",
                    sc["class_0"],
                    sc["class_1"],
                    sc["class_2"],
                    sc["class_3"],
                    sc["class_4"],
                    sc["class_5"],
                    sc["class_6"],
                    sc["class_7"],
                    sc["total_boxes"],
                )
            )
        print("-" * 105)


if __name__ == "__main__":
    main()
