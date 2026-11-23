"""Compare YOLO Pre-Labels vs Exported Reviewed Annotations.

Computes IoU-based bipartite matching between initial pre-labels and human-reviewed labels:
  - same class and IoU >= 0.50 -> kept
  - same class and 0.30 <= IoU < 0.50 -> adjusted
  - pre-label with no match (IoU < 0.30) -> deleted (pre-label false positive)
  - reviewed box with no matching pre-label (IoU < 0.30) -> added (reviewer addition)

Reports per class:
  - kept, adjusted, deleted, added counts
  - pre-label precision (strict >= 0.5, lenient >= 0.3)
  - pre-label recall (strict >= 0.5, lenient >= 0.3)
Runs automated validation checks on exported annotations using validate_labels.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from scripts.validate_labels import (
    BBox,
    NUM_CLASSES,
    compute_iou,
    validate_label_content,
    validate_labels_dir,
    validate_single_file,
)

DEFAULT_ANNOTATION_DIR = ROOT_DIR / "data" / "annotation"
DEFAULT_EXPORTED_DIR = ROOT_DIR / "data" / "annotation" / "exported"
DEFAULT_CLASSES_PATH = ROOT_DIR / "data" / "classes.txt"

CLASS_NAMES = [
    "vendor",
    "invoice_no",
    "date",
    "gstin",
    "table",
    "total",
    "qr",
    "stamp_signature",
]


@dataclass
class ClassMatchStats:
    """Class-level comparison statistics."""

    class_id: int
    class_name: str
    kept: int = 0
    adjusted: int = 0
    deleted: int = 0
    added: int = 0

    @property
    def total_prelabels(self) -> int:
        return self.kept + self.adjusted + self.deleted

    @property
    def total_reviewed(self) -> int:
        return self.kept + self.adjusted + self.added

    @property
    def precision_strict(self) -> float:
        """Precision at IoU >= 0.5."""
        return (self.kept / self.total_prelabels) if self.total_prelabels > 0 else 0.0

    @property
    def recall_strict(self) -> float:
        """Recall at IoU >= 0.5."""
        return (self.kept / self.total_reviewed) if self.total_reviewed > 0 else 0.0

    @property
    def precision_lenient(self) -> float:
        """Precision at IoU >= 0.3 (kept + adjusted)."""
        return ((self.kept + self.adjusted) / self.total_prelabels) if self.total_prelabels > 0 else 0.0

    @property
    def recall_lenient(self) -> float:
        """Recall at IoU >= 0.3 (kept + adjusted)."""
        return ((self.kept + self.adjusted) / self.total_reviewed) if self.total_reviewed > 0 else 0.0


def parse_label_file(file_path: Path) -> List[BBox]:
    """Parse YOLO boxes from a text file."""
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    res = validate_label_content(lines=lines, filename=file_path.name)
    return res.boxes


def match_boxes_for_class(
    pre_boxes: List[BBox],
    rev_boxes: List[BBox],
    strict_thresh: float = 0.50,
    lenient_thresh: float = 0.30,
) -> Tuple[int, int, int, int]:
    """Match boxes belonging to the same class using greedy bipartite matching.

    Returns:
        (kept, adjusted, deleted, added)
    """
    if not pre_boxes and not rev_boxes:
        return 0, 0, 0, 0
    if pre_boxes and not rev_boxes:
        return 0, 0, len(pre_boxes), 0
    if rev_boxes and not pre_boxes:
        return 0, 0, 0, len(rev_boxes)

    # Compute candidate pairs with IoU >= lenient_thresh
    candidates: List[Tuple[float, int, int]] = []
    for p_idx, p_box in enumerate(pre_boxes):
        for r_idx, r_box in enumerate(rev_boxes):
            iou = compute_iou(p_box, r_box)
            if iou >= lenient_thresh:
                candidates.append((iou, p_idx, r_idx))

    # Sort candidates in descending order of IoU
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_pre: Set[int] = set()
    matched_rev: Set[int] = set()

    kept = 0
    adjusted = 0

    for iou, p_idx, r_idx in candidates:
        if p_idx in matched_pre or r_idx in matched_rev:
            continue
        matched_pre.add(p_idx)
        matched_rev.add(r_idx)
        if iou >= strict_thresh:
            kept += 1
        else:
            adjusted += 1

    deleted = len(pre_boxes) - len(matched_pre)
    added = len(rev_boxes) - len(matched_rev)

    return kept, adjusted, deleted, added


def compare_image_boxes(
    pre_boxes: List[BBox],
    rev_boxes: List[BBox],
    num_classes: int = NUM_CLASSES,
) -> Dict[int, Tuple[int, int, int, int]]:
    """Compare pre-labels and reviewed labels for a single image across all classes.

    Returns:
        Dict mapping class_id -> (kept, adjusted, deleted, added)
    """
    class_results: Dict[int, Tuple[int, int, int, int]] = {}

    for cid in range(num_classes):
        p_sub = [b for b in pre_boxes if b.class_id == cid]
        r_sub = [b for b in rev_boxes if b.class_id == cid]
        class_results[cid] = match_boxes_for_class(p_sub, r_sub)

    return class_results


def compare_batch(
    prelabel_dir: Path,
    reviewed_dir: Path,
    num_classes: int = NUM_CLASSES,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare all matching label files between pre-label dir and reviewed dir."""
    if class_names is None:
        class_names = CLASS_NAMES

    pre_files = {f.stem: f for f in prelabel_dir.glob("*.txt") if f.name != "classes.txt"}
    rev_files = {f.stem: f for f in reviewed_dir.glob("*.txt") if f.name != "classes.txt"}

    all_stems = sorted(set(pre_files.keys()).union(set(rev_files.keys())))

    class_stats = {
        cid: ClassMatchStats(class_id=cid, class_name=class_names[cid])
        for cid in range(num_classes)
    }

    per_image_results: Dict[str, Dict[int, Tuple[int, int, int, int]]] = {}

    for stem in all_stems:
        p_path = pre_files.get(stem)
        r_path = rev_files.get(stem)

        p_boxes = parse_label_file(p_path) if p_path else []
        r_boxes = parse_label_file(r_path) if r_path else []

        img_res = compare_image_boxes(p_boxes, r_boxes, num_classes=num_classes)
        per_image_results[stem] = img_res

        for cid, (k, adj, d, a) in img_res.items():
            class_stats[cid].kept += k
            class_stats[cid].adjusted += adj
            class_stats[cid].deleted += d
            class_stats[cid].added += a

    # Run label validation on reviewed folder
    val_report = validate_labels_dir(reviewed_dir, num_classes=num_classes)

    return {
        "prelabel_dir": str(prelabel_dir),
        "reviewed_dir": str(reviewed_dir),
        "total_files": len(all_stems),
        "matching_files": len(set(pre_files.keys()).intersection(set(rev_files.keys()))),
        "class_stats": class_stats,
        "per_image": per_image_results,
        "validation_report": val_report,
    }


def print_comparison_report(report: Dict[str, Any], batch_label: str = "") -> None:
    """Print formatted comparison and quality report."""
    title = f"PRE-LABEL VS REVIEWED COMPARISON REPORT: {batch_label}" if batch_label else "PRE-LABEL VS REVIEWED COMPARISON REPORT"
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(f"Pre-label Dir : {report['prelabel_dir']}")
    print(f"Reviewed Dir  : {report['reviewed_dir']}")
    print(f"Total Files   : {report['total_files']} (Matching: {report['matching_files']})")
    print("-" * 90)

    # Summary table per class
    header = f"{'Class':<16} {'Kept(>=.5)':<11} {'Adj(.3-.5)':<11} {'Deleted':<9} {'Added':<8} {'Prec(Strict)':<13} {'Rec(Strict)':<13} {'Prec(Lenient)':<14}"
    print(header)
    print("-" * 90)

    tot_kept = 0
    tot_adj = 0
    tot_del = 0
    tot_add = 0
    tot_pre = 0
    tot_rev = 0

    for cid in range(NUM_CLASSES):
        stats: ClassMatchStats = report["class_stats"][cid]
        tot_kept += stats.kept
        tot_adj += stats.adjusted
        tot_del += stats.deleted
        tot_add += stats.added
        tot_pre += stats.total_prelabels
        tot_rev += stats.total_reviewed

        prec_s_str = f"{stats.precision_strict * 100:.1f}%" if stats.total_prelabels > 0 else "-"
        rec_s_str = f"{stats.recall_strict * 100:.1f}%" if stats.total_reviewed > 0 else "-"
        prec_l_str = f"{stats.precision_lenient * 100:.1f}%" if stats.total_prelabels > 0 else "-"

        print(
            f"{stats.class_name:<16} "
            f"{stats.kept:<11} "
            f"{stats.adjusted:<11} "
            f"{stats.deleted:<9} "
            f"{stats.added:<8} "
            f"{prec_s_str:<13} "
            f"{rec_s_str:<13} "
            f"{prec_l_str:<14}"
        )

    print("-" * 90)
    overall_prec_s = f"{(tot_kept / tot_pre * 100):.1f}%" if tot_pre > 0 else "-"
    overall_rec_s = f"{(tot_kept / tot_rev * 100):.1f}%" if tot_rev > 0 else "-"
    overall_prec_l = f"{((tot_kept + tot_adj) / tot_pre * 100):.1f}%" if tot_pre > 0 else "-"

    print(
        f"{'TOTAL / MACRO':<16} "
        f"{tot_kept:<11} "
        f"{tot_adj:<11} "
        f"{tot_del:<9} "
        f"{tot_add:<8} "
        f"{overall_prec_s:<13} "
        f"{overall_rec_s:<13} "
        f"{overall_prec_l:<14}"
    )
    print("=" * 90)

    # Validation summary
    val = report.get("validation_report", {})
    print("\n--- Reviewed Labels Validation Check ---")
    print(f"Total checked files: {val.get('total_files', 0)}")
    print(f"Total bounding boxes: {val.get('total_boxes', 0)}")
    print(f"Empty files: {val.get('empty_files', 0)}")
    print(f"Files with coordinate errors: {val.get('error_files', 0)}")
    print(f"Files with warnings (tiny/duplicate): {val.get('warning_files', 0)}")
    print("=" * 90 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO pre-labels with exported reviewed labels.")
    parser.add_argument(
        "--batch",
        type=str,
        default="",
        help="Batch folder name (e.g. batch_01, batch_04_test) or number (1..7).",
    )
    parser.add_argument(
        "--prelabel-dir",
        type=Path,
        default=None,
        help="Custom pre-labels directory path.",
    )
    parser.add_argument(
        "--reviewed-dir",
        type=Path,
        default=None,
        help="Custom reviewed labels directory path.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare all batches found in data/annotation/exported/.",
    )
    return parser.parse_args()


def resolve_batch_dirs(batch_arg: str) -> Tuple[Path, Path]:
    """Resolve pre-label and reviewed directories from a batch identifier."""
    # Handle pure numeric input like '1' or '4'
    if batch_arg.isdigit():
        b_num = int(batch_arg)
        # Search in data/annotation/ for matching folder
        matches = list(DEFAULT_ANNOTATION_DIR.glob(f"batch_{b_num:02d}*"))
        if not matches:
            raise FileNotFoundError(f"No batch directory found for batch number {b_num} in {DEFAULT_ANNOTATION_DIR}")
        batch_folder_name = matches[0].name
    else:
        batch_folder_name = batch_arg

    pre_dir = DEFAULT_ANNOTATION_DIR / batch_folder_name
    rev_dir = DEFAULT_EXPORTED_DIR / batch_folder_name

    # Also check if exported directory has no '_test' suffix or vice versa
    if not rev_dir.exists():
        alt_name = batch_folder_name.replace("_test", "") if "_test" in batch_folder_name else f"{batch_folder_name}_test"
        alt_rev = DEFAULT_EXPORTED_DIR / alt_name
        if alt_rev.exists():
            rev_dir = alt_rev

    return pre_dir, rev_dir


def main() -> int:
    args = parse_args()

    if args.prelabel_dir and args.reviewed_dir:
        report = compare_batch(args.prelabel_dir, args.reviewed_dir)
        print_comparison_report(report, batch_label="Custom Directories")
        return 0

    if args.batch:
        pre_dir, rev_dir = resolve_batch_dirs(args.batch)
        if not pre_dir.exists():
            print(f"Error: Pre-label batch directory not found: {pre_dir}", file=sys.stderr)
            return 1
        if not rev_dir.exists():
            print(f"Warning: Exported reviewed directory not found: {rev_dir}", file=sys.stderr)
            print(f"To review this batch, follow the guide in docs/review_workflow.md", file=sys.stderr)
            print(f"and unzip the exported YOLO package into {rev_dir}", file=sys.stderr)
            return 1

        report = compare_batch(pre_dir, rev_dir)
        print_comparison_report(report, batch_label=args.batch)
        return 0

    if args.all:
        if not DEFAULT_EXPORTED_DIR.exists():
            print(f"No exported directory found at {DEFAULT_EXPORTED_DIR}", file=sys.stderr)
            return 0
        exported_batches = sorted([d for d in DEFAULT_EXPORTED_DIR.iterdir() if d.is_dir()])
        if not exported_batches:
            print(f"No reviewed batch folders found in {DEFAULT_EXPORTED_DIR}", file=sys.stderr)
            return 0
        for b_dir in exported_batches:
            pre_dir, _ = resolve_batch_dirs(b_dir.name)
            if pre_dir.exists():
                report = compare_batch(pre_dir, b_dir)
                print_comparison_report(report, batch_label=b_dir.name)
        return 0

    # Default if no args passed: check if any exported batch exists or show help
    print("Usage: python scripts/compare_prelabels_vs_reviewed.py --batch <batch_id> (e.g. batch_01)")
    print("       python scripts/compare_prelabels_vs_reviewed.py --all")
    print("       python scripts/compare_prelabels_vs_reviewed.py --prelabel-dir <dir> --reviewed-dir <dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
