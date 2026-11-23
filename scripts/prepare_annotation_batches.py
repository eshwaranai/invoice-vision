"""Prepare Annotation Batches for Makesense.ai.

Splits dataset (30 public scans + 29 own personal pages) into bounded annotation
batches (max 15 images/batch), adhering to the following rules:
  1. Split integrity: Every batch is strictly homogeneous to one split (train, val, test).
  2. Source order: All 30 public images first, followed by all 29 own personal pages.
  3. Test isolation: Test-split batches are isolated and clearly labeled '_test'.
  4. Complete file pairs: Each batch folder contains images, matching YOLO .txt pre-labels,
     and classes.txt (from data/classes.txt). If an image has no pre-label, an empty .txt
     file is written.
  5. Privacy: Never writes or prints personal PII.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = ROOT_DIR / "data" / "splits_v2.json"
DEFAULT_CLASSES = ROOT_DIR / "data" / "classes.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "annotation"

PUBLIC_IMG_DIR = ROOT_DIR / "data" / "raw" / "indian_gst"
PUBLIC_LBL_DIR = ROOT_DIR / "data" / "prelabels"

OWN_IMG_DIR = ROOT_DIR / "data" / "raw" / "own"
OWN_LBL_DIR = ROOT_DIR / "data" / "prelabels_own"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare annotation batches for makesense.ai")
    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS,
        help="Path to splits JSON file (default: data/splits_v2.json)",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=DEFAULT_CLASSES,
        help="Path to classes.txt file (default: data/classes.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Target annotation output directory (default: data/annotation)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=15,
        help="Maximum number of images per batch (default: 15)",
    )
    return parser.parse_args()


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    """Split items into sublists of size <= chunk_size."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def prepare_batches(
    splits_path: Path,
    classes_path: Path,
    output_dir: Path,
    max_batch_size: int = 15,
) -> List[Dict[str, Any]]:
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_path}")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    classes_content = classes_path.read_text(encoding="utf-8")

    # Defined sequence of groups to process:
    # 1. Public train, val, test
    # 2. Own train, val, test
    sequence: List[Tuple[str, str, Path, Path, List[str]]] = [
        ("public", "train", PUBLIC_IMG_DIR, PUBLIC_LBL_DIR, splits["public_clean_splits"]["train"]),
        ("public", "val", PUBLIC_IMG_DIR, PUBLIC_LBL_DIR, splits["public_clean_splits"]["val"]),
        ("public", "test", PUBLIC_IMG_DIR, PUBLIC_LBL_DIR, splits["public_clean_splits"]["test"]),
        ("own", "train", OWN_IMG_DIR, OWN_LBL_DIR, splits["own_splits"]["train"]),
        ("own", "val", OWN_IMG_DIR, OWN_LBL_DIR, splits["own_splits"]["val"]),
        ("own", "test", OWN_IMG_DIR, OWN_LBL_DIR, splits["own_splits"]["test"]),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    batch_records: List[Dict[str, Any]] = []
    batch_idx = 1

    for source, split, img_dir, lbl_dir, file_list in sequence:
        chunks = chunk_list(file_list, max_batch_size)
        for chunk in chunks:
            is_test = (split == "test")
            folder_name = f"batch_{batch_idx:02d}_test" if is_test else f"batch_{batch_idx:02d}"
            batch_folder = output_dir / folder_name
            batch_folder.mkdir(parents=True, exist_ok=True)

            # Copy classes.txt
            shutil.copy2(classes_path, batch_folder / "classes.txt")

            copied_images = []
            prelabel_files = []
            empty_prelabels = 0

            for img_name in chunk:
                src_img = img_dir / img_name
                if not src_img.exists():
                    raise FileNotFoundError(f"Source image missing: {src_img}")

                dst_img = batch_folder / img_name
                shutil.copy2(src_img, dst_img)
                copied_images.append(img_name)

                stem = Path(img_name).stem
                src_lbl = lbl_dir / f"{stem}.txt"
                dst_lbl = batch_folder / f"{stem}.txt"

                if src_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    # Write empty pre-label file if none exists
                    dst_lbl.write_text("", encoding="utf-8")
                    empty_prelabels += 1

                prelabel_files.append(dst_lbl.name)

            record: Dict[str, Any] = {
                "batch_id": folder_name,
                "batch_number": batch_idx,
                "folder_name": folder_name,
                "split": split,
                "is_test": is_test,
                "source": source,
                "image_count": len(chunk),
                "images": chunk,
                "empty_prelabels_created": empty_prelabels,
                "path": str(batch_folder.resolve()),
            }

            # Save per-batch manifest
            with open(batch_folder / "batch_manifest.json", "w", encoding="utf-8") as bf:
                json.dump(record, bf, indent=2)

            batch_records.append(record)
            batch_idx += 1

    # Save top-level batches manifest
    manifest_path = output_dir / "batches_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_batches": len(batch_records),
                "total_images": sum(b["image_count"] for b in batch_records),
                "batches": batch_records,
            },
            f,
            indent=2,
        )

    return batch_records


def print_batch_plan(batch_records: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("ANNOTATION BATCH PLAN")
    print("=" * 80)
    print(f"{'Batch ID':<16} {'Split':<8} {'Source':<8} {'Images':<8} {'First File':<20} {'Last File':<20}")
    print("-" * 80)
    total_imgs = 0
    for b in batch_records:
        b_id = b["batch_id"]
        split = b["split"]
        src = b["source"]
        cnt = b["image_count"]
        first_f = b["images"][0] if b["images"] else "-"
        last_f = b["images"][-1] if b["images"] else "-"
        total_imgs += cnt
        print(f"{b_id:<16} {split:<8} {src:<8} {cnt:<8} {first_f:<20} {last_f:<20}")
    print("-" * 80)
    print(f"Total Batches: {len(batch_records)} | Total Images: {total_imgs}")
    print("=" * 80 + "\n")


def main() -> int:
    args = parse_args()
    batches = prepare_batches(
        splits_path=args.splits,
        classes_path=args.classes,
        output_dir=args.output_dir,
        max_batch_size=args.max_batch_size,
    )
    print_batch_plan(batches)
    return 0


if __name__ == "__main__":
    sys.exit(main())
