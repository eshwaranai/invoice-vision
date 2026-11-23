"""Assemble Annotation Batches 10 to 13 for Synthetic GST Invoices.

Batch Structure:
  - data/annotation/batch_10 (15 images: gstsyn_024 .. gstsyn_038)
  - data/annotation/batch_11 (15 images: gstsyn_039 .. gstsyn_053)
  - data/annotation/batch_12 (15 images: gstsyn_054 .. gstsyn_068)
  - data/annotation/batch_13 (15 images: gstsyn_069 .. gstsyn_083)

Each batch contains:
  - image files (.png / .jpg)
  - matching YOLO .txt pre-labels
  - classes.txt
  - labels.txt (identical to classes.txt)
  - images_only/ subfolder containing the 15 image files
  - yolo_import/ subfolder with .txt files and labels.txt only
  - batch_manifest.json

STRICT SAFETY:
  - Do NOT touch existing batches (batch_01 - batch_09).
  - Do NOT touch data/annotation/exported/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_SYNTH_DIR = ROOT_DIR / "data" / "raw" / "gst_synth"
PRELABELS_DIR = ROOT_DIR / "data" / "prelabels_gst_synth"
ANNOTATION_DIR = ROOT_DIR / "data" / "annotation"
CLASSES_TXT_PATH = ROOT_DIR / "data" / "classes.txt"
MANIFEST_CSV = ROOT_DIR / "data" / "gst_synth_manifest.csv"


def assemble_batch(batch_dir: Path, batch_num: int, image_filenames: List[str]) -> None:
    """Create batch directory and populate all required assets."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    images_only_dir = batch_dir / "images_only"
    images_only_dir.mkdir(parents=True, exist_ok=True)
    yolo_import_dir = batch_dir / "yolo_import"
    yolo_import_dir.mkdir(parents=True, exist_ok=True)

    # Read class definitions
    classes_content = CLASSES_TXT_PATH.read_text(encoding="utf-8")

    # Write classes.txt and labels.txt (identical content) in batch root
    (batch_dir / "classes.txt").write_text(classes_content, encoding="utf-8")
    (batch_dir / "labels.txt").write_text(classes_content, encoding="utf-8")

    # In yolo_import/ write labels.txt
    (yolo_import_dir / "labels.txt").write_text(classes_content, encoding="utf-8")

    empty_prelabels = 0

    for fn in image_filenames:
        src_img = RAW_SYNTH_DIR / fn
        stem = Path(fn).stem
        src_txt = PRELABELS_DIR / f"{stem}.txt"

        if not src_img.exists():
            raise FileNotFoundError(f"Missing raw image: {src_img}")

        # Copy image to batch root and images_only/
        shutil.copy2(src_img, batch_dir / fn)
        shutil.copy2(src_img, images_only_dir / fn)

        # Copy .txt prelabel (or create empty if not found)
        txt_content = ""
        if src_txt.exists():
            txt_content = src_txt.read_text(encoding="utf-8")
        else:
            empty_prelabels += 1

        (batch_dir / f"{stem}.txt").write_text(txt_content, encoding="utf-8")
        (yolo_import_dir / f"{stem}.txt").write_text(txt_content, encoding="utf-8")

    # Write batch_manifest.json
    manifest = {
        "batch_id": batch_dir.name,
        "batch_number": batch_num,
        "folder_name": batch_dir.name,
        "split": "train",
        "is_test": False,
        "source": "gst_synth",
        "image_count": len(image_filenames),
        "images": image_filenames,
        "empty_prelabels_created": empty_prelabels,
        "path": str(batch_dir),
    }
    with open(batch_dir / "batch_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    n_imgs = len(list(images_only_dir.glob("*.*")))
    n_lbls = len(list(yolo_import_dir.glob("*.txt"))) - 1
    print(f"Assembled {batch_dir.name}: {n_imgs} images in images_only/, {n_lbls} label files in yolo_import/.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble synthetic annotation batches.")
    parser.add_argument(
        "--batches",
        choices=["10-13", "08-09", "all"],
        default="10-13",
        help="Which batches to assemble (default: 10-13).",
    )
    args = parser.parse_args()

    all_raw_images = sorted(
        [f.name for f in RAW_SYNTH_DIR.glob("gstsyn_*") if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    )

    print("=" * 75)
    print(f"ASSEMBLING ANNOTATION BATCHES ({args.batches})")
    print("=" * 75)

    if args.batches in ("08-09", "all"):
        if len(all_raw_images) < 24:
            raise ValueError(f"Need at least 24 images for batches 08-09, found {len(all_raw_images)}")
        assemble_batch(ANNOTATION_DIR / "batch_08", 8, all_raw_images[:15])
        assemble_batch(ANNOTATION_DIR / "batch_09", 9, all_raw_images[15:24])

    if args.batches in ("10-13", "all"):
        if len(all_raw_images) != 84:
            raise ValueError(f"Expected 84 synthetic images for batches 10-13, found {len(all_raw_images)}")

        b10 = all_raw_images[24:39]
        b11 = all_raw_images[39:54]
        b12 = all_raw_images[54:69]
        b13 = all_raw_images[69:84]

        assemble_batch(ANNOTATION_DIR / "batch_10", 10, b10)
        assemble_batch(ANNOTATION_DIR / "batch_11", 11, b11)
        assemble_batch(ANNOTATION_DIR / "batch_12", 12, b12)
        assemble_batch(ANNOTATION_DIR / "batch_13", 13, b13)

        print("-" * 75)
        print("Successfully assembled batches 10, 11, 12, and 13 (15 images each).")
        print("Existing batches (batch_01 - batch_09) and data/annotation/exported/ remain untouched.")

    print("=" * 75)


if __name__ == "__main__":
    main()
