"""Update Pre-Labels and Rebuild Unreviewed Annotation Batches.

Updates:
  1. data/prelabels_own/*.txt using the new unified suggester.
  2. data/prelabels_own/review_priority.csv with (file, class, confidence).
  3. data/prelabels_own/README.md.
  4. Unreviewed batch folders (batch_02, batch_03, batch_04_test, batch_05,
     batch_06, batch_07_test) with updated prelabels and a yolo_import/ subfolder
     containing the .txt prelabels and labels.txt.

STRICT SAFETY:
  - NEVER modifies data/annotation/batch_01 or data/annotation/exported/.
  - PRIVACY: Only reports class counts and file names, zero personal text.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Tuple

import cv2
import pymupdf  # PyMuPDF

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from invoice_vision.prelabel.suggest import CLASS_NAMES, SuggestedBox, suggest_document_boxes

PDF_TEXT_DIR = ROOT_DIR / "data" / "processed" / "pdf_text"
RAW_OWN_DIR = ROOT_DIR / "data" / "raw" / "own"
PDF_DIR = ROOT_DIR / "data" / "raw" / "own_pdf"
PRELABELS_OWN_DIR = ROOT_DIR / "data" / "prelabels_own"
ANNOTATION_DIR = ROOT_DIR / "data" / "annotation"
OCR_CACHE_DIR = ROOT_DIR / "data" / "processed" / "ocr_cache"
RAW_PUBLIC_DIR = ROOT_DIR / "data" / "raw" / "indian_gst"
CLASSES_TXT = ROOT_DIR / "data" / "classes.txt"


def update_own_prelabels() -> Tuple[Dict[str, int], List[Dict[str, str]]]:
    """Regenerate pre-labels for personal bills from PDF text layers."""
    PRELABELS_OWN_DIR.mkdir(parents=True, exist_ok=True)
    pdf_text_files = sorted(list(PDF_TEXT_DIR.glob("*.json")))

    class_counts: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
    priority_records: List[Dict[str, str]] = []

    for pf in pdf_text_files:
        with open(pf, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_fn = data.get("image_filename", f"{pf.stem}.png")
        img_path = RAW_OWN_DIR / img_fn
        img = cv2.imread(str(img_path)) if img_path.exists() else None
        h = data.get("height_px", 1000)
        w = data.get("width_px", 800)
        if img is not None:
            h, w = img.shape[:2]

        pdf_fn = data.get("pdf_filename")
        page = None
        if pdf_fn:
            pdf_path = PDF_DIR / pdf_fn
            if pdf_path.exists():
                try:
                    doc = pymupdf.open(str(pdf_path))
                    pno = max(0, int(data.get("page_number", 1)) - 1)
                    if pno < len(doc):
                        page = doc[pno]
                except Exception:
                    page = None

        proposals = suggest_document_boxes(data, img=img, pdf_page=page, img_shape=(h, w))

        stem = Path(img_fn).stem
        yolo_lines: List[str] = []

        for p in proposals:
            class_counts[p.class_name] += 1
            xc, yc, bw, bh = p.to_yolo(w, h)
            yolo_lines.append(f"{p.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            priority_records.append(
                {
                    "file": img_fn,
                    "class": p.class_name,
                    "confidence": p.confidence,
                }
            )

        out_txt = PRELABELS_OWN_DIR / f"{stem}.txt"
        out_txt.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

    # Write review_priority.csv
    csv_path = PRELABELS_OWN_DIR / "review_priority.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "class", "confidence"])
        writer.writeheader()
        writer.writerows(priority_records)

    # Update README.md
    readme_path = PRELABELS_OWN_DIR / "README.md"
    readme_path.write_text(
        "# Automated Heuristic Pre-Labels (Personal Bills - v2 Suggester)\n\n"
        "> [!WARNING]\n"
        "> **MANDATORY HUMAN REVIEW REQUIRED BEFORE TRAINING**\n"
        "> Pre-labels regenerated with the unified v2 heuristic suggester.\n"
        "> Review low-confidence candidates first using `review_priority.csv`.\n",
        encoding="utf-8",
    )

    return class_counts, priority_records


def update_unreviewed_batches() -> None:
    """Update pre-labels and create yolo_import/ folders for unreviewed batches."""
    # Find all batches except batch_01
    batch_dirs = sorted([d for d in ANNOTATION_DIR.iterdir() if d.is_dir() and d.name.startswith("batch_")])

    for b_dir in batch_dirs:
        if b_dir.name == "batch_01":
            print(f"Skipping {b_dir.name} (strictly preserved as reviewed ground truth).")
            continue

        manifest_path = b_dir / "batch_manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        source = manifest.get("source", "public")
        images = manifest.get("images", [])

        print(f"Updating {b_dir.name} ({source}, {len(images)} images)...")

        # Create or clear yolo_import folder
        yolo_import_dir = b_dir / "yolo_import"
        yolo_import_dir.mkdir(parents=True, exist_ok=True)

        # Copy classes.txt as labels.txt into yolo_import
        shutil.copy2(CLASSES_TXT, yolo_import_dir / "labels.txt")

        for img_fn in images:
            stem = Path(img_fn).stem
            img_path = (RAW_OWN_DIR if source == "own" else RAW_PUBLIC_DIR) / img_fn
            img = cv2.imread(str(img_path)) if img_path.exists() else None
            h, w = img.shape[:2] if img is not None else (1000, 800)

            if source == "own":
                pdf_json = PDF_TEXT_DIR / f"{stem}.json"
                if not pdf_json.exists():
                    continue
                with open(pdf_json, "r", encoding="utf-8") as f:
                    data = json.load(f)

                pdf_fn = data.get("pdf_filename")
                page = None
                if pdf_fn and (PDF_DIR / pdf_fn).exists():
                    try:
                        doc = pymupdf.open(str(PDF_DIR / pdf_fn))
                        pno = max(0, int(data.get("page_number", 1)) - 1)
                        if pno < len(doc):
                            page = doc[pno]
                    except Exception:
                        page = None

                proposals = suggest_document_boxes(data, img=img, pdf_page=page, img_shape=(h, w))
            else:
                # Public image
                ocr_json = OCR_CACHE_DIR / f"{stem}.jpg.json"
                if not ocr_json.exists():
                    continue
                with open(ocr_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                proposals = suggest_document_boxes(data, img=img, img_shape=(h, w))

            yolo_lines = []
            for p in proposals:
                xc, yc, bw, bh = p.to_yolo(w, h)
                yolo_lines.append(f"{p.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            content = "\n".join(yolo_lines) + "\n" if yolo_lines else ""

            # Update batch .txt
            (b_dir / f"{stem}.txt").write_text(content, encoding="utf-8")
            # Write into yolo_import folder
            (yolo_import_dir / f"{stem}.txt").write_text(content, encoding="utf-8")


def main() -> None:
    print("=" * 75)
    print("REGENERATING PRE-LABELS FOR OWN BILLS")
    print("=" * 75)
    counts, records = update_own_prelabels()
    print("\nClass Coverage for Personal Bills (29 pages):")
    total_boxes = sum(counts.values())
    for c_name in CLASS_NAMES:
        cnt = counts[c_name]
        pct = (cnt / 29.0) * 100
        print(f"  {c_name:<16}: {cnt:>3} boxes ({pct:>5.1f}% page coverage)")
    print("-" * 75)
    print(f"Total Proposed Boxes: {total_boxes} (avg {total_boxes / 29.0:.2f} boxes/page)")

    # Count by confidence
    conf_counts = {"high": 0, "medium": 0, "low": 0}
    for r in records:
        conf_counts[r["confidence"]] += 1
    print("\nConfidence Breakdown for Review Priority:")
    print(f"  High   : {conf_counts['high']:>3} boxes")
    print(f"  Medium : {conf_counts['medium']:>3} boxes")
    print(f"  Low    : {conf_counts['low']:>3} boxes")

    print("\n" + "=" * 75)
    print("UPDATING UNREVIEWED ANNOTATION BATCHES")
    print("=" * 75)
    update_unreviewed_batches()
    print("Done.\n")


if __name__ == "__main__":
    main()
