"""Convert Personal PDF Invoices to 200 DPI PNG Images.

Processes all PDF documents in data/raw/own_pdf/:
  1. Renders each page to RGB PNG at 200 DPI into data/raw/own/
     Naming: own_001_p1.png, own_001_p2.png (sorted order of original PDFs).
  2. Extracts text layer if present (word count and bounding boxes scaled to rendered px).
     Saves word bounding boxes to data/processed/pdf_text/own_001_p1.json.
  3. Writes data/pdf_manifest.csv with:
     pdf_filename, image_filename, page_number, page_count, width_px, height_px, has_text_layer, word_count.
  4. Keeps original PDFs untouched.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import pymupdf

ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT_DIR / "data" / "raw" / "own_pdf"
IMAGE_DIR = ROOT_DIR / "data" / "raw" / "own"
TEXT_DIR = ROOT_DIR / "data" / "processed" / "pdf_text"
MANIFEST_PATH = ROOT_DIR / "data" / "pdf_manifest.csv"

TARGET_DPI = 200.0
PDF_POINTS_PER_INCH = 72.0
ZOOM_FACTOR = TARGET_DPI / PDF_POINTS_PER_INCH


def convert_all_pdfs() -> Dict[str, Any]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(list(PDF_DIR.glob("*.pdf")), key=lambda p: p.name.lower())
    total_pdfs = len(pdf_files)
    print(f"Discovered {total_pdfs} PDF files in {PDF_DIR}")

    manifest_rows: List[Dict[str, Any]] = []
    failed_pdfs: List[Tuple[str, str]] = []
    encrypted_pdfs: List[str] = []
    large_page_pdfs: List[Tuple[str, int]] = []
    pages_with_text_layer = 0
    total_rendered_pages = 0

    render_matrix = pymupdf.Matrix(ZOOM_FACTOR, ZOOM_FACTOR)

    for pdf_idx, pdf_path in enumerate(pdf_files, start=1):
        pdf_code = f"own_{pdf_idx:03d}"
        pdf_name = pdf_path.name

        try:
            doc = pymupdf.open(pdf_path)
        except Exception as e:
            failed_pdfs.append((pdf_name, f"Failed to open: {e}"))
            continue

        if doc.is_encrypted:
            # Attempt blank password
            auth_ok = doc.authenticate("")
            if not auth_ok:
                encrypted_pdfs.append(pdf_name)
                failed_pdfs.append((pdf_name, "Password-protected / encrypted"))
                doc.close()
                continue

        page_count = len(doc)
        if page_count > 3:
            large_page_pdfs.append((pdf_name, page_count))

        for page_num in range(1, page_count + 1):
            page = doc[page_num - 1]
            image_name = f"{pdf_code}_p{page_num}.png"
            image_path = IMAGE_DIR / image_name
            text_json_path = TEXT_DIR / f"{pdf_code}_p{page_num}.json"

            # 1. Render to RGB PNG at 200 DPI
            pix = page.get_pixmap(matrix=render_matrix, alpha=False)
            pix.save(str(image_path))
            total_rendered_pages += 1
            w_px, h_px = pix.width, pix.height

            # 2. Extract Text Layer
            # page.get_text('words') returns tuples: (x0, y0, x1, y1, word, block, line, word_num)
            raw_words = page.get_text("words")
            word_count = len(raw_words)
            has_text = word_count > 0
            if has_text:
                pages_with_text_layer += 1

            # Scale word bounding boxes to pixel coordinates
            scale_x = w_px / page.rect.width if page.rect.width > 0 else ZOOM_FACTOR
            scale_y = h_px / page.rect.height if page.rect.height > 0 else ZOOM_FACTOR

            extracted_words: List[Dict[str, Any]] = []
            for w_tuple in raw_words:
                x0, y0, x1, y1, text = w_tuple[0], w_tuple[1], w_tuple[2], w_tuple[3], w_tuple[4]
                sx0, sy0 = round(x0 * scale_x, 2), round(y0 * scale_y, 2)
                sx1, sy1 = round(x1 * scale_x, 2), round(y1 * scale_y, 2)
                extracted_words.append({
                    "text": text,
                    "box": [[sx0, sy0], [sx1, sy0], [sx1, sy1], [sx0, sy1]],
                    "bbox": [sx0, sy0, sx1, sy1],
                    "block": int(w_tuple[5]),
                    "line": int(w_tuple[6]),
                    "word_idx": int(w_tuple[7]),
                })

            with open(text_json_path, "w", encoding="utf-8") as jf:
                json.dump({
                    "pdf_filename": pdf_name,
                    "image_filename": image_name,
                    "page_number": page_num,
                    "page_count": page_count,
                    "width_px": w_px,
                    "height_px": h_px,
                    "has_text_layer": has_text,
                    "word_count": word_count,
                    "words": extracted_words,
                }, jf, indent=2)

            manifest_rows.append({
                "pdf_filename": pdf_name,
                "image_filename": image_name,
                "page_number": page_num,
                "page_count": page_count,
                "width_px": w_px,
                "height_px": h_px,
                "has_text_layer": has_text,
                "word_count": word_count,
            })

        doc.close()

    # 3. Write data/pdf_manifest.csv
    fieldnames = [
        "pdf_filename",
        "image_filename",
        "page_number",
        "page_count",
        "width_px",
        "height_px",
        "has_text_layer",
        "word_count",
    ]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote PDF Manifest to: {MANIFEST_PATH}")

    return {
        "total_pdfs": total_pdfs,
        "total_rendered_pages": total_rendered_pages,
        "pages_with_text_layer": pages_with_text_layer,
        "failed_pdfs": failed_pdfs,
        "encrypted_pdfs": encrypted_pdfs,
        "large_page_pdfs": large_page_pdfs,
        "manifest_count": len(manifest_rows),
    }


def main() -> None:
    print("=" * 75)
    print("CONVERTING OWN PDF INVOICES TO 200 DPI RGB PNG")
    print("=" * 75)

    stats = convert_all_pdfs()

    print("\nConversion Summary:")
    print("-" * 75)
    print(f"Total PDFs Scanned: {stats['total_pdfs']}")
    print(f"Total Rendered PNG Pages: {stats['total_rendered_pages']}")
    print(f"Pages with Extractable Text Layer: {stats['pages_with_text_layer']} / {stats['total_rendered_pages']} ({stats['pages_with_text_layer']/stats['total_rendered_pages']*100:.1f}%)" if stats['total_rendered_pages'] > 0 else "0 pages")
    print(f"PDFs with > 3 Pages: {len(stats['large_page_pdfs'])}")
    for name, count in stats["large_page_pdfs"]:
        print(f"  * {name}: {count} pages")

    print(f"Failed or Password-Protected PDFs: {len(stats['failed_pdfs'])}")
    for name, reason in stats["failed_pdfs"]:
        print(f"  * {name}: {reason}")


if __name__ == "__main__":
    main()
