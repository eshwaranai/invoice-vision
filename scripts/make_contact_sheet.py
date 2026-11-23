"""Generate Obfuscated Contact Sheet for Own Bill Template Families.

Generates results/own_contact_sheet.png:
  - 4-column grid displaying the representative first page of each template family (T01..T13).
  - Thumbnails scaled to ~250 px width with Gaussian blur (sigma=5.0) to preserve layout structure while obfuscating text for privacy.
  - Captioned with family ID, suggested vendor, and total page count.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_OWN_DIR = ROOT_DIR / "data" / "raw" / "own"
MANIFEST_PATH = ROOT_DIR / "data" / "own_manifest.csv"
OUTPUT_PATH = ROOT_DIR / "results" / "own_contact_sheet.png"

THUMB_WIDTH = 250
GRID_COLS = 4
CELL_PADDING = 12
HEADER_HEIGHT = 44
BORDER_WIDTH = 1


def create_contact_sheet() -> Path:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Group by template family
    families = sorted(df["template_family_suggested"].unique())
    num_families = len(families)
    num_rows = math.ceil(num_families / GRID_COLS)

    print(f"Generating contact sheet for {num_families} template families across {num_rows} rows...")

    # Load and prepare thumbnails
    cells: List[Tuple[str, str, int, np.ndarray]] = []
    max_thumb_h = 350

    for fam_id in families:
        fam_df = df[df["template_family_suggested"] == fam_id]
        page_count = len(fam_df)
        first_row = fam_df.iloc[0]
        img_fn = first_row["filename"]
        vendor_name = str(first_row.get("vendor_suggested", "Unknown Vendor"))

        img_path = RAW_OWN_DIR / img_fn
        img = cv2.imread(str(img_path))
        if img is None:
            # Fallback blank placeholder
            thumb = np.ones((max_thumb_h, THUMB_WIDTH, 3), dtype=np.uint8) * 230
        else:
            h, w = img.shape[:2]
            aspect = h / float(w)
            thumb_h = int(THUMB_WIDTH * aspect)
            thumb_h = min(max(thumb_h, 300), 360)
            thumb = cv2.resize(img, (THUMB_WIDTH, thumb_h), interpolation=cv2.INTER_AREA)

            # Apply light Gaussian blur so layout is preserved but text is unreadable
            thumb = cv2.GaussianBlur(thumb, (17, 17), sigmaX=5.5, sigmaY=5.5)

            # Draw 1px border around page thumbnail
            cv2.rectangle(thumb, (0, 0), (THUMB_WIDTH - 1, thumb.shape[0] - 1), (180, 180, 180), BORDER_WIDTH)

        max_thumb_h = max(max_thumb_h, thumb.shape[0])
        cells.append((fam_id, vendor_name, page_count, thumb))

    cell_w = THUMB_WIDTH + (2 * CELL_PADDING)
    cell_h = max_thumb_h + HEADER_HEIGHT + (2 * CELL_PADDING)

    canvas_w = cell_w * GRID_COLS
    canvas_h = cell_h * num_rows

    # Canvas with clean off-white background
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 242

    for idx, (fam_id, vendor_name, page_count, thumb) in enumerate(cells):
        r = idx // GRID_COLS
        c = idx % GRID_COLS

        x0 = c * cell_w
        y0 = r * cell_h

        # Cell card background
        card_x1 = x0 + 4
        card_y1 = y0 + 4
        card_x2 = x0 + cell_w - 4
        card_y2 = y0 + cell_h - 4
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (255, 255, 255), -1)
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (215, 215, 215), 1)

        # Header caption: Family ID pill + Vendor Name + Count
        caption_fam = f"[{fam_id}]"
        # Shorten vendor to fit cell
        short_vendor = vendor_name
        if len(short_vendor) > 22:
            short_vendor = short_vendor[:20] + ".."
        caption_text = f"{caption_fam} {short_vendor}"
        caption_sub = f"Pages: {page_count}"

        text_x = card_x1 + CELL_PADDING
        text_y1 = card_y1 + 18
        text_y2 = card_y1 + 34

        cv2.putText(canvas, caption_text, (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 30, 45), 1, cv2.LINE_AA)
        cv2.putText(canvas, caption_sub, (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 110, 120), 1, cv2.LINE_AA)

        # Place blurred thumbnail
        th_h, th_w = thumb.shape[:2]
        thumb_x = card_x1 + CELL_PADDING
        thumb_y = card_y1 + HEADER_HEIGHT + 4

        # Center vertically if shorter than max_thumb_h
        y_offset = (max_thumb_h - th_h) // 2
        dest_y = thumb_y + y_offset

        canvas[dest_y : dest_y + th_h, thumb_x : thumb_x + th_w] = thumb

    cv2.imwrite(str(OUTPUT_PATH), canvas)
    print(f"Saved contact sheet to: {OUTPUT_PATH} ({canvas_w}x{canvas_h} px)")
    return OUTPUT_PATH


def main() -> None:
    print("=" * 75)
    print("GENERATING OWN INVOICES CONTACT SHEET (Privacy-Preserving Layout Grid)")
    print("=" * 75)

    out_p = create_contact_sheet()
    print(f"Successfully generated contact sheet at: {out_p}")


if __name__ == "__main__":
    main()
