"""Derive proposed invoice_no box for own_013_p1.png and save proposal files."""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent

# 1. Paths
PDF_JSON_PATH = ROOT_DIR / "data" / "processed" / "pdf_text" / "own_013_p1.json"
IMG_PATH = ROOT_DIR / "data" / "yolo_v2" / "images" / "test" / "own_013_p1.png"
if not IMG_PATH.exists():
    IMG_PATH = ROOT_DIR / "data" / "raw" / "own" / "own_013_p1.png"

EXPORT_LBL_PATH = ROOT_DIR / "data" / "annotation" / "exported" / "batch_07_test" / "own_013_p1.txt"
YOLO_LBL_PATH = ROOT_DIR / "data" / "yolo_v2" / "labels" / "test" / "own_013_p1.txt"

OUT_DIR = ROOT_DIR / "data" / "processed" / "label_fix_proposal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Read PDF text layer to locate the value next to INVOICE#
with open(PDF_JSON_PATH, "r", encoding="utf-8") as f:
    pdf_data = json.load(f)

width_px = pdf_data["width_px"]
height_px = pdf_data["height_px"]

target_word = None
words = pdf_data.get("words", [])
for i, w in enumerate(words):
    if "INVOICE#" in str(w.get("text", "")).upper():
        if i + 1 < len(words):
            target_word = words[i + 1]
            break

if target_word is None:
    raise ValueError("Could not find word adjacent to 'INVOICE#' in PDF text layer")

bx = target_word["bbox"]  # [xmin, ymin, xmax, ymax]

# 2 px margin per pre-label rule
x0 = max(0.0, bx[0] - 2.0)
y0 = max(0.0, bx[1] - 2.0)
x1 = min(float(width_px), bx[2] + 2.0)
y1 = min(float(height_px), bx[3] + 2.0)

bw = x1 - x0
bh = y1 - y0
xc = (x0 + x1) / (2.0 * width_px)
yc = (y0 + y1) / (2.0 * height_px)
nw = bw / width_px
nh = bh / height_px

proposed_line_yolo = f"1 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
proposed_line_export = f"1 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"

# 3. Check overlap with existing boxes in YOLOv2 label file
yolo_lines = YOLO_LBL_PATH.read_text(encoding="utf-8").splitlines() if YOLO_LBL_PATH.exists() else []
prop_rect = [x0, y0, x1, y1]

overlaps = []
for idx, line in enumerate(yolo_lines):
    parts = line.strip().split()
    if len(parts) < 5:
        continue
    cid = int(parts[0])
    c_xc, c_yc, c_nw, c_nh = [float(x) for x in parts[1:5]]
    c_x1 = (c_xc - c_nw / 2.0) * width_px
    c_y1 = (c_yc - c_nh / 2.0) * height_px
    c_x2 = (c_xc + c_nw / 2.0) * width_px
    c_y2 = (c_yc + c_nh / 2.0) * height_px

    ix1 = max(prop_rect[0], c_x1)
    iy1 = max(prop_rect[1], c_y1)
    ix2 = min(prop_rect[2], c_x2)
    iy2 = min(prop_rect[3], c_y2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    prop_area = (x1 - x0) * (y1 - y0)
    box_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    union_area = prop_area + box_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0

    if inter_area > 0:
        overlaps.append((idx, cid, iou))

# 4. Write proposal files
# (a) Makesense exported format
export_lines = EXPORT_LBL_PATH.read_text(encoding="utf-8").splitlines() if EXPORT_LBL_PATH.exists() else []
new_export_lines = export_lines + [proposed_line_export]
out_export_path = OUT_DIR / "exported_batch_07_test_own_013_p1.txt"
out_export_path.write_text("\n".join(new_export_lines) + "\n", encoding="utf-8")

# (b) Remapped yolo_v2 format
new_yolo_lines = yolo_lines + [proposed_line_yolo]
out_yolo_path = OUT_DIR / "yolo_v2_own_013_p1.txt"
out_yolo_path.write_text("\n".join(new_yolo_lines) + "\n", encoding="utf-8")

# 5. Draw overlay image
img = cv2.imread(str(IMG_PATH))
if img is not None:
    overlay = img.copy()
    class_colors = {
        0: (255, 128, 0),    # vendor: blue-cyan
        1: (0, 0, 255),      # invoice_no: red
        2: (255, 0, 255),    # date: magenta
        3: (0, 255, 0),      # gstin: green
        4: (0, 255, 255),    # table: yellow
        5: (255, 255, 0),    # total: cyan
        6: (128, 128, 128),  # qr: gray
        7: (128, 0, 128),    # stamp: purple
    }
    class_names = {0: "vendor", 1: "invoice_no", 2: "date", 3: "gstin", 4: "table", 5: "total", 6: "qr", 7: "stamp"}

    # Draw existing boxes in blue/green
    for line in yolo_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        c_xc, c_yc, c_nw, c_nh = [float(x) for x in parts[1:5]]
        bx1 = int(round((c_xc - c_nw / 2.0) * width_px))
        by1 = int(round((c_yc - c_nh / 2.0) * height_px))
        bx2 = int(round((c_xc + c_nw / 2.0) * width_px))
        by2 = int(round((c_yc + c_nh / 2.0) * height_px))
        color = class_colors.get(cid, (0, 255, 0))
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 2)
        cname = class_names.get(cid, str(cid))
        cv2.putText(overlay, f"{cname} ({cid})", (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw proposed invoice_no box highlighted in bright red with dashed effect / thicker border
    px0, py0, px1, py1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    cv2.rectangle(overlay, (px0, py0), (px1, py1), (0, 0, 255), 3)
    cv2.putText(overlay, "PROPOSED invoice_no (1)", (px0, max(15, py0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    overlay_out_path = OUT_DIR / "own_013_p1_proposal_overlay.png"
    cv2.imwrite(str(overlay_out_path), overlay)

print("=== PART A: PROPOSED INVOICE_NO BOX FOR own_013_p1.png ===")
print(f"Image Dimensions: {width_px} x {height_px} px")
print(f"Absolute Box (2px margin): [{x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}] px")
print(f"Normalized YOLO Box (class_id=1): xc={xc:.6f}, yc={yc:.6f}, w={nw:.6f}, h={nh:.6f}")
print(f"Overlap with existing boxes: {len(overlaps)} overlaps (Overlaps an existing box: {len(overlaps) > 0})")
print(f"Proposal files written:")
print(f"  - {out_export_path.relative_to(ROOT_DIR)}")
print(f"  - {out_yolo_path.relative_to(ROOT_DIR)}")
print(f"  - {overlay_out_path.relative_to(ROOT_DIR)}")
