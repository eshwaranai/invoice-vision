"""Propagate reviewed annotations to synthetic phone photos via perspective transformation.

Transforms reviewed YOLO annotations from data/annotation/exported/ to synthetic
phone photo coordinates using the ground-truth corner mappings in manifest.json.
Calculates axis-aligned bounding box (AABB) loosening metrics and saves an overlay
montage of 6 diverse synthetic samples to results/figures/synth_label_check.jpg.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

CLASS_NAMES = {
    0: "vendor",
    1: "invoice_no",
    2: "date",
    3: "gstin",
    4: "table",
    5: "total",
    6: "qr",
    7: "stamp_signature",
}

CLASS_COLORS = {
    0: (255, 128, 0),    # Blue-ish orange
    1: (0, 200, 0),      # Green
    2: (255, 200, 0),    # Cyan/Yellow
    3: (0, 0, 255),      # Red
    4: (200, 0, 200),    # Magenta
    5: (0, 165, 255),    # Orange
    6: (128, 128, 0),    # Teal
    7: (128, 0, 128),    # Purple
}


def shoelace_area(pts: np.ndarray) -> float:
    """Calculate area of an arbitrary quadrilateral using Shoelace formula."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def load_reviewed_annotations(exported_dir: Path) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    """Load reviewed YOLO annotations from all batches in exported directory."""
    annotations: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
    for txt_path in exported_dir.rglob("*.txt"):
        if txt_path.name in ("classes.txt", "labels.txt"):
            continue
        boxes: List[Tuple[int, float, float, float, float]] = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append((cls_id, xc, yc, w, h))
        stem = txt_path.stem
        # Map stem and filename with .jpg
        annotations[stem] = boxes
        annotations[f"{stem}.jpg"] = boxes
    return annotations


def transform_box_to_synth(
    box: Tuple[int, float, float, float, float],
    H: np.ndarray,
    canv_w: int,
    canv_h: int,
) -> Tuple[Tuple[int, float, float, float, float], float, float]:
    """Transform a normalized box via perspective matrix H and compute AABB loosening.

    Returns:
        transformed_yolo_box: (class_id, xc_norm, yc_norm, w_norm, h_norm)
        quad_area: Area of warped quadrilateral in pixels
        aabb_area: Area of axis-aligned bounding box in pixels
    """
    cls_id, xc, yc, w, h = box

    # 4 corners of original box in normalized coordinates (clockwise from top-left)
    norm_corners = np.array([
        [xc - w / 2.0, yc - h / 2.0],
        [xc + w / 2.0, yc - h / 2.0],
        [xc + w / 2.0, yc + h / 2.0],
        [xc - w / 2.0, yc + h / 2.0],
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to synthetic canvas pixel coordinates
    warped_corners = cv2.perspectiveTransform(norm_corners, H).reshape(-1, 2)

    # Calculate exact quadrilateral area
    quad_area = max(shoelace_area(warped_corners), 1.0)

    # Compute minimal Axis-Aligned Bounding Box (AABB)
    xmin = float(np.min(warped_corners[:, 0]))
    xmax = float(np.max(warped_corners[:, 0]))
    ymin = float(np.min(warped_corners[:, 1]))
    ymax = float(np.max(warped_corners[:, 1]))

    # Clip to canvas boundary
    xmin_clip = max(0.0, min(xmin, float(canv_w - 1)))
    xmax_clip = max(0.0, min(xmax, float(canv_w - 1)))
    ymin_clip = max(0.0, min(ymin, float(canv_h - 1)))
    ymax_clip = max(0.0, min(ymax, float(canv_h - 1)))

    aabb_area = max((xmax_clip - xmin_clip) * (ymax_clip - ymin_clip), 1.0)

    # Convert to normalized YOLO format
    box_w = max(xmax_clip - xmin_clip, 1.0)
    box_h = max(ymax_clip - ymin_clip, 1.0)
    xc_norm = (xmin_clip + xmax_clip) / (2.0 * canv_w)
    yc_norm = (ymin_clip + ymax_clip) / (2.0 * canv_h)
    w_norm = box_w / canv_w
    h_norm = box_h / canv_h

    # Enforce strictly valid YOLO bounds
    xc_norm = min(max(xc_norm, 0.0), 1.0)
    yc_norm = min(max(yc_norm, 0.0), 1.0)
    w_norm = min(max(w_norm, 0.0), 1.0)
    h_norm = min(max(h_norm, 0.0), 1.0)

    return (cls_id, xc_norm, yc_norm, w_norm, h_norm), quad_area, aabb_area


def draw_box_overlay(
    img: np.ndarray,
    boxes: List[Tuple[int, float, float, float, float]],
    title: str,
) -> np.ndarray:
    """Draw bounding boxes and class names onto image."""
    vis = img.copy()
    h, w = vis.shape[:2]

    # Draw header banner
    cv2.rectangle(vis, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    for cls_id, xc, yc, bw, bh in boxes:
        xmin = int((xc - bw / 2.0) * w)
        ymin = int((yc - bh / 2.0) * h)
        xmax = int((xc + bw / 2.0) * w)
        ymax = int((yc + bh / 2.0) * h)

        color = CLASS_COLORS.get(cls_id, (0, 255, 255))
        label = CLASS_NAMES.get(cls_id, str(cls_id))

        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_ymin = max(ymin - 4, text_size[1] + 4)
        cv2.rectangle(vis, (xmin, label_ymin - text_size[1] - 4), (xmin + text_size[0] + 6, label_ymin + 2), color, -1)
        cv2.putText(vis, label, (xmin + 3, label_ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return vis


def create_montage(
    sample_images: List[Tuple[np.ndarray, str]],
    grid_rows: int = 2,
    grid_cols: int = 3,
    tile_size: Tuple[int, int] = (640, 800),  # (w, h)
) -> np.ndarray:
    """Combine sample images into a tiled montage grid."""
    tile_w, tile_h = tile_size
    montage = np.zeros((grid_rows * tile_h, grid_cols * tile_w, 3), dtype=np.uint8)

    for idx, (img, _) in enumerate(sample_images[:grid_rows * grid_cols]):
        r = idx // grid_cols
        c = idx % grid_cols
        resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        montage[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = resized

    return montage


def main() -> None:
    manifest_path = Path("data/processed/synth_photos/manifest.json")
    exported_dir = Path("data/annotation/exported")
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}")
        return
    if not exported_dir.exists():
        print(f"Error: exported annotations not found at {exported_dir}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    reviewed_annotations = load_reviewed_annotations(exported_dir)
    print(f"Loaded reviewed annotations for {len(reviewed_annotations)} source keys.")

    records: List[Dict[str, Any]] = manifest.get("records", [])
    src_unit_square = np.float32([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    labeled_count = 0
    total_boxes_propagated = 0
    loosening_by_severity: Dict[str, List[float]] = {
        "mild": [],
        "medium": [],
        "severe": [],
        "hard": [],
    }

    # Selected candidates for montage (one from each severity + both backgrounds)
    montage_candidates = [
        ("gst_001", "mild", "dark_wood"),
        ("gst_004", "mild", "light_surface"),
        ("gst_005", "medium", "dark_wood"),
        ("gst_015", "medium", "light_surface"),
        ("gst_001", "severe", "dark_wood"),
        ("gst_004", "hard", "light_surface"),
    ]
    montage_samples: List[Tuple[np.ndarray, str]] = []

    for r in records:
        source_img = r["source_image"]
        if source_img not in reviewed_annotations:
            continue

        rel_path = r["relative_path"]
        synth_img_path = Path(rel_path)
        if not synth_img_path.exists():
            continue

        canv_w = int(r["image_width"])
        canv_h = int(r["image_height"])
        dst_corners = np.float32(r["ground_truth_corners"])

        # Perspective transform from unit square to synthetic canvas corners
        H = cv2.getPerspectiveTransform(src_unit_square, dst_corners)

        reviewed_boxes = reviewed_annotations[source_img]
        transferred_boxes: List[Tuple[int, float, float, float, float]] = []

        sev = r.get("severity", "unknown")

        for b in reviewed_boxes:
            t_box, q_area, aabb_area = transform_box_to_synth(b, H, canv_w, canv_h)
            transferred_boxes.append(t_box)
            expansion = aabb_area / q_area
            if sev in loosening_by_severity:
                loosening_by_severity[sev].append(expansion)

        # Write YOLO label file next to synthetic image
        txt_path = synth_img_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f_out:
            f_out.write(f"# AUTO-TRANSFERRED from reviewed labels of {source_img} via perspective transform\n")
            for cls_id, xc, yc, bw, bh in transferred_boxes:
                f_out.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        labeled_count += 1
        total_boxes_propagated += len(transferred_boxes)

        # Check if candidate for montage
        stem_match = Path(source_img).stem
        for m_stem, m_sev, m_bg in montage_candidates:
            if stem_match == m_stem and r.get("severity") == m_sev and r.get("background") == m_bg:
                img_bgr = cv2.imread(str(synth_img_path))
                if img_bgr is not None:
                    title = f"{m_stem} | {m_sev} | {m_bg} ({len(transferred_boxes)} boxes)"
                    vis_img = draw_box_overlay(img_bgr, transferred_boxes, title)
                    montage_samples.append((vis_img, title))
                break

    print(f"\nSuccessfully propagated labels to {labeled_count} synthetic phone photos.")
    print(f"Total boxes propagated: {total_boxes_propagated}")

    print("\n--- AABB Loosening Factor by Degradation Severity ---")
    print("Severity | Sample Count | Mean Loosening Ratio (AABB_area / Quad_area) | Std Dev")
    for sev in ["mild", "medium", "severe", "hard"]:
        vals = loosening_by_severity[sev]
        if vals:
            mean_l = float(np.mean(vals))
            std_l = float(np.std(vals))
            print(f"{sev:<8} | {len(vals):<12} | {mean_l:.4f} ({((mean_l - 1.0) * 100):+.2f}% area expansion) | {std_l:.4f}")
        else:
            print(f"{sev:<8} | 0            | N/A")

    all_vals = [v for sublist in loosening_by_severity.values() for v in sublist]
    if all_vals:
        overall_mean = float(np.mean(all_vals))
        print(f"OVERALL  | {len(all_vals):<12} | {overall_mean:.4f} ({((overall_mean - 1.0) * 100):+.2f}% area expansion) | {float(np.std(all_vals)):.4f}")

    # Generate and save montage
    if montage_samples:
        montage_img = create_montage(montage_samples, grid_rows=2, grid_cols=3)
        out_montage_path = figures_dir / "synth_label_check.jpg"
        cv2.imwrite(str(out_montage_path), montage_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"\nSaved 6-sample overlay montage to: {out_montage_path}")
    else:
        print("\nWarning: No montage samples generated.")


if __name__ == "__main__":
    main()
