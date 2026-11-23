"""Script to generate realistic synthetic phone photos of paper invoices (v2).

Features:
  - 2.5x bicubic upscaling of source page to imitate camera sensor resolution.
  - Placed on canvas so page fills 75-88% of canvas WITHOUT downscaling text.
  - Realistic camera degradations:
      * Blur sigma: 0.35 - 0.95
      * JPEG quality: 76 - 94
      * Sensor noise sigma: 2.0 - 5.5
  - Three severity tiers:
      * mild: tilt <= 8 deg, rot <= 3 deg, blur 0.35, noise 2.0, JPEG Q92
      * medium: tilt <= 15 deg, rot <= 6 deg, blur 0.60, noise 3.5, JPEG Q85
      * severe: tilt <= 25 deg, rot <= 9 deg, blur 0.95, noise 5.5, JPEG Q76
  - Two background surfaces:
      * 'dark_wood': high-contrast dark wooden desk surface
      * 'light_surface': low-contrast light surface (bed sheet / light table)
  - Saves 72 synthetic images (12 images x 3 severities x 2 backgrounds)
  - Manifest with ground-truth corners and exact parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

# Add workspace root and src/
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

RAW_DIR = Path("data/raw/indian_gst")
GROUND_TRUTH_CSV = Path("data/ground_truth_gst.csv")
OUTPUT_BASE_DIR = Path("data/processed/synth_photos")
MANIFEST_PATH = OUTPUT_BASE_DIR / "manifest.json"

RANDOM_SEED = 42

TARGET_IMAGES = [
    ("gst_000.jpg", "Senate Space"),
    ("gst_001.jpg", "Linkedin Singapore Pte. Ltd"),
    ("gst_004.jpg", "Microsoft / Microsoft India"),
    ("gst_005.jpg", "Microsoft / Microsoft India"),
    ("gst_012.jpg", "Net2Source LLP"),
    ("gst_014.jpg", "Bansal Electrovision Pvt. Ltd"),
    ("gst_015.jpg", "Sai Computer Solution"),
    ("gst_020.jpg", "SBI General Insurance Company Limited"),
    ("gst_021.jpg", "SIMEQ Technologies LLP"),
    ("gst_024.jpg", "Amazon - Retailez Private Limited"),
    ("gst_026.jpg", "Amazon - Whole9Yards Online LLP"),
    ("gst_028.jpg", "Amazon - Altaf H Vali Electronics Pvt Ltd"),
]

BACKGROUND_TYPES = ["dark_wood", "light_surface"]

SEVERITY_PROFILES = {
    "mild": {
        "fill_ratio": (0.82, 0.88),
        "max_tilt_deg": 6.5,
        "max_rot_deg": 2.5,
        "blur_sigma": 0.35,
        "shadow_opacity": 0.15,
        "lighting_scale": 0.96,
        "noise_sigma": 2.0,
        "jpeg_quality": 92,
    },
    "medium": {
        "fill_ratio": (0.78, 0.84),
        "max_tilt_deg": 12.0,
        "max_rot_deg": 5.0,
        "blur_sigma": 0.60,
        "shadow_opacity": 0.28,
        "lighting_scale": 0.88,
        "noise_sigma": 3.5,
        "jpeg_quality": 85,
    },
    "severe": {
        "fill_ratio": (0.74, 0.80),
        "max_tilt_deg": 20.0,
        "max_rot_deg": 8.0,
        "blur_sigma": 0.95,
        "shadow_opacity": 0.45,
        "lighting_scale": 0.74,
        "noise_sigma": 5.5,
        "jpeg_quality": 76,
        "downscale_width": None,
    },
    "hard": {
        "fill_ratio": (0.72, 0.78),
        "max_tilt_deg": 22.0,
        "max_rot_deg": 10.0,
        "blur_sigma": 3.0,
        "shadow_opacity": 0.55,
        "lighting_scale": 0.35,
        "noise_sigma": 6.5,
        "jpeg_quality": 50,
        "downscale_width": 800,
    },
}


def generate_background(w: int, h: int, bg_type: str, rng: np.random.RandomState) -> np.ndarray:
    """Procedurally synthesize background surface."""
    y_coords = np.arange(h, dtype=np.float32)[:, None]
    x_coords = np.arange(w, dtype=np.float32)[None, :]

    if bg_type == "dark_wood":
        base_color = np.array([40.0, 70.0, 110.0], dtype=np.float32)  # Wood brown BGR
        grain = np.sin(y_coords * 0.025) * 14.0
        grad = ((x_coords / max(1, w)) * 15.0) + ((y_coords / max(1, h)) * 12.0)
        noise = rng.normal(0.0, 4.0, (h, w, 3)).astype(np.float32)
        bg = base_color + grain[:, :, None] + grad[:, :, None] + noise
        return np.clip(bg, 0, 255).astype(np.uint8)

    elif bg_type == "light_surface":
        # Off-white / light beige / light gray bedsheet or table (mean ~220 BGR)
        base_color = np.array([215.0, 220.0, 225.0], dtype=np.float32)
        # Subtle organic cloth folds / creases
        wrinkles = (np.sin(y_coords * 0.012 + x_coords * 0.008) * 7.0) + (
            np.cos(x_coords * 0.015 - y_coords * 0.005) * 5.0
        )
        grad = ((x_coords / max(1, w)) * 8.0) - ((y_coords / max(1, h)) * 6.0)
        noise = rng.normal(0.0, 3.0, (h, w, 3)).astype(np.float32)
        bg = base_color + wrinkles[:, :, None] + grad[:, :, None] + noise
        return np.clip(bg, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown background type: {bg_type}")


def generate_soft_shadow(
    w: int, h: int, opacity: float, rng: np.random.RandomState
) -> np.ndarray:
    """Generate smooth phone/hand shadow overlay."""
    mask = np.zeros((h, w), dtype=np.float32)
    cx = float(rng.uniform(w * 0.15, w * 0.45))
    cy = float(rng.uniform(h * 0.10, h * 0.35))
    axes = (int(w * rng.uniform(0.35, 0.50)), int(h * rng.uniform(0.30, 0.45)))
    angle = float(rng.uniform(20.0, 50.0))

    cv2.ellipse(mask, (int(cx), int(cy)), axes, angle, 0, 360, 1.0, -1)
    k_size = int(max(w, h) * 0.20) | 1
    blurred = cv2.GaussianBlur(mask, (k_size, k_size), sigmaX=k_size / 3.0)
    blurred = (blurred / (blurred.max() + 1e-6)) * opacity
    return blurred


def generate_illumination_map(
    w: int, h: int, base_scale: float, rng: np.random.RandomState
) -> np.ndarray:
    """Radial lighting falloff simulating ambient overhead lighting."""
    light_x = float(rng.uniform(w * 0.2, w * 0.8))
    light_y = float(rng.uniform(0.0, h * 0.25))

    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = (x_grid - light_x) ** 2 + (y_grid - light_y) ** 2
    max_dist_sq = float(w**2 + h**2)

    falloff = 1.0 - (dist_sq / max_dist_sq) * 0.35
    lighting = base_scale * falloff
    return np.clip(lighting, 0.3, 1.15).astype(np.float32)


def synthesize_phone_photo(
    clean_image: np.ndarray,
    severity: str,
    bg_type: str,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, List[List[float]], Dict[str, Any]]:
    """Synthesize realistic phone photo of invoice paper."""
    params = SEVERITY_PROFILES[severity]

    # 1. First upscale source page 2.5x with bicubic interpolation
    src_h, src_w = clean_image.shape[:2]
    up_w = int(src_w * 2.5)
    up_h = int(src_h * 2.5)
    upscaled = cv2.resize(clean_image, (up_w, up_h), interpolation=cv2.INTER_CUBIC)

    # 2. Place on canvas so the page fills 75-88% of canvas without downscaling
    fill = float(rng.uniform(params["fill_ratio"][0], params["fill_ratio"][1]))
    canv_w = int(up_w / fill)
    canv_h = int(up_h / fill)

    bg = generate_background(canv_w, canv_h, bg_type, rng)

    pad_x = (canv_w - up_w) / 2.0 + float(rng.uniform(-canv_w * 0.02, canv_w * 0.02))
    pad_y = (canv_h - up_h) / 2.0 + float(rng.uniform(-canv_h * 0.02, canv_h * 0.02))

    # Base rectangle
    tl = np.array([pad_x, pad_y], dtype=np.float32)
    tr = np.array([pad_x + up_w - 1, pad_y], dtype=np.float32)
    br = np.array([pad_x + up_w - 1, pad_y + up_h - 1], dtype=np.float32)
    bl = np.array([pad_x, pad_y + up_h - 1], dtype=np.float32)

    # 3. Apply perspective tilt and rotation
    tilt_rad = np.radians(float(rng.uniform(-params["max_tilt_deg"], params["max_tilt_deg"])))
    rot_rad = np.radians(float(rng.uniform(-params["max_rot_deg"], params["max_rot_deg"])))

    tilt_shift_x = float(np.sin(tilt_rad) * up_w * 0.12)
    tilt_shift_y = float(np.sin(tilt_rad) * up_h * 0.10)

    tl += np.array([tilt_shift_x, tilt_shift_y], dtype=np.float32)
    tr += np.array([-tilt_shift_x, tilt_shift_y], dtype=np.float32)
    br += np.array([tilt_shift_x * 0.5, -tilt_shift_y], dtype=np.float32)
    bl += np.array([-tilt_shift_x * 0.5, -tilt_shift_y], dtype=np.float32)

    center = np.array([canv_w / 2.0, canv_h / 2.0], dtype=np.float32)
    cos_r = np.cos(rot_rad)
    sin_r = np.sin(rot_rad)
    rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)

    dst_pts = np.zeros((4, 2), dtype=np.float32)
    for i, pt in enumerate([tl, tr, br, bl]):
        dst_pts[i] = center + rot_mat.dot(pt - center)

    # 4. Warp page onto canvas
    src_pts = np.float32([[0, 0], [up_w - 1, 0], [up_w - 1, up_h - 1], [0, up_h - 1]])
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_page = cv2.warpPerspective(
        upscaled, H, (canv_w, canv_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    mask = np.ones((up_h, up_w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        mask, H, (canv_w, canv_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    mask_smooth = cv2.GaussianBlur(warped_mask.astype(np.float32) / 255.0, (3, 3), 0.5)[:, :, None]
    composite = (warped_page.astype(np.float32) * mask_smooth) + (bg.astype(np.float32) * (1.0 - mask_smooth))

    # 5. Soft shadow overlay
    shadow_mask = generate_soft_shadow(canv_w, canv_h, params["shadow_opacity"], rng)
    composite = composite * (1.0 - shadow_mask[:, :, None])

    # 6. Illumination falloff
    lighting = generate_illumination_map(canv_w, canv_h, params["lighting_scale"], rng)
    composite = composite * lighting[:, :, None]

    # 7. Lens blur
    b_sigma = params["blur_sigma"]
    if b_sigma > 0.1:
        k = int(b_sigma * 4) | 1
        composite = cv2.GaussianBlur(composite, (k, k), sigmaX=b_sigma)

    # 8. Sensor noise
    n_sigma = params["noise_sigma"]
    if n_sigma > 0.0:
        noise = rng.normal(0.0, n_sigma, composite.shape).astype(np.float32)
        composite = np.clip(composite + noise, 0, 255)

    comp_u8 = np.clip(composite, 0, 255).astype(np.uint8)

    # 8b. Downscale to target width if specified (for hard tier)
    downscale_w = params.get("downscale_width")
    if downscale_w is not None:
        target_w = int(downscale_w)
        scale_factor = target_w / float(comp_u8.shape[1])
        target_h = int(round(comp_u8.shape[0] * scale_factor))
        comp_u8 = cv2.resize(comp_u8, (target_w, target_h), interpolation=cv2.INTER_AREA)
        dst_pts = dst_pts * scale_factor

    # 9. JPEG compression
    jq = params["jpeg_quality"]
    _, enc = cv2.imencode(".jpg", comp_u8, [cv2.IMWRITE_JPEG_QUALITY, jq])
    synth_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    corners_list = [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in dst_pts]

    meta = {
        "bg_type": bg_type,
        "upscale_factor": 2.5,
        "fill_ratio": round(fill, 3),
        "tilt_deg": round(float(np.degrees(tilt_rad)), 2),
        "rot_deg": round(float(np.degrees(rot_rad)), 2),
        "blur_sigma": b_sigma,
        "shadow_opacity": params["shadow_opacity"],
        "lighting_scale": params["lighting_scale"],
        "noise_sigma": n_sigma,
        "jpeg_quality": jq,
        "downscale_width": downscale_w,
    }
    return synth_img, corners_list, meta


def main() -> None:
    print("=" * 80)
    print("SYNTHETIC PHONE PHOTO DATASET GENERATION (v2)")
    print(f"Fixed Random Seed: {RANDOM_SEED}")
    print("=" * 80)

    rng = np.random.RandomState(RANDOM_SEED)

    # Ensure output directories exist for all severities & backgrounds
    for sev in SEVERITY_PROFILES.keys():
        for bg in BACKGROUND_TYPES:
            (OUTPUT_BASE_DIR / f"{sev}_{bg}").mkdir(parents=True, exist_ok=True)

    df_gt = pd.read_csv(GROUND_TRUTH_CSV).set_index("filename")
    manifest_records: List[Dict[str, Any]] = []

    for fn, vendor_grp in TARGET_IMAGES:
        src_path = RAW_DIR / fn
        if not src_path.exists():
            continue
        clean_img = cv2.imread(str(src_path))
        if clean_img is None:
            continue

        gt_row = df_gt.loc[fn].to_dict() if fn in df_gt.index else {}
        stem = Path(fn).stem

        print(f"\nProcessing {fn} ({vendor_grp}):")

        for sev in ["mild", "medium", "severe", "hard"]:
            for bg in BACKGROUND_TYPES:
                synth_img, corners, meta = synthesize_phone_photo(clean_img, sev, bg, rng)

                out_fn = f"{stem}_{sev}_{bg}.jpg"
                out_path = OUTPUT_BASE_DIR / f"{sev}_{bg}" / out_fn
                cv2.imwrite(str(out_path), synth_img)

                record = {
                    "id": f"{stem}_{sev}_{bg}",
                    "filename": out_fn,
                    "relative_path": str(out_path.as_posix()),
                    "source_image": fn,
                    "vendor_group": vendor_grp,
                    "severity": sev,
                    "background": bg,
                    "is_synthetic": True,
                    "image_width": synth_img.shape[1],
                    "image_height": synth_img.shape[0],
                    "ground_truth_corners": corners,
                    "degradation_parameters": meta,
                    "ground_truth_invoice": {
                        "invoice_number": str(gt_row.get("invoice_number", "")),
                        "purchase_date": str(gt_row.get("purchase_date", "")),
                        "supplier_gst": str(gt_row.get("supplier_gst", "")),
                        "total_amount": str(gt_row.get("total_amount", "")),
                    },
                }
                manifest_records.append(record)
                print(f"  [{sev.upper():<6} | {bg:<13}] -> {out_fn} (canvas: {synth_img.shape[1]}x{synth_img.shape[0]}, blur: {meta['blur_sigma']}, Q{meta['jpeg_quality']})")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "random_seed": RANDOM_SEED,
                "total_synthetic_images": len(manifest_records),
                "severities": list(SEVERITY_PROFILES.keys()),
                "background_types": BACKGROUND_TYPES,
                "target_images_count": len(TARGET_IMAGES),
                "records": manifest_records,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print(f"Successfully generated {len(manifest_records)} synthetic images across 3 severities and 2 backgrounds.")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
