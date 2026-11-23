"""Script to generate side-by-side visual comparison progression panels.

Stitches 4 stages into a single horizontal comparison figure:
  1. Degraded Raw Phone Photo
  2. Perspective Rectified (+Warp to canonical 1600 px)
  3. Illumination Equalized (+Shadow Removal)
  4. Local Contrast Enhanced (+CLAHE)
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

FIGURES_DIR = Path("results/figures")


def make_progression_figure(stem: str) -> None:
    f_raw = FIGURES_DIR / f"{stem}_01_raw.jpg"
    f_warp = FIGURES_DIR / f"{stem}_02_warped.jpg"
    f_shadow = FIGURES_DIR / f"{stem}_03_shadow_free.jpg"
    f_clahe = FIGURES_DIR / f"{stem}_04_clahe.jpg"

    for f in [f_raw, f_warp, f_shadow, f_clahe]:
        if not f.exists():
            print(f"Skipping {stem}: {f} does not exist yet.")
            return

    img_raw = cv2.imread(str(f_raw))
    img_warp = cv2.imread(str(f_warp))
    img_shadow = cv2.imread(str(f_shadow))
    img_clahe = cv2.imread(str(f_clahe))

    # Standardize display height to 800 px
    target_h = 800

    def resize_to_h(im: np.ndarray, label: str) -> np.ndarray:
        h, w = im.shape[:2]
        new_w = int(round(w * (target_h / h)))
        resized = cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA)

        # Add top banner for stage title
        banner_h = 45
        banner = np.zeros((banner_h, new_w, 3), dtype=np.uint8) + 40
        cv2.putText(
            banner,
            label,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        stacked = np.vstack([banner, resized])
        # Add 3px border
        bordered = cv2.copyMakeBorder(stacked, 0, 0, 2, 2, cv2.BORDER_CONSTANT, value=[100, 100, 100])
        return bordered

    panel1 = resize_to_h(img_raw, "1. Degraded Raw")
    panel2 = resize_to_h(img_warp, "2. + Warp (1600px)")
    panel3 = resize_to_h(img_shadow, "3. + Shadow Free")
    panel4 = resize_to_h(img_clahe, "4. + CLAHE (Full)")

    progression = np.hstack([panel1, panel2, panel3, panel4])
    out_path = FIGURES_DIR / f"{stem}_progression_panel.jpg"
    cv2.imwrite(str(out_path), progression)
    print(f"Saved side-by-side progression figure to: {out_path}")


def main() -> None:
    for s in ["gst_000_mild_dark_wood", "gst_004_mild_dark_wood"]:
        make_progression_figure(s)


if __name__ == "__main__":
    main()
