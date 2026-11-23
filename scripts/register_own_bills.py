"""Register Personal Bill Images and Detect Near-Duplicates via dHash.

Scans rendered bill images in data/raw/own/:
  - Measures width, height, file size, blur variance, mean brightness.
  - Automatically sets source_type = 'pdf_render'.
  - Leaves vendor, template_family, and has_pii EMPTY (strictly unpopulated).
  - Writes data/own_manifest.csv.
  - Detects near-duplicate images using difference hashing (dHash) with numpy and opencv only.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from invoice_vision.preprocess.quality_gate import compute_blur_score, compute_brightness

OWN_DIR = ROOT_DIR / "data" / "raw" / "own"
MANIFEST_PATH = ROOT_DIR / "data" / "own_manifest.csv"

# dHash parameters
DHASH_SIZE = 8  # 8x8 = 64-bit hash
NEAR_DUP_HAMMING_THRESHOLD = 10  # Hamming distance <= 10 indicates high visual similarity


def compute_dhash(gray: np.ndarray, hash_size: int = DHASH_SIZE) -> np.ndarray:
    """Compute difference hash (dHash) using numpy and opencv only.

    Resizes to (hash_size + 1, hash_size), compares adjacent horizontal pixels.
    Returns 1D boolean numpy array of length hash_size * hash_size.
    """
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def compute_hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """Compute Hamming distance between two boolean hash vectors."""
    return int(np.count_nonzero(h1 != h2))


def register_own_bills() -> Dict[str, Any]:
    image_files = sorted(
        [p for p in OWN_DIR.glob("*.png")] + [p for p in OWN_DIR.glob("*.jpg")] + [p for p in OWN_DIR.glob("*.jpeg")],
        key=lambda p: p.name.lower(),
    )

    records: List[Dict[str, Any]] = []
    hashes: Dict[str, np.ndarray] = {}

    for img_path in image_files:
        fn = img_path.name
        fsize_kb = round(img_path.stat().st_size / 1024.0, 2)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: could not read {fn}")
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        blur = round(compute_blur_score(img), 2)
        bright = round(compute_brightness(img), 2)

        # Compute dHash for duplicate detection
        dhash = compute_dhash(gray, hash_size=DHASH_SIZE)
        hashes[fn] = dhash

        records.append({
            "filename": fn,
            "width": w,
            "height": h,
            "file_size_kb": fsize_kb,
            "blur_score": blur,
            "mean_brightness": bright,
            "vendor": "",           # Leave EMPTY per specification
            "template_family": "",  # Leave EMPTY per specification
            "source_type": "pdf_render",
            "has_pii": "",          # Leave EMPTY per specification
        })

    # Write data/own_manifest.csv
    fieldnames = [
        "filename",
        "width",
        "height",
        "file_size_kb",
        "blur_score",
        "mean_brightness",
        "vendor",
        "template_family",
        "source_type",
        "has_pii",
    ]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} entries to {MANIFEST_PATH}")

    # Pairwise near-duplicate detection
    near_duplicates: List[Tuple[str, str, int]] = []
    file_list = list(hashes.keys())
    for i in range(len(file_list)):
        for j in range(i + 1, len(file_list)):
            f1 = file_list[i]
            f2 = file_list[j]
            dist = compute_hamming_distance(hashes[f1], hashes[f2])
            if dist <= NEAR_DUP_HAMMING_THRESHOLD:
                near_duplicates.append((f1, f2, dist))

    near_duplicates.sort(key=lambda x: x[2])

    return {
        "total_registered": len(records),
        "manifest_path": str(MANIFEST_PATH),
        "near_duplicates": near_duplicates,
        "records": records,
    }


def main() -> None:
    print("=" * 75)
    print("REGISTERING OWN BILL IMAGES & DHASH DUPLICATE DETECTION")
    print("=" * 75)

    res = register_own_bills()

    print("\nRegistration Summary:")
    print("-" * 75)
    print(f"Total Registered Images: {res['total_registered']}")
    print(f"Manifest Path: {res['manifest_path']}")

    near_dups = res["near_duplicates"]
    print(f"\nNear-Duplicate Pairs Found (Hamming Distance <= {NEAR_DUP_HAMMING_THRESHOLD} / 64 bits): {len(near_dups)}")
    print("-" * 75)
    if not near_dups:
        print("  No near-duplicate pairs detected.")
    else:
        for f1, f2, dist in near_dups:
            sim_pct = ((64 - dist) / 64.0) * 100.0
            print(f"  * {f1:<16} <--> {f2:<16} (Hamming Distance: {dist:2d}/64, Visual Match: {sim_pct:5.1f}%)")


if __name__ == "__main__":
    main()
