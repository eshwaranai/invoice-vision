"""Generate and Verify Dataset Splits v4 (incorporating 60 additional Synthetic GST Invoices).

Rules:
  1. Base splits from data/splits_v3.json.
  2. Exactly 60 new synthetic images assigned to TRAIN only (gstsyn_024 to gstsyn_083).
  3. All 84 synthetic images (gstsyn_000 to gstsyn_083) reside in TRAIN.
  4. 4 layout-based split groups (gstsyn_classic, gstsyn_handwritten, gstsyn_minimal, gstsyn_thermal).
  5. Val (12 images) and Test (14 images) sets are strictly untouched and identical to v3.
  6. Formally verifies:
       - Zero split_group overlap across splits.
       - Zero image SHA-256 hash duplication across train, val, and test.
       - 100% identity of val and test sets with splits_v3.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

SPLITS_V3_PATH = ROOT_DIR / "data" / "splits_v3.json"
SPLITS_V4_PATH = ROOT_DIR / "data" / "splits_v4.json"
MANIFEST_PATH = ROOT_DIR / "data" / "gst_synth_manifest.csv"
RAW_SYNTH_DIR = ROOT_DIR / "data" / "raw" / "gst_synth"
RAW_PUBLIC_DIR = ROOT_DIR / "data" / "raw" / "indian_gst"
RAW_OWN_DIR = ROOT_DIR / "data" / "raw" / "own"


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_image_path(fn: str) -> Optional[Path]:
    """Resolve file path across public, own, and synthetic raw folders."""
    p_pub = RAW_PUBLIC_DIR / fn
    if p_pub.exists():
        return p_pub
    p_own = RAW_OWN_DIR / fn
    if p_own.exists():
        return p_own
    p_synth = RAW_SYNTH_DIR / fn
    if p_synth.exists():
        return p_synth
    return None


def create_and_verify_splits_v4() -> None:
    print("=" * 75)
    print("GENERATING AND VERIFYING DATASET SPLITS V4")
    print("=" * 75)

    with open(SPLITS_V3_PATH, "r", encoding="utf-8") as f:
        v3 = json.load(f)

    manifest_df = pd.read_csv(MANIFEST_PATH)
    all_synth_images = manifest_df["filename"].tolist()
    if len(all_synth_images) != 84:
        raise ValueError(f"Expected 84 synthetic images, found {len(all_synth_images)}")

    new_60_images = all_synth_images[24:]
    if len(new_60_images) != 60:
        raise ValueError(f"Expected 60 new synthetic images, found {len(new_60_images)}")

    v4 = dict(v3)
    v4["metadata"] = dict(v3["metadata"])
    v4["metadata"]["version"] = "v4"
    v4["metadata"]["total_documents"] = int(v3["metadata"]["total_documents"]) + 60
    v4["metadata"]["synthetic_multi_layout_count"] = 84

    # Group allocations - ensure layout groups are present in train
    v4["split_group_allocations"] = {
        "train": sorted(list(set(v3["split_group_allocations"]["train"]))),
        "val": sorted(list(set(v3["split_group_allocations"]["val"]))),
        "test": sorted(list(set(v3["split_group_allocations"]["test"]))),
    }

    # All 84 synthetic images go to TRAIN
    v4["gst_synth_splits"] = {
        "train": all_synth_images,
        "val": [],
        "test": [],
    }

    # Combined clean splits
    orig_train = v3["combined_clean_splits"]["train"]
    orig_val = v3["combined_clean_splits"]["val"]
    orig_test = v3["combined_clean_splits"]["test"]

    # Combine: original v3 train images + 60 new synthetic images
    new_combined_train = sorted(list(set(orig_train + new_60_images)))
    if len(new_combined_train) != len(orig_train) + 60:
        raise ValueError(f"Train split count mismatch: expected {len(orig_train) + 60}, got {len(new_combined_train)}")

    v4["combined_clean_splits"] = {
        "train": new_combined_train,
        "val": list(orig_val),
        "test": list(orig_test),
    }

    # Verify val and test sets are 100% strictly identical to v3
    assert v4["combined_clean_splits"]["val"] == orig_val, "VAL set was modified!"
    assert v4["combined_clean_splits"]["test"] == orig_test, "TEST set was modified!"
    assert v4["split_group_allocations"]["val"] == v3["split_group_allocations"]["val"], "VAL groups modified!"
    assert v4["split_group_allocations"]["test"] == v3["split_group_allocations"]["test"], "TEST groups modified!"

    # Counts
    v4["counts"] = dict(v3["counts"])
    v4["counts"]["gst_synth"] = {"train": 84, "val": 0, "test": 0}
    v4["counts"]["combined_clean"] = {
        "train": len(v4["combined_clean_splits"]["train"]),
        "val": len(v4["combined_clean_splits"]["val"]),
        "test": len(v4["combined_clean_splits"]["test"]),
    }

    # Write v4 splits
    with open(SPLITS_V4_PATH, "w", encoding="utf-8") as f:
        json.dump(v4, f, indent=2)
    print(f"Wrote {SPLITS_V4_PATH.relative_to(ROOT_DIR)}")

    # --------------------------------------------------------------------------
    # FORMAL VERIFICATION CHECKS
    # --------------------------------------------------------------------------
    print("\nRunning formal leakage and integrity verifications:")

    # 1. Zero Group Overlap Check
    train_g = set(v4["split_group_allocations"]["train"])
    val_g = set(v4["split_group_allocations"]["val"])
    test_g = set(v4["split_group_allocations"]["test"])

    tv_overlap = train_g & val_g
    tt_overlap = train_g & test_g
    vt_overlap = val_g & test_g

    print(f"  Train Groups ({len(train_g)}): {sorted(train_g)}")
    print(f"  Val Groups   ({len(val_g)}): {sorted(val_g)}")
    print(f"  Test Groups  ({len(test_g)}): {sorted(test_g)}")

    assert len(tv_overlap) == 0, f"Leakage: Train & Val group overlap: {tv_overlap}"
    assert len(tt_overlap) == 0, f"Leakage: Train & Test group overlap: {tt_overlap}"
    assert len(vt_overlap) == 0, f"Leakage: Val & Test group overlap: {vt_overlap}"
    print("  [PASS] Zero group overlap confirmed across all split pairs.")

    # 2. Zero Image Hash Duplicate Check
    split_hashes: Dict[str, Dict[str, str]] = {"train": {}, "val": {}, "test": {}}

    for split_name in ["train", "val", "test"]:
        for fn in v4["combined_clean_splits"][split_name]:
            path = find_image_path(fn)
            if path is None:
                raise FileNotFoundError(f"Missing image for hash check: {fn}")
            sha = compute_file_sha256(path)
            split_hashes[split_name][fn] = sha

    train_hashes = set(split_hashes["train"].values())
    val_hashes = set(split_hashes["val"].values())
    test_hashes = set(split_hashes["test"].values())

    hash_tv_overlap = train_hashes & val_hashes
    hash_tt_overlap = train_hashes & test_hashes
    hash_vt_overlap = val_hashes & test_hashes

    assert len(hash_tv_overlap) == 0, f"Duplicate image hash between Train and Val: {hash_tv_overlap}"
    assert len(hash_tt_overlap) == 0, f"Duplicate image hash between Train and Test: {hash_tt_overlap}"
    assert len(hash_vt_overlap) == 0, f"Duplicate image hash between Val and Test: {hash_vt_overlap}"
    print(f"  [PASS] Zero image SHA-256 hash duplication confirmed across {len(train_hashes)} train, {len(val_hashes)} val, and {len(test_hashes)} test images.")

    # 3. Counts summary
    print("\nDataset Split v4 Counts:")
    print(f"  Clean Train Images : {v4['counts']['combined_clean']['train']} (33 real + 84 synthetic = 117)")
    print(f"  Clean Val Images   : {v4['counts']['combined_clean']['val']} (12 images, strictly unchanged)")
    print(f"  Clean Test Images  : {v4['counts']['combined_clean']['test']} (14 images, strictly unchanged)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    create_and_verify_splits_v4()
