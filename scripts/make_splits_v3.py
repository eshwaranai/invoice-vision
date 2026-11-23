"""Generate and Verify Dataset Splits v3 (incorporating Synthetic Multi-Layout Invoices).

Rules:
  1. Everything in data/splits_v2.json remains unchanged.
  2. Exactly 24 new synthetic images assigned to TRAIN only.
  3. One new split_group per layout:
       - gstsyn_classic
       - gstsyn_handwritten
       - gstsyn_minimal
       - gstsyn_thermal
  4. Val and Test sets are strictly untouched.
  5. Formally verifies:
       - Zero split_group overlap across splits.
       - Zero image SHA-256 hash duplication across train, val, and test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

SPLITS_V2_PATH = ROOT_DIR / "data" / "splits_v2.json"
SPLITS_V3_PATH = ROOT_DIR / "data" / "splits_v3.json"
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


def create_and_verify_splits_v3() -> None:
    print("=" * 75)
    print("GENERATING AND VERIFYING DATASET SPLITS V3")
    print("=" * 75)

    with open(SPLITS_V2_PATH, "r", encoding="utf-8") as f:
        v2 = json.load(f)

    manifest_df = pd.read_csv(MANIFEST_PATH)
    new_train_images = manifest_df["filename"].tolist()
    if len(new_train_images) != 24:
        raise ValueError(f"Expected 24 synthetic images, found {len(new_train_images)}")

    # 4 new layout-based split groups
    new_groups = {
        "classic": "gstsyn_classic",
        "handwritten": "gstsyn_handwritten",
        "minimal": "gstsyn_minimal",
        "thermal": "gstsyn_thermal",
    }

    v3 = dict(v2)
    v3["metadata"] = dict(v2["metadata"])
    v3["metadata"]["version"] = "v3"
    v3["metadata"]["total_documents"] = int(v2["metadata"]["total_documents"]) + 24
    v3["metadata"]["synthetic_multi_layout_count"] = 24

    # Group allocations
    orig_train_groups = set(v2["split_group_allocations"]["train"])
    orig_val_groups = set(v2["split_group_allocations"]["val"])
    orig_test_groups = set(v2["split_group_allocations"]["test"])

    v3["split_group_allocations"] = {
        "train": sorted(list(orig_train_groups | set(new_groups.values()))),
        "val": sorted(list(orig_val_groups)),
        "test": sorted(list(orig_test_groups)),
    }

    # Add gst_synth_splits
    v3["gst_synth_splits"] = {
        "train": new_train_images,
        "val": [],
        "test": [],
    }

    # Combined clean splits
    orig_combined = v2["combined_clean_splits"]
    v3["combined_clean_splits"] = {
        "train": sorted(orig_combined["train"] + new_train_images),
        "val": sorted(orig_combined["val"]),
        "test": sorted(orig_combined["test"]),
    }

    # Verify val and test sets are 100% strictly identical
    assert sorted(v3["combined_clean_splits"]["val"]) == sorted(orig_combined["val"]), "VAL set was modified!"
    assert sorted(v3["combined_clean_splits"]["test"]) == sorted(orig_combined["test"]), "TEST set was modified!"
    assert sorted(v3["split_group_allocations"]["val"]) == sorted(v2["split_group_allocations"]["val"]), "VAL groups modified!"
    assert sorted(v3["split_group_allocations"]["test"]) == sorted(v2["split_group_allocations"]["test"]), "TEST groups modified!"

    # Counts
    v3["counts"] = dict(v2["counts"])
    v3["counts"]["gst_synth"] = {"train": 24, "val": 0, "test": 0}
    v3["counts"]["combined_clean"] = {
        "train": len(v3["combined_clean_splits"]["train"]),
        "val": len(v3["combined_clean_splits"]["val"]),
        "test": len(v3["combined_clean_splits"]["test"]),
    }

    # Write v3 splits
    with open(SPLITS_V3_PATH, "w", encoding="utf-8") as f:
        json.dump(v3, f, indent=2)
    print(f"Wrote {SPLITS_V3_PATH.relative_to(ROOT_DIR)}")

    # --------------------------------------------------------------------------
    # FORMAL VERIFICATION CHECKS
    # --------------------------------------------------------------------------
    print("\nRunning formal leakage and integrity verifications:")

    # 1. Zero Group Overlap Check
    train_g = set(v3["split_group_allocations"]["train"])
    val_g = set(v3["split_group_allocations"]["val"])
    test_g = set(v3["split_group_allocations"]["test"])

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
        for fn in v3["combined_clean_splits"][split_name]:
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
    print("\nDataset Split v3 Counts:")
    print(f"  Clean Train Images : {v3['counts']['combined_clean']['train']} (33 real + 24 synthetic)")
    print(f"  Clean Val Images   : {v3['counts']['combined_clean']['val']} (strictly unchanged)")
    print(f"  Clean Test Images  : {v3['counts']['combined_clean']['test']} (strictly unchanged)")
    print("=" * 75 + "\n")


def main() -> None:
    create_and_verify_splits_v3()


if __name__ == "__main__":
    main()
