"""Template-Safe Dataset Splitting Script.

Merges vendor groups sharing identical or related template families:
  - All Microsoft groups -> Microsoft family (8 images)
  - All Amazon-generated invoices -> Amazon family (6 images)
  - LinkedIn Singapore -> LinkedIn family (3 images)
  - Other distinct single-vendor groups

Assigns whole families to train / val / test splits aiming for 60/20/20 ratio (18 / 6 / 6),
ensuring zero template or vendor leakage between splits.
Also maps synthetic phone photos to splits based on their parent clean image.
Writes output to data/splits.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
GROUPS_PATH = ROOT_DIR / "data" / "groups.json"
MANIFEST_PATH = ROOT_DIR / "data" / "processed" / "synth_photos" / "manifest.json"
SPLITS_PATH = ROOT_DIR / "data" / "splits.json"

RANDOM_SEED = 42


def get_merged_families(groups_data: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Merge vendor groups that share the same template family."""
    families: Dict[str, List[str]] = {
        "Microsoft": [],
        "Amazon": [],
        "LinkedIn": [],
        "Sai Computer Solution": [],
        "Net2Source": [],
        "Senate Space": [],
        "Bansal Electrovision": [],
        "SBI General Insurance": [],
        "SIMEQ Technologies": [],
    }

    for vendor, images in groups_data.items():
        v_lower = vendor.lower()
        if "microsoft" in v_lower:
            families["Microsoft"].extend(images)
        elif "amazon" in v_lower:
            families["Amazon"].extend(images)
        elif "linkedin" in v_lower:
            families["LinkedIn"].extend(images)
        elif "sai computer" in v_lower:
            families["Sai Computer Solution"].extend(images)
        elif "net2source" in v_lower:
            families["Net2Source"].extend(images)
        elif "senate" in v_lower:
            families["Senate Space"].extend(images)
        elif "bansal" in v_lower:
            families["Bansal Electrovision"].extend(images)
        elif "sbi" in v_lower:
            families["SBI General Insurance"].extend(images)
        elif "simeq" in v_lower:
            families["SIMEQ Technologies"].extend(images)
        else:
            families[vendor] = list(images)

    # Clean up empty families and sort image lists
    result = {}
    for k, v in families.items():
        if v:
            result[k] = sorted(list(set(v)))
    return result


def find_optimal_splits(
    families: Dict[str, List[str]],
    target_counts: Tuple[int, int, int] = (18, 6, 6),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Partition whole families into train / val / test matching target counts (18, 6, 6)."""
    # Deterministic search over permutations
    items = list(families.items())
    # Sort items deterministically first
    items.sort(key=lambda x: (-len(x[1]), x[0]))

    # Search for an exact match (18, 6, 6)
    # Target sizes: train=18, val=6, test=6
    best_assignment = None
    best_loss = 9999

    import itertools

    fam_keys = [k for k, _ in items]
    fam_sizes = {k: len(v) for k, v in items}

    # Evaluate family assignments to (train, val, test)
    for assignment in itertools.product(["train", "val", "test"], repeat=len(fam_keys)):
        c_train = sum(fam_sizes[fam_keys[i]] for i, a in enumerate(assignment) if a == "train")
        c_val = sum(fam_sizes[fam_keys[i]] for i, a in enumerate(assignment) if a == "val")
        c_test = sum(fam_sizes[fam_keys[i]] for i, a in enumerate(assignment) if a == "test")

        loss = abs(c_train - target_counts[0]) + abs(c_val - target_counts[1]) + abs(c_test - target_counts[2])
        if loss < best_loss:
            best_loss = loss
            best_assignment = assignment
            if loss == 0:
                break

    assert best_assignment is not None
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for i, a in enumerate(best_assignment):
        splits[a].append(fam_keys[i])

    return splits


def main() -> None:
    print("=" * 75)
    print("TEMPLATE-SAFE DATASET SPLITTING (Whole Family Partitioning)")
    print("=" * 75)

    if not GROUPS_PATH.exists():
        raise FileNotFoundError(f"Missing {GROUPS_PATH}")

    with open(GROUPS_PATH, "r", encoding="utf-8") as f:
        groups_raw = json.load(f)["groups"]

    merged_families = get_merged_families(groups_raw)

    print("\nMerged Template Families & Image Lists:")
    print("-" * 75)
    total_imgs = 0
    for fam_name, img_list in sorted(merged_families.items(), key=lambda x: -len(x[1])):
        print(f"  * {fam_name:<25} ({len(img_list):2d} images): {', '.join(img_list)}")
        total_imgs += len(img_list)
    print(f"\nTotal Merged Families: {len(merged_families)}")
    print(f"Total Images: {total_imgs}")

    # Compute partition
    family_split_assignment = find_optimal_splits(merged_families, target_counts=(18, 6, 6), seed=RANDOM_SEED)

    split_images: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for s_name, fam_list in family_split_assignment.items():
        for f in fam_list:
            split_images[s_name].extend(merged_families[f])
        split_images[s_name].sort()

    print("\nSplit Allocation Summary:")
    print("-" * 75)
    for s_name in ["train", "val", "test"]:
        fams = family_split_assignment[s_name]
        imgs = split_images[s_name]
        pct = (len(imgs) / total_imgs) * 100.0
        print(f"\n>>> Split: {s_name.upper()} ({len(imgs)} clean images, {pct:.1f}%):")
        print(f"    Assigned Families ({len(fams)}): {', '.join(fams)}")
        print(f"    Image List: {', '.join(imgs)}")

    # Strict Validation: zero family and zero image overlap
    train_fams = set(family_split_assignment["train"])
    val_fams = set(family_split_assignment["val"])
    test_fams = set(family_split_assignment["test"])

    assert len(train_fams & val_fams) == 0, f"Family leak between train and val: {train_fams & val_fams}"
    assert len(train_fams & test_fams) == 0, f"Family leak between train and test: {train_fams & test_fams}"
    assert len(val_fams & test_fams) == 0, f"Family leak between val and test: {val_fams & test_fams}"

    train_imgs = set(split_images["train"])
    val_imgs = set(split_images["val"])
    test_imgs = set(split_images["test"])

    assert len(train_imgs & val_imgs) == 0, "Image leak train/val"
    assert len(train_imgs & test_imgs) == 0, "Image leak train/test"
    assert len(val_imgs & test_imgs) == 0, "Image leak val/test"
    assert len(train_imgs | val_imgs | test_imgs) == total_imgs, "Missing images in split"

    print("\n[Verification Passed]: No template family or image appears in more than one split.")

    # Synthetic photos mapping
    synth_splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    synth_manifest_present = MANIFEST_PATH.exists()
    if synth_manifest_present:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for rec in manifest["records"]:
            src_img = rec["source_image"]
            assigned_split = None
            for s_name, img_set in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
                if src_img in img_set:
                    assigned_split = s_name
                    break
            if assigned_split is not None:
                synth_splits[assigned_split].append(rec["filename"])

        print("\nSynthetic Photos Partition (Inherited from Source):")
        for s_name in ["train", "val", "test"]:
            print(f"  {s_name.upper():<5}: {len(synth_splits[s_name])} synthetic images")

    # Build output payload
    output_data = {
        "metadata": {
            "total_clean_images": total_imgs,
            "total_families": len(merged_families),
            "target_ratio": "60/20/20",
            "seed": RANDOM_SEED,
            "leakage_verified": True,
        },
        "families": merged_families,
        "family_split_assignment": family_split_assignment,
        "clean_splits": split_images,
        "clean_counts": {k: len(v) for k, v in split_images.items()},
        "synthetic_splits": synth_splits,
        "synthetic_counts": {k: len(v) for k, v in synth_splits.items()},
    }

    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nWrote template-safe splits to: {SPLITS_PATH}")


if __name__ == "__main__":
    main()
