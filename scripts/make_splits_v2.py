"""Generate Template-Safe Dataset Splits v2 (Unified Public & Own Invoices).

Reads data/vendor_groups.yaml and data/splits.json:
  1. Preserves public test families in TEST (Net2Source, Senate Space, Bansal, SBI General, SIMEQ).
  2. Preserves public train/val allocations (Microsoft, Sai Computer, LinkedIn in TRAIN; Amazon in VAL).
  3. Own LinkedIn (T10) joins public LinkedIn in TRAIN (0% vendor leakage).
  4. Generates and evaluates 3 options for own bill test allocation targeting 25-30% own test pages:
       - Option A (Recommended): 6 distinct groups, 8 pages (27.6%), max group size 2.
       - Option B: 5 distinct groups, 8 pages (27.6%), includes own_family_C (3p).
       - Option C: 6 distinct groups, 9 pages (31.0%), includes own_family_C (3p).
  5. Never puts groups with > 3 pages (own_family_A [6], own_family_D [6], own_family_B [4]) into test.
  6. Writes data/splits_v2.json without overwriting data/splits.json.
  7. Formally verifies zero split_group overlap across splits.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
VENDOR_GROUPS_PATH = ROOT_DIR / "data" / "vendor_groups.yaml"
SPLITS_V1_PATH = ROOT_DIR / "data" / "splits.json"
SPLITS_V2_PATH = ROOT_DIR / "data" / "splits_v2.json"


def evaluate_own_test_options(own_groups: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate candidate test set options maximizing distinct groups with <= 3 pages per group."""
    # Filter out LinkedIn (bound to public train) and large groups (> 3 pages)
    # Available small groups:
    #   own_family_C (3)
    #   own_family_H (2)
    #   own_family_G (2)
    #   own_family_I (1)
    #   own_family_F (1)
    #   own_family_J (1)
    #   own_family_E (1)

    options = [
        {
            "option_id": "Option A (Recommended)",
            "description": "Maximizes distinct groups (6 groups) with strictly <= 2 pages per group.",
            "test_groups": ["own_family_H", "own_family_G", "own_family_I", "own_family_F", "own_family_J", "own_family_E"],
            "val_groups": ["own_family_A"],
            "train_groups": ["own_family_D", "own_family_B", "own_family_C", "LinkedIn"],
        },
        {
            "option_id": "Option B",
            "description": "5 distinct groups, includes own_family_C (3p).",
            "test_groups": ["own_family_C", "own_family_H", "own_family_I", "own_family_F", "own_family_J"],
            "val_groups": ["own_family_A"],
            "train_groups": ["own_family_D", "own_family_B", "own_family_G", "own_family_E", "LinkedIn"],
        },
        {
            "option_id": "Option C",
            "description": "6 distinct groups, 9 pages (31.0%), includes own_family_C (3p).",
            "test_groups": ["own_family_C", "own_family_G", "own_family_I", "own_family_F", "own_family_J", "own_family_E"],
            "val_groups": ["own_family_A"],
            "train_groups": ["own_family_D", "own_family_B", "own_family_H", "LinkedIn"],
        },
    ]

    for opt in options:
        t_pages = sum(own_groups[g].get("total_pages") or own_groups[g].get("own_pages", 0) for g in opt["test_groups"])
        v_pages = sum(own_groups[g].get("total_pages") or own_groups[g].get("own_pages", 0) for g in opt["val_groups"])
        tr_pages = sum(own_groups[g].get("total_pages") or own_groups[g].get("own_pages", 0) for g in opt["train_groups"])
        opt["test_pages_count"] = t_pages
        opt["test_pct"] = round((t_pages / 29.0) * 100.0, 1)
        opt["val_pages_count"] = v_pages
        opt["train_pages_count"] = tr_pages
        opt["distinct_test_groups"] = len(opt["test_groups"])

    return options


def main() -> None:
    print("=" * 80)
    print("TEMPLATE-SAFE SPLITTING v2 (Unified Public & Own Invoices)")
    print("=" * 80)

    with open(VENDOR_GROUPS_PATH, "r", encoding="utf-8") as f:
        vgroups_raw = yaml.safe_load(f)["groups"]

    public_groups = {k: v for k, v in vgroups_raw.items() if v["source"] in ("public", "hybrid")}
    own_groups = {k: v for k, v in vgroups_raw.items() if v["source"] in ("own", "hybrid")}

    print(f"Loaded {len(vgroups_raw)} normalized split groups:")
    print(f"  * Public / Hybrid groups: {len(public_groups)}")
    print(f"  * Own / Hybrid groups: {len(own_groups)}")

    # Fixed Public Allocations (from splits.json)
    public_split_alloc = {
        "train": ["Microsoft", "Sai_Computer", "LinkedIn"],
        "val": ["Amazon"],
        "test": ["Net2Source", "Senate_Space", "Bansal", "SBI_General", "SIMEQ"],
    }

    print("\n" + "=" * 80)
    print("OWN BILLS TEST SET CANDIDATE OPTIONS FOR USER REVIEW")
    print("=" * 80)
    options = evaluate_own_test_options(own_groups)

    for opt in options:
        print(f"\n>>> {opt['option_id']}:")
        print(f"    Description: {opt['description']}")
        print(f"    Test Groups ({opt['distinct_test_groups']}): {', '.join(opt['test_groups'])}")
        print(f"    Own Test Pages: {opt['test_pages_count']} / 29 ({opt['test_pct']}%)")
        print(f"    Own Val Pages : {opt['val_pages_count']} / 29 (Group: {', '.join(opt['val_groups'])})")
        print(f"    Own Train Pages: {opt['train_pages_count']} / 29 (Groups: {', '.join(opt['train_groups'])})")

    # Select Option A as active configuration
    selected_opt = options[0]
    print(f"\n[Active Choice]: {selected_opt['option_id']} selected for data/splits_v2.json.")

    # Build Unified Split Allocation
    unified_group_splits: Dict[str, List[str]] = {
        "train": sorted(list(set(public_split_alloc["train"] + selected_opt["train_groups"]))),
        "val": sorted(list(set(public_split_alloc["val"] + selected_opt["val_groups"]))),
        "test": sorted(list(set(public_split_alloc["test"] + selected_opt["test_groups"]))),
    }

    # Verify zero group leakage
    tr_g = set(unified_group_splits["train"])
    v_g = set(unified_group_splits["val"])
    te_g = set(unified_group_splits["test"])

    assert len(tr_g & v_g) == 0, f"Overlap train-val: {tr_g & v_g}"
    assert len(tr_g & te_g) == 0, f"Overlap train-test: {tr_g & te_g}"
    assert len(v_g & te_g) == 0, f"Overlap val-test: {v_g & te_g}"
    print("\n[Verification Passed]: 0% split_group overlap across train, val, and test.")

    # Build file-level image mappings
    # 1. Public images from splits.json
    with open(SPLITS_V1_PATH, "r", encoding="utf-8") as f:
        splits_v1 = json.load(f)

    clean_public_splits = splits_v1["clean_splits"]
    synth_public_splits = splits_v1["synthetic_splits"]

    # 2. Own bill pages
    manifest_df = pd.read_csv(ROOT_DIR / "data" / "own_manifest.csv")
    stem_to_own_group = {}
    for g_name, g_info in own_groups.items():
        if "pages" in g_info:
            for p in g_info["pages"]:
                stem_to_own_group[p] = g_name
    # Own LinkedIn pages (own_014_p1.png, own_014_p2.png)
    stem_to_own_group["own_014_p1.png"] = "LinkedIn"
    stem_to_own_group["own_014_p2.png"] = "LinkedIn"

    own_splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for _, row in manifest_df.iterrows():
        fn = row["filename"]
        g = stem_to_own_group[fn]
        for s_name, g_list in unified_group_splits.items():
            if g in g_list:
                own_splits[s_name].append(fn)
                break
        own_splits[s_name].sort()

    # Unified combined file lists
    combined_clean_splits: Dict[str, List[str]] = {
        s: sorted(clean_public_splits[s] + own_splits[s]) for s in ["train", "val", "test"]
    }

    output_payload = {
        "metadata": {
            "version": "v2",
            "active_option": selected_opt["option_id"],
            "options_considered": options,
            "leakage_verified": True,
            "total_documents": len(combined_clean_splits["train"]) + len(combined_clean_splits["val"]) + len(combined_clean_splits["test"]),
        },
        "split_group_allocations": unified_group_splits,
        "public_clean_splits": clean_public_splits,
        "public_synthetic_splits": synth_public_splits,
        "own_splits": own_splits,
        "combined_clean_splits": combined_clean_splits,
        "counts": {
            "public_clean": {s: len(clean_public_splits[s]) for s in ["train", "val", "test"]},
            "public_synthetic": {s: len(synth_public_splits[s]) for s in ["train", "val", "test"]},
            "own_pages": {s: len(own_splits[s]) for s in ["train", "val", "test"]},
            "combined_clean": {s: len(combined_clean_splits[s]) for s in ["train", "val", "test"]},
        },
    }

    with open(SPLITS_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nWrote unified v2 splits to: {SPLITS_V2_PATH}")
    print("\nSummary Counts per Split:")
    print("-" * 75)
    row_fmt = "{:<8} {:>14} {:>14} {:>12} {:>16}"
    print(row_fmt.format("Split", "Public Clean", "Public Synth", "Own Pages", "Combined Clean"))
    print("-" * 75)
    for s in ["train", "val", "test"]:
        print(row_fmt.format(
            s.upper(),
            len(clean_public_splits[s]),
            len(synth_public_splits[s]),
            len(own_splits[s]),
            len(combined_clean_splits[s]),
        ))
    print("-" * 75)


if __name__ == "__main__":
    main()
