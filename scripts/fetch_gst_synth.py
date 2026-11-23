"""Fetch and Sample Synthetic Indian GST Invoices.

Dataset: Divya52/indian-gst-invoices (config: synthetic)
Original sample: 24 invoices (gstsyn_000 .. gstsyn_023)
Extension: 60 new invoices (gstsyn_024 .. gstsyn_083)
Total: 84 invoices, stratified over layout x level (4 layouts x 3 levels),
maximizing diversity across font, header style, and supply_type.

Saves:
  - data/raw/gst_synth/gstsyn_024.<ext> .. gstsyn_083.<ext>
  - data/processed/gst_synth_labels/gstsyn_024.json .. gstsyn_083.json
  - data/gst_synth_manifest.csv (appended rows)
  - data/gst_synth_README.md
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Dict, List, Set, Tuple

from huggingface_hub import hf_hub_download

ROOT_DIR = Path(__file__).resolve().parent.parent
REPO_ID = "Divya52/indian-gst-invoices"
DATASET_CONFIG = "synthetic"

RAW_OUT_DIR = ROOT_DIR / "data" / "raw" / "gst_synth"
LABELS_OUT_DIR = ROOT_DIR / "data" / "processed" / "gst_synth_labels"
MANIFEST_PATH = ROOT_DIR / "data" / "gst_synth_manifest.csv"
README_PATH = ROOT_DIR / "data" / "gst_synth_README.md"

LAYOUTS = ["classic", "handwritten", "minimal", "thermal"]
LEVELS = ["clean", "photo", "scan"]
SEED = 42


def print_json_key_structure(label_data: Dict[str, Any]) -> None:
    """Print the hierarchical key structure of the raw label JSON."""
    print("\n" + "=" * 75)
    print("RAW LABEL JSON KEY STRUCTURE:")
    print("=" * 75)
    for k, v in label_data.items():
        if isinstance(v, dict):
            subkeys = list(v.keys())
            print(f"  {k} (dict): {subkeys}")
        elif isinstance(v, list):
            sample_keys = list(v[0].keys()) if v and isinstance(v[0], dict) else type(v[0]).__name__ if v else "empty"
            print(f"  {k} (list[{len(v)}]): item keys -> {sample_keys}")
        else:
            print(f"  {k}: {type(v).__name__} (example: {repr(v)[:40]})")
    print("=" * 75 + "\n")


def write_synth_readme(total_samples: int = 84) -> None:
    """Write data/gst_synth_README.md with attribution and dataset details."""
    content = f"""# Synthetic Indian GST Invoices Subset

## Source & Attribution
- **Dataset**: `Divya52/indian-gst-invoices` (Configuration: `synthetic`)
- **Source URL**: [https://huggingface.co/datasets/Divya52/indian-gst-invoices](https://huggingface.co/datasets/Divya52/indian-gst-invoices)
- **Author**: Divya52
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))

## Subset Description
- **Total Sample Count**: Exactly {total_samples} images (`gstsyn_000` through `gstsyn_{total_samples-1:03d}`).
  - Initial batch: 24 images (`gstsyn_000` .. `gstsyn_023`).
  - Extension batch: 60 images (`gstsyn_024` .. `gstsyn_083`).
- **Sampling Strategy**: Stratified across all 12 combinations of:
  - **4 Layouts**: `classic`, `handwritten`, `minimal`, `thermal` (21 images each)
  - **3 Degradation Levels**: `clean`, `photo`, `scan` (28 images each)
- **Diversity Balancing**: Greedily maximized joint diversity across:
  - **Fonts**: `arial`, `helvetica`, `courier`, `times`
  - **Header Styles**: `plain`, `band`, None (thermal)
  - **Supply Types**: `inter_state`, `intra_state`
- **Random Seed**: `42` (deterministic selection).
- **Data Privacy**: All company names, addresses, GSTIN numbers, dates, and amounts are procedurally generated synthetic data. Zero real personal or corporate data.

## Directory Layout
- `data/raw/gst_synth/`: Raw image files (`.png` for clean, `.jpg` for photo/scan).
- `data/processed/gst_synth_labels/`: Complete raw ground-truth JSON labels per invoice.
- `data/gst_synth_manifest.csv`: Flattened metadata and key billing attributes for all {total_samples} invoices.
"""
    README_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {README_PATH.relative_to(ROOT_DIR)}")


def load_existing_manifest() -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Load existing manifest rows and set of already used original IDs."""
    existing_rows: List[Dict[str, Any]] = []
    used_ids: Set[str] = set()
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing_rows.append(r)
                used_ids.add(r["original_id"])
    return existing_rows, used_ids


def sample_diverse_records(
    available_records: List[Dict[str, Any]],
    samples_per_bucket: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Stratified sampling across (layout x level) maximizing diversity across font, header, supply_type."""
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in available_records:
        ly = r.get("layout", "")
        lv = r.get("level", "")
        if ly in LAYOUTS and lv in LEVELS:
            buckets[(ly, lv)].append(r)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []

    for ly in LAYOUTS:
        for lv in LEVELS:
            cand_list = list(buckets[(ly, lv)])
            if len(cand_list) < samples_per_bucket:
                raise ValueError(f"Insufficient samples for ({ly}, {lv}): found {len(cand_list)}, need {samples_per_bucket}")

            # Deterministic sorting
            rng.shuffle(cand_list)
            cand_list.sort(key=lambda x: str(x.get("id", "")))

            chosen: List[Dict[str, Any]] = []
            font_counts: Counter[str] = Counter()
            header_counts: Counter[str] = Counter()
            supply_counts: Counter[str] = Counter()

            for _ in range(samples_per_bucket):
                best_cand = None
                best_score = float("inf")
                for c in cand_list:
                    if c in chosen:
                        continue
                    f = str(c.get("font") or "")
                    h = str(c.get("header") or "")
                    s = str(c.get("supply_type") or "")
                    score = (font_counts[f] * 10) + (header_counts[h] * 5) + (supply_counts[s] * 2)
                    if score < best_score:
                        best_score = score
                        best_cand = c
                if best_cand is None:
                    raise RuntimeError("Failed to pick candidate")
                chosen.append(best_cand)
                font_counts[str(best_cand.get("font") or "")] += 1
                header_counts[str(best_cand.get("header") or "")] += 1
                supply_counts[str(best_cand.get("supply_type") or "")] += 1

            selected.extend(chosen)

    # Sort deterministically
    selected.sort(key=lambda x: (x.get("layout", ""), x.get("level", ""), x.get("id", "")))
    return selected


def fetch_and_sample_60() -> None:
    """Sample 60 new synthetic invoices (gstsyn_024 to gstsyn_083) and append to manifest."""
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_rows, used_ids = load_existing_manifest()
    print(f"Loaded existing manifest with {len(existing_rows)} rows ({len(used_ids)} distinct original IDs).")
    start_idx = len(existing_rows)

    print(f"Downloading metadata for '{REPO_ID}' (config: '{DATASET_CONFIG}')...")
    meta_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{DATASET_CONFIG}/metadata.jsonl",
        repo_type="dataset",
    )

    records: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                r = json.loads(line_str)
                if r.get("id") not in used_ids:
                    records.append(r)

    print(f"Loaded {len(records)} candidate metadata records excluding already used IDs.")

    # Select 60 samples: 5 per (layout x level) pair
    selected = sample_diverse_records(records, samples_per_bucket=5, seed=SEED)
    print(f"Selected exactly {len(selected)} new samples across 12 (layout x level) pairs.")

    # Report counts per attribute
    layouts_cnt = Counter(r.get("layout") for r in selected)
    levels_cnt = Counter(r.get("level") for r in selected)
    fonts_cnt = Counter(r.get("font") for r in selected)
    headers_cnt = Counter(r.get("header") for r in selected)
    supplies_cnt = Counter(r.get("supply_type") for r in selected)

    print("\n" + "=" * 75)
    print(f"ATTRIBUTE BREAKDOWN FOR {len(selected)} NEW SAMPLES (gstsyn_{start_idx:03d} - gstsyn_{start_idx+len(selected)-1:03d}):")
    print("=" * 75)
    print(f"  Layouts:      {dict(layouts_cnt)}")
    print(f"  Levels:       {dict(levels_cnt)}")
    print(f"  Fonts:        {dict(fonts_cnt)}")
    print(f"  Header Style: {dict(headers_cnt)}")
    print(f"  Supply Type:  {dict(supplies_cnt)}")
    print("=" * 75 + "\n")

    new_manifest_rows: List[Dict[str, Any]] = []

    print("Downloading images and writing labels:")
    for offset, sample in enumerate(selected):
        new_idx = start_idx + offset
        new_stem = f"gstsyn_{new_idx:03d}"
        orig_file = str(sample.get("file_name", ""))
        ext = Path(orig_file).suffix.lower() or ".jpg"
        new_filename = f"{new_stem}{ext}"
        orig_id = sample.get("id", "")
        layout = sample.get("layout", "")
        level = sample.get("level", "")

        # Parse label JSON
        lbl_raw = sample.get("label", "{}")
        lbl = json.loads(lbl_raw) if isinstance(lbl_raw, str) else lbl_raw

        # Save raw label JSON
        raw_label_path = LABELS_OUT_DIR / f"{new_stem}.json"
        with open(raw_label_path, "w", encoding="utf-8") as f:
            json.dump(lbl, f, indent=2, ensure_ascii=False)

        # Download original image from Hub
        hub_image_filename = f"{DATASET_CONFIG}/{orig_file}"
        downloaded_cached_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=hub_image_filename,
            repo_type="dataset",
        )
        dest_img_path = RAW_OUT_DIR / new_filename
        shutil.copy2(downloaded_cached_path, dest_img_path)

        # Build flattened manifest record
        supplier = lbl.get("supplier", {}) or {}
        buyer = lbl.get("buyer", {}) or {}
        totals = lbl.get("totals", {}) or {}
        line_items = lbl.get("line_items", []) or []

        row = {
            "filename": new_filename,
            "original_id": orig_id,
            "layout": layout,
            "level": level,
            "invoice_number": lbl.get("invoice_number", ""),
            "invoice_date": lbl.get("invoice_date", ""),
            "supplier_name": supplier.get("name", ""),
            "supplier_gstin": supplier.get("gstin", ""),
            "supplier_address": supplier.get("address", ""),
            "supplier_state": supplier.get("state", ""),
            "supplier_state_code": supplier.get("state_code", ""),
            "buyer_name": buyer.get("name", ""),
            "buyer_gstin": buyer.get("gstin", ""),
            "buyer_address": buyer.get("address", ""),
            "buyer_state": buyer.get("state", ""),
            "buyer_state_code": buyer.get("state_code", ""),
            "place_of_supply": lbl.get("place_of_supply", ""),
            "supply_type": lbl.get("supply_type", ""),
            "reverse_charge": lbl.get("reverse_charge", ""),
            "n_line_items": len(line_items),
            "taxable_value": totals.get("taxable_value", ""),
            "cgst": totals.get("cgst", ""),
            "sgst": totals.get("sgst", ""),
            "igst": totals.get("igst", ""),
            "round_off": totals.get("round_off", ""),
            "grand_total": totals.get("grand_total", ""),
        }
        new_manifest_rows.append(row)
        print(f"  [{offset+1:02d}/60] {new_filename} <- {orig_id} ({layout}, {level}) [font={sample.get('font')}, header={sample.get('header')}, supply={sample.get('supply_type')}]")

    # Combine existing and new manifest rows
    all_manifest_rows = existing_rows + new_manifest_rows
    fieldnames = list(all_manifest_rows[0].keys())
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_manifest_rows)

    print(f"\nUpdated manifest to total {len(all_manifest_rows)} rows in: {MANIFEST_PATH.relative_to(ROOT_DIR)}")
    write_synth_readme(total_samples=len(all_manifest_rows))


def main() -> None:
    fetch_and_sample_60()


if __name__ == "__main__":
    main()
