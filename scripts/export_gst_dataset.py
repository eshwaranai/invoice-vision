"""Export Indian GST invoice dataset from Hugging Face to local workspace.

Downloads 'adityaeucloid/test_dataset' (split 'test'), saves images to
data/raw/indian_gst/, and writes structured annotations to data/ground_truth_gst.csv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset
from PIL import Image


DATASET_NAME = "adityaeucloid/test_dataset"
SPLIT = "test"
OUTPUT_DIR = Path("data/raw/indian_gst")
CSV_PATH = Path("data/ground_truth_gst.csv")


def parse_ground_truth(gt_raw: Any) -> Dict[str, Any]:
    """Safely parse ground_truth field into a dictionary of extracted fields."""
    if isinstance(gt_raw, str):
        try:
            gt_data = json.loads(gt_raw)
        except json.JSONDecodeError:
            return {"raw_text": gt_raw}
    elif isinstance(gt_raw, dict):
        gt_data = gt_raw
    else:
        return {}

    # Extract fields if nested under 'gt_parse'
    if "gt_parse" in gt_data:
        gt_parse = gt_data["gt_parse"]
        if isinstance(gt_parse, str):
            try:
                parsed_fields = json.loads(gt_parse)
            except json.JSONDecodeError:
                parsed_fields = {"gt_parse_raw": gt_parse}
        elif isinstance(gt_parse, dict):
            parsed_fields = dict(gt_parse)
        else:
            parsed_fields = {"gt_parse": gt_parse}
    else:
        parsed_fields = dict(gt_data)

    # Check for original filename in root metadata or parsed fields
    orig_fn = (
        gt_data.get("original_filename")
        or gt_data.get("file_name")
        or gt_data.get("filename")
        or parsed_fields.pop("original_filename", None)
        or parsed_fields.pop("file_name", None)
        or ""
    )

    result: Dict[str, Any] = {"original_filename": orig_fn}

    # Flatten or serialize complex nested types (lists/dicts) for CSV safety
    for key, val in parsed_fields.items():
        if isinstance(val, (dict, list)):
            result[key] = json.dumps(val, ensure_ascii=False)
        else:
            result[key] = val

    return result


def export_dataset() -> None:
    """Download HF dataset, save images, and construct ground truth CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset '{DATASET_NAME}' (split: '{SPLIT}')...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    total_records = len(ds)
    print(f"Loaded {total_records} records.")

    rows: List[Dict[str, Any]] = []
    saved_images_count = 0

    for idx, item in enumerate(ds):
        filename = f"gst_{idx:03d}.jpg"
        img_path = OUTPUT_DIR / filename

        # 1. Save Image as RGB JPEG
        img: Image.Image = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_path, format="JPEG", quality=95)
        saved_images_count += 1

        # 2. Parse Ground Truth
        gt_info = parse_ground_truth(item.get("ground_truth"))
        row_entry: Dict[str, Any] = {
            "filename": filename,
            "original_filename": gt_info.get("original_filename", ""),
        }
        for k, v in gt_info.items():
            if k not in ("filename", "original_filename"):
                row_entry[k] = v

        rows.append(row_entry)

    # 3. Create DataFrame and order columns
    df = pd.DataFrame(rows)

    # Ensure filename and original_filename are the first two columns
    primary_cols = ["filename", "original_filename"]
    other_cols = [c for c in df.columns if c not in primary_cols]
    ordered_cols = primary_cols + sorted(other_cols)
    df = df[ordered_cols]

    # Save to CSV
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"Saved ground truth annotations to: {CSV_PATH}")

    # 4. Print Diagnostics
    print("\n" + "=" * 70)
    print("DATASET EXPORT SUMMARY")
    print("=" * 70)
    print(f"Number of images saved: {saved_images_count} to {OUTPUT_DIR}/")
    print(f"\nCSV Columns ({len(df.columns)} total):")
    for col in df.columns:
        print(f"  - {col}")

    print("\nNon-empty values count per column:")
    for col in df.columns:
        # Check non-null, non-empty string, and non-NaN
        non_empty = df[col].apply(
            lambda x: pd.notna(x) and str(x).strip() != "" and str(x).lower() != "nan"
        ).sum()
        print(f"  {col:<30}: {non_empty}/{len(df)}")

    print("\nFirst 10 rows:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head(10).to_string())
    print("=" * 70)


if __name__ == "__main__":
    export_dataset()
