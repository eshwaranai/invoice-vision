"""Create zip archive of YOLO dataset for cloud training (e.g., Google Colab).

Compresses images, labels, and data.yaml from data/yolo_v1/ or data/yolo_v2/ into zip.
Reports archive file count, compressed size (MB), and uncompressed size.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Tuple
import zipfile

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = ROOT_DIR / "data" / "yolo_v1"
DEFAULT_OUTPUT_ZIP = ROOT_DIR / "data" / "yolo_v1.zip"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("zip_dataset")


def create_dataset_zip(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_zip: Path = DEFAULT_OUTPUT_ZIP,
) -> Tuple[int, float, float]:
    """Create a zip archive of source_dir and return (file_count, size_mb, uncompressed_mb)."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    # Collect all files
    files_to_zip = sorted([f for f in source_dir.rglob("*") if f.is_file()])
    if not files_to_zip:
        raise ValueError(f"No files found in source directory: {source_dir}")

    total_uncompressed_bytes = 0

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files_to_zip:
            rel_path = f.relative_to(source_dir)
            total_uncompressed_bytes += f.stat().st_size
            zf.write(f, arcname=str(rel_path).replace("\\", "/"))

    compressed_bytes = output_zip.stat().st_size
    compressed_mb = compressed_bytes / (1024 * 1024)
    uncompressed_mb = total_uncompressed_bytes / (1024 * 1024)

    logger.info(
        f"Created dataset archive: {output_zip} | Files: {len(files_to_zip)} | "
        f"Compressed: {compressed_mb:.2f} MB | Uncompressed: {uncompressed_mb:.2f} MB"
    )

    return len(files_to_zip), compressed_mb, uncompressed_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Package YOLO dataset into zip archive")
    parser.add_argument(
        "--src",
        "--source",
        dest="source",
        type=str,
        default=str(DEFAULT_SOURCE_DIR),
        help="Source dataset directory (default: data/yolo_v1)",
    )
    parser.add_argument(
        "--out",
        "--output",
        dest="output",
        type=str,
        default=str(DEFAULT_OUTPUT_ZIP),
        help="Output zip archive path (default: data/yolo_v1.zip)",
    )
    args = parser.parse_args()

    count, comp_mb, uncomp_mb = create_dataset_zip(
        source_dir=Path(args.source),
        output_zip=Path(args.output),
    )
    print("\n" + "=" * 60)
    print("YOLO DATASET ARCHIVE SUMMARY")
    print("=" * 60)
    print(f"Archive Path       : {args.output}")
    print(f"Total Files Added  : {count}")
    print(f"Compressed Size    : {comp_mb:.2f} MB")
    print(f"Uncompressed Size  : {uncomp_mb:.2f} MB")
    print(f"Compression Ratio  : {uncomp_mb / comp_mb:.2f}x" if comp_mb > 0 else "")
    print("=" * 60)


if __name__ == "__main__":
    main()
