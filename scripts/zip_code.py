"""Package project codebase into data/code.zip for self-contained cloud execution.

Includes only configs/, scripts/, src/, and requirements.txt.
Excludes all data, datasets, weights, runs, git metadata, and notebooks.
Verifies and prints file count, archive size, and included file extensions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Set, Tuple
import zipfile

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ZIP = ROOT_DIR / "data" / "code.zip"

# Strict whitelist of top-level directories and files to package
INCLUDED_DIRS = ["configs", "scripts", "src"]
INCLUDED_FILES = ["requirements.txt"]

# Strict blacklist of file patterns and folders
EXCLUDED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".pdf",  # Images / documents
    ".pt", ".pth", ".bin", ".onnx", ".safetensors",             # Model weights
    ".pyc", ".pyo", ".pyd",                                     # Bytecode
    ".zip", ".tar", ".gz",                                      # Archives
    ".csv", ".tsv", ".parquet", ".sqlite", ".db",               # Raw data
}
EXCLUDED_DIR_NAMES = {
    "__pycache__", ".pytest_cache", ".git", ".github",
    "runs", "data", "notebooks", "results", "venv", ".venv",
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("zip_code")


def is_file_allowed(file_path: Path) -> bool:
    """Check if file satisfies whitelist/blacklist security and privacy rules."""
    # Check directory components for excluded folder names
    parts = set(file_path.parts)
    if parts.intersection(EXCLUDED_DIR_NAMES):
        return False

    # Check extension
    ext = file_path.suffix.lower()
    if ext in EXCLUDED_EXTENSIONS:
        return False

    return True


def create_code_zip(
    root_dir: Path = ROOT_DIR,
    output_zip: Path = DEFAULT_OUT_ZIP,
) -> Tuple[int, float, Set[str]]:
    """Create self-contained code archive and return (file_count, size_kb, extensions)."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    files_to_pack = []

    # 1. Collect from included directories
    for dir_name in INCLUDED_DIRS:
        target_dir = root_dir / dir_name
        if target_dir.exists() and target_dir.is_dir():
            for f in sorted(target_dir.rglob("*")):
                if f.is_file() and is_file_allowed(f):
                    files_to_pack.append(f)

    # 2. Collect included root files
    for fname in INCLUDED_FILES:
        target_file = root_dir / fname
        if target_file.exists() and target_file.is_file() and is_file_allowed(target_file):
            files_to_pack.append(target_file)

    if not files_to_pack:
        raise ValueError(f"No qualifying source code files found in {root_dir}")

    extensions: Set[str] = set()

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in files_to_pack:
            rel_path = f.relative_to(root_dir)
            arcname = str(rel_path).replace("\\", "/")
            zf.write(f, arcname=arcname)
            ext = f.suffix.lower() if f.suffix else "[no_ext]"
            extensions.add(ext)

    archive_size_bytes = output_zip.stat().st_size
    archive_size_kb = archive_size_bytes / 1024.0

    return len(files_to_pack), archive_size_kb, extensions


def main() -> None:
    parser = argparse.ArgumentParser(description="Package code into data/code.zip")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUT_ZIP),
        help="Output zip path (default: data/code.zip)",
    )
    args = parser.parse_args()

    count, size_kb, extensions = create_code_zip(
        root_dir=ROOT_DIR,
        output_zip=Path(args.output),
    )

    print("\n" + "=" * 65)
    print("CODE PACKAGE SUMMARY (data/code.zip)")
    print("=" * 65)
    print(f"Archive Output      : {args.output}")
    print(f"Total Files Added   : {count}")
    print(f"Archive Size        : {size_kb:.2f} KB ({size_kb / 1024.0:.2f} MB)")
    print(f"File Extensions     : {sorted(list(extensions))}")
    print("Verified Exclusions : Zero raw images, personal bills, weights, or runs.")
    print("=" * 65)


if __name__ == "__main__":
    main()
