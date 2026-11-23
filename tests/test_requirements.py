"""Verify that requirements.txt exactly matches installed environment packages."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = ROOT_DIR / "requirements.txt"


def test_requirements_pins_match_installed() -> None:
    """Every pinned package in requirements.txt must match the installed package version."""
    assert REQUIREMENTS_PATH.exists(), f"Missing {REQUIREMENTS_PATH}"

    mismatches = []
    missing_packages = []

    with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            assert "==" in line, f"Package specification must be strictly pinned with '==': {line}"
            pkg_name, expected_version = line.split("==", 1)
            pkg_name = pkg_name.strip()
            expected_version = expected_version.strip()

            try:
                installed_version = importlib.metadata.version(pkg_name)
                # Normalize case/format comparison
                if installed_version != expected_version:
                    mismatches.append(
                        f"{pkg_name}: required {expected_version}, installed {installed_version}"
                    )
            except importlib.metadata.PackageNotFoundError:
                missing_packages.append(pkg_name)

    assert not missing_packages, f"Packages in requirements.txt not installed in environment: {missing_packages}"
    assert not mismatches, f"Package version discrepancies between requirements.txt and environment:\n" + "\n".join(mismatches)
