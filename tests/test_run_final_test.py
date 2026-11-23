"""Unit tests for scripts/run_final_test.py.

Guarantees:
  1. Never touches real test-split images.
  2. Verifies refusal without --final flag.
  3. Verifies refusal if test_lock.json already exists.
  4. Verifies refusal on weights SHA-256 mismatch or commit lineage mismatch.
  5. Proves that after a successful run, a second run is strictly refused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_final_test import (
    compute_file_sha256,
    main,
    parse_protocol_specs,
    verify_test_preconditions,
    write_test_lock,
)


def test_parse_protocol_specs(tmp_path: Path) -> None:
    dummy_protocol = tmp_path / "protocol.md"
    dummy_protocol.write_text(
        "# Test Protocol\n"
        "- **Weights SHA256**: `c76c6753c52c4d78ba73c966e603ec604fccafc633e95db1091e1ecc50824919`\n"
        "- **Frozen Git Commit**: `aa2e9a64ecbf712ef86cc4e7fa40840c0c64f646`\n",
        encoding="utf-8",
    )
    specs = parse_protocol_specs(dummy_protocol)
    assert specs["expected_weights_sha256"] == "c76c6753c52c4d78ba73c966e603ec604fccafc633e95db1091e1ecc50824919"
    assert specs["expected_commit_hash"] == "aa2e9a64ecbf712ef86cc4e7fa40840c0c64f646"


def test_refusal_without_final_flag(tmp_path: Path) -> None:
    dummy_lock = tmp_path / "test_lock.json"
    dummy_weights = tmp_path / "dummy.pt"
    dummy_weights.write_bytes(b"dummy weights data")
    dummy_sha = hashlib.sha256(b"dummy weights data").hexdigest()

    dummy_protocol = tmp_path / "protocol.md"
    dummy_protocol.write_text(
        f"- **Weights SHA256**: `{dummy_sha}`\n- **Frozen Git Commit**: `aa2e9a64ecbf`\n"
    )

    ret = main([
        "--lock-file", str(dummy_lock),
        "--weights", str(dummy_weights),
        "--protocol", str(dummy_protocol),
        "--bypass-git-check",
    ])
    assert ret == 1
    assert not dummy_lock.exists(), "Lockfile must NOT be written when invocation is refused"


def test_refusal_when_lockfile_exists(tmp_path: Path) -> None:
    dummy_lock = tmp_path / "test_lock.json"
    dummy_lock.write_text(json.dumps({"timestamp": "2026-11-22T00:00:00Z", "lock_active": True}))

    dummy_weights = tmp_path / "dummy.pt"
    dummy_weights.write_bytes(b"dummy weights data")
    dummy_sha = hashlib.sha256(b"dummy weights data").hexdigest()

    dummy_protocol = tmp_path / "protocol.md"
    dummy_protocol.write_text(
        f"- **Weights SHA256**: `{dummy_sha}`\n- **Frozen Git Commit**: `aa2e9a64ecbf`\n"
    )

    ret = main([
        "--final",
        "--lock-file", str(dummy_lock),
        "--weights", str(dummy_weights),
        "--protocol", str(dummy_protocol),
        "--bypass-git-check",
    ])
    assert ret == 1


def test_refusal_on_weights_sha_mismatch(tmp_path: Path) -> None:
    dummy_lock = tmp_path / "test_lock.json"
    dummy_weights = tmp_path / "dummy.pt"
    dummy_weights.write_bytes(b"tampered weights")

    dummy_protocol = tmp_path / "protocol.md"
    dummy_protocol.write_text(
        "- **Weights SHA256**: `0000000000000000000000000000000000000000000000000000000000000000`\n"
        "- **Frozen Git Commit**: `aa2e9a64ecbf`\n"
    )

    ret = main([
        "--final",
        "--lock-file", str(dummy_lock),
        "--weights", str(dummy_weights),
        "--protocol", str(dummy_protocol),
        "--bypass-git-check",
    ])
    assert ret == 1


def test_refusal_on_dirty_tree(tmp_path: Path) -> None:
    dummy_lock = tmp_path / "test_lock.json"
    dummy_weights = tmp_path / "dummy.pt"
    dummy_weights.write_bytes(b"dummy data")
    dummy_sha = hashlib.sha256(b"dummy data").hexdigest()

    dummy_protocol = tmp_path / "protocol.md"
    dummy_protocol.write_text(
        f"- **Weights SHA256**: `{dummy_sha}`\n- **Frozen Git Commit**: `aa2e9a64ecbf`\n"
    )

    with patch("scripts.run_final_test.is_git_tree_clean", return_value=False):
        ret = main([
            "--final",
            "--lock-file", str(dummy_lock),
            "--weights", str(dummy_weights),
            "--protocol", str(dummy_protocol),
        ])
        assert ret == 1


def test_successful_mock_run_and_refuse_second_run(tmp_path: Path) -> None:
    """Prove that first run succeeds with --final, and a second run is immediately refused."""
    dummy_lock = tmp_path / "test_lock.json"
    dummy_weights = tmp_path / "dummy.pt"
    dummy_weights.write_bytes(b"dummy data")
    dummy_sha = hashlib.sha256(b"dummy data").hexdigest()

    dummy_protocol = tmp_path / "protocol.md"
    dummy_protocol.write_text(
        f"- **Weights SHA256**: `{dummy_sha}`\n- **Frozen Git Commit**: `aa2e9a64ecbf`\n"
    )

    dummy_out = tmp_path / "report.md"

    # Mock run_pipeline_test to avoid running actual YOLO inference or touching test images
    mock_run_result = {"test_image_count": 2, "test_images": ["mock_1.jpg", "mock_2.jpg"], "status": "completed"}
    with patch("scripts.run_final_test.run_pipeline_test", return_value=mock_run_result):
        # FIRST RUN: Must succeed and create lockfile
        ret_1 = main([
            "--final",
            "--lock-file", str(dummy_lock),
            "--weights", str(dummy_weights),
            "--protocol", str(dummy_protocol),
            "--out", str(dummy_out),
            "--bypass-git-check",
        ])
        assert ret_1 == 0
        assert dummy_lock.exists()

        lock_data = json.loads(dummy_lock.read_text(encoding="utf-8"))
        assert lock_data["lock_active"] is True
        assert lock_data["test_image_count"] == 2
        assert lock_data["weights_sha256"] == dummy_sha

        # SECOND RUN: Must be strictly refused because dummy_lock exists!
        ret_2 = main([
            "--final",
            "--lock-file", str(dummy_lock),
            "--weights", str(dummy_weights),
            "--protocol", str(dummy_protocol),
            "--out", str(dummy_out),
            "--bypass-git-check",
        ])
        assert ret_2 == 1, "Second run must be strictly refused by the guardrail!"


def test_refusal_on_frozen_source_file_modification(tmp_path: Path) -> None:
    """Verifies that tampering with any frozen source file listed in protocol aborts execution."""
    dummy_lock = tmp_path / "test_lock.json"
    dummy_weights = tmp_path / "dummy.pt"
    dummy_weights.write_bytes(b"dummy data")
    dummy_sha = hashlib.sha256(b"dummy data").hexdigest()

    dummy_src = tmp_path / "fake_src.py"
    dummy_src.write_text("print('hello')", encoding="utf-8")
    tampered_sha = "0" * 64

    dummy_protocol = tmp_path / "protocol.md"
    dummy_protocol.write_text(
        f"- **Weights SHA256**: `{dummy_sha}`\n"
        f"- **Frozen Git Commit**: `aa2e9a64ecbf`\n\n"
        f"| File Path | SHA-256 Checksum |\n"
        f"| :--- | :--- |\n"
        f"| `{dummy_src.as_posix()}` | `{tampered_sha}` |\n",
        encoding="utf-8",
    )

    ret = main([
        "--final",
        "--lock-file", str(dummy_lock),
        "--weights", str(dummy_weights),
        "--protocol", str(dummy_protocol),
        "--bypass-git-check",
    ])
    assert ret == 1
    assert not dummy_lock.exists(), "Lockfile must not be written when frozen file fails integrity check"


def test_refusal_on_test_label_modification(tmp_path: Path) -> None:
    """Verifies that tampering with any test label file or box count aborts execution."""
    dummy_lock = tmp_path / "test_lock.json"
    dummy_weights = tmp_path / "dummy.pt"
    dummy_weights.write_bytes(b"dummy data")
    dummy_sha = hashlib.sha256(b"dummy data").hexdigest()

    dummy_lbl = tmp_path / "dummy_test.txt"
    dummy_lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    tampered_lbl_sha = "1" * 64

    dummy_protocol = tmp_path / "protocol.md"
    dummy_protocol.write_text(
        f"- **Weights SHA256**: `{dummy_sha}`\n"
        f"- **Frozen Git Commit**: `aa2e9a64ecbf`\n"
        f"**Total Test Images**: 1\n"
        f"**Total Test Bounding Boxes**: 1\n\n"
        f"| Test Label File | SHA-256 | Box Count |\n"
        f"| :--- | :--- | :--- |\n"
        f"| `{dummy_lbl.as_posix()}` | `{tampered_lbl_sha}` | 1 |\n",
        encoding="utf-8",
    )

    ret = main([
        "--final",
        "--lock-file", str(dummy_lock),
        "--weights", str(dummy_weights),
        "--protocol", str(dummy_protocol),
        "--bypass-git-check",
    ])
    assert ret == 1
    assert not dummy_lock.exists(), "Lockfile must not be written when test label fails integrity check"
