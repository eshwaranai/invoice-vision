"""Unit tests for the OpenCV preprocessing pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from invoice_vision.preprocess.deskew import deskew_image, estimate_skew_angle
from invoice_vision.preprocess.enhancement import apply_clahe, apply_denoise
from invoice_vision.preprocess.page_detector import (
    detect_page_corners,
    order_corners,
    page_detect_and_warp,
    warp_perspective,
)
from invoice_vision.preprocess.quality_gate import (
    check_image_quality,
    compute_blur_score,
    compute_brightness,
)
from invoice_vision.preprocess.shadow_removal import remove_shadows


def test_order_corners_canonical() -> None:
    # Supply corners in scrambled order
    # Expected: TL=(50, 50), TR=(300, 60), BR=(290, 400), BL=(40, 390)
    scrambled = np.array([[290.0, 400.0], [50.0, 50.0], [40.0, 390.0], [300.0, 60.0]])
    ordered = order_corners(scrambled)

    np.testing.assert_allclose(ordered[0], [50.0, 50.0], atol=1e-3)
    np.testing.assert_allclose(ordered[1], [300.0, 60.0], atol=1e-3)
    np.testing.assert_allclose(ordered[2], [290.0, 400.0], atol=1e-3)
    np.testing.assert_allclose(ordered[3], [40.0, 390.0], atol=1e-3)


def test_warp_perspective_recovery() -> None:
    # Create a 450x360 white rectangle on black background
    canvas = np.zeros((600, 800, 3), dtype=np.uint8)
    src_w, src_h = 450, 360
    src_rect = np.float32([[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]])

    # Perspective destination corners (width ~450, height ~360)
    dst_corners = np.float32([[150.0, 100.0], [600.0, 120.0], [550.0, 480.0], [120.0, 460.0]])
    H = cv2.getPerspectiveTransform(src_rect, dst_corners)
    synthetic_doc = np.ones((src_h, src_w, 3), dtype=np.uint8) * 240
    cv2.putText(synthetic_doc, "INVOICE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    cv2.warpPerspective(synthetic_doc, H, (800, 600), dst=canvas)

    # Detect page corners on synthetic image
    detected_corners, success = detect_page_corners(canvas, min_area_ratio=0.10)
    assert success is True
    assert detected_corners is not None

    # Check recovered corners against known ground-truth corners within 10 px tolerance
    corner_errors = np.linalg.norm(detected_corners - dst_corners, axis=1)
    mean_error = float(np.mean(corner_errors))
    assert mean_error < 10.0, f"Mean corner error {mean_error:.2f} exceeded tolerance 10 px"

    # Warp and verify output dimensions match expected quad dimensions
    warped = warp_perspective(canvas, detected_corners)
    assert abs(warped.shape[1] - src_w) < 20
    assert abs(warped.shape[0] - src_h) < 20


def test_page_detect_and_warp_fallback() -> None:
    # Completely uniform blank image with no edges
    blank = np.ones((400, 400, 3), dtype=np.uint8) * 128
    result_img, detected, corners = page_detect_and_warp(blank, min_area_ratio=0.20)

    assert detected is False
    assert corners is None
    assert result_img.shape == blank.shape
    np.testing.assert_array_equal(result_img, blank)


def test_quality_gate_pass() -> None:
    # Sharp, well-lit synthetic document
    sharp_img = np.ones((800, 600, 3), dtype=np.uint8) * 220
    for y in range(50, 750, 40):
        cv2.line(sharp_img, (50, y), (550, y), (20, 20, 20), 2)
        cv2.putText(sharp_img, f"Text Line {y}", (60, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    result = check_image_quality(sharp_img, min_blur=50.0)
    assert result.status == "PASS"
    assert result.blur_score > 50.0
    assert "PASS" in result.human_readable_reason


def test_quality_gate_blurred_reject() -> None:
    # Heavily blurred image
    sharp_img = np.ones((800, 600, 3), dtype=np.uint8) * 200
    cv2.putText(sharp_img, "Blur Test", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    heavily_blurred = cv2.GaussianBlur(sharp_img, (51, 51), sigmaX=25.0)

    result = check_image_quality(heavily_blurred, min_blur=80.0, reject_blur=35.0)
    assert result.status in ("WARN", "REJECT")
    assert any("blur" in r.lower() for r in result.reasons)


def test_quality_gate_dark_reject() -> None:
    # Critically dark image (mean brightness < 30)
    dark_img = np.ones((800, 600, 3), dtype=np.uint8) * 15
    result = check_image_quality(dark_img)
    assert result.status == "REJECT"
    assert any("underexposed" in r.lower() for r in result.reasons)


def test_quality_gate_low_resolution_reject() -> None:
    tiny_img = np.ones((200, 200, 3), dtype=np.uint8) * 200
    result = check_image_quality(tiny_img, min_resolution=(500, 500))
    assert result.status == "REJECT"
    assert any("resolution" in r.lower() for r in result.reasons)


def test_shadow_removal_flattens_gradient() -> None:
    # Create an image with text and an artificial dark shadow band
    img = np.ones((400, 400, 3), dtype=np.uint8) * 220
    # Add dark text line across image
    img[190:210, 40:360] = 30
    ramp = np.linspace(0.3, 1.0, 400).reshape(1, 400, 1)
    shadowed = np.clip(img * ramp, 0, 255).astype(np.uint8)

    deshadowed = remove_shadows(shadowed, kernel_size=25)
    assert deshadowed.shape == shadowed.shape

    # Left background edge (originally dark ~66) should be brightened significantly (> 200)
    left_bg_before = float(shadowed[:150, :50].mean())
    left_bg_after = float(deshadowed[:150, :50].mean())
    assert left_bg_after > left_bg_before + 100.0

    # Text contrast should remain dark
    text_pixel = int(deshadowed[200, 100, 0])
    assert text_pixel < 80


def test_clahe_and_denoise() -> None:
    img = np.ones((300, 300, 3), dtype=np.uint8) * 128
    enhanced = apply_clahe(img, clip_limit=2.0)
    assert enhanced.shape == img.shape

    noisy = np.clip(img + np.random.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
    denoised = apply_denoise(noisy, method="bilateral")
    assert denoised.shape == noisy.shape
