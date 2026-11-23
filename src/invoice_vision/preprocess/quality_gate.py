"""Image quality assessment gate.

Measures Laplacian blur variance, mean brightness, resolution, and glare saturation,
returning a triage decision: PASS, WARN, or REJECT with human-readable rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class QualityResult:
    """Represents the outcome of document quality assessment."""

    status: str  # "PASS", "WARN", or "REJECT"
    blur_score: float
    brightness: float  # Relative page brightness
    resolution: Tuple[int, int]  # (width, height)
    glare_ratio: float  # Kept for telemetry; non-blocking
    reasons: List[str]
    human_readable_reason: str

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "blur_score": round(self.blur_score, 2),
            "brightness": round(self.brightness, 2),
            "resolution": list(self.resolution),
            "glare_ratio": round(self.glare_ratio, 4),
            "reasons": self.reasons,
            "human_readable_reason": self.human_readable_reason,
        }


def compute_blur_score(image: np.ndarray) -> float:
    """Compute blur variance using the Laplacian operator.

    Higher scores indicate sharper edges; low scores (< 28) indicate severe defocus/motion blur.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_brightness(image: np.ndarray) -> float:
    """Compute mean grayscale pixel intensity across whole canvas (0 to 255)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(gray.mean())


def compute_relative_page_brightness(
    image: np.ndarray, quad: Optional[np.ndarray] = None
) -> float:
    """Compute mean brightness inside the document region rather than the whole canvas.

    If a detected 4-point quadrilateral is provided, computes mean intensity inside the quad mask.
    Otherwise, samples the central 60% window of the image where document paper sits,
    avoiding dark or textured desk border pixels that distort global brightness.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    h, w = gray.shape[:2]

    if quad is not None and len(quad) == 4:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(quad)], 255)
        page_pixels = gray[mask == 255]
        if page_pixels.size > 0:
            return float(page_pixels.mean())

    # Fallback: central 60% window where document paper sits
    y1, y2 = int(h * 0.20), int(h * 0.80)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    if y2 > y1 and x2 > x1:
        return float(gray[y1:y2, x1:x2].mean())

    return float(gray.mean())


def compute_glare_ratio(image: np.ndarray, threshold: int = 250) -> float:
    """Compute fraction of saturated highlight pixels (glare telemetry)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    saturated = (gray >= threshold).sum()
    return float(saturated) / float(gray.size)


def check_image_quality(
    image: np.ndarray,
    page_quad: Optional[np.ndarray] = None,
    min_blur: float = 45.0,
    reject_blur: float = 28.0,
    min_brightness: float = 70.0,
    reject_brightness: float = 45.0,
    min_resolution: Tuple[int, int] = (500, 500),
) -> QualityResult:
    """Assess overall image quality against operational criteria.

    Note on Glare and Overexposure:
        Global intensity thresholds (e.g. max_brightness 230 or glare_ratio 0.15) were removed.
        In clean digital invoices and white paper documents, white background pixels naturally
        saturate at 250-255 across 70-85% of the page area. A naive global glare threshold either
        falsely flags 100% of clean digital scans as defective or requires dummy non-triggerable
        thresholds (glare 0.95 or max_brightness 255).
        Document exposure is measured via relative page brightness to reliably catch underexposed
        photos (< 45.0) without false positives on clean white scans.

    Returns:
        QualityResult with status ('PASS', 'WARN', 'REJECT') and diagnostic reason.
    """
    if image is None or image.size == 0:
        return QualityResult(
            status="REJECT",
            blur_score=0.0,
            brightness=0.0,
            resolution=(0, 0),
            glare_ratio=0.0,
            reasons=["Empty or null image buffer."],
            human_readable_reason="REJECT: Image cannot be read or has 0 bytes.",
        )

    h, w = image.shape[:2]
    res = (w, h)
    blur = compute_blur_score(image)
    page_bright = compute_relative_page_brightness(image, page_quad)
    glare = compute_glare_ratio(image)

    reject_reasons: List[str] = []
    warn_reasons: List[str] = []

    # 1. Resolution Check
    if w < min_resolution[0] or h < min_resolution[1]:
        reject_reasons.append(
            f"Resolution {w}x{h} is below minimum requirement {min_resolution[0]}x{min_resolution[1]}."
        )

    # 2. Blur Check
    if blur < reject_blur:
        reject_reasons.append(f"Severe defocus/motion blur (Laplacian score: {blur:.1f} < {reject_blur}).")
    elif blur < min_blur:
        warn_reasons.append(f"Borderline image sharpness (Laplacian score: {blur:.1f} < {min_blur}).")

    # 3. Page Relative Brightness Check
    if page_bright < reject_brightness:
        reject_reasons.append(f"Critically underexposed page (page brightness: {page_bright:.1f} < {reject_brightness}).")
    elif page_bright < min_brightness:
        warn_reasons.append(f"Dim page lighting conditions (page brightness: {page_bright:.1f} < {min_brightness}).")

    # Final Decision Triage
    if reject_reasons:
        status = "REJECT"
        all_reasons = reject_reasons + warn_reasons
        human_msg = "REJECT: " + "; ".join(reject_reasons)
    elif warn_reasons:
        status = "WARN"
        all_reasons = warn_reasons
        human_msg = "WARN: " + "; ".join(warn_reasons)
    else:
        status = "PASS"
        all_reasons = ["Image meets all quality criteria."]
        human_msg = f"PASS: High quality document (Sharpness: {blur:.1f}, Page Brightness: {page_bright:.1f})."

    return QualityResult(
        status=status,
        blur_score=blur,
        brightness=page_bright,
        resolution=res,
        glare_ratio=glare,
        reasons=all_reasons,
        human_readable_reason=human_msg,
    )
