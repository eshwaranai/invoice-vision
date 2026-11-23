"""Image enhancement utilities (CLAHE, edge-preserving denoising, binarization)."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local image contrast using Contrast Limited Adaptive Histogram Equalization.

    For color images, operates on the lightness (L) channel in LAB color space to
    preserve color saturation and prevent chromatic noise amplification.

    Args:
        image: BGR or grayscale numpy array.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization (e.g. 8x8).

    Returns:
        Contrast-enhanced image.
    """
    if image is None or image.size == 0:
        return image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)


def apply_denoise(
    image: np.ndarray,
    method: str = "bilateral",
    d: int = 5,
    sigma: float = 30.0,
) -> np.ndarray:
    """Apply edge-preserving noise reduction.

    Args:
        image: BGR or grayscale image.
        method: 'bilateral' (fast, CPU-friendly) or 'fast_nl_means' (thorough).
        d: Diameter of pixel neighborhood for bilateral filter.
        sigma: Filter sigma in the color and coordinate space.

    Returns:
        Denoised image.
    """
    if image is None or image.size == 0:
        return image

    if method == "bilateral":
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma, sigmaSpace=sigma)
    elif method == "fast_nl_means":
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
        else:
            return cv2.fastNlMeansDenoising(image, None, h=3, templateWindowSize=7, searchWindowSize=21)
    else:
        # Default fallback to bilateral
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma, sigmaSpace=sigma)


def apply_adaptive_threshold(
    image: np.ndarray,
    block_size: int = 21,
    c: int = 10,
) -> np.ndarray:
    """Binarize document image using adaptive Gaussian thresholding."""
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    b_size = block_size if (block_size % 2 == 1) else (block_size + 1)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b_size, c
    )
