"""Shadow removal and uneven illumination correction module.

Flattens non-uniform lighting and removes shadows cast by phones/hands
using background division with morphological structuring elements.
"""

from __future__ import annotations

import cv2
import numpy as np


def remove_shadows(image: np.ndarray, kernel_size: int = 25) -> np.ndarray:
    """Remove soft shadows and normalize lighting gradients across a document.

    Uses morphological dilation and median filtering to estimate the low-frequency
    background illumination map, then divides the image by the background.

    Args:
        image: BGR or grayscale input document image.
        kernel_size: Size of the morphological structuring kernel (must be odd).

    Returns:
        Illumination-equalized image.
    """
    if image is None or image.size == 0:
        return image

    k = kernel_size if (kernel_size % 2 == 1) else (kernel_size + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    is_color = len(image.shape) == 3
    planes = cv2.split(image) if is_color else [image]

    norm_planes = []
    for plane in planes:
        # Step 1: Dilate to suppress dark text characters and estimate background
        dilated = cv2.dilate(plane, kernel)

        # Step 2: Smooth the estimated background illumination
        bg_illumination = cv2.medianBlur(dilated, k).astype(np.float32)

        # Step 3: Divide plane by estimated background illumination
        divided = (plane.astype(np.float32) / np.maximum(bg_illumination, 1.0)) * 255.0
        norm = np.clip(divided, 0.0, 255.0).astype(np.uint8)
        norm_planes.append(norm)

    if is_color:
        return cv2.merge(norm_planes)
    return norm_planes[0]
