"""Document deskewing module.

Estimates text orientation skew angle using Hough line transform and rotated
bounding geometry, rotating documents back to a horizontal reading axis.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def estimate_skew_angle(image: np.ndarray, max_angle: float = 30.0) -> float:
    """Estimate orientation skew angle of text lines in degrees.

    Args:
        image: BGR or grayscale document image.
        max_angle: Maximum permissible skew angle in degrees.

    Returns:
        Skew angle in degrees (positive indicates clockwise tilt).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Invert image so text is white on black background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to merge letters into words/lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Strategy 1: Probabilistic Hough Lines
    edges = cv2.Canny(dilated, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=15)

    angles: list[float] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < 1e-4:
                continue
            deg = np.degrees(np.arctan2(dy, dx))
            # Keep angles close to horizontal
            if abs(deg) <= max_angle:
                angles.append(deg)

    if angles:
        return float(np.median(angles))

    # Strategy 2 Fallback: minAreaRect on foreground text mask
    pts = np.column_stack(np.where(dilated > 0))
    if len(pts) > 50:
        # minAreaRect expects (x, y)
        pts_xy = np.fliplr(pts)
        rect = cv2.minAreaRect(pts_xy)
        angle = rect[-1]
        if angle < -45.0:
            angle = -(90.0 + angle)
        elif angle > 45.0:
            angle = 90.0 - angle
        else:
            angle = -angle

        if abs(angle) <= max_angle:
            return float(angle)

    return 0.0


def deskew_image(
    image: np.ndarray, max_angle: float = 30.0, min_correct_angle: float = 0.5
) -> Tuple[np.ndarray, float]:
    """Deskew image by rotating back to horizontal alignment.

    Args:
        image: Input document image.
        max_angle: Maximum rotation tolerance in degrees.
        min_correct_angle: Minimum angle threshold below which no rotation is performed.

    Returns:
        (deskewed_image, estimated_angle_degrees)
    """
    angle = estimate_skew_angle(image, max_angle=max_angle)

    if abs(angle) < min_correct_angle:
        return image, 0.0

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute bounding box of rotated image to avoid cropping corners
    cos_val = abs(M[0, 0])
    sin_val = abs(M[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle
