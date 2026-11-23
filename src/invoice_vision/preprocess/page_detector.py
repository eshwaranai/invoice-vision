"""Page boundary detection and perspective rectification module.

Identifies paper document boundaries using edge detection, contour analysis,
and polygon approximation, rectifying tilted camera perspectives into frontal scans.
Provides a safe fallback returning the original image when detection fails.
Supports multi-strategy candidate extraction (Canny+Otsu, Adaptive Threshold,
Morphological Gradient, Background Color Distance, Center-Seeded GrabCut) and
a composite scoring function for robust low-contrast boundary recovery.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points consistently: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    Args:
        pts: Array of shape (4, 2) containing (x, y) coordinates.

    Returns:
        np.ndarray of shape (4, 2) with dtype float32 ordered [TL, TR, BR, BL].
    """
    pts = np.asarray(pts, dtype=np.float32).reshape((4, 2))
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-Left has smallest sum (x + y), Bottom-Right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-Right has smallest difference (y - x), Bottom-Left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def warp_perspective(
    image: np.ndarray, corners: np.ndarray, target_width: Optional[int] = None
) -> np.ndarray:
    """Warp quadrilateral region of an image into a rectangular frontal scan.

    Args:
        image: Source image as BGR or grayscale numpy array.
        corners: Ordered (4, 2) corner coordinates [TL, TR, BR, BL].
        target_width: If set (e.g. 1600), scale output rectangle to have exactly this width,
            preserving the quadrilateral's aspect ratio.

    Returns:
        Perspective-rectified image with dimensions derived from the quad bounds.
    """
    ordered = order_corners(corners)
    tl, tr, br, bl = ordered

    # Compute width as maximum of top and bottom edge lengths
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    max_w = max(int(np.ceil(max(width_top, width_bot))), 50)

    # Compute height as maximum of left and right edge lengths
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = max(int(np.ceil(max(height_left, height_right))), 50)

    if target_width is not None and target_width > 0:
        out_w = int(target_width)
        aspect_ratio = float(max_h) / float(max_w)
        out_h = max(int(round(target_width * aspect_ratio)), 50)
    else:
        out_w = max_w
        out_h = max_h

    dst_pts = np.array(
        [[0.0, 0.0], [float(out_w - 1), 0.0], [float(out_w - 1), float(out_h - 1)], [0.0, float(out_h - 1)]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(ordered, dst_pts)
    warped = cv2.warpPerspective(
        image, M, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return warped


def score_quadrilateral(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
    grad_mag: np.ndarray,
) -> float:
    """Score candidate quadrilateral based on area ratio, rectangularity, and edge gradient support."""
    h, w = image_shape[:2]
    img_area = float(w * h)
    ordered = order_corners(quad)

    # 1. Area ratio penalty
    area = cv2.contourArea(ordered)
    area_ratio = area / img_area
    if area_ratio < 0.20 or area_ratio > 0.98:
        return -1.0
    s_area = max(0.0, 1.0 - 2.0 * abs(area_ratio - 0.78))

    # 2. Geometric regularity & rectangularity
    tl, tr, br, bl = ordered
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)

    if min(w_top, w_bot, h_left, h_right) < 30.0:
        return -1.0

    rw = min(w_top, w_bot) / max(w_top, w_bot, 1e-4)
    rh = min(h_left, h_right) / max(h_left, h_right, 1e-4)

    def angle_cos(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        v1 = p1 - p2
        v2 = p3 - p2
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        return abs(float(np.dot(v1, v2) / norm))

    cos_dev = float(np.mean([
        angle_cos(bl, tl, tr),
        angle_cos(tl, tr, br),
        angle_cos(tr, br, bl),
        angle_cos(br, bl, tl),
    ]))
    s_geom = max(0.0, 0.5 * (rw + rh) - 0.5 * cos_dev)

    # 3. Edge support: sample gradient magnitude along the 4 edges
    samples: List[float] = []
    for p1, p2 in [(tl, tr), (tr, br), (br, bl), (bl, tl)]:
        for alpha in np.linspace(0.05, 0.95, 20):
            pt = p1 + alpha * (p2 - p1)
            px = int(np.clip(round(pt[0]), 0, w - 1))
            py = int(np.clip(round(pt[1]), 0, h - 1))
            samples.append(float(grad_mag[py, px]))

    mean_edge_grad = float(np.mean(samples)) if samples else 0.0
    baseline_grad = float(np.mean(grad_mag)) + 1e-5
    s_edge = min(1.0, mean_edge_grad / (baseline_grad * 1.5))

    # Balanced composite score: prioritizes geometric plausibility when edge contrast is low
    return float(0.25 * s_edge + 0.45 * s_geom + 0.30 * s_area)


def _approx_quad_from_contour(cnt: np.ndarray) -> Optional[np.ndarray]:
    """Try multiple epsilon approximation scales to extract convex 4-point quad."""
    peri = cv2.arcLength(cnt, True)
    if peri < 100.0:
        return None
    for eps in [0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08]:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx.reshape(4, 2))

    # MinAreaRect fallback
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return order_corners(box)


def extract_candidates_canny_otsu(
    blurred: np.ndarray, min_area: float, max_area: float
) -> List[np.ndarray]:
    """Strategy 1: Canny edge detector + Otsu thresholding."""
    candidates = []
    edges = cv2.Canny(blurred, 30, 120)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed1 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)

    for bmap in [closed1, closed2]:
        cnts, _ = cv2.findContours(bmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if min_area <= a <= max_area:
                q = _approx_quad_from_contour(c)
                if q is not None:
                    candidates.append(q)
    return candidates


def extract_candidates_adaptive(
    blurred: np.ndarray, min_area: float, max_area: float
) -> List[np.ndarray]:
    """Strategy 2: Adaptive Gaussian thresholding."""
    candidates = []
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if min_area <= a <= max_area:
            q = _approx_quad_from_contour(c)
            if q is not None:
                candidates.append(q)
    return candidates


def extract_candidates_morph_gradient(
    blurred: np.ndarray, min_area: float, max_area: float
) -> List[np.ndarray]:
    """Strategy 3: Morphological gradient operator."""
    candidates = []
    k_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grad = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, k_grad)
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if min_area <= a <= max_area:
            q = _approx_quad_from_contour(c)
            if q is not None:
                candidates.append(q)
    return candidates


def extract_candidates_color_distance(
    image: np.ndarray, min_area: float, max_area: float
) -> List[np.ndarray]:
    """Strategy 4: Distance from border-estimated background color."""
    if len(image.shape) < 3:
        return []
    h, w = image.shape[:2]
    bw = min(20, min(h, w) // 10)
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[bw : h - bw, bw : w - bw] = False

    bg_color = np.median(image[border_mask], axis=0)
    color_dist = np.linalg.norm(image.astype(np.float32) - bg_color, axis=2)
    max_d = float(color_dist.max()) + 1e-5
    norm_dist = (color_dist / max_d * 255.0).astype(np.uint8)

    _, th = cv2.threshold(norm_dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)

    candidates = []
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if min_area <= a <= max_area:
            q = _approx_quad_from_contour(c)
            if q is not None:
                candidates.append(q)
    return candidates


def extract_candidates_grabcut(
    image: np.ndarray, min_area: float, max_area: float
) -> List[np.ndarray]:
    """Strategy 5: Fast center-seeded GrabCut foreground segmentation."""
    h, w = image.shape[:2]
    scale = 360.0 / float(w)
    small = cv2.resize(image, (360, int(round(h * scale))))
    sh, sw = small.shape[:2]

    mask_gc = np.zeros((sh, sw), np.uint8)
    rect = (int(0.08 * sw), int(0.08 * sh), int(0.84 * sw), int(0.84 * sh))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(small, mask_gc, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return []

    mask_bin = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        a_small = cv2.contourArea(c)
        a_full = a_small / (scale * scale)
        if min_area <= a_full <= max_area:
            q_small = _approx_quad_from_contour(c)
            if q_small is not None:
                q_full = q_small / scale
                candidates.append(q_full.astype(np.float32))
    return candidates


def detect_page_corners(
    image: np.ndarray,
    min_area_ratio: float = 0.20,
    strategy: str = "grabcut_fallback",
) -> Tuple[Optional[np.ndarray], bool]:
    """Detect the 4 corners of a document page in an image.

    Args:
        image: Input image (BGR or grayscale).
        min_area_ratio: Minimum fraction of total image area the page must cover.
        strategy: 'grabcut_fallback' (default: GrabCut primary, contour fallback),
                  'grabcut', 'canny_otsu', 'adaptive', 'morph_grad', 'color_dist',
                  or 'ensemble' (multi-method tournament).

    Returns:
        (ordered_corners, True) if a valid document quadrilateral is detected.
        (None, False) if detection fails.
    """
    if image is None or image.size == 0:
        return None, False

    h, w = image.shape[:2]
    img_area = float(w * h)
    min_area = img_area * min_area_ratio
    max_area = img_area * 0.98

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    if float(gray.std()) < 4.0 or int(np.ptp(gray)) < 12:
        return None, False

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Compute Sobel gradient magnitude once for candidate scoring
    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobel_x, sobel_y)

    # Strategy: GrabCut Primary with Contour Fallback (default recommended pipeline)
    if strategy == "grabcut_fallback":
        # 1. Primary: Center-seeded GrabCut (robust to low-contrast boundaries)
        gc_candidates = extract_candidates_grabcut(image, min_area, max_area)
        scored_gc = []
        for q in gc_candidates:
            score = score_quadrilateral(q, (h, w), grad_mag)
            if score > 0.35:
                scored_gc.append((score, q))

        if scored_gc:
            scored_gc.sort(key=lambda x: x[0], reverse=True)
            return order_corners(scored_gc[0][1]), True

        # 2. Fallback: Morphological gradient + Canny contour analysis
        fallback_candidates = []
        fallback_candidates.extend(extract_candidates_morph_gradient(blurred, min_area, max_area))
        fallback_candidates.extend(extract_candidates_canny_otsu(blurred, min_area, max_area))

        scored_fb = []
        for q in fallback_candidates:
            score = score_quadrilateral(q, (h, w), grad_mag)
            if score > 0.30:
                scored_fb.append((score, q))

        if scored_fb:
            scored_fb.sort(key=lambda x: x[0], reverse=True)
            return order_corners(scored_fb[0][1]), True

        return None, False

    # Standalone or Ensemble strategies
    candidates: List[np.ndarray] = []

    if strategy in ("canny_otsu", "ensemble"):
        candidates.extend(extract_candidates_canny_otsu(blurred, min_area, max_area))
    if strategy in ("adaptive", "ensemble"):
        candidates.extend(extract_candidates_adaptive(blurred, min_area, max_area))
    if strategy in ("morph_grad", "ensemble"):
        candidates.extend(extract_candidates_morph_gradient(blurred, min_area, max_area))
    if strategy in ("color_dist", "ensemble"):
        candidates.extend(extract_candidates_color_distance(image, min_area, max_area))
    if strategy in ("grabcut", "ensemble"):
        candidates.extend(extract_candidates_grabcut(image, min_area, max_area))

    if not candidates:
        return None, False

    # Score each candidate and pick highest score
    scored_candidates = []
    for q in candidates:
        score = score_quadrilateral(q, (h, w), grad_mag)
        if score > 0.30:  # Minimum quality threshold
            scored_candidates.append((score, q))

    if not scored_candidates:
        return None, False

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    best_quad = scored_candidates[0][1]
    return order_corners(best_quad), True


def page_detect_and_warp(
    image: np.ndarray,
    min_area_ratio: float = 0.20,
    target_width: Optional[int] = None,
    strategy: str = "grabcut_fallback",
) -> Tuple[np.ndarray, bool, Optional[np.ndarray]]:
    """Detect document boundaries and rectify perspective with safe fallback.

    Args:
        image: Input document photo (BGR numpy array).
        min_area_ratio: Minimum area fraction for document candidate.
        target_width: If set, scales rectified page to canonical width (e.g. 1600 px).
        strategy: Detection method ('ensemble', 'canny_otsu', etc.).

    Returns:
        (rectified_or_original_image, is_detected, detected_corners)
    """
    corners, detected = detect_page_corners(image, min_area_ratio=min_area_ratio, strategy=strategy)

    if not detected or corners is None:
        return image, False, None

    warped = warp_perspective(image, corners, target_width=target_width)
    return warped, True, corners
