"""Invoice Vision Preprocessing Package."""

from invoice_vision.preprocess.deskew import deskew_image, estimate_skew_angle
from invoice_vision.preprocess.enhancement import apply_adaptive_threshold, apply_clahe, apply_denoise
from invoice_vision.preprocess.page_detector import detect_page_corners, order_corners, page_detect_and_warp, warp_perspective
from invoice_vision.preprocess.pipeline import PreprocessPipeline, PreprocessResult
from invoice_vision.preprocess.quality_gate import QualityResult, check_image_quality, compute_blur_score, compute_brightness
from invoice_vision.preprocess.shadow_removal import remove_shadows

__all__ = [
    "order_corners",
    "warp_perspective",
    "detect_page_corners",
    "page_detect_and_warp",
    "estimate_skew_angle",
    "deskew_image",
    "compute_blur_score",
    "compute_brightness",
    "check_image_quality",
    "QualityResult",
    "remove_shadows",
    "apply_clahe",
    "apply_denoise",
    "apply_adaptive_threshold",
    "PreprocessPipeline",
    "PreprocessResult",
]
