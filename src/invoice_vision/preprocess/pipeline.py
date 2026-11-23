"""Modular preprocessing pipeline orchestrator.

Loads configuration toggles from configs/default.yaml and executes
document rectification, quality assessment, and enhancement stages sequentially.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import yaml

from invoice_vision.preprocess.deskew import deskew_image
from invoice_vision.preprocess.enhancement import apply_clahe, apply_denoise
from invoice_vision.preprocess.page_detector import page_detect_and_warp
from invoice_vision.preprocess.quality_gate import QualityResult, check_image_quality
from invoice_vision.preprocess.shadow_removal import remove_shadows


@dataclass
class PreprocessResult:
    """Encapsulates the output of the full preprocessing pipeline."""

    image: np.ndarray
    stages_applied: List[str] = field(default_factory=list)
    stage_outputs: Dict[str, np.ndarray] = field(default_factory=dict)
    page_detected: bool = False
    corners: Optional[np.ndarray] = None
    quality: Optional[QualityResult] = None
    skew_angle: float = 0.0
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages_applied": self.stages_applied,
            "page_detected": self.page_detected,
            "corners": self.corners.tolist() if self.corners is not None else None,
            "quality": self.quality.to_dict() if self.quality is not None else None,
            "skew_angle": round(self.skew_angle, 2),
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }


class PreprocessPipeline:
    """Config-driven preprocessing pipeline with per-stage toggles."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None, config_dict: Optional[Dict[str, Any]] = None) -> None:
        if config_dict is not None:
            self.config = config_dict
        elif config_path is not None and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            default_cfg_path = Path("configs/default.yaml")
            if default_cfg_path.exists():
                with open(default_cfg_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = {}

        self.stages_cfg = self.config.get("stages", {})

    def process(
        self,
        image: np.ndarray,
        overrides: Optional[Dict[str, bool]] = None,
        save_intermediate: bool = False,
    ) -> PreprocessResult:
        """Run preprocessing pipeline on an input image.

        Args:
            image: BGR document image.
            overrides: Optional dict overriding enable/disable per stage
                       (e.g. {'page_detect_and_warp': True, 'shadow_removal': False}).
            save_intermediate: If True, saves intermediate numpy images into stage_outputs dict.

        Returns:
            PreprocessResult containing rectified image and execution metadata.
        """
        current_img = image.copy()
        result = PreprocessResult(image=current_img)
        if save_intermediate:
            result.stage_outputs["raw"] = current_img.copy()

        def is_enabled(stage_key: str, default: bool = True) -> bool:
            if overrides and stage_key in overrides:
                return overrides[stage_key]
            cfg = self.stages_cfg.get(stage_key, {})
            if isinstance(cfg, dict):
                return bool(cfg.get("enabled", default))
            return bool(cfg)

        # ----------------------------------------------------------------------
        # Stage 1: Document Boundary Detection & Perspective Warp
        # ----------------------------------------------------------------------
        if is_enabled("page_detect_and_warp", default=True):
            t0 = time.perf_counter()
            min_ratio = float(self.stages_cfg.get("page_detection", {}).get("min_quad_area_ratio", 0.20))
            warped, detected, corners = page_detect_and_warp(current_img, min_area_ratio=min_ratio)
            t1 = time.perf_counter()

            result.timings_ms["page_detect_and_warp"] = (t1 - t0) * 1000.0
            result.page_detected = detected
            result.corners = corners
            if detected:
                current_img = warped
                result.stages_applied.append("page_detect_and_warp")
            if save_intermediate:
                result.stage_outputs["page_detect_and_warp"] = current_img.copy()

        # ----------------------------------------------------------------------
        # Stage 2: Deskewing
        # ----------------------------------------------------------------------
        if is_enabled("deskew", default=True):
            t0 = time.perf_counter()
            max_angle = float(self.stages_cfg.get("deskew", {}).get("max_angle", 30.0))
            deskewed, angle = deskew_image(current_img, max_angle=max_angle)
            t1 = time.perf_counter()

            result.timings_ms["deskew"] = (t1 - t0) * 1000.0
            result.skew_angle = angle
            if abs(angle) >= 0.5:
                current_img = deskewed
                result.stages_applied.append("deskew")
            if save_intermediate:
                result.stage_outputs["deskew"] = current_img.copy()

        # ----------------------------------------------------------------------
        # Stage 3: Quality Gate Check
        # ----------------------------------------------------------------------
        if is_enabled("quality_gate", default=True):
            t0 = time.perf_counter()
            q_cfg = self.stages_cfg.get("quality_check", {})
            min_blur = float(q_cfg.get("blur_threshold", 80.0))
            q_res = check_image_quality(current_img, min_blur=min_blur)
            t1 = time.perf_counter()

            result.timings_ms["quality_gate"] = (t1 - t0) * 1000.0
            result.quality = q_res
            result.stages_applied.append("quality_gate")

        # ----------------------------------------------------------------------
        # Stage 4: Shadow Removal (Background Division)
        # ----------------------------------------------------------------------
        if is_enabled("shadow_removal", default=True):
            t0 = time.perf_counter()
            k_size = int(self.stages_cfg.get("shadow_removal", {}).get("kernel_size", 25))
            deshadowed = remove_shadows(current_img, kernel_size=k_size)
            t1 = time.perf_counter()

            result.timings_ms["shadow_removal"] = (t1 - t0) * 1000.0
            current_img = deshadowed
            result.stages_applied.append("shadow_removal")
            if save_intermediate:
                result.stage_outputs["shadow_removal"] = current_img.copy()

        # ----------------------------------------------------------------------
        # Stage 5: CLAHE Local Contrast Enhancement
        # ----------------------------------------------------------------------
        if is_enabled("clahe", default=True):
            t0 = time.perf_counter()
            clahe_cfg = self.stages_cfg.get("clahe", {})
            clip_lim = float(clahe_cfg.get("clip_limit", 2.0))
            enhanced = apply_clahe(current_img, clip_limit=clip_lim)
            t1 = time.perf_counter()

            result.timings_ms["clahe"] = (t1 - t0) * 1000.0
            current_img = enhanced
            result.stages_applied.append("clahe")
            if save_intermediate:
                result.stage_outputs["clahe"] = current_img.copy()

        # ----------------------------------------------------------------------
        # Stage 6: Denoising
        # ----------------------------------------------------------------------
        if is_enabled("denoise", default=False):
            t0 = time.perf_counter()
            denoise_cfg = self.stages_cfg.get("denoise", {})
            method = str(denoise_cfg.get("method", "bilateral"))
            denoised = apply_denoise(current_img, method=method)
            t1 = time.perf_counter()

            result.timings_ms["denoise"] = (t1 - t0) * 1000.0
            current_img = denoised
            result.stages_applied.append("denoise")
            if save_intermediate:
                result.stage_outputs["denoise"] = current_img.copy()

        result.image = current_img
        return result
