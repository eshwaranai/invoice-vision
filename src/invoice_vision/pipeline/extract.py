"""End-to-end invoice field extraction pipeline using YOLOv8 detection and crop-level EasyOCR.

Pipeline:
  Image -> YOLOv8 Detection (imgsz 1024) -> Class-Specific Crop (4-6px pad)
        -> Crop-Level EasyOCR (with 3x bicubic upscale for GSTIN)
        -> Domain-Specific Value Parsers & Checksum Validators
        -> Structured Extraction Output with Confidence & Review Gate
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from dateutil import parser as date_parser

from invoice_vision.parse.spatial_rules import (
    position_aware_gstin_correct,
    validate_gstin_checksum,
)

TARGET_CLASSES = ["vendor", "invoice_no", "date", "gstin", "table", "total"]

INV_NO_CLEAN_PREFIX = re.compile(
    r"^(?:tax\s+)?(?:invoice|inv|bill)\s*(?:no\.?|num(?:ber)?\.?|#)?\s*[:\-\.]*\s*",
    re.IGNORECASE,
)
DATE_CLEAN_PREFIX = re.compile(
    r"^(?:invoice\s*date|bill\s*date|purchase\s*date|dated|date)\s*[:\-\.]*\s*",
    re.IGNORECASE,
)
VENDOR_CLEAN_PREFIX = re.compile(
    r"^(?:sold\s*by|seller|from|supplier)\s*[:\-\.]*\s*",
    re.IGNORECASE,
)
TOTAL_CLEAN_PREFIX = re.compile(
    r"^(?:grand\s*total|net\s*payable|amount\s*payable|total\s*payable|total\s*amount|total)\s*[:\-\.]*\s*",
    re.IGNORECASE,
)
GSTIN_CLEAN_PREFIX = re.compile(
    r"^(?:gstin\s*\/?\s*uin|gstin|gst\s*no\.?|gst\s*number|gst\s*reg\.?)\s*[:\-\.]*\s*",
    re.IGNORECASE,
)

DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b"),
    re.compile(
        r"\b(\d{1,2}[\s\-\/\.](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-\/\.]\d{2,4})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b"),
]


@dataclass
class FieldExtraction:
    """Structured extraction result for a single invoice field."""

    value: Optional[str]
    detector_confidence: float = 0.0
    ocr_confidence: float = 0.0
    validation_result: Union[bool, str] = False
    combined_confidence: float = 0.0
    needs_review: bool = True
    source_box: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "detector_confidence": round(self.detector_confidence, 4),
            "ocr_confidence": round(self.ocr_confidence, 4),
            "validation_result": self.validation_result,
            "combined_confidence": round(self.combined_confidence, 4),
            "needs_review": self.needs_review,
            "source_box": self.source_box,
        }


def parse_invoice_no(raw_text: str) -> Tuple[Optional[str], bool]:
    """Clean invoice number by removing prefix keywords and leading/trailing punctuation."""
    cleaned = INV_NO_CLEAN_PREFIX.sub("", raw_text).strip()
    cleaned = cleaned.strip(":-# \t\n.,")
    if len(cleaned) >= 2:
        return cleaned, True
    return (cleaned if cleaned else None), False


def parse_date_field(raw_text: str) -> Tuple[Optional[str], bool]:
    """Extract and format purchase date as DD-MM-YYYY using dayfirst=True."""
    cleaned = DATE_CLEAN_PREFIX.sub("", raw_text).strip()
    # Search for date pattern
    candidate: Optional[str] = None
    for pat in DATE_PATTERNS:
        m = pat.search(cleaned)
        if m:
            candidate = m.group(1).strip()
            break
    if not candidate:
        candidate = cleaned.strip()

    if candidate:
        try:
            dt = date_parser.parse(candidate, dayfirst=True)
            return dt.strftime("%d-%m-%Y"), True
        except Exception:
            pass

    return (candidate if candidate else None), False


def parse_total_amount(raw_text: str) -> Tuple[Optional[str], bool]:
    """Extract monetary amount without inventing decimal places."""
    cleaned = TOTAL_CLEAN_PREFIX.sub("", raw_text).strip()
    s = re.sub(r"(?:inr|rs\.?|rs/-|₹|<|>|\$)", " ", cleaned, flags=re.IGNORECASE).strip()
    # Match decimal or integer amount
    matches = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?|\d+(?:\.\d+)?", s)
    if not matches:
        return None, False

    cand = matches[-1].replace(",", "").strip()
    try:
        val = float(cand)
        if val > 0.0:
            return cand, True
    except ValueError:
        pass
    return (cand if cand else None), False


GSTIN_PAN_REGEX = re.compile(
    r"\b([0-9A-Za-z]{2}[A-Za-z]{5}[0-9A-Za-z]{4}[A-Za-z]{1}[0-9A-Za-z]{3})\b",
    re.IGNORECASE,
)


def parse_gstin_field(raw_text: str) -> Tuple[Optional[str], bool]:
    """Extract 15-character GSTIN, apply position-aware correction, and validate checksum."""
    cleaned = GSTIN_CLEAN_PREFIX.sub("", raw_text).strip()
    
    # Priority 1: Tokens matching Indian GSTIN / PAN layout (2 alphanumeric + 5 alpha + 4 num + 1 alpha + 3 alphanumeric)
    m = GSTIN_PAN_REGEX.search(cleaned)
    candidates: List[str] = []
    if m:
        candidates.append(m.group(1))

    # Priority 2: Extract any 15-char alphanumeric sequence
    if not candidates:
        raw_alphanum = re.sub(r"[\s\-_:/\.]", "", cleaned)
        candidates = re.findall(r"[0-9A-Za-z]{15}", raw_alphanum)
        if not candidates:
            # Fallback for slight length deviations (e.g. 14 or 16)
            candidates = re.findall(r"[0-9A-Za-z]{14,16}", raw_alphanum)

    if not candidates:
        return (cleaned if cleaned else None), False

    best_corrected: Optional[str] = None
    best_valid: bool = False

    for cand in candidates:
        token = cand[:15]
        corrected = position_aware_gstin_correct(token)
        is_valid = validate_gstin_checksum(corrected)
        if is_valid:
            return corrected, True

        # Check if single-character confusion (I/1, O/0, Z/7) at ambiguous positions allows checksum to pass
        chars = list(corrected)
        for pos in range(len(chars)):
            orig = chars[pos]
            for sub_from, sub_to in [("I", "1"), ("1", "I"), ("O", "0"), ("0", "O"), ("Z", "7"), ("7", "Z")]:
                if orig == sub_from:
                    chars[pos] = sub_to
                    cand_try = "".join(chars)
                    if validate_gstin_checksum(cand_try):
                        return cand_try, True
                    chars[pos] = orig

        if best_corrected is None:
            best_corrected = corrected
            best_valid = is_valid

    return best_corrected, best_valid


def parse_vendor_field(raw_text: str) -> Tuple[Optional[str], bool]:
    """Clean vendor block by joining lines and removing boilerplate prefixes."""
    cleaned = VENDOR_CLEAN_PREFIX.sub("", raw_text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) >= 3:
        return cleaned, True
    return (cleaned if cleaned else None), False


def parse_field_value(class_name: str, raw_text: str) -> Tuple[Optional[str], Union[bool, str]]:
    """Route raw crop text to appropriate class-specific parser."""
    if class_name == "invoice_no":
        return parse_invoice_no(raw_text)
    elif class_name == "date":
        return parse_date_field(raw_text)
    elif class_name == "total":
        return parse_total_amount(raw_text)
    elif class_name == "gstin":
        return parse_gstin_field(raw_text)
    elif class_name == "vendor":
        return parse_vendor_field(raw_text)
    return raw_text.strip(), True


class InvoiceExtractor:
    """Production invoice extraction pipeline combining YOLOv8 and EasyOCR."""

    def __init__(
        self,
        weights_path: Union[str, Path] = "data/models/layouts_s0_best.pt",
        conf_thresh: float = 0.05,
        high_conf_bar: float = 0.25,
        imgsz: int = 1024,
        device: str = "cpu",
        crop_padding: int = 5,
        model_storage_dir: Union[str, Path] = "data/models/easyocr",
        ocr_reader: Any = None,
        yolo_model: Any = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.conf_thresh = conf_thresh
        self.high_conf_bar = high_conf_bar
        self.imgsz = imgsz
        self.device = device
        self.crop_padding = crop_padding
        self.model_storage_dir = Path(model_storage_dir)
        self._reader = ocr_reader
        self._yolo = yolo_model

    def _get_yolo(self) -> Any:
        if self._yolo is None:
            from ultralytics import YOLO

            if not self.weights_path.exists():
                raise FileNotFoundError(f"YOLO weights not found at {self.weights_path}")
            self._yolo = YOLO(str(self.weights_path))
        return self._yolo

    def _get_reader(self) -> Any:
        if self._reader is None:
            import easyocr

            self.model_storage_dir.mkdir(parents=True, exist_ok=True)
            user_net_dir = self.model_storage_dir / "user_network"
            user_net_dir.mkdir(parents=True, exist_ok=True)
            self._reader = easyocr.Reader(
                ["en"],
                gpu=False,
                model_storage_directory=str(self.model_storage_dir),
                user_network_directory=str(user_net_dir),
            )
        return self._reader

    def _crop_and_ocr(
        self,
        img: np.ndarray,
        box_xyxy: List[int],
        class_name: str,
    ) -> Tuple[str, float]:
        """Crop region with padding, optionally upscale, and run EasyOCR."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box_xyxy
        pad = self.crop_padding
        x1_pad = max(0, x1 - pad)
        y1_pad = max(0, y1 - pad)
        x2_pad = min(w, x2 + pad)
        y2_pad = min(h, y2 + pad)

        crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop.size == 0:
            return "", 0.0

        # For GSTIN crops, upscale 3x bicubic if low-res (box height < 25px or crop_h <= 32px) to avoid distortion on high-res pages
        if class_name == "gstin":
            crop_h = crop.shape[0]
            if crop_h <= 32:
                crop = cv2.resize(
                    crop,
                    (crop.shape[1] * 3, crop.shape[0] * 3),
                    interpolation=cv2.INTER_CUBIC,
                )

        reader = self._get_reader()
        ocr_results = reader.readtext(crop)
        if not ocr_results:
            return "", 0.0

        texts = [str(item[1]).strip() for item in ocr_results if item[1]]
        confs = [float(item[2]) for item in ocr_results]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return " ".join(texts), avg_conf

    def extract(
        self,
        image_input: Union[str, Path, np.ndarray],
    ) -> Dict[str, Any]:
        """Execute end-to-end extraction on a single image.

        Returns a structured dictionary with field extractions for:
          vendor, invoice_no, date, gstin, table, total
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Failed to load image from {img_path}")
            image_name = img_path.name
        else:
            img = image_input
            image_name = "in_memory_image"

        img_h, img_w = img.shape[:2]
        yolo = self._get_yolo()
        pred_res = yolo.predict(
            img,
            imgsz=self.imgsz,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False,
        )[0]

        # Organize detections by class
        detections_by_class: Dict[str, List[Tuple[float, List[int]]]] = {
            c: [] for c in TARGET_CLASSES
        }

        for box in pred_res.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = yolo.names[cls_id]
            if cls_name not in detections_by_class:
                continue
            conf = float(box.conf[0].item())
            xyxy = [int(round(coord)) for coord in box.xyxy[0].tolist()]
            detections_by_class[cls_name].append((conf, xyxy))

        # Sort each class detections descending by confidence
        for c in TARGET_CLASSES:
            detections_by_class[c].sort(key=lambda item: item[0], reverse=True)

        extracted_fields: Dict[str, Any] = {}

        # 1. Single highest-confidence fields: vendor, invoice_no, date, table, total
        for cname in ["vendor", "invoice_no", "date", "table", "total"]:
            dets = detections_by_class[cname]
            if not dets:
                extracted_fields[cname] = FieldExtraction(
                    value=None,
                    detector_confidence=0.0,
                    ocr_confidence=0.0,
                    validation_result=False,
                    combined_confidence=0.0,
                    needs_review=True,
                    source_box=None,
                ).to_dict()
                continue

            best_conf, best_box = dets[0]
            if cname == "table":
                # Skip table cell contents (keep the table box for downstream)
                extracted_fields[cname] = FieldExtraction(
                    value=None,
                    detector_confidence=best_conf,
                    ocr_confidence=1.0,
                    validation_result=True,
                    combined_confidence=best_conf,
                    needs_review=(best_conf < 0.60),
                    source_box=best_box,
                ).to_dict()
                continue

            raw_text, ocr_conf = self._crop_and_ocr(img, best_box, cname)
            val, val_res = parse_field_value(cname, raw_text)
            comb_conf = best_conf * ocr_conf
            needs_rev = (
                (val is None)
                or (comb_conf < 0.60)
                or (val_res is False)
                or (best_conf < self.high_conf_bar)
            )

            extracted_fields[cname] = FieldExtraction(
                value=val,
                detector_confidence=best_conf,
                ocr_confidence=ocr_conf,
                validation_result=val_res,
                combined_confidence=comb_conf,
                needs_review=needs_rev,
                source_box=best_box,
            ).to_dict()

        # 2. GSTIN: keep up to 3 boxes (to catch supplier + buyer)
        gstin_dets = detections_by_class["gstin"][:3]
        gstin_candidates: List[Dict[str, Any]] = []

        for g_conf, g_box in gstin_dets:
            raw_text, ocr_conf = self._crop_and_ocr(img, g_box, "gstin")
            val, val_res = parse_field_value("gstin", raw_text)
            comb_conf = g_conf * ocr_conf
            needs_rev = (
                (val is None)
                or (comb_conf < 0.60)
                or (val_res is False)
                or (g_conf < self.high_conf_bar)
            )
            field_data = FieldExtraction(
                value=val,
                detector_confidence=g_conf,
                ocr_confidence=ocr_conf,
                validation_result=val_res,
                combined_confidence=comb_conf,
                needs_review=needs_rev,
                source_box=g_box,
            ).to_dict()
            gstin_candidates.append(field_data)

        if gstin_candidates:
            primary_gstin = dict(gstin_candidates[0])
            primary_gstin["candidates"] = gstin_candidates
            extracted_fields["gstin"] = primary_gstin
            extracted_fields["gstins"] = gstin_candidates
        else:
            empty_gstin = FieldExtraction(
                value=None,
                detector_confidence=0.0,
                ocr_confidence=0.0,
                validation_result=False,
                combined_confidence=0.0,
                needs_review=True,
                source_box=None,
            ).to_dict()
            empty_gstin["candidates"] = []
            extracted_fields["gstin"] = empty_gstin
            extracted_fields["gstins"] = []

        return {
            "image": image_name,
            "width": img_w,
            "height": img_h,
            "fields": extracted_fields,
        }
