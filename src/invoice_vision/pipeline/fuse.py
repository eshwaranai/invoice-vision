"""Multi-modal candidate fusion and arithmetic validation for Indian invoice extraction.

Fuses candidates from:
  (a) YOLOv8 bounding boxes + crop-level EasyOCR (operating threshold conf >= 0.05)
  (b) v1 2D spatial rules on full-page EasyOCR
  (c) Arithmetic relations across candidate monetary amount tokens on the page

Rules:
  - total: Select amount satisfying taxable_value + taxes (+ round_off) = grand_total
    within 0.5 using subsets of up to 4 amount tokens; fallback to YOLO box, then to rules.
    Never invents digits or decimals.
  - invoice_no, date: YOLO value if detector confidence >= 0.25, else the rules value;
    if both exist and disagree, keep the rules value and set needs_review = True.
  - gstin: YOLO crop + checksum pipeline, plus any checksum-valid GSTINs discovered
    by rules not already predicted.
  - Every field extraction carries: source ("yolo", "rules", "arithmetic", "none"),
    confidence, and needs_review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import itertools
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dateutil import parser as date_parser

from invoice_vision.ocr.easyocr_engine import OCRItem
from invoice_vision.parse.spatial_rules import (
    ExtractedField,
    extract_spatial_fields,
    parse_amount,
    position_aware_gstin_correct,
    validate_gstin_checksum,
)
from invoice_vision.pipeline.extract import (
    parse_date_field,
    parse_field_value,
    parse_gstin_field,
    parse_invoice_no,
    parse_total_amount,
)


@dataclass
class FusedField:
    """Standardized output for an extracted field after candidate fusion."""

    value: Optional[str]
    source: str  # "yolo", "rules", "arithmetic", "none"
    confidence: float = 0.0
    needs_review: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "confidence": round(float(self.confidence), 4),
            "needs_review": self.needs_review,
            "details": self.details,
        }


def normalize_invoice_no(val: Optional[str]) -> str:
    """Normalize invoice number for semantic comparison (strip whitespace and symbols)."""
    if not val:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(val)).lower()


def dates_agree(d1: Optional[str], d2: Optional[str]) -> bool:
    """Check if two date strings resolve to the same calendar date."""
    if not d1 or not d2:
        return False
    # Exact normalized match
    if d1.strip().lower() == d2.strip().lower():
        return True
    try:
        dt1 = date_parser.parse(d1.strip(), dayfirst=True)
        dt2 = date_parser.parse(d2.strip(), dayfirst=True)
        return (dt1.year, dt1.month, dt1.day) == (dt2.year, dt2.month, dt2.day)
    except Exception:
        return False


def extract_amounts_from_tokens(
    tokens: List[Union[str, OCRItem, Dict[str, Any]]]
) -> List[Tuple[float, str]]:
    """Extract valid positive monetary amounts and their source text from tokens.

    Preserves duplicates across distinct tokens because multiple line items or
    taxes (e.g. CGST and SGST) regularly share identical values on invoices.
    """
    candidates: List[Tuple[float, str]] = []

    for item in tokens:
        if isinstance(item, str):
            txt = item
        elif isinstance(item, OCRItem):
            txt = item.text
        elif isinstance(item, dict):
            txt = str(item.get("text", ""))
        else:
            continue

        txt_clean = txt.strip()
        if not txt_clean:
            continue

        amt, is_uncertain = parse_amount(txt_clean)
        if amt is not None and amt >= 1.0:
            # Filter out obvious phone numbers / pin codes / dates unless structured as amounts
            str_clean = re.sub(r"[^\d]", "", txt_clean)
            if len(str_clean) == 6 and "." not in txt_clean and amt > 100000:
                # likely Indian PIN code (e.g. 110038, 711303)
                continue
            if len(str_clean) == 10 and "." not in txt_clean and amt > 6000000000:
                # likely mobile number
                continue

            # Format original string cleanly without inventing digits
            # Look for number substring that matches parsed float
            amt_int = int(round(amt))
            amt_str = f"{amt:.2f}"
            if f"{amt:.2f}" in txt_clean:
                val_str = f"{amt:.2f}"
            elif f"{amt_int}" in txt_clean:
                val_str = f"{amt_int}"
            else:
                m_digits = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?|\d+(?:\.\d+)?", txt_clean)
                val_str = m_digits[-1].replace(",", "").strip() if m_digits else f"{amt:.2f}"

            candidates.append((round(amt, 2), val_str))

    return candidates


def find_arithmetic_grand_total(
    amount_candidates: List[Tuple[float, str]],
    yolo_amount: Optional[float] = None,
    rules_amount: Optional[float] = None,
    tolerance: float = 0.5,
) -> Optional[Tuple[float, str, List[float], float]]:
    """Find the grand total satisfying taxable + taxes (+ round_off) = grand_total.

    Searches subsets of 2, 3, or 4 other amounts on the page whose sum matches
    a candidate grand total within tolerance.

    Returns:
        (grand_total_float, grand_total_str, combo_amounts, difference) or None.
    """
    if len(amount_candidates) < 3:
        return None

    # Sort candidates descending by float value
    all_floats = [c[0] for c in amount_candidates]
    str_map = {c[0]: c[1] for c in amount_candidates}
    unique_floats = sorted(list(set(all_floats)), reverse=True)

    verified_relations: List[Tuple[float, List[float], float]] = []

    for G in unique_floats:
        # Components must strictly be smaller than G
        idx = all_floats.index(G)
        remaining = all_floats[:idx] + all_floats[idx + 1 :]
        sub_components = [a for a in remaining if a < G]

        if len(sub_components) < 2:
            continue

        best_combo: Optional[Tuple[List[float], float]] = None
        min_diff = float("inf")

        for k in [2, 3, 4]:
            if len(sub_components) < k:
                continue
            for combo in itertools.combinations(sub_components, k):
                diff = abs(sum(combo) - G)
                if diff <= tolerance:
                    if diff < min_diff:
                        min_diff = diff
                        best_combo = (list(combo), round(diff, 4))
                    if diff <= 0.01:
                        break
            if best_combo and best_combo[1] <= 0.01:
                break

        if best_combo:
            verified_relations.append((G, best_combo[0], best_combo[1]))

    if not verified_relations:
        return None

    # Prioritization:
    # 1. Corroborated by YOLO or Rules candidate
    for G, combo, diff in verified_relations:
        if yolo_amount is not None and abs(G - yolo_amount) <= tolerance:
            return (G, str_map.get(G, f"{G:.2f}"), combo, diff)
        if rules_amount is not None and abs(G - rules_amount) <= tolerance:
            return (G, str_map.get(G, f"{G:.2f}"), combo, diff)

    # 2. Maximum grand total satisfying arithmetic
    # Filter out unreasonable outliers (> 100,000,000)
    reasonable = [r for r in verified_relations if r[0] < 100_000_000.0]
    if reasonable:
        best_r = reasonable[0]
        return (best_r[0], str_map.get(best_r[0], f"{best_r[0]:.2f}"), best_r[1], best_r[2])

    best_r = verified_relations[0]
    return (best_r[0], str_map.get(best_r[0], f"{best_r[0]:.2f}"), best_r[1], best_r[2])


class InvoiceFuser:
    """Fuses detection crops, spatial rules, and page-level arithmetic validation."""

    def __init__(
        self,
        detector_high_conf: float = 0.25,
        arithmetic_tolerance: float = 0.5,
    ) -> None:
        self.detector_high_conf = detector_high_conf
        self.arithmetic_tolerance = arithmetic_tolerance

    def fuse_total(
        self,
        yolo_val: Optional[str],
        yolo_det_conf: float,
        yolo_ocr_conf: float,
        yolo_needs_review: bool,
        rules_val: Optional[str],
        rules_conf: float,
        rules_needs_review: bool,
        amount_candidates: List[Tuple[float, str]],
    ) -> FusedField:
        """Fuse total amount using arithmetic validation with YOLO/rules fallbacks."""
        yolo_num: Optional[float] = None
        if yolo_val:
            m = re.search(r"\d+(?:,\d+)*(?:\.\d+)?|\d+(?:\.\d+)?", str(yolo_val).replace(",", ""))
            if m:
                try:
                    yolo_num = float(m.group(0))
                except ValueError:
                    pass

        rules_num: Optional[float] = None
        if rules_val:
            m = re.search(r"\d+(?:,\d+)*(?:\.\d+)?|\d+(?:\.\d+)?", str(rules_val).replace(",", ""))
            if m:
                try:
                    rules_num = float(m.group(0))
                except ValueError:
                    pass

        # 1. Arithmetic validation on page amounts
        arith_result = find_arithmetic_grand_total(
            amount_candidates=amount_candidates,
            yolo_amount=yolo_num,
            rules_amount=rules_num,
            tolerance=self.arithmetic_tolerance,
        )

        if arith_result is not None:
            g_float, g_str, combo, diff = arith_result
            return FusedField(
                value=g_str,
                source="arithmetic",
                confidence=0.95,
                needs_review=False,
                details={
                    "arithmetic_validated": True,
                    "combo_components": combo,
                    "error": diff,
                    "corroborated_by_yolo": (yolo_num is not None and abs(g_float - yolo_num) <= self.arithmetic_tolerance),
                    "corroborated_by_rules": (rules_num is not None and abs(g_float - rules_num) <= self.arithmetic_tolerance),
                },
            )

        # 2. Fallback to YOLO box value
        if yolo_val and yolo_num is not None:
            # Auto-accept if detector confidence is high and OCR confidence is acceptable
            auto_accept = (yolo_det_conf >= self.detector_high_conf) and not yolo_needs_review
            return FusedField(
                value=yolo_val,
                source="yolo",
                confidence=yolo_det_conf * yolo_ocr_conf if yolo_ocr_conf > 0 else yolo_det_conf,
                needs_review=not auto_accept,
                details={
                    "detector_confidence": yolo_det_conf,
                    "ocr_confidence": yolo_ocr_conf,
                    "arithmetic_validated": False,
                },
            )

        # 3. Fallback to spatial rules value
        if rules_val and rules_num is not None:
            return FusedField(
                value=rules_val,
                source="rules",
                confidence=rules_conf,
                needs_review=rules_needs_review,
                details={
                    "rules_confidence": rules_conf,
                    "arithmetic_validated": False,
                },
            )

        # 4. No candidate found
        return FusedField(
            value=None,
            source="none",
            confidence=0.0,
            needs_review=True,
            details={"arithmetic_validated": False},
        )

    def fuse_key_value(
        self,
        field_name: str,
        yolo_val: Optional[str],
        yolo_det_conf: float,
        yolo_ocr_conf: float,
        yolo_needs_review: bool,
        rules_val: Optional[str],
        rules_conf: float,
        rules_needs_review: bool,
    ) -> FusedField:
        """Fuse invoice_no or date field based on confidence threshold and agreement."""
        y_has = bool(yolo_val and str(yolo_val).strip())
        r_has = bool(rules_val and str(rules_val).strip())

        # Check agreement
        is_agree = False
        if y_has and r_has:
            if field_name == "invoice_no":
                norm_y = normalize_invoice_no(yolo_val)
                norm_r = normalize_invoice_no(rules_val)
                is_agree = bool(norm_y and norm_r and (norm_y == norm_r or norm_y in norm_r or norm_r in norm_y))
            elif field_name == "date":
                is_agree = dates_agree(yolo_val, rules_val)

        # Rule: "YOLO value if detector confidence >= 0.25, else the rules value;
        #        if both exist and disagree keep the rules value and set needs_review = true."
        if y_has and yolo_det_conf >= self.detector_high_conf:
            if r_has:
                if is_agree:
                    # High confidence YOLO confirmed by spatial rules
                    return FusedField(
                        value=yolo_val,
                        source="yolo",
                        confidence=max(yolo_det_conf, rules_conf),
                        needs_review=False,
                        details={"agreement": True, "detector_confidence": yolo_det_conf},
                    )
                else:
                    # Disagreement: keep the rules value and set needs_review = True
                    return FusedField(
                        value=rules_val,
                        source="rules",
                        confidence=rules_conf,
                        needs_review=True,
                        details={
                            "agreement": False,
                            "disagree_with_yolo": yolo_val,
                            "detector_confidence": yolo_det_conf,
                        },
                    )
            else:
                # High confidence YOLO with no rules candidate
                return FusedField(
                    value=yolo_val,
                    source="yolo",
                    confidence=yolo_det_conf,
                    needs_review=False,
                    details={"agreement": None, "detector_confidence": yolo_det_conf},
                )
        else:
            # Low detector confidence (< 0.25) or no YOLO value
            if r_has:
                if y_has:
                    if is_agree:
                        # Even though detector confidence was low, rules and YOLO agree!
                        return FusedField(
                            value=rules_val,
                            source="rules",
                            confidence=max(0.80, rules_conf),
                            needs_review=False,
                            details={"agreement": True, "detector_confidence": yolo_det_conf},
                        )
                    else:
                        # Disagreement: keep rules value, flag for review
                        return FusedField(
                            value=rules_val,
                            source="rules",
                            confidence=rules_conf,
                            needs_review=True,
                            details={
                                "agreement": False,
                                "disagree_with_yolo": yolo_val,
                                "detector_confidence": yolo_det_conf,
                            },
                        )
                else:
                    # Only rules value exists
                    return FusedField(
                        value=rules_val,
                        source="rules",
                        confidence=rules_conf,
                        needs_review=rules_needs_review,
                        details={"agreement": None, "detector_confidence": 0.0},
                    )
            elif y_has:
                # Only YOLO value exists, but detector confidence < 0.25: fallback with review
                return FusedField(
                    value=yolo_val,
                    source="yolo",
                    confidence=yolo_det_conf,
                    needs_review=True,
                    details={"agreement": None, "detector_confidence": yolo_det_conf},
                )
            else:
                # Neither found
                return FusedField(
                    value=None,
                    source="none",
                    confidence=0.0,
                    needs_review=True,
                    details={"agreement": None},
                )

    def fuse_gstin(
        self,
        yolo_gstin_field: Dict[str, Any],
        yolo_all_candidates: List[Dict[str, Any]],
        rules_gstin_val: Optional[str],
        page_tokens: List[Union[str, OCRItem, Dict[str, Any]]],
    ) -> FusedField:
        """Fuse GSTIN: YOLO crop + checksum pipeline, plus any valid GSTINs found by rules."""
        predicted_gstins: List[str] = []
        gstin_entries: List[Dict[str, Any]] = []

        # 1. Add YOLO candidates that passed parsing
        for cand in yolo_all_candidates:
            v = cand.get("value")
            if v and v not in predicted_gstins:
                predicted_gstins.append(v)
                gstin_entries.append({
                    "value": v,
                    "source": "yolo",
                    "valid_checksum": bool(cand.get("validation_result", False)),
                    "confidence": cand.get("detector_confidence", 0.0),
                    "needs_review": cand.get("needs_review", True),
                })

        # 2. Add spatial rules GSTIN if checksum valid and not already present
        if rules_gstin_val and len(rules_gstin_val) == 15:
            corr = position_aware_gstin_correct(rules_gstin_val)
            if validate_gstin_checksum(corr) and corr not in predicted_gstins:
                predicted_gstins.append(corr)
                gstin_entries.append({
                    "value": corr,
                    "source": "rules",
                    "valid_checksum": True,
                    "confidence": 0.85,
                    "needs_review": False,
                })

        # 3. Scan full page tokens for any additional checksum-valid GSTINs (e.g. buyer GSTIN)
        for tok in page_tokens:
            txt = tok.text if isinstance(tok, OCRItem) else (tok.get("text", "") if isinstance(tok, dict) else str(tok))
            # Split words by delimiters to cleanly isolate 15-character tokens
            words = re.split(r"[\s\-_:/\,\;\(\)\[\]\{\}\<\>\|\\]+", txt)
            for cand15 in words:
                clean_cand = re.sub(r"[^\w]", "", cand15).strip()
                if len(clean_cand) == 15:
                    corr = position_aware_gstin_correct(clean_cand)
                    if validate_gstin_checksum(corr) and corr not in predicted_gstins:
                        predicted_gstins.append(corr)
                        gstin_entries.append({
                            "value": corr,
                            "source": "rules",
                            "valid_checksum": True,
                            "confidence": 0.85,
                            "needs_review": False,
                        })

        # Determine primary GSTIN
        if not gstin_entries:
            return FusedField(
                value=None,
                source="none",
                confidence=0.0,
                needs_review=True,
                details={"all_gstins": []},
            )

        # Prioritize checksum-valid entry
        valid_entries = [e for e in gstin_entries if e["valid_checksum"]]
        primary = valid_entries[0] if valid_entries else gstin_entries[0]

        return FusedField(
            value=primary["value"],
            source=primary["source"],
            confidence=primary["confidence"],
            needs_review=primary["needs_review"],
            details={"all_gstins": [e["value"] for e in gstin_entries], "candidates": gstin_entries},
        )

    def fuse(
        self,
        yolo_result: Dict[str, Any],
        spatial_rules_result: Dict[str, ExtractedField],
        ocr_items: List[OCRItem],
        page_text_tokens: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end multi-modal fusion on a single invoice."""
        fields = yolo_result.get("fields", {})
        y_inv = fields.get("invoice_no", {})
        y_dt = fields.get("date", {})
        y_tot = fields.get("total", {})
        y_gst = fields.get("gstin", {})
        y_vend = fields.get("vendor", {})
        y_gstins = fields.get("gstins", y_gst.get("candidates", []))

        r_inv = spatial_rules_result.get("invoice_number", ExtractedField(None))
        r_dt = spatial_rules_result.get("purchase_date", ExtractedField(None))
        r_tot = spatial_rules_result.get("total_amount", ExtractedField(None))
        r_gst = spatial_rules_result.get("supplier_gst", ExtractedField(None))

        # Extract amount candidates from page tokens
        tokens_for_amounts: List[Any] = list(ocr_items)
        if page_text_tokens:
            tokens_for_amounts.extend(page_text_tokens)

        amount_candidates = extract_amounts_from_tokens(tokens_for_amounts)

        # 1. Total (Arithmetic validation + fallbacks)
        fused_total = self.fuse_total(
            yolo_val=y_tot.get("value"),
            yolo_det_conf=y_tot.get("detector_confidence", 0.0),
            yolo_ocr_conf=y_tot.get("ocr_confidence", 0.0),
            yolo_needs_review=y_tot.get("needs_review", True),
            rules_val=r_tot.value,
            rules_conf=r_tot.confidence,
            rules_needs_review=r_tot.needs_review,
            amount_candidates=amount_candidates,
        )

        # 2. Invoice Number
        fused_inv = self.fuse_key_value(
            field_name="invoice_no",
            yolo_val=y_inv.get("value"),
            yolo_det_conf=y_inv.get("detector_confidence", 0.0),
            yolo_ocr_conf=y_inv.get("ocr_confidence", 0.0),
            yolo_needs_review=y_inv.get("needs_review", True),
            rules_val=r_inv.value,
            rules_conf=r_inv.confidence,
            rules_needs_review=r_inv.needs_review,
        )

        # 3. Purchase Date
        fused_date = self.fuse_key_value(
            field_name="date",
            yolo_val=y_dt.get("value"),
            yolo_det_conf=y_dt.get("detector_confidence", 0.0),
            yolo_ocr_conf=y_dt.get("ocr_confidence", 0.0),
            yolo_needs_review=y_dt.get("needs_review", True),
            rules_val=r_dt.value,
            rules_conf=r_dt.confidence,
            rules_needs_review=r_dt.needs_review,
        )

        # 4. GSTIN
        tokens_for_gstin = list(ocr_items)
        if page_text_tokens:
            tokens_for_gstin.extend(page_text_tokens)

        fused_gstin = self.fuse_gstin(
            yolo_gstin_field=y_gst,
            yolo_all_candidates=y_gstins,
            rules_gstin_val=r_gst.value,
            page_tokens=tokens_for_gstin,
        )

        # 5. Vendor (Evaluated by detection localization recall only)
        fused_vendor = FusedField(
            value=y_vend.get("value"),
            source="yolo" if y_vend.get("value") else "none",
            confidence=y_vend.get("detector_confidence", 0.0),
            needs_review=y_vend.get("needs_review", True),
            details={"detector_confidence": y_vend.get("detector_confidence", 0.0)},
        )

        return {
            "image": yolo_result.get("image", ""),
            "width": yolo_result.get("width", 0),
            "height": yolo_result.get("height", 0),
            "fields": {
                "vendor": fused_vendor.to_dict(),
                "invoice_no": fused_inv.to_dict(),
                "date": fused_date.to_dict(),
                "gstin": fused_gstin.to_dict(),
                "total": fused_total.to_dict(),
            },
        }
