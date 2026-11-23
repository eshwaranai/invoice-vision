"""Evaluation of Multi-Modal Fusion and Leave-One-Family-Out (LOFO) on TRAIN and VAL families.

Guarantees:
  - ZERO test set leakage: Test split and test families are strictly untouched.
  - Compares Fused Pipeline vs YOLO-only on the 12 VAL images.
  - Executes LOFO using rules-only and arithmetic-only candidates across all 12 TRAIN and VAL families.
  - Strict privacy: for own bills, prints no extracted personal text, only counts, flags, and metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from dateutil import parser as date_parser

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from invoice_vision.ocr.easyocr_engine import EasyOCREngine, OCRItem
from invoice_vision.parse.spatial_rules import (
    ExtractedField,
    extract_spatial_fields,
    parse_amount,
    position_aware_gstin_correct,
    validate_gstin_checksum,
)
from invoice_vision.pipeline.extract import InvoiceExtractor
from invoice_vision.pipeline.fuse import (
    FusedField,
    InvoiceFuser,
    dates_agree,
    extract_amounts_from_tokens,
    find_arithmetic_grand_total,
    normalize_invoice_no,
)


def load_all_ground_truth(
    data_root: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    """Load ground truth, family mapping, and split mapping for all train and val images."""
    splits_path = data_root / "splits_v3.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    train_imgs = set(splits_data.get("combined_clean_splits", {}).get("train", []))
    val_imgs = set(splits_data.get("combined_clean_splits", {}).get("val", []))
    allowed_imgs = train_imgs.union(val_imgs)

    img_to_split: Dict[str, str] = {}
    for img in train_imgs:
        img_to_split[img] = "train"
    for img in val_imgs:
        img_to_split[img] = "val"

    # 1. Family mapping from vendor_groups.yaml, family_map.csv, groups.json, gst_synth_manifest.csv
    file_to_fam: Dict[str, str] = {}
    vg_path = data_root / "vendor_groups.yaml"
    with open(vg_path, "r", encoding="utf-8") as f:
        vg_data = yaml.safe_load(f).get("groups", {})
        for gname, ginfo in vg_data.items():
            if "pages" in ginfo:
                for p in ginfo["pages"]:
                    file_to_fam[p] = gname

    fm_path = data_root / "family_map.csv"
    if fm_path.exists():
        with open(fm_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                file_to_fam[r["filename"]] = r["split_group"]

    gj_path = data_root / "groups.json"
    if gj_path.exists():
        with open(gj_path, "r", encoding="utf-8") as f:
            gj_data = json.load(f).get("groups", {})
            for vname, fnames in gj_data.items():
                for gname, ginfo in vg_data.items():
                    for tf in ginfo.get("template_families", []):
                        if tf.lower() in vname.lower():
                            for fn in fnames:
                                file_to_fam[fn] = gname

    synth_path = data_root / "gst_synth_manifest.csv"
    synth_gt: Dict[str, Dict[str, Any]] = {}
    if synth_path.exists():
        with open(synth_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                fn = r["filename"]
                file_to_fam[fn] = f"gstsyn_{r['layout']}"
                synth_gt[fn] = {
                    "invoice_no": str(r["invoice_number"]).strip(),
                    "date": str(r["invoice_date"]).strip(),
                    "vendor": str(r["supplier_name"]).strip(),
                    "gstin": str(r["supplier_gstin"]).strip(),
                    "total": str(r["grand_total"]).strip(),
                    "all_gstins": [str(r["supplier_gstin"]).strip(), str(r.get("buyer_gstin", "")).strip()],
                }

    # 2. Public invoices GT
    public_gt_path = data_root / "ground_truth_gst.csv"
    public_df = pd.read_csv(public_gt_path) if public_gt_path.exists() else pd.DataFrame()
    public_gt: Dict[str, Dict[str, Any]] = {}
    for _, r in public_df.iterrows():
        fn = str(r["filename"]).strip()
        all_gsts = []
        if pd.notna(r.get("supplier_gst")) and str(r["supplier_gst"]).strip():
            all_gsts.append(str(r["supplier_gst"]).strip())
        if pd.notna(r.get("customer_gst")) and str(r["customer_gst"]).strip():
            all_gsts.append(str(r["customer_gst"]).strip())

        public_gt[fn] = {
            "invoice_no": str(r["invoice_number"]).strip() if pd.notna(r.get("invoice_number")) else None,
            "date": str(r["purchase_date"]).strip() if pd.notna(r.get("purchase_date")) else None,
            "vendor": str(r["supplier_name"]).strip() if pd.notna(r.get("supplier_name")) else None,
            "gstin": all_gsts[0] if all_gsts else None,
            "total": str(r["total_amount"]).strip() if pd.notna(r.get("total_amount")) else None,
            "all_gstins": all_gsts,
        }

    # 3. Assemble unified GT for all train and val images
    unified_gt: Dict[str, Dict[str, Any]] = {}
    class_map = {0: "vendor", 1: "invoice_no", 2: "date", 3: "gstin", 4: "table", 5: "total"}

    for fn in allowed_imgs:
        stem = Path(fn).stem
        if fn.startswith("gst_"):
            unified_gt[fn] = public_gt.get(fn, {})
        elif fn.startswith("gstsyn_"):
            unified_gt[fn] = synth_gt.get(fn, {})
        else:
            # Own bills: precise extraction from PDF text + human annotation boxes
            pdf_path = data_root / "processed" / "pdf_text" / f"{stem}.json"
            # Check labels in val or train
            lbl_path = data_root / "yolo_v2" / "labels" / img_to_split[fn] / f"{stem}.txt"
            if not pdf_path.exists() or not lbl_path.exists():
                unified_gt[fn] = {}
                continue

            with open(pdf_path, "r", encoding="utf-8") as f:
                pdf_data = json.load(f)
            w_px, h_px = pdf_data["width_px"], pdf_data["height_px"]
            words = pdf_data.get("words", [])

            gt_f: Dict[str, Any] = {}
            all_gst: List[str] = []

            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                cname = class_map.get(cid)
                if not cname or cname == "table":
                    continue
                xc, yc, bw, bh = map(float, parts[1:5])
                bx0 = (xc - bw / 2.0) * w_px
                by0 = (yc - bh / 2.0) * h_px
                bx1 = (xc + bw / 2.0) * w_px
                by1 = (yc + bh / 2.0) * h_px

                in_words = [
                    w["text"]
                    for w in words
                    if bx0 <= (w["bbox"][0] + w["bbox"][2]) / 2.0 <= bx1
                    and by0 <= (w["bbox"][1] + w["bbox"][3]) / 2.0 <= by1
                ]
                raw_val = " ".join(in_words).strip()

                if cname == "vendor":
                    gt_f["vendor"] = raw_val
                elif cname == "invoice_no":
                    gt_f["invoice_no"] = raw_val.strip(":-# \t")
                elif cname == "date":
                    m_d = re.search(r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}", raw_val)
                    gt_f["date"] = m_d.group(0) if m_d else raw_val
                elif cname == "gstin":
                    clean_g = raw_val.strip(":-# \t")
                    if len(clean_g) >= 15:
                        cand15 = clean_g[:15]
                        all_gst.append(cand15)
                        gt_f["gstin"] = cand15
                elif cname == "total":
                    m_tot = re.search(r"\d+(?:,\d+)*(?:\.\d+)?|\d+(?:\.\d+)?", raw_val.replace("INR", "").replace("Rs.", ""))
                    gt_f["total"] = m_tot.group(0) if m_tot else raw_val

            gt_f["all_gstins"] = all_gst
            unified_gt[fn] = gt_f

    return unified_gt, file_to_fam, img_to_split


def check_match_lenient(field: str, pred: Optional[str], gt: Optional[str]) -> bool:
    """Check lenient match according to protocol rules."""
    if not pred or not gt:
        return False

    if field == "invoice_no":
        ny = normalize_invoice_no(pred)
        ng = normalize_invoice_no(gt)
        return bool(ny and ng and (ny == ng or ny in ng or ng in ny))

    elif field == "date":
        return dates_agree(pred, gt)

    elif field == "total":
        try:
            m_p = re.search(r"\d+(?:,\d+)*(?:\.\d+)?|\d+(?:\.\d+)?", str(pred).replace(",", ""))
            m_g = re.search(r"\d+(?:,\d+)*(?:\.\d+)?|\d+(?:\.\d+)?", str(gt).replace(",", ""))
            if m_p and m_g:
                fp = float(m_p.group(0))
                fg = float(m_g.group(0))
                return abs(fp - fg) <= 0.01
        except Exception:
            pass
        return False

    elif field == "gstin":
        corr_p = position_aware_gstin_correct(str(pred).strip())
        corr_g = position_aware_gstin_correct(str(gt).strip())
        return corr_p == corr_g

    return False


def run_val_fusion_evaluation(
    data_root: Path,
    extractor: InvoiceExtractor,
    fuser: InvoiceFuser,
    gt_dict: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run full evaluation on the 12 VAL images comparing YOLO-only vs Fused Pipeline."""
    splits_path = data_root / "splits_v3.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        val_images = json.load(f)["combined_clean_splits"]["val"]

    yolo_matches = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}
    fused_matches_ignore_flag = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}
    fused_matches_auto_accept = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}
    fused_flagged_counts = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}
    field_counts = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}

    distinct_invoices = 9  # 3 public Amazon + 6 own own_family_A (3 are continuation pages)
    per_image_results = []

    for fn in val_images:
        gt = gt_dict.get(fn, {})
        is_public = fn.startswith("gst_")
        img_path = data_root / "yolo_v2" / "images" / "val" / fn
        stem = Path(fn).stem

        # 1. YOLO extraction
        yolo_res = extractor.extract(img_path)
        y_fields = yolo_res.get("fields", {})

        # 2. Spatial rules extraction
        if is_public:
            cp = data_root / "processed" / "ocr_cache" / f"{fn}.json"
            ocr_data = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else []
            items = [OCRItem.from_dict(d) for d in ocr_data]
            page_words = []
        else:
            pdf_p = data_root / "processed" / "pdf_text" / f"{stem}.json"
            pdf_data = json.loads(pdf_p.read_text(encoding="utf-8")) if pdf_p.exists() else {}
            items = []
            page_words = pdf_data.get("words", [])
            for w in page_words:
                bx = w.get("bbox", [0, 0, 0, 0])
                box = [[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]]]
                items.append(OCRItem(box=box, text=w["text"], confidence=1.0))

        rules_res = extract_spatial_fields(items)

        # 3. Multi-modal Fusion
        fused_res = fuser.fuse(
            yolo_result=yolo_res,
            spatial_rules_result=rules_res,
            ocr_items=items,
            page_text_tokens=page_words,
        )
        f_fields = fused_res.get("fields", {})

        img_report: Dict[str, Any] = {"filename": fn, "fields": {}}

        for fld in ["invoice_no", "date", "total", "gstin"]:
            gt_val = gt.get(fld)
            if not gt_val:
                continue

            field_counts[fld] += 1

            # YOLO evaluation
            y_val = y_fields.get(fld, {}).get("value")
            y_flag = y_fields.get(fld, {}).get("needs_review", True)
            y_match = check_match_lenient(fld, y_val, gt_val)
            if y_match:
                yolo_matches[fld] += 1

            # Fused evaluation
            f_item = f_fields.get(fld, {})
            f_val = f_item.get("value")
            f_flag = f_item.get("needs_review", True)
            f_src = f_item.get("source", "none")

            if f_flag:
                fused_flagged_counts[fld] += 1

            # Check primary match or set match for GSTIN
            if fld == "gstin":
                all_pred = f_item.get("details", {}).get("all_gstins", [f_val] if f_val else [])
                all_gt = gt.get("all_gstins", [gt_val])
                f_match = any(check_match_lenient("gstin", p, g) for p in all_pred for g in all_gt)
            else:
                f_match = check_match_lenient(fld, f_val, gt_val)

            if f_match:
                fused_matches_ignore_flag[fld] += 1
                if not f_flag:
                    fused_matches_auto_accept[fld] += 1

            img_report["fields"][fld] = {
                "gt_present": True,
                "yolo_match": y_match,
                "fused_match": f_match,
                "fused_source": f_src,
                "fused_flagged": f_flag,
            }

        per_image_results.append(img_report)

    # Compile table
    summary_table = []
    for fld in ["invoice_no", "date", "total", "gstin"]:
        n = field_counts[fld]
        y_cnt = yolo_matches[fld]
        f_raw_cnt = fused_matches_ignore_flag[fld]
        f_auto_cnt = fused_matches_auto_accept[fld]
        f_flag_cnt = fused_flagged_counts[fld]

        summary_table.append({
            "field": fld,
            "n": n,
            "yolo_matches": y_cnt,
            "yolo_acc": f"{y_cnt / n * 100:.1f}% ({y_cnt}/{n})" if n > 0 else "N/A",
            "fused_raw_matches": f_raw_cnt,
            "fused_raw_acc": f"{f_raw_cnt / n * 100:.1f}% ({f_raw_cnt}/{n})" if n > 0 else "N/A",
            "fused_auto_matches": f_auto_cnt,
            "fused_auto_acc": f"{f_auto_cnt / n * 100:.1f}% ({f_auto_cnt}/{n})" if n > 0 else "N/A",
            "fused_flagged_share": f"{f_flag_cnt / n * 100:.1f}% ({f_flag_cnt}/{n})" if n > 0 else "N/A",
            "delta_matches": f_raw_cnt - y_cnt,
        })

    return {
        "distinct_invoices": distinct_invoices,
        "summary_table": summary_table,
        "per_image": per_image_results,
    }


def run_lofo_evaluation(
    data_root: Path,
    fuser: InvoiceFuser,
    unified_gt: Dict[str, Dict[str, Any]],
    file_to_fam: Dict[str, str],
    img_to_split: Dict[str, str],
) -> Dict[str, Any]:
    """Execute Leave-One-Family-Out (LOFO) evaluation over TRAIN and VAL families.

    Per prompt specification:
      Uses rules-only and arithmetic-only candidates (since YOLO was trained on train families).
      TEST images and families are strictly excluded.
    """
    allowed_images = [fn for fn, sp in img_to_split.items() if sp in ["train", "val"]]
    families = sorted(list(set(file_to_fam[fn] for fn in allowed_images)))

    fam_results = {}
    total_invoices_evaluated = 0

    overall_counts = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}
    overall_raw_matches = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}
    overall_auto_matches = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}
    overall_flagged = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}
    field_families_present = {"invoice_no": set(), "date": set(), "total": set(), "gstin": set()}

    for held_out_fam in families:
        fam_imgs = [fn for fn in allowed_images if file_to_fam.get(fn) == held_out_fam]
        fam_field_counts = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}
        fam_raw_matches = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}
        fam_auto_matches = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}
        fam_flagged = {"invoice_no": 0, "date": 0, "total": 0, "gstin": 0}

        for fn in fam_imgs:
            gt = unified_gt.get(fn, {})
            stem = Path(fn).stem
            is_public = fn.startswith("gst_")
            is_synth = fn.startswith("gstsyn_")

            # Load OCR items & page words
            if is_public or is_synth:
                cp = data_root / "processed" / "ocr_cache" / f"{fn}.json"
                ocr_data = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else []
                items = [OCRItem.from_dict(d) for d in ocr_data]
                page_words = []
            else:
                pdf_p = data_root / "processed" / "pdf_text" / f"{stem}.json"
                pdf_data = json.loads(pdf_p.read_text(encoding="utf-8")) if pdf_p.exists() else {}
                page_words = pdf_data.get("words", [])
                items = []
                for w in page_words:
                    bx = w.get("bbox", [0, 0, 0, 0])
                    box = [[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]]]
                    items.append(OCRItem(box=box, text=w["text"], confidence=1.0))

            # Run deterministic spatial rules
            rules_res = extract_spatial_fields(items)

            # Run deterministic arithmetic validation
            tokens_for_amts: List[Any] = list(items)
            if page_words:
                tokens_for_amts.extend(page_words)
            amt_cands = extract_amounts_from_tokens(tokens_for_amts)

            # LOFO rules-only & arithmetic-only candidates:
            # total: arithmetic with spatial rules fallback
            r_tot = rules_res.get("total_amount", ExtractedField(None))
            fused_tot = fuser.fuse_total(
                yolo_val=None,
                yolo_det_conf=0.0,
                yolo_ocr_conf=0.0,
                yolo_needs_review=True,
                rules_val=r_tot.value,
                rules_conf=r_tot.confidence,
                rules_needs_review=r_tot.needs_review,
                amount_candidates=amt_cands,
            )

            # invoice_no: spatial rules candidate
            r_inv = rules_res.get("invoice_number", ExtractedField(None))
            fused_inv = fuser.fuse_key_value(
                field_name="invoice_no",
                yolo_val=None,
                yolo_det_conf=0.0,
                yolo_ocr_conf=0.0,
                yolo_needs_review=True,
                rules_val=r_inv.value,
                rules_conf=r_inv.confidence,
                rules_needs_review=r_inv.needs_review,
            )

            # date: spatial rules candidate
            r_dt = rules_res.get("purchase_date", ExtractedField(None))
            fused_dt = fuser.fuse_key_value(
                field_name="date",
                yolo_val=None,
                yolo_det_conf=0.0,
                yolo_ocr_conf=0.0,
                yolo_needs_review=True,
                rules_val=r_dt.value,
                rules_conf=r_dt.confidence,
                rules_needs_review=r_dt.needs_review,
            )

            # gstin: spatial rules candidate + checksum valid page discoveries
            r_gst = rules_res.get("supplier_gst", ExtractedField(None))
            fused_gst = fuser.fuse_gstin(
                yolo_gstin_field={},
                yolo_all_candidates=[],
                rules_gstin_val=r_gst.value,
                page_tokens=tokens_for_amts,
            )

            predictions = {
                "invoice_no": fused_inv,
                "date": fused_dt,
                "total": fused_tot,
                "gstin": fused_gst,
            }

            for fld in ["invoice_no", "date", "total", "gstin"]:
                gt_val = gt.get(fld)
                if not gt_val:
                    continue

                fam_field_counts[fld] += 1
                overall_counts[fld] += 1
                field_families_present[fld].add(held_out_fam)

                pred_field = predictions[fld]
                p_val = pred_field.value
                p_flag = pred_field.needs_review

                if p_flag:
                    fam_flagged[fld] += 1
                    overall_flagged[fld] += 1

                if fld == "gstin":
                    all_pred = pred_field.details.get("all_gstins", [p_val] if p_val else [])
                    all_gt = gt.get("all_gstins", [gt_val])
                    m_ok = any(check_match_lenient("gstin", p, g) for p in all_pred for g in all_gt)
                else:
                    m_ok = check_match_lenient(fld, p_val, gt_val)

                if m_ok:
                    fam_raw_matches[fld] += 1
                    overall_raw_matches[fld] += 1
                    if not p_flag:
                        fam_auto_matches[fld] += 1
                        overall_auto_matches[fld] += 1

        fam_results[held_out_fam] = {
            "image_count": len(fam_imgs),
            "counts": fam_field_counts,
            "raw_matches": fam_raw_matches,
            "auto_matches": fam_auto_matches,
            "flagged": fam_flagged,
        }

    # Aggregate metrics per field
    aggregate_table = []
    for fld in ["invoice_no", "date", "total", "gstin"]:
        n = overall_counts[fld]
        raw_m = overall_raw_matches[fld]
        auto_m = overall_auto_matches[fld]
        flg_m = overall_flagged[fld]
        n_fams = len(field_families_present[fld])

        raw_acc_str = f"{raw_m / n * 100:.1f}% ({raw_m}/{n})" if n > 0 else "N/A"
        auto_acc_str = f"{auto_m / n * 100:.1f}% ({auto_m}/{n})" if n > 0 else "N/A"
        flag_share_str = f"{flg_m / n * 100:.1f}% ({flg_m}/{n})" if n > 0 else "N/A"
        flag_caveat = "(too few to measure)" if n < 15 else "Robust (n >= 15)"

        aggregate_table.append({
            "field": fld,
            "n": n,
            "n_families": n_fams,
            "sample_flag": flag_caveat,
            "raw_accuracy": raw_acc_str,
            "auto_accepted_accuracy": auto_acc_str,
            "review_routing_share": flag_share_str,
        })

    return {
        "families_evaluated": len(families),
        "total_images": len(allowed_images),
        "aggregate_table": aggregate_table,
        "per_family": fam_results,
    }


def main() -> None:
    data_root = ROOT_DIR / "data"
    print("Loading Ground Truth, Family Mapping, and Splits...")
    unified_gt, file_to_fam, img_to_split = load_all_ground_truth(data_root)

    extractor = InvoiceExtractor(
        weights_path=data_root / "models" / "layouts_s0_best.pt",
        conf_thresh=0.05,
        high_conf_bar=0.25,
    )
    fuser = InvoiceFuser(detector_high_conf=0.25, arithmetic_tolerance=0.5)

    print("\n=======================================================")
    print("1. EVALUATION ON THE 12 VAL IMAGES (YOLO vs FUSION)")
    print("=======================================================")
    val_res = run_val_fusion_evaluation(data_root, extractor, fuser, unified_gt)
    print(f"Distinct Invoices Evaluated: {val_res['distinct_invoices']}")
    print("| Field | n | YOLO Acc (Setting a) | Fused Raw Acc | Fused Auto-Accepted Acc | Review Routed Share | Delta Invoices |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in val_res["summary_table"]:
        print(f"| `{row['field']}` | {row['n']} | {row['yolo_acc']} | {row['fused_raw_acc']} | {row['fused_auto_acc']} | {row['fused_flagged_share']} | **+{row['delta_matches']}** |")

    # Check pre-declared decision rule
    total_delta = next(r["delta_matches"] for r in val_res["summary_table"] if r["field"] == "total")
    other_deltas = [r["delta_matches"] for r in val_res["summary_table"] if r["field"] != "total"]
    rule_passed = (total_delta >= 3) and all(d >= -1 for d in other_deltas)

    print("\n--- PRE-DECLARED DECISION RULE VERIFICATION ---")
    print(f"Rule: Total accuracy rises by >= 3 invoices AND no other field drops by > 1 invoice.")
    print(f"Observed: Total Delta = +{total_delta} invoices | Other field Deltas = {other_deltas}")
    print(f"Pre-declared Decision Rule Status: {'PASSED' if rule_passed else 'FAILED'}")

    print("\n=======================================================")
    print("2. LEAVE-ONE-FAMILY-OUT (LOFO) ON TRAIN & VAL FAMILIES")
    print("=======================================================")
    lofo_res = run_lofo_evaluation(data_root, fuser, unified_gt, file_to_fam, img_to_split)
    print(f"Evaluated {lofo_res['total_images']} images across {lofo_res['families_evaluated']} distinct template families.")
    print("| Field | n | Distinct Families | Statistical Flag | Accuracy (Ignoring Flag) | Auto-Accepted Accuracy | Routed to Review (%) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in lofo_res["aggregate_table"]:
        print(f"| `{row['field']}` | {row['n']} | {row['n_families']} | {row['sample_flag']} | {row['raw_accuracy']} | {row['auto_accepted_accuracy']} | {row['review_routing_share']} |")

    # Output per-family breakdown
    print("\n--- Per-Family Extraction Breakdown (LOFO) ---")
    for fam, f_data in lofo_res["per_family"].items():
        c = f_data["counts"]
        rm = f_data["raw_matches"]
        print(f"Family `{fam}` ({f_data['image_count']} images):")
        print(f"  total: {rm['total']}/{c['total']} | invoice_no: {rm['invoice_no']}/{c['invoice_no']} | date: {rm['date']}/{c['date']} | gstin: {rm['gstin']}/{c['gstin']}")


if __name__ == "__main__":
    main()
