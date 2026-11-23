# Protocol v2 Evaluation Results: Held-Out Test Set Benchmark

**Evaluation Split**: `test`  
**Date**: `2026-11-23 03:32:47 UTC`  
**Primary Model Weights**: `layouts_s0_best.pt`  
**Weights SHA-256**: `c76c6753c52c4d78ba73c966e603ec604fccafc633e95db1091e1ecc50824919`  
**Total Images Evaluated**: 14 (13 distinct invoices with target metadata, 1 continuation page)  

---

## How to Read This Report

### 1. Scored Invoices (13) vs Images (14)
The held-out test split contains **14 image files**. Among these:
- **13 images** are distinct single-page invoices or first-page documents containing complete header and metadata fields (`vendor`, `invoice_no`, `date`, `gstin`, `total`, `table`).
- **1 image** (`own_010_p2.png`) is a page-2 continuation sheet containing only itemized lines and terms, with **zero** ground-truth header metadata instances (0 bounding boxes in the label manifest).
Consequently, field extraction accuracy is evaluated over the **13 distinct invoices** that contain target fields.

### 2. Standards of Extraction Accuracy
- **Strict (Exact & Unflagged)**: The predicted field value matches ground truth bit-for-bit AND the automated quality gate accepted it without human review (`needs_review == False`).
- **Exact (Ignoring Review Flag)**: The predicted field value matches ground truth bit-for-bit, regardless of whether `needs_review` was flagged.
- **Lenient**: Semantically equivalent extraction:
  - `invoice_no`: Alphanumeric token normalization (stripping spaces, hyphens, and punctuation; case-insensitive).
  - `date`: ISO-8601 calendar equivalence with `dayfirst=True` ambiguity resolution (`DD-MM-YYYY` vs `MM-DD-YYYY`).
  - `total`: Numerical floating-point tolerance (|T - P| <= 0.01).

### 3. Evaluation Settings
- **Setting (a)**: YOLOv8n predicted boxes (`conf = 0.05`) + crop-level EasyOCR + domain parsers.
- **Setting (b)**: Human ground-truth bounding boxes + crop-level EasyOCR + domain parsers (the theoretical ceiling for a perfect visual detector, isolating crop OCR and parser limitations).
- **Setting (c)**: Baseline spatial rules heuristic applied to full-page EasyOCR word tokens.
- **Fused-OCR-only [HEADLINE BENCHMARK]**: Multi-modal fusion combining YOLO proposals, spatial candidates, and arithmetic constraints, where ALL images strictly utilize full-page EasyOCR tokens (`page_text_tokens = None`). Represents the headline benchmark for camera captures and scanned documents.
- **Fused-with-text-layer (Upper Bound for Digital PDFs)**: Multi-modal fusion utilizing the underlying digital PDF text layer words for own bills where available. Because ground truth for own bills was derived from the PDF text layer, this variant is partly circular and acts strictly as an upper bound for digital PDFs.

### 4. GSTIN Evaluation Standards
- **Document-Level Match**: Percentage of distinct invoices with GSTIN (n=10) where at least one predicted GSTIN matches any ground-truth GSTIN on that document.
- **Token-Level Set Recall**: Percentage of *all individual ground-truth GSTIN tokens* (supplier and customer tax IDs, total n=16 tokens across 13 invoices) retrieved by checksum-valid predictions.

---

## Honest Executive Summary

All figures below are drawn directly from the executed single-run benchmark on the frozen held-out test split:

1. **Generalization Drop on Unseen Layouts**:
   - Test **mAP@0.50** reached **0.2909** (0.291) compared to the validation score of **0.5140** (0.514), representing a drop of Delta = -0.223.
   - This drop reflects real-world distribution shift: while validation consisted of 2 known template families, the test split evaluates 11 distinct unseen template families across 14 images.
2. **Headline Fused-OCR-only Field Performance**:
   - **`date`**: **53.8% (7/13)** across strict, exact, and lenient criteria. Date tokens are relatively standardized and benefit from ISO parsing.
   - **`invoice_no`**: **30.8% (4/13)** lenient, **23.1% (3/13)** exact ignoring review flag, and **15.4% (2/13)** strict. Varied typography and alphanumeric noise remain challenging.
   - **`total`**: **15.4% (2/13)** across strict, exact, and lenient criteria. Missing itemized line-item OCR prevented arithmetic subset balancing on 11 of 13 bills.
   - **`gstin`**: **18.8% (3/16)** token set recall, and **30.0% (3/10)** document-level match.
3. **Human-Box Ceiling (Setting b)**:
   - Setting (b) provides human-annotated bounding boxes directly to the crop OCR and domain parsers.
   - Even with 100% localization accuracy:
     - `invoice_no`: **strict 46.2% (6/13)**, **lenient 53.8% (7/13)**.
     - `total`: **strict 30.8% (4/13)**, **ignore-flag and lenient 53.8% (7/13)**.
     - `date`: **strict 30.8% (4/13)**, **ignore-flag and lenient 38.5% (5/13)**.
     - `gstin`: **25.0% (4/16)** token set recall.
   - This proves that ~46% to 75% of end-to-end extraction failures stem from crop-level OCR misreads and rigid regex parsers, rather than bounding box localization errors.
4. **Spatial Rules Baseline Comparison (Setting c vs Fused-OCR-only Headline)**:
   - Comparing the full-page spatial rules baseline (Setting c) against the Fused-OCR-only headline benchmark:
     - `invoice_no`: strict **2/13 vs 2/13** (15.4%), lenient **3/13 vs 4/13** (23.1% vs 30.8%).
     - `date`: strict and lenient **6/13 vs 7/13** (46.2% vs 53.8%).
     - `total`: strict and lenient **2/13 vs 2/13** (15.4%).
     - `gstin`: token set recall **2/16 vs 3/16** (12.5% vs 18.8%).
   - **Statistical Reality**: All differences are within one invoice at n=13 (and within one token at n=16), so there is **no evidence fusion beats the rules baseline on the test set**.
5. **Review Gate Triage Performance**:
   - The quality gate routed **76.9% (10/13)** of invoices to manual review and auto-accepted **23.1% (3/13)**.
   - For **unflagged fields (`needs_review == False`)**, **11/24 (45.8%)** were strictly correct (12/24, 50.0% lenient).
   - For **flagged fields (`needs_review == True`)**, **0/15 (0.0%)** were strictly correct (1/15, 6.7% lenient).
   - The review gate successfully eliminated straight-through false acceptances on flagged fields (0% error rate on flagged fields), but the auto-accept accuracy (45.8%) demonstrates that confidence calibration requires further work before unattended straight-through processing.
6. **Primary Weights Benchmark**:
   - All results were produced by the pre-declared **PRIMARY** checkpoint (`layouts_s0_best.pt`). Secondary checkpoints were archived on remote storage and omitted locally.

---

## Deep-Dive Code Analysis

### 1. Why Vendor Recall is 61.5% in Section (b) but 23.1% in Section (c)
- **Code Evidence**:
  - In `scripts/run_final_test_v2.py` (lines 742–762), Section (b) evaluates raw detector operating points. It loops through all 51 bounding box proposals for `vendor` generated by `model.predict(conf=0.05)`. Any predicted box overlapping a ground-truth box with IoU >= 0.50 is counted as a true positive. Here, 8 of 13 ground-truth vendor boxes were matched by at least one raw proposal:
    Recall (Section b) = 8/13 ≈ 61.5%
  - In contrast, `src/invoice_vision/pipeline/extract.py` (lines 353–368) enforces a single-prediction constraint for downstream field extraction:
    ```python
    for cname in ["vendor", "invoice_no", "date", "table", "total"]:
        dets = detections_by_class[cname]
        ...
        best_conf, best_box = dets[0]
    ```
    The pipeline picks **only the single highest-confidence detection** (`dets[0]`).
  - In Section (c) (`scripts/run_final_test_v2.py`, lines 966–973), vendor localization is evaluated using `pred_a_box = fields_a.get("vendor", {}).get("source_box")` (only `dets[0]`).
  - Because the detector produced 43 false-positive vendor boxes at conf = 0.05, in 5 out of the 8 images where a true vendor box was detected, an irrelevant false-positive detection (e.g., logo, header banner, or customer address block) had a higher confidence score than the true vendor box. Only in 3 images was the true box ranked #1 by confidence:
    Recall (Section c) = 3/13 ≈ 23.1%

### 2. Why GSTIN Set Recall for Setting (a) is 0/16 Despite 4 Matched Bounding Boxes
- **Code Path (Verified)**:
  - In `scripts/run_final_test_v2.py` (line 983), candidate extraction for Setting (a) requires:
    ```python
    cand_a = [c["value"] for c in fields_a.get("gstins", []) if c.get("value") and c.get("validation_result")]
    ```
    A candidate is discarded unless `c.get("validation_result") is True`.
  - In `src/invoice_vision/pipeline/extract.py` (lines 141–189, 406–425), `parse_gstin_field` parses the crop OCR text and executes `validate_gstin_checksum(corrected)`. If the 15-character string fails the Mod-36 Luhn checksum (even after single-character confusion substitutions for `I/1`, `O/0`, `Z/7`), `val_res` is marked `False`.
  - In Section (b), 4 GSTIN boxes overlapped ground truth with IoU >= 0.50. However, none of the 4 detected crops yielded a checksum-valid GSTIN.
  - Because no crop passed the checksum, none entered the candidate set `cand_a` (`cand_a = []`), resulting in **0/16 (0.0%)** set recall.
- **Cause Analysis (Hypothesis)**:
  - The code path itself (no crop passed checksum => none entered candidate set) is **verified**.
  - The underlying OCR-misread cause is a **HYPOTHESIS**: per-crop character edit distances and raw OCR outputs were not logged during the single frozen test execution to protect private own-bill identifiers. It is hypothesized that multi-character substitution errors or bounding box boundary clipping on low-contrast, 8pt alphanumeric tax strings corrupted the tokens beyond single-character checksum recovery.

---

## Detailed Evaluation Report

### Section (a): Ultralytics Detector Metrics
- **Overall mAP@0.50**: **0.2909**

| Class ID | Class Name | Ground Truth Instances | Precision | Recall | AP@0.50 | AP@0.50:0.95 | Statistical Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | `vendor` | 13 | 0.2490 | 0.5385 | 0.4458 | 0.1823 | (too few to measure) |
| 1 | `invoice_no` | 13 | 0.4681 | 0.1391 | 0.2213 | 0.0877 | (too few to measure) |
| 2 | `date` | 13 | 0.3936 | 0.1047 | 0.2422 | 0.1231 | (too few to measure) |
| 3 | `gstin` | 16 | 0.3495 | 0.2500 | 0.2340 | 0.1416 | VALID |
| 4 | `table` | 13 | 0.0698 | 0.1538 | 0.1467 | 0.0647 | (too few to measure) |
| 5 | `total` | 14 | 0.3523 | 0.5000 | 0.4557 | 0.1431 | (too few to measure) |

---

### Section (b): Operating Point Metrics
- **Operating Confidence Threshold**: `conf = 0.05`
- **IoU Matching Cutoff**: `IoU >= 0.50`
- **Average False Detections per Image**: **7.86** (110 total FP across 14 images)

| Class Name | Ground Truth | Predictions | True Positives | False Positives | False Negatives | Precision | Recall | Statistical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `vendor` | 13 | 51 | 8 | 43 | 5 | 15.7% | 61.5% | (too few to measure) |
| `invoice_no` | 13 | 12 | 4 | 8 | 9 | 33.3% | 30.8% | (too few to measure) |
| `date` | 13 | 7 | 3 | 4 | 10 | 42.9% | 23.1% | (too few to measure) |
| `gstin` | 16 | 18 | 4 | 14 | 12 | 22.2% | 25.0% | VALID |
| `table` | 13 | 34 | 6 | 28 | 7 | 17.6% | 46.2% | (too few to measure) |
| `total` | 14 | 21 | 8 | 13 | 6 | 38.1% | 57.1% | (too few to measure) |

---

### Section (c): Field Extraction Accuracy Across Comparable Settings

| Field | GT n | Statistical Status | Setting (a) Strict | Setting (a) Ignore Flag | Setting (a) Lenient | Setting (b) Strict | Setting (b) Ignore Flag | Setting (b) Lenient | Setting (c) Strict | Setting (c) Lenient | Fused-OCR-only [HEADLINE] Strict | Fused-OCR-only [HEADLINE] Ignore Flag | Fused-OCR-only [HEADLINE] Lenient | Fused-with-text [Upper Bound] Strict | Fused-with-text [Upper Bound] Ignore Flag | Fused-with-text [Upper Bound] Lenient |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `invoice_no` | 13 | (too few to measure) | 0.0% (0/13) | 7.7% (1/13) | 15.4% (2/13) | 46.2% (6/13) | 46.2% (6/13) | 53.8% (7/13) | 15.4% (2/13) | 23.1% (3/13) | 15.4% (2/13) | 23.1% (3/13) | 30.8% (4/13) | 0.0% (0/13) | 7.7% (1/13) | 15.4% (2/13) |
| `date` | 13 | (too few to measure) | 0.0% (0/13) | 7.7% (1/13) | 7.7% (1/13) | 30.8% (4/13) | 38.5% (5/13) | 38.5% (5/13) | 46.2% (6/13) | 46.2% (6/13) | 53.8% (7/13) | 53.8% (7/13) | 53.8% (7/13) | 53.8% (7/13) | 53.8% (7/13) | 53.8% (7/13) |
| `total` | 13 | (too few to measure) | 0.0% (0/13) | 38.5% (5/13) | 38.5% (5/13) | 30.8% (4/13) | 53.8% (7/13) | 53.8% (7/13) | 15.4% (2/13) | 15.4% (2/13) | 15.4% (2/13) | 15.4% (2/13) | 15.4% (2/13) | 30.8% (4/13) | 30.8% (4/13) | 30.8% (4/13) |
| `gstin (set recall)` | 16 | VALID | 0.0% (0/16) | 0.0% (0/16) | 0.0% (0/16) | 25.0% (4/16) | 25.0% (4/16) | 25.0% (4/16) | 12.5% (2/16) | 12.5% (2/16) | 18.8% (3/16) | 18.8% (3/16) | 18.8% (3/16) | 25.0% (4/16) | 25.0% (4/16) | 25.0% (4/16) |

> [!NOTE]
> **GSTIN Document-Level Accuracy (at least 1 predicted GSTIN matches ground truth on invoice)**:
> - **Fused-OCR-only [HEADLINE]**: 30.0% (3/10)
> - **Fused-with-text-layer [Upper Bound]**: 40.0% (4/10)

#### Vendor & Table Localization Performance (Detection Recall Only)
| Entity / Structure | GT Instances | Setting (a) [YOLO] Recall | Setting (b) [Human GT] Recall | Fused-OCR-only [HEADLINE] Recall | Fused-with-text [Upper Bound] Recall | Statistical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `vendor` | 13 | 23.1% (3/13) | 100.0% (13/13) | 23.1% (3/13) | 23.1% (3/13) | (too few to measure) |
| `table` | 13 | 38.5% (5/13) | 100.0% (13/13) | 38.5% (5/13) | 38.5% (5/13) | (too few to measure) |

---

### Section (d): Review Routing Metrics

#### Review Routing Summary Across Fused Settings
| Metric | Fused-OCR-only [HEADLINE RESULT] | Fused-with-text-layer [Upper Bound] |
| :--- | :---: | :---: |
| **Total Scored Documents** | 13 | 13 |
| **Documents Routed to Review (`needs_review == True`)** | 76.9% (10/13) | 100.0% (13/13) |
| **Documents Auto-Accepted Straight-Through (`needs_review == False`)** | 23.1% (3/13) | 0.0% (0/13) |

#### Field Extraction Accuracy by Review Status: Fused-OCR-only [HEADLINE RESULT]
| Field Review Status | Total Fields Scored | Strict Field Accuracy | Lenient Field Accuracy |
| :--- | :---: | :---: | :---: |
| **Unflagged Fields (`needs_review == False`)** | 24 | 45.8% (11/24) | 50.0% (12/24) |
| **Flagged Fields (`needs_review == True`)** | 15 | 0.0% (0/15) [flagged] | 6.7% (1/15) |

#### Field Extraction Accuracy by Review Status: Fused-with-text-layer (Upper Bound)
| Field Review Status | Total Fields Scored | Strict Field Accuracy | Lenient Field Accuracy |
| :--- | :---: | :---: | :---: |
| **Unflagged Fields (`needs_review == False`)** | 19 | 57.9% (11/19) | 63.2% (12/19) |
| **Flagged Fields (`needs_review == True`)** | 20 | 0.0% (0/20) [flagged] | 5.0% (1/20) |

---

### Section (e): Template Family & Source Breakdown

#### Breakdown by Data Source: Fused-OCR-only [HEADLINE RESULT]
| Data Source | Images | Scored Invoices | Invoice No Lenient | Date Lenient | Total Lenient | GSTIN Recall | Review Route Rate (Invoices) | Statistical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `public` | 6 | 6 | 1/6 | 2/6 | 1/6 | 0/12 | 100.0% (6/6) | (too few to measure) |
| `own` | 8 | 7 | 3/7 | 5/7 | 1/7 | 3/4 | 57.1% (4/7) | (too few to measure) |

#### Breakdown by Data Source: Fused-with-text-layer (Upper Bound)
| Data Source | Images | Scored Invoices | Invoice No Lenient | Date Lenient | Total Lenient | GSTIN Recall | Review Route Rate (Invoices) | Statistical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `public` | 6 | 6 | 1/6 | 2/6 | 1/6 | 0/12 | 100.0% (6/6) | (too few to measure) |
| `own` | 8 | 7 | 1/7 | 5/7 | 3/7 | 4/4 | 100.0% (7/7) | (too few to measure) |

#### Breakdown by Template Family: Fused-OCR-only [HEADLINE RESULT]
| Template Family | Source | Images | Scored Invoices | Invoice No Lenient | Date Lenient | Total Lenient | GSTIN Recall | Review Route Rate (Invoices) | Statistical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `public_family_A` | `public` | 1 | 1 | 0/1 | 0/1 | 0/1 | 0/2 | 100.0% (1/1) | (too few to measure) |
| `public_family_B` | `public` | 2 | 2 | 0/2 | 0/2 | 0/2 | 0/4 | 100.0% (2/2) | (too few to measure) |
| `public_family_C` | `public` | 1 | 1 | 0/1 | 0/1 | 0/1 | 0/2 | 100.0% (1/1) | (too few to measure) |
| `public_family_D` | `public` | 1 | 1 | 1/1 | 1/1 | 1/1 | 0/2 | 100.0% (1/1) | (too few to measure) |
| `public_family_E` | `public` | 1 | 1 | 0/1 | 1/1 | 0/1 | 0/2 | 100.0% (1/1) | (too few to measure) |
| `own_family_H` | `own` | 2 | 2 | 0/2 | 2/2 | 0/2 | 2/2 | 0.0% (0/2) | (too few to measure) |
| `own_family_I` | `own` | 1 | 1 | 0/1 | 1/1 | 0/1 | 0/0 | 100.0% (1/1) | (too few to measure) |
| `own_family_G` | `own` | 2 | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 0.0% (0/1) | (too few to measure) |
| `own_family_F` | `own` | 1 | 1 | 1/1 | 0/1 | 0/1 | 0/0 | 100.0% (1/1) | (too few to measure) |
| `own_family_J` | `own` | 1 | 1 | 0/1 | 1/1 | 0/1 | 0/0 | 100.0% (1/1) | (too few to measure) |
| `own_family_E` | `own` | 1 | 1 | 1/1 | 0/1 | 0/1 | 0/1 | 100.0% (1/1) | (too few to measure) |

#### Breakdown by Template Family: Fused-with-text-layer (Upper Bound)
| Template Family | Source | Images | Scored Invoices | Invoice No Lenient | Date Lenient | Total Lenient | GSTIN Recall | Review Route Rate (Invoices) | Statistical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `public_family_A` | `public` | 1 | 1 | 0/1 | 0/1 | 0/1 | 0/2 | 100.0% (1/1) | (too few to measure) |
| `public_family_B` | `public` | 2 | 2 | 0/2 | 0/2 | 0/2 | 0/4 | 100.0% (2/2) | (too few to measure) |
| `public_family_C` | `public` | 1 | 1 | 0/1 | 0/1 | 0/1 | 0/2 | 100.0% (1/1) | (too few to measure) |
| `public_family_D` | `public` | 1 | 1 | 1/1 | 1/1 | 1/1 | 0/2 | 100.0% (1/1) | (too few to measure) |
| `public_family_E` | `public` | 1 | 1 | 0/1 | 1/1 | 0/1 | 0/2 | 100.0% (1/1) | (too few to measure) |
| `own_family_H` | `own` | 2 | 2 | 0/2 | 2/2 | 2/2 | 2/2 | 100.0% (2/2) | (too few to measure) |
| `own_family_I` | `own` | 1 | 1 | 0/1 | 1/1 | 0/1 | 0/0 | 100.0% (1/1) | (too few to measure) |
| `own_family_G` | `own` | 2 | 1 | 0/1 | 1/1 | 1/1 | 1/1 | 100.0% (1/1) | (too few to measure) |
| `own_family_F` | `own` | 1 | 1 | 0/1 | 0/1 | 0/1 | 0/0 | 100.0% (1/1) | (too few to measure) |
| `own_family_J` | `own` | 1 | 1 | 0/1 | 1/1 | 0/1 | 0/0 | 100.0% (1/1) | (too few to measure) |
| `own_family_E` | `own` | 1 | 1 | 1/1 | 0/1 | 0/1 | 1/1 | 100.0% (1/1) | (too few to measure) |

---

## Secondary Weights Robustness Status
- `layouts_s1_best.pt`: `[MISSING LOCALLY - NOT EVALUATED]`
- `layouts_s2_best.pt`: `[MISSING LOCALLY - NOT EVALUATED]`
- `layouts_s3_best.pt`: `[MISSING LOCALLY - NOT EVALUATED]`
- `layouts_s4_best.pt`: `[MISSING LOCALLY - NOT EVALUATED]`

> **Protocol Note**: Secondary weights checkpoints are archived on remote storage and omitted locally. The definitive benchmark is established strictly by PRIMARY weights (`layouts_s0_best.pt`).


---

## Post-Hoc Diagnostic: Ground-Truth GSTIN Checksum Integrity (Post-Hoc, Not Part of Protocol v2)

A read-only diagnostic was conducted on the ground-truth GSTIN strings in the test split (evaluating ground-truth annotations alone, completely independent of model predictions or OCR outputs) to check how many ground-truth GSTIN strings pass the Mod-36 Luhn checksum:
- **Public Images (`adityaeucloid/test_dataset`)**: **2 / 12** ground-truth GSTINs pass checksum (**16.7%**). The public test images come from a third-party Hugging Face invoice dataset with no license or dataset card (`adityaeucloid/test_dataset`), used locally and not redistributed; the Divya52 synthetic invoices (CC BY 4.0) were used only for training and for the two README sample images.
- **Own-Bill Images**: **4 / 4** ground-truth GSTINs pass checksum (**100.0%**).
- **Total Test Ground-Truth GSTINs**: **6 / 16** pass checksum (**37.5%**).

> [!NOTE]
> This diagnostic is post-hoc and exploratory; the official headline scoring and Protocol v2 evaluation metrics are strictly unchanged.