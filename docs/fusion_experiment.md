# Experiment 3: Candidate Fusion & Arithmetic Validation

**Date**: 2026-11-22  
**Status**: Pre-Test Validation Completed (TEST Split Strictly Untouched)  
**Author**: Senior Computer Vision & Document Intelligence Engineer  

---

## 1. Motivation & Fusion Architecture

Analysis of the YOLOv8-only validation results (results/pipeline_val_v2.md) revealed key structural failure modes:
1. **Total Amount Localization Failure**: YOLO detector missed the `total` bounding box on 77.8% (7/9) of validation invoices, achieving 0% extraction accuracy.
2. **Review Routing Gating**: While raw crop-level recognition succeeded on 66.7% of invoice numbers and 88.9% of dates, 100% of these detections fell in the [0.05, 0.25) confidence interval, causing them to be routed to manual review.
3. **Redundant Deterministic Evidence**: Full-page OCR text contains domain constraints (arithmetic equations between taxable amounts and GST taxes) and spatial key-value anchors that were unused by the purely visual detector.

### Multi-Modal Candidate Fusion (`src/invoice_vision/pipeline/fuse.py`)
- **Total Amount**: Evaluates candidate amounts across the page to find an amount G satisfying:
  taxable_value + taxes (+ round_off) = grand_total (± 0.5)
  using subsets of 2, 3, or 4 tokens. Falls back to the YOLO bounding box value, then to the v1 2D spatial rules. Never invents digits or decimals.
- **Invoice Number & Date**:
  - If YOLO detector confidence >= 0.25: auto-accepts or corroborates with spatial rules.
  - If both YOLO and spatial rules exist and disagree: **keeps the rules value and flags for review (`needs_review = True`)**.
  - If YOLO detector confidence < 0.25: falls back to spatial rules candidate.
- **GSTIN**: Combines YOLO crop-level OCR (with 3x adaptive bicubic upscaling and Luhn checksum correction) with any checksum-valid tax identifiers discovered across full-page tokens (capturing buyer GSTINs without visual drift).
- **Provenance & Review Gate**: Every output field explicitly tags its provenance (`source`: `"arithmetic"`, `"yolo"`, `"rules"`, `"none"`), confidence, and boolean `needs_review` flag.

---

## 2. Validation Split Results (12 Images, 9 Distinct Invoices)

Evaluated on the exact 12 validation images (`data/splits_v3.json`) comparing the YOLO-only baseline (Setting a, `conf = 0.05`) against the multi-modal fused pipeline:

| Target Field | GT n | Statistical Flag | YOLO Acc (Setting a) | Fused Raw Acc (Ignore Flag) | Fused Auto-Accepted Acc | Review Routed Share | Delta Invoices vs YOLO |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `invoice_no` | 9 | (too few to measure) | 66.7% (6/9) | **88.9% (8/9)** | 11.1% (1/9) | 88.9% (8/9) | **+2** |
| `date` | 9 | (too few to measure) | 88.9% (8/9) | **100.0% (9/9)** | **88.9% (8/9)** | 11.1% (1/9) | **+1** |
| `total` | 9 | (too few to measure) | 0.0% (0/9) | **66.7% (6/9)** | **66.7% (6/9)** | 0.0% (0/9) | **+6** |
| `gstin` | 9 | (too few to measure) | 88.9% (8/9) | **100.0% (9/9)** | 11.1% (1/9) | 88.9% (8/9) | **+1** |

### Pre-Declared Decision Rule Verification
- **Rule**: Fusion counts as an empirical improvement only if VAL `total` accuracy rises by at least 3 invoices (Delta >= +3) AND no other field drops by more than 1 invoice (Delta >= -1).
- **Observed**:
  - `total` Delta: **+6 invoices** (>= +3, PASSED)
  - `invoice_no` Delta: **+2 invoices** (>= -1, PASSED)
  - `date` Delta: **+1 invoice** (>= -1, PASSED)
  - `gstin` Delta: **+1 invoice** (>= -1, PASSED)
- **Status**: **PASSED**.

> [!WARNING]
> **Scientific Governance Caveat**:
> The fusion and arithmetic rules were designed after analyzing specific validation failure modes (namely, low YOLO confidence on own_family_A templates and missing total detections). Therefore, **these validation metrics are optimistic (in-sample tuned)**. Unbiased verification of generalizability requires evaluation on the held-out, frozen test split under Protocol v2.

---

## 3. Leave-One-Family-Out (LOFO) Evaluation (TRAIN & VAL Families)

To evaluate cross-template generalizability without training bias, we conducted a Leave-One-Family-Out evaluation across all **12 template families** in the TRAIN and VAL splits (69 total images).

Because the YOLO detector was trained on the training families, evaluating YOLO on those families would be contaminated. Per protocol, **LOFO evaluates the deterministic spatial rules and arithmetic validation candidates only**, which contain zero learned weights:

| Field | Total n | Distinct Families | Statistical Flag | Raw Accuracy (Ignore Flag) | Auto-Accepted Accuracy | Share Routed to Review |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `invoice_no` | 63 | 12 | Adequate sample (n >= 15) | **25.4% (16/63)** | 20.6% (13/63) | 77.8% (49/63) |
| `date` | 65 | 12 | Adequate sample (n >= 15) | **58.5% (38/65)** | 55.4% (36/65) | 15.4% (10/65) |
| `total` | 62 | 12 | Adequate sample (n >= 15) | **41.9% (26/62)** | 41.9% (26/62) | 1.6% (1/62) |
| `gstin` | 60 | 12 | Adequate sample (n >= 15) | **55.0% (33/60)** | 55.0% (33/60) | 43.3% (26/60) |

### Per-Family LOFO Extraction Breakdown
- **`own_family_C` (3 images)**: total: 3/3 (100%) | invoice_no: 0/3 | date: 3/3 (100%) | gstin: 3/3 (100%)
- **`Amazon` (6 images)**: total: 0/3 | invoice_no: 3/3 (100%) | date: 3/3 (100%) | gstin: 0/3
- **`LinkedIn` (5 images)**: total: 3/4 (75%) | invoice_no: 2/4 (50%) | date: 2/4 (50%) | gstin: 2/5 (40%)
- **`Microsoft` (8 images)**: total: 1/8 (12.5%) | invoice_no: 0/8 | date: 2/8 (25%) | gstin: 0/4
- **`own_family_B` (4 images)**: total: 2/4 (50%) | invoice_no: 0/2 | date: 2/4 (50%) | gstin: 2/2 (100%)
- **`own_family_A` (6 images)**: total: 6/6 (100%) | invoice_no: 0/6 | date: 6/6 (100%) | gstin: 6/6 (100%)
- **`Sai_Computer` (7 images)**: total: 0/4 | invoice_no: 0/7 | date: 6/7 (85.7%) | gstin: 0/7
- **`own_family_D` (6 images)**: total: 0/6 | invoice_no: 0/6 | date: 5/6 (83.3%) | gstin: 6/6 (100%)
- **`gstsyn_classic` (6 images)**: total: 2/6 (33.3%) | invoice_no: 3/6 (50%) | date: 2/6 (33.3%) | gstin: 4/6 (66.7%)
- **`gstsyn_handwritten` (6 images)**: total: 5/6 (83.3%) | invoice_no: 4/6 (66.7%) | date: 5/6 (83.3%) | gstin: 3/6 (50%)
- **`gstsyn_minimal` (6 images)**: total: 3/6 (50%) | invoice_no: 4/6 (66.7%) | date: 1/6 (16.7%) | gstin: 3/6 (50%)
- **`gstsyn_thermal` (6 images)**: total: 1/6 (16.7%) | invoice_no: 0/6 | date: 1/6 (16.7%) | gstin: 4/6 (66.7%)

---

## 4. Part A: Label-Fix Proposal for `own_013_p1.png`

Blind label audit revealed that `own_013_p1.png` had 0 bounding boxes labeled for `invoice_no` despite containing a valid token in its PDF text layer.

### Proposed Bounding Box
- **Anchor Keyword**: `INVOICE#`
- **Target Value Word**: Word immediately adjacent to `INVOICE#`
- **Absolute Box (2 px margin)**: [863.22, 458.59, 1034.97, 483.08] px (W=1653, H=2339 px)
- **Normalized YOLO Box (`class_id = 1`)**:
  `1 0.574165 0.201298 0.103902 0.010470`
- **Overlap Check**: `0 overlaps` with existing boxes (**False**).
- **Proposal Files Generated** (existing label files strictly unmodified):
  - `data/processed/label_fix_proposal/exported_batch_07_test_own_013_p1.txt`
  - `data/processed/label_fix_proposal/yolo_v2_own_013_p1.txt`
  - `data/processed/label_fix_proposal/own_013_p1_proposal_overlay.png`