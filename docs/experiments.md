# Model Training Experiments Log

This document records training experiments, ablations, empirical metrics, and statistical defenses for the Indian invoice document field extraction system.

---

## Experiment 1: Real-Only vs. Real + Auto-Transferred Synthetic Photos

- **Date**: 2026-11-21
- **Status**: Completed
- **Evaluation Split**: Validation set only (n=12 images, 57 ground-truth bounding boxes, template families `Amazon` and `own_family_A` / `T13`).
- **Test Split**: **Untouched**. The test split (n=14 images) remains strictly isolated for final held-out benchmark evaluation.

### 1. Experimental Setup & Hardware Configuration
- **Model Architecture**: YOLOv8n (`yolov8n.pt` pretrained backbone, 3.01M parameters)
- **Configuration**: [configs/yolo_v1.yaml](configs/yolo_v1.yaml)
- **Input Canvas**: 1024×1024 px
- **Schedule**: 100 epochs, patience 30, batch size 8
- **Augmentation Pipeline**: Document-specific (`fliplr: 0`, `flipud: 0`, `degrees: 5.0`, `perspective: 0.0005`, `scale: 0.3`, `translate: 0.1`, HSV jitter, `mosaic: 1.0` with `close_mosaic: 10`)
- **Random Seeds**: Seeds 0, 1, 2 (3 independent runs per condition to measure seed variance)
- **Software**: `ultralytics==8.4.215`
- **Hardware Platform**: Google Colab (NVIDIA T4 Tensor Core GPU, 16 GB VRAM)

### 2. Dataset Conditions
1. **Real-Only Baseline**:
   - 33 real training images (18 public clean scans from `data/raw/indian_gst` + 15 personal bills from `data/raw/own`).
   - Ground-truth boxes manually verified in makesense.ai.
2. **Real + Synthetic Photos (`--with-synth`)**:
   - 33 real training images + 32 auto-transferred synthetic photo variations (`mild`, `medium`, `severe`, `hard` perspective/blur variants).
   - Total: 65 training images.
   - Note: The 32 synthetic photos derive from only 4 seed invoice templates; their bounding boxes were automatically homography-transferred and unreviewed.

---

### 3. Empirical Results (Validation Split)

> [!NOTE]
> **Changelog (2026-11-22)**: Earlier recorded per-seed values (0.352, 0.391, 0.337 and 0.360, 0.378, 0.225) were invented from aggregate means and ranges and have been corrected to the exact values from the Colab comparison printout (real-only: 0.337, 0.391, 0.353; real+synthetic copies: 0.378, 0.361, 0.225). Aggregate means (0.360 and 0.321) and per-class ranges remain unchanged.

#### 3.1 Aggregate Performance (mAP@0.50)
| Condition | Training Images | Seed 0 | Seed 1 | Seed 2 | Mean mAP@0.50 | Seed Spread (Min – Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real-Only Baseline** | **33** | 0.337 | 0.391 | 0.353 | **0.360** | **0.337 – 0.391** (Delta = 0.054) |
| **Real + Synthetic Photos** | **65** | 0.378 | 0.361 | 0.225 | **0.321** | **0.225 – 0.378** (Delta = 0.153) |

#### 3.2 Per-Class AP@0.50 Breakdown (Mean and Min – Max Range)
| Class Name | Real-Only Mean AP50 | Real-Only Range (Min – Max) | Synth+Real Mean AP50 | Synth+Real Range (Min – Max) |
| :--- | :---: | :---: | :---: | :---: |
| `vendor` | 0.504 | 0.476 – 0.525 | **0.660** | 0.541 – 0.767 |
| `invoice_no` | **0.229** | 0.007 – 0.433 | 0.055 | 0.037 – 0.072 |
| `date` | **0.372** | 0.140 – 0.594 | 0.166 | 0.030 – 0.308 |
| `gstin` | 0.717 | 0.391 – 0.968 | **0.768** | 0.492 – 0.943 |
| `table` | **0.214** | 0.188 – 0.236 | 0.175 | 0.150 – 0.211 |
| `total` | **0.127** | 0.023 – 0.208 | 0.105 | 0.013 – 0.235 |

---

### 4. Technical Analysis & Interview Defense

> [!IMPORTANT]
> **Key Finding**: The difference in mean validation mAP50 (0.360 vs 0.321) is fully within the seed-to-seed variance (Delta = 0.054 for real, Delta = 0.153 for synth). There is **no empirical evidence that auto-transferred synthetic photo copies improve detection performance**.

#### Critical Engineering Limitations:
1. **Unreviewed Auto-Transferred Labels**: The 32 synthetic photo variations derive from only 4 seed documents. Bounding boxes were generated via geometric homography transfer without human review. Sub-pixel interpolation error, warp clipping, and severe blur degraded ground-truth box quality, introducing label noise.
2. **Template Diversity vs. Image Count**: Adding 32 photos increased image volume by 97%, but added **zero template diversity**. The network overfitted to repetitive features of the 4 seed documents, leading to severe regression on validation classes like `invoice_no` (0.229 -> 0.055) and `date` (0.372 -> 0.166).
3. **Small Sample Validation Gating**: Per-class validation statistics rest on only 9 to 12 ground-truth boxes per class across 12 images. A single detection error alters class recall by 8.3% - 11.1%. The observed spread across seeds (e.g. `invoice_no` ranging from 0.007 to 0.433) highlights the high sensitivity of small evaluation sets.
4. **Validation Optimism**: The checkpoint (`best.pt`) for each run was selected based on the peak validation score achieved during 100 epochs on this exact validation set. Reported numbers are therefore slightly optimistic.
5. **Test Split Integrity**: The test split (n=14 images) has not been evaluated or tuned against in this experiment, ensuring strict integrity for downstream benchmark reporting.

### 5. Architectural Decision
**The baseline for all future experiments is strictly Real-Only.** Auto-transferred synthetic photos from the 4 seed invoices will not be included in production training splits.

---

## Experiment 2: Real Baseline vs. Real + Multi-Layout Synthetic Invoices (`data/yolo_v2`)

- **Date**: 2026-11-22
- **Status**: Completed
- **Evaluation Split**: Validation set only (n=12 images, 57 ground-truth boxes across `Amazon` and `own_family_A` / `T13`).
- **Test Split**: **Untouched**. The test split (n=14 images) remains strictly isolated and is NOT used for any model selection, hyperparameter tuning, or experimental decisions.

### 1. Hypothesis
Adding 24 layout-varied synthetic invoices (sampled across 4 distinct layouts: `classic`, `handwritten`, `minimal`, and `thermal` from `Divya52/indian-gst-invoices`) to the 33 real training images (total: 57 training images in `data/yolo_v2`) improves validation mAP50 over the real-only baseline (which achieved a 3-seed mean of 0.360, range: 0.337 - 0.391).

### 2. Experimental Setup & Configuration
- **Model Architecture**: YOLOv8n (`yolov8n.pt` pretrained backbone, 3.01M parameters)
- **Configuration**: [configs/yolo_v1.yaml](configs/yolo_v1.yaml) (strictly unchanged)
- **Training Dataset**: `data/yolo_v2` (57 train images: 18 public clean + 15 own clean + 24 synthetic multi-layout; zero photo copies)
- **Validation Dataset**: `data/yolo_v2/images/val` (12 images, identical file set and SHA-256 hashes as `data/yolo_v1`)
- **Seeds**: 5 independent seeds (`seed=0, 1, 2, 3, 4`)
- **Software**: `ultralytics==8.4.215`
- **Evaluation**: Validation split only (`--splits val`)

### 3. Pre-Registered Decision Rule
To eliminate post-hoc rationalization, the threshold for adopting `data/yolo_v2` was defined in advance:
1. **Mean Gain Requirement**: The 5-seed mean validation mAP50 must exceed the baseline mean by at least +0.05 (i.e., mean validation mAP50 >= 0.410).
2. **Consistency Requirement**: At least 4 of 5 same-seed pairs must favor the new dataset:
   count(mAP50_v2(s) > mAP50_real(s)) >= 4 for s in {0, 1, 2, 3, 4}
3. **Outcome Resolution**:
   - If **both** conditions are satisfied: Declare an **empirical improvement** and adopt `data/yolo_v2` as the new training standard.
   - If **either** condition fails: Explicitly report **"no evidence of improvement"** and retain the 33-image real-only baseline.
   - The test set is not evaluated under any circumstance during this decision.

---

### 4. Empirical Results (Validation Split)

#### 4.1 Aggregate Performance (mAP@0.50 across 5 Seeds)
| Seed | Real-Only Baseline (33 images) | Layouts Dataset `yolo_v2` (57 images) | Same-Seed Difference (Delta) | Favors Layouts? |
| :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | 0.337 | **0.514** | +0.177 | Yes |
| **Seed 1** | 0.391 | **0.460** | +0.069 | Yes |
| **Seed 2** | 0.353 | **0.408** | +0.055 | Yes |
| **Seed 3** | 0.371 | **0.416** | +0.045 | Yes |
| **Seed 4** | 0.429 | **0.430** | +0.001 | Yes (Tie / Slight) |
| **Mean** | **0.376** | **0.446** | **+0.069** | **5 / 5 pairs** |

> **Decision Rule Verification**:
> - Mean validation mAP50 gain: +0.069 >= +0.05 (**PASS**)
> - Same-seed consistency: 5 of 5 pairs favor layouts >= 4/5 (**PASS**)
> - **Pre-Registered Decision Rule: PASSED**.

#### 4.2 Per-Class AP@0.50 Breakdown (Mean and Min – Max Range)
| Class Name | Real-Only Mean AP50 | Real-Only Range (Min – Max) | Layouts `yolo_v2` Mean AP50 | Layouts `yolo_v2` Range (Min – Max) |
| :--- | :---: | :---: | :---: | :---: |
| `vendor` | 0.531 | 0.476 – 0.610 | **0.629** | 0.486 – 0.995 |
| `invoice_no` | **0.261** | 0.007 – 0.433 | 0.255 | 0.063 – 0.684 |
| `date` | 0.370 | 0.140 – 0.594 | **0.599** | 0.226 – 0.765 |
| `gstin` | 0.774 | 0.391 – 0.968 | **0.840** | 0.705 – 0.922 |
| `table` | 0.205 | 0.188 – 0.236 | **0.245** | 0.139 – 0.412 |
| `total` | **0.117** | 0.010 – 0.208 | 0.105 | 0.032 – 0.214 |

---

### 5. Critical Statistical Caveats & Limitations
1. **Disproportionate Impact of Seed 0**: The overall mean gain (+0.069) is heavily driven by Seed 0 (0.337 -> 0.514, Delta = +0.177). If Seed 0 is excluded, the mean gain across seeds 1–4 drops to +0.042 (which would fall below the +0.05 threshold). In addition, Seed 4 was effectively a statistical tie (0.429 vs 0.430, Delta = +0.001).
2. **Overlapping Per-Class Ranges**: Across all 6 classes, the min–max ranges of the real baseline and `yolo_v2` significantly overlap (e.g. `invoice_no` 0.007-0.433 vs 0.063-0.684; `vendor` 0.476-0.610 vs 0.486-0.995).
3. **Small Sample Validation Gating**: Per-class validation statistics rest on only 9 to 12 ground-truth boxes per class across 12 images. High variance is expected with sample sizes this small.
4. **Validation Optimism**: The validation set consists of only 12 images from 2 template families (`Amazon` and `own_family_A`). The best model checkpoints were selected based on the peak validation epoch on this exact set.
5. **Causal Attribution is a Hypothesis**: The attribution of performance gains specifically to "layout variety" remains a working hypothesis. Experiment 1 and Experiment 2 differ along multiple dimensions simultaneously (sample count: 33 vs 57 images; human-reviewed vs unreviewed labels; synthetic distortion vs clean digital rendering).
6. **No Transfer Established for Phone Photos**: All added synthetic images are clean or digitally transformed; zero physical camera capture or ambient illumination variations were included. Performance on real handheld phone photos remains unestablished.
7. **Test Split Strict Isolation**: The held-out test split (n=14 images) remains 100% untouched and uninspected.

### 6. Frozen Pipeline Setup
Based on the pre-registered decision rule passing, the frozen detector configuration for the end-to-end extraction pipeline is:
- **Architecture**: YOLOv8n (`yolov8n.pt` backbone)
- **Hyperparameter Config**: [configs/yolo_v1.yaml](configs/yolo_v1.yaml)
- **Dataset**: `data/yolo_v2` (n=57 training images)
- **Model Checkpoint**: `layouts_s0` weights (data/models/layouts_s0_best.pt), which achieved the peak validation mAP50 (0.514) across the 5 seeds (acknowledged as an optimistic pick on validation data).

---

## Experiment 3: Expanded Multi-Layout Synthetic Invoices (`data/yolo_v3`)

- **Date**: 2026-11-22
- **Status**: Completed (Evaluated on Validation Split)
- **Evaluation Split**: Validation set only (n=12 images, 57 ground-truth boxes across `Amazon` and `own_family_A` / `T13`, identical to `data/yolo_v1` and `data/yolo_v2`).
- **Test Split**: **Untouched**. The held-out test split (n=14 images) remains strictly isolated and is NOT used for any model selection, hyperparameter tuning, or experimental decisions.

### 1. Hypothesis
Expanding the multi-layout synthetic training set from 24 to 84 invoices (stratified across 4 layouts: `classic`, `handwritten`, `minimal`, `thermal`, 21 images each; total training set: 117 images: 33 real + 84 synthetic) improves validation mAP50 over the `yolo_v2` layouts baseline (which achieved a 5-seed mean of 0.446, range: 0.408 - 0.514).

### 2. Experimental Setup & Configuration
- **Model Architecture**: YOLOv8n (`yolov8n.pt` pretrained backbone, 3.01M parameters)
- **Configuration**: [configs/yolo_v1.yaml](configs/yolo_v1.yaml) (strictly unchanged)
- **Training Dataset**: `data/yolo_v3` (117 train images: 18 public clean + 15 own clean + 84 synthetic multi-layout; zero photo copies)
- **Validation Dataset**: `data/yolo_v3/images/val` (12 images, identical file set, image SHA-256 hashes, and label SHA-256 hashes as `data/yolo_v2`)
- **Seeds**: 5 independent seeds (`seed=0, 1, 2, 3, 4`) with run names `v3_s0` to `v3_s4`
- **Baseline Values (`yolo_v2` Layouts)**:
  - Seed 0: 0.514
  - Seed 1: 0.460
  - Seed 2: 0.408
  - Seed 3: 0.416
  - Seed 4: 0.430
  - **Mean Baseline mAP50**: 0.446
- **Software**: `ultralytics==8.4.215`
- **Evaluation**: Validation split only (`--splits val`)

### 3. Pre-Registered Decision Rule

> [!NOTE]
> **Changelog (2026-11-22)**: The earlier text stated Delta >= +0.05 (target >= 0.496), which differed from the rule specified in the original instructions; corrected to mean validation mAP50 gain >= +0.03 over the measured `yolo_v2` mean 0.446 (target >= 0.476) AND >= 4 of 5 same-seed pairs favor `yolo_v3` before any Experiment 3 result existed.

1. **Mean Gain Requirement**: The 5-seed mean validation mAP50 must exceed the measured `yolo_v2` layouts baseline mean (0.446) by at least +0.03 (i.e., target mean validation mAP50 >= 0.476).
2. **Consistency Requirement**: At least 4 of 5 same-seed pairs must favor `data/yolo_v3`:
   count(mAP50_v3(s) > mAP50_layouts(s)) >= 4 for s in {0, 1, 2, 3, 4}
3. **Outcome Resolution**:
   - If **both** conditions are satisfied: Declare an **empirical improvement** and adopt `data/yolo_v3` as the new training baseline.
   - If **either** condition fails: Explicitly report **"no evidence of improvement"** and retain the `data/yolo_v2` baseline.

---

### 4. Empirical Results (Validation Split)

#### 4.1 Aggregate Performance (mAP@0.50 across 5 Seeds)
| Seed | Layouts Dataset `yolo_v2` (57 images) | Expanded Dataset `yolo_v3` (117 images) | Same-Seed Difference (Delta) | Favors v3? |
| :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | **0.514** | 0.455 | -0.059 | No |
| **Seed 1** | **0.460** | 0.441 | -0.018 | No |
| **Seed 2** | **0.408** | 0.371 | -0.037 | No |
| **Seed 3** | **0.416** | 0.397 | -0.019 | No |
| **Seed 4** | 0.430 | **0.536** | +0.106 | Yes |
| **Mean** | **0.446** | **0.440** | **-0.005** | **1 / 5 pairs** |

> **Decision Rule Verification**:
> - Mean validation mAP50 gain: -0.005 (target was >= +0.03) (**FAIL**)
> - Same-seed consistency: 1 of 5 pairs favor v3 (target was >= 4/5) (**FAIL**)
> - **Pre-Registered Decision Rule: NOT MET**.
> - **Architectural Conclusion**: There is no evidence that 60 extra synthetic invoices of the same 4 layouts help; keep `data/yolo_v2` / `layouts`.

---

### 5. Exploratory Analysis & Statistical Observations (Not Pre-Registered)

1. **Comparison Against Real-Only Baseline**:
   Comparing `yolo_v3` (117 images) directly against the 33-image real-only baseline per seed:
   - Seed 0: 0.337 -> 0.455 (Delta = +0.118)
   - Seed 1: 0.391 -> 0.441 (Delta = +0.050)
   - Seed 2: 0.353 -> 0.371 (Delta = +0.018)
   - Seed 3: 0.371 -> 0.397 (Delta = +0.026)
   - Seed 4: 0.429 -> 0.536 (Delta = +0.107)
   - **Mean Gain**: 0.376 -> 0.440 (Delta = +0.064), which is consistent with the Experiment 2 gain of +0.069 (0.376 -> 0.446).
2. **Seed-to-Seed Spread for the Same Recipe**:
   The variance across random seeds for the identical training recipe is large:
   - `layouts` (`yolo_v2`): 0.408 - 0.514 (spread Delta = 0.106)
   - `yolo_v3`: 0.371 - 0.536 (spread Delta = 0.165)
3. **Small Sample Validation Gating**:
   Validation is measured on only 12 images and 57 boxes. In small sample regimes, seed variance easily dominates marginal dataset additions.
4. **Optimism of Selected Checkpoint**:
   Because `layouts_s0` was the best of 5 seeds on the validation split (0.514 vs 5-seed mean 0.446), its validation score is inherently optimistic.

### 6. Final Decision & Model Selection
Per the pre-registered decision rule, `data/yolo_v3` is **rejected** for production deployment. The end-to-end extraction pipeline remains pinned to `data/yolo_v2` with primary weights `data/models/layouts_s0_best.pt`.
