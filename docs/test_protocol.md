# Frozen Test Set Evaluation Protocol & Guardrail Specification

**Status**: FROZEN (Pre-Registered BEFORE Any Test Evaluation)  
**Date**: 2026-11-22  
**Author**: Computer Vision & Invoice Pipeline Engineering Team  

---

## 1. Purpose & Scientific Governance

This protocol establishes the immutable specification for evaluating the end-to-end Indian invoice extraction system on the unseen **Test Split** (14 images, `data/splits_v3.json`).

### Non-Negotiable Test Governance Rules
1. **Zero Test Contamination**: The test split has never been evaluated, previewed, or tuned against during the development of detection models, confidence thresholds, spatial rules, or OCR preprocessing heuristics.
2. **One-Shot Execution**: Evaluation on the test split may be executed **EXACTLY ONCE**.
3. **Lock Enforcement**: Upon completion of the test run, `data/test_lock.json` is written atomically. The runner script (`scripts/run_final_test.py`) will permanently refuse any subsequent execution while this lock exists.
4. **Post-Test Change Policy**: Any code, configuration, weight, or threshold change attempted after the test execution **invalidates the test split**. Any subsequent iteration constitutes a new system version (e.g., v2.0) requiring a completely fresh, untouched held-out evaluation dataset.

---

## 2. Frozen System Specification

The end-to-end pipeline is frozen at the following exact configuration:

| Component | Specification / Value | Verification Mechanism |
| :--- | :--- | :--- |
| **Model Weights** | `data/models/layouts_s0_best.pt` | File Size: `6,556,522` bytes (6.25 MB) |
| **Weights SHA256** | `c76c6753c52c4d78ba73c966e603ec604fccafc633e95db1091e1ecc50824919` | Checked via `hashlib.sha256()` before inference |
| **Architecture** | YOLOv8n (6 classes: `vendor`, `invoice_no`, `date`, `gstin`, `table`, `total`) | Verified matching `data/yolo_v2/data.yaml` |
| **Operating Conf Threshold** | `conf_thresh = 0.05` | Frozen from validation F1 optimization sweep |
| **Review Confidence Bar** | `high_conf_bar = 0.25` | Flagged with `needs_review = True` if `conf < 0.25` |
| **GSTIN Upscale Heuristic** | Adaptive 3x bicubic if `crop_h <= 32` px, 1x otherwise | Preserves high-res 300 DPI text; enhances small scans |
| **GSTIN Parsers & Validators** | `position_aware_gstin_correct` + Mod-36 Luhn checksum | Fixed 15-char regex + character substitution table |
| **Date & Amount Parsers** | ISO normalization (dayfirst priority) + numeric parse | Strict float <= 0.001, Lenient <= 0.01 |
| **Scoring Code** | `scripts/eval_pipeline.py` / `scripts/run_final_test.py` | Exact matching evaluation logic |
| **Frozen Git Commit Hash** | `aa2e9a64ecbf712ef86cc4e7fa40840c0c64f646` | Verified against git tree status |

### Frozen Files & Artifact Integrity Manifest

Every tracked source file, configuration, runner script, and model weight file is pinned by SHA-256 hash. If any file hash differs at runtime, `scripts/run_final_test.py` strictly refuses execution:

| File Path | SHA-256 Checksum |
| :--- | :--- |
| `src/invoice_vision/__init__.py` | `d0c22a3a229f7eeb9a1d48f939bce525d7ebc4fc2d0022bbc8f53369ebc34a82` |
| `src/invoice_vision/detect/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/invoice_vision/ocr/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/invoice_vision/ocr/easyocr_engine.py` | `af07e6b3de5c1f93fa2a9a07d5b7de8ad687dadf176232e036d536485420f123` |
| `src/invoice_vision/parse/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/invoice_vision/parse/baseline_rules.py` | `81ef633260820f41aa5459cfa9072fbe6342357e917fc7db9d07f93b0de912f6` |
| `src/invoice_vision/parse/spatial_rules.py` | `e02db5a19ca0220fbb8b64931ca9057654f220f829ca5467af731f76cba8b8f3` |
| `src/invoice_vision/pipeline/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/invoice_vision/pipeline/extract.py` | `8896e9f936b0801559c5a17152617446db914a9f3d09c12ac4b57881864c2528` |
| `src/invoice_vision/prelabel/__init__.py` | `30c7ab24c108b9d8d4c6555d2e33ed4aa8ffb299b5396cab0d018d158ed9fd47` |
| `src/invoice_vision/prelabel/suggest.py` | `480fc469a056c3d7aed1897602baeeff51e7b65972fdf97c5cb61783d0673baf` |
| `src/invoice_vision/preprocess/__init__.py` | `e6a34005bcfa9235154e780742da68ddbf04106683cde9b974d34d7c34eb242a` |
| `src/invoice_vision/preprocess/deskew.py` | `701353d44e6b51640db7cbbbd7baef72e4dac5efee416ca3b5fba35c22bf16a7` |
| `src/invoice_vision/preprocess/enhancement.py` | `cb5b1e2a53eeb6aec64d0c467ec14e0ec4334e80492aa62d634c9f5c072ec14c` |
| `src/invoice_vision/preprocess/page_detector.py` | `d5cd5a99403898619eea5ce9e4aa949e71abc31ca788d905db8c39ba9ac0af9f` |
| `src/invoice_vision/preprocess/pipeline.py` | `09e675349179f91d0e26b5042b793c058b839fcd79afdc19b9dcb54a18ebe43f` |
| `src/invoice_vision/preprocess/quality_gate.py` | `8aff7589401a148f77c9de08b6f804397b66f3c165686b27e642c0b5df81c3e0` |
| `src/invoice_vision/preprocess/shadow_removal.py` | `5682cce316170424390abab47f20535a112c7d0b3a4031a929288475a1cdb9fd` |
| `src/invoice_vision/validate/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `configs/yolo_v1.yaml` | `528c684becb084911ba67f1e750dccc886bebc8ae7b3875a0ceae61bab9e14de` |
| `scripts/eval_pipeline.py` | `1350ff068dde41f2ff5556ae60fc5f68e17b87bf75ee339df6f4bdae3ac7a21f` |
| `scripts/run_final_test.py` | `063b43501a199842987c594efd752d60eeec72ef3f83cc089d2ce6255df5dab5` |
| `data/models/layouts_s0_best.pt` | `c76c6753c52c4d78ba73c966e603ec604fccafc633e95db1091e1ecc50824919` |

---

## 3. Test Dataset & Effective Sample Size

- **Dataset Split**: `data/splits_v3.json` (`combined_clean_splits['test']`)
- **Total Test Images**: 14
- **Total Test Bounding Boxes**: 81
  - **Public Invoices (n=6)**: `gst_000.jpg`, `gst_012.jpg`, `gst_013.jpg`, `gst_014.jpg`, `gst_020.jpg`, `gst_021.jpg`
  - **Own Invoices (n=8)**: `own_001_p1.png`, `own_002_p1.png`, `own_003_p1.png`, `own_010_p1.png`, `own_010_p2.png`, `own_011_p1.png`, `own_012_p1.png`, `own_013_p1.png`
- **Template Diversity**: Spans diverse commercial formats (including Microsoft, Sai Computer, Net2Source, utility, and retail templates).
- **Mandatory Reporting Flag**:
  Because the number of target entity instances on the test split satisfies n < 15 for individual fields, every reported metric must carry the explicit statistical caveat flag:
  `"(too few to measure)"`.

### Frozen Test Label Manifest

| Test Label File | SHA-256 Checksum | Bounding Boxes |
| :--- | :--- | :--- |
| `data/yolo_v2/labels/test/gst_000.txt` | `698b407c5e0d15b5f0de410fedd0eba1774f556e58b12989499b7ec149dedaf7` | 7 |
| `data/yolo_v2/labels/test/gst_012.txt` | `63b3d892a25fbca71b3cc64d45b366f11d41fad4520536073c9a47930926c270` | 7 |
| `data/yolo_v2/labels/test/gst_013.txt` | `923e56e338933c3c308e979a6fb5fed7e9a1c58b53be14cff58b4d8fac64912c` | 8 |
| `data/yolo_v2/labels/test/gst_014.txt` | `5445236b95f56f816fe0470ac5754d08c85235b5820edb3ea089df5720412356` | 6 |
| `data/yolo_v2/labels/test/gst_020.txt` | `0019634e9eade24bf22afb727ed9d1e6bbcfc6a38c86c5541d6fc364ee3e6e1c` | 7 |
| `data/yolo_v2/labels/test/gst_021.txt` | `f3549482768d7aa905ccf2f27f542cec0c09f4d7ab964a9869581a53ca649ec0` | 7 |
| `data/yolo_v2/labels/test/own_001_p1.txt` | `3d6c90e57425002f980e87db007b0ada7a25559b6621c046183d1f6d357766ee` | 6 |
| `data/yolo_v2/labels/test/own_002_p1.txt` | `eefc7a5d75899d967f3f576d40a9842443932fe1005c88c368bfa4180b10bf36` | 6 |
| `data/yolo_v2/labels/test/own_003_p1.txt` | `1245b6eab5e29b2bab3dec5f0449c2ac4da21c48c352ff58382391d60ac18d65` | 6 |
| `data/yolo_v2/labels/test/own_010_p1.txt` | `cf492891c05a16b7491006f50078ad83b8e4307cd6d779d839ba0db391ebdd1d` | 6 |
| `data/yolo_v2/labels/test/own_010_p2.txt` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 0 |
| `data/yolo_v2/labels/test/own_011_p1.txt` | `d6153715fb67ab2c0b0823ea14fc6d42f4863e8b68e06a312dcf5584205d1cf0` | 5 |
| `data/yolo_v2/labels/test/own_012_p1.txt` | `183c0c8bdf37a8e24568c2c51b27c7b74e3d4f4941de576ee222c471b26ee6ac` | 5 |
| `data/yolo_v2/labels/test/own_013_p1.txt` | `1443933e60e8f49a2ae10be09f1aca1d447e2c6e5142172db33fb65aea16bf24` | 5 |

---

## 4. Required Test Metrics & Experimental Settings

The test evaluation must execute and report the following metrics across three standardized settings:

### Experimental Settings
1. **Setting (a) - End-to-End Pipeline**: YOLO candidate boxes (`conf = 0.05`) + adaptive crop OCR + field parsers.
2. **Setting (b) - Human Ground-Truth Boxes**: Reviewed human annotation boxes + adaptive crop OCR + field parsers (isolates OCR/parser error from detector localization error).
3. **Setting (c) - Spatial-Rules Baseline**: Full-page EasyOCR token clustering + 2D reading-row heuristics (v1 rule baseline).

### Required Reporting Sections
1. **Ultralytics Object Detection Metrics**:
   - Validation mAP50 and mAP50-95 computed on the 14 test images.
   - Per-class AP50 for `vendor`, `invoice_no`, `date`, `gstin`, `table`, `total`.
2. **Operating Point Localization & Quality (`conf = 0.05`)**:
   - Detection Recall and Precision at IoU >= 0.5 per class.
   - Average false detections per image.
3. **Field Extraction Accuracy**:
   - Strict and lenient accuracy for `invoice_no`, `date`, `total`.
   - Detection miss rate per field.
4. **Vendor Localization**:
   - Evaluated strictly by detection localization recall (IoU >= 0.5).
5. **GSTIN Set Retrieval**:
   - Total ground-truth tax identifiers (supplier + buyer).
   - Setting (a) unconstrained recall & precision.
   - Setting (a) constrained to human annotation box overlap (IoU >= 0.3).
   - Setting (b) and Setting (c) recall & precision.
6. **Breakdown by Template Family**:
   - Performance aggregated across distinct invoice template groups.
7. **Privacy Preservation Guarantee**:
   - For own bills (`own_*`), **zero extracted text** may appear in output reports or logs. Only boolean match flags, edit distances, character mismatch indices, and counts are permissible. Public bills (`gst_*`) may display extracted strings.

---

## 5. Execution Guardrail (`scripts/run_final_test.py`)

The test evaluation script enforces the following automated pre-flight security checks:
1. **Refusal without Flag**: If invoked without `--final`, the script aborts immediately without touching test data.
2. **Lock Check**: If `data/test_lock.json` exists on disk, the script aborts with an error indicating the test set was already run and is permanently locked.
3. **Clean Tree Check**: Verifies `git diff --quiet` and `git diff --cached --quiet` to ensure no unstaged or uncommitted code changes exist.
4. **Weights Integrity Check**: Computes SHA-256 of `data/models/layouts_s0_best.pt` and ensures exact equality with `c76c6753c52c4d78ba73c966e603ec604fccafc633e95db1091e1ecc50824919`.
5. **Commit Lineage Check**: Verifies that the current HEAD commit contains the frozen code commit (`aa2e9a64ecbf712ef86cc4e7fa40840c0c64f646`) in its git history.
6. **Lockfile Generation**: Upon completing evaluation, writes `data/test_lock.json` containing:
   ```json
   {
     "timestamp": "<ISO-8601>",
     "commit_hash": "<HEAD-HASH>",
     "weights_sha256": "c76c6753c52c4d78ba73c966e603ec604fccafc633e95db1091e1ecc50824919",
     "test_image_count": 14,
     "output_report": "results/pipeline_test_v1.md"
   }
   ```