# Project Limitations & Empirical Constraints

This document records known experimental, dataset, and architectural limitations of the Indian Invoice Vision extraction pipeline. Maintaining an honest record of limitations is essential for reproducible research and rigorous technical interview defense.

---

## 1. Dataset Limitations

### 1.1 Synthetic Degradation vs Real Camera Optics
- **Nature of Synthetic Set**: The degraded phone photo evaluation set (`data/processed/synth_photos/`) was generated procedurally by applying synthetic affine/perspective warps, 2D Gaussian blur, synthetic Gaussian sensor noise, linear gradient shadows, and JPEG re-compression to digital PDF scans.
- **Missing Real-World Factors**: Procedural transformations do not capture physical phenomena encountered in real mobile captures:
  - Non-planar page curl and creases (cylindrical/barrel distortion).
  - Variable lens distortion (radial aberration, chromatic aberration).
  - Spatial depth-of-field falloff (defocus blur varying across the page quad).
  - Localized specular flash hot-spots and reflective surface glares.
  - Complex non-desk backgrounds (crumpled bedsheets, patterned fabrics, hands/fingers holding the paper).
- **Labeling Requirement**: Every dataset, report, and script must explicitly label these samples as **`SYNTHETIC`**.

### 1.2 Small Sample Size ($n=12$ Invoices)
- The ablation study was conducted on 12 representative invoice templates ($12 \times 4 \text{ severities} \times 2 \text{ backgrounds} = 96$ synthetic images).
- While covering major Indian invoice archetypes (tax invoices, utility statements, retail slips, Amazon seller invoices), $n=12$ is insufficient to capture the full statistical variance of long-tail Indian SME invoices (handwritten additions, regional script stamps, degraded thermal printouts).

---

## 2. Preprocessing & Quality Assessment Limitations

### 2.1 Center-Seeded GrabCut Assumptions
- The primary page boundary detector relies on center-seeded GrabCut segmentation downscaled to $360\text{ px}$.
- **Constraint**: The initialization assumes the document occupies the central region of the image frame (document bounds initialized with a border margin of $15\text{--}20\%$).
- If a user captures an extreme off-center photo where the paper sits entirely along the edge or corner of the camera frame, center-seeded GrabCut will segment the background as foreground, resulting in detection failure.

### 2.2 Template Leakage in Initial Quality Gate Split
- In the initial calibration/evaluation split (v3), `gst_004.jpg` (Microsoft Commercial Invoice) was placed in the calibration set while `gst_005.jpg` (Microsoft Statement) was placed in the evaluation set.
- Although the layout and typography differ slightly (commercial tax invoice vs monthly account summary), both belong to the Microsoft template family, introducing potential layout correlation across splits.
- This has been resolved by the template-safe family merge in `data/splits.json` where all 8 Microsoft invoices are strictly isolated into a single split.

### 2.3 Page Detection Success Tolerance ($25\text{ px} \to 150\text{ px}$)
- In the initial clean baseline, page detection tolerance was set to $\le 25\text{ px}$ mean corner error.
- For synthetic phone photos on low-contrast light surfaces, corner localization errors averaged $11\text{--}35\text{ px}$ due to slight edge rounding under Gaussian blur.
- To prevent false failure flags when the entire text block is successfully captured within the warped quad, the benchmark success tolerance was widened to $\le 150\text{ px}$ (approx. $5\text{--}8\%$ of full $2000\text{ px}$ image diagonal).
- While adequate for OCR text retention, a $150\text{ px}$ boundary offset may leave small margin borders or clip extreme outer headers on tight framing.

### 2.4 Ensemble Selector Underperformance
- In multi-strategy page detection benchmarks across 96 images:
  - **GrabCut Alone**: **96.9%** overall accuracy ($45/48 = 93.8\%$ on low-contrast `light_surface`).
  - **Ensemble Selector**: **82.3%** overall accuracy ($31/48 = 64.6\%$ on `light_surface`).
- **Why Ensemble Underperformed**: The composite scoring function weighted edge gradient magnitude ($S_{\text{edge}}$). On low-contrast light surfaces, background table borders or internal lines produce sharper local gradients than the subtle document boundary, causing the ensemble to occasionally pick a fragmented Canny candidate over the correct GrabCut quad.
- **Architectural Decision**: The production page detector defaults to **GrabCut-first detection with contour-based fallback (`canny_otsu` / `morph_grad`)**, rather than trusting the uncalibrated ensemble selector.

---

## 3. Scope & Operational Limitations
- **Language**: English and numerical text only. EasyOCR is configured for English; vernacular scripts (Hindi, Tamil, Marathi) are not yet handled.
- **Hardware Profile**: Designed strictly for CPU execution (Intel/AMD x86_64, 8 GB RAM). No GPU-dependent inference paths are assumed.

---

## 4. Annotation & Pre-Label Evaluation Limitations (Dev-Set Contamination Rules)

### 4.1 In-Sample Overfitting of Heuristic Suggesters
- The rule-based pre-label suggester (`src/invoice_vision/prelabel/suggest.py`) was formulated after inspecting failure modes on `batch_01` (15 public invoices).
- Consequently, metrics reported on `batch_01` (71.6% coverage, 55% reduction in hand-drawn boxes) represent **in-sample upper bounds**, not generalized system accuracy.
- `batch_02` comprises invoices from identical template families (Microsoft, Sai Computer, Net2Source, LinkedIn) and also does not provide independent out-of-sample validation.

### 4.2 Out-of-Sample Drop on `batch_03` (Amazon Template)
- The first independent test on `batch_03` ($n=6$, Amazon seller invoices including 3 continuation pages) exhibited severe layout brittleness:
  - Strict precision fell to **28.6%** (4 of 14 proposals matched; 8 proposals were deleted as false positives).
  - Lenient coverage reached only **24.0%** (6 of 25 reviewed boxes matched; 19 boxes had to be drawn by hand).
  - Root causes identified from data: failure to strip `"GST Registration No:"` prefixes (missing GSTINs), failure to constrain vendor blocks without right-aligned metadata barriers on single-column headers (oversized vendor boxes spanning width $\approx 0.88$), and selecting item table summary tokens instead of grand totals.

### 4.3 Mandatory Evaluation Protocol: `batch_03` Designated as Dev Data
- **Operating Rule**: From now on, **`batch_03` counts strictly as development data (dev data)**.
- Rules, regex patterns, or spatial heuristics may be inspected or iterated against `batch_03` **ONLY IF** a completely fresh, previously unseen batch—such as the personal own-bill batch `batch_05`—is held out and used to report subsequent evaluation results.
- No performance improvements may be claimed or published by testing against `batch_01`, `batch_02`, or `batch_03` once their ground truth has been inspected.

---

## 5. Validation Split Pipeline & Baseline Evaluation Limitations

### 5.1 Baseline Outperforming YOLO on Selected Header Fields
- On the 12 validation images (`data/splits_v3.json`), the v1 rule-based spatial baseline on full-page EasyOCR matched or exceeded the YOLO pipeline:
  - `invoice_no`: Rule-based baseline achieved **100.0%** lenient accuracy ($9/9$), whereas Setting (a) [YOLO + OCR] achieved **66.7%** ($6/9$).
  - `date`: Rule-based baseline achieved **100.0%** lenient accuracy ($9/9$), matching Setting (a) at **88.9%** ($8/9$).
- Spatial heuristics perform well when key-value pairs sit on predictable single-line patterns, whereas the detector suffers from lower recall or false crop boundaries on unseen layout styles.

### 5.2 Detector Misses on Total Amount at Operating Point
- At the operating threshold `conf = 0.05`, `total` detection recall is severely limited:
  - Detection miss rate is **77.8%** ($7/9$ invoices have no detected total box).
  - Setting (a) total extraction accuracy is **0.0%** ($0/9$), despite Setting (b) [Human Boxes] achieving **66.7%** ($6/9$).
  - Grand total labels often appear near bottom-right summaries or within complex subtotal blocks where YOLOv8n fails to generate confident proposals above $0.05$.

### 5.3 Small Effective Sample Size & Low Template Variance
- Although the validation set contains 12 image files, the **effective sample size is only 9 invoices** across **2 template families** (`Amazon India Seller Invoices`, $n=3$, and `own_family_A Corporation`, $n=6$; the remaining 3 public images are continuation pages with no header metadata).
- All 6 own bills in the validation split belong to a single template family and share **exactly 1 distinct GSTIN string**. Consequently, perfect GSTIN extraction on own validation bills reflects repeated performance on a single string rather than generalized token recognition across diverse Indian state codes and PAN structures.
- All target fields have $n = 9$ (or $n=12$ for GSTIN candidates), strictly $< 15$, carrying the mandatory warning **`(too few to measure)`**.

### 5.4 Optimistic Validation Metrics (Dev-Set Tuning)
- The detector operating threshold (`conf = 0.05`) and the adaptive GSTIN upscaling rule ($\le 32$px crop height) were selected and tuned directly after analyzing failure modes on the validation split.
- Therefore, validation accuracy numbers are **optimistic (in-sample tuned)**.
- Unbiased performance can only be established via a frozen, guarded execution on the held-out test split.

### 5.5 Inconsistent Vendor Class Semantics Across Batches
- The `vendor` class suffers from semantic drift across annotation batches:
  - In `batch_03` (Amazon marketplace invoices), annotators enclosed the top-left marketplace logo (`amazon.in`).
  - In other batches, annotators enclosed the registered commercial supplier name, or an entire composite block including branch addresses, state codes, and corporate identifiers.
- Because the visual crop content does not consistently map to legal supplier names across template families, strict string match and fuzzy token overlap produce artificial zero scores on valid visual boxes.
- **Reporting Decision**: Vendor is evaluated and reported strictly by **detection localization recall (IoU $\ge 0.5$) only**.

### 5.6 Batch 03 Annotations Predating the Dual-GSTIN Protocol
- Human annotation of `batch_03` (`gst_024` through `gst_029`) was completed before the project formalized the two-GSTIN protocol (extracting both supplier and buyer tax identifiers).
- In `batch_03`, annotators labeled only the supplier GSTIN, leaving buyer GSTINs unannotated in the `.txt` label files even though both exist on the document and in `data/ground_truth_gst.csv`.
- This causes human-box Setting (b) to achieve only $75.0\%$ GSTIN set recall ($9/12$), while Setting (a) autonomously localizes the unannotated buyer box on `gst_024.jpg`, reaching $83.3\%$ ($10/12$) unconstrained. Constraining Setting (a) to human box overlap ($\text{IoU} \ge 0.3$) restores equivalence at $75.0\%$.
