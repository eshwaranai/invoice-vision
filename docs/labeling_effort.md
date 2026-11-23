# Manual Labeling Effort & Pre-Label Suggester Evaluation

## 1. Executive Summary & Problem Context

During the manual annotation of `batch_01` (15 public Indian GST invoices) using makesense.ai, labeling speed was severely bottlenecked by missing and inaccurate pre-labels:
- Out of **95 total reviewed bounding boxes**, only **35 (36.8%)** originated from initial pre-labels.
- **60 boxes (63.2%)** had to be drawn entirely by hand (an average of **4.0 hand-drawn boxes per page**).
- Major omissions: `vendor` (0/15 pre-labeled) and `table` (0/15 pre-labeled) had zero suggestions, while `invoice_no` (10 added), `gstin` (7 added), and `total` (7 added + 3 false positive deletions) required substantial manual correction.

To accelerate annotation throughput, a unified heuristic suggester was developed (`src/invoice_vision/prelabel/suggest.py`).

> [!WARNING]
> **Methodological Caveat: In-Sample vs Out-of-Sample Evaluation**
> - The heuristic rules in `suggest.py` were designed specifically after inspecting failure modes on `batch_01`. Therefore, **all `batch_01` evaluation metrics are strictly in-sample (best case)**.
> - `batch_02` consists of invoices from the same template families as `batch_01` (Microsoft, Sai Computer, Net2Source, LinkedIn) and is **also not independent**.
> - **`batch_03` (Amazon template, $n=6$ images, including 3 continuation pages) represents the first true out-of-sample evaluation.**
> - On `batch_03`, performance dropped precipitously: only **6 of 25 reviewed boxes (24.0%)** were kept or adjusted, strict precision was **4 of 14 proposals (28.6%)**, and **8 of 14 proposals were deleted as false positives**.
> - Any prior claim of a generalized "55% reduction in labeling effort" is **invalid across unseen templates**. That 55% reduction was observed strictly in-sample on `batch_01`. Out-of-sample, annotators still had to draw 76% of boxes by hand (19/25), while also spending time deleting spurious false positive proposals.

---

## 2. In-Sample Baseline Performance (`batch_01`)

The baseline pre-labels were produced by the initial rule-based pipeline and exported from `data/annotation/batch_01`. The ground truth was established by human review in makesense.ai and exported to `data/annotation/exported/batch_01`.

```
==========================================================================================
BASELINE PRE-LABEL VS REVIEWED COMPARISON (batch_01, n=15 images, 95 reviewed boxes)
==========================================================================================
Class            Kept (IoU>=0.5)  Adj (0.3<=IoU<0.5)  Deleted  Added (Hand-Drawn)  Prec(Strict)  Rec(Strict)
------------------------------------------------------------------------------------------
vendor           0                0                   0        15                  N/A           0.0%
invoice_no       6                0                   0        10                  100.0%        37.5%
date             9                0                   0        6                   100.0%        60.0%
gstin            12               0                   0        7                   100.0%        63.2%
table            0                0                   0        15                  N/A           0.0%
total            8                0                   3        7                   72.7%         53.3%
qr               0                0                   0        0                   N/A           N/A
stamp_signature  0                0                   0        0                   N/A           N/A
------------------------------------------------------------------------------------------
TOTAL / MACRO    35               0                   3        60                  92.1%         36.8%
==========================================================================================
```

---

## 3. In-Sample Suggester Evaluation (`batch_01` and `batch_02`)

### 3.1 `batch_01` (In-Sample Calibration Batch, $n=15$)
```
==========================================================================================
NEW SUGGESTER VS REVIEWED: batch_01 (IN-SAMPLE, n=15 images, 95 reviewed boxes)
==========================================================================================
Class            Kept (IoU>=0.5)  Adj (0.3<=IoU<0.5)  Deleted  Added (Hand-Drawn)  Prec(Strict)  Rec(Strict)  Prec(Lenient) Rec(Lenient)
------------------------------------------------------------------------------------------
vendor           5                6                   4        4                   33.3%         33.3%        73.3%         73.3%
invoice_no       3                0                   0        13                  100.0%        18.8%        100.0%        18.8%
date             9                0                   6        6                   60.0%         60.0%        60.0%         60.0%
gstin            19               0                   0        0                   100.0%        100.0%       100.0%        100.0%
table            13               2                   0        0                   86.7%         86.7%        100.0%        100.0%
total            10               1                   2        4                   76.9%         66.7%        84.6%         73.3%
qr               0                0                   0        0                   N/A           N/A          N/A           N/A
stamp_signature  0                0                   0        0                   N/A           N/A          N/A           N/A
------------------------------------------------------------------------------------------
TOTAL / MACRO    59               9                   12       27                  73.8%         62.1%        85.0%         71.6%
==========================================================================================
```

### 3.2 `batch_02` (In-Sample Template Family, $n=3$, 20 reviewed boxes)
`batch_02` contains images belonging to the same template families as `batch_01`. Because the heuristics were tuned on these exact layout patterns, performance remained artificially high:

```
==========================================================================================
PRE-LABEL VS REVIEWED: batch_02 (IN-SAMPLE TEMPLATE FAMILIES, n=3 images, 20 reviewed boxes)
==========================================================================================
Class            Kept (IoU>=0.5)  Adj (0.3<=IoU<0.5)  Deleted   Added (Hand-Drawn)  Prec(Strict)  Rec(Strict)   Prec(Lenient) 
------------------------------------------------------------------------------------------
vendor           3                0                   0         0                   100.0%        100.0%        100.0%        
invoice_no       0                0                   0         3                   N/A           0.0%          N/A           
date             3                0                   0         0                   100.0%        100.0%        100.0%        
gstin            6                0                   0         0                   100.0%        100.0%        100.0%        
table            3                0                   0         0                   100.0%        100.0%        100.0%        
total            2                0                   0         0                   100.0%        100.0%        100.0%        
qr               0                0                   0         0                   N/A           N/A           N/A           
stamp_signature  0                0                   0         0                   N/A           N/A           N/A           
------------------------------------------------------------------------------------------
TOTAL / MACRO    17               0                   0         3                   100.0%        85.0%         100.0%        
==========================================================================================
```

---

## 4. First Out-of-Sample Evaluation: `batch_03` (Amazon Template)

`batch_03` represents the first evaluation on a template family not inspected during rule development. It consists of 6 public images (`gst_024` through `gst_029`):
- 3 primary invoice pages (`gst_024`, `gst_026`, `gst_028`).
- 3 multi-page continuation sheets (`gst_025`, `gst_027`, `gst_029`).

```
==========================================================================================
PRE-LABEL VS REVIEWED: batch_03 (FIRST OUT-OF-SAMPLE TEMPLATE, n=6 images, 25 reviewed boxes)
==========================================================================================
Class            Kept (IoU>=0.5)  Adj (0.3<=IoU<0.5)  Deleted   Added (Hand-Drawn)  Prec(Strict)  Rec(Strict)   Prec(Lenient) 
------------------------------------------------------------------------------------------
vendor           1                0                   5         5                   16.7%         16.7%         16.7%         
invoice_no       0                0                   0         3                   N/A           0.0%          N/A           
date             1                2                   0         0                   33.3%         33.3%         100.0%        
gstin            0                0                   0         3                   N/A           0.0%          N/A           
table            1                0                   0         2                   100.0%        33.3%         100.0%        
total            0                0                   3         3                   0.0%          0.0%          0.0%          
qr               1                0                   0         1                   100.0%        50.0%         100.0%        
stamp_signature  0                0                   0         2                   N/A           0.0%          N/A           
------------------------------------------------------------------------------------------
TOTAL / MACRO    4                2                   8         19                  28.6%         16.0%         42.9%         
==========================================================================================
```

### 4.1 Summary of Out-of-Sample Metrics
- **Reviewed Boxes Covered**: Only **6 of 25 (24.0% lenient recall)**.
- **Strictly Kept**: Only **4 of 25 (16.0% strict recall)**.
- **Proposals Generated**: 14 total proposals across 6 images.
  - Strictly correct: 4 / 14 (**28.6% strict precision**).
  - Leniently correct: 6 / 14 (**42.9% lenient precision**).
  - False positive deletions: **8 of 14 proposals (57.1%) were deleted by the reviewer**.
- **Manual Workload**: **19 of 25 boxes (76.0%) had to be drawn by hand**.

---

## 5. Failure Analysis on `batch_03`

To adhere to factual reporting, failure causes are categorized as **Verified from Data** (proven by inspecting cached OCR tokens and ground-truth coordinates) or **Hypothesis** (unconfirmed conjecture).

### 5.1 GSTIN Extractor: 0/3 Detected (Verified from Data)
- **Verified Fact**: In `data/processed/ocr_cache/gst_024.jpg.json` and `gst_028.jpg.json`, Amazon invoices print GSTINs under the label:
  ```
  "GST Registration No: 06BCDEF7890G1Z5"
  ```
- **Root Cause**: The prefix stripping regex in `suggest.py` was defined as `r"^(?:GSTIN|GST|UIN)[\s:/,\.\-]+"` (tuned on Microsoft and Sai Computer invoices). It stripped `GST:`, but failed to strip `Registration No:`. The candidate string remained `"Registration No: 06BCDEF7890G1Z5"`, which failed `len(val_str.split()) > 1` (it had 3 words) and failed the 15-character length check.
- *Hypothesis*: *Broadening prefix stripping to include `Registration No` would recover Amazon GSTINs, but doing so without evaluating on unseen batches would constitute dev-set leakage.*

### 5.2 Vendor Header Extractor: 1/6 Kept, 5 Deleted (Verified from Data)
- **Verified Fact**: On Amazon invoices, the vendor logo and registered address sit strictly on the top-left quadrant ($x \approx 0.08 \dots 0.35$, width $\approx 0.29$). There is no parallel right-aligned invoice metadata column at that vertical height.
- **Root Cause**: In `suggest.py`, the dynamic horizontal barrier ($x_{split}$) is only activated if right-aligned metadata rows exist. In their absence, the algorithm grouped all text in the top 35% of the page across the full page width ($w \approx 0.88$ normalized). The resulting IoU against the tight ground-truth box ($w \approx 0.29$) fell to $\approx 0.20$ (below the $0.3$ lenient threshold), causing 5 proposals to be rejected as false positives.
- **Continuation Pages**: `gst_025`, `gst_027`, and `gst_029` are continuation sheets containing only a small top logo and disclaimer paragraphs. The suggester grouped the entire top block, proposing invalid vendor boxes on pages where annotators marked only the small logo.

### 5.3 Total Extractor: 0/3 Kept, 3 Deleted (Verified from Data)
- **Verified Fact**: Amazon invoices feature multi-line itemized charges with intermediate totals (`TOTALS`, `Total Tax`, `Shipping Charges`, `Invoice Value`).
- **Root Cause**: The suggester's right-most amount heuristic matched intermediate table columns or taxable sub-values rather than the final grand total at the bottom-right summary row, resulting in 3 deletions and 3 manual additions. Furthermore, on continuation sheets containing no totals, the fallback logic proposed arbitrary numbers from the bottom 35% of the page as totals.

### 5.4 Invoice Number Extractor: 0/3 Detected (Hypothesis)
- *Hypothesis*: *Amazon invoices format invoice numbers with labels like `Invoice Number:` with distinct line-break spacing or inside table cells that did not trigger the immediate single-line adjacency check in `suggest_invoice_no_box`.*

---

## 6. Workload & Time Savings Reality Check

| Batch | Type | Reviewed Boxes | Kept + Adj (Coverage) | Hand-Drawn Boxes | Deleted (False Proposals) | Reviewer Experience |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`batch_01`** | In-Sample (Rule Design) | 95 | **68 (71.6%)** | 27 (28.4%) | 12 | Substantial time saved; 55% reduction in drawing. |
| **`batch_02`** | In-Sample (Same Families) | 20 | **17 (85.0%)** | 3 (15.0%) | 0 | Fast review; rules matched template layouts. |
| **`batch_03`** | **Out-of-Sample (Amazon)** | 25 | **6 (24.0%)** | **19 (76.0%)** | **8** | **High friction; 76% hand-drawn + 8 deletions.** |

### Key Engineering Takeaway
Rule-based heuristic suggesters exhibit extreme template sensitivity. While they drastically cut labeling time on familiar document structures (71%--85% coverage), they generalize poorly to novel layout archetypes (24% coverage on Amazon). Propagating pre-labels without acknowledging in-sample bias leads to brittle workflows; future phases must rely on trainable learned models (YOLO field detectors) rather than endlessly compounding hand-crafted spatial rules.

---

## 7. Synthetic Phone Photo Label Propagation & AABB Loosening

### Coordinate Transformation
To leverage the reviewed ground-truth annotations without re-annotating degraded phone photos, annotations were mathematically propagated using the known ground-truth page corners from `data/processed/synth_photos/manifest.json`.

For a source page, the perspective transformation matrix $H$ maps the unit square $[0, 1] \times [0, 1]$ directly to the four synthetic canvas corner coordinates $P_{dst}$:
$$H = \text{cv2.getPerspectiveTransform}(P_{src}, P_{dst})$$
Where $P_{src} = \begin{bmatrix} [0, 0], & [1, 0], & [1, 1], & [0, 1] \end{bmatrix}^T$.

Each normalized bounding box $(x_c, y_c, w, h)$ is decomposed into its 4 corner vertices:
$$P_{box} = \begin{bmatrix} x_c - w/2 & y_c - h/2 \\ x_c + w/2 & y_c - h/2 \\ x_c + w/2 & y_c + h/2 \\ x_c - w/2 & y_c + h/2 \end{bmatrix}$$
And warped via $P'_{box} = H \cdot P_{box}$.

### Axis-Aligned Bounding Box (AABB) Loosening
Standard YOLO object detection formats require axis-aligned bounding boxes (AABBs):
$$x_{min} = \min(P'[:, 0]), \quad x_{max} = \max(P'[:, 0]), \quad y_{min} = \min(P'[:, 1]), \quad y_{max} = \max(P'[:, 1])$$

Because the invoice is subjected to non-zero perspective tilt and in-plane rotation ($\theta \ne 0$), the warped bounding box is an oriented quadrilateral. Enclosing this quadrilateral within an axis-aligned box geometrically expands the bounding box area:
$$\text{Area}_{AABB} = (x_{max} - x_{min})(y_{max} - y_{min}) \ge \text{Area}_{quad}$$
Where $\text{Area}_{quad}$ is computed via the Shoelace formula:
$$\text{Area}_{quad} = \frac{1}{2} \left| \sum_{i=1}^{4} (x_i y_{i+1} - y_i x_{i+1}) \right|$$

### Empirical Loosening Measurements
Evaluated across all **200 propagated bounding boxes** (32 synthetic phone photos):

| Degradation Severity | Sample Boxes | Mean Loosening Ratio ($\text{Area}_{AABB} / \text{Area}_{quad}$) | Mean Area Expansion (%) | Std Dev |
| :--- | :--- | :--- | :--- | :--- |
| **mild** | 50 | **1.1193** | **+11.93%** | 0.0681 |
| **medium** | 50 | **1.2033** | **+20.33%** | 0.1505 |
| **severe** | 50 | **1.2337** | **+23.37%** | 0.2178 |
| **hard** | 50 | **1.3159** | **+31.59%** | 0.2400 |
| **OVERALL** | **200** | **1.2180** | **+21.80%** | **0.1950** |

#### Engineering Implications of AABB Loosening:
1. **Background Contamination**: At severe/hard tilt (up to $25^\circ$), an average of 23% to 32% extra background pixels (desk texture, adjacent table rows) enter the bounding box.
2. **IoU Penalty**: Models trained on AABBs of rotated documents experience slightly degraded IoU evaluations against tight ground truths. For production deployments with extreme camera tilts, an Oriented Bounding Box (OBB) model (e.g., YOLO-OBB) should be preferred over standard 2D AABB.
3. **Verification Montage**: Visual inspection of 6 diverse synthetic samples (`mild`, `medium`, `severe`, `hard` on dark wood and light table) was verified in `results/figures/synth_label_check.jpg`. All boxes tightly envelop their respective text elements without clipped corners.

---

## 8. Key Engineering Decisions & Defensible Tradeoffs

### Decision 1: High-Precision vs High-Recall on Invoice Numbers
- **Tradeoff**: Accepting relaxed prefix matching would increase recall (e.g., catching numbers near `Ref:`, `Order:`, or header `Tax Invoice`), but introduces false positive boxes that confuse annotators and degrade downstream parsing.
- **Decision**: Enforced strict adjacent label rules (`Invoice No`, `Bill No`) and negative lookaheads (`Tax Invoice`, `Receipt`). Strict precision achieved **100.0%**. Annotators only need to add missing boxes rather than diagnose and delete misleading false proposals.

### Decision 2: Spatial Metadata Barrier ($x_{split}$) for Vendor Blocks
- **Tradeoff**: Taking all text lines in the top 35% of the page as the vendor box causes company name blocks to merge with right-aligned metadata columns (Invoice Date, GSTIN, PO number).
- **Decision**: Partitioned the top region with a dynamic horizontal barrier: if right-aligned metadata rows exist, the vendor block is horizontally clamped to the left of the minimum metadata $x$-coordinate, preserving clean vendor boundaries (73.3% lenient recall on in-sample data).

### Decision 3: Character-Ratio Sub-Bounding Box Extraction for Merged OCR Tokens
- **Tradeoff**: When OCR engines (such as EasyOCR) group labels and values into a single token (e.g., `Total Charges INR 12,450.751 RUPEES...`), returning the whole token produces an over-sized box containing unwanted label text.
- **Decision**: Regex patterns isolate the target value substring within the token, and the sub-box is estimated proportional to character indices:
  $$x_{min}^{sub} = x_{min} + \frac{\text{start\_idx}}{\text{token\_len}} \times (x_{max} - x_{min})$$
  This eliminates label contamination while preserving exact value coordinates.
