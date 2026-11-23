# MakeSense.ai Bounding Box Review & Annotation Workflow

This document provides the standard operating procedure for reviewing, refining, and exporting YOLO bounding box annotations for the Indian invoice extraction pipeline using [MakeSense.ai](https://www.makesense.ai/).

---

## 1. Overview & Tool Architecture

- **Tool**: [MakeSense.ai](https://www.makesense.ai/) (Free, open-source online annotation tool).
- **Privacy Assurance**: MakeSense.ai runs **100% client-side in the browser**. No images or bounding box annotations are transferred or uploaded to external servers or cloud services, guaranteeing that personal documents remain completely confidential on the local machine.
- **Batch Architecture**: Document batches are pre-partitioned into bounded folders ($\le 15$ images each in `data/annotation/batch_XX`) to prevent browser canvas memory spikes and keep review sessions manageable.

---

## 2. Step-by-Step Review Procedure

### Step 1: Open Annotation Tool
1. In Google Chrome or Mozilla Firefox, navigate to: `https://www.makesense.ai/`
2. Click the central **"Get Started"** button.

### Step 2: Load Batch Images
1. Click **"Drop images"** or use the file picker.
2. Select all image files (`.jpg`, `.png`) from your current batch folder:
   ```text
   data/annotation/batch_XX/
   ```
   *(e.g., all 15 images in `data/annotation/batch_01/`)*

### Step 3: Select Annotation Mode
1. When prompted with "What would you like to do?", select **"Object Detection"**.

### Step 4: Import Class Labels
1. In the "Create label list" dialog, click **"Load labels from file"**.
2. Select the `classes.txt` file located directly inside your batch folder (`data/annotation/batch_XX/classes.txt`).
3. Verify that the 8 classes appear in exact index order:
   - `0`: `vendor`
   - `1`: `invoice_no`
   - `2`: `date`
   - `3`: `gstin`
   - `4`: `table`
   - `5`: `total`
   - `6`: `qr`
   - `7`: `stamp_signature`
4. Click **"Start project"**.

### Step 5: Import YOLO Pre-Labels
1. In the top navigation bar, click **`Actions`** $\rightarrow$ **`Import Annotations`**.
2. Select **`Single file or multiple files in YOLO format`**.
3. In the file dialog, select **all `.txt` files** from the batch folder (`data/annotation/batch_XX/*.txt`).
4. MakeSense will automatically map the coordinates to the image dimensions and display the pre-labeled bounding boxes.

### Step 6: Review & Refine Every Image
Go through each image sequentially. Use keyboard shortcuts for fast navigation:
- `A` / `D`: Previous / Next image
- `W`: Draw new bounding box
- `Delete` / `Backspace`: Delete selected box
- `Tab`: Cycle active class label

For every image, perform the following quality corrections:
1. **Fix Shifted or Loose Boxes**:
   - Drag handles to adjust box boundaries so they tightly enclose text ink with $\approx 2\text{ px}$ of padding.
   - Ensure bounding boxes do **NOT** extend into table gridlines or neighboring text rows.
2. **Delete False Positives**:
   - Remove spurious boxes (e.g., phone numbers or PIN codes mistakenly tagged as `invoice_no`, or subtotal amounts tagged as `total`).
3. **Add Missing Ground-Truth Fields**:
   - Draw missing boxes for all present classes:
     - **`vendor` (0)**: Header logo and primary supplier trade name block.
     - **`invoice_no` (1)**: Value token only (e.g. `INV/2024/089`). Exclude labels like `"Invoice No:"`.
     - **`date` (2)**: Value token only (e.g. `15/08/2024`). Exclude labels like `"Date:"`.
     - **`gstin` (3)**: 15-character GSTIN token only. **Annotate BOTH supplier and customer GSTINs**.
     - **`table` (4)**: Outer bounding rectangle covering column headers down to the last item row.
     - **`total` (5)**: Final payable Grand Total amount token only (exclude `"Grand Total:"` label).
     - **`qr` (6)**: Payment QR code or B2B e-invoice IRN QR code.
     - **`stamp_signature` (7)**: Company rubber seal, authorized signatory signature, or signature block.

> [!CAUTION]
> **CRITICAL WARNING: BROWSER VOLATILE STORAGE**
> MakeSense.ai stores annotations strictly in ephemeral browser memory. If you refresh (`F5` / `Ctrl+R`), navigate away, or close the browser tab before exporting, **ALL WORK ON THAT BATCH WILL BE IRRETRIEVABLY LOST**.
> Always finish and export the batch before closing the browser.

### Step 7: Export Annotations
1. Once all images in the batch are verified, click **`Actions`** $\rightarrow$ **`Export Annotations`**.
2. Select: **`A .zip package containing files in YOLO format`**.
3. Click **"Export"** to download the archive (typically named `labels_my-project_...zip`).

### Step 8: Unpack Exported Annotations
1. Unzip the downloaded `.zip` file into the designated exported directory for that batch:
   ```text
   data/annotation/exported/batch_XX/
   ```
   *(e.g., `data/annotation/exported/batch_01/` or `data/annotation/exported/batch_04_test/`)*
2. Verify that each image has a corresponding `.txt` file in `data/annotation/exported/batch_XX/`.

---

## 3. Core Annotation Guidelines & Rules

### 3.1 Strict Label Independence (Values Only)
For `invoice_no`, `date`, `gstin`, and `total`, **annotate ONLY the VALUE string**.
- **Correct**: Bounding box strictly around `29ABCDE1234F1ZW`.
- **Incorrect**: Bounding box covering `"GSTIN: 29ABCDE1234F1ZW"`.
- *Engineering Decision*: Downstream parsing relies on crop-level OCR. If the keyword is inside the crop, the OCR output contains both label and value, requiring complex regex heuristics and introducing parsing ambiguity. A value-only crop feeds cleanly into token parsers and checksum validators.

### 3.2 Box Tightness (Approx. 2 px Margin)
- Box borders must fit snugly around the characters with $\approx 2\text{ px}$ of margin.
- Generous white margins allow adjacent line items or vertical table borders to leak into the OCR crop, corrupting character segmentation.

### 3.3 Final Grand Total Disambiguation
- Invoices frequently contain multiple financial figures: `Subtotal`, `Taxable Value`, `CGST`, `SGST`, `IGST`, `Round Off`, and `Grand Total`.
- Annotate **ONLY the final payable Grand Total** as class `5 total`.
- Intermediate tax totals and subtotals belong to the table / financial summary grid and must not be annotated as class `5`.

### 3.4 Line-Items Table (`table`)
- The `table` bounding box must encompass:
  - Top column header row (`Item`, `HSN/SAC`, `Rate`, `Qty`, `Taxable Amount`, `Total`).
  - All line items down through the final item row.
- Exclude standalone document headers, bank detail footers, and terms & conditions notes unless they are physically embedded within the drawn table border.

### 3.5 Multi-Page Documents & Continuation Sheets
- **Page 1**: Typically contains `vendor`, `invoice_no`, `date`, `gstin`, and `table` header.
- **Page 2+**: If the item table continues onto page 2, annotate the continuation grid as class `4 table`.
- If `invoice_no` repeats in the running header of page 2, annotate it as class `1 invoice_no`.
- Annotate `total` on the specific page where the final grand total payable figure appears (usually the last page).

---

## 4. Crucial Revision: Handling Partially Cutoff & Hard-to-Read Values

> [!IMPORTANT]
> **REVISED RULE FOR DEGRADED / PARTIALLY VISIBLE VALUES**:
> **Label anything visible, even if partially truncated or degraded, and maintain a note for hard cases.**

### Why the Previous "< 50% readable $\rightarrow$ do not annotate" Rule Was Flawed
In supervised object detection (such as YOLOv8/YOLOv11):
1. **Unlabeled Objects are Treated as Background**:
   Any pixel region without a bounding box is treated as negative background by the focal/cross-entropy loss during training.
2. **Detector Penalization**:
   If a partially visible `gstin` or `invoice_no` is left unlabeled, YOLO is actively penalized with background loss whenever it correctly predicts an anchor on that text.
3. **Degraded Recall on Mobile Photos**:
   In real-world camera captures with perspective skew, slight cropping, or lighting variation, the model learns to suppress predictions on any non-pristine text.

### Practical Protocol:
- If character glyphs of an invoice field are partially visible (even 20%–40% visible or clipped at the page edge), **draw a tight box around all visible ink** and assign the correct class.
- If a value is genuinely 100% illegible or completely outside the image frame, omit it.
- Record any unusual edge cases in your review notes for downstream inspection.
