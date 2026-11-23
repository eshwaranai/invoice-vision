# Bounding Box Annotation Guidelines (YOLO Object Detection)

This guideline defines the standard operating procedure for annotating Indian invoices and retail bills for YOLO-based field detection. Adherence to these rules ensures consistency, high model mAP, and seamless integration with downstream crop-level OCR.

---

## 1. Class Taxonomy & Schema

| Class ID | Class Name | Description | Annotation Scope |
| :---: | :--- | :--- | :--- |
| **0** | `vendor` | Supplier / issuer company name, trade name, and logo block at the top of the invoice. | Enclose logo and company title. Exclude separate address/contact details unless tightly enclosed in the header logo block. |
| **1** | `invoice_no` | Invoice / Bill Number **VALUE only**. | Bounding box around the numeric or alphanumeric sequence (e.g. `INV/2023-24/0912`, `984210`). **NEVER include labels** like `"Invoice No:"`, `"Bill #:"`. |
| **2** | `date` | Invoice issuance date **VALUE only**. | Bounding box around date string (e.g. `15-08-2023`, `12/04/2024`, `03 Oct 2023`). **NEVER include labels** like `"Date:"`, `"Billing Date:"`. |
| **3** | `gstin` | 15-character GSTIN **VALUE only**. | Annotate **every** valid GSTIN on the page (supplier GSTIN and customer/buyer GSTIN). Exclude `"GSTIN:"` / `"GST No:"` label. |
| **4** | `table` | Complete line-items table. | Outer bounding quad enclosing the column header row (`Item`, `HSN`, `Qty`, `Rate`, `Amount`) down to the last item row. |
| **5** | `total` | Final Grand Total payable **VALUE only**. | Bounding box around final payable numerical amount (e.g. `14,850.00`). Include currency symbol if contiguous (`₹14,850.00`). Exclude `"Grand Total:"` label. |
| **6** | `qr` | QR Code block. | B2B e-invoice IRN QR code, UPI dynamic payment QR code, or digital validation code. |
| **7** | `stamp_signature` | Official stamp, seal, or signature. | Rubber stamp, company seal, physical handwritten signature, or designated "Authorized Signatory" signing box. |

---

## 2. Bounding Box Tightness & Margin Rules

1. **Tight Margins (approx. 2 px)**:
   - The bounding box must tightly enclose character glyphs with approximately **2 pixels of padding**.
   - Do **NOT** leave generous white margins around tokens. Generous boxes degrade crop-level OCR because adjacent table borders or text lines get included in the crop.
2. **Exclusion of Key Labels**:
   - For `invoice_no`, `date`, `gstin`, and `total`, **ONLY the VALUE** is annotated.
   - *Rationale*: Downstream parsers match values directly; including the keyword inside the crop forces OCR to read both label and value together, complicating regex extraction and causing parsing ambiguity.
3. **Currency Symbols**:
   - If the currency symbol (`₹`, `Rs.`, `INR`) directly touches or precedes the number on the same line, enclose it within the `total` box. If it is located in a distant table column header, enclose only the numeric value.

---

## 3. Edge Cases & Special Handling

### 3.1 Subtotal vs Grand Total
- Invoices frequently contain multiple summary amounts: `Subtotal`, `Taxable Value`, `CGST`, `SGST`, `IGST`, `Round Off`, and `Grand Total`.
- **Rule**: Annotate **ONLY the final payable Grand Total** as class `5 total`.
- **Subtotals and intermediate tax breakdowns are NOT annotated as class 5**. They remain part of the table or footer area and are parsed via crop-level OCR table extractors.

### 3.2 Multi-Line and Partially Visible Values
- **Multi-Line Invoice Numbers**: If an invoice number wraps onto a second line, draw a single tight bounding box enclosing both lines if contiguous. If separated by unrelated metadata, annotate the primary identification segment.
- **Partially Cutoff Characters**: If a value is partially cut off by camera framing but still human-readable (e.g. top of `INV-001` slightly clipped), annotate it. If fewer than 50% of character glyphs are present or the string is illegible, do **not** annotate.

### 3.3 Multiple Dates on the Page
- Invoices often display `Invoice Date`, `Due Date`, `Challan Date`, `PO Date`, and `Supply Date`.
- **Rule**: Annotate **only the primary Invoice Issuance Date** as class `2 date`.
- If only one date is present, annotate it as class `2`.

### 3.4 Multi-Page Invoices & Second Pages
- **Page 1**: Typically contains `vendor`, `invoice_no`, `date`, `gstin`, `table` (top rows), and sometimes `total`.
- **Page 2 / Continuation Sheets**:
  - Contains table continuation, footer notes, and `total`.
  - If a second page contains continuation items, annotate the continuation grid as class `4 table`.
  - If `invoice_no` is repeated in the running header of page 2, annotate it as class `1 invoice_no`.
  - Annotate `total` on the specific page where the final grand total appears (typically the last page).

### 3.5 Line-Items Table (`table`)
- The table box must enclose:
  - The top header row (`S.No`, `Description`, `Qty`, `Unit Price`, `Total`).
  - All itemized product/service rows.
- **Exclude**:
  - Document header metadata (supplier address, invoice number block).
  - Bottom summary blocks (bank account details, terms & conditions, signatory block) unless physically integrated inside the drawn table grid border.

---

## 4. Annotation Checklist for Quality Assurance

Before committing any batch of annotations to the dataset, verify every image against this 8-point checklist:

- [ ] **1. Label Independence**: Are all `invoice_no`, `date`, `gstin`, and `total` boxes strictly values-only (zero keyword labels included)?
- [ ] **2. Box Tightness**: Are box boundaries within 2 pixels of character ink, without excessive whitespace?
- [ ] **3. GSTIN Completeness**: Are **both** supplier and customer GSTINs labeled if present?
- [ ] **4. Total Disambiguation**: Is the labeled `total` strictly the final payable Grand Total (not Subtotal, Net Amount, or Taxable Value)?
- [ ] **5. Table Coverage**: Does the `table` box cleanly cover the column headers down through the last item row?
- [ ] **6. No Class Confusion**: Are QR codes labeled as `qr` (6) and official stamps/signatures labeled as `stamp_signature` (7)?
- [ ] **7. Coordinate Validity**: Are all normalized YOLO coordinates strictly within $[0.0, 1.0]$?
- [ ] **8. No Overlapping Duplicates**: Are there any redundant double-boxes on the exact same token?
