# Data Directory

This directory stores datasets and intermediate outputs for the invoice extraction pipeline.

## Directory Structure
- `data/raw/`: Original bill and invoice images (scans, photos, PDF conversions) across categories:
  - GST tax invoices
  - Retail receipts
  - Restaurant bills
  - Fuel slips
  - Pharmacy receipts
- `data/processed/`: Processed outputs including cropped fields, binarized images, warped perspective outputs, annotations, and intermediate artifacts.

## Privacy & Security Policy
> **CRITICAL**: Never commit raw bill images, personal data, or customer-identifiable information (PII) to version control.
> 
> All files in `data/` (except this `README.md` and `.gitkeep` placeholders) are strictly ignored by `.gitignore`.
> Any sample data used for unit testing or integration testing must be completely synthetic, redacted, or anonymized.
