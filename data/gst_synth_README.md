# Synthetic Indian GST Invoices Subset

## Source & Attribution
- **Dataset**: `Divya52/indian-gst-invoices` (Configuration: `synthetic`)
- **Source URL**: [https://huggingface.co/datasets/Divya52/indian-gst-invoices](https://huggingface.co/datasets/Divya52/indian-gst-invoices)
- **Author**: Divya52
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))

## Subset Description
- **Total Sample Count**: Exactly 84 images (`gstsyn_000` through `gstsyn_083`).
  - Initial batch: 24 images (`gstsyn_000` .. `gstsyn_023`).
  - Extension batch: 60 images (`gstsyn_024` .. `gstsyn_083`).
- **Sampling Strategy**: Stratified across all 12 combinations of:
  - **4 Layouts**: `classic`, `handwritten`, `minimal`, `thermal` (21 images each)
  - **3 Degradation Levels**: `clean`, `photo`, `scan` (28 images each)
- **Diversity Balancing**: Greedily maximized joint diversity across:
  - **Fonts**: `arial`, `helvetica`, `courier`, `times`
  - **Header Styles**: `plain`, `band`, None (thermal)
  - **Supply Types**: `inter_state`, `intra_state`
- **Random Seed**: `42` (deterministic selection).
- **Data Privacy**: All company names, addresses, GSTIN numbers, dates, and amounts are procedurally generated synthetic data. Zero real personal or corporate data.

## Directory Layout
- `data/raw/gst_synth/`: Raw image files (`.png` for clean, `.jpg` for photo/scan).
- `data/processed/gst_synth_labels/`: Complete raw ground-truth JSON labels per invoice.
- `data/gst_synth_manifest.csv`: Flattened metadata and key billing attributes for all 84 invoices.
